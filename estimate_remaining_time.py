import argparse
import hashlib
import os
import sqlite3
import sys

import cv2

VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v',
    '.mpg', '.mpeg', '.3gp', '.f4v', '.ts', '.vob', '.rmvb', '.rm',
    '.asf', '.divx', '.xvid', '.m2ts', '.mts'
}

ARTIFACT_SUFFIXES = [
    '_frames.mp4',
    '_objects.mp4',
    '_detections.mp4',
    '_mosaic.jpg',
    '.txt',
    '.done',
]

DONE_SUFFIX = '.done'


def is_video_file(file_path):
    if not file_path:
        return False
    base = file_path.lower()
    if base.endswith('_frames.mp4') or base.endswith('_objects.mp4') or base.endswith('_detections.mp4'):
        return False
    if '_frames.part' in base or '_objects.part' in base or '_detections.part' in base:
        return False
    _, ext = os.path.splitext(base)
    return ext in VIDEO_EXTENSIONS


def safe_artifact_basename(video_path):
    """当前格式: 直接使用原始文件名基础名, 在资源管理器中紧跟原视频排序。"""
    return os.path.splitext(os.path.basename(video_path))[0]


def _sanitize_basename(video_path):
    base_name = os.path.basename(os.path.splitext(video_path)[0])
    sanitized = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in base_name)
    return sanitized or 'video'


def legacy_artifact_basename(video_path, max_length=80):
    """旧格式 v2: sanitized + stat-hash."""
    sanitized = _sanitize_basename(video_path)
    try:
        st = os.stat(video_path)
        size = int(getattr(st, 'st_size', 0) or 0)
        mtime = float(getattr(st, 'st_mtime', 0.0) or 0.0)
        digest = hashlib.md5(f"{size}|{mtime}".encode('utf-8', 'ignore')).hexdigest()[:8]
    except Exception:
        digest = hashlib.md5(sanitized.encode('utf-8', 'ignore')).hexdigest()[:8]
    limit = max(8, max_length - len(digest) - 1)
    if len(sanitized) > limit:
        sanitized = sanitized[:limit]
    return f"{sanitized}_{digest}"


def _legacy_artifact_basename_v1(video_path, max_length=80):
    """旧格式 v1: sanitized + path-hash."""
    sanitized = _sanitize_basename(video_path)
    digest = hashlib.md5(str(video_path).encode('utf-8', 'ignore')).hexdigest()[:8]
    limit = max(8, max_length - len(digest) - 1)
    if len(sanitized) > limit:
        sanitized = sanitized[:limit]
    return f"{sanitized}_{digest}"


def get_shared_state_dir():
    shared = os.environ.get('FINDINVIDEO_SHARED_STATE_DIR')
    if not shared:
        shared = r"\\192.168.31.9\\d\\md5_list"
    return shared or None


def get_yoloed_md5_path():
    yoloed_path = os.environ.get('FINDINVIDEO_YOLOED_PATH')
    if not yoloed_path:
        shared = get_shared_state_dir()
        if shared:
            yoloed_path = os.path.join(shared, 'yoloed.txt')
        else:
            yoloed_path = r"D:\\md5_list\\yoloed.txt"
    return yoloed_path


def load_yoloed_paths():
    path = get_yoloed_md5_path()
    path_set = set()
    if not path or not os.path.exists(path):
        return path_set
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                entry = line.strip()
                if not entry:
                    continue
                parts = entry.split('|', 1)
                if len(parts) == 2:
                    raw_path = parts[1].strip()
                    if raw_path:
                        path_set.add(raw_path)
                        path_set.add(os.path.normpath(raw_path))
    except Exception:
        return path_set
    return path_set


def has_existing_artifacts(video_path):
    try:
        video_dir = os.path.dirname(video_path) or '.'
        bases = [safe_artifact_basename(video_path),
                 legacy_artifact_basename(video_path),
                 _legacy_artifact_basename_v1(video_path)]
        for base in bases:
            done_path = os.path.join(video_dir, base + DONE_SUFFIX)
            if os.path.exists(done_path):
                return True
            for suffix in ARTIFACT_SUFFIXES:
                artifact_path = os.path.join(video_dir, base + suffix)
                if os.path.exists(artifact_path):
                    return True
    except Exception:
        return False
    return False


def iter_videos_from_db(db_path, root_path):
    if not os.path.exists(db_path):
        return []
    root_norm = os.path.normpath(root_path)
    like_pattern = root_norm.rstrip(os.sep) + os.sep + '%'
    rows = []
    conn = sqlite3.connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        query = (
            "SELECT v.dir_path, v.file_name "
            "FROM videos v "
            "JOIN directories d ON v.dir_path = d.path "
            "WHERE v.is_video = 1 AND d.excluded = 0 AND (v.dir_path = ? OR v.dir_path LIKE ?)"
        )
        rows = conn.execute(query, (root_norm, like_pattern)).fetchall()
    finally:
        conn.close()
    results = []
    for row in rows:
        dir_path = row['dir_path']
        file_name = row['file_name']
        if not dir_path or not file_name:
            continue
        results.append(os.path.join(dir_path, file_name))
    return results


def iter_videos_by_walk(root_path):
    results = []
    for root, _, files in os.walk(root_path):
        for name in files:
            if is_video_file(name):
                results.append(os.path.join(root, name))
    return results


def get_frame_count(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return None
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if frame_count and frame_count > 0:
        return int(frame_count)
    return None


def main():
    parser = argparse.ArgumentParser(description='Estimate remaining processing time based on total frames.')
    parser.add_argument('--root', default=r'D:\\z', help='Root path to scan for videos')
    parser.add_argument('--db', default='directory_index.db', help='Path to directory_index.db')
    parser.add_argument('--fps', type=float, default=10.0, help='Average processing speed in frames/sec')
    args = parser.parse_args()

    root_path = os.path.normpath(args.root)
    db_path = os.path.normpath(args.db)

    if not os.path.isdir(root_path):
        print(f"Root path not found or not a directory: {root_path}")
        sys.exit(1)

    yoloed_paths = load_yoloed_paths()

    videos = iter_videos_from_db(db_path, root_path)
    if not videos:
        videos = iter_videos_by_walk(root_path)

    total_frames = 0
    pending_videos = 0
    skipped_artifact = 0
    skipped_path = 0
    unreadable = 0

    for video_path in videos:
        if not is_video_file(video_path):
            continue
        if has_existing_artifacts(video_path):
            skipped_artifact += 1
            continue
        if video_path in yoloed_paths or os.path.normpath(video_path) in yoloed_paths:
            skipped_path += 1
            continue

        frame_count = get_frame_count(video_path)
        if frame_count is None:
            unreadable += 1
            continue
        total_frames += frame_count
        pending_videos += 1

    if args.fps <= 0:
        print('FPS must be > 0')
        sys.exit(1)

    total_seconds = total_frames / args.fps if total_frames > 0 else 0.0
    total_days = total_seconds / 86400.0

    print(f"Root: {root_path}")
    print(f"DB: {db_path}")
    print(f"Pending videos: {pending_videos}")
    print(f"Total frames: {total_frames}")
    print(f"Average speed: {args.fps:.2f} frames/sec")
    print(f"Estimated time: {total_seconds:.2f} seconds ({total_days:.2f} days)")
    print(f"Skipped (artifacts/done): {skipped_artifact}")
    print(f"Skipped (yoloed path): {skipped_path}")
    print(f"Unreadable videos: {unreadable}")


if __name__ == '__main__':
    main()
