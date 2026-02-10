from ultralytics import YOLO
import cv2
import numpy as np
import os
import hashlib
import json
import sqlite3
import atexit
from tqdm import tqdm
import gc  # 导入垃圾回收模块
import time

# 常见的视频文件扩展名
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

DIR_ARTIFACT_SKIP_SUFFIXES = (
    '_mosaic.jpg',
    '_detection.mp4',
    '_detections.mp4',
)

# 需要排除的路径变量
_EXCLUDE_PATHS_RAW = [
    os.path.abspath(r"D:\$RECYCLE.BIN"),
    os.path.abspath(r"D:\System Volume Information"),
    # 在这里添加其他需要排除的路径
    os.path.abspath(r"D:\z\gtll 10yue"),
]
# 自动生成 UNC 变体，确保排除路径在 UNC 格式扫描时也能匹配
EXCLUDE_PATHS = list(_EXCLUDE_PATHS_RAW)
for _ep in _EXCLUDE_PATHS_RAW:
    _unc = windows_path_to_unc(_ep)
    if _unc and _unc not in EXCLUDE_PATHS:
        EXCLUDE_PATHS.append(_unc)

DIRECTORY_INDEX_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'directory_index.db')


class DirectoryIndex:
    """SQLite-backed cache for directory metadata and video listings."""

    def __init__(self, db_path=None):
        self.db_path = db_path or ':memory:'
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        with self.conn:
            self.conn.execute('PRAGMA foreign_keys = ON;')
            self.conn.execute('PRAGMA journal_mode = WAL;')
        self._ensure_schema()

    def _ensure_schema(self):
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS directories (
                    path TEXT PRIMARY KEY,
                    parent_path TEXT,
                    dir_mtime REAL,
                    last_scan REAL,
                    is_leaf INTEGER,
                    video_count INTEGER,
                    has_artifact INTEGER,
                    excluded INTEGER DEFAULT 0
                )
                """
            )
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS videos (
                    dir_path TEXT,
                    file_name TEXT,
                    file_mtime REAL,
                    file_size INTEGER,
                    is_video INTEGER,
                    PRIMARY KEY (dir_path, file_name),
                    FOREIGN KEY (dir_path) REFERENCES directories(path) ON DELETE CASCADE
                )
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_directories_parent ON directories(parent_path)
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_videos_dir ON videos(dir_path)
                """
            )

    def close(self):
        try:
            self.conn.close()
        except sqlite3.Error:
            pass

    def reopen(self, new_db_path):
        """关闭当前连接并在新路径重新打开数据库。"""
        self.close()
        self.db_path = new_db_path
        folder = os.path.dirname(new_db_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        with self.conn:
            self.conn.execute('PRAGMA foreign_keys = ON;')
            self.conn.execute('PRAGMA journal_mode = WAL;')
        self._ensure_schema()

    def _normalize_path(self, path):
        if not path:
            return None
        try:
            return normalize_posix_path_with_fs(path)
        except Exception:
            return os.path.normpath(path)

    def _build_exclusion_set(self, exclusions):
        normalized = set()
        if not exclusions:
            return normalized
        for entry in exclusions:
            if not entry:
                continue
            normalized.add(entry)
            normalized.add(os.path.normpath(entry))
            # 盘符 → UNC，确保排除路径能匹配 UNC 格式的扫描路径
            if os.name == 'nt':
                unc = windows_path_to_unc(entry)
                if unc:
                    normalized.add(unc)
                    normalized.add(os.path.normpath(unc))
            if os.name == 'posix':
                converted = windows_path_to_wsl(entry)
                if converted:
                    normalized.add(self._normalize_path(converted))
        normalized.discard(None)
        return normalized

    def _is_excluded(self, path, exclusions):
        if not exclusions:
            return False
        for ex in exclusions:
            if ex and path.startswith(ex):
                return True
        return False

    def refresh(self, root_path, exclusions=None):
        normalized_root = self._normalize_path(root_path)
        if not normalized_root or not os.path.isdir(normalized_root):
            return
        exclusion_set = self._build_exclusion_set(exclusions)
        self._refresh_directory(normalized_root, exclusion_set, parent_path=None)

    def _refresh_directory(self, dir_path, exclusions, parent_path):
        normalized = self._normalize_path(dir_path)
        if not normalized:
            return
        if self._is_excluded(normalized, exclusions):
            self._mark_excluded(normalized, parent_path)
            return
        if not os.path.isdir(normalized):
            self._remove_directory_recursive(normalized)
            return
        try:
            current_mtime = os.stat(normalized).st_mtime
        except (PermissionError, FileNotFoundError, OSError):
            self._remove_directory_recursive(normalized)
            return

        row = self._get_directory(normalized)
        if row and not row['excluded'] and row['dir_mtime'] == current_mtime:
            if row['parent_path'] != parent_path:
                with self.conn:
                    self.conn.execute(
                        "UPDATE directories SET parent_path=? WHERE path=?",
                        (parent_path, normalized)
                    )
            for child in self._get_child_paths(normalized):
                self._refresh_directory(child, exclusions, normalized)
            return

        stored_children = set(self._get_child_paths(normalized))
        child_dirs = []
        video_records = []
        has_artifact = False
        try:
            with os.scandir(normalized) as iterator:
                for entry in iterator:
                    try:
                        entry_path = self._normalize_path(entry.path)
                        if entry_path is None:
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            child_dirs.append(entry_path)
                        elif entry.is_file(follow_symlinks=False):
                            lower_name = entry.name.lower()
                            if lower_name.endswith(DIR_ARTIFACT_SKIP_SUFFIXES):
                                has_artifact = True
                            if is_video_file(entry.name) or is_video_file(entry_path):
                                try:
                                    stat_info = entry.stat(follow_symlinks=False)
                                    video_records.append(
                                        (entry.name, stat_info.st_mtime, stat_info.st_size)
                                    )
                                except OSError:
                                    continue
                    except (PermissionError, FileNotFoundError, OSError):
                        continue
        except (PermissionError, FileNotFoundError, OSError):
            self._remove_directory_recursive(normalized)
            return

        child_dirs = [c for c in child_dirs if c]
        child_dir_set = set(child_dirs)
        for removed_child in stored_children - child_dir_set:
            self._remove_directory_recursive(removed_child)

        now = time.time()
        is_leaf = 1 if not child_dirs else 0
        video_count = len(video_records)
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO directories (path, parent_path, dir_mtime, last_scan, is_leaf, video_count, has_artifact, excluded)
                VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(path) DO UPDATE SET
                    parent_path=excluded.parent_path,
                    dir_mtime=excluded.dir_mtime,
                    last_scan=excluded.last_scan,
                    is_leaf=excluded.is_leaf,
                    video_count=excluded.video_count,
                    has_artifact=excluded.has_artifact,
                    excluded=0
                """,
                (normalized, parent_path, current_mtime, now, is_leaf, video_count, 1 if has_artifact else 0)
            )
            self.conn.execute('DELETE FROM videos WHERE dir_path=?', (normalized,))
            for name, mtime, size in video_records:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO videos (dir_path, file_name, file_mtime, file_size, is_video)
                    VALUES (?, ?, ?, ?, 1)
                    """,
                    (normalized, name, mtime, size)
                )

        for child_path in child_dirs:
            self._refresh_directory(child_path, exclusions, normalized)

    def _get_directory(self, path):
        return self.conn.execute('SELECT * FROM directories WHERE path=?', (path,)).fetchone()

    def _get_child_paths(self, path):
        rows = self.conn.execute('SELECT path FROM directories WHERE parent_path=?', (path,)).fetchall()
        return [row['path'] for row in rows]

    def _remove_directory_recursive(self, path):
        normalized = self._normalize_path(path)
        if not normalized:
            return
        child_rows = self.conn.execute('SELECT path FROM directories WHERE parent_path=?', (normalized,)).fetchall()
        for row in child_rows:
            self._remove_directory_recursive(row['path'])
        with self.conn:
            self.conn.execute('DELETE FROM directories WHERE path=?', (normalized,))

    def _mark_excluded(self, path, parent_path):
        current_mtime = 0.0
        try:
            current_mtime = os.stat(path).st_mtime
        except (PermissionError, FileNotFoundError, OSError):
            current_mtime = 0.0
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO directories (path, parent_path, dir_mtime, last_scan, is_leaf, video_count, has_artifact, excluded)
                VALUES (?, ?, ?, ?, 1, 0, 0, 1)
                ON CONFLICT(path) DO UPDATE SET
                    parent_path=excluded.parent_path,
                    dir_mtime=excluded.dir_mtime,
                    last_scan=excluded.last_scan,
                    is_leaf=1,
                    video_count=0,
                    has_artifact=0,
                    excluded=1
                """,
                (path, parent_path, current_mtime, time.time())
            )
            self.conn.execute('DELETE FROM videos WHERE dir_path=?', (path,))
        for child in self._get_child_paths(path):
            self._remove_directory_recursive(child)

    def get_leaf_directories(self, root_path):
        normalized_root = self._normalize_path(root_path)
        if not normalized_root:
            return []
        like_pattern = f"{normalized_root.rstrip(os.sep)}{os.sep}%"
        rows = self.conn.execute(
            """
            SELECT path, video_count, has_artifact FROM directories
            WHERE excluded=0 AND video_count>0 AND is_leaf=1
              AND (path=? OR path LIKE ?)
            """,
            (normalized_root, like_pattern)
        ).fetchall()
        return [(row['path'], row['video_count']) for row in rows if not row['has_artifact']]

    def get_video_count(self, dir_path):
        normalized = self._normalize_path(dir_path)
        if not normalized:
            return None
        row = self._get_directory(normalized)
        if row and not row['excluded']:
            return row['video_count'] or 0
        return None

    def directory_has_artifacts(self, dir_path):
        normalized = self._normalize_path(dir_path)
        if not normalized:
            return None
        row = self._get_directory(normalized)
        if row and not row['excluded']:
            return bool(row['has_artifact'])
        return None

    def get_videos(self, dir_path):
        normalized = self._normalize_path(dir_path)
        if not normalized:
            return []
        rows = self.conn.execute(
            'SELECT file_name FROM videos WHERE dir_path=? ORDER BY file_name',
            (normalized,)
        ).fetchall()
        return [row['file_name'] for row in rows]


DIRECTORY_INDEX = DirectoryIndex()
atexit.register(DIRECTORY_INDEX.close)


class PauseRequested(Exception):
    """Raised to request a graceful stop with checkpoint saved."""


CHECKPOINT_SUFFIX = '.checkpoint.json'


def _truthy_env(name, default=False):
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in ('1', 'true', 'yes', 'y', 'on')


def _get_pause_file_path():
    """Return the pause flag path (best-effort)."""
    explicit = os.environ.get('FINDINVIDEO_PAUSE_FILE')
    if explicit:
        return explicit
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        base_dir = os.getcwd()
    return os.path.join(base_dir, 'pause.flag')


def _pause_requested(pause_file_path=None):
    path = pause_file_path or _get_pause_file_path()
    try:
        return bool(path) and os.path.exists(path)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 衍生文件（artifact）命名策略
# ---------------------------------------------------------------------------
# 新策略:  直接使用原始文件名基础名（不含扩展名），不做任何字符替换或 hash
# 后缀，确保在 Windows 资源管理器按名称排序时衍生文件紧跟在原视频旁边。
# 例: 原视频 "1 (44).mp4" → "1 (44)_frames.mp4", "1 (44)_mosaic.jpg" 等。
# ---------------------------------------------------------------------------

def safe_artifact_basename(video_path):
    """返回用于创建衍生文件的基础名（= 原视频文件名去掉扩展名）。

    相比旧版（sanitized + hash），这种方式保持了和原文件完全相同的前缀，
    使得 Windows 资源管理器 StrCmpLogicalW 排序时衍生文件紧贴原视频。
    """
    return os.path.splitext(os.path.basename(video_path))[0]


def _sanitize_basename(video_path):
    """将文件名中的非 alnum / - / _ 字符替换为 _（旧格式辅助函数）。"""
    base_name = os.path.basename(os.path.splitext(video_path)[0])
    sanitized = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in base_name)
    return sanitized or 'video'


def legacy_artifact_basename(video_path, max_length=80):
    """旧格式 v2: sanitized_name + stat-based hash — 仅用于检测已有衍生文件。"""
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
    """旧格式 v1: sanitized_name + path-based hash — 仅用于检测已有衍生文件。"""
    sanitized = _sanitize_basename(video_path)
    digest = hashlib.md5(str(video_path).encode('utf-8', 'ignore')).hexdigest()[:8]
    limit = max(8, max_length - len(digest) - 1)
    if len(sanitized) > limit:
        sanitized = sanitized[:limit]
    return f"{sanitized}_{digest}"


def _checkpoint_path(video_path):
    video_dir = os.path.dirname(video_path) or '.'
    base = safe_artifact_basename(video_path)
    return os.path.join(video_dir, base + CHECKPOINT_SUFFIX)


def _load_checkpoint(video_path):
    path = _checkpoint_path(video_path)
    try:
        if not os.path.exists(path):
            return None
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
        try:
            st = os.stat(video_path)
            if data.get('size') not in (None, st.st_size):
                return None
            if data.get('mtime') is not None and abs(float(data.get('mtime')) - float(st.st_mtime)) > 2.0:
                return None
        except Exception:
            pass
        return data
    except Exception:
        return None


def _save_checkpoint(video_path, next_frame, detections, last_detected):
    path = _checkpoint_path(video_path)
    payload = {
        'version': 1,
        'next_frame': int(max(0, next_frame or 0)),
        'detections': detections or [],
        'last_detected': float(last_detected) if last_detected is not None else -5.0,
        'saved_at': time.time(),
    }
    try:
        st = os.stat(video_path)
        payload['size'] = st.st_size
        payload['mtime'] = st.st_mtime
    except Exception:
        pass
    try:
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8', errors='ignore') as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception:
        pass


def _clear_checkpoint(video_path):
    path = _checkpoint_path(video_path)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def is_video_file(file_path):
    """检查文件是否为视频文件"""
    # 直接用原始文件名，避免特殊字符被截断
    _, ext = os.path.splitext(file_path.lower())
    base = file_path.lower()
    if base.endswith('_frames.mp4') or base.endswith('_objects.mp4') or base.endswith('_detections.mp4'):
        return False
    # 续跑/分段输出文件，避免被当作输入视频再次处理
    if '_frames.part' in base or '_objects.part' in base or '_detections.part' in base:
        return False
    return ext in VIDEO_EXTENSIONS


def has_existing_artifacts(video_path):
    """检查当前视频是否已生成同名衍生文件。"""
    # 若存在 checkpoint，说明上次是“暂停/中断未完成”，不能当作已完成跳过
    try:
        if os.path.exists(_checkpoint_path(video_path)):
            return False
    except Exception:
        pass
    video_dir = os.path.dirname(video_path) or '.'
    bases = [safe_artifact_basename(video_path),
             legacy_artifact_basename(video_path),
             _legacy_artifact_basename_v1(video_path)]
    for base in bases:
        # Fast marker first
        done_path = os.path.join(video_dir, base + DONE_SUFFIX)
        if os.path.exists(done_path):
            return True
        for suffix in ARTIFACT_SUFFIXES:
            artifact_path = os.path.join(video_dir, base + suffix)
            if os.path.exists(artifact_path):
                return True
    return False


def write_done_marker(video_path):
    """Write a small marker file so future runs can skip without hashing."""
    try:
        video_dir = os.path.dirname(video_path) or '.'
        base = safe_artifact_basename(video_path)
        marker = os.path.join(video_dir, base + DONE_SUFFIX)
        if os.path.exists(marker):
            return
        with open(marker, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(f"done\n")
    except Exception:
        pass

def is_leaf_directory(dir_path):
    """检查目录是否为叶子节点（不包含子目录）"""
    try:
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path):
                return False
        return True
    except (PermissionError, FileNotFoundError):
        return False

def count_videos_in_directory(dir_path):
    """统计目录中的视频文件数量"""
    cached = DIRECTORY_INDEX.get_video_count(dir_path)
    if cached is not None:
        return cached
    count = 0
    try:
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path) and is_video_file(file_path):
                count += 1
    except (PermissionError, FileNotFoundError):
        pass
    return count

def find_leaf_directories_with_videos(root_path, exclusions=None, refresh_index=True):
    """使用目录索引查找包含视频文件的叶子节点目录。"""
    if exclusions is None:
        exclusions = EXCLUDE_PATHS
    try:
        if refresh_index:
            DIRECTORY_INDEX.refresh(root_path, exclusions)
        leaf_dirs = DIRECTORY_INDEX.get_leaf_directories(root_path)
    except Exception as exc:
        print(f"索引查找叶子目录失败，退回遍历模式: {exc}")
        leaf_dirs = []
    if not leaf_dirs:
        # 兼容性回退
        try:
            for root, dirs, files in os.walk(root_path):
                is_excluded = any(root.startswith(ex_path) for ex_path in exclusions)
                if is_excluded:
                    continue
                dirs[:] = [d for d in dirs if not any(os.path.join(root, d).startswith(ex_path) for ex_path in exclusions)]
                if is_leaf_directory(root):
                    video_count = count_videos_in_directory(root)
                    if video_count > 0:
                        leaf_dirs.append((root, video_count))
        except (PermissionError, FileNotFoundError) as e:
            print(f"警告: 无法访问目录 '{root_path}': {e}")
    leaf_dirs.sort(key=lambda x: x[1], reverse=True)
    return leaf_dirs

def save_mosaic_batch(crops_batch, batch_idx, dir_name, base_name, max_cols=8):
    """保存一批检测到的目标区域为拼接图像"""
    if not crops_batch:
        return
    
    rows = []
    row = []
    for i, crop in enumerate(crops_batch):
        row.append(crop)
        if (i + 1) % max_cols == 0:
            rows.append(np.hstack(row))
            row = []
    
    if row:
        missing = max_cols - len(row)
        blank = np.zeros_like(crops_batch[0])
        row.extend([blank] * missing)
        rows.append(np.hstack(row))
    
    if rows:
        batch_mosaic = np.vstack(rows)
        # 根据批次保存不同的文件名
        mosaic_path = os.path.join(dir_name, f"{base_name}_mosaic_{batch_idx}.jpg")
        cv2.imwrite(mosaic_path, batch_mosaic)
        print(f"已保存第{batch_idx}批拼接图片至: {mosaic_path}")

    # 清理内存
    del crops_batch, rows, row, batch_mosaic
    gc.collect()
    
def detect_objects_in_video(video_path, target_class,
                          show_window=False, save_crops=False,
                          save_training_data=False,
                          all_objects=False):
    # 如果不开启全量检测，则保证 target_class 为列表
    if not all_objects and isinstance(target_class, str):
        target_class = [target_class]

    pause_file = _get_pause_file_path()
    resume_enabled = _truthy_env('FINDINVIDEO_RESUME', default=True)
    imgsz_env = os.environ.get('FINDINVIDEO_IMGSZ')
    try:
        imgsz = int(imgsz_env) if imgsz_env else 1920
    except Exception:
        imgsz = 1920

    # 加载模型
    model = YOLO('models/yolov11l-face.pt')

    video_dir = os.path.dirname(video_path)
    artifact_base = safe_artifact_basename(video_path)
    txt_save_path = os.path.join(video_dir, artifact_base + '.txt')
    mosaic_path = os.path.join(video_dir, artifact_base + '_mosaic.jpg')
    video_save_path = os.path.join(video_dir, artifact_base + '_frames.mp4')

    # 若需要生成训练数据，则构造保存目录及生成 classes.txt 文件
    if save_training_data:
        training_folder = os.path.join(video_dir, "training_data")
        os.makedirs(training_folder, exist_ok=True)
        classes_txt = os.path.join(training_folder, "classes.txt")
        if not os.path.exists(classes_txt):
            with open(classes_txt, "w") as f:
                if all_objects:
                    # 写入模型中所有的类别，按 key 排序
                    for key in sorted(model.names.keys()):
                        f.write(model.names[key] + "\n")
                else:
                    for cls in target_class:
                        f.write(cls + "\n")
    
    ckpt = _load_checkpoint(video_path) if resume_enabled else None
    start_frame = int(ckpt.get('next_frame', 0)) if ckpt else 0
    detections = list(ckpt.get('detections', [])) if ckpt else []
    last_detected = float(ckpt.get('last_detected', detections[-1] if detections else -5.0)) if ckpt else -5

    # 视频处理初始化
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return []
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_safe = fps if fps and fps > 0 else 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if start_frame > 0:
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        except Exception:
            pass

    # 初始化进度条
    pbar = tqdm(total=total_frames, initial=min(start_frame, total_frames), desc=f"处理视频: {os.path.basename(video_path)}")

    frame_count = start_frame
    
    # 截图存储配置（用于拼接大图）
    crop_size = (160, 160)  # 统一缩放到的小图尺寸
    max_cols = 8  # 拼接大图每行最多显示数量
    crops_batch = []  # 存储当前批次的目标区域
    batch_size = 200  # 每批处理的目标数量
    batch_idx = 1  # 批次计数器
    
    paused = False
    try:
        while cap.isOpened():
            if _pause_requested(pause_file):
                _save_checkpoint(video_path, next_frame=frame_count, detections=detections, last_detected=last_detected)
                paused = True
                raise PauseRequested()
            success, frame = cap.read()
            if not success:
                break
            
            # 更新进度条
            pbar.update(1)
            
            if save_training_data:
                frame_annotations = []
            
            current_time = frame_count / fps_safe
            results = model.predict(frame, conf=0.5, verbose=False)
            detected = False
            
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    # 如果全量检测开启，或者检测到的类别在 target_class 内则处理
                    if all_objects or model.names[cls_id] in target_class:
                        if current_time - last_detected >= 0.1:
                            detections.append(current_time)
                            last_detected = current_time
                            detected = True
                        
                        if save_crops:
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = map(int, xyxy)
                            # 边界检查
                            h, w = frame.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            
                            if x2 > x1 and y2 > y1:  # 确保有效区域
                                crop = frame[y1:y2, x1:x2].copy()  # 使用.copy()避免引用原始帧
                                try:
                                    resized = cv2.resize(crop, crop_size)
                                    crops_batch.append(resized)
                                    
                                    # 批处理，避免占用过多内存
                                    if len(crops_batch) >= batch_size:
                                        dir_name = os.path.dirname(video_path)
                                        base_name = os.path.splitext(os.path.basename(video_path))[0]
                                        save_mosaic_batch(crops_batch, batch_idx, dir_name, base_name, max_cols)
                                        batch_idx += 1
                                        crops_batch = []
                                        # 强制垃圾回收
                                        gc.collect()
                                except Exception as e:
                                    print(f"处理裁剪图像时出错: {e}")
                        
                        if show_window:
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = map(int, xyxy)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        if save_training_data:
                            h, w, _ = frame.shape
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = map(int, xyxy)
                            cx = ((x1 + x2) / 2) / w
                            cy = ((y1 + y2) / 2) / h
                            bw = (x2 - x1) / w
                            bh = (y2 - y1) / h
                            # 如果全量检测，类别编号直接用 cls_id；否则使用 target_class 的索引
                            if all_objects:
                                class_index = cls_id
                            else:
                                class_index = target_class.index(model.names[cls_id])
                            annotation_line = f"{class_index} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                            frame_annotations.append(annotation_line)
            
            if show_window and detected:
                cv2.imshow('Detection Preview', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if save_training_data and frame_annotations:
                video_base = os.path.splitext(os.path.basename(video_path))[0]
                training_image_path = os.path.join(training_folder, f"{video_base}_{frame_count}.jpg")
                training_annotation_path = os.path.splitext(training_image_path)[0] + ".txt"
                cv2.imwrite(training_image_path, frame)
                with open(training_annotation_path, 'w') as f:
                    for line in frame_annotations:
                        f.write(line + "\n")
            
            frame_count += 1
            
            # 每100帧清理一次内存
            if frame_count % 100 == 0:
                # 手动触发垃圾回收
                gc.collect()
                
                # 为减轻内存压力，暂停一小段时间
                if frame_count % 500 == 0:
                    time.sleep(0.1)
                
            # 释放当前帧
            del frame
    
    except KeyboardInterrupt:
        _save_checkpoint(video_path, next_frame=frame_count, detections=detections, last_detected=last_detected)
        paused = True
    except PauseRequested:
        paused = True
    except Exception as e:
        print(f"处理视频时发生错误: {e}")
    
    finally:
        # 确保资源释放
        cap.release()
        pbar.close()
        if show_window:
            cv2.destroyAllWindows()

    if not paused and (frame_count >= total_frames or total_frames == 0):
        _clear_checkpoint(video_path)
    
    # 处理剩余的裁剪图像
    if save_crops and crops_batch:
        dir_name = os.path.dirname(video_path)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        save_mosaic_batch(crops_batch, batch_idx, dir_name, base_name, max_cols)
    
    # 保存检测结果
    txt_save_path = video_path + ".txt"
    with open(txt_save_path, 'w') as f:
        f.write("检测到目标的时间位置（秒）:\n")
        for t in detections:
            f.write(f"{t:.2f}\n")
    print(f"已保存检测时间戳至: {txt_save_path}")
    
    # 创建总拼接图
    if save_crops and batch_idx > 1:
        try:
            dir_name = os.path.dirname(video_path)
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            # 合并所有批次图片
            mosaic_images = []
            for i in range(1, batch_idx + 1):
                mosaic_path = os.path.join(dir_name, f"{base_name}_mosaic_{i}.jpg")
                if os.path.exists(mosaic_path):
                    img = cv2.imread(mosaic_path)
                    if img is not None:
                        mosaic_images.append(img)
            
            # 如果有多个批次图片，创建最终拼接图
            if len(mosaic_images) > 0:
                # 计算总拼接图的高度
                final_height = sum(img.shape[0] for img in mosaic_images)
                final_width = max(img.shape[1] for img in mosaic_images)
                
                # 创建大小合适的画布
                final_mosaic = np.zeros((final_height, final_width, 3), dtype=np.uint8)
                
                # 填充图像
                y_offset = 0
                for img in mosaic_images:
                    h, w = img.shape[:2]
                    final_mosaic[y_offset:y_offset+h, 0:w] = img
                    y_offset += h
                
                # 保存最终拼接图
                final_mosaic_path = os.path.join(dir_name, f"{base_name}_mosaic.jpg")
                cv2.imwrite(final_mosaic_path, final_mosaic)
                print(f"已保存最终拼接图片至: {final_mosaic_path}")
                
                # 删除临时批次图片
                for i in range(1, batch_idx + 1):
                    temp_path = os.path.join(dir_name, f"{base_name}_mosaic_{i}.jpg")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        except Exception as e:
            print(f"创建最终拼接图时出错: {e}")
    
    # 清理内存
    gc.collect()
    return detections

def get_yoloed_md5_path():
    # 支持路径转换；默认使用处理目标目录下的 md5_list/yoloed.txt
    yoloed_path = _get_env_path('FINDINVIDEO_YOLOED_PATH')
    if not yoloed_path:
        base_dir = _PROCESSING_ROOT_DIR
        if not base_dir:
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
            except Exception:
                base_dir = os.getcwd()
        yoloed_path = os.path.join(base_dir, 'md5_list', 'yoloed.txt')
    if os.name == 'posix':
        yoloed_path = windows_path_to_wsl(yoloed_path)
        if yoloed_path:
            yoloed_path = normalize_posix_path_with_fs(yoloed_path)
    return yoloed_path


def get_file_md5_cached(file_path):
    """Cached wrapper around get_file_md5 to avoid recompute in one run."""
    if not file_path:
        return None
    try:
        st = os.stat(file_path)
        key = (str(file_path), int(getattr(st, 'st_size', 0) or 0), float(getattr(st, 'st_mtime', 0.0) or 0.0))
    except Exception:
        key = (str(file_path), None, None)
    cached = _FILE_MD5_CACHE.get(key)
    if cached is not None:
        return cached
    digest = get_file_md5(file_path)
    # Best-effort bound memory
    if len(_FILE_MD5_CACHE) > 2048:
        _FILE_MD5_CACHE.clear()
    _FILE_MD5_CACHE[key] = digest
    return digest

def ensure_yoloed_storage():
    path = get_yoloed_md5_path()
    if not path:
        return None
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception as e:
            print(f"创建MD5存储目录失败: {e}")
            return None
    if not os.path.exists(path):
        try:
            open(path, 'a', encoding='utf-8').close()
        except Exception as e:
            print(f"初始化MD5存储文件失败: {e}")
            return None
    return path

def load_yoloed_md5(reload=False):
    global _YOLOED_MD5_CACHE, _YOLOED_MD5_CACHE_MTIME, _YOLOED_PATH_CACHE
    path = ensure_yoloed_storage()
    if not path:
        _YOLOED_MD5_CACHE = set()
        _YOLOED_MD5_CACHE_MTIME = None
        _YOLOED_PATH_CACHE = set()
        return _YOLOED_MD5_CACHE

    try:
        mtime = os.path.getmtime(path)
    except Exception:
        mtime = None

    if (
        not reload
        and _YOLOED_MD5_CACHE is not None
        and _YOLOED_PATH_CACHE is not None
        and mtime is not None
        and _YOLOED_MD5_CACHE_MTIME == mtime
    ):
        return _YOLOED_MD5_CACHE

    md5_set = set()
    path_set = set()
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                entry = line.strip()
                if not entry:
                    continue
                parts = entry.split('|', 1)
                md5_part = parts[0].strip()
                if md5_part:
                    md5_set.add(md5_part)

                if len(parts) == 2:
                    raw_path = parts[1].strip()
                    if raw_path:
                        path_set.add(raw_path)
                        path_set.add(os.path.normpath(raw_path))
                        canon = canonical_video_path(raw_path)
                        if canon:
                            path_set.add(canon)
                        # 跨平台: 若为 WSL 路径 (/mnt/<drive>/...), 同时生成 Windows 盘符 + UNC 变体
                        if raw_path.startswith('/mnt/') and len(raw_path) > 5 and raw_path[5].isalpha():
                            _drv = raw_path[5]
                            _rest = raw_path[6:].lstrip('/').replace('/', '\\')
                            _win = f"{_drv.upper()}:\\{_rest}" if _rest else f"{_drv.upper()}:\\"
                            path_set.add(_win)
                            _win_norm = os.path.normpath(_win)
                            path_set.add(_win_norm)
                            _unc = canonical_video_path(_win)
                            if _unc:
                                path_set.add(_unc)
    except Exception as e:
        print(f"读取已识别MD5列表失败: {e}")
    _YOLOED_MD5_CACHE = md5_set
    _YOLOED_PATH_CACHE = path_set
    _YOLOED_MD5_CACHE_MTIME = mtime
    return _YOLOED_MD5_CACHE

def append_yoloed_md5(md5, file_path=None):
    global _YOLOED_MD5_CACHE, _YOLOED_MD5_CACHE_MTIME, _YOLOED_PATH_CACHE
    path = ensure_yoloed_storage()
    try:
        lock_path = path + '.lock'
        release = _with_lockfile(lock_path, timeout_seconds=30, stale_seconds=3600)
        entry = md5 if not file_path else f"{md5}|{file_path}"
        try:
            with open(path, 'a', encoding='utf-8', errors='ignore') as f:
                f.write(entry + '\n')
        finally:
            release()

        # refresh cache mtime
        try:
            _YOLOED_MD5_CACHE_MTIME = os.path.getmtime(path)
        except Exception:
            _YOLOED_MD5_CACHE_MTIME = None
        if _YOLOED_MD5_CACHE is not None:
            _YOLOED_MD5_CACHE.add(md5)
        if file_path and _YOLOED_PATH_CACHE is not None:
            _fp = str(file_path)
            _YOLOED_PATH_CACHE.add(_fp)
            _YOLOED_PATH_CACHE.add(os.path.normpath(_fp))
            canon = canonical_video_path(file_path)
            if canon:
                _YOLOED_PATH_CACHE.add(canon)
            # 跨平台: WSL 路径同时添加 Windows 变体
            if _fp.startswith('/mnt/') and len(_fp) > 5 and _fp[5].isalpha():
                _drv = _fp[5]
                _rest = _fp[6:].lstrip('/').replace('/', '\\')
                _win = f"{_drv.upper()}:\\{_rest}" if _rest else f"{_drv.upper()}:\\"
                _YOLOED_PATH_CACHE.add(_win)
                _YOLOED_PATH_CACHE.add(os.path.normpath(_win))
                _unc = canonical_video_path(_win)
                if _unc:
                    _YOLOED_PATH_CACHE.add(_unc)
    except Exception as e:
        print(f"写入已识别MD5失败: {e}")


def is_path_already_yoloed(file_path):
    """Fast check using yoloed.txt stored paths (no hashing)."""
    load_yoloed_md5(reload=False)
    if not _YOLOED_PATH_CACHE:
        return False
    candidates = []
    p = str(file_path)
    candidates.append(p)
    candidates.append(os.path.normpath(p))
    canon = canonical_video_path(p)
    if canon:
        candidates.append(canon)
    return any(c in _YOLOED_PATH_CACHE for c in candidates)

def process_directory_videos(dir_path, target_item, all_objects_switch=False, skip_long_videos=True):
    """处理目录中的所有视频文件"""
    if os.name == 'posix' and is_windows_style_path(dir_path):
        converted = windows_path_to_wsl(dir_path)
        if converted:
            dir_path = normalize_posix_path_with_fs(converted)
    if directory_has_artifact_outputs(dir_path):
        print(f"检测到目录内已存在_mosaic/_detection文件，跳过目录: {dir_path}")
        return
    video_files = []
    
    try:
        for file in os.listdir(dir_path):
            if is_video_file(file):
                file_path = os.path.join(dir_path, file)
                if should_process(file_path):
                    # 检查视频时长
                    cap = cv2.VideoCapture(file_path)
                    if not cap.isOpened():
                        print(f"无法打开视频: {file_path}")
                        continue
                        
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    cap.release()
                    duration = frame_count / fps if fps > 0 else float('inf')
                    
                    if duration <= 3600:  # 小于等于1小时的视频
                        video_files.append(file_path)
                    else:
                        print(f"视频时长 {duration:.2f}秒超过一小时，跳过处理: {file_path}")
                else:
                    print(f"已存在拼接图片，跳过处理: {file_path}")
    except (PermissionError, FileNotFoundError) as e:
        print(f"警告: 无法访问目录 '{dir_path}': {e}")
        return
    
    # 处理视频文件
    for video_file, duration, claim_path, md5 in video_files:
        if duration == float('inf'):
            print(f"提示: 无法获取视频时长，仍尝试处理: {video_file}")
        print(f"开始处理视频文件: {video_file}")
        detect_objects_in_video(video_file, target_item,
                                show_window=False,
                                save_crops=True,
                                save_training_data=False,
                                all_objects=all_objects_switch)
        # 视频处理完成后强制垃圾回收
        gc.collect()
        # 短暂休眠，让系统有时间释放资源
        time.sleep(1)

if __name__ == "__main__":
    video_path = r"D:\z"  # 可设置为视频文件或目录
    # 如要检测所有模型内对象，则将 target_item 设置为任意值并启用全量检测开关
    target_item = "face"  # 当 all_objects 为 True 时，该值不再限制检测
    all_objects_switch = False  # 设置为 True 表示显示所有检测对象
    
    # 新增功能：按叶子节点视频数量排序处理
    use_leaf_node_processing = True  # 设置为 True 启用叶子节点处理模式
    
    if use_leaf_node_processing and os.path.isdir(video_path):
        print(f"启用叶子节点处理模式，正在扫描目录: {video_path}")
        print("正在查找包含视频文件的叶子节点目录...")

        root_video_count = count_videos_in_directory(root_path)
        if root_video_count > 0:
            print(f"\n=== 处理根目录: {root_path} ({root_video_count} 个视频) ===")
            process_directory_videos(root_path, target_item, all_objects_switch, skip_long_videos)

        leaf_dirs = find_leaf_directories_with_videos(root_path, EXCLUDE_PATHS, refresh_index=False)

        if not leaf_dirs:
            print(f"未找到包含视频文件的叶子节点目录")
        else:
            print(f"\n找到 {len(leaf_dirs)} 个包含视频文件的叶子节点目录:")
            for i, (dir_path, video_count) in enumerate(leaf_dirs, 1):
                relative_path = os.path.relpath(dir_path, video_path)
                print(f"{i:3d}. {relative_path} ({video_count} 个视频文件)")
            
            print(f"\n开始按视频数量从多到少的顺序处理叶子节点目录...")
            
            # 按顺序处理每个叶子目录
            for i, (dir_path, video_count) in enumerate(leaf_dirs, 1):
                relative_path = os.path.relpath(dir_path, video_path)
                print(f"\n=== 处理第 {i}/{len(leaf_dirs)} 个目录: {relative_path} ({video_count} 个视频) ===")
                process_directory_videos(dir_path, target_item, all_objects_switch)
                
                # 每处理完一个目录后强制垃圾回收
                gc.collect()
                # 让系统有时间释放资源
                time.sleep(2)
    
    # 原有的处理逻辑（当 use_leaf_node_processing 为 False 时使用）
    elif os.path.isdir(video_path):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        for root, dirs, files in os.walk(root_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in video_extensions:
                    file_path = os.path.join(root, file)
                    if should_process(file_path):
                        # 获取视频时长
                        cap = cv2.VideoCapture(file_path)
                        if not cap.isOpened():
                            print(f"无法打开视频: {file_path}")
                            continue
                            
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        cap.release()
                        duration = frame_count / fps if fps > 0 else float('inf')
                        if duration > 3600:
                            print(f"视频时长 {duration:.2f}秒超过一小时，跳过处理: {file_path}")
                            continue
                        
                        print(f"开始处理视频文件: {file_path}")
                        detect_objects_in_video(file_path, target_item,
                                                show_window=False,
                                                save_crops=True,
                                                save_training_data=True,
                                                all_objects=all_objects_switch)
                        
                        # 强制垃圾回收
                        gc.collect()
                        time.sleep(1)
                    else:
                        print(f"已存在拼接图片，跳过处理: {file_path}")
    else:
        # 处理单个视频文件前检查视频时长
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            duration = frame_count / fps if fps > 0 else float('inf')
            if duration > 3600:
                print(f"视频时长 {duration:.2f}秒超过一小时，跳过处理: {video_path}")
            else:
                if should_process(video_path):
                    detect_objects_in_video(video_path, target_item,
                                           show_window=False,
                                           save_crops=True,
                                           save_training_data=True,
                                           all_objects=all_objects_switch)
                else:
                    print(f"已存在拼接图片，跳过处理: {video_path}")