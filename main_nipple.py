from ultralytics import YOLO
import cv2
import numpy as np
import os
import hashlib
import json
import sqlite3
import atexit
import subprocess
import errno
import socket
import ntpath
import ctypes
from ctypes import wintypes
from tqdm import tqdm
import gc  # 导入垃圾回收模块
import time

_PROCESSING_ROOT_DIR = None
_YOLOED_MD5_CACHE = None
_YOLOED_MD5_CACHE_MTIME = None
_YOLOED_PATH_CACHE = None
_FILE_MD5_CACHE = {}


def is_windows_style_path(path):
    if not path or not isinstance(path, str):
        return False
    if len(path) < 2:
        return False
    drive, sep = path[0], path[1]
    return drive.isalpha() and sep == ':'


def windows_path_to_wsl(path):
    if not path:
        return None
    try:
        result = subprocess.run(['wslpath', '-a', path], check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding='utf-8', errors='ignore')
        converted = result.stdout.strip()
        return converted or None
    except Exception:
        if is_windows_style_path(path):
            drive = path[0].lower()
            remainder = path[2:].replace('\\', '/').lstrip('/')
            return f"/mnt/{drive}/{remainder}" if remainder else f"/mnt/{drive}"
        return None


def wsl_path_to_windows(path):
    if not path:
        return None
    try:
        result = subprocess.run(['wslpath', '-w', path], check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding='utf-8', errors='ignore')
        converted = result.stdout.strip()
        return converted or None
    except Exception:
        prefix = '/mnt/'
        if path.startswith(prefix) and len(path) > len(prefix):
            drive = path[len(prefix)]
            if drive.isalpha():
                remainder = path[len(prefix) + 1:].lstrip('/')
                return f"{drive.upper()}:\\{remainder.replace('/', '\\')}" if remainder else f"{drive.upper()}:\\"
        return None


def normalize_posix_path_with_fs(path):
    if not path:
        return None
    try:
        resolved = os.path.realpath(path)
        return resolved
    except OSError:
        return os.path.normpath(path)


def windows_path_to_unc(path):
    if os.name != 'nt' or not path:
        return None
    try:
        drive, tail = ntpath.splitdrive(str(path))
        if not drive or len(drive) < 2 or drive[1] != ':':
            return None
        drive_root = drive[0].upper() + ':'

        mpr = ctypes.WinDLL('mpr')
        WNetGetConnectionW = mpr.WNetGetConnectionW
        WNetGetConnectionW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
        WNetGetConnectionW.restype = wintypes.DWORD

        buf_len = wintypes.DWORD(1024)
        buf = ctypes.create_unicode_buffer(buf_len.value)
        rc = WNetGetConnectionW(drive_root, buf, ctypes.byref(buf_len))
        if rc != 0:
            return None
        unc_root = buf.value
        remainder = tail.lstrip('\\/')
        return unc_root.rstrip('\\/') + ('\\' + remainder if remainder else '')
    except Exception:
        return None


def unc_to_drive_letter(path):
    """将UNC路径（如 \\\\192.168.6.100\\d\\...）转换回本机映射的盘符路径（如 D:\\...）。
    遍历所有已映射的网络驱动器，找到匹配的UNC前缀并替换为盘符。"""
    if os.name != 'nt' or not path:
        return None
    p = str(path)
    if not p.startswith('\\\\'):
        return None
    try:
        mpr = ctypes.WinDLL('mpr')
        WNetGetConnectionW = mpr.WNetGetConnectionW
        WNetGetConnectionW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)]
        WNetGetConnectionW.restype = wintypes.DWORD

        p_norm = os.path.normpath(p).lower()
        for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
            drive = f'{letter}:'
            buf_len = wintypes.DWORD(1024)
            buf = ctypes.create_unicode_buffer(buf_len.value)
            rc = WNetGetConnectionW(drive, buf, ctypes.byref(buf_len))
            if rc != 0:
                continue
            unc_root = os.path.normpath(buf.value).lower()
            if p_norm == unc_root or p_norm.startswith(unc_root + '\\'):
                remainder = os.path.normpath(p)[len(unc_root):]
                return drive + remainder
        return None
    except Exception:
        return None


def _safe_relpath(path, start):
    """安全计算相对路径，处理UNC路径与盘符路径不在同一挂载点的情况。"""
    try:
        return os.path.relpath(path, start)
    except ValueError:
        # 尝试将UNC路径转换为盘符路径后重试
        converted_path = unc_to_drive_letter(path) if path.startswith('\\\\') else path
        converted_start = unc_to_drive_letter(start) if start.startswith('\\\\') else start
        p = converted_path or path
        s = converted_start or start
        try:
            return os.path.relpath(p, s)
        except ValueError:
            # 仍然无法计算相对路径，返回原始路径
            return path


def canonical_video_path(path):
    if not path:
        return None
    p = str(path)
    if os.name == 'nt':
        p_norm = ntpath.normpath(p)
        unc = windows_path_to_unc(p_norm)
        if unc:
            return ntpath.normpath(unc)
        return p_norm
    return normalize_posix_path_with_fs(p)


def _get_env_path(name):
    value = os.environ.get(name)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def get_shared_state_dir():
    shared = _get_env_path('FINDINVIDEO_SHARED_STATE_DIR')
    if not shared:
        return None
    if os.name == 'posix' and is_windows_style_path(shared):
        converted = windows_path_to_wsl(shared)
        if converted:
            shared = normalize_posix_path_with_fs(converted)
    return shared


def ensure_shared_state_dir():
    shared = get_shared_state_dir()
    if not shared:
        return None
    try:
        os.makedirs(shared, exist_ok=True)
        return shared
    except Exception as e:
        print(f"共享状态目录不可用: {shared}, 错误: {e}")
        return None


def _atomic_create_file(path, content):
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        fd = os.open(path, flags)
    except FileExistsError:
        return False
    except OSError as e:
        if getattr(e, 'errno', None) == errno.EEXIST:
            return False
        raise
    try:
        with os.fdopen(fd, 'w', encoding='utf-8', errors='ignore') as f:
            f.write(content)
        return True
    except Exception:
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            os.unlink(path)
        except Exception:
            pass
        raise


def _with_lockfile(lock_path, timeout_seconds=30, stale_seconds=3600):
    start = time.time()
    payload = f"host={socket.gethostname()} pid={os.getpid()} start={start}\n"
    while True:
        try:
            created = _atomic_create_file(lock_path, payload)
            if created:
                def _release():
                    try:
                        os.unlink(lock_path)
                    except Exception:
                        pass
                return _release
        except Exception:
            pass
        try:
            st = os.stat(lock_path)
            age = time.time() - float(getattr(st, 'st_mtime', time.time()) or time.time())
            if stale_seconds > 0 and age > stale_seconds:
                try:
                    os.unlink(lock_path)
                except Exception:
                    pass
        except Exception:
            pass
        if time.time() - start >= timeout_seconds:
            raise TimeoutError(f"等待锁超时: {lock_path}")
        time.sleep(0.2)

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

# 在判断叶子目录时忽略这些子目录名（工具生成的输出目录，不影响目录结构判断）
_IGNORED_SUBDIRS = {'_detected', 'yolov5_output', '__pycache__', '.git', '$RECYCLE.BIN', 'System Volume Information'}

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
            # 已处理视频记录表（基于文件MD5，跨机器通用）
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS processed_videos (
                    file_md5 TEXT PRIMARY KEY,
                    video_path TEXT,
                    processed_at REAL,
                    detection_count INTEGER,
                    model_name TEXT
                )
                """
            )
            self.conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_processed_videos_path ON processed_videos(video_path)
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
        all_file_names_lower = set()  # 收集目录内所有文件名（小写），用于判断衍生文件
        try:
            with os.scandir(normalized) as iterator:
                for entry in iterator:
                    try:
                        entry_path = self._normalize_path(entry.path)
                        if entry_path is None:
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name not in _IGNORED_SUBDIRS:
                                child_dirs.append(entry_path)
                        elif entry.is_file(follow_symlinks=False):
                            all_file_names_lower.add(entry.name.lower())
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

        # 精确判断：所有视频是否都已有衍生文件（纯内存比对，无额外I/O）
        if video_records:
            processed_count = 0
            for v_name, _, _ in video_records:
                v_base = os.path.splitext(v_name)[0].lower()
                # 检查当前命名格式的衍生文件（base + artifact suffix）
                if any((v_base + s.lower()) in all_file_names_lower for s in ARTIFACT_SUFFIXES):
                    processed_count += 1
            has_artifact = (processed_count == len(video_records))
        else:
            has_artifact = False

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
        return [(row['path'], row['video_count'], bool(row['has_artifact'])) for row in rows]

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

    def get_directory_info(self, dir_path):
        """获取目录的 has_artifact 和 dir_mtime 信息，返回 (has_artifact, dir_mtime) 或 None。"""
        normalized = self._normalize_path(dir_path)
        if not normalized:
            return None
        row = self._get_directory(normalized)
        if row and not row['excluded']:
            return (bool(row['has_artifact']), row['dir_mtime'])
        return None

    def mark_directory_processed(self, dir_path):
        """将目录标记为已全部处理（has_artifact=1），同时更新 dir_mtime 为当前文件系统的值。"""
        normalized = self._normalize_path(dir_path)
        if not normalized:
            return
        try:
            current_mtime = os.stat(normalized).st_mtime
        except (PermissionError, FileNotFoundError, OSError):
            current_mtime = None
        try:
            with self.conn:
                self.conn.execute(
                    """
                    UPDATE directories SET has_artifact=1, dir_mtime=?
                    WHERE path=?
                    """,
                    (current_mtime, normalized)
                )
        except Exception as e:
            print(f'标记目录已处理失败: {e}')

    def get_videos(self, dir_path):
        normalized = self._normalize_path(dir_path)
        if not normalized:
            return []
        rows = self.conn.execute(
            'SELECT file_name FROM videos WHERE dir_path=? ORDER BY file_name',
            (normalized,)
        ).fetchall()
        return [row['file_name'] for row in rows]

    def mark_video_processed(self, file_md5, video_path, detection_count=0, model_name=None):
        """将视频标记为已处理（基于文件MD5，跨机器通用）。"""
        if not file_md5:
            return
        now = time.time()
        try:
            with self.conn:
                self.conn.execute(
                    """
                    INSERT INTO processed_videos (file_md5, video_path, processed_at, detection_count, model_name)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(file_md5) DO UPDATE SET
                        video_path=excluded.video_path,
                        processed_at=excluded.processed_at,
                        detection_count=excluded.detection_count,
                        model_name=excluded.model_name
                    """,
                    (file_md5, str(video_path) if video_path else None, now, detection_count, model_name)
                )
        except Exception as e:
            print(f'标记视频已处理失败: {e}')

    def is_video_processed_by_md5(self, file_md5):
        """根据文件MD5查询视频是否已处理过。"""
        if not file_md5:
            return False
        try:
            row = self.conn.execute(
                'SELECT 1 FROM processed_videos WHERE file_md5=?',
                (file_md5,)
            ).fetchone()
            return row is not None
        except Exception:
            return False


DIRECTORY_INDEX = DirectoryIndex()
atexit.register(DIRECTORY_INDEX.close)


def _init_processing_root(root_dir):
    """设置处理根目录，并将数据库迁移到 <root_dir>/md5_list/directory_index.db。"""
    global _PROCESSING_ROOT_DIR
    if not root_dir or not os.path.isdir(root_dir):
        return
    _PROCESSING_ROOT_DIR = root_dir
    db_dir = os.path.join(root_dir, 'md5_list')
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, 'directory_index.db')
    DIRECTORY_INDEX.reopen(db_path)
    print(f'数据库已打开: {db_path}')


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


def directory_has_artifact_outputs(dir_path):
    try:
        for name in os.listdir(dir_path):
            lower = name.lower()
            if lower.endswith(DIR_ARTIFACT_SKIP_SUFFIXES):
                return True
    except (PermissionError, FileNotFoundError, OSError):
        return False
    return False

def is_leaf_directory(dir_path):
    """检查目录是否为叶子节点（不包含子目录，忽略工具生成的输出目录）"""
    try:
        for item in os.listdir(dir_path):
            if item in _IGNORED_SUBDIRS:
                continue
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
                        leaf_dirs.append((root, video_count, False))
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
                          all_objects=False,
                          save_mosaic=False,
                          save_timestamps=False):
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
    model = YOLO('models/nipple-nano-02-21.pt')

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

    # ---- 检测帧视频写入器（懒初始化，首次检测到目标时创建）----
    video_writer = None
    frame_w, frame_h = 0, 0
    
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
                                        if save_mosaic:
                                            dir_name = os.path.dirname(video_path)
                                            base_name = os.path.splitext(os.path.basename(video_path))[0]
                                            save_mosaic_batch(crops_batch, batch_idx, dir_name, base_name, max_cols)
                                            batch_idx += 1
                                        crops_batch = []
                                        # 强制垃圾回收
                                        gc.collect()
                                except Exception as e:
                                    print(f"处理裁剪图像时出错: {e}")
                        
                        # 在帧上绘制检测框（供视频写入和窗口预览共用）
                        xyxy_draw = box.xyxy[0].cpu().numpy()
                        dx1, dy1, dx2, dy2 = map(int, xyxy_draw)
                        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                        label = model.names[cls_id]
                        cv2.putText(frame, label, (dx1, max(dy1 - 6, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
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
            
            # 若本帧有检测结果，写入输出视频
            if detected:
                if video_writer is None:
                    frame_h, frame_w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(video_save_path, fourcc, fps_safe, (frame_w, frame_h))
                video_writer.write(frame)

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
        if video_writer is not None:
            video_writer.release()
        cap.release()
        pbar.close()
        if show_window:
            cv2.destroyAllWindows()

    if not paused and (frame_count >= total_frames or total_frames == 0):
        _clear_checkpoint(video_path)
    
    # 处理剩余的裁剪图像
    if save_mosaic and save_crops and crops_batch:
        dir_name = os.path.dirname(video_path)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        save_mosaic_batch(crops_batch, batch_idx, dir_name, base_name, max_cols)

    # 输出视频结果提示
    if video_writer is not None:
        print(f"已保存检测帧视频至: {video_save_path}")
    else:
        print(f"未检测到目标，未生成帧视频: {os.path.basename(video_path)}")
    
    # 保存检测结果
    if save_timestamps:
        txt_save_path = video_path + ".txt"
        with open(txt_save_path, 'w') as f:
            f.write("检测到目标的时间位置（秒）:\n")
            for t in detections:
                f.write(f"{t:.2f}\n")
        print(f"已保存检测时间戳至: {txt_save_path}")
    
    # 创建总拼接图
    if save_mosaic and save_crops and batch_idx > 1:
        try:
            dir_name = os.path.dirname(video_path)
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            mosaic_images = []
            for i in range(1, batch_idx + 1):
                mosaic_path = os.path.join(dir_name, f"{base_name}_mosaic_{i}.jpg")
                if os.path.exists(mosaic_path):
                    img = cv2.imread(mosaic_path)
                    if img is not None:
                        mosaic_images.append(img)
            if len(mosaic_images) > 0:
                final_height = sum(img.shape[0] for img in mosaic_images)
                final_width = max(img.shape[1] for img in mosaic_images)
                final_mosaic = np.zeros((final_height, final_width, 3), dtype=np.uint8)
                y_offset = 0
                for img in mosaic_images:
                    h, w = img.shape[:2]
                    final_mosaic[y_offset:y_offset+h, 0:w] = img
                    y_offset += h
                final_mosaic_path = os.path.join(dir_name, f"{base_name}_mosaic.jpg")
                cv2.imwrite(final_mosaic_path, final_mosaic)
                print(f"已保存最终拼接图片至: {final_mosaic_path}")
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


def get_file_md5(file_path):
    """计算文件的MD5哈希值，兼容极长的 Windows 路径。"""
    if not file_path:
        return None

    def candidate_paths(original_path):
        seen = set()
        original = str(original_path)
        for variant in (original,):
            if variant and variant not in seen:
                seen.add(variant)
                yield variant
        if os.name == 'posix':
            win = wsl_path_to_windows(original)
            if win and win not in seen:
                seen.add(win)
                yield win

    short_dir = '/tmp/findinvideo_md5'

    def open_stream(path):
        hash_md5 = hashlib.md5()
        with open(path, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b''):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def attempt_symlink(path):
        os.makedirs(short_dir, exist_ok=True)
        ext = os.path.splitext(path)[1] or ''
        hashed = hashlib.md5(path.encode('utf-8', 'ignore')).hexdigest()
        link_path = os.path.join(short_dir, hashed + ext)
        try:
            if os.path.exists(link_path):
                return link_path
            os.symlink(path, link_path)
            return link_path
        except Exception:
            return None

    def windows_copy(path):
        win_src = path if is_windows_style_path(path) else wsl_path_to_windows(path)
        if not win_src:
            return None
        win_src = win_src.replace('/', '\\')
        ext = os.path.splitext(win_src)[1] or '.bin'
        hashed = hashlib.md5(win_src.encode('utf-8', 'ignore')).hexdigest()
        base = os.environ.get('FINDINVIDEO_WIN_TEMP', r'C:\Temp\findinvideo_md5') or r'C:\Temp\findinvideo_md5'
        base = base.replace('/', '\\').rstrip('\\') or r'C:\Temp\findinvideo_md5'
        drive, tail = ntpath.splitdrive(base)
        base = ntpath.join('C:\\', base.lstrip('\\')) if not drive else ntpath.normpath(drive + tail)
        dest_win = ntpath.normpath(ntpath.join(base, hashed + ext))
        dest_dir = ntpath.dirname(dest_win)

        sanitized_dest_dir = dest_dir.replace("'", "''")
        sanitized_src = win_src.replace("'", "''")
        sanitized_dst = dest_win.replace("'", "''")
        ensure_script = (
            "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
            f"$dir='\\\\?\\{sanitized_dest_dir}';"
            "if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }"
        )
        copy_script = (
            "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
            f"$src='\\\\?\\{sanitized_src}';"
            f"$dst='\\\\?\\{sanitized_dst}';"
            "Copy-Item -LiteralPath $src -Destination $dst -Force"
        )
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command", ensure_script], check=True, capture_output=True, text=True)
            subprocess.run(["powershell", "-NoProfile", "-Command", copy_script], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            print(f"复制文件用于MD5失败: {exc}")
            return None
        dest_wsl = windows_path_to_wsl(dest_win)
        return dest_wsl

    last_error = None
    for candidate in candidate_paths(file_path):
        try:
            return open_stream(candidate)
        except (FileNotFoundError, NotADirectoryError) as e:
            last_error = e
        except OSError as e:
            last_error = e
            if os.name == 'posix':
                link = attempt_symlink(candidate)
                if link:
                    try:
                        return open_stream(link)
                    except Exception as e2:
                        last_error = e2
                copied = windows_copy(candidate)
                if copied:
                    try:
                        return open_stream(copied)
                    except Exception as e3:
                        last_error = e3
        except Exception as e:
            last_error = e

    print(f"计算MD5错误: {file_path}, 错误: {last_error or '未找到可访问路径'}")
    return None


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


def should_process(file_path):
    if has_existing_artifacts(file_path):
        return False
    if is_path_already_yoloed(file_path):
        return False
    md5 = get_file_md5_cached(file_path)
    if not md5:
        return False
    # 数据库查询：基于MD5判断是否已处理（跨机器通用）
    if DIRECTORY_INDEX.is_video_processed_by_md5(md5):
        return False
    yoloed_md5 = load_yoloed_md5(reload=False)
    if md5 in yoloed_md5:
        return False
    return True


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

def _record_video_processed(video_path, detections):
    """将已处理的视频记录到数据库和yoloed.txt中（确保零检测视频也不会被重复处理）。"""
    try:
        md5 = get_file_md5_cached(video_path)
        if md5:
            detection_count = len(detections) if detections else 0
            DIRECTORY_INDEX.mark_video_processed(
                file_md5=md5,
                video_path=str(video_path),
                detection_count=detection_count,
                model_name='yolov11l-face'
            )
            # 同时写入yoloed.txt以保持向后兼容
            append_yoloed_md5(md5, file_path=video_path)
            print(f'已记录视频处理完成（检测数={detection_count}）: {os.path.basename(video_path)}')
    except Exception as e:
        print(f'记录视频处理状态失败: {e}')

def _mark_directory_done(dir_path, video_file_names):
    """将目录标记为已全部处理，并将MD5缓存中已有的视频批量写入processed_videos表。"""
    try:
        DIRECTORY_INDEX.mark_directory_processed(dir_path)
        # 批量记录MD5已缓存的视频（这些视频在 should_process() 中已计算过MD5，无额外I/O）
        # 注意: _FILE_MD5_CACHE 的键是 (path_str, size, mtime) 元组，需要按此格式查找
        batch_count = 0
        for vf in video_file_names:
            vf_path = os.path.join(dir_path, vf)
            # 构造与 get_file_md5_cached 相同格式的缓存键
            try:
                st = os.stat(vf_path)
                cache_key = (str(vf_path), int(getattr(st, 'st_size', 0) or 0),
                             float(getattr(st, 'st_mtime', 0.0) or 0.0))
            except (OSError, PermissionError):
                cache_key = (str(vf_path), None, None)
            cached_md5 = _FILE_MD5_CACHE.get(cache_key)
            if cached_md5:
                DIRECTORY_INDEX.mark_video_processed(
                    file_md5=cached_md5,
                    video_path=vf_path,
                    detection_count=-1,  # -1 表示"通过衍生文件/跳过逻辑确认已处理，非本次检测"
                    model_name=None
                )
                batch_count += 1
        if batch_count > 0:
            print(f'已将 {batch_count} 个视频的处理记录写入数据库')
    except Exception as e:
        print(f'标记目录完成状态失败: {e}')


def process_directory_videos(dir_path, target_item, all_objects_switch=False, skip_long_videos=True):
    """处理目录中的所有视频文件"""
    if os.name == 'posix' and is_windows_style_path(dir_path):
        converted = windows_path_to_wsl(dir_path)
        if converted:
            dir_path = normalize_posix_path_with_fs(converted)

    # 一次 listdir 获取所有文件，后续判断全部在内存中完成
    try:
        all_files = os.listdir(dir_path)
    except (PermissionError, FileNotFoundError) as e:
        print(f"警告: 无法访问目录 '{dir_path}': {e}")
        return

    all_names_lower = set(f.lower() for f in all_files)
    video_file_names = [f for f in all_files if is_video_file(f)]

    if not video_file_names:
        return

    # 快速目录级预检查：纯内存比对，判断所有视频是否都已有衍生文件（零额外I/O）
    unprocessed_videos = []
    for vf in video_file_names:
        v_base = os.path.splitext(vf)[0].lower()
        if any((v_base + s.lower()) in all_names_lower for s in ARTIFACT_SUFFIXES):
            continue  # 该视频已有衍生文件
        unprocessed_videos.append(vf)

    if not unprocessed_videos:
        print(f"目录中所有 {len(video_file_names)} 个视频已有衍生文件，跳过整个目录")
        _mark_directory_done(dir_path, video_file_names)
        return

    skipped_count = len(video_file_names) - len(unprocessed_videos)
    if skipped_count > 0:
        print(f"跳过 {skipped_count}/{len(video_file_names)} 个已有衍生文件的视频")

    # 对未处理的视频逐一检查（should_process 包含MD5/数据库等更精确的判断）
    video_files = []
    for file in unprocessed_videos:
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
                video_files.append((file_path, duration))
            else:
                print(f"视频时长 {duration:.2f}秒超过一小时，跳过处理: {file_path}")
    
    if not video_files:
        if unprocessed_videos:
            print(f"目录中剩余 {len(unprocessed_videos)} 个视频经精确检查后均无需处理")
        _mark_directory_done(dir_path, video_file_names)
        return
    
    # 处理视频文件
    for video_file, duration in video_files:
        if duration == float('inf'):
            print(f"提示: 无法获取视频时长，仍尝试处理: {video_file}")
        print(f"开始处理视频文件: {video_file}")
        detections = detect_objects_in_video(video_file, target_item,
                                show_window=False,
                                save_crops=True,
                                save_training_data=False,
                                all_objects=all_objects_switch,
                                save_mosaic=save_mosaic_switch,
                                save_timestamps=save_timestamps_switch)
        _record_video_processed(video_file, detections)
        # 视频处理完成后强制垃圾回收
        gc.collect()
        # 短暂休眠，让系统有时间释放资源
        time.sleep(1)

    # 所有视频处理完成后，标记目录为已全部处理
    _mark_directory_done(dir_path, video_file_names)

if __name__ == "__main__":
    video_path = r"C:\Users\f1094\Downloads\香菜炸地球\20251024_223812.fix_p001.flv"  # 可设置为视频文件或目录
    # 如要检测所有模型内对象，则将 target_item 设置为任意值并启用全量检测开关
    target_item = "nipple"  # 当 all_objects 为 True 时，该值不再限制检测
    all_objects_switch = False  # 设置为 True 表示显示所有检测对象
    save_mosaic_switch = False  # 设置为 True 启用拼接图片保存
    save_timestamps_switch = False  # 设置为 True 启用检测时间戳txt保存
    
    # 初始化处理根目录，数据库和yoloed.txt都放在 <video_path>/md5_list/ 下
    _root = video_path if os.path.isdir(video_path) else os.path.dirname(video_path)
    _init_processing_root(_root)
    
    # 新增功能：按叶子节点视频数量排序处理
    use_leaf_node_processing = True  # 设置为 True 启用叶子节点处理模式
    
    if use_leaf_node_processing and os.path.isdir(video_path):
        print(f"启用叶子节点处理模式，正在扫描目录: {video_path}")
        print("正在查找包含视频文件的叶子节点目录...")

        root_video_count = count_videos_in_directory(video_path)
        if root_video_count > 0:
            print(f"\n=== 处理根目录: {video_path} ({root_video_count} 个视频) ===")
            process_directory_videos(video_path, target_item, all_objects_switch)

        leaf_dirs = find_leaf_directories_with_videos(video_path, EXCLUDE_PATHS, refresh_index=False)

        if not leaf_dirs:
            print(f"未找到包含视频文件的叶子节点目录")
        else:
            print(f"\n找到 {len(leaf_dirs)} 个包含视频文件的叶子节点目录:")
            for i, (dir_path, video_count, all_processed) in enumerate(leaf_dirs, 1):
                relative_path = _safe_relpath(dir_path, video_path)
                status = ' [已处理]' if all_processed else ''
                print(f"{i:3d}. {relative_path} ({video_count} 个视频文件){status}")
            
            print(f"\n开始按视频数量从多到少的顺序处理叶子节点目录...")
            
            # 按顺序处理每个叶子目录
            db_skipped = 0
            for i, (dir_path, video_count, _) in enumerate(leaf_dirs, 1):
                relative_path = _safe_relpath(dir_path, video_path)

                # 数据库快速跳过：has_artifact=True 且目录 mtime 未变 → 无需任何I/O
                dir_info = DIRECTORY_INDEX.get_directory_info(dir_path)
                if dir_info:
                    db_has_artifact, db_mtime = dir_info
                    if db_has_artifact and db_mtime is not None:
                        try:
                            current_mtime = os.stat(dir_path).st_mtime
                            if current_mtime == db_mtime:
                                db_skipped += 1
                                continue  # 目录未变且已全部处理，完全跳过
                        except (PermissionError, FileNotFoundError, OSError):
                            pass  # 无法获取mtime，退回到正常处理流程

                print(f"\n=== [{i}/{len(leaf_dirs)}] {relative_path} ({video_count} 个视频) ===")
                process_directory_videos(dir_path, target_item, all_objects_switch)
                
                # 每处理完一个目录后强制垃圾回收
                gc.collect()
                # 让系统有时间释放资源
                time.sleep(2)

            if db_skipped > 0:
                print(f"\n数据库快速跳过了 {db_skipped}/{len(leaf_dirs)} 个已确认处理完成的目录")
    
    # 原有的处理逻辑（当 use_leaf_node_processing 为 False 时使用）
    elif os.path.isdir(video_path):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        for root, dirs, files in os.walk(video_path):
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
                        detections = detect_objects_in_video(file_path, target_item,
                                                show_window=False,
                                                save_crops=True,
                                                save_training_data=True,
                                                all_objects=all_objects_switch,
                                                save_mosaic=save_mosaic_switch,
                                                save_timestamps=save_timestamps_switch)
                        _record_video_processed(file_path, detections)
                        
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
                    detections = detect_objects_in_video(video_path, target_item,
                                           show_window=False,
                                           save_crops=True,
                                           save_training_data=True,
                                           all_objects=all_objects_switch,
                                           save_mosaic=save_mosaic_switch,
                                           save_timestamps=save_timestamps_switch)
                    _record_video_processed(video_path, detections)
                else:
                    print(f"已存在拼接图片，跳过处理: {video_path}")