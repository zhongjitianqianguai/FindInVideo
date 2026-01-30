from ultralytics import YOLO
import cv2
import numpy as np
import os
from tqdm import tqdm  # 新增导入tqdm库用于进度条显示
import hashlib
import subprocess
import errno
import ntpath
import sqlite3
import time
import atexit
import sys
import logging
import traceback
import faulthandler
import json
import socket
import ctypes
from ctypes import wintypes
import signal

_YOLOED_MD5_CACHE = None
_YOLOED_MD5_CACHE_MTIME = None
_YOLOED_PATH_CACHE = None
_FILE_MD5_CACHE = {}

_LOGGER = logging.getLogger("findinvideo")
_CRASH_LOG_FH = None


class PauseRequested(Exception):
    """Raised to request a graceful stop with checkpoint saved."""


_STOP_REQUESTED = False


def _truthy_env(name, default=False):
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in ('1', 'true', 'yes', 'y', 'on')


def _get_pause_file_path():
    """Return the pause flag path (best-effort).

    If this file exists, the process will pause at a safe point.
    """
    explicit = _get_env_path('FINDINVIDEO_PAUSE_FILE') if '_get_env_path' in globals() else os.environ.get('FINDINVIDEO_PAUSE_FILE')
    if explicit:
        return explicit
    shared = get_shared_state_dir() if 'get_shared_state_dir' in globals() else None
    if shared:
        return os.path.join(shared, 'pause.flag')
    # fallback to workspace directory
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except Exception:
        base_dir = os.getcwd()
    return os.path.join(base_dir, 'pause.flag')


def _pause_requested(pause_file_path=None):
    if _STOP_REQUESTED:
        return True
    path = pause_file_path or _get_pause_file_path()
    try:
        return bool(path) and os.path.exists(path)
    except Exception:
        return False


def _install_pause_signal_handler():
    """Make Ctrl+C request a graceful pause (not a hard abort)."""
    def _handler(signum, frame):
        global _STOP_REQUESTED
        _STOP_REQUESTED = True
        try:
            print("\n收到 Ctrl+C：将于安全点暂停并保存进度（可直接再次运行续跑）。")
        except Exception:
            pass
    try:
        signal.signal(signal.SIGINT, _handler)
    except Exception:
        pass


CHECKPOINT_SUFFIX = '.checkpoint.json'


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
        # basic validation to avoid resuming wrong file
        try:
            st = os.stat(video_path)
            if data.get('size') not in (None, st.st_size):
                return None
            # allow small drift in mtime rounding
            if data.get('mtime') is not None and abs(float(data.get('mtime')) - float(st.st_mtime)) > 2.0:
                return None
        except Exception:
            pass
        return data
    except Exception:
        return None


def _save_checkpoint(video_path, next_frame, detections, last_detected, part_index):
    path = _checkpoint_path(video_path)
    payload = {
        'version': 1,
        'next_frame': int(max(0, next_frame or 0)),
        'detections': detections or [],
        'last_detected': float(last_detected) if last_detected is not None else -5.0,
        'part_index': int(max(0, part_index or 0)),
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
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        # best-effort only
        pass


def _clear_checkpoint(video_path):
    path = _checkpoint_path(video_path)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _setup_diagnostics():
    """Best-effort logging and crash diagnostics.

    - logs/run.log: Python-level logs + uncaught exceptions
    - logs/crash.log: faulthandler output for hard crashes (when possible)
    """
    global _CRASH_LOG_FH
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(base_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)

        run_log_path = os.path.join(log_dir, 'run.log')
        crash_log_path = os.path.join(log_dir, 'crash.log')

        if not _LOGGER.handlers:
            _LOGGER.setLevel(logging.INFO)
            fmt = logging.Formatter(
                fmt='%(asctime)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler = logging.FileHandler(run_log_path, encoding='utf-8')
            file_handler.setFormatter(fmt)
            _LOGGER.addHandler(file_handler)
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(fmt)
            _LOGGER.addHandler(stream_handler)

        if _CRASH_LOG_FH is None:
            _CRASH_LOG_FH = open(crash_log_path, 'a', encoding='utf-8', errors='ignore')
            faulthandler.enable(file=_CRASH_LOG_FH, all_threads=True)

        def _excepthook(exc_type, exc_value, exc_tb):
            try:
                _LOGGER.error("未捕获异常:\n%s", ''.join(traceback.format_exception(exc_type, exc_value, exc_tb)))
            except Exception:
                pass
            sys.__excepthook__(exc_type, exc_value, exc_tb)

        sys.excepthook = _excepthook
        _LOGGER.info("Diagnostics enabled. Logs: %s", run_log_path)
    except Exception:
        # Never fail the main program due to diagnostics setup.
        pass

def is_windows_style_path(path):
    """Return True if the path looks like a Windows drive path (e.g. C:\\)."""
    if not path or not isinstance(path, str):
        return False
    if len(path) < 2:
        return False
    drive, sep = path[0], path[1]
    return drive.isalpha() and sep == ':'


def windows_path_to_wsl(path):
    """Convert a Windows path to a WSL-accessible path."""
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
    """Convert a WSL path back to Windows format."""
    if not path:
        return None
    try:
        result = subprocess.run(['wslpath', '-w', path], check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding='utf-8', errors='ignore')
        converted = result.stdout.strip()
        return converted or None
    except Exception:
        # Best-effort manual conversion for /mnt/<drive>/...
        prefix = '/mnt/'
        if path.startswith(prefix) and len(path) > len(prefix):
            drive = path[len(prefix)]
            if drive.isalpha():
                remainder = path[len(prefix) + 1:].replace('/', '\\')
                return f"{drive.upper()}:\\{remainder}" if remainder else f"{drive.upper()}:\\"
        return None


def normalize_posix_path_with_fs(path):
    """Resolve symlinks and normalize a POSIX path if the filesystem allows."""
    if not path:
        return None
    try:
        resolved = os.path.realpath(path)
        return resolved
    except OSError:
        return os.path.normpath(path)


def safe_relpath(path, start):
    """Best-effort relpath; falls back to original path if mounts differ."""
    try:
        return os.path.relpath(path, start)
    except Exception:
        return str(path)


def windows_path_to_unc(path):
    r"""Convert a mapped drive path (e.g. D:\foo) to UNC (e.g. \\server\share\foo) if possible."""
    if os.name != 'nt' or not path:
        return None
    try:
        drive, tail = ntpath.splitdrive(str(path))
        if not drive or len(drive) < 2 or drive[1] != ':':
            return None
        drive_root = drive[0].upper() + ':'

        # WNetGetConnectionW returns UNC path for a mapped drive.
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
        # Join UNC root with remainder
        remainder = tail.lstrip('\\/')
        return unc_root.rstrip('\\/') + ('\\' + remainder if remainder else '')
    except Exception:
        return None


def canonical_video_path(path):
    """Best-effort canonicalization to improve cross-machine/path-mapping matches."""
    if not path:
        return None
    p = str(path)

    # Windows: normalize separators and try to convert mapped drive to UNC
    if os.name == 'nt':
        p_norm = ntpath.normpath(p)
        unc = windows_path_to_unc(p_norm)
        if unc:
            return ntpath.normpath(unc)
        return p_norm

    # POSIX: normalize and resolve symlinks when possible
    return normalize_posix_path_with_fs(p)


def safe_artifact_basename(video_path, max_length=80):
    """Generate a filesystem-friendly, bounded-length base name for artifacts.

    IMPORTANT: To work across multiple machines/OS and path mappings (Windows drive, UNC,
    WSL mount, Debian mount), this should NOT depend on the absolute path.
    We prefer a cheap stat-based fingerprint so artifacts have consistent names.

    Backward compatibility: callers may still need legacy base names (path-based).
    Use legacy_artifact_basename() for that.
    """
    base_name = os.path.basename(os.path.splitext(video_path)[0])
    sanitized = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in base_name)
    if not sanitized:
        sanitized = 'video'

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


def legacy_artifact_basename(video_path, max_length=80):
    """Legacy artifact base name (path-dependent). Keep for backward compatibility."""
    base_name = os.path.basename(os.path.splitext(video_path)[0])
    sanitized = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in base_name)
    if not sanitized:
        sanitized = 'video'
    digest = hashlib.md5(str(video_path).encode('utf-8', 'ignore')).hexdigest()[:8]
    limit = max(8, max_length - len(digest) - 1)
    if len(sanitized) > limit:
        sanitized = sanitized[:limit]
    return f"{sanitized}_{digest}"


def _get_env_path(name):
    """Return env var value stripped, or None."""
    value = os.environ.get(name)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def get_shared_state_dir():
    """Shared state directory for multi-machine coordination.

    Set FINDINVIDEO_SHARED_STATE_DIR to a shared/network path, e.g.:
      \\NAS\\share\\findinvideo_state
    """
    shared = _get_env_path('FINDINVIDEO_SHARED_STATE_DIR')
    if not shared:
        shared = r"\\192.168.31.9\\d\\md5_list"
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
    """Create a file atomically. Return True if created, False if exists."""
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


def _read_json_file_best_effort(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return json.load(f)
    except Exception:
        return None


def _claim_key_for_file(file_path, md5_hint=None):
    """Generate a stable claim key for a video.

    Prefer md5 when available; fall back to (basename + size + mtime).
    """
    # Prefer content fingerprint so different paths (Windows/WSL/Linux) still map to the same video.
    if md5_hint:
        return f"md5_{md5_hint}"

    # Fallback when md5 is unavailable: use stat fingerprint only (path-independent).
    try:
        st = os.stat(file_path)
        size = int(getattr(st, 'st_size', 0) or 0)
        mtime = float(getattr(st, 'st_mtime', 0.0) or 0.0)
    except Exception:
        size, mtime = 0, 0.0
    raw = f"{size}|{mtime}"
    return f"stat_{hashlib.md5(raw.encode('utf-8', 'ignore')).hexdigest()}"


def try_acquire_video_claim(file_path, md5_hint=None):
    """Try to claim a video for processing across multiple machines.

    Uses atomic lock-file creation in shared state dir.
    Returns claim_file_path if claimed, else None.
    """
    shared = ensure_shared_state_dir()
    if not shared:
        # No shared dir configured -> single-machine behavior.
        return None

    claims_dir = os.path.join(shared, 'claims')
    os.makedirs(claims_dir, exist_ok=True)

    ttl_seconds = int(os.environ.get('FINDINVIDEO_CLAIM_TTL_SECONDS', '86400') or '86400')
    key = _claim_key_for_file(file_path, md5_hint=md5_hint)
    claim_path = os.path.join(claims_dir, key + '.json')
    now = time.time()
    payload = {
        'path': str(file_path),
        'host': socket.gethostname(),
        'pid': os.getpid(),
        'started_at': now,
        'key': key,
    }

    # Fast path: create claim
    try:
        created = _atomic_create_file(claim_path, json.dumps(payload, ensure_ascii=False))
        if created:
            return claim_path
    except Exception as e:
        print(f"创建claim失败: {claim_path}, 错误: {e}")
        return None

    # Exists: maybe stale -> reclaim
    try:
        st = os.stat(claim_path)
        age = now - float(getattr(st, 'st_mtime', now) or now)
    except Exception:
        age = 0
    if ttl_seconds > 0 and age > ttl_seconds:
        existing = _read_json_file_best_effort(claim_path)
        print(f"检测到过期claim，尝试回收: {claim_path} (age={age:.0f}s, owner={existing})")
        try:
            os.unlink(claim_path)
        except Exception:
            return None
        try:
            created = _atomic_create_file(claim_path, json.dumps(payload, ensure_ascii=False))
            if created:
                return claim_path
        except Exception:
            return None

    return None


def release_video_claim(claim_path):
    if not claim_path:
        return
    try:
        os.unlink(claim_path)
    except Exception:
        pass


def _with_lockfile(lock_path, timeout_seconds=30, stale_seconds=3600):
    """A tiny lockfile helper (no external deps).

    Returns a callable that must be invoked to release the lock.
    """
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

        # If lock exists too long, try to break it
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

# 需要排除的路径变量
EXCLUDE_PATHS = [
    os.path.abspath(r"D:\$RECYCLE.BIN"),
    os.path.abspath(r"D:\System Volume Information"),
    # 在这里添加其他需要排除的路径
]

DIRECTORY_INDEX_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'directory_index.db')


class DirectoryIndex:
    """SQLite-backed cache for directory metadata and video listings."""

    def __init__(self, db_path=None):
        self.db_path = db_path or DIRECTORY_INDEX_DB_PATH
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
    bases = [safe_artifact_basename(video_path), legacy_artifact_basename(video_path)]
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


def directory_has_artifact_outputs(dir_path):
    """Return True if directory already contains generated artifacts we should skip."""
    indexed = DIRECTORY_INDEX.directory_has_artifacts(dir_path)
    if indexed is not None:
        return indexed
    try:
        for name in os.listdir(dir_path):
            lower_name = name.lower()
            if lower_name.endswith(DIR_ARTIFACT_SKIP_SUFFIXES):
                return True
    except (PermissionError, FileNotFoundError):
        return False
    return False

def detect_objects_in_video(video_path, target_class,
                            show_window=False, save_crops=False,
                            save_training_data=False,
                            all_objects=False,
                            save_mosaic=False,
                            save_video=False,
                            save_txt=False):
    # ...existing code...
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
    part_index = int(ckpt.get('part_index', 0)) if ckpt else 0
    # When resume is enabled, always write to part files to avoid "append" problems.
    use_parts = bool(resume_enabled and save_video)
    part_video_path = os.path.join(video_dir, f"{artifact_base}_frames.part{part_index:03d}.mp4")

    # 视频处理初始化
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"警告: 无法打开视频: {video_path}")
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

    frame_index = start_frame
    
    # 截图存储配置（用于拼接大图）
    crop_size = (160, 160)  # 统一缩放到的小图尺寸
    max_cols = 8  # 拼接大图每行最多显示数量
    crops = []  # 存储所有截取的目标区域
    video_writer = None

    try:
        while cap.isOpened():
            if _pause_requested(pause_file):
                _save_checkpoint(video_path, next_frame=frame_index, detections=detections, last_detected=last_detected, part_index=part_index + (1 if use_parts else 0))
                raise PauseRequested()

            success, frame = cap.read()
            if not success:
                break
            if frame is None:
                _LOGGER.warning("读取到空帧，已跳过: %s (frame=%s)", video_path, frame_index)
                pbar.update(1)
                frame_index += 1
                continue
            # 更新进度条
            pbar.update(1)
            frame_height, frame_width = frame.shape[:2]
            if save_training_data:
                frame_annotations = []
            # 某些编码/容器在 Windows 下会返回 fps=0，直接相除会导致异常退出。
            # 优先使用 POS_MSEC 获取真实时间戳（更稳定），回退到 fps_safe 计算。
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_msec and pos_msec > 0:
                current_time = float(pos_msec) / 1000.0
            else:
                current_time = frame_index / fps_safe
            try:
                results = model.predict(frame, conf=0.5, verbose=False, imgsz=imgsz)
            except KeyboardInterrupt:
                # Treat as pause: save checkpoint and exit gracefully.
                _save_checkpoint(video_path, next_frame=frame_index, detections=detections, last_detected=last_detected, part_index=part_index + (1 if use_parts else 0))
                raise PauseRequested()
            detected = False
            annotated_frame = None
            need_annotation = show_window or save_video
            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    if all_objects or model.names[cls_id] in target_class:
                        if current_time - last_detected >= 0.1:
                            detections.append(current_time)
                            last_detected = current_time
                            detected = True
                        if save_training_data or need_annotation or (save_crops and save_mosaic):
                            xyxy = box.xyxy[0].cpu().numpy()
                            if not np.isfinite(xyxy).all():
                                continue
                            x1, y1, x2, y2 = xyxy
                            x1i = int(np.clip(np.floor(x1), 0, frame_width - 1))
                            y1i = int(np.clip(np.floor(y1), 0, frame_height - 1))
                            x2i = int(np.clip(np.ceil(x2), 1, frame_width))
                            y2i = int(np.clip(np.ceil(y2), 1, frame_height))
                            if x2i <= x1i or y2i <= y1i:
                                continue
                        if need_annotation:
                            if annotated_frame is None:
                                annotated_frame = frame.copy()
                            cv2.rectangle(annotated_frame, (x1i, y1i), (x2i - 1, y2i - 1), (0, 255, 0), 2)
                        if save_crops and save_mosaic:
                            crop = frame[y1i:y2i, x1i:x2i]
                            if crop.size > 0:
                                resized = cv2.resize(crop, crop_size)
                                crops.append(resized)
                        if save_training_data:
                            h, w, _ = frame.shape
                            cx = ((x1i + x2i) / 2) / w
                            cy = ((y1i + y2i) / 2) / h
                            bw = (x2i - x1i) / w
                            bh = (y2i - y1i) / h
                            if all_objects:
                                class_index = cls_id
                            else:
                                class_index = target_class.index(model.names[cls_id])
                            annotation_line = f"{class_index} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                            frame_annotations.append(annotation_line)
            if detected and save_video and annotated_frame is not None:
                if video_writer is None:
                    h, w, _ = annotated_frame.shape
                    out_path = part_video_path if use_parts else video_save_path
                    video_writer = cv2.VideoWriter(
                        out_path,
                        cv2.VideoWriter_fourcc(*'mp4v'),
                        fps_safe,
                        (w, h)
                    )
                video_writer.write(annotated_frame)
            if show_window and detected:
                preview_frame = annotated_frame if annotated_frame is not None else frame
                cv2.imshow('Detection Preview', preview_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    pbar.close()
                    if show_window:
                        cv2.destroyAllWindows()
                    if video_writer is not None:
                        video_writer.release()
                    return detections
            if save_training_data and frame_annotations:
                training_base = artifact_base
                training_image_path = os.path.join(training_folder, f"{training_base}_{frame_index}.jpg")
                training_annotation_path = os.path.splitext(training_image_path)[0] + ".txt"
                cv2.imwrite(training_image_path, frame)
                with open(training_annotation_path, 'w') as f:
                    for line in frame_annotations:
                        f.write(line + "\n")
            frame_index += 1

    finally:
        try:
            cap.release()
        except Exception:
            pass
        try:
            pbar.close()
        except Exception:
            pass
        if show_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        if video_writer is not None:
            try:
                video_writer.release()
            except Exception:
                pass

    # 正常跑完整个视频：清理 checkpoint
    if (not _pause_requested(pause_file)) and (frame_index >= total_frames or total_frames == 0):
        _clear_checkpoint(video_path)

    # 如果只产生了单个 part 文件，则在完成后重命名为标准输出文件名
    if (not _pause_requested(pause_file)) and use_parts and (frame_index >= total_frames or total_frames == 0):
        try:
            if os.path.exists(part_video_path):
                os.replace(part_video_path, video_save_path)
        except Exception:
            pass

    if save_txt:
        with open(txt_save_path, 'w') as f:
            f.write("检测到目标的时间位置（秒）:\n")
            for t in detections:
                f.write(f"{t:.2f}\n")
        print(f"已保存检测时间戳至: {txt_save_path}")
    # 缩略图生成开关
    if save_mosaic and save_crops and crops:
        rows = []
        row = []
        for i, crop in enumerate(crops):
            row.append(crop)
            if (i + 1) % max_cols == 0:
                rows.append(np.hstack(row))
                row = []
        if row:
            missing = max_cols - len(row)
            blank = np.zeros_like(row[0])
            row.extend([blank] * missing)
            rows.append(np.hstack(row))
        final_mosaic = np.vstack(rows)
        if show_window:
            cv2.imshow('All Detected Faces', final_mosaic)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.imwrite(mosaic_path, final_mosaic)
            print(f"已保存拼接图片至: {mosaic_path}")
    if save_video:
        out_path = video_save_path if os.path.exists(video_save_path) else (part_video_path if use_parts else video_save_path)
        if out_path and os.path.exists(out_path):
            print(f"已保存检测帧视频至: {out_path}")
    return detections

def get_yoloed_md5_path():
    # 支持路径转换
    yoloed_path = _get_env_path('FINDINVIDEO_YOLOED_PATH')
    if not yoloed_path:
        shared = get_shared_state_dir()
        if shared:
            yoloed_path = os.path.join(shared, 'yoloed.txt')
        else:
            yoloed_path = r"D:\md5_list\yoloed.txt"
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
            _YOLOED_PATH_CACHE.add(str(file_path))
            _YOLOED_PATH_CACHE.add(os.path.normpath(str(file_path)))
            canon = canonical_video_path(file_path)
            if canon:
                _YOLOED_PATH_CACHE.add(canon)
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

def should_process(file_path):
    """返回(是否处理, 跳过原因)。"""
    if has_existing_artifacts(file_path):
        return False, 'artifact'
    md5 = get_file_md5_cached(file_path)
    if not md5:
        return True, 'md5-unavailable'
    # 多机情况下：根据 yoloed.txt 的 mtime 自动刷新缓存
    yoloed_md5 = load_yoloed_md5(reload=False)
    if md5 in yoloed_md5:
        return False, 'md5'
    return True, None


def should_process_with_md5(file_path):
    """返回(是否处理, 跳过原因, md5或None)。"""
    if has_existing_artifacts(file_path):
        return False, 'artifact', None
    md5 = get_file_md5_cached(file_path)
    if not md5:
        return True, 'md5-unavailable', None
    yoloed_md5 = load_yoloed_md5(reload=False)
    if md5 in yoloed_md5:
        return False, 'md5', md5
    return True, None, md5

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
    indexed_videos = DIRECTORY_INDEX.get_videos(dir_path)
    if indexed_videos:
        candidate_names = indexed_videos
    else:
        candidate_names = []
        try:
            candidate_names = [name for name in os.listdir(dir_path) if is_video_file(name)]
        except (PermissionError, FileNotFoundError) as e:
            print(f"警告: 无法访问目录 '{dir_path}': {e}")
            return

    for file in candidate_names:
        file_path = os.path.join(dir_path, file)
        if not os.path.isfile(file_path):
            continue
        # 先用 artifact/.done 做快速跳过，避免网络盘上大量计算 MD5
        if has_existing_artifacts(file_path):
            print(f"检测到同名衍生文件/完成标记，跳过处理: {file_path}")
            continue

        # 先抢占（不用 md5），避免多机重复做后续的 hash/推理
        claim_path = try_acquire_video_claim(file_path, md5_hint=None)
        if ensure_shared_state_dir() and not claim_path:
            print(f"已被其他机器抢占(处理中)，跳过: {file_path}")
            continue

        # 快速路径：如果 yoloed.txt 里记录了相同路径（或映射到 UNC 后相同），直接跳过
        if is_path_already_yoloed(file_path):
            release_video_claim(claim_path)
            print(f"已在路径记录中，跳过处理: {file_path}")
            continue

        md5 = get_file_md5_cached(file_path)
        if md5:
            if md5 in load_yoloed_md5(reload=False):
                # 已处理：释放 claim 并跳过
                release_video_claim(claim_path)
                print(f"已在MD5记录中，跳过处理: {file_path}")
                continue
        else:
            # md5 拿不到也允许继续（但可能会重复）；保留行为
            pass

        # 再计算时长过滤
        cap = cv2.VideoCapture(file_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        duration = frame_count / fps if fps > 0 else float('inf')
        if not skip_long_videos or duration <= 3600:
            video_files.append((file_path, duration, claim_path, md5))
        else:
            release_video_claim(claim_path)
            print(f"视频时长 {duration:.2f}秒超过一小时，跳过处理: {file_path}")
    
    # 处理视频文件
    for video_file, duration, claim_path, md5 in video_files:
        if duration == float('inf'):
            print(f"提示: 无法获取视频时长，仍尝试处理: {video_file}")
        print(f"开始处理视频文件: {video_file}")
        try:
            _LOGGER.info("Start video: %s", video_file)
            detect_objects_in_video(video_file, target_item,
                                    show_window=False,
                                    save_crops=True,
                                    save_training_data=False,
                                    all_objects=all_objects_switch,
                                    save_mosaic=False,  # 默认关闭缩略图
                                    save_video=True)    # 默认开启帧视频
            _LOGGER.info("Done video: %s", video_file)
        except PauseRequested:
            _LOGGER.info("Pause requested; saving checkpoint and stopping: %s", video_file)
            release_video_claim(claim_path)
            raise
        except Exception as exc:
            _LOGGER.error("Video failed: %s\n%s", video_file, traceback.format_exc())
            print(f"处理视频异常，已记录日志，跳过该视频: {video_file}\n错误: {exc}")
            release_video_claim(claim_path)
            continue
        # 处理完成后将MD5写入yoloed.txt
        md5 = md5 or get_file_md5_cached(video_file)
        if md5:
            append_yoloed_md5(md5, video_file)
        # 无论是否有 detections，都写一个 done 标记避免下次扫描再算 MD5
        write_done_marker(video_file)
        release_video_claim(claim_path)


def process_root_directory(root_path, target_item, all_objects_switch, skip_long_videos, use_leaf_node_processing):
    """根据配置处理根目录下的所有视频目录/文件"""
    if os.name == 'posix' and is_windows_style_path(root_path):
        converted_root = windows_path_to_wsl(root_path)
        if converted_root:
            root_path = normalize_posix_path_with_fs(converted_root)
    if not os.path.isdir(root_path):
        print(f"警告: 目录 '{root_path}' 无法通过常规方式访问，尝试继续处理。")
    try:
        DIRECTORY_INDEX.refresh(root_path, EXCLUDE_PATHS)
    except Exception as exc:
        print(f"目录索引刷新失败({root_path}): {exc}")
    md5_cache = load_yoloed_md5()
    print(f"已加载已识别MD5数量: {len(md5_cache)}")
    if use_leaf_node_processing:
        print(f"启用叶子节点处理模式，正在扫描目录: {root_path}")
        print("正在查找包含视频文件的叶子节点目录...")

        root_video_count = count_videos_in_directory(root_path)
        if root_video_count > 0:
            print(f"\n=== 处理根目录: {root_path} ({root_video_count} 个视频) ===")
            process_directory_videos(root_path, target_item, all_objects_switch, skip_long_videos)

        leaf_dirs = find_leaf_directories_with_videos(root_path, EXCLUDE_PATHS, refresh_index=False)

        if not leaf_dirs:
            print("未找到包含视频文件的叶子节点目录")
            return

        print(f"\n找到 {len(leaf_dirs)} 个包含视频文件的叶子节点目录:")
        for i, (dir_path, video_count) in enumerate(leaf_dirs, 1):
            relative_path = safe_relpath(dir_path, root_path)
            print(f"{i:3d}. {relative_path} ({video_count} 个视频文件)")

        print("\n开始按视频数量从多到少的顺序处理叶子节点目录...")
        for i, (dir_path, video_count) in enumerate(leaf_dirs, 1):
            relative_path = safe_relpath(dir_path, root_path)
            print(f"\n=== 处理第 {i}/{len(leaf_dirs)} 个目录: {relative_path} ({video_count} 个视频) ===")
            process_directory_videos(dir_path, target_item, all_objects_switch, skip_long_videos)
    else:
        print(f"使用原有目录遍历逻辑处理: {root_path}")
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        for root, dirs, files in os.walk(root_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in video_extensions:
                    file_path = os.path.join(root, file)
                    if has_existing_artifacts(file_path):
                        print(f"检测到同名衍生文件/完成标记，跳过处理: {file_path}")
                        continue

                    claim_path = try_acquire_video_claim(file_path, md5_hint=None)
                    if ensure_shared_state_dir() and not claim_path:
                        print(f"已被其他机器抢占(处理中)，跳过: {file_path}")
                        continue

                    if is_path_already_yoloed(file_path):
                        release_video_claim(claim_path)
                        print(f"已在路径记录中，跳过处理: {file_path}")
                        continue

                    md5 = get_file_md5_cached(file_path)
                    if md5 and md5 in load_yoloed_md5(reload=False):
                        release_video_claim(claim_path)
                        print(f"已在MD5记录中，跳过处理: {file_path}")
                        continue

                    cap = cv2.VideoCapture(file_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    cap.release()
                    duration = frame_count / fps if fps > 0 else float('inf')
                    if skip_long_videos and duration > 3600:
                        print(f"视频时长 {duration:.2f}秒超过一小时，跳过处理: {file_path}")
                        release_video_claim(claim_path)
                        continue
                    if duration == float('inf'):
                        print(f"提示: 无法获取视频时长，仍尝试处理: {file_path}")
                    elif duration > 3600:
                        print(f"提示: 视频时长 {duration:.2f}秒超过一小时，但仍根据配置处理: {file_path}")

                    print(f"开始处理视频文件: {file_path}")
                    try:
                        detect_objects_in_video(file_path, target_item,
                                                show_window=False,
                                                save_crops=True,
                                                save_training_data=True,
                                                all_objects=all_objects_switch,
                                                save_mosaic=False,  # 默认关闭缩略图
                                                save_video=True)    # 默认开启帧视频
                        md5_done = md5 or get_file_md5_cached(file_path)
                        if md5_done:
                            append_yoloed_md5(md5_done, file_path)
                        write_done_marker(file_path)
                    finally:
                        release_video_claim(claim_path)

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
            if is_windows_style_path(original) or '\\' in original:
                converted = windows_path_to_wsl(original)
                for derived in (normalize_posix_path_with_fs(converted) if converted else None, converted):
                    if derived and derived not in seen:
                        seen.add(derived)
                        yield derived
            else:
                normalized = normalize_posix_path_with_fs(original)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    yield normalized

    short_dir = '/tmp/findinvideo_md5'

    def open_stream(path):
        hash_md5 = hashlib.md5()
        with open(path, 'rb') as fh:
            max_bytes = 10 * 1024 * 1024
            read = 0
            for chunk in iter(lambda: fh.read(4096), b''):
                hash_md5.update(chunk)
                read += len(chunk)
                if read >= max_bytes:
                    break
        return hash_md5.hexdigest()

    def attempt_symlink(path):
        os.makedirs(short_dir, exist_ok=True)
        ext = os.path.splitext(path)[1] or ''
        hashed = hashlib.md5(path.encode('utf-8', 'ignore')).hexdigest()
        link_path = os.path.join(short_dir, hashed + ext)
        try:
            if os.path.lexists(link_path):
                if not os.path.islink(link_path) or os.readlink(link_path) != path:
                    os.unlink(link_path)
            if not os.path.lexists(link_path):
                os.symlink(path, link_path)
            digest = open_stream(link_path)
            print(f"[MD5] 通过短路径计算: {link_path}")
            return digest
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
            subprocess.run(['powershell.exe', '-NoProfile', '-Command', ensure_script],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           text=True, encoding='utf-8', errors='ignore')
            subprocess.run(['powershell.exe', '-NoProfile', '-Command', copy_script],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           text=True, encoding='utf-8', errors='ignore')
        except subprocess.CalledProcessError as exc:
            msg = exc.stderr.strip() if exc.stderr else str(exc)
            print(f"[MD5] Windows Copy-Item 失败: {msg}")
            return None
        dest_wsl = windows_path_to_wsl(dest_win)
        return dest_wsl

    last_error = None
    for candidate in candidate_paths(file_path):
        try:
            digest = open_stream(candidate)
            if candidate != file_path:
                print(f"[MD5] 使用备用路径计算: {candidate}")
            return digest
        except (FileNotFoundError, NotADirectoryError) as e:
            last_error = e
            continue
        except OSError as e:
            last_error = e
            if os.name != 'posix':
                continue
            need_short = getattr(e, 'errno', None) == errno.ENAMETOOLONG or 'File name too long' in str(e)
            candidate_is_file = False
            try:
                candidate_is_file = os.path.isfile(candidate)
            except OSError as exists_err:
                if getattr(exists_err, 'errno', None) == errno.ENAMETOOLONG:
                    candidate_is_file = True
            if not need_short and len(candidate) >= 240 and candidate_is_file:
                need_short = True
            if need_short and candidate_is_file:
                digest = attempt_symlink(candidate)
                if digest:
                    return digest
            alt = windows_copy(candidate)
            if alt:
                try:
                    digest = open_stream(alt)
                    print(f"[MD5] 通过 Windows 缩短路径计算: {alt}")
                    return digest
                except Exception as copy_err:
                    last_error = copy_err
            continue
        except Exception as e:
            last_error = e
            break

    print(f"计算MD5错误: {file_path}, 错误: {last_error or '未找到可访问路径'}")
    return None

if __name__ == "__main__":
    _setup_diagnostics()
    _install_pause_signal_handler()
    video_path = r"""D:\z"""  # 可设置为视频文件或目录
    original_video_path = video_path
    if os.name == 'posix' and is_windows_style_path(video_path):
        converted = windows_path_to_wsl(video_path)
        normalized = normalize_posix_path_with_fs(converted) if converted else None
        if normalized:
            video_path = normalized
            print(f"检测到 Windows 路径，已转换为 WSL 路径处理: {video_path}")
        else:
            print(f"警告: 无法转换 Windows 路径 {original_video_path}，将尝试直接使用。")
    # 如要检测所有模型内对象，则将 target_item 设置为任意值并启用全量检测开关
    target_item = "face"  # 当 all_objects 为 True 时，该值不再限制检测
    all_objects_switch = False  # 设置为 True 表示显示所有检测对象
    skip_long_videos = False  # 设置为 True 时跳过超过一小时的视频
    
    # 新增功能：按叶子节点视频数量排序处理
    use_leaf_node_processing = True  # 设置为 True 启用叶子节点处理模式
    
    try:
        if os.path.isdir(video_path):
            process_root_directory(video_path, target_item, all_objects_switch, skip_long_videos, use_leaf_node_processing)
        elif os.path.isfile(video_path) and is_video_file(video_path):
            # 处理单个视频文件前检查视频时长
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            duration = frame_count / fps if fps > 0 else float('inf')
            if skip_long_videos and duration > 3600:
                print(f"视频时长 {duration:.2f}秒超过一小时，跳过处理: {video_path}")
            else:
                if duration == float('inf'):
                    print(f"提示: 无法获取视频时长，仍尝试处理: {video_path}")
                elif duration > 3600:
                    print(f"提示: 视频时长 {duration:.2f}秒超过一小时，但仍根据配置处理: {video_path}")
                if has_existing_artifacts(video_path):
                    print(f"检测到同名衍生文件/完成标记，跳过处理: {video_path}")
                else:
                    claim_path = try_acquire_video_claim(video_path, md5_hint=None)
                    if ensure_shared_state_dir() and not claim_path:
                        print(f"已被其他机器抢占(处理中)，跳过: {video_path}")
                    else:
                        try:
                            if is_path_already_yoloed(video_path):
                                print(f"已在路径记录中，跳过处理: {video_path}")
                            else:
                                md5 = get_file_md5_cached(video_path)
                                if md5 and md5 in load_yoloed_md5(reload=False):
                                    print(f"已在MD5记录中，跳过处理: {video_path}")
                                else:
                                    detect_objects_in_video(video_path, target_item,
                                                            show_window=False,
                                                            save_crops=True,
                                                            save_training_data=True,
                                                            all_objects=all_objects_switch,
                                                            save_mosaic=False,  # 默认关闭缩略图
                                                            save_video=True)    # 默认开启帧视频
                                    md5_done = md5 or get_file_md5_cached(video_path)
                                    if md5_done:
                                        append_yoloed_md5(md5_done, video_path)
                                    write_done_marker(video_path)
                        finally:
                            release_video_claim(claim_path)
        else:
            if not is_video_file(video_path):
                print(f"提示: '{video_path}' 不像是视频文件，按目录进行处理尝试。")
                process_root_directory(video_path, target_item, all_objects_switch, skip_long_videos, use_leaf_node_processing)
            else:
                print(f"路径不存在或无法访问: {video_path}")
    except PauseRequested:
        print("已暂停：已保存进度（checkpoint）。删除 pause.flag 或再次运行即可续跑。")
        sys.exit(0)
    except Exception as exc:
        _LOGGER.error("Fatal error:\n%s", traceback.format_exc())
        raise