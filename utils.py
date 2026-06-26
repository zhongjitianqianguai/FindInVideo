"""utils.py — 公共工具函数和常量

从 main.py、main_nipple.py、main_nipple_rog.py、main_yolov5.py 中提取的重复代码。
"""

import os
import json
import time
import hashlib
import sqlite3
import socket
import signal
import errno
import subprocess
import threading
import ntpath
import ctypes
from ctypes import wintypes
import atexit


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

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

_IGNORED_SUBDIRS = {
    '_detected', 'yolov5_output', '__pycache__', '.git',
    '$RECYCLE.BIN', 'System Volume Information', 'training_data', 'md5_list',
}

CHECKPOINT_SUFFIX = '.checkpoint.json'

CLAIM_HEARTBEAT_INTERVAL_FRAMES = 100


# ---------------------------------------------------------------------------
# 路径处理
# ---------------------------------------------------------------------------

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
        result = subprocess.run(
            ['wslpath', '-a', path], check=True,
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
        result = subprocess.run(
            ['wslpath', '-w', path], check=True,
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
    """将UNC路径（如 \\\\192.168.6.100\\d\\...）转换回本机映射的盘符路径（如 D:\\...）。"""
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
        converted_path = unc_to_drive_letter(path) if path.startswith('\\\\') else path
        converted_start = unc_to_drive_letter(start) if start.startswith('\\\\') else start
        p = converted_path or path
        s = converted_start or start
        try:
            return os.path.relpath(p, s)
        except ValueError:
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


# ---------------------------------------------------------------------------
# 环境变量和共享状态
# ---------------------------------------------------------------------------

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


def _truthy_env(name, default=False):
    val = os.environ.get(name)
    if val is None:
        return default
    return str(val).strip().lower() in ('1', 'true', 'yes', 'y', 'on')


# ---------------------------------------------------------------------------
# 文件锁
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 视频文件判断和命名
# ---------------------------------------------------------------------------

def is_video_file(file_path):
    """检查文件是否为视频文件"""
    _, ext = os.path.splitext(file_path.lower())
    base = file_path.lower()
    if base.endswith('_frames.mp4') or base.endswith('_objects.mp4') or base.endswith('_detections.mp4'):
        return False
    if '_frames.part' in base or '_objects.part' in base or '_detections.part' in base:
        return False
    return ext in VIDEO_EXTENSIONS


def safe_artifact_basename(video_path):
    """返回用于创建衍生文件的基础名（= 原视频文件名去掉扩展名）。"""
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


# ---------------------------------------------------------------------------
# Checkpoint 系统
# ---------------------------------------------------------------------------

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


def _checkpoint_owner_snapshot(claim_md5=None):
    """采样当前 checkpoint 写入者信息，便于恢复和排障。"""
    owner = {
        'host_name': socket.gethostname(),
        'pid': os.getpid(),
    }
    if claim_md5:
        owner['claim_md5'] = claim_md5
    return owner


def _save_checkpoint(video_path, next_frame, detections, last_detected, claim_md5=None, last_success_frame=None):
    path = _checkpoint_path(video_path)
    payload = {
        'version': 1,
        'next_frame': int(max(0, next_frame or 0)),
        'detections': detections or [],
        'last_detected': float(last_detected) if last_detected is not None else -5.0,
        'saved_at': time.time(),
        'last_success_frame': int(last_success_frame) if last_success_frame is not None else int(max(-1, (next_frame or 0) - 1)),
        'checkpoint_owner': _checkpoint_owner_snapshot(claim_md5),
        'claim_heartbeat_at': time.time() if claim_md5 else None,
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
        pass


def _clear_checkpoint(video_path):
    path = _checkpoint_path(video_path)
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 暂停系统
# ---------------------------------------------------------------------------

_STOP_REQUESTED = False


class PauseRequested(Exception):
    """Raised to request a graceful stop with checkpoint saved."""


def _install_pause_signal_handler():
    """将 Ctrl+C 转成安全暂停请求，在循环安全点统一落 checkpoint。"""
    def _handler(signum, frame):
        global _STOP_REQUESTED
        _STOP_REQUESTED = True
        try:
            print("\n收到 Ctrl+C：已请求暂停，当前帧结束后将保存进度并退出，不会继续处理后续视频。")
        except Exception:
            pass
    try:
        signal.signal(signal.SIGINT, _handler)
    except Exception:
        pass


def _get_pause_file_path():
    """Return the pause flag path (best-effort)."""
    explicit = os.environ.get('FINDINVIDEO_PAUSE_FILE')
    if explicit:
        return explicit
    shared = get_shared_state_dir()
    if shared:
        return os.path.join(shared, 'pause.flag')
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


# ---------------------------------------------------------------------------
# 视频帧读取超时包装
# ---------------------------------------------------------------------------

_READ_TIMEOUT_SEC = 30


def _read_frame_with_timeout(cap, timeout=_READ_TIMEOUT_SEC):
    """在子线程中调用 cap.read()，超时后返回 (False, None, True)。

    返回 (success, frame, timed_out)。
    """
    result = [False, None]

    def _reader():
        try:
            result[0], result[1] = cap.read()
        except Exception:
            result[0], result[1] = False, None

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        return False, None, True
    return result[0], result[1], False


# ---------------------------------------------------------------------------
# DirectoryIndex 类
# ---------------------------------------------------------------------------

class DirectoryIndex:
    """SQLite-backed cache for directory metadata and video listings."""

    def __init__(self, db_path=None):
        self.db_path = db_path or ':memory:'
        self.conn = sqlite3.connect(self.db_path, timeout=60)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA busy_timeout = 30000;')
        self.conn.execute('PRAGMA foreign_keys = ON;')
        # 网络共享上 WAL 模式不安全，使用 DELETE 模式
        try:
            self.conn.execute('PRAGMA journal_mode = DELETE;')
        except sqlite3.OperationalError as e:
            if self._is_lock_error(e):
                print(f"设置journal_mode被锁，跳过: {e}")
            else:
                raise
        self._ensure_schema()

    def _ensure_schema(self):
        """创建所需表和索引（如果不存在）。"""
        for _attempt in range(3):
            try:
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
                    self.conn.execute(
                        """
                        CREATE TABLE IF NOT EXISTS processing_claims (
                            file_md5 TEXT PRIMARY KEY,
                            video_path TEXT,
                            claimed_at REAL,
                            heartbeat_at REAL,
                            host_name TEXT,
                            pid INTEGER
                        )
                        """
                    )
                    self._ensure_claim_schema_compat()
                return
            except sqlite3.OperationalError as e:
                if self._is_lock_error(e):
                    if _attempt < 2:
                        print(f"创建表结构被锁，重试中({_attempt + 1}/3)...")
                        time.sleep(5)
                    else:
                        print(f"创建表结构被锁，跳过（表可能已存在）: {e}")
                else:
                    raise

    def close(self):
        try:
            self.conn.close()
        except sqlite3.Error:
            pass

    def _ensure_claim_schema_compat(self):
        """为旧数据库补齐 processing_claims 新字段。"""
        try:
            rows = self.conn.execute('PRAGMA table_info(processing_claims)').fetchall()
        except sqlite3.DatabaseError:
            return
        columns = {row['name'] if isinstance(row, sqlite3.Row) else row[1] for row in rows}
        if 'heartbeat_at' not in columns:
            self.conn.execute('ALTER TABLE processing_claims ADD COLUMN heartbeat_at REAL')

    def reopen(self, new_db_path):
        """关闭当前连接并在新路径重新打开数据库。"""
        self.close()
        self.db_path = new_db_path
        folder = os.path.dirname(new_db_path)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, timeout=60)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA busy_timeout = 30000;')
        self.conn.execute('PRAGMA foreign_keys = ON;')
        # 网络共享上 WAL 模式不安全，使用 DELETE 模式
        for _attempt in range(3):
            try:
                self.conn.execute('PRAGMA journal_mode = DELETE;')
                break
            except sqlite3.OperationalError as e:
                if self._is_lock_error(e):
                    if _attempt < 2:
                        print(f"设置journal_mode被锁，重试中({_attempt + 1}/3)...")
                        time.sleep(5)
                    else:
                        print(f"设置journal_mode被锁，跳过: {e}")
                else:
                    raise
        self._ensure_schema()
        print(f"数据库已打开: {self.db_path}")

    def _check_integrity(self):
        """执行 PRAGMA integrity_check，返回数据库是否完好。"""
        try:
            result = self.conn.execute('PRAGMA integrity_check;').fetchone()
            return result and result[0] == 'ok'
        except sqlite3.DatabaseError as e:
            if self._is_lock_error(e):
                print(f"数据库被锁定，跳过完整性检查（视为完好）: {e}")
                return True
            return False

    @staticmethod
    def _is_lock_error(exc):
        """判断异常是否为数据库锁冲突或网络瞬态错误。"""
        msg = str(exc).lower()
        return ('locked' in msg or 'busy' in msg or
                'disk i/o error' in msg or 'unable to open' in msg)

    @staticmethod
    def _remove_db_files(db_path):
        """将损坏的数据库文件备份为 .corrupt 后删除。"""
        import time as _time
        timestamp = _time.strftime('%Y%m%d_%H%M%S')
        main_db_cleared = False
        for suffix in ('', '-wal', '-shm'):
            p = db_path + suffix
            try:
                if os.path.exists(p):
                    backup = f"{p}.corrupt.{timestamp}"
                    try:
                        os.rename(p, backup)
                        print(f"已备份损坏文件: {p} -> {backup}")
                        if suffix == '':
                            main_db_cleared = True
                    except OSError:
                        try:
                            os.remove(p)
                            print(f"备份失败，已直接删除: {p}")
                            if suffix == '':
                                main_db_cleared = True
                        except OSError as e2:
                            print(f"处理 {p} 失败: {e2}")
                else:
                    if suffix == '':
                        main_db_cleared = True
            except OSError as e:
                print(f"处理 {p} 失败: {e}")
        return main_db_cleared

    def _fallback_to_memory_db(self, reason):
        """回退到内存数据库。"""
        print(f"回退到内存数据库（{reason}）")
        try:
            self.conn.close()
        except Exception:
            pass
        self.conn = sqlite3.connect(':memory:')
        self.conn.row_factory = sqlite3.Row
        self.conn.execute('PRAGMA foreign_keys = ON;')
        self._ensure_schema()

    def _safe_reconnect(self, context_msg):
        """安全地重新连接到文件数据库。"""
        try:
            self.conn.close()
        except Exception:
            pass
        try:
            self.conn = sqlite3.connect(self.db_path, timeout=60)
            self.conn.row_factory = sqlite3.Row
            self.conn.execute('PRAGMA busy_timeout = 30000;')
            self.conn.execute('SELECT 1;').fetchone()
            return True
        except Exception as e:
            print(f"{context_msg}重连数据库失败: {e}")
            self._fallback_to_memory_db(f"{context_msg}重连失败")
            return True

    def _rebuild_if_corrupt(self):
        """检测到数据库操作异常时的恢复策略。"""
        if self.db_path == ':memory:':
            return False
        print(f"数据库操作异常，尝试重连: {self.db_path}")
        self._safe_reconnect('操作异常后')
        try:
            self.conn.execute('SELECT 1;').fetchone()
            return True
        except sqlite3.DatabaseError:
            pass
        if self._check_integrity():
            return True
        lock_path = self.db_path + '.rebuild.lock'
        try:
            release = _with_lockfile(lock_path, timeout_seconds=60, stale_seconds=120)
        except (TimeoutError, Exception) as e:
            print(f"另一个实例正在重建数据库，等待完成: {e}")
            time.sleep(10)
            self._safe_reconnect('等待其他实例重建后')
            return True
        try:
            if self._check_integrity():
                return True
            print(f"数据库确认损坏，正在备份并重建: {self.db_path}")
            try:
                self.conn.close()
            except Exception:
                pass
            db_cleared = self._remove_db_files(self.db_path)
            if not db_cleared:
                self._fallback_to_memory_db('损坏的数据库文件被锁定，无法删除')
                return True
            try:
                self.conn = sqlite3.connect(self.db_path, timeout=60)
                self.conn.row_factory = sqlite3.Row
                self.conn.execute('PRAGMA busy_timeout = 30000;')
                self.conn.execute('PRAGMA foreign_keys = ON;')
                self.conn.execute('PRAGMA journal_mode = DELETE;')
                self._ensure_schema()
            except Exception as e:
                print(f"重建数据库文件失败: {e}")
                self._fallback_to_memory_db('重建数据库文件失败')
        finally:
            release()
        return True

    def _normalize_path(self, path):
        if not path:
            return None
        try:
            canon = canonical_video_path(path)
            if canon:
                return canon
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
                        'UPDATE directories SET parent_path=? WHERE path=?',
                        (parent_path, normalized))
            stored_children = set(self._get_child_paths(normalized))
            try:
                with os.scandir(normalized) as iterator:
                    for entry in iterator:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                if entry.name not in _IGNORED_SUBDIRS:
                                    entry_path = self._normalize_path(entry.path)
                                    if entry_path and entry_path not in stored_children:
                                        self._refresh_directory(entry_path, exclusions, normalized)
                        except (PermissionError, FileNotFoundError, OSError):
                            continue
            except (PermissionError, FileNotFoundError, OSError):
                pass
            for child in self._get_child_paths(normalized):
                self._refresh_directory(child, exclusions, normalized)
            return

        stored_children = set(self._get_child_paths(normalized))
        child_dirs = []
        video_records = []
        all_file_names_lower = set()
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
                                    video_records.append((entry.name, stat_info.st_mtime, stat_info.st_size))
                                except OSError:
                                    continue
                    except (PermissionError, FileNotFoundError, OSError):
                        continue
        except (PermissionError, FileNotFoundError, OSError):
            self._remove_directory_recursive(normalized)
            return

        if video_records:
            processed_count = 0
            for v_name, _, _ in video_records:
                v_base = os.path.splitext(v_name)[0].lower()
                if any((v_base + s.lower()) in all_file_names_lower for s in ARTIFACT_SUFFIXES):
                    processed_count += 1
            has_artifact = processed_count == len(video_records)
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
                (normalized, parent_path, current_mtime, now, is_leaf, video_count, 1 if has_artifact else 0))
            self.conn.execute('DELETE FROM videos WHERE dir_path=?', (normalized,))
            for name, mtime, size in video_records:
                self.conn.execute(
                    """
                    INSERT OR REPLACE INTO videos (dir_path, file_name, file_mtime, file_size, is_video)
                    VALUES (?, ?, ?, ?, 1)
                    """,
                    (normalized, name, mtime, size))

        for child_path in child_dirs:
            self._refresh_directory(child_path, exclusions, normalized)

    def _get_directory(self, path):
        try:
            return self.conn.execute(
                'SELECT * FROM directories WHERE path=?', (path,)).fetchone()
        except sqlite3.DatabaseError as e:
            if self._is_lock_error(e):
                print(f"数据库被锁定，跳过查询: {e}")
                return None
            if self._rebuild_if_corrupt():
                return None
            raise

    def _get_child_paths(self, path):
        try:
            rows = self.conn.execute(
                'SELECT path FROM directories WHERE parent_path=?', (path,)).fetchall()
            return [row['path'] for row in rows]
        except sqlite3.DatabaseError as e:
            if self._is_lock_error(e):
                print(f"数据库被锁定，跳过查询: {e}")
                return []
            if self._rebuild_if_corrupt():
                return []
            raise

    def _remove_directory_recursive(self, path):
        normalized = self._normalize_path(path)
        if not normalized:
            return
        try:
            child_rows = self.conn.execute(
                'SELECT path FROM directories WHERE parent_path=?', (normalized,)).fetchall()
        except sqlite3.DatabaseError as e:
            if self._is_lock_error(e):
                print(f"数据库被锁定，跳过删除: {e}")
                return
            if self._rebuild_if_corrupt():
                return
            raise
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
                (path, parent_path, current_mtime, time.time()))
            self.conn.execute('DELETE FROM videos WHERE dir_path=?', (path,))
        for child in self._get_child_paths(path):
            self._remove_directory_recursive(child)

    def get_leaf_directories(self, root_path):
        normalized_root = self._normalize_path(root_path)
        if not normalized_root:
            return []
        like_pattern = f"{normalized_root.rstrip(os.sep)}{os.sep}%"
        try:
            rows = self.conn.execute(
                """
                SELECT path, video_count, has_artifact FROM directories
                WHERE excluded=0 AND video_count>0 AND is_leaf=1
                  AND (path=? OR path LIKE ?)
                """,
                (normalized_root, like_pattern)).fetchall()
        except sqlite3.DatabaseError as e:
            if self._is_lock_error(e):
                print(f"数据库被锁定，跳过查询叶子目录: {e}")
                return []
            if self._rebuild_if_corrupt():
                return []
            raise
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
        """获取目录的 has_artifact 和 dir_mtime 信息。"""
        normalized = self._normalize_path(dir_path)
        if not normalized:
            return None
        row = self._get_directory(normalized)
        if row and not row['excluded']:
            return (bool(row['has_artifact']), row['dir_mtime'])
        return None

    def mark_directory_processed(self, dir_path):
        """将目录标记为已全部处理（has_artifact=1）。"""
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
                    'UPDATE directories SET has_artifact=1, dir_mtime=? WHERE path=?',
                    (current_mtime, normalized))
        except Exception as e:
            print(f"标记目录已处理失败: {e}")

    def get_videos(self, dir_path):
        normalized = self._normalize_path(dir_path)
        if not normalized:
            return []
        try:
            rows = self.conn.execute(
                'SELECT file_name FROM videos WHERE dir_path=? ORDER BY file_name',
                (normalized,)).fetchall()
        except sqlite3.DatabaseError as e:
            if self._is_lock_error(e):
                print(f"数据库被锁定，跳过查询视频列表: {e}")
                return []
            if self._rebuild_if_corrupt():
                return []
            raise
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
                    (file_md5, str(video_path) if video_path else None, now, detection_count, model_name))
                self.conn.execute(
                    'DELETE FROM processing_claims WHERE file_md5=?', (file_md5,))
        except Exception as e:
            print(f"标记视频已处理失败: {e}")

    def is_video_processed_by_md5(self, file_md5):
        """根据文件MD5查询视频是否已处理过。"""
        if not file_md5:
            return False
        try:
            row = self.conn.execute(
                'SELECT 1 FROM processed_videos WHERE file_md5=?', (file_md5,)).fetchone()
            return row is not None
        except Exception:
            return False

    def try_claim_video(self, file_md5, video_path):
        """尝试声明视频为"正在处理"状态。"""
        if not file_md5:
            return False
        now = time.time()
        host_name = socket.gethostname()
        pid = os.getpid()
        try:
            self.conn.execute(
                """
                INSERT INTO processing_claims (file_md5, video_path, claimed_at, heartbeat_at, host_name, pid)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(file_md5) DO NOTHING
                """,
                (file_md5, str(video_path) if video_path else None, now, now, host_name, pid))
            self.conn.commit()
            row = self.conn.execute(
                'SELECT host_name, pid FROM processing_claims WHERE file_md5=?',
                (file_md5,)).fetchone()
            if row and row['host_name'] == host_name and row['pid'] == pid:
                return True
            return False
        except Exception as e:
            print(f"声明视频处理失败: {e}")
            return False

    def refresh_claim(self, file_md5):
        """刷新当前进程持有的 claim 心跳。"""
        if not file_md5:
            return False
        try:
            now = time.time()
            host_name = socket.gethostname()
            pid = os.getpid()
            with self.conn:
                cur = self.conn.execute(
                    """
                    UPDATE processing_claims
                    SET heartbeat_at=?
                    WHERE file_md5=? AND host_name=? AND pid=?
                    """,
                    (now, file_md5, host_name, pid))
            return bool(cur.rowcount)
        except Exception as e:
            print(f"刷新视频声明心跳失败: {e}")
            return False

    def release_claim(self, file_md5):
        """释放视频声明，允许其他机器处理。"""
        if not file_md5:
            return
        try:
            self.conn.execute(
                'DELETE FROM processing_claims WHERE file_md5=?', (file_md5,))
            self.conn.commit()
        except Exception as e:
            print(f"释放视频声明失败: {e}")

    def is_video_claimed(self, file_md5):
        """检查视频是否已被其他机器声明。"""
        if not file_md5:
            return False
        try:
            row = self.conn.execute(
                'SELECT host_name, pid, claimed_at, heartbeat_at FROM processing_claims WHERE file_md5=?',
                (file_md5,)).fetchone()
            if not row:
                return False
            claimed_at = row['heartbeat_at'] or row['claimed_at']
            ttl = int(os.environ.get('FINDINVIDEO_CLAIM_TTL_SECONDS', '86400'))
            if time.time() - claimed_at > ttl:
                self.release_claim(file_md5)
                return False
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

DIRECTORY_INDEX = DirectoryIndex()
atexit.register(DIRECTORY_INDEX.close)
