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
import uuid


def _get_process_started_at(pid):
    """获取本机进程启动时间，用于识别 PID 复用；无法确认时返回 None。"""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0:
        return None

    if os.name == 'nt':
        try:
            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            open_process = kernel32.OpenProcess
            open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            open_process.restype = wintypes.HANDLE
            get_process_times = kernel32.GetProcessTimes
            get_process_times.argtypes = [
                wintypes.HANDLE,
                ctypes.POINTER(wintypes.FILETIME),
                ctypes.POINTER(wintypes.FILETIME),
                ctypes.POINTER(wintypes.FILETIME),
                ctypes.POINTER(wintypes.FILETIME),
            ]
            get_process_times.restype = wintypes.BOOL
            close_handle = kernel32.CloseHandle
            close_handle.argtypes = [wintypes.HANDLE]
            close_handle.restype = wintypes.BOOL

            handle = open_process(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
            if not handle:
                return None
            try:
                creation = wintypes.FILETIME()
                exit_time = wintypes.FILETIME()
                kernel_time = wintypes.FILETIME()
                user_time = wintypes.FILETIME()
                if not get_process_times(
                    handle,
                    ctypes.byref(creation),
                    ctypes.byref(exit_time),
                    ctypes.byref(kernel_time),
                    ctypes.byref(user_time),
                ):
                    return None
                ticks = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
                return ticks / 10_000_000.0 - 11_644_473_600.0
            finally:
                close_handle(handle)
        except Exception:
            return None

    try:
        with open(f'/proc/{pid}/stat', 'r', encoding='utf-8', errors='ignore') as f:
            stat_line = f.read()
        right_paren = stat_line.rfind(')')
        if right_paren < 0:
            return None
        fields = stat_line[right_paren + 2:].split()
        start_ticks = int(fields[19])
        clock_ticks = int(os.sysconf('SC_CLK_TCK'))
        boot_time = None
        with open('/proc/stat', 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('btime '):
                    boot_time = float(line.split()[1])
                    break
        if boot_time is None or clock_ticks <= 0:
            return None
        return boot_time + start_ticks / clock_ticks
    except Exception:
        return None


def _is_process_alive(pid):
    """判断本机 PID 是否仍存活；权限不足等未知情况返回 None。"""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False

    if os.name == 'nt':
        try:
            kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
            open_process = kernel32.OpenProcess
            open_process.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            open_process.restype = wintypes.HANDLE
            wait_for_single_object = kernel32.WaitForSingleObject
            wait_for_single_object.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            wait_for_single_object.restype = wintypes.DWORD
            close_handle = kernel32.CloseHandle
            close_handle.argtypes = [wintypes.HANDLE]
            close_handle.restype = wintypes.BOOL

            handle = open_process(0x00100000 | 0x1000, False, pid)  # SYNCHRONIZE | QUERY
            if not handle:
                error = ctypes.get_last_error()
                if error == 87:  # ERROR_INVALID_PARAMETER
                    return False
                return None
            try:
                status = wait_for_single_object(handle, 0)
                if status == 0:  # WAIT_OBJECT_0
                    return False
                if status == 258:  # WAIT_TIMEOUT
                    return True
                return None
            finally:
                close_handle(handle)
        except Exception:
            return None

    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    except OSError:
        return False


def _get_machine_id():
    """获取稳定且脱敏的本机标识；无法确认时返回 None。"""
    raw_id = None
    source = None
    if os.name == 'nt':
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r'SOFTWARE\Microsoft\Cryptography',
            ) as key:
                raw_id = str(winreg.QueryValueEx(key, 'MachineGuid')[0]).strip()
                source = 'windows'
        except Exception:
            return None
    else:
        for path in ('/etc/machine-id', '/var/lib/dbus/machine-id'):
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    raw_id = f.read().strip()
                if raw_id:
                    source = 'linux'
                    break
            except Exception:
                continue
    if not raw_id or not source:
        return None
    return hashlib.sha256(f'{source}:{raw_id}'.encode('utf-8')).hexdigest()


_PROCESS_HOST_NAME = socket.gethostname()
_PROCESS_HOST_ID = _get_machine_id()
_PROCESS_PID = os.getpid()
_PROCESS_STARTED_AT = _get_process_started_at(_PROCESS_PID)
_PROCESS_OWNER_TOKEN = uuid.uuid4().hex


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

CLAIM_TTL_DEFAULT_SECONDS = 86400
CLAIM_TTL_MIN_SECONDS = 60
CLAIM_HEARTBEAT_MAX_SECONDS = 60.0


def get_claim_ttl_seconds():
    """读取 claim TTL，并拒绝短于一分钟的不安全配置。"""
    try:
        configured = int(
            os.environ.get(
                'FINDINVIDEO_CLAIM_TTL_SECONDS', str(CLAIM_TTL_DEFAULT_SECONDS)
            )
        )
    except (TypeError, ValueError):
        configured = CLAIM_TTL_DEFAULT_SECONDS
    return max(CLAIM_TTL_MIN_SECONDS, configured)


def get_claim_heartbeat_interval_seconds():
    """让心跳始终显著短于 TTL，避免活跃 claim 被误回收。"""
    return max(
        1.0,
        min(CLAIM_HEARTBEAT_MAX_SECONDS, get_claim_ttl_seconds() / 3.0),
    )


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
    except (PermissionError, FileNotFoundError, OSError):
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
        'host_name': _PROCESS_HOST_NAME,
        'host_id': _PROCESS_HOST_ID,
        'pid': _PROCESS_PID,
        'owner_token': _PROCESS_OWNER_TOKEN,
        'owner_started_at': _PROCESS_STARTED_AT,
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


def has_completed_artifact(video_path, existing_names_lower=None):
    """判断视频是否有可靠的完成产物。

    所有命名版本都只认 ``.done``，避免异常退出留下的 ``_frames.mp4``、
    ``.txt`` 等半成品被误判为完成。
    """
    try:
        if os.path.exists(_checkpoint_path(video_path)):
            return False
    except Exception:
        pass

    video_dir = os.path.dirname(video_path) or '.'

    def _exists(file_name):
        if existing_names_lower is not None:
            return file_name.lower() in existing_names_lower
        return os.path.exists(os.path.join(video_dir, file_name))

    for base in (
        safe_artifact_basename(video_path),
        legacy_artifact_basename(video_path),
        _legacy_artifact_basename_v1(video_path),
    ):
        if _exists(base + DONE_SUFFIX):
            return True
    return False


_CLAIM_MATCH_MD5_CACHE = {}


def _claim_path_aliases(path):
    """生成不依赖当前操作系统的路径别名，用于 Windows/WSL claim 对照。"""
    aliases = set()

    def _add(value):
        if value is None:
            return
        normalized = str(value).strip().replace('\\', '/')
        if normalized:
            aliases.add(normalized.rstrip('/').casefold())

    raw = str(path or '')
    _add(raw)
    try:
        _add(canonical_video_path(raw))
    except Exception:
        pass

    if is_windows_style_path(raw):
        drive = raw[0].lower()
        remainder = raw[2:].replace('\\', '/').lstrip('/')
        _add(f'/mnt/{drive}/{remainder}' if remainder else f'/mnt/{drive}')
    else:
        normalized = raw.replace('\\', '/')
        prefix = '/mnt/'
        if normalized.casefold().startswith(prefix) and len(normalized) > len(prefix):
            drive = normalized[len(prefix)]
            if drive.isalpha():
                remainder = normalized[len(prefix) + 1:].lstrip('/')
                _add(f'{drive.upper()}:/{remainder}' if remainder else f'{drive.upper()}:/')
    return aliases


def _claim_path_basename(path):
    normalized = str(path or '').replace('\\', '/').rstrip('/')
    return normalized.rsplit('/', 1)[-1].casefold() if normalized else ''


def _claim_match_file_md5(path):
    """只在跨平台路径无法直接对应时计算 MD5，结果按文件状态缓存。"""
    try:
        stat_info = os.stat(path)
        key = (str(path), int(stat_info.st_size), float(stat_info.st_mtime))
        cached = _CLAIM_MATCH_MD5_CACHE.get(key)
        if cached is not None:
            return cached
        digest = hashlib.md5()
        with open(path, 'rb') as stream:
            while True:
                chunk = stream.read(4 * 1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        value = digest.hexdigest()
        if len(_CLAIM_MATCH_MD5_CACHE) > 2048:
            _CLAIM_MATCH_MD5_CACHE.clear()
        _CLAIM_MATCH_MD5_CACHE[key] = value
        return value
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 暂停系统
# ---------------------------------------------------------------------------

_STOP_REQUESTED = False


class PauseRequested(Exception):
    """Raised to request a graceful stop with checkpoint saved."""


class ClaimLostError(RuntimeError):
    """处理中的视频 claim 已被回收或转移。"""


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

    def __init__(
        self,
        db_path=None,
        owner_token=None,
        host_name=None,
        host_id=None,
        pid=None,
        process_started_at=None,
    ):
        self.owner_token = owner_token if owner_token is not None else _PROCESS_OWNER_TOKEN
        self.host_name = host_name if host_name is not None else _PROCESS_HOST_NAME
        self.host_id = host_id if host_id is not None else _PROCESS_HOST_ID
        self.pid = int(pid if pid is not None else _PROCESS_PID)
        started_at = (
            process_started_at
            if process_started_at is not None
            else _PROCESS_STARTED_AT
        )
        self.process_started_at = (
            float(started_at) if started_at is not None else None
        )
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
                            host_id TEXT,
                            pid INTEGER,
                            owner_token TEXT,
                            owner_started_at REAL
                        )
                        """
                    )
                    self._ensure_claim_schema_compat()
                    self.conn.execute(
                        """
                        CREATE INDEX IF NOT EXISTS idx_processing_claims_heartbeat
                        ON processing_claims(heartbeat_at)
                        """
                    )
                    self._ensure_completion_semantics_compat()
                return
            except sqlite3.OperationalError as e:
                if self._is_lock_error(e):
                    if _attempt < 2:
                        print(f"创建表结构被锁，重试中({_attempt + 1}/3)...")
                        time.sleep(5)
                    else:
                        raise RuntimeError(f"创建必要表结构失败: {e}") from e
                else:
                    raise

    def close(self):
        try:
            self.conn.close()
        except sqlite3.Error:
            pass

    def _ensure_claim_schema_compat(self):
        """为旧数据库补齐 processing_claims 新字段。"""
        additions = {
            'heartbeat_at': 'REAL',
            'host_id': 'TEXT',
            'owner_token': 'TEXT',
            'owner_started_at': 'REAL',
        }
        for column, column_type in additions.items():
            try:
                rows = self.conn.execute('PRAGMA table_info(processing_claims)').fetchall()
                columns = {
                    row['name'] if isinstance(row, sqlite3.Row) else row[1]
                    for row in rows
                }
                if column in columns:
                    continue
                self.conn.execute(
                    f'ALTER TABLE processing_claims ADD COLUMN {column} {column_type}'
                )
            except sqlite3.OperationalError as e:
                # 多实例可能同时迁移；若另一实例已经补列，则视为成功。
                rows = self.conn.execute('PRAGMA table_info(processing_claims)').fetchall()
                columns = {
                    row['name'] if isinstance(row, sqlite3.Row) else row[1]
                    for row in rows
                }
                if column not in columns:
                    raise e
        rows = self.conn.execute('PRAGMA table_info(processing_claims)').fetchall()
        columns = {
            row['name'] if isinstance(row, sqlite3.Row) else row[1]
            for row in rows
        }
        missing = set(additions) - columns
        if missing:
            raise RuntimeError(
                f"processing_claims 缺少必要字段: {', '.join(sorted(missing))}"
            )

    def _ensure_completion_semantics_compat(self):
        """完成产物改为只认 .done 时，使旧目录缓存强制重新扫描一次。"""
        marker_key = 'completion_requires_done_v1'
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS index_metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )
        claimed = self.conn.execute(
            'INSERT OR IGNORE INTO index_metadata (key, value) VALUES (?, ?)',
            (marker_key, str(time.time())),
        )
        if claimed.rowcount != 1:
            return
        self.conn.execute(
            'UPDATE directories SET has_artifact=0, dir_mtime=NULL'
        )

    def reopen(self, new_db_path):
        """关闭当前连接并在新路径重新打开数据库。"""
        self.release_all_claims()
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
        cleaned = self.cleanup_stale_claims()
        if cleaned:
            print(f"已清理 {cleaned} 条失效视频声明")
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
            video_paths = []
            for v_name, _, _ in video_records:
                video_path = os.path.join(normalized, v_name)
                video_paths.append(video_path)
                if has_completed_artifact(video_path, all_file_names_lower):
                    processed_count += 1
            has_artifact = (
                processed_count == len(video_records)
                and not self.has_active_claims_for_paths(video_paths)
            )
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

    @staticmethod
    def _directory_completion_snapshot(dir_path, video_paths):
        """在写锁外用一次目录扫描取得稳定的视频/checkpoint 快照。"""
        expected_videos = {
            _claim_path_basename(path) for path in video_paths or []
        }
        try:
            before = os.stat(dir_path)
            names_lower = set()
            actual_videos = set()
            with os.scandir(dir_path) as iterator:
                for entry in iterator:
                    name_key = entry.name.casefold()
                    names_lower.add(name_key)
                    if is_video_file(entry.name):
                        actual_videos.add(_claim_path_basename(entry.name))
            after = os.stat(dir_path)
        except (PermissionError, FileNotFoundError, OSError):
            return None

        before_token = (
            int(getattr(before, 'st_mtime_ns', before.st_mtime * 1_000_000_000)),
            int(getattr(before, 'st_size', 0)),
        )
        after_token = (
            int(getattr(after, 'st_mtime_ns', after.st_mtime * 1_000_000_000)),
            int(getattr(after, 'st_size', 0)),
        )
        if before_token != after_token or actual_videos != expected_videos:
            return None
        for video_name in actual_videos:
            checkpoint_name = (
                os.path.splitext(video_name)[0] + CHECKPOINT_SUFFIX
            ).casefold()
            if checkpoint_name in names_lower:
                return None
        return {
            'mtime': float(after.st_mtime),
            'token': after_token,
        }

    def mark_directory_processed_if_idle(self, dir_path, video_paths):
        """在写锁内确认无 checkpoint/claim 后标记目录完成。

        这里有意对整个共享索引中的 claim 保守：目录完成标记只是快速跳过优化，
        宁可延后一轮写入，也不能在另一实例仍工作时制造完成态竞态。
        """
        normalized = self._normalize_path(dir_path)
        if not normalized:
            return False
        snapshot = self._directory_completion_snapshot(normalized, video_paths)
        if snapshot is None:
            return False
        self.cleanup_stale_claims()
        try:
            self.conn.execute('BEGIN IMMEDIATE')
            if self.conn.execute(
                'SELECT 1 FROM processing_claims LIMIT 1'
            ).fetchone() is not None:
                self.conn.rollback()
                return False
            try:
                locked_stat = os.stat(normalized)
            except (PermissionError, FileNotFoundError, OSError):
                self.conn.rollback()
                return False
            locked_token = (
                int(
                    getattr(
                        locked_stat,
                        'st_mtime_ns',
                        locked_stat.st_mtime * 1_000_000_000,
                    )
                ),
                int(getattr(locked_stat, 'st_size', 0)),
            )
            if locked_token != snapshot['token']:
                self.conn.rollback()
                return False
            self.conn.execute(
                'UPDATE directories SET has_artifact=1, dir_mtime=? WHERE path=?',
                (snapshot['mtime'], normalized),
            )
            self.conn.commit()
            return True
        except Exception as e:
            try:
                self.conn.rollback()
            except Exception:
                pass
            print(f"原子标记目录已处理失败: {e}")
            return False

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

    def _upsert_processed_video(
        self, file_md5, video_path, detection_count=0, model_name=None
    ):
        self.conn.execute(
            """
            INSERT INTO processed_videos (
                file_md5, video_path, processed_at, detection_count, model_name
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(file_md5) DO UPDATE SET
                video_path=excluded.video_path,
                processed_at=excluded.processed_at,
                detection_count=excluded.detection_count,
                model_name=excluded.model_name
            """,
            (
                file_md5,
                str(video_path) if video_path else None,
                time.time(),
                detection_count,
                model_name,
            ),
        )

    def mark_video_processed(self, file_md5, video_path, detection_count=0, model_name=None):
        """补录视频完成状态，不触碰可能属于其他进程的 claim。"""
        if not file_md5:
            return False
        try:
            with self.conn:
                cur = self.conn.execute(
                    """
                    INSERT INTO processed_videos (
                        file_md5, video_path, processed_at, detection_count, model_name
                    )
                    SELECT ?, ?, ?, ?, ?
                    WHERE NOT EXISTS (
                        SELECT 1 FROM processing_claims WHERE file_md5=?
                    )
                    ON CONFLICT(file_md5) DO UPDATE SET
                        video_path=excluded.video_path,
                        processed_at=excluded.processed_at,
                        detection_count=excluded.detection_count,
                        model_name=excluded.model_name
                    WHERE NOT EXISTS (
                        SELECT 1 FROM processing_claims
                        WHERE file_md5=excluded.file_md5
                    )
                    """,
                    (
                        file_md5,
                        str(video_path) if video_path else None,
                        time.time(),
                        detection_count,
                        model_name,
                        file_md5,
                    ),
                )
            return cur.rowcount == 1
        except Exception as e:
            print(f"标记视频已处理失败: {e}")
            return False

    def complete_claimed_video(
        self, file_md5, video_path, detection_count=0, model_name=None
    ):
        """仅允许当前 claim 持有者在同一事务中消费 claim 并写入完成态。"""
        if not file_md5:
            return False
        try:
            with self.conn:
                cur = self.conn.execute(
                    """
                    DELETE FROM processing_claims
                    WHERE file_md5=? AND owner_token=?
                    """,
                    (file_md5, self.owner_token),
                )
                if cur.rowcount != 1:
                    return False
                self._upsert_processed_video(
                    file_md5, video_path, detection_count, model_name
                )
            return True
        except Exception as e:
            print(f"完成视频 claim 失败: {e}")
            return False

    def is_video_processed_by_md5(self, file_md5, include_directory_backfill=True):
        """根据文件MD5查询视频是否已处理过。"""
        if not file_md5:
            return False
        try:
            row = self.conn.execute(
                'SELECT detection_count FROM processed_videos WHERE file_md5=?',
                (file_md5,),
            ).fetchone()
            if row is None:
                return False
            if not include_directory_backfill and row['detection_count'] == -1:
                return False
            return True
        except Exception:
            return False

    @staticmethod
    def _claim_ttl_seconds():
        return get_claim_ttl_seconds()

    def _get_claim_row(self, file_md5):
        return self.conn.execute(
            """
            SELECT file_md5, video_path, claimed_at, heartbeat_at,
                   host_name, host_id, pid, owner_token, owner_started_at
            FROM processing_claims
            WHERE file_md5=?
            """,
            (file_md5,),
        ).fetchone()

    @staticmethod
    def _claim_snapshot_values(row):
        return (
            row['file_md5'],
            row['video_path'],
            row['claimed_at'],
            row['heartbeat_at'],
            row['host_name'],
            row['host_id'],
            row['pid'],
            row['owner_token'],
            row['owner_started_at'],
        )

    def _local_claim_owner_alive(self, row):
        """返回本机 claim 持有者是否仍是原进程；无法确认时返回 None。"""
        if row['owner_token'] and row['owner_token'] == self.owner_token:
            return True
        alive = _is_process_alive(row['pid'])
        if alive is not True:
            return alive

        actual_started_at = _get_process_started_at(row['pid'])
        expected_started_at = row['owner_started_at']
        if actual_started_at is not None and expected_started_at is not None:
            return float(actual_started_at) == float(expected_started_at)

        # 兼容旧表：若当前 PID 的进程启动时间晚于 claim 创建时间，说明 PID 已复用。
        if actual_started_at is not None and row['claimed_at'] is not None:
            if float(actual_started_at) > float(row['claimed_at']):
                return False
        # PID 存活但无法核对启动身份时不能永久保留；返回 None 让 TTL 决定。
        return None

    def _claim_is_reclaimable(self, row, now=None):
        now = time.time() if now is None else float(now)
        same_machine = bool(row['host_id'] and self.host_id) and (
            str(row['host_id']) == str(self.host_id)
        )
        legacy_same_machine = (
            not row['host_id']
            and not row['owner_token']
            and bool(row['host_name'] and self.host_name)
            and str(row['host_name']).casefold() == str(self.host_name).casefold()
        )
        if same_machine or legacy_same_machine:
            alive = self._local_claim_owner_alive(row)
            if alive is True:
                return False
            if alive is False:
                return True

        last_heartbeat = row['heartbeat_at'] or row['claimed_at'] or 0
        return now - float(last_heartbeat) > self._claim_ttl_seconds()

    def _invalidate_claim_directory(self, video_path):
        """领取视频时撤销父目录完成标记；调用方必须位于同一写事务内。"""
        if not video_path:
            return
        parent = self._normalize_path(os.path.dirname(str(video_path)))
        if parent:
            self.conn.execute(
                'UPDATE directories SET has_artifact=0 WHERE path=?',
                (parent,),
            )

    def _replace_claim_snapshot(self, row, video_path, now):
        """仅在 claim 快照未变化时原子接管，避免误删或覆盖新持有者。"""
        with self.conn:
            cur = self.conn.execute(
                """
                UPDATE processing_claims
                SET video_path=?, claimed_at=?, heartbeat_at=?, host_name=?, host_id=?,
                    pid=?, owner_token=?, owner_started_at=?
                WHERE file_md5=?
                  AND video_path IS ?
                  AND claimed_at IS ?
                  AND heartbeat_at IS ?
                  AND host_name IS ?
                  AND host_id IS ?
                  AND pid IS ?
                  AND owner_token IS ?
                  AND owner_started_at IS ?
                  AND NOT EXISTS (
                      SELECT 1 FROM processed_videos
                      WHERE processed_videos.file_md5=processing_claims.file_md5
                  )
                """,
                (
                    str(video_path) if video_path else None,
                    now,
                    now,
                    self.host_name,
                    self.host_id,
                    self.pid,
                    self.owner_token,
                    self.process_started_at,
                    *self._claim_snapshot_values(row),
                ),
            )
            if cur.rowcount == 1:
                self._invalidate_claim_directory(video_path)
        return cur.rowcount == 1

    def _delete_claim_snapshot(self, row):
        with self.conn:
            cur = self.conn.execute(
                """
                DELETE FROM processing_claims
                WHERE file_md5=?
                  AND video_path IS ?
                  AND claimed_at IS ?
                  AND heartbeat_at IS ?
                  AND host_name IS ?
                  AND host_id IS ?
                  AND pid IS ?
                  AND owner_token IS ?
                  AND owner_started_at IS ?
                """,
                self._claim_snapshot_values(row),
            )
        return cur.rowcount == 1

    def try_claim_video(self, file_md5, video_path):
        """在开工前原子领取视频；同机死进程可立即接管。"""
        if not file_md5:
            return False
        try:
            resume_incomplete = bool(video_path) and os.path.exists(
                _checkpoint_path(video_path)
            )
        except Exception:
            resume_incomplete = False
        try:
            for _attempt in range(3):
                now = time.time()
                with self.conn:
                    if resume_incomplete:
                        self.conn.execute(
                            """
                            DELETE FROM processed_videos
                            WHERE file_md5=? AND detection_count=-1
                            """,
                            (file_md5,),
                        )
                    cur = self.conn.execute(
                        """
                        INSERT INTO processing_claims (
                            file_md5, video_path, claimed_at, heartbeat_at,
                            host_name, host_id, pid, owner_token, owner_started_at
                        )
                        SELECT ?, ?, ?, ?, ?, ?, ?, ?, ?
                        WHERE NOT EXISTS (
                            SELECT 1 FROM processed_videos WHERE file_md5=?
                        )
                        ON CONFLICT(file_md5) DO NOTHING
                        """,
                        (
                            file_md5,
                            str(video_path) if video_path else None,
                            now,
                            now,
                            self.host_name,
                            self.host_id,
                            self.pid,
                            self.owner_token,
                            self.process_started_at,
                            file_md5,
                        ),
                    )
                    if cur.rowcount == 1:
                        self._invalidate_claim_directory(video_path)
                if cur.rowcount == 1:
                    return True

                row = self._get_claim_row(file_md5)
                if not row:
                    continue
                if self.is_video_processed_by_md5(file_md5):
                    return False
                if row['owner_token'] == self.owner_token:
                    return self.refresh_claim(file_md5)
                if not self._claim_is_reclaimable(row, now=now):
                    return False
                if self._replace_claim_snapshot(row, video_path, now):
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
            with self.conn:
                cur = self.conn.execute(
                    """
                    UPDATE processing_claims
                    SET heartbeat_at=?
                    WHERE file_md5=? AND owner_token=?
                    """,
                    (now, file_md5, self.owner_token),
                )
            return bool(cur.rowcount)
        except Exception as e:
            print(f"刷新视频声明心跳失败: {e}")
            return False

    def release_claim(self, file_md5):
        """只释放当前进程会话持有的视频声明。"""
        if not file_md5:
            return False
        try:
            with self.conn:
                cur = self.conn.execute(
                    """
                    DELETE FROM processing_claims
                    WHERE file_md5=? AND owner_token=?
                    """,
                    (file_md5, self.owner_token),
                )
            return cur.rowcount == 1
        except Exception as e:
            print(f"释放视频声明失败: {e}")
            return False

    def release_all_claims(self):
        """释放当前进程会话的全部 claim，用于正常退出和切换数据库。"""
        try:
            with self.conn:
                cur = self.conn.execute(
                    'DELETE FROM processing_claims WHERE owner_token=?',
                    (self.owner_token,),
                )
            return max(0, int(cur.rowcount or 0))
        except Exception:
            return 0

    def cleanup_stale_claims(self):
        """清理异机过期 claim 和本机已死亡进程 claim。"""
        cleaned = 0
        try:
            rows = self.conn.execute(
                """
                SELECT file_md5, video_path, claimed_at, heartbeat_at,
                       host_name, host_id, pid, owner_token, owner_started_at
                FROM processing_claims
                """
            ).fetchall()
            now = time.time()
            for row in rows:
                if self._claim_is_reclaimable(row, now=now):
                    cleaned += int(self._delete_claim_snapshot(row))
        except Exception as e:
            print(f"清理失效视频声明失败: {e}")
        return cleaned

    def is_video_claimed(self, file_md5):
        """检查视频是否已有有效 claim，并安全清理确认失效的快照。"""
        if not file_md5:
            return False
        try:
            for _attempt in range(2):
                row = self._get_claim_row(file_md5)
                if not row:
                    return False
                if not self._claim_is_reclaimable(row):
                    return True
                if self._delete_claim_snapshot(row):
                    return False
            return self._get_claim_row(file_md5) is not None
        except Exception as e:
            print(f"检查视频声明失败: {e}")
            return False

    def has_active_claims_for_paths(self, video_paths):
        """检查给定视频路径中是否存在有效 claim，兼容 Windows/WSL 别名。"""
        target_records = []
        target_aliases = set()
        for path in video_paths or []:
            aliases = _claim_path_aliases(path)
            target_aliases.update(aliases)
            target_records.append((path, _claim_path_basename(path)))
        if not target_records:
            return False
        try:
            rows = self.conn.execute(
                """
                SELECT file_md5, video_path, claimed_at, heartbeat_at,
                       host_name, host_id, pid, owner_token, owner_started_at
                FROM processing_claims
                """
            ).fetchall()
            unmatched_active_rows = []
            for row in rows:
                if self._claim_is_reclaimable(row):
                    if not self._delete_claim_snapshot(row):
                        # 快照已变化，说明 claim 被刷新或接管；保守视为仍在使用。
                        return True
                    continue
                if _claim_path_aliases(row['video_path']) & target_aliases:
                    return True
                unmatched_active_rows.append(row)

            # UNC 与自定义 WSL 挂载无法仅靠字符串互转时，用 claim 的 MD5 核实。
            for row in unmatched_active_rows:
                claim_basename = _claim_path_basename(row['video_path'])
                if not claim_basename:
                    continue
                for path, target_basename in target_records:
                    if target_basename != claim_basename:
                        continue
                    digest = _claim_match_file_md5(path)
                    if digest is None:
                        # 同名候选无法核实时宁可延后目录完成，避免误标。
                        return True
                    if digest == row['file_md5']:
                        return True
            return False
        except Exception as e:
            print(f"检查目录视频声明失败: {e}")
            # 无法确认时保守阻止目录完成，避免把在途视频误标完成。
            return True


# ---------------------------------------------------------------------------
# 全局单例
# ---------------------------------------------------------------------------

DIRECTORY_INDEX = DirectoryIndex()
atexit.register(DIRECTORY_INDEX.close)
atexit.register(DIRECTORY_INDEX.release_all_claims)
