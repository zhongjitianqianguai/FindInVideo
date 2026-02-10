import sys
import os
import shutil
# 添加yolov5文件夹到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

import torch
import cv2
import numpy as np
from tqdm import tqdm
import subprocess
import json
import re
import time
import hashlib
import logging
import signal
import socket
import threading
import ctypes
import errno
import traceback
import faulthandler
import ntpath
from ctypes import wintypes

# 常见的视频文件扩展名
VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v',
    '.mpg', '.mpeg', '.3gp', '.f4v', '.ts', '.vob', '.rmvb', '.rm',
    '.asf', '.divx', '.xvid', '.m2ts', '.mts'
}

# 需要排除的路径变量
EXCLUDE_PATHS = [
    os.path.abspath(r"D:\$RECYCLE.BIN"),
    os.path.abspath(r"D:\System Volume Information"),
    # 在这里添加其他需要排除的路径
]

_LOGGER = logging.getLogger("findinvideo_yolov5")
_CRASH_LOG_FH = None


class PauseRequested(Exception):
    """Raised to request a graceful stop with checkpoint saved."""


_STOP_REQUESTED = False

# ---------- cap.read() 超时包装 ----------
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


def _setup_diagnostics():
    """Best-effort logging and crash diagnostics."""
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
                _LOGGER.error("Uncaught exception:\n%s", ''.join(traceback.format_exception(exc_type, exc_value, exc_tb)))
            except Exception:
                pass
            sys.__excepthook__(exc_type, exc_value, exc_tb)

        sys.excepthook = _excepthook
        _LOGGER.info("Diagnostics enabled. Logs: %s", run_log_path)
    except Exception:
        pass


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


def safe_artifact_basename(video_path, max_length=80):
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


def _read_json_file_best_effort(path):
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return json.load(f)
    except Exception:
        return None


def _claim_key_for_file(file_path, md5_hint=None):
    if md5_hint:
        return f"md5_{md5_hint}"
    try:
        st = os.stat(file_path)
        size = int(getattr(st, 'st_size', 0) or 0)
        mtime = float(getattr(st, 'st_mtime', 0.0) or 0.0)
    except Exception:
        size, mtime = 0, 0.0
    raw = f"{size}|{mtime}"
    return f"stat_{hashlib.md5(raw.encode('utf-8', 'ignore')).hexdigest()}"


def try_acquire_video_claim(file_path, md5_hint=None):
    shared = ensure_shared_state_dir()
    if not shared:
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
    try:
        created = _atomic_create_file(claim_path, json.dumps(payload, ensure_ascii=False))
        if created:
            return claim_path
    except Exception as e:
        print(f"创建claim失败: {claim_path}, 错误: {e}")
        return None

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


ARTIFACT_SUFFIXES = [
    '_mosaic.jpg',
    '.txt',
    '.done',
]

DONE_SUFFIX = '.done'


def has_existing_artifacts(video_path):
    try:
        if os.path.exists(_checkpoint_path(video_path)):
            return False
    except Exception:
        pass
    video_dir = os.path.dirname(video_path) or '.'
    bases = [safe_artifact_basename(video_path), legacy_artifact_basename(video_path)]
    for base in bases:
        done_path = os.path.join(video_dir, base + DONE_SUFFIX)
        if os.path.exists(done_path):
            return True
        for suffix in ARTIFACT_SUFFIXES:
            path = os.path.join(video_dir, base + suffix)
            if os.path.exists(path):
                return True
    return False


def write_done_marker(video_path):
    try:
        video_dir = os.path.dirname(video_path) or '.'
        base = safe_artifact_basename(video_path)
        marker = os.path.join(video_dir, base + DONE_SUFFIX)
        if os.path.exists(marker):
            return
        with open(marker, 'w', encoding='utf-8', errors='ignore') as f:
            f.write('done\n')
    except Exception:
        pass


_YOLOED_MD5_CACHE = None
_YOLOED_MD5_CACHE_MTIME = None
_YOLOED_PATH_CACHE = None
_FILE_MD5_CACHE = {}


def get_yoloed_md5_path():
    yoloed_path = _get_env_path('FINDINVIDEO_YOLOED_PATH_YOLOV5')
    if not yoloed_path:
        shared = get_shared_state_dir()
        if shared:
            yoloed_path = os.path.join(shared, 'yoloed_yolov5.txt')
        else:
            yoloed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'yoloed_yolov5.txt')
    if os.name == 'posix':
        yoloed_path = windows_path_to_wsl(yoloed_path) or yoloed_path
    return yoloed_path


def get_file_md5_cached(file_path):
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
            print(f"yoloed目录创建失败: {e}")
    if not os.path.exists(path):
        try:
            with open(path, 'a', encoding='utf-8', errors='ignore'):
                pass
        except Exception as e:
            print(f"yoloed文件创建失败: {e}")
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
                line = line.strip()
                if not line:
                    continue
                if '|' in line:
                    md5, p = line.split('|', 1)
                    md5_set.add(md5.strip())
                    if p:
                        path_set.add(p.strip())
                else:
                    md5_set.add(line)
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
        try:
            _YOLOED_MD5_CACHE_MTIME = os.path.getmtime(path)
        except Exception:
            _YOLOED_MD5_CACHE_MTIME = None
        if _YOLOED_MD5_CACHE is not None:
            _YOLOED_MD5_CACHE.add(md5)
        if file_path and _YOLOED_PATH_CACHE is not None:
            _YOLOED_PATH_CACHE.add(file_path)
    except Exception as e:
        print(f"写入已识别MD5失败: {e}")


def is_path_already_yoloed(file_path):
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

def is_video_file(file_path):
    """检查文件是否为视频文件"""
    _, ext = os.path.splitext(file_path.lower())
    return ext in VIDEO_EXTENSIONS

_IGNORED_SUBDIRS = {'yolov5_output', '__pycache__', '.git', '$RECYCLE.BIN', 'System Volume Information'}

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
    count = 0
    try:
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path) and is_video_file(file_path):
                count += 1
    except (PermissionError, FileNotFoundError, OSError):
        pass
    return count

def find_leaf_directories_with_videos(root_path, exclusions=None):
    """
    递归查找所有包含视频文件的叶子节点目录
    返回: [(目录路径, 视频数量), ...] 按视频数量降序排列
    """
    if exclusions is None:
        exclusions = EXCLUDE_PATHS
    
    leaf_dirs = []
    
    # 检查根路径是否被排除
    for ex_path in exclusions:
        if root_path.startswith(ex_path):
            return leaf_dirs
    
    try:
        for root, dirs, files in os.walk(root_path):
            # 检查当前目录是否被排除
            is_excluded = any(root.startswith(ex_path) for ex_path in exclusions)
            if is_excluded:
                continue
            
            # 过滤掉被排除的子目录
            dirs[:] = [d for d in dirs if not any(os.path.join(root, d).startswith(ex_path) for ex_path in exclusions)]
            
            # 检查是否为叶子节点
            if is_leaf_directory(root):
                video_count = count_videos_in_directory(root)
                if video_count > 0:
                    leaf_dirs.append((root, video_count))
    
    except (PermissionError, FileNotFoundError, OSError) as e:
        print(f"警告: 无法访问目录 '{root_path}': {e}")
    
    # 按视频数量降序排列
    leaf_dirs.sort(key=lambda x: x[1], reverse=True)
    return leaf_dirs

def load_yolov5_model(model_path, device='cpu'):
    """加载YOLOv5模型"""
    try:
        # 直接导入本地YOLOv5模块
        from yolov5.models.experimental import attempt_load
        from yolov5.utils.general import non_max_suppression
        from yolov5.utils.torch_utils import select_device
        
        # 尝试导入新版本的scale_boxes，如果失败则尝试scale_coords
        try:
            from yolov5.utils.general import scale_boxes as scale_coords_func
        except ImportError:
            try:
                from yolov5.utils.general import scale_coords as scale_coords_func
            except ImportError:
                print("警告: 无法导入坐标缩放函数，将使用手动缩放")
                scale_coords_func = None
        
        print(f"成功导入本地YOLOv5模块")
        
        device = select_device(device)
        
        # 尝试不同的参数组合来加载模型
        try:
            # 新版本YOLOv5使用weights参数
            model = attempt_load(weights=model_path, device=device)
        except TypeError:
            try:
                # 旧版本使用map_location参数
                model = attempt_load(model_path, map_location=device)
            except TypeError:
                try:
                    # 最简单的调用方式
                    model = attempt_load(model_path)
                    model = model.to(device)
                except Exception as e:
                    print(f"所有加载方式都失败: {e}")
                    raise e
        
        model.eval()
        
        return model, non_max_suppression, scale_coords_func, device
        
    except ImportError as e:
        print(f"导入YOLOv5模块失败: {e}")
        print("请确保yolov5文件夹在项目目录下，并包含必要的模块文件")
        raise e
    except Exception as e:
        print(f"加载YOLOv5模型失败: {e}")
        raise e

def detect_objects_in_video_yolov5(video_path, target_class,
                                   show_window=False, save_crops=False,
                                   save_training_data=False,
                                   all_objects=False,
                                   model_path='models/breast.pt'):
    """使用YOLOv5进行视频目标检测"""
    
    # 如果不开启全量检测，则保证 target_class 为列表
    if not all_objects and isinstance(target_class, str):
        target_class = [target_class]

    # 加载YOLOv5模型
    try:
        model, nms_func, scale_coords_func, device = load_yolov5_model(model_path)
        print(f"成功加载YOLOv5模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return []

    # 获取类别名称
    if hasattr(model, 'names'):
        class_names = model.names
    elif hasattr(model, 'module') and hasattr(model.module, 'names'):
        class_names = model.module.names
    else:
        # 默认COCO类别
        class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
                      6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
                      # ... 可以添加更多COCO类别
                      }

    # 若需要生成训练数据，则构造保存目录及生成 classes.txt 文件
    if save_training_data:
        training_folder = os.path.join(os.path.dirname(video_path), "training_data")
        os.makedirs(training_folder, exist_ok=True)
        classes_txt = os.path.join(training_folder, "classes.txt")
        if not os.path.exists(classes_txt):
            with open(classes_txt, "w") as f:
                if all_objects:
                    # 写入模型中所有的类别，按 key 排序
                    for key in sorted(class_names.keys()):
                        f.write(class_names[key] + "\n")
                else:
                    for cls in target_class:
                        f.write(cls + "\n")

    pause_file = _get_pause_file_path()
    resume_enabled = _truthy_env('FINDINVIDEO_RESUME', default=True)
    ckpt = _load_checkpoint(video_path) if resume_enabled else None
    start_frame = int(ckpt.get('next_frame', 0)) if ckpt else 0
    detections = list(ckpt.get('detections', [])) if ckpt else []
    last_detected = float(ckpt.get('last_detected', detections[-1] if detections else -5.0)) if ckpt else -5
    
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

    frame_count = start_frame
    
    # 截图存储配置（用于拼接大图）
    crop_size = (160, 160)  # 统一缩放到的小图尺寸
    max_cols = 8  # 拼接大图每行最多显示数量
    crops = []  # 存储所有截取的目标区域
    
    # 输入尺寸
    img_size = 640
    
    while cap.isOpened():
        if _pause_requested(pause_file):
            _save_checkpoint(video_path, next_frame=frame_count, detections=detections, last_detected=last_detected)
            raise PauseRequested()
        success, frame, timed_out = _read_frame_with_timeout(cap)
        if not success:
            if timed_out:
                _LOGGER.warning("读帧超时(>%ds)，尝试跳过: %s (frame=%d)", _READ_TIMEOUT_SEC, video_path, frame_count)
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count + 1)
                except Exception:
                    break
                pbar.update(1)
                frame_count += 1
                continue
            break  # 正常 EOF
        if frame is None:
            print(f"读取到空帧，已跳过: {video_path} (frame={frame_count})")
            pbar.update(1)
            frame_count += 1
            continue
        
        # 更新进度条
        pbar.update(1)
        
        if save_training_data:
            frame_annotations = []
        
        pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_msec and pos_msec > 0:
            current_time = float(pos_msec) / 1000.0
        else:
            current_time = frame_count / fps_safe
        
        # 预处理图像
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (img_size, img_size))
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        detected = False
        
        try:
            # YOLOv5推理
            if device is not None:
                img_tensor = img_tensor.to(device)
            
            with torch.no_grad():
                # 使用本地YOLOv5模型进行推理
                pred = model(img_tensor)[0]
                
                # 使用NMS函数
                pred = nms_func(pred, conf_thres=0.5, iou_thres=0.45)[0]
                
                if pred is not None and len(pred):
                    # 克隆张量以避免原地更新错误
                    pred = pred.clone()
                    
                    # 缩放坐标到原始图像尺寸
                    if scale_coords_func is not None:
                        # 新版本使用scale_boxes，参数顺序可能不同
                        try:
                            pred[:, :4] = scale_coords_func(img_tensor.shape[2:], pred[:, :4], frame.shape).round()
                        except:
                            # 如果参数顺序不对，尝试新的参数顺序
                            pred[:, :4] = scale_coords_func(pred[:, :4], frame.shape, img_tensor.shape[2:]).round()
                    else:
                        # 手动缩放坐标
                        h, w = frame.shape[:2]
                        pred[:, 0] *= w / img_size  # x1
                        pred[:, 1] *= h / img_size  # y1
                        pred[:, 2] *= w / img_size  # x2
                        pred[:, 3] *= h / img_size  # y2
                    
                    for det in pred:
                        x1, y1, x2, y2, conf, cls_id = det[:6]
                        cls_id = int(cls_id)
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        
                        # 获取类别名称
                        class_name = class_names.get(cls_id, f'class_{cls_id}')
                        
                        # 如果全量检测开启，或者检测到的类别在 target_class 内则处理
                        if all_objects or class_name in target_class:
                            if current_time - last_detected >= 0.1:
                                detections.append(current_time)
                                last_detected = current_time
                                detected = True
                            
                            if save_crops:
                                crop = frame[y1:y2, x1:x2]
                                if crop.size > 0:
                                    resized = cv2.resize(crop, crop_size)
                                    crops.append(resized)
                            
                            if show_window:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            if save_training_data:
                                h, w, _ = frame.shape
                                cx = ((x1 + x2) / 2) / w
                                cy = ((y1 + y2) / 2) / h
                                bw = (x2 - x1) / w
                                bh = (y2 - y1) / h
                                # 如果全量检测，类别编号直接用 cls_id；否则使用 target_class 的索引
                                if all_objects:
                                    class_index = cls_id
                                else:
                                    class_index = target_class.index(class_name)
                                annotation_line = f"{class_index} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                                frame_annotations.append(annotation_line)
        
        except KeyboardInterrupt:
            _save_checkpoint(video_path, next_frame=frame_count, detections=detections, last_detected=last_detected)
            raise PauseRequested()
        except Exception as e:
            print(f"处理帧 {frame_count} 时出错: {e}")
            # 继续处理下一帧，而不是中断
            pass
        
        if show_window and detected:
            cv2.imshow('Detection Preview', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if save_training_data and frame_annotations:
            training_base = safe_artifact_basename(video_path)
            training_image_path = os.path.join(training_folder, f"{training_base}_{frame_count}.jpg")
            training_annotation_path = os.path.splitext(training_image_path)[0] + ".txt"
            cv2.imwrite(training_image_path, frame)
            with open(training_annotation_path, 'w') as f:
                for line in frame_annotations:
                    f.write(line + "\n")
        
        frame_count += 1
    
    cap.release()
    pbar.close()
    if show_window:
        cv2.destroyAllWindows()

    if (not _pause_requested(pause_file)) and (frame_count >= total_frames or total_frames == 0):
        _clear_checkpoint(video_path)

    video_dir = os.path.dirname(video_path) or '.'
    artifact_base = safe_artifact_basename(video_path)
    txt_save_path = os.path.join(video_dir, artifact_base + ".txt")
    with open(txt_save_path, 'w') as f:
        f.write("检测到目标的时间位置（秒）:\n")
        for t in detections:
            f.write(f"{t:.2f}\n")
    print(f"已保存检测时间戳至: {txt_save_path}")
    
    if save_crops and crops:
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
            cv2.imshow('All Detected Objects', final_mosaic)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
                mosaic_path = os.path.join(video_dir, artifact_base + "_mosaic.jpg")
                cv2.imwrite(mosaic_path, final_mosaic)
                print(f"已保存拼接图片至: {mosaic_path}")
                write_done_marker(video_path)
    return detections

def should_process(file_path):
    if has_existing_artifacts(file_path):
        return
    md5 = get_file_md5_cached(file_path)
    if not md5:
        return
    yoloed_md5 = load_yoloed_md5(reload=False)
    if md5 in yoloed_md5:
        return
    return True, None


def create_mosaic_from_crops(crops_dir, video_path):
    """从裁剪图片创建拼接图"""
    crops = []
    crop_size = (160, 160)
    max_cols = 8
    
    # 收集所有裁剪的图片
    for root, dirs, files in os.walk(crops_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is not None:
                    resized = cv2.resize(img, crop_size)
                    crops.append(resized)
    
    if crops:
        # 创建拼接图
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
        
        if rows:
            final_mosaic = np.vstack(rows)
            
            dir_name = os.path.dirname(video_path) or '.'
            base_name = safe_artifact_basename(video_path)
            mosaic_path = os.path.join(dir_name, base_name + "_mosaic.jpg")
            cv2.imwrite(mosaic_path, final_mosaic)
            print(f"已保存拼接图片至: {mosaic_path}")

def detect_objects_with_frame_analysis(video_path, target_class,
                                       show_window=False, save_crops=False,
                                       save_training_data=False,
                                       all_objects=False,
                                       model_path='models/breast.pt'):
    """通过逐帧分析获取准确的时间戳"""
    
    # 首先使用detect.py进行检测
    detect_script = os.path.join(os.path.dirname(__file__), 'yolov5', 'detect.py')
    output_dir = os.path.join(os.path.dirname(video_path), "yolov5_output")
    
    # 获取当前Python解释器路径（虚拟环境中的Python）
    python_exe = sys.executable
    
    cmd = [
        python_exe, detect_script,  # 使用当前Python解释器
        '--weights', model_path,
        '--source', video_path,
        '--project', output_dir,
        '--name', 'exp',
        '--exist-ok',
        '--conf-thres', '0.5',
        '--save-txt',  # 临时保存标签用于解析时间戳，解析后自动清理
    ]
    
    if save_crops:
        cmd.append('--save-crop')
    
    print(f"运行YOLOv5 detect.py进行检测...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode != 0:
            print(f"detect.py运行失败: {result.stderr}")
            return []
        
        # 分析视频帧和检测结果
        pause_file = _get_pause_file_path()
        resume_enabled = _truthy_env('FINDINVIDEO_RESUME', default=True)
        ckpt = _load_checkpoint(video_path) if resume_enabled else None
        start_frame = int(ckpt.get('next_frame', 0)) if ckpt else 0
        detections = list(ckpt.get('detections', [])) if ckpt else []
        last_detected = float(ckpt.get('last_detected', detections[-1] if detections else -5.0)) if ckpt else -5

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
        
        results_dir = os.path.join(output_dir, 'exp')
        labels_dir = os.path.join(results_dir, 'labels')
        
        print(f"分析视频帧以获取准确时间戳...")
        pbar = tqdm(total=total_frames, initial=min(start_frame, total_frames), desc=f"分析视频: {os.path.basename(video_path)}")

        frame_count = start_frame
        
        while cap.isOpened():
            if _pause_requested(pause_file):
                _save_checkpoint(video_path, next_frame=frame_count, detections=detections, last_detected=last_detected)
                raise PauseRequested()
            success, frame, timed_out = _read_frame_with_timeout(cap)
            if not success:
                if timed_out:
                    _LOGGER.warning("读帧超时(>%ds)，尝试跳过: %s (frame=%d)", _READ_TIMEOUT_SEC, video_path, frame_count)
                    try:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count + 1)
                    except Exception:
                        break
                    pbar.update(1)
                    frame_count += 1
                    continue
                break  # 正常 EOF
            if frame is None:
                print(f"读取到空帧，已跳过: {video_path} (frame={frame_count})")
                pbar.update(1)
                frame_count += 1
                continue
            
            pbar.update(1)
            pos_msec = cap.get(cv2.CAP_PROP_POS_MSEC)
            if pos_msec and pos_msec > 0:
                current_time = float(pos_msec) / 1000.0
            else:
                current_time = frame_count / fps_safe
            
            # 检查对应的标签文件是否存在
            label_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_{frame_count:06d}.txt"
            label_path = os.path.join(labels_dir, label_filename)
            
            if os.path.exists(label_path):
                # 读取检测结果
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                if lines and current_time - last_detected >= 0.1:
                    detections.append(current_time)
                    last_detected = current_time
            
            frame_count += 1
        
        cap.release()
        pbar.close()

        if (not _pause_requested(pause_file)) and (frame_count >= total_frames or total_frames == 0):
            _clear_checkpoint(video_path)
        
        # 保存时间戳
        video_dir = os.path.dirname(video_path) or '.'
        artifact_base = safe_artifact_basename(video_path)
        txt_save_path = os.path.join(video_dir, artifact_base + ".txt")
        with open(txt_save_path, 'w') as f:
            f.write("检测到目标的时间位置（秒）:\n")
            for t in detections:
                f.write(f"{t:.2f}\n")
        print(f"已保存检测时间戳至: {txt_save_path}")
        
        # 清理临时 labels 目录（仅用于解析时间戳，不需要保留）
        if os.path.exists(labels_dir):
            try:
                shutil.rmtree(labels_dir)
            except Exception:
                pass

        # 处理拼接图片
        if save_crops:
            crops_dir = os.path.join(results_dir, 'crops')
            if os.path.exists(crops_dir):
                create_mosaic_from_crops(crops_dir, video_path)
        write_done_marker(video_path)
        return detections
        
    except Exception as e:
        print(f"检测失败: {e}")
        return []

def process_directory_videos(dir_path, target_item, all_objects_switch=False, model_path='models/yolov5s.pt', use_detectpy=False, skip_long_videos=True):
    """处理目录中的所有视频文件"""
    video_files = []

    try:
        for file in os.listdir(dir_path):
            if not is_video_file(file):
                continue
            file_path = os.path.join(dir_path, file)
            if not os.path.isfile(file_path):
                continue

            if has_existing_artifacts(file_path):
                print(f"检测到已有输出，跳过: {file_path}")
                continue

            claim_path = try_acquire_video_claim(file_path, md5_hint=None)
            if ensure_shared_state_dir() and not claim_path:
                print(f"已被其它机器处理，跳过: {file_path}")
                continue

            if is_path_already_yoloed(file_path):
                print(f"路径已存在于yoloed.txt，跳过: {file_path}")
                release_video_claim(claim_path)
                continue

            md5 = get_file_md5_cached(file_path)
            if md5:
                yoloed_md5 = load_yoloed_md5(reload=False)
                if md5 in yoloed_md5:
                    print(f"MD5已处理，跳过: {file_path}")
                    release_video_claim(claim_path)
                    continue
            else:
                print(f"无法计算MD5，跳过: {file_path}")
                release_video_claim(claim_path)
                continue

            cap = cv2.VideoCapture(file_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            duration = frame_count / fps if fps > 0 else float('inf')
            if skip_long_videos and duration > 3600:
                print(f"视频过长(>1h)，跳过: {file_path}")
                release_video_claim(claim_path)
                continue

            video_files.append((file_path, duration, claim_path, md5))
    except (PermissionError, FileNotFoundError) as e:
        print(f"警告: 无法访问目录 '{dir_path}': {e}")
        return

    for video_file, duration, claim_path, md5 in video_files:
        print(f"开始处理视频文件: {video_file}")
        try:
            if use_detectpy:
                detect_objects_with_frame_analysis(video_file, target_item,
                                                 show_window=False,
                                                 save_crops=True,
                                                 save_training_data=False,
                                                 all_objects=all_objects_switch,
                                                 model_path=model_path)
            else:
                detect_objects_in_video_yolov5(video_file, target_item,
                                              show_window=False,
                                              save_crops=True,
                                              save_training_data=False,
                                              all_objects=all_objects_switch,
                                              model_path=model_path)
        except PauseRequested:
            release_video_claim(claim_path)
            raise
        except Exception as exc:
            print(f"处理视频失败: {video_file}, 错误: {exc}")
            _LOGGER.error("Video failed: %s\n%s", video_file, traceback.format_exc())
        md5 = md5 or get_file_md5_cached(video_file)
        if md5:
            append_yoloed_md5(md5, file_path=video_file)
        write_done_marker(video_file)
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

if __name__ == "__main__":
    _setup_diagnostics()
    _install_pause_signal_handler()
    video_path = r"C:\Users\f1094\Downloads\果汁晗"  # 可设置为视频文件或目录
    # 如要检测所有模型内对象，则将 target_item 设置为任意值并启用全量检测开关
    target_item = "breast"  # 当 all_objects 为 True 时，该值不再限制检测
    all_objects_switch = True  # 设置为 True 表示显示所有检测对象
    
    # YOLOv5模型路径 - 使用相对路径
    yolov5_model_path = "models/breast.pt"  # 您可以根据需要修改模型路径
    
    # 新增选项：是否使用detect.py进行推理
    use_detectpy = True  # 设置为 True 使用detect.py，False 使用原有方法
    
    # 新增功能：按叶子节点视频数量排序处理
    use_leaf_node_processing = True  # 设置为 True 启用叶子节点处理模式
    skip_long_videos = False  # 设置为 True 时跳过超过一小时的视频
    try:
        if use_leaf_node_processing and os.path.isdir(video_path):
            print(f"启用叶子节点处理模式，正在扫描目录: {video_path}")
            print("正在查找包含视频文件的叶子节点目录...")

            # 查找所有叶子目录
            leaf_dirs = find_leaf_directories_with_videos(video_path, EXCLUDE_PATHS)

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
                    process_directory_videos(dir_path, target_item, all_objects_switch, yolov5_model_path, use_detectpy, skip_long_videos)

        # 原有的处理逻辑（当 use_leaf_node_processing 为 False 时使用）
        elif os.path.isdir(video_path):
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
            for root, dirs, files in os.walk(video_path):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in video_extensions:
                        file_path = os.path.join(root, file)
                        if should_process(file_path):
                            print(f"开始处理视频文件: {file_path}")
                            detect_objects_in_video_yolov5(file_path, target_item,
                                                          show_window=False,
                                                          save_crops=True,
                                                          save_training_data=False,
                                                          all_objects=all_objects_switch,
                                                          model_path=yolov5_model_path)
                            md5 = get_file_md5_cached(file_path)
                            if md5:
                                append_yoloed_md5(md5, file_path=file_path)
                        else:
                            print(f"已存在拼接图片，跳过处理: {file_path}")
        else:
            # 处理单个视频文件
            if should_process(video_path):
                detect_objects_in_video_yolov5(video_path, target_item,
                                              show_window=False,
                                              save_crops=True,
                                              save_training_data=False,
                                              all_objects=all_objects_switch,
                                              model_path=yolov5_model_path)
                md5 = get_file_md5_cached(video_path)
                if md5:
                    append_yoloed_md5(md5, file_path=video_path)
            else:
                print(f"已存在拼接图片，跳过处理: {video_path}")
    except PauseRequested:
        print("已暂停：已保存进度（checkpoint）。删除 pause.flag 或再次运行即可续跑。")
        sys.exit(0)
    except Exception as exc:
        _LOGGER.error("Fatal error:\n%s", traceback.format_exc())
        raise