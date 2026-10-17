"""记录 FindInVideo 运行、视频失败和进程退出诊断。"""

import atexit
import ctypes
import faulthandler
import json
import math
import os
import socket
import sys
import threading
import traceback
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone


_OMIT_KEYS = {'argv', 'env', 'locals'}
_RUN_ID = uuid.uuid4().hex
_ENTRYPOINT = None
_PID = os.getpid()
try:
    _HOST = socket.gethostname()
except OSError:
    _HOST = 'unknown'

_INSTALLED = False
_EXIT_WRITTEN = False
_EXIT_REASON = None
_VIDEO_FAILURE_COUNT = 0
_LAST_FAILURE = None
_CURRENT_VIDEO = None
_CURRENT_MD5 = None
_CURRENT_PIPELINE_ID = None
_HANDLED_EXCEPTION_IDS = set()
_IO_ERROR_REPORTED = False
_LOG_LOCK = threading.RLock()
_ORIGINAL_SYS_EXCEPTHOOK = None
_ORIGINAL_THREAD_EXCEPTHOOK = None
_CRASH_STREAM = None
_CONSOLE_HANDLER = None


def _now():
    """返回 UTC ISO 时间。"""
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def _safe_value(value, depth=0):
    """把诊断字段转换成不包含对象局部状态的 JSON 值。"""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if depth >= 8:
        return f'<{type(value).__name__}>'
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, BaseException):
        return {
            'exception_type': type(value).__name__,
            'message': str(value),
        }
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            name = str(key)
            if name.casefold() not in _OMIT_KEYS:
                result[name] = _safe_value(item, depth + 1)
        return result
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe_value(item, depth + 1) for item in value]
    return f'<{type(value).__name__}>'


def _event(event_type, fields):
    """构造带当前运行标识的事件。"""
    event = {
        'event_type': str(event_type),
        'run_id': _RUN_ID,
        'entrypoint': _safe_value(_ENTRYPOINT),
        'pid': _PID,
        'host': _HOST,
        'time': _now(),
    }
    for key, value in fields.items():
        name = str(key)
        if name.casefold() not in _OMIT_KEYS and name not in {
            'event_type', 'run_id', 'entrypoint', 'pid', 'host', 'time'
        }:
            event[name] = _safe_value(value)
    return event


def _runtime_log_path():
    """返回本机 JSONL 诊断日志路径。"""
    return os.environ.get('FINDINVIDEO_RUNTIME_LOG_PATH') or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'logs', 'runtime_events.jsonl'
    )


def _crash_log_path():
    """返回 faulthandler 日志路径。"""
    return os.environ.get('FINDINVIDEO_CRASH_LOG_PATH') or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'logs', 'crash.log'
    )


def _report_io_error(message):
    """本次运行最多向 stderr 报告一次诊断写入错误。"""
    global _IO_ERROR_REPORTED
    if _IO_ERROR_REPORTED:
        return
    _IO_ERROR_REPORTED = True
    try:
        print(f'运行诊断写入失败：{message}', file=sys.stderr, flush=True)
    except Exception:
        pass


def _append_local(event):
    """将 JSON 事件追加到本机文件并同步到磁盘。"""
    path = _runtime_log_path()
    try:
        with _LOG_LOCK:
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
            with open(path, 'a', encoding='utf-8', newline='') as stream:
                stream.write(json.dumps(event, ensure_ascii=False, separators=(',', ':')) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        return True
    except Exception as error:
        _report_io_error(f'{type(error).__name__}: {error}')
        return False


def _mirror_event(event):
    """仅调用已经载入的 utils 审计函数，不主动导入数据库模块。"""
    module = sys.modules.get('utils')
    writer = getattr(module, 'record_processing_event', None)
    if not callable(writer):
        return
    reserved = {
        'event_type', 'pid', 'host', 'time',
        'video_path', 'file_md5', 'pipeline_id',
    }
    details = {key: value for key, value in event.items() if key not in reserved}
    details['runtime_time'] = event['time']
    try:
        writer(
            event['event_type'],
            video_path=event.get('video_path'),
            file_md5=event.get('file_md5'),
            pipeline_id=event.get('pipeline_id'),
            **details,
        )
    except Exception:
        # 共享数据库只是本机日志之后的尽力镜像。
        pass


def _emit(event_type, mirror=True, **fields):
    """先写本机 JSONL，再尽力镜像到已载入的共享审计函数。"""
    event = _event(event_type, fields)
    _append_local(event)
    if mirror and event_type != 'video_started':
        _mirror_event(event)
    return event


def _write_crash_line(line):
    """将关联信息同步写入 faulthandler 文件。"""
    if _CRASH_STREAM is None:
        return
    try:
        _CRASH_STREAM.write(line.rstrip('\n') + '\n')
        _CRASH_STREAM.flush()
        os.fsync(_CRASH_STREAM.fileno())
    except Exception as error:
        _report_io_error(f'crash.log {type(error).__name__}: {error}')


def _ensure_crash_log(enable_fault_handler):
    """打开并保留 crash.log，必要时启用所有线程的 faulthandler。"""
    global _CRASH_STREAM
    if _CRASH_STREAM is None:
        path = _crash_log_path()
        try:
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)
            _CRASH_STREAM = open(path, 'a', encoding='utf-8', buffering=1)
            _write_crash_line(
                f'运行开始 run_id={_RUN_ID} pid={_PID} host={_HOST} '
                f'entrypoint={_ENTRYPOINT or "<unknown>"} time={_now()}'
            )
        except Exception as error:
            _CRASH_STREAM = None
            _report_io_error(f'crash.log {type(error).__name__}: {error}')
    if enable_fault_handler and _CRASH_STREAM is not None:
        try:
            faulthandler.enable(file=_CRASH_STREAM, all_threads=True)
        except Exception as error:
            _report_io_error(f'faulthandler {type(error).__name__}: {error}')


def _format_exception(exc_type, exc_value, exc_traceback):
    """格式化异常及其显式或隐式异常链。"""
    if exc_type is None or exc_value is None:
        return ''
    try:
        return ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
    except Exception:
        return f'{exc_type.__name__}: {exc_value}'


def _exception_fields(error, exc_traceback=None):
    """提取异常类型、消息、系统错误码和完整堆栈。"""
    if not isinstance(error, BaseException):
        return {
            'exception_type': None,
            'message': None if error is None else str(error),
            'errno': None,
            'winerror': None,
            'traceback': '',
        }
    tb = exc_traceback if exc_traceback is not None else error.__traceback__
    return {
        'exception_type': type(error).__name__,
        'message': str(error),
        'errno': getattr(error, 'errno', None),
        'winerror': getattr(error, 'winerror', None),
        'traceback': _format_exception(type(error), error, tb),
    }


def _exit_code_for_system_exit(error):
    """映射 SystemExit 到解释器使用的整数退出码。"""
    code = error.code
    if code is None:
        return 0
    if isinstance(code, int):
        return code & (0xffffffff if os.name == 'nt' else 0xff)
    return 1


def _unhandled_exit_code(error):
    """保持解释器退出行为，并记录当前平台实际报告的退出码。"""
    if isinstance(error, KeyboardInterrupt):
        return 0xc000013a if os.name == 'nt' else 130
    return 1


def _record_process_exit(reason, exit_code=None, error=None, exc_traceback=None, system_exit_code=None):
    """每次运行只写一条进程退出摘要。"""
    global _EXIT_WRITTEN
    if _EXIT_WRITTEN:
        return
    _EXIT_WRITTEN = True
    fields = {
        'reason': reason,
        'exit_code': exit_code,
        'video_failure_count': _VIDEO_FAILURE_COUNT,
        'last_failure': _LAST_FAILURE,
        'video_path': _CURRENT_VIDEO,
        'file_md5': _CURRENT_MD5,
        'pipeline_id': _CURRENT_PIPELINE_ID,
    }
    if error is not None:
        fields.update(_exception_fields(error, exc_traceback))
        _HANDLED_EXCEPTION_IDS.add(id(error))
    if system_exit_code is not None:
        fields['system_exit_code'] = system_exit_code
    _emit('process_exit', **fields)


def _atexit_fallback():
    """补记没有经过外层诊断包装器的退出，不推断其为正常完成。"""
    if not _EXIT_WRITTEN:
        _record_process_exit(_EXIT_REASON or 'unknown_exit')


def _sys_excepthook(exc_type, exc_value, exc_traceback):
    """记录未捕获异常，然后继续调用安装前的系统 hook。"""
    fields = _exception_fields(exc_value, exc_traceback)
    _emit('uncaught_exception', **fields)
    if id(exc_value) not in _HANDLED_EXCEPTION_IDS:
        _record_process_exit(
            _EXIT_REASON or 'uncaught_exception',
            exit_code=_unhandled_exit_code(exc_value),
            error=exc_value,
            exc_traceback=exc_traceback,
        )
    if _ORIGINAL_SYS_EXCEPTHOOK is not None:
        try:
            _ORIGINAL_SYS_EXCEPTHOOK(exc_type, exc_value, exc_traceback)
        except Exception:
            pass


def _thread_excepthook(args):
    """记录线程异常但不将其视作进程退出。"""
    fields = _exception_fields(args.exc_value, args.exc_traceback)
    fields['thread_name'] = getattr(args.thread, 'name', None)
    _emit('thread_exception', **fields)
    if _ORIGINAL_THREAD_EXCEPTHOOK is not None:
        try:
            _ORIGINAL_THREAD_EXCEPTHOOK(args)
        except Exception:
            pass


def _console_control_handler(control_type):
    """记录 Windows 关闭类控制通知，并交还 Windows 默认处理。"""
    reasons = {
        1: 'CTRL_BREAK',
        2: 'CTRL_CLOSE',
        5: 'LOGOFF',
        6: 'SHUTDOWN',
    }
    reason = reasons.get(int(control_type))
    if reason is not None:
        _emit('process_termination_requested', mirror=False, reason=reason)
    return False


def _install_console_handler():
    """在 Windows 注册返回 False 的控制台处理器。"""
    global _CONSOLE_HANDLER
    if os.name != 'nt' or _CONSOLE_HANDLER is not None:
        return
    try:
        # Windows BOOL 是 32 位，不能使用 8 位 c_bool 作为回调返回值。
        callback_type = ctypes.WINFUNCTYPE(ctypes.c_int, ctypes.c_uint)
        _CONSOLE_HANDLER = callback_type(_console_control_handler)
        register = ctypes.windll.kernel32.SetConsoleCtrlHandler
        register.argtypes = [callback_type, ctypes.c_int]
        register.restype = ctypes.c_int
        if not register(_CONSOLE_HANDLER, True):
            _CONSOLE_HANDLER = None
    except Exception as error:
        _CONSOLE_HANDLER = None
        _report_io_error(f'Windows 控制台 handler {type(error).__name__}: {error}')


def install_runtime_diagnostics(entrypoint):
    """幂等安装本机运行日志、异常 hook 和 faulthandler。"""
    global _INSTALLED, _ENTRYPOINT, _ORIGINAL_SYS_EXCEPTHOOK, _ORIGINAL_THREAD_EXCEPTHOOK
    if _INSTALLED:
        return
    _INSTALLED = True
    _ENTRYPOINT = os.fspath(entrypoint) if entrypoint is not None else None
    _ORIGINAL_SYS_EXCEPTHOOK = sys.excepthook
    sys.excepthook = _sys_excepthook
    if hasattr(threading, 'excepthook'):
        _ORIGINAL_THREAD_EXCEPTHOOK = threading.excepthook
        threading.excepthook = _thread_excepthook
    atexit.register(_atexit_fallback)
    _ensure_crash_log(enable_fault_handler=True)
    _emit('process_started')
    _install_console_handler()


def record_video_started(video_path, file_md5=None, pipeline_id=None):
    """记录当前视频，并将路径同步到 crash.log 以便关联原生崩溃。"""
    global _CURRENT_VIDEO, _CURRENT_MD5, _CURRENT_PIPELINE_ID
    _CURRENT_VIDEO = os.fspath(video_path) if video_path is not None else None
    _CURRENT_MD5 = file_md5
    _CURRENT_PIPELINE_ID = pipeline_id
    _emit(
        'video_started',
        mirror=False,
        video_path=_CURRENT_VIDEO,
        file_md5=file_md5,
        pipeline_id=pipeline_id,
    )
    _ensure_crash_log(enable_fault_handler=False)
    _write_crash_line(
        f'视频开始 run_id={_RUN_ID} video_path={_CURRENT_VIDEO} '
        f'file_md5={file_md5} pipeline_id={pipeline_id} time={_now()}'
    )


def record_video_failure(
    video_path,
    error=None,
    message=None,
    file_md5=None,
    pipeline_id=None,
    stage='video_processing',
    **details,
):
    """持久化单视频失败原因及真实异常链，不改变处理流程。"""
    global _VIDEO_FAILURE_COUNT, _LAST_FAILURE, _CURRENT_VIDEO, _CURRENT_MD5, _CURRENT_PIPELINE_ID
    if error is None:
        error = sys.exc_info()[1]
    exception = _exception_fields(error)
    if message is not None:
        exception['message'] = str(message)
    _VIDEO_FAILURE_COUNT += 1
    failure_id = uuid.uuid4().hex
    video_path = os.fspath(video_path) if video_path is not None else None
    _CURRENT_VIDEO = video_path
    _CURRENT_MD5 = file_md5
    _CURRENT_PIPELINE_ID = pipeline_id
    _LAST_FAILURE = {
        'failure_id': failure_id,
        'video_path': video_path,
        'stage': stage,
        'exception_type': exception['exception_type'],
        'message': exception['message'],
    }
    fields = dict(details)
    fields.update({
        'failure_id': failure_id,
        'video_path': video_path,
        'file_md5': file_md5,
        'pipeline_id': pipeline_id,
        'stage': stage,
        **exception,
    })
    _emit('video_failed', **fields)
    return failure_id


def set_exit_reason(reason):
    """保存调用方明确知道的暂停或其他主动退出原因。"""
    global _EXIT_REASON
    _EXIT_REASON = str(reason)


def record_process_event(event_type, **details):
    """记录调用方明确提供的进程事件，例如 process_restart_requested。"""
    return _emit(event_type, **details)


@contextmanager
def process_diagnostics(entrypoint):
    """用同一进程级退出记录包装原入口代码，保留其全局变量语义。"""
    install_runtime_diagnostics(entrypoint)
    try:
        yield
    except SystemExit as error:
        _record_process_exit(
            _EXIT_REASON or 'system_exit',
            exit_code=_exit_code_for_system_exit(error),
            error=error,
            system_exit_code=error.code,
        )
        raise
    except KeyboardInterrupt as error:
        _record_process_exit(
            _EXIT_REASON or 'keyboard_interrupt',
            exit_code=_unhandled_exit_code(error),
            error=error,
        )
        raise
    except BaseException as error:
        pause_type = getattr(sys.modules.get('utils'), 'PauseRequested', None)
        reason = 'pause_requested' if pause_type is not None and isinstance(error, pause_type) else 'uncaught_exception'
        _record_process_exit(
            _EXIT_REASON or reason,
            exit_code=_unhandled_exit_code(error),
            error=error,
        )
        raise
    else:
        reason = _EXIT_REASON or (
            'queue_exhausted_with_video_failures'
            if _VIDEO_FAILURE_COUNT
            else 'queue_exhausted'
        )
        _record_process_exit(reason, exit_code=0)


def run_with_diagnostics(operation, entrypoint):
    """在诊断上下文中运行 callable，并原样返回其结果。"""
    with process_diagnostics(entrypoint):
        return operation()
