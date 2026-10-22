"""通过 Windows WMI broker 在独立控制台中启动并监控 main.py。"""

import ctypes
from ctypes import wintypes
import datetime
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time
import uuid


HANDSHAKE_TIMEOUT_SECONDS = 30
RESTART_EXIT_CODE = 75
LAUNCH_TOKEN_ENV = 'FINDINVIDEO_LAUNCH_TOKEN'
CREATE_FLAGS = 0x01000000 | 0x10 | 0x400
BROKER_SCRIPT = r'''
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$request = [Console]::In.ReadToEnd() | ConvertFrom-Json
$startup = New-CimInstance -ClassName Win32_ProcessStartup -ClientOnly -Property @{
    CreateFlags = [uint32](0x01000000 -bor 0x10 -bor 0x400)
    ShowWindow = [uint16]1
    EnvironmentVariables = [string[]]$request.environment
}
$result = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
    CommandLine = [string]$request.command
    CurrentDirectory = [string]$request.root
    ProcessStartupInformation = $startup
}
[pscustomobject]@{ ReturnValue = $result.ReturnValue; ProcessId = $result.ProcessId } |
    ConvertTo-Json -Compress
'''


class LauncherError(RuntimeError):
    """独立启动校验失败。"""


def _is_windows():
    return os.name == 'nt'


def _validate_token(token):
    if not isinstance(token, str) or len(token) != 32 or any(char not in '0123456789abcdef' for char in token):
        raise LauncherError('启动握手令牌格式错误')


def _paths(root, token):
    _validate_token(token)
    logs = Path(root) / 'logs'
    return {
        'result': logs / f'launcher_startup_{token}.json',
        'main_ready': logs / f'launcher_handshake_{token}_main.json',
        'main_ack': logs / f'launcher_handshake_{token}_main_ack.json',
        'restart': logs / f'launcher_restart_{token}.json',
    }


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(f'{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp')
    try:
        with temporary.open('w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        # Windows 读者短暂持有文件时可能拒绝替换；保留原文件并有界重试。
        for attempt in range(40):
            try:
                os.replace(temporary, path)
                break
            except PermissionError:
                if attempt == 39:
                    raise
                time.sleep(0.025)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _read_json(path):
    with Path(path).open('r', encoding='utf-8') as stream:
        return json.load(stream)


def _try_read_json(path):
    try:
        return _read_json(path)
    except (FileNotFoundError, PermissionError, json.JSONDecodeError):
        return None


def _wait_document(path, token, deadline, states=None):
    while time.monotonic() < deadline:
        document = _try_read_json(path)
        if document is not None:
            if document.get('token') != token:
                raise LauncherError('启动握手令牌不匹配')
            state = document.get('state')
            if state in ('failed', 'rejected'):
                raise LauncherError(document.get('error', '独立启动握手被拒绝'))
            if states is None or state in states:
                return document
        time.sleep(0.05)
    raise LauncherError('独立启动握手超时')


def _kernel32():
    if os.name != 'nt':
        raise LauncherError('独立启动只支持 Windows')
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenProcess.restype = wintypes.HANDLE
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD),
    ]
    api.GetProcessTimes.argtypes = [
        wintypes.HANDLE, ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
        ctypes.POINTER(wintypes.FILETIME), ctypes.POINTER(wintypes.FILETIME),
    ]
    api.ProcessIdToSessionId.argtypes = [wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
    api.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
    return api


def _creation_time(filetime):
    ticks = (filetime.dwHighDateTime << 32) | filetime.dwLowDateTime
    return ticks / 10000000 - 11644473600


def _console_visible():
    api = _kernel32()
    api.GetConsoleWindow.restype = wintypes.HWND
    hwnd = api.GetConsoleWindow()
    user32 = ctypes.WinDLL('user32', use_last_error=True)
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    return bool(hwnd and user32.IsWindowVisible(hwnd))


def _inspect_process(pid):
    """读取指定进程的实际映像、创建时间、会话和 Job 状态。"""
    api = _kernel32()
    handle = api.OpenProcess(0x1000, False, int(pid))
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        size = wintypes.DWORD(32768)
        image = ctypes.create_unicode_buffer(size.value)
        if not api.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
        if not api.GetProcessTimes(
            handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(kernel), ctypes.byref(user),
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        in_job = wintypes.BOOL()
        if not api.IsProcessInJob(handle, None, ctypes.byref(in_job)):
            raise ctypes.WinError(ctypes.get_last_error())
        session = wintypes.DWORD()
        if not api.ProcessIdToSessionId(int(pid), ctypes.byref(session)):
            raise ctypes.WinError(ctypes.get_last_error())
        return {
            'pid': int(pid),
            'image': image.value,
            'created_at': _creation_time(created),
            'session': session.value,
            'in_job': bool(in_job.value),
        }
    finally:
        api.CloseHandle(handle)


def is_in_job(pid=None):
    """查询当前进程或指定 PID 是否仍属于 Windows Job。"""
    return bool(_inspect_process(os.getpid() if pid is None else int(pid))['in_job'])


def _current_identity():
    identity = _inspect_process(os.getpid())
    identity['console_visible'] = _console_visible()
    return identity


def _base_python():
    executable = Path(getattr(sys, '_base_executable', sys.executable))
    if executable.name.lower() == 'pythonw.exe':
        executable = executable.with_name('python.exe')
    executable = executable.resolve()
    if not executable.is_file():
        raise LauncherError('找不到基础 Python 解释器')
    return executable


def _child_environment(root, venv_python=None):
    environment = os.environ.copy()
    candidate = Path(root) / '.venv' / 'Scripts' / 'python.exe'
    if venv_python is not None:
        candidate = Path(venv_python)
    environment['__PYVENV_LAUNCHER__'] = str(candidate if candidate.is_file() else Path(sys.executable).resolve())
    environment['PYTHONUTF8'] = '1'
    return environment


def _invoke_broker(command, root, environment, timeout):
    system_root = os.environ.get('SystemRoot', r'C:\Windows')
    powershell = Path(system_root) / 'System32' / 'WindowsPowerShell' / 'v1.0' / 'powershell.exe'
    if not powershell.is_file():
        raise LauncherError('找不到 Windows PowerShell CIM broker')
    payload = {
        'command': command,
        'root': str(root),
        'environment': [f'{key}={value}' for key, value in environment.items()],
    }
    try:
        completed = subprocess.run(
            [str(powershell), '-NoProfile', '-NonInteractive', '-WindowStyle', 'Hidden', '-Command', BROKER_SCRIPT],
            input=json.dumps(payload), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding='utf-8', timeout=timeout,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0x08000000),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LauncherError(f'Windows CIM broker 启动失败：{exc}') from exc
    if completed.returncode != 0:
        raise LauncherError(f'Windows CIM broker 返回错误码 {completed.returncode}')
    try:
        result = json.loads(completed.stdout.lstrip('\ufeff').strip())
    except (TypeError, json.JSONDecodeError) as exc:
        raise LauncherError('Windows CIM broker 返回了无效结果') from exc
    if result.get('ReturnValue') != 0 or not result.get('ProcessId'):
        raise LauncherError(f"Windows CIM broker 创建失败，返回码 {result.get('ReturnValue')}")
    return int(result['ProcessId'])


def _same_image(actual, expected):
    return os.path.normcase(os.path.abspath(actual)) == os.path.normcase(os.path.abspath(expected))


def _validate_identity(record, actual, token, pid, expected_image, expected_session, started_at):
    """校验握手记录与操作系统查询到的进程身份一致。"""
    _validate_token(token)
    if not isinstance(record, dict) or not isinstance(actual, dict):
        raise LauncherError('启动握手身份记录缺失')
    if record.get('token') != token or int(record.get('pid', -1)) != int(pid):
        raise LauncherError('启动握手 PID 或 token 不匹配')
    if int(actual.get('pid', -1)) != int(pid):
        raise LauncherError('操作系统返回的进程 PID 不匹配')
    if not _same_image(record.get('image', ''), expected_image) or not _same_image(actual.get('image', ''), expected_image):
        raise LauncherError('启动进程映像路径不匹配')
    created_at = float(actual.get('created_at', 0))
    if created_at < float(started_at) - 1.0 or created_at > time.time() + 1.0:
        raise LauncherError('启动进程创建时间不匹配')
    if abs(float(record.get('created_at', 0)) - created_at) > 1.0:
        raise LauncherError('启动握手创建时间与系统记录不匹配')
    if int(record.get('session', -1)) != int(expected_session) or int(actual.get('session', -1)) != int(expected_session):
        raise LauncherError('启动进程 Windows 会话不匹配')
    if record.get('in_job') is not False or actual.get('in_job') is not False:
        raise LauncherError('启动进程仍处于 Windows Job')
    if record.get('console_visible') is not True:
        raise LauncherError('启动进程没有可见控制台')


def _write_startup_event(root, event, **fields):
    timestamp = datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S%z')
    details = ' '.join(f'{key}={value}' for key, value in fields.items())
    line = f'[{timestamp}] event={event} entrypoint=main.py {details}\n'
    path = Path(root) / 'logs' / 'launcher_events.log'
    with path.open('a', encoding='utf-8', newline='') as stream:
        stream.write(line)
        stream.flush()
        os.fsync(stream.fileno())


def _worker_deadline(document):
    return time.monotonic() + max(0.0, float(document['deadline']) - time.time())


def _wait_main_ready(path, process, token, deadline):
    while time.monotonic() < deadline:
        document = _try_read_json(path)
        if document is not None:
            if document.get('token') != token:
                raise LauncherError('main worker 握手令牌不匹配')
            if document.get('state') == 'ready':
                return document
        exit_code = process.poll()
        if exit_code is not None:
            raise LauncherError(f'main worker 在握手前退出，退出码 {exit_code}')
        time.sleep(0.05)
    raise LauncherError('main worker 握手超时')


def _worker_fail(path, token, error):
    try:
        current = _try_read_json(path) or {}
        if current.get('token') == token and current.get('state') not in ('running', 'complete'):
            current.update({'state': 'failed', 'error': str(error)})
            _write_json(path, current)
    except OSError:
        pass


def _main_worker(root, token):
    """在检查通过前等待，不执行 main.py。"""
    paths = _paths(root, token)
    startup = paths['result']
    deadline = time.monotonic() + HANDSHAKE_TIMEOUT_SECONDS
    document = _wait_document(startup, token, deadline, {'main_starting'})
    deadline = _worker_deadline(document)
    main_started_at = float(document['main_started_at'])
    identity = _current_identity()
    _validate_identity(
        {'token': token, **identity}, identity, token, os.getpid(), str(_base_python()),
        int(document['session']), main_started_at,
    )
    _write_json(paths['main_ready'], {'token': token, 'state': 'ready', **identity})
    document = _wait_document(startup, token, deadline, {'running'})
    if document.get('main', {}).get('pid') != os.getpid() or document.get('main_pid') != os.getpid():
        raise LauncherError('main.py 启动 PID 与握手 PID 不一致')
    main_path = Path(root) / 'main.py'
    os.chdir(root)
    if sys.path:
        sys.path[0] = str(root)
    else:
        sys.path.insert(0, str(root))
    sys.argv = [str(main_path)]
    os.environ[LAUNCH_TOKEN_ENV] = token
    return main_path


def _supervisor_worker(root, token):
    """验证 supervisor 与 main 子进程，再等待并记录真实退出码。"""
    # Ctrl+C 由实际处理进程保存断点，监护进程必须继续等待真实退出码。
    signal.signal(signal.SIGINT, lambda *_: None)
    paths = _paths(root, token)
    startup = paths['result']
    process = None
    released_main = False
    try:
        deadline = time.monotonic() + HANDSHAKE_TIMEOUT_SECONDS
        document = _wait_document(startup, token, deadline, {'pending'})
        deadline = _worker_deadline(document)
        while not document.get('broker_pid'):
            time.sleep(0.05)
            document = _wait_document(startup, token, deadline, {'pending'})
        if int(document['broker_pid']) != os.getpid():
            raise LauncherError('WMI 返回 PID 与 supervisor PID 不匹配')
        started_at = float(document['started_at'])
        session = int(document['session'])
        identity = _current_identity()
        _validate_identity(
            {'token': token, **identity}, identity, token, os.getpid(), str(_base_python()), session, started_at,
        )
        document.update({'state': 'supervisor_ready', 'supervisor': {'token': token, **identity}})
        _write_json(startup, document)

        document = _wait_document(startup, token, deadline, {'supervisor_accepted'})
        if int(document.get('supervisor_pid', -1)) != os.getpid():
            raise LauncherError('supervisor 接受回执 PID 不匹配')

        main_started_at = time.time()
        document.update({'state': 'main_starting', 'main_started_at': main_started_at})
        _write_json(startup, document)
        executable = _base_python()
        script = Path(__file__).resolve()
        process = subprocess.Popen(
            [str(executable), '-X', 'utf8', str(script), '--main-worker', token],
            cwd=str(root), env=_child_environment(root),
        )
        main_ready = _wait_main_ready(paths['main_ready'], process, token, deadline)
        actual_main = _inspect_process(process.pid)
        _validate_identity(
            main_ready, actual_main, token, process.pid, str(executable), session, main_started_at,
        )
        document = _wait_document(startup, token, deadline, {'main_starting'})
        document.update({'state': 'awaiting_acceptance', 'main': main_ready, 'main_pid': process.pid})
        _write_json(startup, document)

        document = _wait_document(startup, token, deadline, {'accepted'})
        if int(document.get('supervisor_pid', -1)) != os.getpid():
            raise LauncherError('最终启动回执 supervisor PID 不匹配')
        if int(document.get('main', {}).get('pid', -1)) != process.pid:
            raise LauncherError('最终启动回执 main PID 不匹配')
        document.update({'state': 'running', 'accepted_at': time.time()})
        _write_json(startup, document)
        _write_startup_event(root, 'launcher_started', token=token, pid=process.pid, supervisor_pid=os.getpid())
        released_main = True
        while True:
            exit_code = process.wait()
            request = _try_read_json(paths['restart'])
            if not (
                exit_code == RESTART_EXIT_CODE and request
                and request.get('token') == token and request.get('main_pid') == process.pid
            ):
                break
            # 旧解释器已完全退出，模型、显存和视频资源随进程释放。
            paths['restart'].unlink()
            paths['main_ready'].unlink(missing_ok=True)
            previous_pid = process.pid
            main_started_at = time.time()
            deadline = time.monotonic() + HANDSHAKE_TIMEOUT_SECONDS
            document.update({
                'state': 'main_starting', 'main_started_at': main_started_at,
                'deadline': main_started_at + HANDSHAKE_TIMEOUT_SECONDS,
                'restart_count': int(document.get('restart_count', 0)) + 1,
            })
            released_main = False
            _write_json(startup, document)
            process = subprocess.Popen(
                [str(executable), '-X', 'utf8', str(script), '--main-worker', token],
                cwd=str(root), env=_child_environment(root),
            )
            main_ready = _wait_main_ready(paths['main_ready'], process, token, deadline)
            _validate_identity(
                main_ready, _inspect_process(process.pid), token, process.pid,
                str(executable), session, main_started_at,
            )
            document.update({'state': 'running', 'main': main_ready, 'main_pid': process.pid})
            _write_json(startup, document)
            released_main = True
            _write_startup_event(root, 'launcher_restarted', token=token,
                                 pid=process.pid, previous_pid=previous_pid)
        document = _try_read_json(startup) or document
        document.update({'state': 'complete', 'exit_code': int(exit_code), 'finished_at': time.time()})
        _write_startup_event(root, 'launcher_exit', token=token, exit_code=int(exit_code), pid=process.pid)
        _write_json(startup, document)
        return int(exit_code)
    except Exception as exc:
        _worker_fail(startup, token, exc)
        if process is not None and not released_main:
            try:
                document = _try_read_json(startup) or {}
                if document.get('token') == token:
                    document.update({'state': 'rejected', 'error': str(exc)})
                    _write_json(startup, document)
            except OSError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # 子 worker 只会等待握手；身份已由本次 token 及其 Popen 句柄绑定。
                process.terminate()
                process.wait(timeout=5)
        raise
    finally:
        for key in ('main_ready', 'main_ack', 'restart'):
            try:
                paths[key].unlink()
            except FileNotFoundError:
                pass


def restart_main(entrypoint, args=()):
    """通知独立监护进程重启同一入口，直接运行时保留原 execv 行为。"""
    path = Path(entrypoint).resolve()
    token = os.environ.get(LAUNCH_TOKEN_ENV)
    if not token:
        os.execv(sys.executable, [sys.executable, str(path), *args])
        return
    paths = _paths(path.parent, token)
    document = _read_json(paths['result'])
    if (document.get('token') != token or document.get('state') != 'running'
            or document.get('main_pid') != os.getpid()):
        raise LauncherError('恢复重启请求与当前受监控进程不匹配')
    _write_json(paths['restart'], {'token': token, 'main_pid': os.getpid()})
    raise SystemExit(RESTART_EXIT_CODE)


def launch_main(root=None, timeout=HANDSHAKE_TIMEOUT_SECONDS):
    """启动独立 supervisor，验收后返回 PID 与持久化启动结果路径。"""
    if not _is_windows():
        raise LauncherError('独立启动只支持 Windows')
    if timeout <= 0:
        raise ValueError('timeout 必须大于 0')
    root = Path(root or Path(__file__).resolve().parent).resolve()
    script = root / 'independent_launcher.py'
    main_script = root / 'main.py'
    if not script.is_file() or not main_script.is_file():
        raise LauncherError('启动目录必须包含 independent_launcher.py 和 main.py')
    token = uuid.uuid4().hex
    paths = _paths(root, token)
    paths['result'].parent.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    session = wintypes.DWORD()
    if not _kernel32().ProcessIdToSessionId(os.getpid(), ctypes.byref(session)):
        raise ctypes.WinError(ctypes.get_last_error())
    document = {
        'token': token,
        'state': 'pending',
        'started_at': started_at,
        'deadline': started_at + timeout,
        'session': session.value,
    }
    _write_json(paths['result'], document)
    executable = _base_python()
    environment = _child_environment(root)
    command = subprocess.list2cmdline([str(executable), '-X', 'utf8', str(script), '--worker', token])
    deadline = time.monotonic() + timeout
    worker_pid = None
    accepted = False
    try:
        worker_pid = _invoke_broker(command, root, environment, min(timeout, 20))
        document.update({'broker_pid': worker_pid})
        _write_json(paths['result'], document)
        ready = _wait_document(paths['result'], token, deadline, {'supervisor_ready'})
        supervisor = ready.get('supervisor', {})
        actual_supervisor = _inspect_process(worker_pid)
        _validate_identity(
            supervisor, actual_supervisor, token, worker_pid, str(executable), session.value, started_at,
        )
        document = dict(ready, state='supervisor_accepted', supervisor_pid=worker_pid)
        _write_json(paths['result'], document)

        report = _wait_document(paths['result'], token, deadline, {'awaiting_acceptance'})
        main = report.get('main', {})
        actual_main = _inspect_process(int(main.get('pid', -1)))
        _validate_identity(
            main, actual_main, token, int(main.get('pid', -1)), str(executable), session.value,
            float(report.get('main_started_at', started_at)),
        )
        if int(report.get('supervisor', {}).get('pid', -1)) != worker_pid:
            raise LauncherError('启动结果中的 supervisor PID 不匹配')
        _validate_identity(
            report['supervisor'], actual_supervisor, token, worker_pid, str(executable), session.value, started_at,
        )
        result = dict(
            report, state='accepted', accepted=True, supervisor_pid=worker_pid,
            main_pid=int(main['pid']), startup_result=str(paths['result']),
        )
        _write_json(paths['result'], result)
        released = _wait_document(paths['result'], token, deadline, {'running', 'complete'})
        if int(released.get('supervisor_pid', -1)) != worker_pid or int(released.get('main_pid', -1)) != int(main['pid']):
            raise LauncherError('最终运行状态 PID 与验收结果不一致')
        accepted = True
        return result
    except Exception as exc:
        if not accepted:
            current = _try_read_json(paths['result']) or document
            if current.get('token') == token and current.get('state') not in ('running', 'complete'):
                current.update({'state': 'rejected', 'error': str(exc)})
                try:
                    _write_json(paths['result'], current)
                except OSError:
                    pass
        if isinstance(exc, LauncherError):
            raise
        raise LauncherError(f'独立启动验收失败：{exc}') from exc


def main():
    """处理公开启动和两个受 token 约束的内部 worker 入口。"""
    root = Path(__file__).resolve().parent
    if len(sys.argv) == 1:
        try:
            result = launch_main(root=root)
        except Exception as exc:
            print(f'独立启动失败：{exc}', file=sys.stderr)
            return 1
        print(
            f"独立启动验收成功：supervisor PID={result['supervisor_pid']}，"
            f"main PID={result['main']['pid']}，启动结果={result['startup_result']}"
        )
        return 0
    if len(sys.argv) == 3 and sys.argv[1] in ('--worker', '--main-worker'):
        token = sys.argv[2]
        if sys.argv[1] == '--main-worker':
            try:
                _validate_token(token)
                main_path = _main_worker(root, token)
            except Exception as exc:
                print(f'独立启动门禁失败：{exc}', file=sys.stderr)
                return 1
            # 放行后的业务异常走标准 traceback 和 main.py 已安装的诊断钩子。
            runpy.run_path(str(main_path), run_name='__main__')
            return 0
        try:
            _validate_token(token)
            return _supervisor_worker(root, token)
        except Exception as exc:
            print(f'独立启动 worker 失败：{exc}', file=sys.stderr)
            return 1
    print('参数无效；请通过 run_main.bat 启动。', file=sys.stderr)
    return 2


if __name__ == '__main__':
    sys.exit(main())
