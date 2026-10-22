"""独立启动的身份校验、broker 拒绝和原生异步退出码验收。"""

import ctypes
from ctypes import wintypes
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

import independent_launcher as launcher


ROOT = pathlib.Path(__file__).resolve().parents[1]


class IndependentLauncherTests(unittest.TestCase):
    def test_handshake_writer_retries_while_reader_holds_file(self):
        with tempfile.TemporaryDirectory(prefix='.launcher-json-', dir=ROOT) as temporary:
            path = pathlib.Path(temporary) / 'handshake.json'
            launcher._write_json(path, {'state': 'before'})
            original_load = json.load
            original_replace = os.replace
            attempted = threading.Event()
            errors = []

            def replace(source, destination):
                try:
                    return original_replace(source, destination)
                finally:
                    attempted.set()

            def writer():
                try:
                    launcher._write_json(path, {'state': 'after'})
                except Exception as exc:
                    errors.append(exc)

            thread = threading.Thread(target=writer)

            def replace_while_reading(stream):
                thread.start()
                self.assertTrue(attempted.wait(timeout=2))
                return original_load(stream)

            with mock.patch.object(launcher.json, 'load', side_effect=replace_while_reading), \
                 mock.patch.object(launcher.os, 'replace', side_effect=replace):
                try:
                    previous = launcher._read_json(path)
                finally:
                    thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(previous['state'], 'before')
            self.assertEqual(launcher._read_json(path)['state'], 'after')

    def make_identity(self):
        created_at = time.time()
        record = {
            'token': 'a' * 32,
            'pid': 321,
            'image': str(pathlib.Path(sys.executable).resolve()),
            'created_at': created_at,
            'session': 7,
            'in_job': False,
            'console_visible': True,
        }
        return record, {key: value for key, value in record.items() if key != 'token'}

    def validate(self, record, actual):
        launcher._validate_identity(
            record, actual, 'a' * 32, 321, str(pathlib.Path(sys.executable).resolve()), 7,
            time.time() - 0.5,
        )

    def test_valid_identity_is_accepted(self):
        record, actual = self.make_identity()
        self.validate(record, actual)

    def test_job_membership_is_rejected_before_main_acceptance(self):
        record, actual = self.make_identity()
        actual['in_job'] = True
        with self.assertRaisesRegex(launcher.LauncherError, 'Windows Job'):
            self.validate(record, actual)

    def test_token_pid_session_and_console_mismatches_are_rejected(self):
        for field, value in (
            ('token', 'b' * 32),
            ('pid', 654),
            ('session', 8),
            ('console_visible', False),
        ):
            with self.subTest(field=field):
                record, actual = self.make_identity()
                record[field] = value
                with self.assertRaises(launcher.LauncherError):
                    self.validate(record, actual)

    def test_broker_nonzero_exit_is_a_hard_failure(self):
        completed = subprocess.CompletedProcess([], 17, stdout='', stderr='broker marker')
        with mock.patch.object(pathlib.Path, 'is_file', return_value=True), mock.patch.object(
            launcher.subprocess, 'run', return_value=completed,
        ):
            with self.assertRaisesRegex(launcher.LauncherError, '返回错误码 17'):
                launcher._invoke_broker('worker command', ROOT, {}, timeout=1)

    def test_broker_payload_preserves_unicode_root_and_environment(self):
        completed = subprocess.CompletedProcess([], 0, stdout='{"ReturnValue":0,"ProcessId":91}', stderr='')
        root_path = pathlib.Path('C:/视频/输入')
        with mock.patch.object(pathlib.Path, 'is_file', return_value=True), mock.patch.object(
            launcher.subprocess, 'run', return_value=completed,
        ) as run:
            pid = launcher._invoke_broker(
                'worker command', root_path, {'FINDINVIDEO_TEST_VALUE': '包含中文'}, timeout=1,
            )
        payload_text = run.call_args.kwargs['input']
        self.assertTrue(payload_text.isascii())
        payload = json.loads(payload_text)
        self.assertEqual(payload['root'], str(root_path))
        self.assertEqual(payload['environment'], ['FINDINVIDEO_TEST_VALUE=包含中文'])
        self.assertEqual(pid, 91)

    def test_main_import_failure_keeps_runtime_traceback_and_exit_code(self):
        with tempfile.TemporaryDirectory(prefix='.launcher-import-failure-', dir=ROOT) as temporary:
            root = pathlib.Path(temporary)
            logs = root / 'logs'
            logs.mkdir()
            shutil.copy2(ROOT / 'runtime_diagnostics.py', root / 'runtime_diagnostics.py')
            main_path = root / 'main.py'
            main_path.write_text(
                'from runtime_diagnostics import install_runtime_diagnostics\n'
                'install_runtime_diagnostics(__file__)\n'
                'import launcher_missing_dependency_marker\n',
                encoding='utf-8',
            )
            driver = (
                'import pathlib, sys\n'
                'fixture = pathlib.Path(sys.argv[1])\n'
                'main_path = pathlib.Path(sys.argv[2])\n'
                f'sys.path.insert(0, {str(ROOT)!r})\n'
                'import independent_launcher as launcher\n'
                'launcher._main_worker = lambda root, token: main_path\n'
                'sys.argv = [str(launcher.__file__), "--main-worker", "' + 'a' * 32 + '"]\n'
                'sys.exit(launcher.main())\n'
            )
            environment = os.environ.copy()
            environment['FINDINVIDEO_RUNTIME_LOG_PATH'] = str(logs / 'runtime_events.jsonl')
            environment['FINDINVIDEO_CRASH_LOG_PATH'] = str(logs / 'crash.log')
            result = subprocess.run(
                [sys.executable, '-c', driver, str(root), str(main_path)], cwd=ROOT, env=environment,
                capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=20,
            )
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn('Traceback (most recent call last)', result.stderr)
            self.assertIn('launcher_missing_dependency_marker', result.stderr)
            events = [
                json.loads(line) for line in (logs / 'runtime_events.jsonl').read_text(encoding='utf-8').splitlines()
            ]
            exits = [event for event in events if event.get('event_type') == 'process_exit']
            self.assertEqual(len(exits), 1)
            self.assertEqual(exits[0]['reason'], 'uncaught_exception')
            self.assertEqual(exits[0]['exit_code'], 1)
            self.assertIn('launcher_missing_dependency_marker', exits[0]['traceback'])

    def test_launch_main_completes_token_bound_supervisor_and_main_handshake(self):
        with tempfile.TemporaryDirectory(prefix='.launcher-handshake-', dir=ROOT) as temporary:
            root = pathlib.Path(temporary)
            logs = root / 'logs'
            logs.mkdir()
            (root / 'independent_launcher.py').write_text('# fixture entrypoint\n', encoding='utf-8')
            (root / 'main.py').write_text('# fixture main\n', encoding='utf-8')
            supervisor_pid = 4040
            main_pid = 5050
            identities = {}
            thread_errors = []

            def identity(pid, token):
                return {
                    'token': token,
                    'pid': pid,
                    'image': str(pathlib.Path(sys.executable).resolve()),
                    'created_at': time.time(),
                    'session': 7,
                    'in_job': False,
                    'console_visible': True,
                }

            def actual_identity(pid):
                actual = dict(identities[pid])
                actual.pop('token')
                return actual

            def read_until(path, token, state, deadline):
                while time.monotonic() < deadline:
                    document = launcher._try_read_json(path)
                    if document and document.get('token') == token and document.get('state') == state:
                        return document
                    time.sleep(0.01)
                raise TimeoutError(f'等待测试状态超时: {state}')

            def fake_supervisor(token, result_path):
                try:
                    deadline = time.monotonic() + 4
                    pending = read_until(result_path, token, 'pending', deadline)
                    while not pending.get('broker_pid'):
                        time.sleep(0.01)
                        pending = launcher._read_json(result_path)
                    supervisor = identity(supervisor_pid, token)
                    identities[supervisor_pid] = supervisor
                    pending.update({'state': 'supervisor_ready', 'supervisor': supervisor})
                    launcher._write_json(result_path, pending)

                    document = read_until(result_path, token, 'supervisor_accepted', deadline)
                    main_started_at = time.time()
                    main = identity(main_pid, token)
                    main['created_at'] = main_started_at
                    identities[main_pid] = main
                    document.update({'state': 'main_starting', 'main_started_at': main_started_at})
                    launcher._write_json(result_path, document)

                    paths = launcher._paths(root, token)
                    main_ready = {'state': 'ready', **main}
                    launcher._write_json(paths['main_ready'], main_ready)
                    document.update({'state': 'awaiting_acceptance', 'main': main_ready, 'main_pid': main_pid})
                    launcher._write_json(result_path, document)

                    document = read_until(result_path, token, 'accepted', deadline)
                    document.update({'state': 'running'})
                    launcher._write_json(result_path, document)
                except Exception as exc:
                    thread_errors.append(exc)

            class FakeKernel:
                @staticmethod
                def ProcessIdToSessionId(pid, output):
                    ctypes.cast(output, ctypes.POINTER(wintypes.DWORD)).contents.value = 7
                    return 1

            def fake_broker(command, broker_root, environment, timeout):
                result_path = next(logs.glob('launcher_startup_*.json'))
                token = json.loads(result_path.read_text(encoding='utf-8'))['token']
                thread = threading.Thread(target=fake_supervisor, args=(token, result_path), daemon=True)
                thread.start()
                return supervisor_pid

            expected_image = str(pathlib.Path(sys.executable).resolve())
            with mock.patch.object(launcher, '_is_windows', return_value=True), \
                 mock.patch.object(launcher, '_kernel32', return_value=FakeKernel()), \
                 mock.patch.object(launcher, '_base_python', return_value=pathlib.Path(expected_image)), \
                 mock.patch.object(launcher, '_child_environment', return_value={}), \
                 mock.patch.object(launcher, '_invoke_broker', side_effect=fake_broker), \
                 mock.patch.object(launcher, '_inspect_process', side_effect=actual_identity):
                try:
                    result = launcher.launch_main(root=root, timeout=4)
                except Exception as exc:
                    self.fail(f'{exc}; 测试监护线程错误: {thread_errors!r}')

            self.assertEqual(result['state'], 'accepted')
            self.assertEqual(result['supervisor_pid'], supervisor_pid)
            self.assertEqual(result['main_pid'], main_pid)
            self.assertEqual(result['main']['token'], result['token'])
            self.assertTrue(pathlib.Path(result['startup_result']).is_file())
            self.assertFalse(thread_errors, thread_errors)

    @unittest.skipUnless(
        os.name == 'nt' and os.environ.get('FINDINVIDEO_RUN_NATIVE_LAUNCHER_TESTS') == '1',
        '需显式开启 Windows WMI 原生启动验收',
    )
    def test_bat_returns_acceptance_and_persists_fake_main_exit_code(self):
        with tempfile.TemporaryDirectory(prefix='.independent-launcher-', dir=ROOT) as temporary:
            root = pathlib.Path(temporary)
            logs = root / 'logs'
            logs.mkdir()
            shutil.copy2(ROOT / 'independent_launcher.py', root / 'independent_launcher.py')
            bat_text = (ROOT / 'run_main.bat').read_text(encoding='ascii')
            expected_call = '".venv\\Scripts\\python.exe"'
            self.assertIn(expected_call, bat_text)
            bat_text = bat_text.replace(expected_call, f'"{sys.executable}"')
            (root / 'run_main.bat').write_bytes(bat_text.replace('\n', '\r\n').encode('ascii'))
            (root / 'main.py').write_text('import os\nos._exit(7)\n', encoding='utf-8')

            command_shell = os.environ.get('COMSPEC', r'C:\Windows\System32\cmd.exe')
            environment = os.environ.copy()
            environment['FINDINVIDEO_RUNTIME_LOG_PATH'] = str(root / 'logs' / 'runtime_events.jsonl')
            environment['FINDINVIDEO_CRASH_LOG_PATH'] = str(root / 'logs' / 'crash.log')

            def run_batch():
                result = subprocess.run(
                    [command_shell, '/d', '/c', str(root / 'run_main.bat')], cwd=root, env=environment,
                    capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=45,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

            deadline = time.monotonic() + 30
            run_batch()
            result_files = list((root / 'logs').glob('launcher_startup_*.json'))
            self.assertEqual(len(result_files), 1)
            startup_path = result_files[0]
            startup = None
            while time.monotonic() < deadline:
                startup = json.loads(startup_path.read_text(encoding='utf-8'))
                if startup.get('state') in ('complete', 'failed', 'rejected'):
                    break
                time.sleep(0.05)
            self.assertEqual(startup.get('state'), 'complete', startup)
            self.assertEqual(startup.get('exit_code'), 7, startup)
            events = (root / 'logs' / 'launcher_events.log').read_text(encoding='utf-8')
            self.assertIn('event=launcher_started entrypoint=main.py', events)
            self.assertIn('event=launcher_exit entrypoint=main.py', events)
            self.assertIn('exit_code=7', events)

            shutil.copy2(ROOT / 'runtime_diagnostics.py', root / 'runtime_diagnostics.py')
            (root / 'main.py').write_text(
                'from runtime_diagnostics import install_runtime_diagnostics\n'
                'install_runtime_diagnostics(__file__)\n'
                'import launcher_missing_dependency_marker\n',
                encoding='utf-8',
            )
            run_batch()
            result_files = list((root / 'logs').glob('launcher_startup_*.json'))
            self.assertEqual(len(result_files), 2)
            second_path = next(path for path in result_files if path != startup_path)
            deadline = time.monotonic() + 30
            failure_startup = None
            while time.monotonic() < deadline:
                failure_startup = json.loads(second_path.read_text(encoding='utf-8'))
                if failure_startup.get('state') in ('complete', 'failed', 'rejected'):
                    break
                time.sleep(0.05)
            self.assertEqual(failure_startup.get('state'), 'complete', failure_startup)
            self.assertEqual(failure_startup.get('exit_code'), 1, failure_startup)
            runtime_events = [
                json.loads(line) for line in (logs / 'runtime_events.jsonl').read_text(encoding='utf-8').splitlines()
            ]
            exits = [event for event in runtime_events if event.get('event_type') == 'process_exit']
            self.assertEqual(len(exits), 1)
            self.assertEqual(exits[0]['reason'], 'uncaught_exception')
            self.assertEqual(exits[0]['exception_type'], 'ModuleNotFoundError')
            self.assertIn('launcher_missing_dependency_marker', exits[0]['traceback'])
            events = (root / 'logs' / 'launcher_events.log').read_text(encoding='utf-8')
            self.assertIn('exit_code=1', events)


    @unittest.skipUnless(
        os.name == 'nt' and os.environ.get('FINDINVIDEO_RUN_NATIVE_LAUNCHER_TESTS') == '1',
        '需显式开启 Windows WMI 原生启动验收',
    )
    def test_repeated_recovery_restart_and_ctrl_c_preserve_final_exit_code(self):
        with tempfile.TemporaryDirectory(prefix='.launcher-restart-', dir=ROOT) as temporary:
            root = pathlib.Path(temporary)
            shutil.copy2(ROOT / 'independent_launcher.py', root / 'independent_launcher.py')
            (root / 'main.py').write_text(
                'import ctypes, json, os, pathlib, signal, time\n'
                'from independent_launcher import restart_main, is_in_job\n'
                'root=pathlib.Path(__file__).parent\n'
                "history=root/'stages.jsonl'\n"
                "stage=len(history.read_text().splitlines()) if history.exists() else 0\n"
                "with (root/'stages.jsonl').open('a') as stream:\n"
                "    stream.write(json.dumps({'stage':stage,'pid':os.getpid(),'in_job':is_in_job()})+'\\n')\n"
                'if stage < 2:\n'
                '    restart_main(__file__)\n'
                'seen=[]\n'
                'signal.signal(signal.SIGINT, lambda *_: seen.append(True))\n'
                'if not ctypes.windll.kernel32.GenerateConsoleCtrlEvent(0, 0): raise ctypes.WinError()\n'
                'deadline=time.monotonic()+3\n'
                'while not seen and time.monotonic()<deadline: time.sleep(.01)\n'
                'assert seen, "处理进程未收到 Ctrl+C"\n'
                'os._exit(-1073741819)\n',
                encoding='utf-8',
            )
            result = launcher.launch_main(root)
            startup_path = pathlib.Path(result['startup_result'])
            deadline = time.monotonic() + 15
            startup = {}
            while time.monotonic() < deadline:
                startup = launcher._read_json(startup_path)
                if startup.get('state') == 'complete':
                    break
                time.sleep(.05)
            self.assertEqual(startup.get('state'), 'complete', startup)
            self.assertEqual(startup.get('exit_code'), 0xC0000005, startup)
            self.assertEqual(startup.get('restart_count'), 2, startup)
            stages = [json.loads(line) for line in (root / 'stages.jsonl').read_text().splitlines()]
            self.assertEqual([stage['stage'] for stage in stages], [0, 1, 2])
            self.assertEqual(len({stage['pid'] for stage in stages}), 3)
            self.assertTrue(all(stage['in_job'] is False for stage in stages))
            for stage in stages[:2]:
                with self.assertRaises(OSError):
                    launcher._inspect_process(stage['pid'])
            # 等待本次监护进程完全退出，避免其工作目录妨碍临时目录清理。
            api = launcher._kernel32()
            api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            handle = api.OpenProcess(0x100000, False, result['supervisor_pid'])
            if handle:
                try:
                    self.assertEqual(api.WaitForSingleObject(handle, 5000), 0)
                finally:
                    api.CloseHandle(handle)
            self.assertIn('exit_code=3221225477', (root / 'logs/launcher_events.log').read_text())


if __name__ == '__main__':
    unittest.main()
