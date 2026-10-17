"""运行时诊断模块的隔离测试。"""

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path
from unittest import mock

import runtime_diagnostics


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RuntimeDiagnosticsTests(unittest.TestCase):
    """验证进程退出、视频错误和本机日志行为。"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(
            prefix='.runtime-diagnostics-test-', dir=PROJECT_ROOT
        )
        self.temp_path = Path(self.temp_dir.name)
        self.runtime_log = self.temp_path / 'runtime.jsonl'
        self.crash_log = self.temp_path / 'crash.log'

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_child(self, code, runtime_log=None):
        """在临时日志路径运行隔离子进程。"""
        environment = os.environ.copy()
        environment['PYTHONIOENCODING'] = 'utf-8'
        environment['FINDINVIDEO_RUNTIME_LOG_PATH'] = str(runtime_log or self.runtime_log)
        environment['FINDINVIDEO_CRASH_LOG_PATH'] = str(self.crash_log)
        return subprocess.run(
            [sys.executable, '-c', textwrap.dedent(code)],
            cwd=PROJECT_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=30,
        )

    def read_events(self, path=None):
        """读取临时 JSONL 事件。"""
        path = Path(path or self.runtime_log)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

    def test_clean_return_records_queue_exit_once(self):
        result = self.run_child("""
            import runtime_diagnostics as diagnostics

            entry_marker = '保留原入口局部全局语义'
            with diagnostics.process_diagnostics('main.py'):
                pass
            print(entry_marker)
        """)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('保留原入口局部全局语义', result.stdout)
        events = self.read_events()
        exits = [event for event in events if event['event_type'] == 'process_exit']
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0]['reason'], 'queue_exhausted')
        self.assertEqual(exits[0]['exit_code'], 0)
        self.assertEqual(exits[0]['video_failure_count'], 0)
        self.assertTrue(exits[0]['run_id'])
        self.assertEqual(exits[0]['entrypoint'], 'main.py')
        self.assertEqual(exits[0]['pid'], result_pid(events))
        self.assertTrue(exits[0]['host'])

    def test_failure_chain_mirrors_without_importing_utils(self):
        mirror_path = self.temp_path / 'mirror.jsonl'
        result = self.run_child(f"""
            import json
            import sys
            import types

            mirror_path = {str(mirror_path)!r}
            def record_processing_event(event_type, **details):
                with open(mirror_path, 'a', encoding='utf-8') as stream:
                    stream.write(json.dumps({{'event_type': event_type, **details}}) + '\\n')
            sys.modules['utils'] = types.SimpleNamespace(record_processing_event=record_processing_event)

            import runtime_diagnostics as diagnostics
            with diagnostics.process_diagnostics('main.py'):
                diagnostics.record_video_started('sample.mp4', file_md5='md5-1', pipeline_id='p-1')
                try:
                    try:
                        raise ValueError('inner-marker')
                    except ValueError as inner:
                        raise RuntimeError('outer-marker') from inner
                except RuntimeError as error:
                    diagnostics.record_video_failure(
                        'sample.mp4', error=error, file_md5='md5-1', pipeline_id='p-1', stage='decode'
                    )
        """)

        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.read_events()
        failures = [event for event in events if event['event_type'] == 'video_failed']
        self.assertEqual(len(failures), 1)
        failure = failures[0]
        self.assertEqual(failure['exception_type'], 'RuntimeError')
        self.assertEqual(failure['stage'], 'decode')
        self.assertEqual(failure['video_path'], 'sample.mp4')
        self.assertIn('inner-marker', failure['traceback'])
        self.assertIn('outer-marker', failure['traceback'])
        self.assertIn('direct cause', failure['traceback'])
        self.assertNotIn('video_started', [item['event_type'] for item in self._read_jsonl(mirror_path)])
        mirrored = self._read_jsonl(mirror_path)
        mirrored_failure = next(item for item in mirrored if item['event_type'] == 'video_failed')
        self.assertEqual(mirrored_failure['run_id'], failure['run_id'])
        self.assertEqual(mirrored_failure['entrypoint'], 'main.py')
        self.assertEqual(mirrored_failure['runtime_time'], failure['time'])
        exit_event = next(event for event in events if event['event_type'] == 'process_exit')
        self.assertEqual(exit_event['reason'], 'queue_exhausted_with_video_failures')
        self.assertEqual(exit_event['video_failure_count'], 1)
        self.assertEqual(exit_event['video_path'], 'sample.mp4')
        self.assertEqual(exit_event['file_md5'], 'md5-1')
        self.assertEqual(exit_event['pipeline_id'], 'p-1')
        self.assertEqual(exit_event['last_failure']['failure_id'], failure['failure_id'])

    def _read_jsonl(self, path):
        return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]

    def test_raised_exception_keeps_traceback_and_exit_code(self):
        result = self.run_child("""
            import runtime_diagnostics as diagnostics
            with diagnostics.process_diagnostics('main.py'):
                raise RuntimeError('raised-marker')
        """)

        self.assertEqual(result.returncode, 1)
        events = self.read_events()
        exits = [event for event in events if event['event_type'] == 'process_exit']
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0]['reason'], 'uncaught_exception')
        self.assertEqual(exits[0]['exit_code'], 1)
        self.assertEqual(exits[0]['exception_type'], 'RuntimeError')
        self.assertIn('raised-marker', exits[0]['traceback'])
        uncaught = [event for event in events if event['event_type'] == 'uncaught_exception']
        self.assertEqual(len(uncaught), 1)

    def test_explicit_system_exit_keeps_code_and_reason(self):
        result = self.run_child("""
            import runtime_diagnostics as diagnostics
            with diagnostics.process_diagnostics('main.py'):
                raise SystemExit(23)
        """)

        self.assertEqual(result.returncode, 23)
        exits = [event for event in self.read_events() if event['event_type'] == 'process_exit']
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0]['reason'], 'system_exit')
        self.assertEqual(exits[0]['exit_code'], 23)
        self.assertEqual(exits[0]['system_exit_code'], 23)
        self.assertEqual(exits[0]['exception_type'], 'SystemExit')
        self.assertIn('SystemExit: 23', exits[0]['traceback'])

    def test_uncaught_import_exception_is_logged(self):
        failing_module = self.temp_path / 'module_that_fails.py'
        failing_module.write_text("raise RuntimeError('import-phase-marker')\n", encoding='utf-8')
        result = self.run_child(f"""
            import sys
            sys.path.insert(0, {str(self.temp_path)!r})
            import runtime_diagnostics as diagnostics
            diagnostics.install_runtime_diagnostics('main.py')
            import module_that_fails
        """)

        self.assertEqual(result.returncode, 1)
        events = self.read_events()
        uncaught = [event for event in events if event['event_type'] == 'uncaught_exception']
        exits = [event for event in events if event['event_type'] == 'process_exit']
        self.assertEqual(len(uncaught), 1)
        self.assertEqual(len(exits), 1)
        self.assertEqual(uncaught[0]['exception_type'], 'RuntimeError')
        self.assertIn('import-phase-marker', uncaught[0]['traceback'])
        self.assertEqual(exits[0]['reason'], 'uncaught_exception')
        self.assertEqual(exits[0]['exit_code'], 1)

    def test_faulthandler_writes_native_crash_to_isolated_child_log(self):
        result = self.run_child("""
            import os
            import runtime_diagnostics as diagnostics
            diagnostics.install_runtime_diagnostics('main.py')
            diagnostics.record_video_started('native-crash.mp4', file_md5='md5-native')
            os.abort()
        """)

        self.assertNotEqual(result.returncode, 0)
        content = self.crash_log.read_text(encoding='utf-8', errors='replace')
        self.assertIn('native-crash.mp4', content)
        self.assertIn('Fatal Python error', content)
        self.assertIn('Current thread', content)

    def test_unwritable_runtime_log_does_not_mask_original_exception(self):
        blocker = self.temp_path / 'not-a-directory'
        blocker.write_text('occupied', encoding='utf-8')
        impossible_path = blocker / 'runtime.jsonl'
        result = self.run_child("""
            import runtime_diagnostics as diagnostics
            with diagnostics.process_diagnostics('main.py'):
                raise ValueError('original-marker')
        """, runtime_log=impossible_path)

        self.assertEqual(result.returncode, 1)
        self.assertIn('ValueError: original-marker', result.stderr)
        self.assertEqual(result.stderr.count('运行诊断写入失败'), 1)

    def test_console_handler_is_local_and_returns_false(self):
        previous_runtime_path = os.environ.get('FINDINVIDEO_RUNTIME_LOG_PATH')
        os.environ['FINDINVIDEO_RUNTIME_LOG_PATH'] = str(self.runtime_log)
        fake_utils = mock.Mock()
        try:
            with mock.patch.dict(sys.modules, {'utils': fake_utils}):
                for control_type in (1, 2, 5, 6, 0):
                    self.assertIs(runtime_diagnostics._console_control_handler(control_type), False)
        finally:
            if previous_runtime_path is None:
                os.environ.pop('FINDINVIDEO_RUNTIME_LOG_PATH', None)
            else:
                os.environ['FINDINVIDEO_RUNTIME_LOG_PATH'] = previous_runtime_path

        events = self.read_events()
        requested = [event for event in events if event['event_type'] == 'process_termination_requested']
        self.assertEqual([event['reason'] for event in requested], [
            'CTRL_BREAK', 'CTRL_CLOSE', 'LOGOFF', 'SHUTDOWN'
        ])
        fake_utils.record_processing_event.assert_not_called()
        self.assertFalse(any(event['event_type'] == 'process_exit' for event in events))

    def test_keyboard_interrupt_records_platform_exit_code(self):
        result = self.run_child("""
            import runtime_diagnostics as diagnostics
            with diagnostics.process_diagnostics('main.py'):
                raise KeyboardInterrupt('keyboard-marker')
        """)
        exits = [event for event in self.read_events() if event['event_type'] == 'process_exit']
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0]['reason'], 'keyboard_interrupt')
        self.assertEqual(exits[0]['exit_code'], result.returncode)
        self.assertIn('keyboard-marker', exits[0]['traceback'])

    def test_thread_failure_keeps_traceback_without_claiming_process_exit(self):
        result = self.run_child("""
            import threading
            import runtime_diagnostics as diagnostics
            def fail_thread():
                raise RuntimeError('thread-marker')
            with diagnostics.process_diagnostics('main.py'):
                worker = threading.Thread(target=fail_thread, name='diagnostic-worker')
                worker.start()
                worker.join()
        """)
        self.assertEqual(result.returncode, 0, result.stderr)
        events = self.read_events()
        failures = [event for event in events if event['event_type'] == 'thread_exception']
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]['thread_name'], 'diagnostic-worker')
        self.assertIn('fail_thread', failures[0]['traceback'])
        self.assertIn('thread-marker', failures[0]['traceback'])
        self.assertEqual([event for event in events if event['event_type'] == 'process_exit'][0]['reason'], 'queue_exhausted')


def result_pid(events):
    """返回首个事件中的 PID，便于校验运行身份一致。"""
    return events[0]['pid'] if events else None


if __name__ == '__main__':
    unittest.main()
