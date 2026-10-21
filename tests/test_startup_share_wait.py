"""启动共享盘不可访问时的等待、恢复和入口流程回归测试。"""

import ast
import io
import json
import os
import pathlib
import sqlite3
import tempfile
import unittest
from contextlib import nullcontext, redirect_stdout
from unittest import mock

import utils
from test_video_completion_guards import load_entrypoint


ROOT = pathlib.Path(__file__).resolve().parents[1]


class StartupShareWaitTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix='.startup-share-test-', dir=ROOT)
        self.root = pathlib.Path(self.tempdir.name)
        self.log = self.root / 'recovery.jsonl'
        self.environment = mock.patch.dict(os.environ, {
            'FINDINVIDEO_RECOVERY_LOG_PATH': str(self.log),
            'FINDINVIDEO_RUNTIME_LOG_PATH': str(self.root / 'runtime.jsonl'),
            'FINDINVIDEO_CRASH_LOG_PATH': str(self.root / 'crash.log'),
        })
        self.environment.start()
        os.environ.pop('FINDINVIDEO_RECOVERY_RETRY_SECONDS', None)
        self.module = load_entrypoint(ROOT / 'main.py')
        self.index = utils.DirectoryIndex()
        self.index_patch = mock.patch.object(utils, 'DIRECTORY_INDEX', self.index)
        self.index_patch.start()
        self.pause = mock.patch.object(self.module, '_pause_requested', return_value=False)
        self.pause_mock = self.pause.start()
        self.delays = []

    def tearDown(self):
        self.pause.stop()
        self.index_patch.stop()
        self.index.close()
        self.environment.stop()
        self.tempdir.cleanup()

    def wait_without_real_sleep(self, failure, **kwargs):
        """只替换等待时钟，实际执行生产恢复探测及日志路径。"""
        return utils.wait_for_recovery(failure, sleep_fn=self.delays.append, **kwargs)

    def events(self):
        return [json.loads(line) for line in self.log.read_text(encoding='utf-8').splitlines()]

    def test_missing_share_waits_three_intervals_then_continues(self):
        target = self.root / 'share'

        def sleep(seconds):
            self.delays.append(seconds)
            if sum(self.delays) >= 900 and not target.exists():
                target.mkdir()
                (target / 'sample.mp4').write_bytes(b'video')

        def waiter(failure, **kwargs):
            return utils.wait_for_recovery(failure, sleep_fn=sleep, **kwargs)

        output = io.StringIO()
        with mock.patch.object(self.module, 'wait_for_recovery', side_effect=waiter), \
             mock.patch.object(utils, 'probe_recovery_target') as parent_probe, \
             redirect_stdout(output):
            readiness = self.module._wait_for_startup_input(str(target))

        self.assertTrue(readiness['is_directory'])
        self.assertFalse(readiness['empty'])
        self.assertEqual(sum(self.delays), 900)
        parent_probe.assert_not_called()
        self.assertEqual(len(output.getvalue().splitlines()), 4)
        events = self.events()
        self.assertEqual(sum(event['event_type'] == 'recovery_wait' for event in events), 3)
        self.assertEqual(sum(event['event_type'] == 'recovery_probe_failed' for event in events), 2)
        self.assertEqual(events[-1]['event_type'], 'recovery_probe_succeeded')
        self.assertEqual(events[-1]['stage'], 'startup_input')

    def test_unreadable_directory_retries_listing_before_resuming(self):
        target = self.root / 'share'
        target.mkdir()
        real_listdir = os.listdir
        attempts = []

        def listdir(path):
            if str(path) == str(target):
                attempts.append(True)
                if len(attempts) < 3:
                    raise PermissionError(13, '共享目录暂时不可读')
                return ['sample.mp4']
            return real_listdir(path)

        with mock.patch.object(os, 'listdir', side_effect=listdir), \
             mock.patch.object(self.module, 'wait_for_recovery', side_effect=self.wait_without_real_sleep):
            readiness = self.module._wait_for_startup_input(str(target))
        self.assertFalse(readiness['empty'])
        self.assertEqual(len(attempts), 3)
        self.assertEqual(sum(self.delays), 600)
        self.assertEqual(self.events()[0]['exception_class'], 'PermissionError')

    def test_missing_single_file_does_not_resume_just_because_parent_exists(self):
        target = self.root / 'sample.mp4'

        def sleep(seconds):
            self.delays.append(seconds)
            if sum(self.delays) >= 300:
                target.write_bytes(b'video')

        with mock.patch.object(self.module, 'wait_for_recovery', side_effect=lambda failure, **kwargs: utils.wait_for_recovery(failure, sleep_fn=sleep, **kwargs)):
            readiness = self.module._wait_for_startup_input(str(target))
        self.assertFalse(readiness['is_directory'])
        self.assertFalse(readiness['empty'])
        self.assertEqual(sum(self.delays), 300)

    def test_reachable_empty_directory_does_not_wait(self):
        with mock.patch.object(self.module, 'wait_for_recovery') as wait:
            target = self.root / 'empty'
            target.mkdir()
            readiness = self.module._wait_for_startup_input(str(target))
        self.assertTrue(readiness['empty'])
        wait.assert_not_called()

    def test_active_pause_interrupts_wait_without_exiting_as_missing_input(self):
        self.pause_mock.side_effect = [False, False, True]
        with mock.patch.object(self.module, 'wait_for_recovery', side_effect=self.wait_without_real_sleep):
            with self.assertRaises(utils.PauseRequested):
                self.module._wait_for_startup_input(str(self.root / 'missing'))
        self.assertEqual(self.delays, [])
        self.assertEqual(self.events()[-1]['event_type'], 'pause_requested')

    def test_startup_database_failure_retries_even_when_database_folder_is_missing(self):
        target = self.root / 'share'
        target.mkdir()
        fake_index = mock.Mock()
        fake_index.reopen.side_effect = [sqlite3.OperationalError('unable to open database file'), None]
        real_makedirs = os.makedirs
        failed = []

        def makedirs(path, **kwargs):
            if str(path) == str(target / 'md5_list') and not failed:
                failed.append(True)
                raise PermissionError(13, '共享盘暂时不可写')
            return real_makedirs(path, **kwargs)

        with mock.patch.object(self.module, 'DIRECTORY_INDEX', fake_index), \
             mock.patch.object(os, 'makedirs', side_effect=makedirs):
            utils.retry_with_recovery(
                lambda: self.module._init_processing_root(str(target)), target_dir=str(target),
                pause_checker=lambda: False, wait_fn=self.wait_without_real_sleep,
            )
        self.assertEqual(fake_index.reopen.call_count, 2)
        self.assertTrue((target / 'md5_list').is_dir())
        self.assertEqual(sum(self.delays), 600)
        self.assertEqual(self.events()[-1]['event_type'], 'recovery_resumed')

    def test_permanent_database_error_is_not_infinite_storage_retry(self):
        fake_index = mock.Mock()
        fake_index.reopen.side_effect = sqlite3.DatabaseError('file is not a database')
        with mock.patch.object(self.module, 'DIRECTORY_INDEX', fake_index):
            with self.assertRaises(sqlite3.DatabaseError):
                self.module._init_processing_root(str(self.root))

    def test_wrapped_schema_io_error_remains_recoverable(self):
        try:
            try:
                raise sqlite3.OperationalError('disk I/O error')
            except sqlite3.OperationalError as cause:
                raise RuntimeError('创建必要表结构失败') from cause
        except RuntimeError as error:
            wrapped = error
        fake_index = mock.Mock()
        fake_index.reopen.side_effect = wrapped
        with mock.patch.object(self.module, 'DIRECTORY_INDEX', fake_index):
            with self.assertRaises(utils.RecoverableProcessingError) as caught:
                self.module._init_processing_root(str(self.root))
        self.assertIs(caught.exception.__cause__, wrapped)
        self.assertIn('disk I/O error', self.events()[0]['details']['traceback'])

    def test_missing_root_is_not_created_as_empty_local_input(self):
        target = self.root / 'missing'
        with self.assertRaises(utils.RecoverableProcessingError):
            self.module._init_processing_root(str(target))
        self.assertFalse(target.exists())

    def test_actual_entry_keeps_empty_directory_exit_behavior(self):
        tree = ast.parse((ROOT / 'main.py').read_text(encoding='utf-8'))
        entry = [node for node in tree.body if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"][-1]
        target = self.root / 'empty'
        target.mkdir()
        for node in ast.walk(entry):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'video_path' for t in node.targets):
                node.value = ast.Constant(str(target))
        ast.fix_missing_locations(entry)
        namespace = self.module.__dict__
        namespace['__name__'] = '__main__'
        with mock.patch.object(self.module, '_install_pause_signal_handler'), \
             mock.patch.object(self.module, 'process_diagnostics', side_effect=lambda _: nullcontext()), \
             mock.patch.object(self.module, 'wait_for_recovery') as wait, \
             mock.patch.object(self.module, 'set_exit_reason') as reason, \
             mock.patch.object(self.module, 'YOLO') as model:
            with self.assertRaises(SystemExit) as caught:
                exec(compile(ast.Module(body=[entry], type_ignores=[]), str(ROOT / 'main.py'), 'exec'), namespace)
        self.assertEqual(caught.exception.code, 1)
        wait.assert_not_called()
        model.assert_not_called()
        reason.assert_called_once_with('input_directory_empty')

    def test_actual_entry_waits_then_initializes_database_and_reaches_processing(self):
        source = (ROOT / 'main.py').read_text(encoding='utf-8')
        tree = ast.parse(source)
        entry = [node for node in tree.body if isinstance(node, ast.If) and ast.unparse(node.test) == "__name__ == '__main__'"][-1]
        namespace = self.module.__dict__
        namespace['__name__'] = '__main__'
        target = self.root / 'share'
        # 只替换原入口的配置路径；保留实际启动检查、模型顺序和数据库初始化控制流。
        for node in ast.walk(entry):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'video_path' for t in node.targets):
                node.value = ast.Constant(str(target))
        ast.fix_missing_locations(entry)
        lifecycle = []

        def sleep(seconds):
            self.delays.append(seconds)
            if sum(self.delays) >= 300 and not target.exists():
                target.mkdir()
                (target / 'sample.mp4').write_bytes(b'video')

        def load_model(*args):
            self.assertTrue(target.exists())
            lifecycle.append('model')
            return object()

        fake_index = mock.Mock()
        fake_index.reopen.side_effect = lambda path: lifecycle.append('database')
        with mock.patch.object(self.module, 'YOLO', side_effect=load_model), \
             mock.patch.object(self.module, '_activate_pipeline'), \
             mock.patch.object(self.module, '_install_pause_signal_handler'), \
             mock.patch.object(self.module, 'DIRECTORY_INDEX', fake_index), \
             mock.patch.object(self.module, 'wait_for_recovery', side_effect=lambda failure, **kwargs: utils.wait_for_recovery(failure, sleep_fn=sleep, **kwargs)), \
             mock.patch.object(self.module, 'process_diagnostics', side_effect=lambda _: nullcontext()), \
             mock.patch.object(self.module.utils, 'cleanup_orphan_artifacts'), \
             mock.patch.object(self.module, 'count_videos_in_directory', return_value=1), \
             mock.patch.object(self.module, 'process_directory_videos', side_effect=lambda *args: lifecycle.append('processing')), \
             mock.patch.object(self.module, 'find_leaf_directories_with_videos', return_value=[]):
            exec(compile(ast.Module(body=[entry], type_ignores=[]), str(ROOT / 'main.py'), 'exec'), namespace)
        self.assertEqual(lifecycle, ['model', 'database', 'processing'])
        self.assertEqual(sum(self.delays), 300)
        fake_index.reopen.assert_called_once_with(str(target / 'md5_list' / 'directory_index.db'))


if __name__ == '__main__':
    unittest.main()
