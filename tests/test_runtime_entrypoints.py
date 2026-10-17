"""用真实入口控制流验证失败与退出诊断，全部业务输入和日志位于临时目录。"""

import json
import os
import pathlib
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import unittest
from contextlib import closing


ROOT = pathlib.Path(__file__).resolve().parents[1]


class RuntimeEntrypointTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(prefix='.runtime-entrypoint-', dir=ROOT)
        self.root = pathlib.Path(self.tempdir.name)
        self.log = self.root / 'runtime.jsonl'

    def tearDown(self):
        self.tempdir.cleanup()

    def run_child(self, code, **environment):
        env = os.environ.copy()
        env.update({
            'PYTHONIOENCODING': 'utf-8',
            'FINDINVIDEO_RUNTIME_LOG_PATH': str(self.log),
            'FINDINVIDEO_CRASH_LOG_PATH': str(self.root / 'crash.log'),
            **environment,
        })
        return subprocess.run(
            [sys.executable, '-c', textwrap.dedent(code)], cwd=ROOT,
            env=env, capture_output=True, text=True, encoding='utf-8', timeout=45,
        )

    def events(self):
        return [json.loads(line) for line in self.log.read_text(encoding='utf-8').splitlines()]

    def test_directory_failure_releases_claim_and_persists_both_logs(self):
        for entrypoint in ('main.py', 'main_nipple.py', 'main_nipple_rog.py'):
            with self.subTest(entrypoint=entrypoint):
                self.log = self.root / (entrypoint + '.jsonl')
                db = self.root / (entrypoint + '.sqlite3')
                video_dir = self.root / entrypoint
                video_dir.mkdir()
                (video_dir / 'sample.mp4').write_bytes(b'video-input-placeholder')
                result = self.run_child('''
                    import os, pathlib, sys
                    from unittest import mock
                    sys.path.insert(0, str(pathlib.Path('tests').resolve()))
                    from test_video_completion_guards import load_entrypoint
                    import runtime_diagnostics as diagnostics
                    import utils

                    module = load_entrypoint(pathlib.Path(os.environ['AUDIT_ENTRYPOINT']))
                    index = utils.DirectoryIndex(os.environ['AUDIT_DB'])
                    utils.DIRECTORY_INDEX = index
                    module.DIRECTORY_INDEX = index
                    module._ACTIVE_PIPELINE_ID = 'audit-flow'
                    module.save_mosaic_switch = False
                    module.save_timestamps_switch = False

                    class Capture:
                        def isOpened(self): return True
                        def get(self, prop): return 25
                        def release(self): pass

                    def fail_video(*args, **kwargs):
                        try:
                            raise OSError(5, 'decode-inner-marker')
                        except OSError as inner:
                            raise RuntimeError('video-outer-marker') from inner

                    with mock.patch.object(module, 'has_existing_artifacts', return_value=False), \\
                         mock.patch.object(module, '_get_processing_decision', return_value=('audit-md5', None)), \\
                         mock.patch.object(module.cv2, 'VideoCapture', return_value=Capture()), \\
                         mock.patch.object(module, '_try_claim_video', return_value=True), \\
                         mock.patch.object(module, '_pause_requested', return_value=False), \\
                         mock.patch.object(module, 'detect_objects_in_video', side_effect=fail_video), \\
                         mock.patch.object(module, '_release_claim_safely') as release, \\
                         mock.patch.object(module, '_mark_directory_done') as mark:
                        with diagnostics.process_diagnostics(os.environ['AUDIT_ENTRYPOINT']):
                            kwargs = {} if pathlib.Path(os.environ['AUDIT_ENTRYPOINT']).name == 'main.py' else {'model': object()}
                            assert module.process_directory_videos(os.environ['AUDIT_VIDEO_DIR'], 'face', **kwargs) == 0
                        release.assert_called_once_with('audit-md5')
                        mark.assert_not_called()
                    index.close()
                ''', AUDIT_ENTRYPOINT=str(ROOT / entrypoint), AUDIT_DB=str(db), AUDIT_VIDEO_DIR=str(video_dir))
                self.assertEqual(result.returncode, 0, result.stderr)
                failures = [event for event in self.events() if event['event_type'] == 'video_failed']
                self.assertEqual(len(failures), 1)
                failure = failures[0]
                self.assertEqual(failure['exception_type'], 'RuntimeError')
                self.assertIn('decode-inner-marker', failure['traceback'])
                self.assertIn('video-outer-marker', failure['traceback'])
                self.assertIn('fail_video', failure['traceback'])
                self.assertEqual(failure['file_md5'], 'audit-md5')
                self.assertEqual(failure['pipeline_id'], 'audit-flow')
                self.assertEqual(failure['video_path'], str(video_dir / 'sample.mp4'))
                exit_event = [event for event in self.events() if event['event_type'] == 'process_exit'][0]
                self.assertEqual(exit_event['reason'], 'queue_exhausted_with_video_failures')
                self.assertEqual(exit_event['video_failure_count'], 1)
                with closing(sqlite3.connect(db)) as connection:
                    details = json.loads(connection.execute(
                        "SELECT details_json FROM processing_events WHERE event_type='video_failed'"
                    ).fetchone()[0])
                self.assertEqual(details['traceback'], failure['traceback'])
                self.assertEqual(details['run_id'], failure['run_id'])

    def test_all_entrypoints_capture_dependency_import_failure(self):
        for name in ('main.py', 'main_nipple.py', 'main_nipple_rog.py', 'main_yolov5.py'):
            with self.subTest(entrypoint=name):
                self.log = self.root / (name + '-import.jsonl')
                result = self.run_child('''
                    import importlib.abc, os, runpy, sys
                    class BrokenDependency(importlib.abc.MetaPathFinder):
                        def find_spec(self, fullname, path=None, target=None):
                            if fullname in ('ultralytics', 'torch'):
                                raise ModuleNotFoundError('dependency-import-marker')
                    sys.meta_path.insert(0, BrokenDependency())
                    runpy.run_path(os.environ['AUDIT_ENTRYPOINT'], run_name='__main__')
                ''', AUDIT_ENTRYPOINT=str(ROOT / name))
                self.assertNotEqual(result.returncode, 0)
                exits = [event for event in self.events() if event['event_type'] == 'process_exit']
                self.assertEqual(len(exits), 1)
                self.assertEqual(exits[0]['reason'], 'uncaught_exception')
                self.assertIn('dependency-import-marker', exits[0]['traceback'])
                self.assertEqual(exits[0]['entrypoint'], str(ROOT / name))

    def test_yolov5_failed_results_keep_original_stack_and_child_stderr(self):
        result = self.run_child('''
            import pathlib, subprocess, sys
            from unittest import mock
            sys.path.insert(0, str(pathlib.Path('tests').resolve()))
            from test_yolov5_processing import load_yolov5_module
            import runtime_diagnostics as diagnostics
            module = load_yolov5_module()
            def fail_model(*args, **kwargs):
                raise ValueError('model-load-marker')
            with diagnostics.process_diagnostics('main_yolov5.py'):
                with mock.patch.object(module, 'load_yolov5_model', side_effect=fail_model):
                    result = module.detect_objects_in_video_yolov5('sample.mp4', 'face')
                    assert not result.success
                child = subprocess.CompletedProcess([], 7, stdout='child-output', stderr='child-traceback-marker')
                with mock.patch.object(module, '_run_detect_command', return_value=child):
                    result = module.detect_objects_with_frame_analysis('second.mp4', 'face')
                    assert not result.success
        ''')
        self.assertEqual(result.returncode, 0, result.stderr)
        failures = [event for event in self.events() if event['event_type'] == 'video_failed']
        self.assertEqual(len(failures), 2)
        self.assertIn('fail_model', failures[0]['traceback'])
        self.assertEqual(failures[0]['exception_type'], 'ValueError')
        self.assertEqual(failures[1]['subprocess_returncode'], 7)
        self.assertEqual(failures[1]['subprocess_stderr'], 'child-traceback-marker')
        self.assertEqual(failures[1]['traceback'], '')

    def test_pause_source_is_local_even_without_shared_disk(self):
        result = self.run_child('''
            import runtime_diagnostics as diagnostics
            import utils
            with diagnostics.process_diagnostics('main.py'):
                utils._STOP_REQUESTED = True
                assert utils._pause_requested()
                diagnostics.set_exit_reason('pause_requested')
                raise SystemExit(0)
        ''')
        self.assertEqual(result.returncode, 0, result.stderr)
        pause = [event for event in self.events() if event['event_type'] == 'pause_requested'][0]
        self.assertEqual(pause['source'], 'sigint')
        exit_event = [event for event in self.events() if event['event_type'] == 'process_exit'][0]
        self.assertEqual(exit_event['reason'], 'pause_requested')
        self.assertEqual(exit_event['exit_code'], 0)

    @unittest.skipUnless(os.name == 'nt', '仅 Windows 有 BAT 启动入口')
    def test_launcher_persists_exit_code_even_without_python_cleanup(self):
        script = (ROOT / 'run_main.bat').read_text(encoding='utf-8')
        script = script.replace('".venv\\Scripts\\python.exe"', '"' + sys.executable + '"')
        launcher = self.root / 'run_main.bat'
        launcher.write_bytes(script.replace('\r\n', '\n').replace('\n', '\r\n').encode('utf-8'))
        (self.root / 'main.py').write_text('import os\nos._exit(7)\n', encoding='utf-8')
        result = subprocess.run(
            ['cmd.exe', '/d', '/c', str(launcher)], cwd=self.root,
            capture_output=True, timeout=30,
        )
        self.assertEqual(result.returncode, 7, result.stderr)
        log = (self.root / 'logs' / 'launcher_events.log').read_text(encoding='utf-8')
        self.assertIn('event=launcher_started', log)
        self.assertIn('event=launcher_exit entrypoint=main.py exit_code=7', log)


if __name__ == '__main__':
    unittest.main()
