"""共享主机离线恢复与真实坏视频处理语义的回归测试。"""

import io
import json
import os
import pathlib
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import utils
import runtime_diagnostics
from test_video_completion_guards import load_entrypoint, FakeProgress, EmptyModel


ROOT = pathlib.Path(__file__).resolve().parents[1]


def offline_error():
    error = OSError('共享主机已关机')
    error.winerror = 64
    return error


class ShareOfflineResumeTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(dir=ROOT, prefix='.offline-test-')
        self.root = pathlib.Path(self.tempdir.name)
        self.video = self.root / 'sample.mp4'
        self.video.write_bytes(b'video-placeholder')
        self.module = load_entrypoint(ROOT / 'main.py')
        self.module._PROCESSING_ROOT_DIR = str(self.root)
        self.module._ACTIVE_PIPELINE_ID = 'offline-test'
        self.index = utils.DirectoryIndex()
        self.module.DIRECTORY_INDEX = self.index
        self.patch_index = mock.patch.object(utils, 'DIRECTORY_INDEX', self.index)
        self.patch_index.start()
        self.environment = mock.patch.dict(os.environ, {
            'FINDINVIDEO_RECOVERY_RETRY_SECONDS': '300',
            'FINDINVIDEO_RECOVERY_LOG_PATH': str(self.root / 'recovery.jsonl'),
            'FINDINVIDEO_RUNTIME_LOG_PATH': str(self.root / 'runtime.jsonl'),
            'FINDINVIDEO_CRASH_LOG_PATH': str(self.root / 'crash.log'),
        })
        self.environment.start()
        self.crash_setup = mock.patch.object(runtime_diagnostics, '_ensure_crash_log')
        self.crash_write = mock.patch.object(runtime_diagnostics, '_write_crash_line')
        self.crash_setup.start()
        self.crash_write.start()

    def tearDown(self):
        self.crash_write.stop()
        self.crash_setup.stop()
        self.environment.stop()
        self.patch_index.stop()
        self.index.close()
        self.tempdir.cleanup()

    def test_scan_offline_preserves_existing_directory_index(self):
        self.index.refresh(str(self.root))
        previous = [tuple(row) for row in self.index.conn.execute('SELECT * FROM directories')]
        videos = [tuple(row) for row in self.index.conn.execute('SELECT * FROM videos')]
        with mock.patch.object(os, 'scandir', side_effect=offline_error()):
            with self.assertRaises(utils.RecoverableProcessingError):
                self.module.find_leaf_directories_with_videos(str(self.root))
        self.assertEqual(previous, [tuple(row) for row in self.index.conn.execute('SELECT * FROM directories')])
        self.assertEqual(videos, [tuple(row) for row in self.index.conn.execute('SELECT * FROM videos')])

    def test_fallback_walk_cannot_return_partial_results_when_share_goes_offline(self):
        def walk(path, **kwargs):
            yield str(self.root), [], ['sample.mp4']
            kwargs['onerror'](offline_error())

        with mock.patch.object(self.index, 'refresh'), \
             mock.patch.object(self.index, 'get_leaf_directories', return_value=[]), \
             mock.patch.object(self.module, 'is_leaf_directory', return_value=True), \
             mock.patch.object(self.module, 'count_videos_in_directory', return_value=1), \
             mock.patch.object(self.module, '_check_all_videos_have_artifacts', return_value=False), \
             mock.patch.object(os, 'walk', side_effect=walk):
            with self.assertRaises(utils.RecoverableProcessingError):
                self.module.find_leaf_directories_with_videos(str(self.root))

    def test_directory_listing_is_retried_after_two_sparse_probes(self):
        online = [False]
        real_scandir = os.scandir
        real_listdir = os.listdir

        def listdir(path):
            if not online[0]:
                raise offline_error()
            return real_listdir(path)

        def scandir(path):
            if not online[0]:
                raise offline_error()
            return real_scandir(path)

        delays = []
        processed = []

        def sleep(seconds):
            delays.append(seconds)
            if sum(delays) >= 600:
                online[0] = True

        def waiter(failure, **kwargs):
            return utils.wait_for_recovery(failure, sleep_fn=sleep, **kwargs)

        real_probe = utils.probe_recovery_target
        def probe(path, **kwargs):
            return real_probe(path, **kwargs) if online[0] else (False, offline_error())

        self.module.save_mosaic_switch = False
        self.module.save_timestamps_switch = False
        output = io.StringIO()
        with mock.patch.object(os, 'listdir', side_effect=listdir), \
             mock.patch.object(os, 'scandir', side_effect=scandir), \
             mock.patch.object(self.module, 'has_existing_artifacts', return_value=False), \
             mock.patch.object(self.module, '_get_processing_decision', return_value=('sample-md5', 'ready')), \
             mock.patch.object(self.module, '_video_duration', return_value=10), \
             mock.patch.object(self.module, '_try_claim_video', return_value=True), \
             mock.patch.object(self.module, 'detect_objects_in_video', side_effect=lambda *a, **kw: processed.append(True) or []), \
             mock.patch.object(self.module, '_mark_video_completed', return_value=True), \
             mock.patch.object(self.module, '_mark_directory_done', return_value=True), \
             mock.patch.object(self.module, '_release_claim_safely'), \
             mock.patch.object(self.module, '_pause_requested', return_value=False), \
             mock.patch.object(utils, 'probe_recovery_target', side_effect=probe), \
             redirect_stdout(output):
            count = utils.retry_with_recovery(
                lambda: self.module.process_directory_videos(str(self.root), 'face'),
                target_dir=str(self.root), pause_checker=lambda: False, wait_fn=waiter,
            )
        self.assertEqual(count, 1)
        self.assertEqual(processed, [True])
        self.assertEqual(sum(delays), 600)
        self.assertEqual(output.getvalue().count('存储仍未恢复'), 1)
        events = [json.loads(s) for s in (self.root / 'recovery.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(events[-1]['event_type'], 'recovery_resumed')

    def test_offline_frame_read_reopens_at_saved_checkpoint(self):
        module = self.module
        module._save_pipeline_checkpoint(str(self.video), next_frame=2, detections=[], last_detected=-5, reason='periodic', frame_video_segments=[])
        checkpoint = pathlib.Path(module._pipeline_checkpoint_path(str(self.video)))
        original = checkpoint.read_bytes()
        online = [True]
        attempts = []
        seeks = []
        real_scandir = os.scandir

        class Capture:
            def __init__(self):
                self.position = 0
                self.first_attempt = not attempts
                attempts.append(self)
            def isOpened(self): return True
            def get(self, prop):
                if prop == module.cv2.CAP_PROP_FPS: return 25
                if prop == module.cv2.CAP_PROP_FRAME_COUNT: return 4
                return self.position
            def set(self, prop, value):
                self.position = int(value)
                seeks.append(self.position)
                return True
            def read(self):
                if self.first_attempt:
                    online[0] = False
                    return False, None
                if self.position >= 4: return False, None
                self.position += 1
                return True, object()
            def release(self): pass

        class FrameVideo:
            requires_full_rebuild = False
            def __init__(self, *a, **kw): pass
            def seal_segment(self): return False
            def checkpoint_segments(self): return []
            def abort_active_segment(self): pass
            def finish(self): return []

        def scandir(path):
            if not online[0]: raise offline_error()
            return real_scandir(path)

        delays = []
        released = []
        def sleep(seconds):
            delays.append(seconds)
            if sum(delays) >= 300:
                self.assertEqual(checkpoint.read_bytes(), original)
                online[0] = True
        def waiter(failure, **kwargs):
            return utils.wait_for_recovery(failure, sleep_fn=sleep, **kwargs)

        with mock.patch.object(module, '_open_video_capture', side_effect=lambda _: Capture()), \
             mock.patch.object(module, 'ResumableFrameVideo', FrameVideo), \
             mock.patch.object(module, 'tqdm', return_value=FakeProgress()), \
             mock.patch.object(module, '_pause_requested', return_value=False), \
             mock.patch.object(os, 'scandir', side_effect=scandir):
            result = utils.retry_with_recovery(
                lambda: module.detect_objects_in_video(str(self.video), 'face', model=EmptyModel()),
                target_dir=str(self.root), pause_checker=lambda: False, wait_fn=waiter,
                release_claim=lambda: released.append(True), reacquire_claim=lambda: True,
            )
        self.assertEqual(result, [])
        self.assertEqual(seeks, [2, 2])
        self.assertEqual(len(attempts), 2)
        self.assertEqual(len(released), 1)
        self.assertEqual(sum(delays), 300)
        self.assertEqual(json.loads(checkpoint.read_text(encoding='utf-8'))['next_frame'], 4)

    def test_real_bad_video_keeps_early_eof_backoff_without_wait(self):
        module = self.module
        class Capture:
            position = 0
            def isOpened(self): return True
            def get(self, prop): return 25 if prop == module.cv2.CAP_PROP_FPS else 100
            def read(self):
                self.position += 1
                return (True, object()) if self.position == 1 else (False, None)
            def release(self): pass

        with mock.patch.object(module, '_open_video_capture', return_value=Capture()), \
             mock.patch.object(module, 'ResumableFrameVideo') as segments, \
             mock.patch.object(module, 'tqdm', return_value=FakeProgress()), \
             mock.patch.object(module, '_pause_requested', return_value=False), \
             mock.patch.object(module, 'wait_for_recovery') as wait:
            segments.return_value.requires_full_rebuild = False
            segments.return_value.checkpoint_segments.return_value = []
            with self.assertRaises(RuntimeError) as caught:
                module.detect_objects_in_video(str(self.video), 'face', model=EmptyModel())
        self.assertNotIsInstance(caught.exception, utils.RecoverableProcessingError)
        wait.assert_not_called()
        checkpoint = json.loads(pathlib.Path(module._pipeline_checkpoint_path(str(self.video))).read_text(encoding='utf-8'))
        self.assertEqual(checkpoint['reason'], 'unexpected_early_eof')
        self.assertEqual(checkpoint['early_eof_retry_count'], 1)
        self.assertGreater(checkpoint['retry_not_before'], checkpoint['saved_at'])

    def test_read_timeout_with_healthy_storage_remains_single_video_failure(self):
        with mock.patch.object(self.module, '_open_video_capture', side_effect=TimeoutError('坏解码器阻塞')):
            self.assertIsNone(self.module._video_duration(str(self.video)))
        event = json.loads((self.root / 'runtime.jsonl').read_text(encoding='utf-8').splitlines()[-1])
        self.assertEqual(event['exception_type'], 'TimeoutError')
        self.assertEqual(event['event_type'], 'video_failed')

    def test_missing_file_with_healthy_root_is_not_shared_host_outage(self):
        self.module._raise_if_storage_offline(str(self.root / 'missing.mp4'), verify_source=True)
        self.assertFalse((self.root / 'recovery.jsonl').exists())

    def test_unopenable_video_on_offline_share_does_not_become_skipped_candidate(self):
        with mock.patch.object(self.module, '_open_video_capture') as opened, \
             mock.patch.object(os, 'scandir', side_effect=offline_error()):
            opened.return_value.isOpened.return_value = False
            with self.assertRaises(utils.RecoverableProcessingError):
                self.module._video_duration(str(self.video))
        opened.return_value.release.assert_called_once()

    def test_shared_capture_uses_bounded_reader_and_local_capture_keeps_backend(self):
        import bounded_capture
        with mock.patch.object(bounded_capture, 'BoundedVideoCapture') as bounded, \
             mock.patch.object(self.module.cv2, 'VideoCapture') as local:
            path = r'\\host\share\sample.mp4'
            self.assertIs(self.module._open_video_capture(path), bounded.return_value)
            bounded.assert_called_once_with(path)
            local.assert_not_called()
        with mock.patch.object(self.module, 'windows_path_to_unc', return_value=None), \
             mock.patch.object(self.module.cv2, 'VideoCapture') as local:
            self.assertIs(self.module._open_video_capture(str(self.video)), local.return_value)

    def test_real_deleted_directory_still_updates_index_when_host_is_available(self):
        child = self.root / 'removed'
        child.mkdir()
        source = child / 'gone.mp4'
        source.write_bytes(b'video')
        self.index.refresh(str(self.root))
        source.unlink()
        child.rmdir()
        self.index.refresh(str(self.root), on_error=self.module._on_scan_error)
        self.assertIsNone(self.index.get_directory_info(str(child)))

    def test_cleanup_offline_guard_prevents_deleting_artifacts(self):
        artifact = self.root / 'gone.mp4.findinvideo-123abc_frames.mp4'
        artifact.write_bytes(b'keep-artifact')
        with mock.patch.object(os, 'scandir', side_effect=offline_error()):
            with self.assertRaises(utils.RecoverableProcessingError):
                utils.cleanup_orphan_artifacts(
                    str(self.root), directory_index=self.index,
                    check_storage=lambda: self.module._on_scan_error(None, str(self.root)),
                )
        self.assertTrue(artifact.exists())

    def test_segment_validation_propagates_offline_without_rebuilding_saved_output(self):
        artifact = self.root / 'saved_frames.mp4'
        artifact.write_bytes(b'saved-segment')
        failure = utils.RecoverableProcessingError('共享盘离线', stage='source_read')
        with mock.patch.object(self.module, '_open_video_capture', side_effect=failure):
            with self.assertRaises(utils.RecoverableProcessingError):
                utils.ResumableFrameVideo(
                    str(artifact), 25, self.module._StorageVideoIO(),
                    checkpoint={'next_frame': 5, 'detections': [0.1], 'frame_video_segments': [artifact.name]},
                )
        self.assertEqual(artifact.read_bytes(), b'saved-segment')


if __name__ == '__main__':
    unittest.main()
