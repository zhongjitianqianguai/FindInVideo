import importlib.util
import pathlib
import sys
import tempfile
import time
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / 'main.py'


def load_main_module():
    ultralytics = types.ModuleType('ultralytics')
    ultralytics.YOLO = type('FakeYOLO', (), {})
    sys.modules['ultralytics'] = ultralytics

    cv2 = types.ModuleType('cv2')
    cv2.CAP_PROP_FPS = 5
    cv2.CAP_PROP_FRAME_COUNT = 7
    cv2.CAP_PROP_POS_FRAMES = 1
    cv2.FONT_HERSHEY_SIMPLEX = 0
    cv2.VideoCapture = object
    cv2.VideoWriter = object
    cv2.VideoWriter_fourcc = lambda *args: 0
    cv2.destroyAllWindows = lambda: None
    cv2.rectangle = lambda *args, **kwargs: None
    cv2.putText = lambda *args, **kwargs: None
    cv2.imwrite = lambda *args, **kwargs: True
    sys.modules['cv2'] = cv2

    numpy = types.ModuleType('numpy')
    numpy.uint8 = 'uint8'
    numpy.zeros = lambda shape, dtype=None: {'shape': shape, 'dtype': dtype}
    sys.modules['numpy'] = numpy

    tqdm_mod = types.ModuleType('tqdm')
    tqdm_mod.tqdm = lambda *args, **kwargs: None
    sys.modules['tqdm'] = tqdm_mod

    module_name = f'findinvideo_main_test_{time.time_ns()}'
    spec = importlib.util.spec_from_file_location(module_name, MAIN_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProcessingStateTests(unittest.TestCase):
    def setUp(self):
        self.main_module = load_main_module()
        self.main_module.all_objects_switch = False
        self.main_module.save_mosaic_switch = False
        self.main_module.save_timestamps_switch = True
        self.main_module._LOGGER = types.SimpleNamespace(error=lambda *args, **kwargs: None)

    def test_pause_requested_respects_stop_flag(self):
        self.main_module._STOP_REQUESTED = True
        self.assertTrue(self.main_module._pause_requested())

    def test_claim_heartbeat_prevents_stale_reclaim(self):
        idx = self.main_module.DirectoryIndex(':memory:')
        original_ttl = self.main_module.os.environ.get('FINDINVIDEO_CLAIM_TTL_SECONDS')
        self.main_module.os.environ['FINDINVIDEO_CLAIM_TTL_SECONDS'] = '5'
        try:
            self.assertTrue(idx.try_claim_video('md5-a', 'video.mp4'))

            now = time.time()
            with idx.conn:
                idx.conn.execute(
                    '''
                    UPDATE processing_claims
                    SET claimed_at=?, heartbeat_at=?
                    WHERE file_md5=?
                    ''',
                    (now - 100, now, 'md5-a'),
                )
            self.assertTrue(idx.is_video_claimed('md5-a'))

            with idx.conn:
                idx.conn.execute(
                    'UPDATE processing_claims SET heartbeat_at=? WHERE file_md5=?',
                    (now - 100, 'md5-a'),
                )
            self.assertFalse(idx.is_video_claimed('md5-a'))
        finally:
            if original_ttl is None:
                self.main_module.os.environ.pop('FINDINVIDEO_CLAIM_TTL_SECONDS', None)
            else:
                self.main_module.os.environ['FINDINVIDEO_CLAIM_TTL_SECONDS'] = original_ttl
            idx.close()

    def test_mark_video_completed_writes_all_completion_state(self):
        recorded = []
        yoloed = []
        done = []

        class FakeDirectoryIndex:
            def mark_video_processed(self, **kwargs):
                recorded.append(kwargs)

        with mock.patch.object(self.main_module, 'DIRECTORY_INDEX', FakeDirectoryIndex()), \
             mock.patch.object(
                 self.main_module,
                 'append_yoloed_md5',
                 lambda md5, file_path=None: yoloed.append((md5, file_path)),
             ), \
             mock.patch.object(
                 self.main_module,
                 'write_done_marker',
                 lambda path: done.append(path),
             ):
            self.main_module._mark_video_completed('video.mp4', [1.0, 2.0], file_md5='md5-b')

        self.assertEqual(
            recorded,
            [
                {
                    'file_md5': 'md5-b',
                    'video_path': 'video.mp4',
                    'detection_count': 2,
                    'model_name': 'yolov11l-face',
                }
            ],
        )
        self.assertEqual(yoloed, [('md5-b', 'video.mp4')])
        self.assertEqual(done, ['video.mp4'])

    def test_save_checkpoint_includes_owner_and_progress_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = pathlib.Path(tmpdir) / 'video.mp4'
            video_path.write_bytes(b'data')

            self.main_module._save_checkpoint(
                str(video_path),
                next_frame=15,
                detections=[1.2],
                last_detected=1.2,
                claim_md5='md5-meta',
                last_success_frame=14,
            )

            payload = self.main_module._load_checkpoint(str(video_path))

        self.assertEqual(payload['next_frame'], 15)
        self.assertEqual(payload['last_success_frame'], 14)
        self.assertEqual(payload['checkpoint_owner']['claim_md5'], 'md5-meta')
        self.assertEqual(payload['checkpoint_owner']['pid'], self.main_module.os.getpid())
        self.assertIsNotNone(payload['claim_heartbeat_at'])

    def test_process_directory_videos_does_not_mark_directory_done_after_failure(self):
        released = []
        marked_dirs = []

        class FakeCapture:
            def isOpened(self):
                return True

            def get(self, prop):
                if prop == self_main.cv2.CAP_PROP_FPS:
                    return 30
                if prop == self_main.cv2.CAP_PROP_FRAME_COUNT:
                    return 300
                return 0

            def release(self):
                return None

        self_main = self.main_module
        with mock.patch.object(self_main.os, 'listdir', lambda path: ['video.mp4']), \
             mock.patch.object(self_main, 'is_video_file', lambda path: path.endswith('.mp4')), \
             mock.patch.object(self_main.cv2, 'VideoCapture', lambda path: FakeCapture()), \
             mock.patch.object(self_main, 'should_process', lambda path: 'md5-c'), \
             mock.patch.object(
                 self_main,
                 'detect_objects_in_video',
                 lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('boom')),
             ), \
             mock.patch.object(
                 self_main,
                 '_release_claim_safely',
                 lambda md5: released.append(md5),
             ), \
             mock.patch.object(
                 self_main,
                 '_mark_directory_done',
                 lambda dir_path, names: marked_dirs.append(dir_path),
             ):
            result = self_main.process_directory_videos('root', 'person')

        self.assertEqual(result, 0)
        self.assertEqual(released, ['md5-c'])
        self.assertEqual(marked_dirs, [])

    def test_process_directory_videos_releases_claim_on_pause(self):
        released = []

        class FakeCapture:
            def isOpened(self):
                return True

            def get(self, prop):
                if prop == self_main.cv2.CAP_PROP_FPS:
                    return 30
                if prop == self_main.cv2.CAP_PROP_FRAME_COUNT:
                    return 300
                return 0

            def release(self):
                return None

        self_main = self.main_module
        with mock.patch.object(self_main.os, 'listdir', lambda path: ['video.mp4']), \
             mock.patch.object(self_main, 'is_video_file', lambda path: path.endswith('.mp4')), \
             mock.patch.object(self_main.cv2, 'VideoCapture', lambda path: FakeCapture()), \
             mock.patch.object(self_main, 'should_process', lambda path: 'md5-d'), \
             mock.patch.object(
                 self_main,
                 'detect_objects_in_video',
                 lambda *args, **kwargs: (_ for _ in ()).throw(self_main.PauseRequested()),
             ), \
             mock.patch.object(
                 self_main,
                 '_release_claim_safely',
                 lambda md5: released.append(md5),
             ):
            with self.assertRaises(self_main.PauseRequested):
                self_main.process_directory_videos('root', 'person')

        self.assertEqual(released, ['md5-d'])

    def test_detect_objects_in_video_reraises_pause_requested(self):
        class FakeCapture:
            def isOpened(self):
                return True

            def get(self, prop):
                if prop == self_main.cv2.CAP_PROP_FPS:
                    return 30
                if prop == self_main.cv2.CAP_PROP_FRAME_COUNT:
                    return 300
                return 0

            def release(self):
                return None

        class FakeProgress:
            def close(self):
                return None

        class FakeYOLO:
            def __init__(self, *args, **kwargs):
                self.names = {}

        self_main = self.main_module
        with mock.patch.object(self_main, 'YOLO', FakeYOLO), \
             mock.patch.object(self_main.cv2, 'VideoCapture', lambda path: FakeCapture()), \
             mock.patch.object(self_main, 'tqdm', lambda *args, **kwargs: FakeProgress()), \
             mock.patch.object(self_main, '_pause_requested', lambda pause_file=None: True), \
             mock.patch.object(self_main, '_save_checkpoint', lambda *args, **kwargs: None), \
             mock.patch.object(self_main, '_load_checkpoint', lambda path: None), \
             mock.patch.object(self_main, '_truthy_env', lambda *args, **kwargs: True):
            with self.assertRaises(self_main.PauseRequested):
                self_main.detect_objects_in_video('video.mp4', 'person')

    def test_process_directory_videos_does_not_mark_completed_after_stop_requested(self):
        released = []
        completed = []
        pause_checks = iter([False, True])

        class FakeCapture:
            def isOpened(self):
                return True

            def get(self, prop):
                if prop == self_main.cv2.CAP_PROP_FPS:
                    return 30
                if prop == self_main.cv2.CAP_PROP_FRAME_COUNT:
                    return 300
                return 0

            def release(self):
                return None

        self_main = self.main_module
        with mock.patch.object(self_main.os, 'listdir', lambda path: ['video.mp4']), \
             mock.patch.object(self_main, 'is_video_file', lambda path: path.endswith('.mp4')), \
             mock.patch.object(self_main.cv2, 'VideoCapture', lambda path: FakeCapture()), \
             mock.patch.object(self_main, 'should_process', lambda path: 'md5-e'), \
             mock.patch.object(self_main, 'detect_objects_in_video', lambda *args, **kwargs: [1.0]), \
             mock.patch.object(self_main, '_pause_requested', lambda pause_file=None: next(pause_checks)), \
             mock.patch.object(
                 self_main,
                 '_release_claim_safely',
                 lambda md5: released.append(md5),
             ), \
             mock.patch.object(
                 self_main,
                 '_mark_video_completed',
                 lambda *args, **kwargs: completed.append(args),
             ):
            with self.assertRaises(self_main.PauseRequested):
                self_main.process_directory_videos('root', 'person')

        self.assertEqual(released, ['md5-e'])
        self.assertEqual(completed, [])

    def test_top_level_pause_prints_clean_stop_message(self):
        with mock.patch('builtins.print') as print_mock:
            try:
                raise self.main_module.PauseRequested()
            except self.main_module.PauseRequested:
                print("\n已按请求暂停，本轮处理已停止。")

        print_mock.assert_called_once_with("\n已按请求暂停，本轮处理已停止。")


if __name__ == '__main__':
    unittest.main()
