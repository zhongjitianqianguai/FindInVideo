import importlib.util
import pathlib
import sys
import tempfile
import time
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_YOLOV5_PATH = ROOT / 'main_yolov5.py'


class _Progress:
    def __init__(self, *args, **kwargs):
        self.updated = 0

    def update(self, amount):
        self.updated += amount

    def close(self):
        return None


def load_yolov5_module():
    cv2 = types.ModuleType('cv2')
    cv2.CAP_PROP_POS_FRAMES = 1
    cv2.CAP_PROP_POS_MSEC = 0
    cv2.CAP_PROP_FPS = 5
    cv2.CAP_PROP_FRAME_COUNT = 7
    cv2.FONT_HERSHEY_SIMPLEX = 0
    cv2.VideoCapture = object
    cv2.destroyAllWindows = lambda: None
    cv2.imshow = lambda *args, **kwargs: None
    cv2.waitKey = lambda *args, **kwargs: -1
    cv2.imwrite = lambda *args, **kwargs: True

    numpy = types.ModuleType('numpy')
    numpy.hstack = lambda values: values
    numpy.vstack = lambda values: values
    numpy.zeros_like = lambda value: value

    torch = types.ModuleType('torch')
    tqdm_module = types.ModuleType('tqdm')
    tqdm_module.tqdm = _Progress

    module_name = f'findinvideo_yolov5_test_{time.time_ns()}'
    spec = importlib.util.spec_from_file_location(module_name, MAIN_YOLOV5_PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            'cv2': cv2,
            'numpy': numpy,
            'torch': torch,
            'tqdm': tqdm_module,
        },
    ):
        spec.loader.exec_module(module)
    module._LOGGER = types.SimpleNamespace(
        error=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
    )
    return module


class _FrameCapture:
    def __init__(self, frame_count=3, opened=True):
        self.frame_count = frame_count
        self.opened = opened
        self.position = 0
        self.released = False

    def isOpened(self):
        return self.opened and self.position <= self.frame_count

    def read(self):
        if self.position >= self.frame_count:
            return False, None
        self.position += 1
        return True, object()

    def get(self, prop):
        if prop == 5:
            return 10
        if prop == 7:
            return self.frame_count
        return 0

    def set(self, prop, value):
        self.position = int(value)
        return True

    def release(self):
        self.released = True


class YoloV5ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.module = load_yolov5_module()

    def test_detectpy_uses_one_based_non_padded_label_names(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = pathlib.Path(tmpdir) / 'sample.mp4'
            video_path.write_bytes(b'video')
            labels_dir = pathlib.Path(tmpdir) / 'yolov5_output' / 'exp' / 'labels'
            labels_dir.mkdir(parents=True)
            (labels_dir / 'sample_1.txt').write_text('0 0.5 0.5 0.2 0.2\n', encoding='utf-8')
            (labels_dir / 'sample_3.txt').write_text('0 0.5 0.5 0.2 0.2\n', encoding='utf-8')
            capture = _FrameCapture(frame_count=3)
            refreshed = []
            fake_index = types.SimpleNamespace(
                refresh_claim=lambda md5: refreshed.append(md5) or True
            )

            with mock.patch.object(
                self.module,
                '_run_detect_command',
                return_value=types.SimpleNamespace(returncode=0, stdout='', stderr=''),
            ), mock.patch.object(
                self.module.cv2, 'VideoCapture', return_value=capture
            ), mock.patch.object(
                self.module,
                '_read_frame_with_timeout',
                side_effect=lambda cap: (*cap.read(), False),
            ), mock.patch.object(
                self.module, '_pause_requested', return_value=False
            ), mock.patch.object(
                self.module, '_load_checkpoint', return_value=None
            ), mock.patch.object(
                self.module, '_clear_checkpoint'
            ), mock.patch.object(
                self.module, 'DIRECTORY_INDEX', fake_index
            ):
                result = self.module.detect_objects_with_frame_analysis(
                    str(video_path),
                    'person',
                    claim_md5='md5-sample',
                )

            self.assertTrue(result.success)
            self.assertEqual(result.detections, [0.0, 0.2])
            self.assertGreaterEqual(len(refreshed), 2)
            self.assertTrue(capture.released)

    def test_detectpy_subprocess_refreshes_claim_while_running(self):
        refreshed = []
        module = self.module

        class FakeProcess:
            def __init__(self):
                self.args = ['python', 'detect.py']
                self.returncode = 0
                self.communicate_calls = 0

            def communicate(self, timeout=None):
                self.communicate_calls += 1
                if self.communicate_calls == 1:
                    raise module.subprocess.TimeoutExpired(self.args, timeout)
                return 'stdout', 'stderr'

            def terminate(self):
                return None

            def kill(self):
                return None

        process = FakeProcess()
        fake_index = types.SimpleNamespace(
            refresh_claim=lambda md5: refreshed.append(md5) or True
        )
        with mock.patch.object(
            self.module.subprocess, 'Popen', return_value=process
        ), mock.patch.object(
            self.module, 'DIRECTORY_INDEX', fake_index
        ):
            result = self.module._run_detect_command(
                ['python', 'detect.py'],
                cwd='.',
                claim_md5='md5-sample',
                video_path='sample.mp4',
                heartbeat_interval=0.01,
            )

        self.assertEqual(result.returncode, 0)
        self.assertEqual(refreshed, ['md5-sample'])

    def test_detectpy_subprocess_stops_on_pause_request(self):
        module = self.module

        class FakeProcess:
            def __init__(self):
                self.returncode = None
                self.terminated = False

            def communicate(self, timeout=None):
                if self.terminated:
                    return '', ''
                raise module.subprocess.TimeoutExpired(['python'], timeout)

            def terminate(self):
                self.terminated = True

            def kill(self):
                self.terminated = True

        process = FakeProcess()
        fake_index = types.SimpleNamespace(refresh_claim=lambda md5: False)
        with mock.patch.object(
            self.module.subprocess, 'Popen', return_value=process
        ), mock.patch.object(
            self.module, 'DIRECTORY_INDEX', fake_index
        ), mock.patch.object(
            self.module, '_pause_requested', return_value=True
        ):
            with self.assertRaises(self.module.PauseRequested):
                self.module._run_detect_command(
                    ['python', 'detect.py'],
                    cwd='.',
                    claim_md5='md5-sample',
                    video_path='sample.mp4',
                    heartbeat_interval=0.01,
                )

        self.assertTrue(process.terminated)

    def test_detection_failures_are_explicit_results(self):
        with mock.patch.object(
            self.module,
            'load_yolov5_model',
            side_effect=RuntimeError('模型损坏'),
        ):
            model_result = self.module.detect_objects_in_video_yolov5(
                'video.mp4', 'person'
            )

        closed_capture = _FrameCapture(opened=False)
        with mock.patch.object(
            self.module,
            'load_yolov5_model',
            return_value=(object(), object(), None, None),
        ), mock.patch.object(
            self.module.cv2,
            'VideoCapture',
            return_value=closed_capture,
        ):
            video_result = self.module.detect_objects_in_video_yolov5(
                'video.mp4', 'person'
            )

        with mock.patch.object(
            self.module.subprocess,
            'run',
            return_value=types.SimpleNamespace(
                returncode=2, stdout='', stderr='detect failed'
            ),
        ):
            detectpy_result = self.module.detect_objects_with_frame_analysis(
                'video.mp4', 'person'
            )

        for result in (model_result, video_result, detectpy_result):
            with self.subTest(result=result):
                self.assertFalse(result.success)
                self.assertEqual(result.detections, [])
                self.assertTrue(result.error)
        self.assertTrue(closed_capture.released)

    def test_partial_artifacts_do_not_count_as_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = pathlib.Path(tmpdir) / 'sample.mp4'
            video_path.write_bytes(b'video')
            artifact_base = self.module.safe_artifact_basename(str(video_path))
            (pathlib.Path(tmpdir) / f'{artifact_base}.txt').write_text(
                'partial', encoding='utf-8'
            )
            (pathlib.Path(tmpdir) / f'{artifact_base}_mosaic.jpg').write_bytes(
                b'partial'
            )

            self.assertFalse(self.module.has_existing_artifacts(str(video_path)))

            (pathlib.Path(tmpdir) / f'{artifact_base}.done').write_text(
                'done\n', encoding='utf-8'
            )
            self.assertTrue(self.module.has_existing_artifacts(str(video_path)))

    def test_checkpoint_bypasses_legacy_yoloed_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = pathlib.Path(tmpdir) / 'sample.mp4'
            video_path.write_bytes(b'video')
            pathlib.Path(self.module._checkpoint_path(str(video_path))).write_text(
                '{}', encoding='utf-8'
            )
            with mock.patch.object(
                self.module, 'get_file_md5_cached', return_value='md5-sample'
            ), mock.patch.object(
                self.module, 'load_yoloed_md5', return_value={'md5-sample'}
            ):
                self.assertTrue(self.module.should_process(str(video_path)))

    def test_directory_failure_never_writes_completion_state(self):
        events = []

        class FakeDirectoryIndex:
            def try_claim_video(self, md5, path):
                events.append(('claim', md5, path))
                return True

            def refresh_claim(self, md5):
                events.append(('refresh', md5))
                return True

            def complete_claimed_video(self, **kwargs):
                events.append(('complete', kwargs))
                return True

            def release_claim(self, md5):
                events.append(('release', md5))
                return True

        failed = self.module.VideoProcessingResult.failed('detect.py 失败')
        appended = []
        markers = []
        with mock.patch.object(
            self.module.os, 'listdir', return_value=['sample.mp4']
        ), mock.patch.object(
            self.module, 'is_video_file', return_value=True
        ), mock.patch.object(
            self.module, 'should_process', return_value=(True, None)
        ), mock.patch.object(
            self.module, 'get_file_md5_cached', return_value='md5-sample'
        ), mock.patch.object(
            self.module, 'DIRECTORY_INDEX', FakeDirectoryIndex()
        ), mock.patch.object(
            self.module,
            'detect_objects_with_frame_analysis',
            return_value=failed,
        ), mock.patch.object(
            self.module,
            'append_yoloed_md5',
            side_effect=lambda *args, **kwargs: appended.append((args, kwargs)),
        ), mock.patch.object(
            self.module,
            'write_done_marker',
            side_effect=lambda path: markers.append(path),
        ):
            completed = self.module.process_directory_videos(
                'root', 'person', use_detectpy=True
            )

        self.assertEqual(completed, 0)
        self.assertFalse(any(event[0] == 'complete' for event in events))
        self.assertEqual(appended, [])
        self.assertEqual(markers, [])
        self.assertEqual(events[-1], ('release', 'md5-sample'))

    def test_directory_success_consumes_claim_before_writing_markers(self):
        events = []
        detector_calls = []

        class FakeDirectoryIndex:
            def try_claim_video(self, md5, path):
                events.append(('claim', md5, path))
                return True

            def refresh_claim(self, md5):
                events.append(('refresh', md5))
                return True

            def complete_claimed_video(self, **kwargs):
                events.append(('complete', kwargs))
                return True

            def release_claim(self, md5):
                events.append(('release', md5))
                return True

        def detect(*args, **kwargs):
            detector_calls.append((args, kwargs))
            return self.module.VideoProcessingResult.succeeded([1.0, 2.0])

        with mock.patch.object(
            self.module.os, 'listdir', return_value=['sample.mp4']
        ), mock.patch.object(
            self.module, 'is_video_file', return_value=True
        ), mock.patch.object(
            self.module, 'should_process', return_value=(True, None)
        ), mock.patch.object(
            self.module, 'get_file_md5_cached', return_value='md5-sample'
        ), mock.patch.object(
            self.module, 'DIRECTORY_INDEX', FakeDirectoryIndex()
        ), mock.patch.object(
            self.module, 'detect_objects_with_frame_analysis', side_effect=detect
        ), mock.patch.object(
            self.module,
            'append_yoloed_md5',
            side_effect=lambda *args, **kwargs: events.append(('yoloed', args, kwargs)),
        ), mock.patch.object(
            self.module,
            'write_done_marker',
            side_effect=lambda path: events.append(('done', path)) or True,
        ), mock.patch.object(
            self.module, '_clear_checkpoint'
        ):
            completed = self.module.process_directory_videos(
                'root', 'person', model_path='models/test.pt', use_detectpy=True
            )

        self.assertEqual(completed, 1)
        self.assertEqual(detector_calls[0][1]['claim_md5'], 'md5-sample')
        complete_event = next(event for event in events if event[0] == 'complete')
        self.assertEqual(complete_event[1]['detection_count'], 2)
        self.assertEqual(complete_event[1]['model_name'], 'models/test.pt')
        self.assertLess(
            next(i for i, event in enumerate(events) if event[0] == 'complete'),
            next(i for i, event in enumerate(events) if event[0] == 'done'),
        )
        self.assertEqual(events[-1], ('release', 'md5-sample'))

    def test_explicit_pipeline_uses_one_snapshot_for_claim_completion_and_marker(self):
        pipeline_id = 'yolov5:test-model:profile'
        events = []

        class FakeDirectoryIndex:
            def try_claim_video(self, md5, path, **kwargs):
                events.append(('claim', md5, path, kwargs))
                return True

            def complete_claimed_video(self, **kwargs):
                events.append(('complete', kwargs))
                return True

            def release_claim(self, md5, **kwargs):
                events.append(('release', md5, kwargs))
                return True

        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = pathlib.Path(tmpdir) / 'sample.mp4'
            video_path.write_bytes(b'video')
            success = self.module.VideoProcessingResult.succeeded([1.0])
            with mock.patch.object(
                self.module, 'should_process', return_value=(True, None)
            ), mock.patch.object(
                self.module, 'get_file_md5_cached', return_value='md5-sample'
            ), mock.patch.object(
                self.module, 'DIRECTORY_INDEX', FakeDirectoryIndex()
            ), mock.patch.object(
                self.module,
                'detect_objects_with_frame_analysis',
                return_value=success,
            ), mock.patch.object(
                self.module, 'append_yoloed_md5'
            ) as append_yoloed:
                completed = self.module.process_directory_videos(
                    tmpdir,
                    'person',
                    model_path='models/test.pt',
                    use_detectpy=True,
                    pipeline_id=pipeline_id,
                )

            self.assertEqual(completed, 1)
            claim_event = next(event for event in events if event[0] == 'claim')
            complete_event = next(event for event in events if event[0] == 'complete')
            self.assertEqual(claim_event[3]['pipeline_id'], pipeline_id)
            self.assertEqual(complete_event[1]['pipeline_id'], pipeline_id)
            self.assertEqual(
                claim_event[3]['source_snapshot'],
                complete_event[1]['source_snapshot'],
            )
            marker_base = self.module.safe_artifact_basename(
                str(video_path), pipeline_id=pipeline_id
            )
            marker_path = video_path.parent / f'{marker_base}.done'
            self.assertTrue(marker_path.exists())
            append_yoloed.assert_not_called()
            self.assertEqual(events[-1][0], 'release')
            self.assertEqual(events[-1][2]['pipeline_id'], pipeline_id)

    def test_marker_failure_rolls_back_pipeline_completion(self):
        pipeline_id = 'yolov5:test-model:profile'
        source_snapshot = {'size': 5, 'mtime_ns': 7}
        events = []

        class FakeDirectoryIndex:
            def complete_claimed_video(self, **kwargs):
                events.append(('complete', kwargs))
                return True

            def rollback_video_completion(self, md5, **kwargs):
                events.append(('rollback', md5, kwargs))
                return True

        with mock.patch.object(
            self.module, 'DIRECTORY_INDEX', FakeDirectoryIndex()
        ), mock.patch.object(
            self.module, 'write_done_marker', return_value=False
        ), mock.patch.object(
            self.module, 'append_yoloed_md5'
        ) as append_yoloed:
            marked = self.module._mark_video_completed(
                'sample.mp4',
                self.module.VideoProcessingResult.succeeded([]),
                file_md5='md5-sample',
                model_path='models/test.pt',
                pipeline_id=pipeline_id,
                source_snapshot=source_snapshot,
            )

        self.assertFalse(marked)
        self.assertEqual(events[-1][0], 'rollback')
        self.assertEqual(events[-1][2]['pipeline_id'], pipeline_id)
        append_yoloed.assert_not_called()

    def test_explicit_pipeline_never_uses_legacy_yoloed_skip(self):
        pipeline_id = 'yolov5:test-model:profile'
        fake_index = types.SimpleNamespace(
            is_video_processed_by_md5=lambda md5, **kwargs: False
        )
        with mock.patch.object(
            self.module, 'has_existing_artifacts', return_value=False
        ), mock.patch.object(
            self.module, 'get_file_md5_cached', return_value='md5-sample'
        ), mock.patch.object(
            self.module, 'DIRECTORY_INDEX', fake_index
        ), mock.patch.object(
            self.module,
            'load_yoloed_md5',
            side_effect=AssertionError('显式流水线不得读取旧 yoloed'),
        ):
            decision = self.module.should_process(
                'sample.mp4', pipeline_id=pipeline_id
            )

        self.assertTrue(decision)

    def test_lost_claim_never_writes_compatibility_markers(self):
        events = []

        class FakeDirectoryIndex:
            def try_claim_video(self, md5, path):
                events.append(('claim', md5, path))
                return True

            def complete_claimed_video(self, **kwargs):
                events.append(('complete', kwargs))
                return False

            def release_claim(self, md5):
                events.append(('release', md5))
                return True

        appended = []
        markers = []
        success = self.module.VideoProcessingResult.succeeded([])
        with mock.patch.object(
            self.module.os, 'listdir', return_value=['sample.mp4']
        ), mock.patch.object(
            self.module, 'is_video_file', return_value=True
        ), mock.patch.object(
            self.module, 'should_process', return_value=(True, None)
        ), mock.patch.object(
            self.module, 'get_file_md5_cached', return_value='md5-sample'
        ), mock.patch.object(
            self.module, 'DIRECTORY_INDEX', FakeDirectoryIndex()
        ), mock.patch.object(
            self.module,
            'detect_objects_with_frame_analysis',
            return_value=success,
        ), mock.patch.object(
            self.module,
            'append_yoloed_md5',
            side_effect=lambda *args, **kwargs: appended.append((args, kwargs)),
        ), mock.patch.object(
            self.module,
            'write_done_marker',
            side_effect=lambda path: markers.append(path),
        ):
            completed = self.module.process_directory_videos(
                'root', 'person', use_detectpy=True
            )

        self.assertEqual(completed, 0)
        self.assertEqual(appended, [])
        self.assertEqual(markers, [])
        self.assertEqual(events[-1], ('release', 'md5-sample'))

    def test_pause_propagates_and_releases_claim_without_completion(self):
        events = []

        class FakeDirectoryIndex:
            def try_claim_video(self, md5, path):
                events.append(('claim', md5, path))
                return True

            def complete_claimed_video(self, **kwargs):
                events.append(('complete', kwargs))
                return True

            def release_claim(self, md5):
                events.append(('release', md5))
                return True

        with mock.patch.object(
            self.module.os, 'listdir', return_value=['sample.mp4']
        ), mock.patch.object(
            self.module, 'is_video_file', return_value=True
        ), mock.patch.object(
            self.module, 'should_process', return_value=(True, None)
        ), mock.patch.object(
            self.module, 'get_file_md5_cached', return_value='md5-sample'
        ), mock.patch.object(
            self.module, 'DIRECTORY_INDEX', FakeDirectoryIndex()
        ), mock.patch.object(
            self.module,
            'detect_objects_with_frame_analysis',
            side_effect=self.module.PauseRequested(),
        ), mock.patch.object(
            self.module, 'append_yoloed_md5'
        ), mock.patch.object(
            self.module, 'write_done_marker'
        ):
            with self.assertRaises(self.module.PauseRequested):
                self.module.process_directory_videos(
                    'root', 'person', use_detectpy=True
                )

        self.assertFalse(any(event[0] == 'complete' for event in events))
        self.assertEqual(events[-1], ('release', 'md5-sample'))


if __name__ == '__main__':
    unittest.main()
