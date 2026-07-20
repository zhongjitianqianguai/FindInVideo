import importlib.util
import os
import pathlib
import sys
import tempfile
import time
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
ENTRYPOINTS = (
    (ROOT / 'main.py', False),
    (ROOT / 'main_nipple.py', True),
    (ROOT / 'main_nipple_rog.py', True),
)
PIPELINE_ID = 'yolov11:test-pipeline'


def load_entrypoint(module_path):
    """使用轻量替身加载视频入口，避免测试依赖真实推理环境。"""
    ultralytics = types.ModuleType('ultralytics')
    ultralytics.YOLO = type('FakeYOLO', (), {})

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

    numpy = types.ModuleType('numpy')
    numpy.uint8 = 'uint8'
    numpy.zeros = lambda shape, dtype=None: {'shape': shape, 'dtype': dtype}

    tqdm_module = types.ModuleType('tqdm')
    tqdm_module.tqdm = lambda *args, **kwargs: None

    module_name = f'findinvideo_completion_guard_{module_path.stem}_{time.time_ns()}'
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            'ultralytics': ultralytics,
            'cv2': cv2,
            'numpy': numpy,
            'tqdm': tqdm_module,
        },
    ):
        spec.loader.exec_module(module)
    return module


class FakeProgress:
    def update(self, value):
        return None

    def close(self):
        return None


class EmptyModel:
    names = {}

    def predict(self, *args, **kwargs):
        return []


class VideoCompletionGuardTests(unittest.TestCase):
    def _detect(self, module, video_path, capture):
        args = [str(video_path), 'person']
        if module.__file__.endswith(('main_nipple.py', 'main_nipple_rog.py')):
            args.append(EmptyModel())
        with mock.patch.object(module.cv2, 'VideoCapture', lambda path: capture), \
             mock.patch.object(module, 'tqdm', lambda *args, **kwargs: FakeProgress()), \
             mock.patch.object(module, '_pause_requested', return_value=False):
            if module.__file__.endswith('main.py'):
                return module.detect_objects_in_video(*args, model=EmptyModel())
            return module.detect_objects_in_video(*args)

    def test_pipeline_decision_ignores_legacy_yoloed_caches(self):
        """非 legacy pipeline 不得被共享路径、basename 或 MD5 缓存跳过。"""
        for module_path, _needs_model in ENTRYPOINTS:
            with self.subTest(entrypoint=module_path.name), tempfile.TemporaryDirectory() as tmpdir:
                module = load_entrypoint(module_path)
                module._ACTIVE_PIPELINE_ID = PIPELINE_ID
                video_path = pathlib.Path(tmpdir) / 'video.mp4'
                video_path.write_bytes(b'video')
                old = time.time() - 60
                os.utime(video_path, (old, old))

                observed = []

                class FakeIndex:
                    def is_video_processed_by_md5(self, md5, **kwargs):
                        observed.append(('processed', md5, kwargs))
                        return False

                    def is_video_claimed(self, md5):
                        return False

                module._YOLOED_BASENAME_CACHE = {'video.mp4': {video_path.parent.name}}
                with mock.patch.object(module, 'DIRECTORY_INDEX', FakeIndex()), \
                     mock.patch.object(module, 'has_existing_artifacts', return_value=False), \
                     mock.patch.object(module, 'get_file_md5_cached', return_value='same-md5'), \
                     mock.patch.object(
                         module,
                         'is_path_already_yoloed',
                         side_effect=AssertionError('新 pipeline 不应查询旧路径缓存'),
                     ), mock.patch.object(
                         module,
                         'load_yoloed_md5',
                         side_effect=AssertionError('新 pipeline 不应查询旧 MD5 缓存'),
                     ):
                    decision = module._get_processing_decision(
                        str(video_path), acquire_claim=False
                    )

                self.assertEqual(decision, ('same-md5', 'ready'))
                self.assertEqual(
                    observed,
                    [('processed', 'same-md5', {'pipeline_id': PIPELINE_ID})],
                )

    def test_fresh_or_changed_source_is_not_processed(self):
        """刚写入或在 MD5 期间变化的视频必须返回 source_unstable。"""
        for module_path, _needs_model in ENTRYPOINTS:
            with self.subTest(entrypoint=module_path.name), tempfile.TemporaryDirectory() as tmpdir:
                module = load_entrypoint(module_path)
                module._ACTIVE_PIPELINE_ID = PIPELINE_ID
                video_path = pathlib.Path(tmpdir) / 'video.mp4'
                video_path.write_bytes(b'fresh')

                with mock.patch.object(
                    module,
                    'get_file_md5_cached',
                    side_effect=AssertionError('过新的文件不应开始计算 MD5'),
                ):
                    self.assertEqual(
                        module._get_processing_decision(
                            str(video_path), acquire_claim=False
                        ),
                        (None, 'source_unstable'),
                    )

                old = time.time() - 60
                os.utime(video_path, (old, old))

                def mutate_source(path):
                    pathlib.Path(path).write_bytes(b'changed-during-hash')
                    return 'stale-md5'

                with mock.patch.object(
                    module, 'get_file_md5_cached', side_effect=mutate_source
                ):
                    self.assertEqual(
                        module._get_processing_decision(
                            str(video_path), acquire_claim=False
                        ),
                        (None, 'source_unstable'),
                    )

    def test_failed_seek_and_early_eof_are_not_reported_as_success(self):
        """断点定位失败或已知总帧下提前读失败时必须抛错。"""
        for module_path, _needs_model in ENTRYPOINTS:
            with self.subTest(entrypoint=f'{module_path.name}:seek'), tempfile.TemporaryDirectory() as tmpdir:
                module = load_entrypoint(module_path)
                module._ACTIVE_PIPELINE_ID = PIPELINE_ID
                video_path = pathlib.Path(tmpdir) / 'video.mp4'
                video_path.write_bytes(b'video')

                class SeekFailureCapture:
                    def isOpened(self):
                        return True

                    def get(self, prop):
                        return 30 if prop == module.cv2.CAP_PROP_FPS else 10

                    def set(self, prop, value):
                        return False

                    def release(self):
                        return None

                with mock.patch.object(
                    module, '_load_checkpoint', return_value={'next_frame': 2}
                ), self.assertRaisesRegex(RuntimeError, '断点'):
                    self._detect(module, video_path, SeekFailureCapture())

            with self.subTest(entrypoint=f'{module_path.name}:eof'), tempfile.TemporaryDirectory() as tmpdir:
                module = load_entrypoint(module_path)
                module._ACTIVE_PIPELINE_ID = PIPELINE_ID
                video_path = pathlib.Path(tmpdir) / 'video.mp4'
                video_path.write_bytes(b'video')

                class EarlyEofCapture:
                    def __init__(self):
                        self.read_count = 0

                    def isOpened(self):
                        return True

                    def get(self, prop):
                        return 30 if prop == module.cv2.CAP_PROP_FPS else 5

                    def read(self):
                        self.read_count += 1
                        return (True, object()) if self.read_count == 1 else (False, None)

                    def release(self):
                        return None

                with mock.patch.object(module, '_load_checkpoint', return_value=None), \
                     self.assertRaisesRegex(RuntimeError, '提前结束'):
                    self._detect(module, video_path, EarlyEofCapture())

    def test_unknown_total_frame_count_allows_normal_eof(self):
        """无法获知总帧数时，read=False 仍是合法 EOF，兼容流式输入和测试替身。"""
        for module_path, _needs_model in ENTRYPOINTS:
            with self.subTest(entrypoint=module_path.name), tempfile.TemporaryDirectory() as tmpdir:
                module = load_entrypoint(module_path)
                video_path = pathlib.Path(tmpdir) / 'video.mp4'
                video_path.write_bytes(b'video')

                class UnknownLengthCapture:
                    def isOpened(self):
                        return True

                    def get(self, prop):
                        return 30 if prop == module.cv2.CAP_PROP_FPS else 0

                    def read(self):
                        return False, None

                    def release(self):
                        return None

                with mock.patch.object(module, '_load_checkpoint', return_value=None):
                    self.assertEqual(
                        self._detect(module, video_path, UnknownLengthCapture()), []
                    )

    def test_marker_failure_rolls_back_pipeline_completion(self):
        """数据库完成后 marker 写入失败时必须撤销同一 pipeline 的完成记录。"""
        for module_path, _needs_model in ENTRYPOINTS:
            with self.subTest(entrypoint=module_path.name):
                module = load_entrypoint(module_path)
                module._ACTIVE_PIPELINE_ID = PIPELINE_ID
                calls = []

                class FakeIndex:
                    def complete_claimed_video(self, **kwargs):
                        calls.append(('complete', kwargs))
                        return True

                    def rollback_video_completion(self, md5, **kwargs):
                        calls.append(('rollback', md5, kwargs))
                        return True

                snapshot = {'size': 5, 'mtime_ns': 123}
                with mock.patch.object(module, 'DIRECTORY_INDEX', FakeIndex()), \
                     mock.patch.object(module, 'write_done_marker', return_value=False), \
                     mock.patch.object(module, '_clear_checkpoint') as clear_mock:
                    self.assertFalse(
                        module._mark_video_completed(
                            'video.mp4',
                            [1.0],
                            file_md5='video-md5',
                            source_snapshot=snapshot,
                        )
                    )

                complete_kwargs = calls[0][1]
                self.assertEqual(complete_kwargs['pipeline_id'], PIPELINE_ID)
                self.assertEqual(complete_kwargs['source_snapshot'], snapshot)
                self.assertEqual(
                    calls[1],
                    ('rollback', 'video-md5', {'pipeline_id': PIPELINE_ID}),
                )
                clear_mock.assert_not_called()


if __name__ == '__main__':
    unittest.main()
