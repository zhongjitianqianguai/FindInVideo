import importlib.util
import os
import pathlib
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

from utils import pipeline_artifact_key


ROOT = pathlib.Path(__file__).resolve().parents[1]
IMAGE_ENTRYPOINTS = (ROOT / 'main_image.py', ROOT / 'main_image_yolov5.py')


def load_image_entrypoint(module_path):
    """使用轻量替身加载图片入口，避免测试依赖真实模型。"""
    cv2 = types.ModuleType('cv2')
    cv2.imread = lambda path: object()
    cv2.imwrite = lambda path, image: True
    cv2.rectangle = lambda *args, **kwargs: None
    cv2.putText = lambda *args, **kwargs: None
    cv2.cvtColor = lambda value, code: value
    cv2.FONT_HERSHEY_SIMPLEX = 0
    cv2.COLOR_RGB2BGR = 1

    numpy = types.ModuleType('numpy')
    ultralytics = types.ModuleType('ultralytics')
    ultralytics.YOLO = type('FakeYOLO', (), {})
    torch = types.ModuleType('torch')
    tqdm_module = types.ModuleType('tqdm')
    tqdm_module.tqdm = lambda iterable, **kwargs: iterable

    module_name = f'findinvideo_image_state_{module_path.stem}_{time.time_ns()}'
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            'cv2': cv2,
            'numpy': numpy,
            'torch': torch,
            'tqdm': tqdm_module,
            'ultralytics': ultralytics,
        },
    ):
        spec.loader.exec_module(module)
    return module


class ImageProcessingStateTests(unittest.TestCase):
    def test_failure_is_not_negative_cached(self):
        """读取或推理失败不得写成“未检测到”完成态。"""
        for module_path in IMAGE_ENTRYPOINTS:
            with self.subTest(entrypoint=module_path.name), tempfile.TemporaryDirectory() as tmpdir:
                module = load_image_entrypoint(module_path)
                image = pathlib.Path(tmpdir) / 'broken.jpg'
                image.write_bytes(b'image')

                with mock.patch.object(
                    module,
                    'detect_and_annotate',
                    side_effect=RuntimeError('无法读取图片'),
                ):
                    processed, detected = module.process_directory_images(
                        tmpdir, object(), pipeline_id='image:test'
                    )

                self.assertEqual((processed, detected), (0, 0))
                profile_dir = pathlib.Path(tmpdir) / '_detected' / pipeline_artifact_key('image:test')
                self.assertFalse((profile_dir / 'broken.jpg.done').exists())

    def test_pipeline_and_source_fingerprint_control_skip(self):
        """不同流水线互不跳过，源文件变化后旧 marker 必须失效。"""
        for module_path in IMAGE_ENTRYPOINTS:
            with self.subTest(entrypoint=module_path.name), tempfile.TemporaryDirectory() as tmpdir:
                module = load_image_entrypoint(module_path)
                image = pathlib.Path(tmpdir) / 'photo.jpg'
                image.write_bytes(b'first')
                detect = mock.Mock(return_value=(None, 0))

                with mock.patch.object(module, 'detect_and_annotate', detect):
                    self.assertEqual(
                        module.process_directory_images(
                            tmpdir, object(), pipeline_id='image:a'
                        ),
                        (1, 0),
                    )
                    self.assertEqual(
                        module.process_directory_images(
                            tmpdir, object(), pipeline_id='image:a'
                        ),
                        (0, 0),
                    )
                    self.assertEqual(
                        module.process_directory_images(
                            tmpdir, object(), pipeline_id='image:b'
                        ),
                        (1, 0),
                    )

                    time.sleep(0.002)
                    image.write_bytes(b'second-version')
                    os.utime(image, None)
                    self.assertEqual(
                        module.process_directory_images(
                            tmpdir, object(), pipeline_id='image:a'
                        ),
                        (1, 0),
                    )

                self.assertEqual(detect.call_count, 3)

    def test_same_stem_extensions_have_distinct_markers(self):
        """同名不同扩展名的图片不能共享负缓存。"""
        for module_path in IMAGE_ENTRYPOINTS:
            with self.subTest(entrypoint=module_path.name), tempfile.TemporaryDirectory() as tmpdir:
                module = load_image_entrypoint(module_path)
                pathlib.Path(tmpdir, 'same.jpg').write_bytes(b'jpg')
                pathlib.Path(tmpdir, 'same.png').write_bytes(b'png')

                with mock.patch.object(
                    module, 'detect_and_annotate', return_value=(None, 0)
                ):
                    self.assertEqual(
                        module.process_directory_images(
                            tmpdir, object(), pipeline_id='image:test'
                        ),
                        (2, 0),
                    )

                profile_dir = pathlib.Path(tmpdir) / '_detected' / pipeline_artifact_key('image:test')
                self.assertTrue((profile_dir / 'same.jpg.done').exists())
                self.assertTrue((profile_dir / 'same.png.done').exists())

    def test_failed_output_write_is_retried(self):
        """带框图片落盘失败时不得生成完成 marker。"""
        for module_path in IMAGE_ENTRYPOINTS:
            with self.subTest(entrypoint=module_path.name), tempfile.TemporaryDirectory() as tmpdir:
                module = load_image_entrypoint(module_path)
                pathlib.Path(tmpdir, 'photo.jpg').write_bytes(b'image')

                with mock.patch.object(
                    module, 'detect_and_annotate', return_value=(object(), 1)
                ), mock.patch.object(module.cv2, 'imwrite', return_value=False):
                    self.assertEqual(
                        module.process_directory_images(
                            tmpdir, object(), pipeline_id='image:test'
                        ),
                        (0, 0),
                    )

                profile_dir = pathlib.Path(tmpdir) / '_detected' / pipeline_artifact_key('image:test')
                self.assertFalse((profile_dir / 'photo.jpg.done').exists())

    def test_gc_uses_attempt_count_not_detection_count(self):
        """连续无检测结果不能在每张图片后触发完整 GC。"""
        for module_path in IMAGE_ENTRYPOINTS:
            with self.subTest(entrypoint=module_path.name), tempfile.TemporaryDirectory() as tmpdir:
                module = load_image_entrypoint(module_path)
                pathlib.Path(tmpdir, 'one.jpg').write_bytes(b'1')
                pathlib.Path(tmpdir, 'two.jpg').write_bytes(b'2')

                with mock.patch.object(
                    module, 'detect_and_annotate', return_value=(None, 0)
                ), mock.patch.object(module.gc, 'collect') as collect:
                    module.process_directory_images(
                        tmpdir, object(), pipeline_id='image:test'
                    )

                collect.assert_not_called()


if __name__ == '__main__':
    unittest.main()
