import importlib.util
import json
import os
import pathlib
import sys
import tempfile
import time
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / 'main.py'
PIPELINE_FACE = 'video:yolov11l-face:v1'
PIPELINE_CUSTOM = 'video:custom-nipple:v1'


def load_main_module():
    """在不加载真实 YOLO、OpenCV 和 NumPy 的情况下导入主入口。"""
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

    module_name = f'findinvideo_artifact_identity_{time.time_ns()}'
    spec = importlib.util.spec_from_file_location(module_name, MAIN_PATH)
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


class ArtifactIdentityTests(unittest.TestCase):
    def setUp(self):
        self.main_module = load_main_module()

    def _done_marker_path(self, video_path, pipeline_id):
        base = self.main_module.safe_artifact_basename(
            str(video_path), pipeline_id=pipeline_id
        )
        return video_path.parent / f'{base}{self.main_module.DONE_SUFFIX}'

    def test_same_stem_with_different_extensions_has_distinct_artifacts_and_checkpoints(self):
        """同目录同 stem 的不同视频文件不得共用任何衍生文件身份。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            mp4_path = root / 'foo.mp4'
            mkv_path = root / 'foo.mkv'
            mp4_path.write_bytes(b'mp4')
            mkv_path.write_bytes(b'mkv')

            mp4_base = self.main_module.safe_artifact_basename(
                str(mp4_path), pipeline_id=PIPELINE_FACE
            )
            mkv_base = self.main_module.safe_artifact_basename(
                str(mkv_path), pipeline_id=PIPELINE_FACE
            )
            self.assertNotEqual(mp4_base, mkv_base)

            mp4_artifacts = {
                f'{mp4_base}.txt',
                f'{mp4_base}_frames.mp4',
                f'{mp4_base}_mosaic.jpg',
                f'{mp4_base}.done',
            }
            mkv_artifacts = {
                f'{mkv_base}.txt',
                f'{mkv_base}_frames.mp4',
                f'{mkv_base}_mosaic.jpg',
                f'{mkv_base}.done',
            }
            self.assertTrue(mp4_artifacts.isdisjoint(mkv_artifacts))
            self.assertNotEqual(
                self.main_module._checkpoint_path(
                    str(mp4_path), pipeline_id=PIPELINE_FACE
                ),
                self.main_module._checkpoint_path(
                    str(mkv_path), pipeline_id=PIPELINE_FACE
                ),
            )

    def test_done_marker_is_json_and_invalidates_after_source_size_or_mtime_changes(self):
        """完成标记必须绑定源文件大小和修改时间，源文件变化后不得继续命中。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = pathlib.Path(tmpdir) / 'video.mp4'
            video_path.write_bytes(b'first-version')

            self.main_module.write_done_marker(
                str(video_path), pipeline_id=PIPELINE_FACE
            )
            marker_path = self._done_marker_path(video_path, PIPELINE_FACE)
            payload = json.loads(marker_path.read_text(encoding='utf-8'))
            self.assertIsInstance(payload, dict)
            self.assertEqual(payload.get('pipeline_id'), PIPELINE_FACE)
            self.assertTrue(
                self.main_module.has_existing_artifacts(
                    str(video_path), pipeline_id=PIPELINE_FACE
                )
            )

            video_path.write_bytes(b'a-different-and-longer-version')
            self.assertFalse(
                self.main_module.has_existing_artifacts(
                    str(video_path), pipeline_id=PIPELINE_FACE
                ),
                '源文件大小变化后旧完成标记必须失效',
            )

            marker_path.unlink()
            self.main_module.write_done_marker(
                str(video_path), pipeline_id=PIPELINE_FACE
            )
            self.assertTrue(
                self.main_module.has_existing_artifacts(
                    str(video_path), pipeline_id=PIPELINE_FACE
                )
            )
            stat_result = video_path.stat()
            os.utime(
                video_path,
                ns=(stat_result.st_atime_ns, stat_result.st_mtime_ns + 5_000_000_000),
            )
            self.assertFalse(
                self.main_module.has_existing_artifacts(
                    str(video_path), pipeline_id=PIPELINE_FACE
                ),
                '源文件修改时间变化后旧完成标记必须失效',
            )

    def test_done_marker_and_database_completion_are_isolated_by_pipeline(self):
        """同一源文件在不同检测 pipeline 下必须拥有独立完成态。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = pathlib.Path(tmpdir) / 'video.mp4'
            video_path.write_bytes(b'video')

            face_base = self.main_module.safe_artifact_basename(
                str(video_path), pipeline_id=PIPELINE_FACE
            )
            custom_base = self.main_module.safe_artifact_basename(
                str(video_path), pipeline_id=PIPELINE_CUSTOM
            )
            self.assertNotEqual(
                face_base,
                custom_base,
                '不同 pipeline 的普通输出产物必须能够同时存在',
            )
            self.assertNotEqual(
                self.main_module._checkpoint_path(
                    str(video_path), pipeline_id=PIPELINE_FACE
                ),
                self.main_module._checkpoint_path(
                    str(video_path), pipeline_id=PIPELINE_CUSTOM
                ),
                '不同 pipeline 的 checkpoint 必须能够同时存在',
            )

            self.main_module.write_done_marker(
                str(video_path), pipeline_id=PIPELINE_FACE
            )
            self.assertTrue(
                self.main_module.has_existing_artifacts(
                    str(video_path), pipeline_id=PIPELINE_FACE
                )
            )
            self.assertFalse(
                self.main_module.has_existing_artifacts(
                    str(video_path), pipeline_id=PIPELINE_CUSTOM
                )
            )

            index = self.main_module.utils.DirectoryIndex(':memory:')
            try:
                self.assertTrue(
                    index.mark_video_processed(
                        file_md5='same-content-md5',
                        video_path=str(video_path),
                        detection_count=3,
                        model_name='face-model',
                        pipeline_id=PIPELINE_FACE,
                    )
                )
                self.assertTrue(
                    index.is_video_processed_by_md5(
                        'same-content-md5', pipeline_id=PIPELINE_FACE
                    )
                )
                self.assertFalse(
                    index.is_video_processed_by_md5(
                        'same-content-md5', pipeline_id=PIPELINE_CUSTOM
                    )
                )

                self.assertTrue(
                    index.mark_video_processed(
                        file_md5='same-content-md5',
                        video_path=str(video_path),
                        detection_count=1,
                        model_name='custom-model',
                        pipeline_id=PIPELINE_CUSTOM,
                    )
                )
                row_count = index.conn.execute(
                    'SELECT COUNT(*) FROM processed_videos WHERE file_md5=?',
                    ('same-content-md5',),
                ).fetchone()[0]
                self.assertEqual(row_count, 2)
            finally:
                index.close()

    def test_directory_backfill_does_not_overwrite_real_detection_metadata(self):
        """目录收尾补录不得把真实检测数和模型名降级为占位值。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = pathlib.Path(tmpdir) / 'video.mp4'
            video_path.write_bytes(b'video')
            stat_result = video_path.stat()
            cache_key = (
                str(video_path),
                int(stat_result.st_size),
                float(stat_result.st_mtime),
            )

            index = self.main_module.utils.DirectoryIndex(':memory:')
            try:
                self.assertTrue(
                    index.mark_video_processed(
                        file_md5='real-result-md5',
                        video_path=str(video_path),
                        detection_count=9,
                        model_name='verified-model',
                        pipeline_id=PIPELINE_FACE,
                    )
                )
                self.main_module._FILE_MD5_CACHE.clear()
                self.main_module._FILE_MD5_CACHE[cache_key] = 'real-result-md5'

                with mock.patch.object(
                    self.main_module, 'DIRECTORY_INDEX', index
                ), mock.patch.object(
                    self.main_module, 'append_yoloed_md5'
                ):
                    self.assertTrue(
                        self.main_module._mark_directory_done(
                            tmpdir,
                            [video_path.name],
                            pipeline_id=PIPELINE_FACE,
                        )
                    )

                row = index.conn.execute(
                    '''
                    SELECT detection_count, model_name
                    FROM processed_videos
                    WHERE file_md5=?
                    ''',
                    ('real-result-md5',),
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row['detection_count'], 9)
                self.assertEqual(row['model_name'], 'verified-model')
            finally:
                index.close()


if __name__ == '__main__':
    unittest.main()
