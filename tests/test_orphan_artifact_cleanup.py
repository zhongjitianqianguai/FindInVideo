import pathlib
import tempfile
import unittest
from unittest import mock

import utils


PIPELINE_A = '0123456789ab'
PIPELINE_B = 'abcdefabcdef'
MANAGED_TAILS = (
    '.txt',
    '.done',
    '.checkpoint.json',
    '.checkpoint.json.tmp',
    '_frames.mp4',
    '_objects.mp4',
    '_detections.mp4',
    '_mosaic.jpg',
    '_merge_tmp_frames.mp4',
    '_mosaic_12.jpg',
    '_part_007_ABC123_frames.mp4',
    '.done.1234.' + ('a' * 32) + '.tmp',
)


class FakeDirectoryIndex:
    def __init__(self, active_sources=None, error=False):
        self.active_sources = {str(path) for path in (active_sources or ())}
        self.error = error
        self.calls = []

    def has_active_claims_for_paths(self, video_paths):
        self.calls.append(list(video_paths))
        if self.error:
            raise RuntimeError('声明检查失败')
        return any(str(path) in self.active_sources for path in video_paths)


class OrphanArtifactCleanupTests(unittest.TestCase):
    def _artifact(self, directory, source_name, pipeline_id, tail, content=b'x'):
        path = directory / f'{source_name}.findinvideo-{pipeline_id}{tail}'
        path.write_bytes(content)
        return path

    def test_missing_source_deletes_all_managed_tails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            source_name = 'record.mp4'
            artifact_paths = [
                self._artifact(root, source_name, PIPELINE_A, tail, bytes([index + 1]))
                for index, tail in enumerate(MANAGED_TAILS)
            ]
            expected_bytes = sum(path.stat().st_size for path in artifact_paths)
            index = FakeDirectoryIndex()

            result = utils.cleanup_orphan_artifacts(
                root,
                directory_index=index,
            )

            self.assertEqual(result['removed_files'], len(MANAGED_TAILS))
            self.assertEqual(result['removed_bytes'], expected_bytes)
            self.assertEqual(result['skipped_active_sources'], 0)
            self.assertEqual(result['failed_files'], 0)
            self.assertTrue(all(not path.exists() for path in artifact_paths))
            self.assertEqual(index.calls, [[str(root / source_name)]])

    def test_existing_source_keeps_managed_artifacts(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            source = root / 'record.mp4'
            source.write_bytes(b'video')
            artifact = self._artifact(root, source.name, PIPELINE_A, '.done')

            result = utils.cleanup_orphan_artifacts(
                root,
                directory_index=FakeDirectoryIndex(),
            )

            self.assertEqual(result['removed_files'], 0)
            self.assertTrue(artifact.exists())

    def test_similar_and_legacy_names_are_kept(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            names = (
                'record_frames.mp4',
                'record.done',
                'record.mp4.findinvideo-0123456789a.txt',
                'record.mp4.findinvideo-0123456789abc.txt',
                'record.txt.findinvideo-0123456789ab.txt',
                'record.mp4.findinvideo-0123456789ab_extra.txt',
                'record.mp4.findinvideo-0123456789ab_part_7_aa_frames.mp4',
                'record.mp4.findinvideo-0123456789ab_part_007_GG_frames.mp4',
            )
            paths = []
            for name in names:
                path = root / name
                path.write_bytes(b'keep')
                paths.append(path)

            result = utils.cleanup_orphan_artifacts(
                root,
                directory_index=FakeDirectoryIndex(),
            )

            self.assertEqual(result['removed_files'], 0)
            self.assertTrue(all(path.exists() for path in paths))

    def test_extensions_and_pipeline_groups_are_independent(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            artifacts = (
                self._artifact(root, 'same.mp4', PIPELINE_A, '.txt'),
                self._artifact(root, 'same.mp4', PIPELINE_B, '.done'),
                self._artifact(root, 'same.mkv', PIPELINE_A, '_mosaic.jpg'),
            )
            index = FakeDirectoryIndex()

            result = utils.cleanup_orphan_artifacts(root, directory_index=index)

            self.assertEqual(result['removed_files'], len(artifacts))
            self.assertEqual(
                sorted(index.calls),
                sorted([[str(root / 'same.mp4')], [str(root / 'same.mkv')]]),
            )

    def test_active_claim_keeps_entire_source_group(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            artifacts = (
                self._artifact(root, 'record.mp4', PIPELINE_A, '.txt'),
                self._artifact(root, 'record.mp4', PIPELINE_B, '_frames.mp4'),
            )
            source = root / 'record.mp4'
            index = FakeDirectoryIndex(active_sources=[source])

            result = utils.cleanup_orphan_artifacts(root, directory_index=index)

            self.assertEqual(result['removed_files'], 0)
            self.assertEqual(result['skipped_active_sources'], 1)
            self.assertTrue(all(path.exists() for path in artifacts))

    def test_excluded_and_ignored_directories_are_kept(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            excluded = root / 'excluded'
            ignored = root / '_detected'
            excluded.mkdir()
            ignored.mkdir()
            excluded_artifact = self._artifact(
                excluded, 'record.mp4', PIPELINE_A, '.done'
            )
            ignored_artifact = self._artifact(
                ignored, 'record.mp4', PIPELINE_A, '.done'
            )
            normal_artifact = self._artifact(root, 'record.mp4', PIPELINE_A, '.done')

            result = utils.cleanup_orphan_artifacts(
                root,
                exclusions=[excluded],
                directory_index=FakeDirectoryIndex(),
            )

            self.assertEqual(result['removed_files'], 1)
            self.assertFalse(normal_artifact.exists())
            self.assertTrue(excluded_artifact.exists())
            self.assertTrue(ignored_artifact.exists())

    def test_delete_failure_is_counted_and_scan_continues(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            first = self._artifact(root, 'record.mp4', PIPELINE_A, '.txt')
            failed = self._artifact(root, 'record.mp4', PIPELINE_A, '.done')
            last = self._artifact(root, 'record.mp4', PIPELINE_A, '_mosaic.jpg')
            real_remove = utils.os.remove

            def remove(path):
                if pathlib.Path(path) == failed:
                    raise PermissionError('测试删除失败')
                return real_remove(path)

            with mock.patch.object(utils.os, 'remove', side_effect=remove):
                result = utils.cleanup_orphan_artifacts(
                    root,
                    directory_index=FakeDirectoryIndex(),
                )

            self.assertEqual(result['removed_files'], 2)
            self.assertEqual(result['failed_files'], 1)
            self.assertFalse(first.exists())
            self.assertTrue(failed.exists())
            self.assertFalse(last.exists())

    def test_missing_or_failing_claim_interface_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            artifact = self._artifact(root, 'record.mp4', PIPELINE_A, '.done')

            result_missing = utils.cleanup_orphan_artifacts(root, directory_index=object())
            self.assertEqual(result_missing['removed_files'], 0)
            self.assertEqual(result_missing['skipped_active_sources'], 1)
            self.assertTrue(artifact.exists())

            failing_index = FakeDirectoryIndex(error=True)
            result_error = utils.cleanup_orphan_artifacts(
                root,
                directory_index=failing_index,
            )
            self.assertEqual(result_error['removed_files'], 0)
            self.assertEqual(result_error['skipped_active_sources'], 1)
            self.assertTrue(artifact.exists())


if __name__ == '__main__':
    unittest.main()
