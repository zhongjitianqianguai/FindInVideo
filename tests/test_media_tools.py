import contextlib
import io
import math
import os
import pathlib
import tempfile
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]


class MediaTimeParsingTests(unittest.TestCase):
    def test_all_tools_reject_negative_or_non_finite_positions(self):
        import clip_video
        import clip_video_for_upload_telegram
        import clip_video_silence
        import extract_frames

        modules = (
            clip_video,
            clip_video_silence,
            clip_video_for_upload_telegram,
            extract_frames,
        )
        invalid_inputs = (
            '-1',
            'nan',
            'inf',
            '-inf',
            '-1,30',
            '1,nan',
            '1,inf',
            '-1:30',
            '1:-2',
            '00:00:nan',
            '00:00:inf',
        )

        for module in modules:
            for value in invalid_inputs:
                with self.subTest(module=module.__name__, value=value):
                    with self.assertRaises(ValueError):
                        module.parse_position(value)

    def test_range_validation_rejects_invalid_direct_arguments(self):
        from media_tool_utils import validate_time_range

        invalid_ranges = (
            (-1, 1),
            (0, -1),
            (math.nan, 1),
            (0, math.nan),
            (math.inf, 2),
            (0, math.inf),
            (2, 2),
            (3, 2),
        )
        for start_seconds, end_seconds in invalid_ranges:
            with self.subTest(start=start_seconds, end=end_seconds):
                with self.assertRaises(ValueError):
                    validate_time_range(start_seconds, end_seconds)


class MediaBinaryResolutionTests(unittest.TestCase):
    def test_environment_directory_has_highest_priority(self):
        from media_tool_utils import resolve_media_binary

        with tempfile.TemporaryDirectory() as tmpdir:
            ffmpeg = pathlib.Path(tmpdir) / 'ffmpeg.exe'
            ffprobe = pathlib.Path(tmpdir) / 'ffprobe.exe'
            ffmpeg.write_bytes(b'ffmpeg')
            ffprobe.write_bytes(b'ffprobe')

            with mock.patch.dict(
                os.environ,
                {'FINDINVIDEO_FFMPEG_BIN': tmpdir},
                clear=False,
            ), mock.patch('shutil.which', return_value='PATH-TOOL') as which_mock:
                self.assertEqual(resolve_media_binary('ffmpeg'), str(ffmpeg))
                self.assertEqual(resolve_media_binary('ffprobe'), str(ffprobe))

            which_mock.assert_not_called()

    def test_path_lookup_precedes_legacy_compatibility_path(self):
        import media_tool_utils

        with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            'shutil.which', return_value=r'C:\Tools\ffmpeg.exe'
        ):
            self.assertEqual(
                media_tool_utils.resolve_media_binary('ffmpeg'),
                r'C:\Tools\ffmpeg.exe',
            )

    def test_legacy_path_is_last_fallback(self):
        import media_tool_utils

        expected = os.path.join(
            media_tool_utils.LEGACY_FFMPEG_BIN,
            'ffprobe.exe' if os.name == 'nt' else 'ffprobe',
        )
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
            'shutil.which', return_value=None
        ):
            self.assertEqual(
                media_tool_utils.resolve_media_binary('ffprobe'),
                expected,
            )


class TelegramClipTests(unittest.TestCase):
    def test_telegram_output_always_uses_mp4_container(self):
        import clip_video_for_upload_telegram as telegram

        commands = []
        with tempfile.TemporaryDirectory() as tmpdir:
            source = pathlib.Path(tmpdir) / 'source.mkv'
            source.write_bytes(b'video')

            def fake_run(command, **kwargs):
                commands.append(command)
                pathlib.Path(command[-2]).write_bytes(b'clip')
                return types.SimpleNamespace(returncode=0, stderr='')

            with mock.patch.object(telegram, 'FFMPEG', 'ffmpeg'), mock.patch.object(
                telegram.subprocess, 'run', side_effect=fake_run
            ), contextlib.redirect_stdout(io.StringIO()):
                telegram.clip_video_for_upload_telegram(str(source), 1.0, 2.0)

        self.assertEqual(len(commands), 1)
        self.assertTrue(commands[0][-2].endswith('_telegram.mp4'))


class ExtractFramesTests(unittest.TestCase):
    def test_output_directory_isolated_by_source_identity_and_range(self):
        import extract_frames

        with tempfile.TemporaryDirectory() as tmpdir:
            first = pathlib.Path(tmpdir) / 'same.mp4'
            second = pathlib.Path(tmpdir) / 'same.mkv'
            first.write_bytes(b'first')
            second.write_bytes(b'second')

            first_range = extract_frames.build_frame_output_dir(
                str(first), 1.0, 2.0
            )
            second_range = extract_frames.build_frame_output_dir(
                str(first), 2.0, 3.0
            )
            other_source = extract_frames.build_frame_output_dir(
                str(second), 1.0, 2.0
            )

        self.assertEqual(len({first_range, second_range, other_source}), 3)
        self.assertEqual(pathlib.Path(first_range).parent, first.parent)

    def test_rename_failure_is_not_counted_as_saved(self):
        import extract_frames

        with tempfile.TemporaryDirectory() as tmpdir:
            source = pathlib.Path(tmpdir) / 'source.mp4'
            source.write_bytes(b'video')

            def fake_run(command, **kwargs):
                frame_pattern = pathlib.Path(command[-2])
                pathlib.Path(str(frame_pattern).replace('%08d', '00000001')).write_bytes(
                    b'frame'
                )
                return types.SimpleNamespace(returncode=0, stderr='')

            output = io.StringIO()
            with mock.patch.object(
                extract_frames, 'probe_video', return_value=(25.0, 250, 10.0)
            ), mock.patch.object(
                extract_frames.subprocess, 'run', side_effect=fake_run
            ), mock.patch.object(
                extract_frames.os, 'rename', side_effect=OSError('rename failed')
            ), mock.patch.object(
                extract_frames, 'clip_video'
            ), contextlib.redirect_stdout(output):
                extract_frames.extract_frames(str(source), 1.0, 2.0)

        self.assertIn('共保存 0 帧', output.getvalue())
        self.assertIn('重命名失败', output.getvalue())

    def test_probe_video_handles_malformed_fields_without_raising(self):
        import extract_frames

        malformed_payloads = (
            '{not json',
            '{"streams": [{"codec_type": "video", "r_frame_rate": "N/A"}]}',
            '{"streams": [{"codec_type": "video", "r_frame_rate": "1/0", '
            '"duration": "10"}]}',
            '{"streams": [{"codec_type": "video", "r_frame_rate": "25/1", '
            '"nb_frames": "bad", "duration": "nan"}], '
            '"format": {"duration": "inf"}}',
        )

        for payload in malformed_payloads:
            with self.subTest(payload=payload), mock.patch.object(
                extract_frames.subprocess,
                'run',
                return_value=types.SimpleNamespace(
                    returncode=0,
                    stdout=payload,
                    stderr='',
                ),
            ), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    extract_frames.probe_video('video.mp4'),
                    (None, None, None),
                )

    def test_probe_video_derives_frame_count_when_optional_field_is_invalid(self):
        import extract_frames

        payload = (
            '{"streams": [{"codec_type": "video", "r_frame_rate": "25/1", '
            '"nb_frames": "N/A", "duration": "10"}], '
            '"format": {"duration": "10"}}'
        )
        with mock.patch.object(
            extract_frames.subprocess,
            'run',
            return_value=types.SimpleNamespace(
                returncode=0,
                stdout=payload,
                stderr='',
            ),
        ):
            self.assertEqual(
                extract_frames.probe_video('video.mp4'),
                (25.0, 250, 10.0),
            )


class BatchLauncherTests(unittest.TestCase):
    def test_extract_all_frames_is_fail_closed_and_counts_only_current_run(self):
        script = (ROOT / 'extract_all_frames.bat').read_text(encoding='utf-8')

        self.assertIn('FINDINVIDEO_FFMPEG_BIN', script)
        self.assertIn('where ffmpeg', script.lower())
        self.assertGreaterEqual(script.lower().count('if errorlevel 1'), 2)
        self.assertIn('"%RUN_DIR%\\*.jpg"', script)
        self.assertNotIn('echo Video: %VIDEO%', script)
        self.assertNotIn('echo Output: %OUT_DIR%', script)

    def test_rog_launcher_uses_script_directory_and_virtual_environment(self):
        script = (ROOT / 'run_main_nipple_rog.bat').read_text(encoding='utf-8')
        lower = script.lower()

        self.assertIn('cd /d "%~dp0"', lower)
        self.assertIn('.venv\\scripts\\python.exe', lower)
        self.assertIn('main_nipple_rog.py', lower)
        self.assertNotIn('"main_nipple.py"', lower)


if __name__ == '__main__':
    unittest.main()
