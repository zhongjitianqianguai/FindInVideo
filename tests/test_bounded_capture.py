"""有界共享盘视频读取器的 focused 测试。"""

import os
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import bounded_capture
from bounded_capture import BoundedVideoCapture


class BoundedCaptureTests(unittest.TestCase):
    def _temporary_directory(self) -> tempfile.TemporaryDirectory:
        return tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)

    def _make_video(self, path: str) -> list[np.ndarray]:
        frames = [
            np.full((16, 24, 3), (index * 40, 100, 220 - index * 20), dtype=np.uint8)
            for index in range(4)
        ]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 5.0, (24, 16))
        self.assertTrue(writer.isOpened(), '当前 OpenCV 环境无法创建临时 AVI 测试视频')
        try:
            for frame in frames:
                writer.write(frame)
        finally:
            writer.release()
        return frames

    def test_read_get_seek_and_eof_match_opencv(self):
        with self._temporary_directory() as temp_dir:
            path = os.path.join(temp_dir, 'tiny.avi')
            self._make_video(path)
            native = cv2.VideoCapture(path)
            bounded = BoundedVideoCapture(path, timeout=10.0)
            try:
                self.assertTrue(native.isOpened())
                self.assertTrue(bounded.isOpened())
                for prop in (
                    cv2.CAP_PROP_FRAME_COUNT,
                    cv2.CAP_PROP_FPS,
                    cv2.CAP_PROP_FRAME_WIDTH,
                    cv2.CAP_PROP_FRAME_HEIGHT,
                ):
                    self.assertEqual(bounded.get(prop), native.get(prop))

                for _ in range(4):
                    native_ok, native_frame = native.read()
                    bounded_ok, bounded_frame = bounded.read()
                    self.assertEqual(bounded_ok, native_ok)
                    self.assertTrue(np.array_equal(bounded_frame, native_frame))

                self.assertEqual(bounded.read(), (False, None))
                self.assertEqual(native.read(), (False, None))

                self.assertEqual(bounded.set(cv2.CAP_PROP_POS_FRAMES, 2), native.set(cv2.CAP_PROP_POS_FRAMES, 2))
                self.assertEqual(bounded.get(cv2.CAP_PROP_POS_FRAMES), native.get(cv2.CAP_PROP_POS_FRAMES))
                native_ok, native_frame = native.read()
                bounded_ok, bounded_frame = bounded.read()
                self.assertEqual(bounded_ok, native_ok)
                self.assertTrue(np.array_equal(bounded_frame, native_frame))
            finally:
                bounded.release()
                native.release()

    def test_failed_open_release_is_idempotent_and_does_not_retry(self):
        with self._temporary_directory() as temp_dir:
            missing_path = os.path.join(temp_dir, 'missing.avi')
            original_popen = subprocess.Popen
            with mock.patch.object(bounded_capture.subprocess, 'Popen', wraps=original_popen) as popen:
                capture = BoundedVideoCapture(missing_path, timeout=10.0)
                self.assertFalse(capture.isOpened())
                capture.release()
                capture.release()
                self.assertEqual(popen.call_count, 1)

    def test_blocked_worker_times_out_and_reclaims_process_pipe_and_reader(self):
        with self._temporary_directory() as temp_dir:
            worker_path = Path(temp_dir) / 'blocked_worker.py'
            worker_path.write_text(
                "import pickle, sys, time\n"
                "pickle.dump({'ok': True, 'opened': True}, sys.stdout.buffer)\n"
                "sys.stdout.buffer.flush()\n"
                "time.sleep(60)\n",
                encoding='utf-8',
            )
            with mock.patch.object(bounded_capture, '__file__', str(worker_path)):
                capture = BoundedVideoCapture('unused.avi', timeout=0.2)
            process = capture._process
            reader = capture._reader
            started = time.monotonic()
            try:
                with self.assertRaises(TimeoutError):
                    capture.read()
                self.assertLess(time.monotonic() - started, 3.0)
                self.assertIsNotNone(process.poll())
                self.assertFalse(reader.is_alive())
                self.assertTrue(process.stdin.closed)
                self.assertTrue(process.stdout.closed)
            finally:
                capture.release()


if __name__ == '__main__':
    unittest.main()
