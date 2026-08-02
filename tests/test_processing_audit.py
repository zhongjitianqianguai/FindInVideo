"""处理中断、声明释放与断点恢复审计日志回归测试。"""

import pathlib
import tempfile
import time
import unittest
from unittest import mock

import cv2
import numpy as np

import main as main_entrypoint
import utils


class ProcessingAuditTests(unittest.TestCase):
    """验证审计事件在 SQLite 重开后仍可还原恢复链路。"""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.tempdir.name)
        self.video_path = self.root / 'resume.mp4'
        self.video_path.write_text('测试视频占位内容', encoding='utf-8')
        self.db_path = self.root / 'md5_list' / 'directory_index.db'
        self.db_path.parent.mkdir()
        self.previous_index = utils.DIRECTORY_INDEX
        self.first_index = utils.DirectoryIndex(
            str(self.db_path),
            owner_token='first-session',
            host_name='audit-host',
            host_id='audit-host-a',
            pid=10001,
            process_started_at=100.0,
        )
        utils.DIRECTORY_INDEX = self.first_index
        self.file_md5 = 'audit-md5'
        self.pipeline_id = 'audit-pipeline'

    def tearDown(self):
        current_index = utils.DIRECTORY_INDEX
        if current_index is not self.previous_index:
            current_index.close()
        utils.DIRECTORY_INDEX = self.previous_index
        self.tempdir.cleanup()

    def _create_test_video(self, name, frame_count=3):
        """创建用于逐帧推理回归测试的小视频。"""
        video_path = self.root / name
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*'MJPG'),
            10.0,
            (32, 24),
        )
        if not writer.isOpened():
            self.skipTest('当前 OpenCV 环境无法创建内存异常测试视频')
        try:
            for frame_number in range(frame_count):
                writer.write(np.full(
                    (24, 32, 3), frame_number * 20, dtype=np.uint8,
                ))
        finally:
            writer.release()
        return video_path

    def test_interrupt_release_and_resume_are_persisted(self):
        """中断后的新会话能领取、加载检查点并留下可查询的完整事件链。"""
        self.assertTrue(self.first_index.try_claim_video(
            self.file_md5,
            str(self.video_path),
            pipeline_id=self.pipeline_id,
        ))
        self.assertTrue(utils._save_checkpoint(
            str(self.video_path),
            next_frame=37,
            detections=[1.0, 2.0],
            last_detected=2.0,
            claim_md5=self.file_md5,
            last_success_frame=36,
            pipeline_id=self.pipeline_id,
            reason='pause_requested',
        ))
        self.assertTrue(self.first_index.release_claim(
            self.file_md5, pipeline_id=self.pipeline_id,
        ))
        self.first_index.close()

        resumed_index = utils.DirectoryIndex(
            str(self.db_path),
            owner_token='resumed-session',
            host_name='audit-host',
            host_id='audit-host-b',
            pid=10002,
            process_started_at=200.0,
        )
        utils.DIRECTORY_INDEX = resumed_index
        self.assertTrue(resumed_index.try_claim_video(
            self.file_md5,
            str(self.video_path),
            pipeline_id=self.pipeline_id,
        ))
        checkpoint = utils._load_checkpoint(
            str(self.video_path), pipeline_id=self.pipeline_id,
        )
        self.assertEqual(checkpoint['next_frame'], 37)
        self.assertTrue(utils.record_resume_seek(
            str(self.video_path),
            requested_frame=37,
            seek_ok=True,
            reported_frame=37,
            file_md5=self.file_md5,
            pipeline_id=self.pipeline_id,
        ))

        events = resumed_index.list_processing_events(
            limit=50,
            file_md5=self.file_md5,
            pipeline_id=self.pipeline_id,
        )
        event_types = [event['event_type'] for event in events]
        self.assertEqual(
            event_types,
            [
                'claim_acquired',
                'checkpoint_saved',
                'claim_released',
                'claim_acquired',
                'checkpoint_loaded_for_resume',
                'resume_seek_applied',
            ],
        )
        self.assertEqual(events[1]['details']['reason'], 'pause_requested')
        self.assertTrue(events[3]['details']['checkpoint_present'])
        self.assertTrue(events[-1]['details']['position_verified'])
        self.assertEqual(events[-1]['details']['reported_frame'], 37)

    def test_real_opencv_seek_is_persisted_after_checkpoint_resume(self):
        """真实 OpenCV 文件定位的报告帧会被持久化，避免只记录计划恢复帧。"""
        video_path = self.root / 'seek.avi'
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*'MJPG'),
            10.0,
            (32, 24),
        )
        if not writer.isOpened():
            self.skipTest('当前 OpenCV 环境无法创建可验证的测试视频')
        try:
            for frame_number in range(12):
                frame = np.full((24, 32, 3), frame_number, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()

        self.assertTrue(self.first_index.try_claim_video(
            'seek-md5', str(video_path), pipeline_id=self.pipeline_id,
        ))
        self.assertTrue(utils._save_checkpoint(
            str(video_path),
            next_frame=7,
            detections=[],
            last_detected=-5.0,
            claim_md5='seek-md5',
            last_success_frame=6,
            pipeline_id=self.pipeline_id,
            reason='pause_requested',
        ))
        self.assertTrue(self.first_index.release_claim(
            'seek-md5', pipeline_id=self.pipeline_id,
        ))
        checkpoint = utils._load_checkpoint(
            str(video_path), pipeline_id=self.pipeline_id,
        )
        cap = cv2.VideoCapture(str(video_path))
        self.assertTrue(cap.isOpened())
        try:
            seek_ok = cap.set(cv2.CAP_PROP_POS_FRAMES, checkpoint['next_frame'])
            reported_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        finally:
            cap.release()
        self.assertTrue(seek_ok)
        self.assertTrue(utils.record_resume_seek(
            str(video_path),
            requested_frame=checkpoint['next_frame'],
            seek_ok=seek_ok,
            reported_frame=reported_frame,
            file_md5='seek-md5',
            pipeline_id=self.pipeline_id,
        ))

        events = self.first_index.list_processing_events(
            limit=20,
            file_md5='seek-md5',
            pipeline_id=self.pipeline_id,
        )
        self.assertEqual(events[-1]['event_type'], 'resume_seek_applied')
        self.assertTrue(events[-1]['details']['position_verified'])
        self.assertEqual(events[-1]['details']['reported_frame'], 7)

    def test_main_entrypoint_pauses_releases_and_resumes_from_checkpoint(self):
        """主入口在真实视频上暂停后，下一会话会从审计记录的帧继续。"""
        video_path = self.root / 'entrypoint.avi'
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*'MJPG'),
            10.0,
            (32, 24),
        )
        if not writer.isOpened():
            self.skipTest('当前 OpenCV 环境无法创建主入口恢复测试视频')
        try:
            for frame_number in range(12):
                frame = np.full((24, 32, 3), frame_number, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()

        class EmptyModel:
            """不产生检测结果的轻量模型替身。"""

            names = {}

            def predict(self, *args, **kwargs):
                return []

        file_md5 = 'entrypoint-md5'
        self.assertTrue(self.first_index.try_claim_video(
            file_md5, str(video_path), pipeline_id=self.pipeline_id,
        ))
        pause_checks = iter((False, False, True))
        with mock.patch.object(
            main_entrypoint, 'DIRECTORY_INDEX', self.first_index,
        ), mock.patch.object(
            main_entrypoint, '_ACTIVE_PIPELINE_ID', self.pipeline_id,
        ), mock.patch.object(
            main_entrypoint, '_pause_requested', side_effect=lambda *args: next(pause_checks),
        ):
            with self.assertRaises(main_entrypoint.PauseRequested):
                main_entrypoint.detect_objects_in_video(
                    str(video_path),
                    target_class='face',
                    claim_md5=file_md5,
                    model=EmptyModel(),
                )
            main_entrypoint._release_claim_safely(file_md5)

        checkpoint = utils._load_checkpoint(
            str(video_path), pipeline_id=self.pipeline_id,
        )
        self.assertEqual(checkpoint['next_frame'], 2)
        self.first_index.close()

        resumed_index = utils.DirectoryIndex(
            str(self.db_path),
            owner_token='entrypoint-resumed-session',
            host_name='audit-host',
            host_id='audit-host-c',
            pid=10003,
            process_started_at=300.0,
        )
        utils.DIRECTORY_INDEX = resumed_index
        self.assertTrue(resumed_index.try_claim_video(
            file_md5, str(video_path), pipeline_id=self.pipeline_id,
        ))
        with mock.patch.object(
            main_entrypoint, 'DIRECTORY_INDEX', resumed_index,
        ), mock.patch.object(
            main_entrypoint, '_ACTIVE_PIPELINE_ID', self.pipeline_id,
        ), mock.patch.object(main_entrypoint, '_pause_requested', return_value=False):
            detections = main_entrypoint.detect_objects_in_video(
                str(video_path),
                target_class='face',
                claim_md5=file_md5,
                model=EmptyModel(),
            )
        self.assertEqual(detections, [])
        self.assertTrue(resumed_index.release_claim(
            file_md5, pipeline_id=self.pipeline_id,
        ))

        events = resumed_index.list_processing_events(
            limit=50,
            file_md5=file_md5,
            pipeline_id=self.pipeline_id,
        )
        event_types = [event['event_type'] for event in events]
        self.assertIn('checkpoint_saved', event_types)
        self.assertIn('claim_released', event_types)
        self.assertIn('checkpoint_loaded_for_resume', event_types)
        resume_event = next(
            event for event in events
            if event['event_type'] == 'resume_seek_applied'
        )
        self.assertEqual(resume_event['details']['requested_frame'], 2)
        self.assertEqual(resume_event['details']['reported_frame'], 2)
        self.assertTrue(resume_event['details']['position_verified'])

    def test_main_memory_error_retries_current_frame_and_releases_state(self):
        """单次内存分配失败应清理预测状态并原帧重试。"""
        video_path = self._create_test_video('memory-retry.avi')

        class RetryModel:
            names = {}

            def __init__(self):
                self.calls = 0
                self.predictor = type('PredictorState', (), {})()

            def predict(self, *args, **kwargs):
                self.calls += 1
                self.predictor.results = ['模拟结果']
                self.predictor.batch = object()
                self.predictor.dataset = object()
                if self.calls == 1:
                    raise MemoryError('模拟一次内存分配失败')
                return []

        model = RetryModel()
        with mock.patch.object(
            main_entrypoint, '_ACTIVE_PIPELINE_ID', self.pipeline_id,
        ), mock.patch.object(
            main_entrypoint, '_pause_requested', return_value=False,
        ), mock.patch.object(
            main_entrypoint, 'MEMORY_RETRY_DELAY_SECONDS', 0,
        ):
            detections = main_entrypoint.detect_objects_in_video(
                str(video_path), target_class='face', model=model,
            )

        self.assertEqual(detections, [])
        self.assertEqual(model.calls, 4)
        self.assertIsNone(model.predictor.results)
        self.assertIsNone(model.predictor.batch)
        self.assertIsNone(model.predictor.dataset)

    def test_main_repeated_memory_error_saves_exact_checkpoint(self):
        """内存重试仍失败时应保存当前未处理帧的即时检查点。"""
        video_path = self._create_test_video('memory-checkpoint.avi')

        class RepeatedFailureModel:
            names = {}

            def __init__(self):
                self.calls = 0
                self.predictor = type('PredictorState', (), {})()

            def predict(self, *args, **kwargs):
                self.calls += 1
                self.predictor.results = ['模拟结果']
                self.predictor.batch = object()
                self.predictor.dataset = object()
                if self.calls >= 2:
                    raise MemoryError('模拟持续内存分配失败')
                return []

        model = RepeatedFailureModel()
        with mock.patch.object(
            main_entrypoint, '_ACTIVE_PIPELINE_ID', self.pipeline_id,
        ), mock.patch.object(
            main_entrypoint, '_pause_requested', return_value=False,
        ), mock.patch.object(
            main_entrypoint, 'MEMORY_RETRY_DELAY_SECONDS', 0,
        ):
            with self.assertRaisesRegex(MemoryError, '模拟持续内存分配失败'):
                main_entrypoint.detect_objects_in_video(
                    str(video_path), target_class='face', model=model,
                )

        checkpoint = utils._load_checkpoint(
            str(video_path), pipeline_id=self.pipeline_id,
        )
        self.assertIsNotNone(checkpoint)
        self.assertEqual(checkpoint['next_frame'], 1)
        self.assertEqual(checkpoint['last_success_frame'], 0)
        self.assertEqual(checkpoint['reason'], 'memory_allocation_error')
        self.assertEqual(checkpoint['frame_video_segments'], [])
        self.assertEqual(model.calls, 3)
        self.assertIsNone(model.predictor.results)
        self.assertIsNone(model.predictor.batch)
        self.assertIsNone(model.predictor.dataset)

    def test_resumed_frame_video_merges_pre_and_post_interrupt_segments(self):
        """断点后的检测帧写入新分段，完成时必须保留中断前后的全部帧。"""
        final_path = self.root / 'detected_frames.mp4'
        first_frames = [
            np.full((24, 32, 3), value, dtype=np.uint8)
            for value in (20, 40, 60)
        ]
        resumed_frames = [
            np.full((24, 32, 3), value, dtype=np.uint8)
            for value in (80, 100)
        ]

        first_session = utils.ResumableFrameVideo(
            str(final_path),
            fps=10.0,
            cv2_module=cv2,
            checkpoint=None,
        )
        for frame_number, frame in enumerate(first_frames):
            first_session.write(frame, frame_number)
        first_session.seal_segment()
        if not final_path.exists():
            self.skipTest('当前 OpenCV 环境无法创建可验证的 MP4 分段')

        checkpoint_segments = first_session.checkpoint_segments()
        self.assertEqual(checkpoint_segments, [final_path.name])
        self.assertTrue(utils._save_checkpoint(
            str(self.video_path),
            next_frame=len(first_frames),
            detections=[0.0, 0.1, 0.2],
            last_detected=0.2,
            last_success_frame=2,
            pipeline_id=self.pipeline_id,
            reason='periodic',
            frame_video_segments=checkpoint_segments,
        ))
        checkpoint = utils._load_checkpoint(
            str(self.video_path), pipeline_id=self.pipeline_id,
        )

        resumed_session = utils.ResumableFrameVideo(
            str(final_path),
            fps=10.0,
            cv2_module=cv2,
            checkpoint=checkpoint,
        )
        for offset, frame in enumerate(resumed_frames, start=len(first_frames)):
            resumed_session.write(frame, offset)
        resumed_session.finish()

        cap = cv2.VideoCapture(str(final_path))
        self.assertTrue(cap.isOpened())
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        self.assertEqual(frame_count, len(first_frames) + len(resumed_frames))
        self.assertEqual(
            list(self.root.glob('*_part_*_frames.mp4')),
            [],
            '合并成功后不应残留恢复分段',
        )

    def test_early_eof_checkpoint_defers_retry_until_deadline(self):
        """提前 EOF 检查点必须带退避时间，避免每次启动立刻重复失败。"""
        retry_not_before = time.time() + 600
        self.assertTrue(utils._save_checkpoint(
            str(self.video_path),
            next_frame=5,
            detections=[],
            last_detected=-5.0,
            last_success_frame=4,
            pipeline_id=self.pipeline_id,
            reason='unexpected_early_eof',
            early_eof_retry_count=2,
            retry_not_before=retry_not_before,
        ))

        self.assertTrue(utils.is_checkpoint_retry_deferred(
            str(self.video_path), pipeline_id=self.pipeline_id,
        ))
        self.assertFalse(utils.is_checkpoint_retry_deferred(
            str(self.video_path),
            pipeline_id=self.pipeline_id,
            now=retry_not_before + 1,
        ))

    def test_main_resume_keeps_detected_frames_and_writes_periodic_checkpoints(self):
        """主入口恢复后应合并检测帧，并把周期进度持久化。"""
        video_path = self.root / 'detected-entrypoint.avi'
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*'MJPG'),
            10.0,
            (32, 24),
        )
        if not writer.isOpened():
            self.skipTest('当前 OpenCV 环境无法创建主入口检测帧测试视频')
        try:
            for frame_number in range(10):
                writer.write(np.full(
                    (24, 32, 3), frame_number * 20, dtype=np.uint8,
                ))
        finally:
            writer.release()

        class FakeCoordinates:
            """提供与模型框坐标相同的最小接口。"""

            def cpu(self):
                return self

            def numpy(self):
                return np.array([2, 2, 20, 20], dtype=np.float32)

        class FakeXyxy:
            def __getitem__(self, index):
                return FakeCoordinates()

        class FakeBox:
            cls = 0
            xyxy = FakeXyxy()

        class DetectEveryFrameModel:
            names = {0: 'face'}

            def predict(self, *args, **kwargs):
                return [type('Result', (), {'boxes': [FakeBox()]})()]

        file_md5 = 'detected-entrypoint-md5'
        self.assertTrue(self.first_index.try_claim_video(
            file_md5, str(video_path), pipeline_id=self.pipeline_id,
        ))
        pause_checks = iter((False, False, True))
        with mock.patch.object(
            main_entrypoint, 'DIRECTORY_INDEX', self.first_index,
        ), mock.patch.object(
            main_entrypoint, '_ACTIVE_PIPELINE_ID', self.pipeline_id,
        ), mock.patch.object(
            main_entrypoint,
            '_pause_requested',
            side_effect=lambda *args: next(pause_checks),
        ):
            with self.assertRaises(main_entrypoint.PauseRequested):
                main_entrypoint.detect_objects_in_video(
                    str(video_path),
                    target_class='face',
                    claim_md5=file_md5,
                    model=DetectEveryFrameModel(),
                )
            main_entrypoint._release_claim_safely(file_md5)

        interrupted_checkpoint = utils._load_checkpoint(
            str(video_path), pipeline_id=self.pipeline_id,
        )
        self.assertEqual(interrupted_checkpoint['next_frame'], 2)
        self.assertEqual(len(interrupted_checkpoint['frame_video_segments']), 1)
        self.first_index.close()

        resumed_index = utils.DirectoryIndex(
            str(self.db_path),
            owner_token='detected-resumed-session',
            host_name='audit-host',
            host_id='audit-host-d',
            pid=10004,
            process_started_at=400.0,
        )
        utils.DIRECTORY_INDEX = resumed_index
        self.assertTrue(resumed_index.try_claim_video(
            file_md5, str(video_path), pipeline_id=self.pipeline_id,
        ))
        with mock.patch.object(
            main_entrypoint, 'DIRECTORY_INDEX', resumed_index,
        ), mock.patch.object(
            main_entrypoint, '_ACTIVE_PIPELINE_ID', self.pipeline_id,
        ), mock.patch.object(
            main_entrypoint, '_pause_requested', return_value=False,
        ), mock.patch.object(
            main_entrypoint, 'get_checkpoint_interval_seconds', return_value=0,
        ):
            detections = main_entrypoint.detect_objects_in_video(
                str(video_path),
                target_class='face',
                claim_md5=file_md5,
                model=DetectEveryFrameModel(),
            )
        self.assertTrue(resumed_index.release_claim(
            file_md5, pipeline_id=self.pipeline_id,
        ))

        artifact_base = utils.safe_artifact_basename(
            str(video_path), pipeline_id=self.pipeline_id,
        )
        final_path = self.root / f'{artifact_base}_frames.mp4'
        cap = cv2.VideoCapture(str(final_path))
        self.assertTrue(cap.isOpened())
        try:
            output_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        finally:
            cap.release()
        self.assertEqual(output_frame_count, len(detections))
        self.assertGreater(output_frame_count, 2)
        self.assertEqual(list(self.root.glob('*_part_*_frames.mp4')), [])

        final_checkpoint = utils._load_checkpoint(
            str(video_path), pipeline_id=self.pipeline_id,
        )
        self.assertEqual(
            final_checkpoint['reason'],
            'processing_complete_pending_commit',
        )
        events = resumed_index.list_processing_events(
            limit=100,
            file_md5=file_md5,
            pipeline_id=self.pipeline_id,
        )
        self.assertTrue(any(
            event['event_type'] == 'checkpoint_saved'
            and event['details'].get('reason') == 'periodic'
            for event in events
        ))


if __name__ == '__main__':
    unittest.main()
