"""处理中断、声明释放与断点恢复审计日志回归测试。"""

import pathlib
import tempfile
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


if __name__ == '__main__':
    unittest.main()
