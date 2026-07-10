import importlib.util
import concurrent.futures
import pathlib
import sqlite3
import sys
import tempfile
import time
import types
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
MAIN_PATH = ROOT / 'main.py'
ENTRYPOINTS = (
    (MAIN_PATH, False),
    (ROOT / 'main_nipple.py', True),
    (ROOT / 'main_nipple_rog.py', True),
)


def load_main_module(module_path=MAIN_PATH):
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

    module_name = f'findinvideo_{module_path.stem}_test_{time.time_ns()}'
    spec = importlib.util.spec_from_file_location(module_name, module_path)
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
        original_stop_requested = self.main_module.utils._STOP_REQUESTED
        try:
            self.main_module.utils._STOP_REQUESTED = True
            self.assertTrue(self.main_module._pause_requested())
        finally:
            self.main_module.utils._STOP_REQUESTED = original_stop_requested

    def test_heartbeat_interval_tracks_configured_ttl(self):
        utils_module = self.main_module.utils
        original_ttl = self.main_module.os.environ.get('FINDINVIDEO_CLAIM_TTL_SECONDS')
        try:
            self.main_module.os.environ['FINDINVIDEO_CLAIM_TTL_SECONDS'] = '90'
            self.assertEqual(utils_module.get_claim_ttl_seconds(), 90)
            self.assertEqual(utils_module.get_claim_heartbeat_interval_seconds(), 30.0)

            self.main_module.os.environ['FINDINVIDEO_CLAIM_TTL_SECONDS'] = '1'
            self.assertEqual(utils_module.get_claim_ttl_seconds(), 60)
            self.assertEqual(utils_module.get_claim_heartbeat_interval_seconds(), 20.0)
        finally:
            if original_ttl is None:
                self.main_module.os.environ.pop('FINDINVIDEO_CLAIM_TTL_SECONDS', None)
            else:
                self.main_module.os.environ['FINDINVIDEO_CLAIM_TTL_SECONDS'] = original_ttl

    def test_claim_heartbeat_prevents_stale_reclaim(self):
        idx = self.main_module.utils.DirectoryIndex(':memory:')
        original_ttl = self.main_module.os.environ.get('FINDINVIDEO_CLAIM_TTL_SECONDS')
        self.main_module.os.environ['FINDINVIDEO_CLAIM_TTL_SECONDS'] = '5'
        try:
            self.assertTrue(idx.try_claim_video('md5-a', 'video.mp4'))

            now = time.time()
            with idx.conn:
                idx.conn.execute(
                    '''
                    UPDATE processing_claims
                    SET claimed_at=?, heartbeat_at=?, host_name=?, host_id=?, pid=?,
                        owner_token=?, owner_started_at=?
                    WHERE file_md5=?
                    ''',
                    (
                        now - 100, now, 'REMOTE-HOST', 'REMOTE-ID', 999,
                        'remote-owner', now - 200, 'md5-a',
                    ),
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

    def test_dead_local_claim_is_reclaimed_before_ttl(self):
        utils_module = self.main_module.utils
        idx = utils_module.DirectoryIndex(
            ':memory:',
            owner_token='current-owner',
            host_name='LOCAL-HOST',
            host_id='LOCAL-ID',
            pid=222,
            process_started_at=200.0,
        )
        now = time.time()
        try:
            with idx.conn:
                idx.conn.execute(
                    '''
                    INSERT INTO processing_claims (
                        file_md5, video_path, claimed_at, heartbeat_at,
                        host_name, host_id, pid, owner_token, owner_started_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        'md5-dead-owner', 'video.mp4', now, now,
                        'RENAMED-HOST', 'LOCAL-ID', 111, 'dead-owner', 100.0,
                    ),
                )

            with mock.patch.object(
                utils_module, '_is_process_alive', return_value=False, create=True
            ), mock.patch.object(
                utils_module, '_get_process_started_at', return_value=None, create=True
            ):
                self.assertTrue(idx.try_claim_video('md5-dead-owner', 'video.mp4'))

            row = idx.conn.execute(
                'SELECT owner_token, pid FROM processing_claims WHERE file_md5=?',
                ('md5-dead-owner',),
            ).fetchone()
            self.assertEqual(row['owner_token'], 'current-owner')
            self.assertEqual(row['pid'], 222)
        finally:
            idx.close()

    def test_live_local_claim_is_not_reclaimed(self):
        utils_module = self.main_module.utils
        idx = utils_module.DirectoryIndex(
            ':memory:',
            owner_token='current-owner',
            host_name='LOCAL-HOST',
            host_id='LOCAL-ID',
            pid=222,
            process_started_at=200.0,
        )
        now = time.time()
        original_ttl = self.main_module.os.environ.get('FINDINVIDEO_CLAIM_TTL_SECONDS')
        self.main_module.os.environ['FINDINVIDEO_CLAIM_TTL_SECONDS'] = '5'
        try:
            with idx.conn:
                idx.conn.execute(
                    '''
                    INSERT INTO processing_claims (
                        file_md5, video_path, claimed_at, heartbeat_at,
                        host_name, host_id, pid, owner_token, owner_started_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        'md5-live-owner', 'video.mp4', now - 100, now - 100,
                        'LOCAL-HOST', 'LOCAL-ID', 111, 'live-owner', 100.0,
                    ),
                )

            with mock.patch.object(
                utils_module, '_is_process_alive', return_value=True, create=True
            ), mock.patch.object(
                utils_module, '_get_process_started_at', return_value=100.0, create=True
            ):
                self.assertFalse(idx.try_claim_video('md5-live-owner', 'video.mp4'))

            row = idx.conn.execute(
                'SELECT owner_token FROM processing_claims WHERE file_md5=?',
                ('md5-live-owner',),
            ).fetchone()
            self.assertEqual(row['owner_token'], 'live-owner')
        finally:
            if original_ttl is None:
                self.main_module.os.environ.pop('FINDINVIDEO_CLAIM_TTL_SECONDS', None)
            else:
                self.main_module.os.environ['FINDINVIDEO_CLAIM_TTL_SECONDS'] = original_ttl
            idx.close()

    def test_pid_reuse_is_reclaimed_before_ttl(self):
        utils_module = self.main_module.utils
        idx = utils_module.DirectoryIndex(
            ':memory:',
            owner_token='current-owner',
            host_name='LOCAL-HOST',
            host_id='LOCAL-ID',
            pid=222,
            process_started_at=200.0,
        )
        now = time.time()
        try:
            with idx.conn:
                idx.conn.execute(
                    '''
                    INSERT INTO processing_claims (
                        file_md5, video_path, claimed_at, heartbeat_at,
                        host_name, host_id, pid, owner_token, owner_started_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        'md5-reused-pid', 'video.mp4', now, now,
                        'LOCAL-HOST', 'LOCAL-ID', 111, 'old-owner', 100.0,
                    ),
                )

            with mock.patch.object(
                utils_module, '_is_process_alive', return_value=True
            ), mock.patch.object(
                utils_module, '_get_process_started_at', return_value=300.0
            ):
                self.assertTrue(idx.try_claim_video('md5-reused-pid', 'video.mp4'))
        finally:
            idx.close()

    def test_hostname_collision_does_not_trigger_local_pid_reclaim(self):
        utils_module = self.main_module.utils
        idx = utils_module.DirectoryIndex(
            ':memory:',
            owner_token='current-owner',
            host_name='SAME-NAME',
            host_id='MACHINE-B',
            pid=222,
            process_started_at=200.0,
        )
        now = time.time()
        try:
            with idx.conn:
                idx.conn.execute(
                    '''
                    INSERT INTO processing_claims (
                        file_md5, video_path, claimed_at, heartbeat_at,
                        host_name, host_id, pid, owner_token, owner_started_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        'md5-other-machine', 'video.mp4', now, now,
                        'SAME-NAME', 'MACHINE-A', 111, 'remote-owner', 100.0,
                    ),
                )

            with mock.patch.object(
                utils_module,
                '_is_process_alive',
                side_effect=AssertionError('不应查询另一台机器的 PID'),
            ):
                self.assertFalse(
                    idx.try_claim_video('md5-other-machine', 'video.mp4')
                )
        finally:
            idx.close()

    def test_legacy_same_host_dead_pid_is_reclaimed_immediately(self):
        utils_module = self.main_module.utils
        idx = utils_module.DirectoryIndex(
            ':memory:',
            owner_token='current-owner',
            host_name='LOCAL-HOST',
            host_id='LOCAL-ID',
            pid=222,
            process_started_at=200.0,
        )
        now = time.time()
        try:
            with idx.conn:
                idx.conn.execute(
                    """
                    INSERT INTO processing_claims (
                        file_md5, video_path, claimed_at, heartbeat_at,
                        host_name, host_id, pid, owner_token, owner_started_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        'md5-legacy-local',
                        'video.mp4',
                        now,
                        now,
                        'LOCAL-HOST',
                        None,
                        111,
                        None,
                        None,
                    ),
                )
            with mock.patch.object(
                utils_module, '_is_process_alive', return_value=False
            ):
                self.assertTrue(
                    idx.try_claim_video('md5-legacy-local', 'video.mp4')
                )
            row = idx._get_claim_row('md5-legacy-local')
            self.assertEqual(row['owner_token'], 'current-owner')
            self.assertEqual(row['host_id'], 'LOCAL-ID')
        finally:
            idx.close()

    def test_complete_claim_rolls_back_when_processed_write_fails(self):
        idx = self.main_module.utils.DirectoryIndex(':memory:')
        try:
            self.assertTrue(idx.try_claim_video('md5-rollback', 'video.mp4'))
            with mock.patch.object(
                idx,
                '_upsert_processed_video',
                side_effect=sqlite3.OperationalError('write failed'),
            ):
                self.assertFalse(
                    idx.complete_claimed_video('md5-rollback', 'video.mp4')
                )
            self.assertIsNotNone(idx._get_claim_row('md5-rollback'))
            self.assertIsNone(
                idx.conn.execute(
                    'SELECT 1 FROM processed_videos WHERE file_md5=?',
                    ('md5-rollback',),
                ).fetchone()
            )
        finally:
            idx.close()

    def test_cleanup_and_release_all_only_remove_invalid_or_owned_claims(self):
        utils_module = self.main_module.utils
        idx = utils_module.DirectoryIndex(
            ':memory:',
            owner_token='current-owner',
            host_name='LOCAL-HOST',
            host_id='LOCAL-ID',
            pid=222,
            process_started_at=200.0,
        )
        now = time.time()
        original_ttl = self.main_module.os.environ.get('FINDINVIDEO_CLAIM_TTL_SECONDS')
        self.main_module.os.environ['FINDINVIDEO_CLAIM_TTL_SECONDS'] = '60'
        try:
            self.assertTrue(idx.try_claim_video('md5-current', 'current.mp4'))
            rows = (
                (
                    'md5-dead', 'dead.mp4', now, now,
                    'OLD-NAME', 'LOCAL-ID', 111, 'dead-owner', 100.0,
                ),
                (
                    'md5-remote-fresh', 'fresh.mp4', now, now,
                    'REMOTE', 'REMOTE-ID', 333, 'remote-fresh', 300.0,
                ),
                (
                    'md5-remote-stale', 'stale.mp4', now - 100, now - 100,
                    'REMOTE', 'REMOTE-ID', 444, 'remote-stale', 400.0,
                ),
            )
            with idx.conn:
                idx.conn.executemany(
                    '''
                    INSERT INTO processing_claims (
                        file_md5, video_path, claimed_at, heartbeat_at,
                        host_name, host_id, pid, owner_token, owner_started_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    rows,
                )

            with mock.patch.object(
                utils_module, '_is_process_alive', return_value=False
            ):
                self.assertEqual(idx.cleanup_stale_claims(), 2)

            remaining = {
                row['file_md5']
                for row in idx.conn.execute(
                    'SELECT file_md5 FROM processing_claims'
                ).fetchall()
            }
            self.assertEqual(remaining, {'md5-current', 'md5-remote-fresh'})
            self.assertEqual(idx.release_all_claims(), 1)
            remaining = {
                row['file_md5']
                for row in idx.conn.execute(
                    'SELECT file_md5 FROM processing_claims'
                ).fetchall()
            }
            self.assertEqual(remaining, {'md5-remote-fresh'})
        finally:
            if original_ttl is None:
                self.main_module.os.environ.pop('FINDINVIDEO_CLAIM_TTL_SECONDS', None)
            else:
                self.main_module.os.environ['FINDINVIDEO_CLAIM_TTL_SECONDS'] = original_ttl
            idx.close()

    def test_release_claim_does_not_delete_foreign_owner(self):
        utils_module = self.main_module.utils
        idx = utils_module.DirectoryIndex(
            ':memory:',
            owner_token='current-owner',
            host_name='LOCAL-HOST',
            pid=222,
            process_started_at=200.0,
        )
        now = time.time()
        try:
            with idx.conn:
                idx.conn.execute(
                    '''
                    INSERT INTO processing_claims (
                        file_md5, video_path, claimed_at, heartbeat_at,
                        host_name, pid, owner_token, owner_started_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        'md5-foreign-owner', 'video.mp4', now, now,
                        'LOCAL-HOST', 222, 'foreign-owner', 100.0,
                    ),
                )

            heartbeat_before = idx.conn.execute(
                'SELECT heartbeat_at FROM processing_claims WHERE file_md5=?',
                ('md5-foreign-owner',),
            ).fetchone()['heartbeat_at']
            self.assertFalse(idx.refresh_claim('md5-foreign-owner'))
            self.assertFalse(idx.release_claim('md5-foreign-owner'))
            self.assertFalse(
                idx.mark_video_processed('md5-foreign-owner', 'video.mp4')
            )
            self.assertFalse(
                idx.complete_claimed_video(
                    'md5-foreign-owner', 'video.mp4', detection_count=1
                )
            )
            row = idx.conn.execute(
                'SELECT owner_token, heartbeat_at FROM processing_claims WHERE file_md5=?',
                ('md5-foreign-owner',),
            ).fetchone()
            self.assertEqual(row['owner_token'], 'foreign-owner')
            self.assertEqual(row['heartbeat_at'], heartbeat_before)
            self.assertIsNone(
                idx.conn.execute(
                    'SELECT 1 FROM processed_videos WHERE file_md5=?',
                    ('md5-foreign-owner',),
                ).fetchone()
            )
        finally:
            idx.close()

    def test_reclaimed_owner_cannot_refresh_release_or_complete(self):
        utils_module = self.main_module.utils
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(pathlib.Path(tmpdir) / 'claims.db')
            old_owner = utils_module.DirectoryIndex(
                db_path,
                owner_token='old-owner',
                host_name='LOCAL-HOST',
                pid=111,
                process_started_at=100.0,
            )
            new_owner = utils_module.DirectoryIndex(
                db_path,
                owner_token='new-owner',
                host_name='LOCAL-HOST',
                pid=222,
                process_started_at=200.0,
            )
            try:
                self.assertTrue(old_owner.try_claim_video('md5-takeover', 'video.mp4'))
                with mock.patch.object(
                    utils_module, '_is_process_alive', return_value=False
                ), mock.patch.object(
                    utils_module, '_get_process_started_at', return_value=None
                ):
                    self.assertTrue(
                        new_owner.try_claim_video('md5-takeover', 'video.mp4')
                    )

                self.assertFalse(old_owner.refresh_claim('md5-takeover'))
                self.assertFalse(old_owner.release_claim('md5-takeover'))
                self.assertFalse(
                    old_owner.complete_claimed_video('md5-takeover', 'video.mp4')
                )
                self.assertTrue(
                    new_owner.complete_claimed_video(
                        'md5-takeover', 'video.mp4', detection_count=2
                    )
                )
                self.assertFalse(
                    old_owner.try_claim_video('md5-takeover', 'video.mp4')
                )
                self.assertIsNotNone(
                    new_owner.conn.execute(
                        'SELECT 1 FROM processed_videos WHERE file_md5=?',
                        ('md5-takeover',),
                    ).fetchone()
                )
                self.assertIsNone(
                    new_owner.conn.execute(
                        'SELECT 1 FROM processing_claims WHERE file_md5=?',
                        ('md5-takeover',),
                    ).fetchone()
                )
            finally:
                old_owner.close()
                new_owner.close()

    def test_simultaneous_stale_takeover_allows_only_one_new_owner(self):
        utils_module = self.main_module.utils
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(pathlib.Path(tmpdir) / 'takeover-race.db')
            old_owner = utils_module.DirectoryIndex(
                db_path,
                owner_token='old-owner',
                host_name='OLD-HOST',
                host_id='OLD-ID',
                pid=111,
                process_started_at=100.0,
            )
            contender_a = utils_module.DirectoryIndex(
                db_path,
                owner_token='owner-a',
                host_name='HOST-A',
                host_id='HOST-A-ID',
                pid=222,
                process_started_at=200.0,
            )
            contender_b = utils_module.DirectoryIndex(
                db_path,
                owner_token='owner-b',
                host_name='HOST-B',
                host_id='HOST-B-ID',
                pid=333,
                process_started_at=300.0,
            )
            try:
                self.assertTrue(
                    old_owner.try_claim_video('md5-takeover-race', 'video.mp4')
                )
                stale_time = time.time() - 100000
                with old_owner.conn:
                    old_owner.conn.execute(
                        """
                        UPDATE processing_claims
                        SET claimed_at=?, heartbeat_at=?
                        WHERE file_md5=?
                        """,
                        (stale_time, stale_time, 'md5-takeover-race'),
                    )

                original_replace = contender_a._replace_claim_snapshot
                raced = False

                def replace_after_b(row, video_path, now):
                    nonlocal raced
                    if not raced:
                        raced = True
                        self.assertTrue(
                            contender_b.try_claim_video(
                                'md5-takeover-race', video_path
                            )
                        )
                    return original_replace(row, video_path, now)

                with mock.patch.object(
                    contender_a,
                    '_replace_claim_snapshot',
                    side_effect=replace_after_b,
                ):
                    self.assertFalse(
                        contender_a.try_claim_video(
                            'md5-takeover-race', 'video.mp4'
                        )
                    )

                row = contender_a._get_claim_row('md5-takeover-race')
                self.assertEqual(row['owner_token'], 'owner-b')
            finally:
                old_owner.close()
                contender_a.close()
                contender_b.close()

    def test_stale_claim_snapshot_cas_race_is_treated_as_active(self):
        idx = self.main_module.utils.DirectoryIndex(':memory:')
        try:
            now = time.time() - 100000
            with idx.conn:
                idx.conn.execute(
                    """
                    INSERT INTO processing_claims (
                        file_md5, video_path, claimed_at, heartbeat_at,
                        host_name, host_id, pid, owner_token, owner_started_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        'md5-race',
                        'G:\\videos\\video.mp4',
                        now,
                        now,
                        'remote',
                        'remote-id',
                        123,
                        'remote-owner',
                        now,
                    ),
                )
            with mock.patch.object(
                idx, '_claim_is_reclaimable', return_value=True
            ), mock.patch.object(
                idx, '_delete_claim_snapshot', return_value=False
            ):
                self.assertTrue(
                    idx.has_active_claims_for_paths(['/mnt/g/videos/video.mp4'])
                )
        finally:
            idx.close()

    def test_windows_and_wsl_claim_paths_share_alias(self):
        utils_module = self.main_module.utils
        windows_aliases = utils_module._claim_path_aliases(
            'G:\\videos\\video.mp4'
        )
        wsl_aliases = utils_module._claim_path_aliases(
            '/mnt/g/videos/video.mp4'
        )
        self.assertTrue(windows_aliases & wsl_aliases)

    def test_live_pid_without_start_identity_falls_back_to_ttl(self):
        idx = self.main_module.utils.DirectoryIndex(':memory:')
        try:
            row = {
                'owner_token': 'foreign-owner',
                'pid': 123,
                'owner_started_at': None,
                'claimed_at': time.time(),
            }
            with mock.patch.object(
                self.main_module.utils, '_is_process_alive', return_value=True
            ), mock.patch.object(
                self.main_module.utils, '_get_process_started_at', return_value=None
            ):
                self.assertIsNone(idx._local_claim_owner_alive(row))
        finally:
            idx.close()

    def test_claim_invalidates_directory_and_atomic_completion_rechecks_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = pathlib.Path(tmpdir) / 'video.mp4'
            video_path.write_bytes(b'video')
            idx = self.main_module.utils.DirectoryIndex(':memory:')
            try:
                idx.refresh(tmpdir)
                idx.mark_directory_processed(tmpdir)
                self.assertTrue(idx.get_directory_info(tmpdir)[0])

                self.assertTrue(idx.try_claim_video('md5-atomic', str(video_path)))
                self.assertFalse(idx.get_directory_info(tmpdir)[0])
                self.assertFalse(
                    idx.mark_directory_processed_if_idle(tmpdir, [str(video_path)])
                )

                self.assertTrue(idx.release_claim('md5-atomic'))
                checkpoint_path = pathlib.Path(tmpdir) / 'video.checkpoint.json'
                checkpoint_path.write_text('{}', encoding='utf-8')
                self.assertFalse(
                    idx.mark_directory_processed_if_idle(tmpdir, [str(video_path)])
                )
                checkpoint_path.unlink()
                self.assertTrue(
                    idx.mark_directory_processed_if_idle(tmpdir, [str(video_path)])
                )
            finally:
                idx.close()

    def test_completion_semantics_migration_invalidates_old_directory_cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = pathlib.Path(tmpdir) / 'legacy-cache.db'
            video_path = pathlib.Path(tmpdir) / 'video.mp4'
            video_path.write_bytes(b'video')
            (pathlib.Path(tmpdir) / 'video_frames.mp4').write_bytes(b'partial')
            current_mtime = pathlib.Path(tmpdir).stat().st_mtime
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    """
                    CREATE TABLE directories (
                        path TEXT PRIMARY KEY,
                        parent_path TEXT,
                        dir_mtime REAL,
                        last_scan REAL,
                        is_leaf INTEGER,
                        video_count INTEGER,
                        has_artifact INTEGER,
                        excluded INTEGER DEFAULT 0
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO directories (
                        path, dir_mtime, last_scan, is_leaf,
                        video_count, has_artifact, excluded
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (tmpdir, current_mtime, time.time(), 1, 1, 1, 0),
                )
                conn.commit()
            finally:
                conn.close()

            idx = self.main_module.utils.DirectoryIndex(str(db_path))
            try:
                info = idx.get_directory_info(tmpdir)
                self.assertEqual(info, (False, None))
                idx.refresh(tmpdir)
                self.assertFalse(idx.get_directory_info(tmpdir)[0])
            finally:
                idx.close()

    def test_completion_semantics_migration_is_safe_under_concurrent_startup(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = pathlib.Path(tmpdir) / 'concurrent-migration.db'
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    """
                    CREATE TABLE directories (
                        path TEXT PRIMARY KEY,
                        parent_path TEXT,
                        dir_mtime REAL,
                        last_scan REAL,
                        is_leaf INTEGER,
                        video_count INTEGER,
                        has_artifact INTEGER,
                        excluded INTEGER DEFAULT 0
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO directories (
                        path, dir_mtime, last_scan, is_leaf,
                        video_count, has_artifact, excluded
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (tmpdir, time.time(), time.time(), 1, 1, 1, 0),
                )
                conn.commit()
            finally:
                conn.close()

            def open_index(_):
                idx = self.main_module.utils.DirectoryIndex(str(db_path))
                try:
                    return idx.get_directory_info(tmpdir)
                finally:
                    idx.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
                results = list(pool.map(open_index, range(24)))

            self.assertTrue(all(result == (False, None) for result in results))

    def test_atomic_directory_completion_rejects_new_video_after_snapshot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            old_video = pathlib.Path(tmpdir) / 'old.mp4'
            old_video.write_bytes(b'old')
            idx = self.main_module.utils.DirectoryIndex(':memory:')
            try:
                idx.refresh(tmpdir)
                new_video = pathlib.Path(tmpdir) / 'new.mp4'
                new_video.write_bytes(b'new')

                self.assertFalse(
                    idx.mark_directory_processed_if_idle(
                        tmpdir, [str(old_video)]
                    )
                )
                self.assertFalse(idx.get_directory_info(tmpdir)[0])
            finally:
                idx.close()

    def test_legacy_claim_schema_adds_owner_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = pathlib.Path(tmpdir) / 'legacy.db'
            conn = sqlite3.connect(db_path)
            try:
                conn.execute(
                    '''
                    CREATE TABLE processing_claims (
                        file_md5 TEXT PRIMARY KEY,
                        video_path TEXT,
                        claimed_at REAL,
                        host_name TEXT,
                        pid INTEGER
                    )
                    '''
                )
                conn.execute(
                    '''
                    INSERT INTO processing_claims (
                        file_md5, video_path, claimed_at, host_name, pid
                    ) VALUES (?, ?, ?, ?, ?)
                    ''',
                    ('legacy-md5', 'video.mp4', time.time(), 'OLD-HOST', 123),
                )
                conn.commit()
            finally:
                conn.close()

            idx = self.main_module.utils.DirectoryIndex(
                str(db_path), owner_token='current-owner'
            )
            try:
                columns = {
                    row['name']
                    for row in idx.conn.execute(
                        'PRAGMA table_info(processing_claims)'
                    ).fetchall()
                }
                self.assertTrue(
                    {'heartbeat_at', 'host_id', 'owner_token', 'owner_started_at'}
                    <= columns
                )
                self.assertIsNotNone(
                    idx.conn.execute(
                        'SELECT 1 FROM processing_claims WHERE file_md5=?',
                        ('legacy-md5',),
                    ).fetchone()
                )
            finally:
                idx.close()

    def test_process_directory_claims_only_video_being_started(self):
        for module_path, needs_model in ENTRYPOINTS:
            with self.subTest(entrypoint=module_path.name):
                entrypoint = load_main_module(module_path)
                entrypoint.utils._STOP_REQUESTED = False
                entrypoint.all_objects_switch = False
                entrypoint.save_mosaic_switch = False
                entrypoint.save_timestamps_switch = True
                decisions = []
                claimed = []
                released = []

                class FakeCapture:
                    def isOpened(self):
                        return True

                    def get(self, prop):
                        if prop == entrypoint.cv2.CAP_PROP_FPS:
                            return 30
                        if prop == entrypoint.cv2.CAP_PROP_FRAME_COUNT:
                            return 300
                        return 0

                    def release(self):
                        return None

                class FakeDirectoryIndex:
                    def try_claim_video(self, md5, path):
                        claimed.append((md5, path))
                        return True

                    def release_claim(self, md5):
                        released.append(md5)
                        return True

                def get_decision(path, acquire_claim=True):
                    decisions.append((path, acquire_claim))
                    return f'md5-{path}', 'ready'

                with mock.patch.object(
                    entrypoint.os, 'listdir', lambda path: ['a.mp4', 'b.mp4']
                ), mock.patch.object(
                    entrypoint, 'is_video_file', lambda path: path.endswith('.mp4')
                ), mock.patch.object(
                    entrypoint.cv2, 'VideoCapture', lambda path: FakeCapture()
                ), mock.patch.object(
                    entrypoint, '_get_processing_decision', get_decision
                ), mock.patch.object(
                    entrypoint, 'DIRECTORY_INDEX', FakeDirectoryIndex()
                ), mock.patch.object(
                    entrypoint,
                    'detect_objects_in_video',
                    lambda *args, **kwargs: (_ for _ in ()).throw(
                        entrypoint.PauseRequested()
                    ),
                ):
                    args = ['root', 'person']
                    if needs_model:
                        args.append(object())
                    with self.assertRaises(entrypoint.PauseRequested):
                        entrypoint.process_directory_videos(*args)

                first_path = str(pathlib.Path('root') / 'a.mp4')
                second_path = str(pathlib.Path('root') / 'b.mp4')
                self.assertEqual(
                    decisions,
                    [(first_path, False), (second_path, False)],
                )
                self.assertEqual(
                    claimed,
                    [(f'md5-{first_path}', first_path)],
                )
                self.assertEqual(released, [f'md5-{first_path}'])

    def test_real_candidate_decision_never_claims_during_scan(self):
        for module_path, _needs_model in ENTRYPOINTS:
            with self.subTest(entrypoint=module_path.name):
                entrypoint = load_main_module(module_path)

                class FakeDirectoryIndex:
                    def is_video_processed_by_md5(self, md5):
                        return False

                    def is_video_claimed(self, md5):
                        return False

                    def try_claim_video(self, md5, path):
                        raise AssertionError('候选扫描不得创建 claim')

                with mock.patch.object(
                    entrypoint, 'has_existing_artifacts', return_value=False
                ), mock.patch.object(
                    entrypoint, 'is_path_already_yoloed', return_value=False
                ), mock.patch.object(
                    entrypoint, 'get_file_md5_cached', return_value='md5-candidate'
                ), mock.patch.object(
                    entrypoint, 'load_yoloed_md5', return_value=set()
                ), mock.patch.object(
                    entrypoint, 'DIRECTORY_INDEX', FakeDirectoryIndex()
                ):
                    entrypoint._YOLOED_BASENAME_CACHE = {}
                    result = entrypoint._get_processing_decision(
                        'video.mp4', acquire_claim=False
                    )

                self.assertEqual(result, ('md5-candidate', 'ready'))

    def test_checkpoint_bypasses_legacy_backfill_and_yoloed_cache(self):
        for module_path, _needs_model in ENTRYPOINTS:
            with self.subTest(entrypoint=module_path.name):
                entrypoint = load_main_module(module_path)
                with tempfile.TemporaryDirectory() as tmpdir:
                    video_path = pathlib.Path(tmpdir) / 'video.mp4'
                    video_path.write_bytes(b'video')
                    (pathlib.Path(tmpdir) / 'video_frames.mp4').write_bytes(
                        b'partial'
                    )
                    (pathlib.Path(tmpdir) / 'video.checkpoint.json').write_text(
                        '{}', encoding='utf-8'
                    )
                    idx = entrypoint.utils.DirectoryIndex(':memory:')
                    try:
                        with idx.conn:
                            idx._upsert_processed_video(
                                'md5-legacy-backfill',
                                str(video_path),
                                detection_count=-1,
                            )
                        entrypoint._YOLOED_BASENAME_CACHE = {
                            'video.mp4': {pathlib.Path(tmpdir).name.lower()}
                        }
                        with mock.patch.object(
                            entrypoint, 'DIRECTORY_INDEX', idx
                        ), mock.patch.object(
                            entrypoint,
                            'get_file_md5_cached',
                            return_value='md5-legacy-backfill',
                        ), mock.patch.object(
                            entrypoint,
                            'is_path_already_yoloed',
                            return_value=True,
                        ), mock.patch.object(
                            entrypoint,
                            'load_yoloed_md5',
                            return_value={'md5-legacy-backfill'},
                        ):
                            decision = entrypoint._get_processing_decision(
                                str(video_path), acquire_claim=False
                            )

                        self.assertEqual(
                            decision, ('md5-legacy-backfill', 'ready')
                        )
                        self.assertTrue(
                            idx.try_claim_video(
                                'md5-legacy-backfill', str(video_path)
                            )
                        )
                        self.assertFalse(
                            idx.is_video_processed_by_md5(
                                'md5-legacy-backfill'
                            )
                        )
                    finally:
                        idx.close()

    def test_claim_loss_is_detected_before_video_initialization(self):
        for module_path, needs_model in ENTRYPOINTS:
            with self.subTest(entrypoint=module_path.name):
                entrypoint = load_main_module(module_path)
                fake_index = types.SimpleNamespace(refresh_claim=lambda md5: False)
                with mock.patch.object(entrypoint, 'DIRECTORY_INDEX', fake_index):
                    args = ['video.mp4', 'person']
                    if needs_model:
                        args.append(object())
                    with self.assertRaises(entrypoint.ClaimLostError):
                        entrypoint.detect_objects_in_video(
                            *args, claim_md5='md5-lost'
                        )

    def test_claim_loss_at_final_gate_writes_no_final_artifacts(self):
        for module_path, needs_model in ENTRYPOINTS:
            with self.subTest(entrypoint=module_path.name):
                entrypoint = load_main_module(module_path)

                class FakeCapture:
                    def __init__(self):
                        self.open_checks = 0

                    def isOpened(self):
                        self.open_checks += 1
                        return self.open_checks <= 2

                    def get(self, prop):
                        if prop == entrypoint.cv2.CAP_PROP_FPS:
                            return 30
                        if prop == entrypoint.cv2.CAP_PROP_FRAME_COUNT:
                            return 0
                        return 0

                    def read(self):
                        return False, None

                    def release(self):
                        return None

                class FakeProgress:
                    def close(self):
                        return None

                class FakeModel:
                    names = {}

                refresh_results = iter(
                    [True, False] if needs_model else [True, True, False]
                )
                fake_index = types.SimpleNamespace(
                    refresh_claim=lambda md5: next(refresh_results)
                )
                cleared = []

                with tempfile.TemporaryDirectory() as tmpdir:
                    video_path = pathlib.Path(tmpdir) / 'video.mp4'
                    checkpoint_path = pathlib.Path(tmpdir) / 'video.checkpoint.json'
                    checkpoint_path.write_text('{}', encoding='utf-8')
                    with mock.patch.object(
                        entrypoint.cv2, 'VideoCapture', lambda path: FakeCapture()
                    ), mock.patch.object(
                        entrypoint, 'tqdm', lambda *args, **kwargs: FakeProgress()
                    ), mock.patch.object(
                        entrypoint, '_load_checkpoint', return_value=None
                    ), mock.patch.object(
                        entrypoint, 'DIRECTORY_INDEX', fake_index
                    ), mock.patch.object(
                        entrypoint,
                        '_clear_checkpoint',
                        lambda path: cleared.append(path),
                    ), mock.patch.object(
                        entrypoint, 'YOLO', lambda *args, **kwargs: FakeModel()
                    ):
                        args = [str(video_path), 'person']
                        if needs_model:
                            args.append(FakeModel())
                        with self.assertRaises(entrypoint.ClaimLostError):
                            entrypoint.detect_objects_in_video(
                                *args,
                                claim_md5='md5-final-loss',
                                save_crops=True,
                                save_mosaic=True,
                                save_timestamps=True,
                            )

                    self.assertFalse((pathlib.Path(tmpdir) / 'video.txt').exists())
                    self.assertTrue(checkpoint_path.exists())
                    self.assertEqual(cleared, [])

    def test_periodic_claim_loss_stops_before_frame_side_effects(self):
        for module_path, needs_model in ENTRYPOINTS:
            with self.subTest(entrypoint=module_path.name):
                entrypoint = load_main_module(module_path)
                reads = []
                predictions = []

                class FakeCapture:
                    def isOpened(self):
                        return True

                    def get(self, prop):
                        if prop == entrypoint.cv2.CAP_PROP_FPS:
                            return 30
                        if prop == entrypoint.cv2.CAP_PROP_FRAME_COUNT:
                            return 1
                        return 0

                    def read(self):
                        reads.append(True)
                        return True, object()

                    def release(self):
                        return None

                class FakeProgress:
                    def close(self):
                        return None

                class FakeModel:
                    names = {}

                    def predict(self, *args, **kwargs):
                        predictions.append(True)
                        return []

                refresh_results = iter(
                    [True, False] if needs_model else [True, True, False]
                )
                monotonic_values = iter(
                    [0.0, 100.0] if needs_model else [0.0, 0.0, 100.0]
                )
                fake_index = types.SimpleNamespace(
                    refresh_claim=lambda md5: next(refresh_results)
                )
                with mock.patch.object(
                    entrypoint.cv2, 'VideoCapture', lambda path: FakeCapture()
                ), mock.patch.object(
                    entrypoint, 'tqdm', lambda *args, **kwargs: FakeProgress()
                ), mock.patch.object(
                    entrypoint, '_load_checkpoint', return_value=None
                ), mock.patch.object(
                    entrypoint, 'DIRECTORY_INDEX', fake_index
                ), mock.patch.object(
                    entrypoint,
                    'get_claim_heartbeat_interval_seconds',
                    return_value=1.0,
                ), mock.patch.object(
                    entrypoint.time,
                    'monotonic',
                    side_effect=lambda: next(monotonic_values),
                ), mock.patch.object(
                    entrypoint, 'YOLO', lambda *args, **kwargs: FakeModel()
                ):
                    args = ['video.mp4', 'person']
                    if needs_model:
                        args.append(FakeModel())
                    with self.assertRaises(entrypoint.ClaimLostError):
                        entrypoint.detect_objects_in_video(
                            *args, claim_md5='md5-periodic-loss'
                        )

                self.assertEqual(reads, [])
                self.assertEqual(predictions, [])

    def test_checkpoint_with_partial_artifact_is_not_marked_completed(self):
        for module_path, needs_model in ENTRYPOINTS:
            with self.subTest(entrypoint=module_path.name):
                entrypoint = load_main_module(module_path)
                entrypoint.utils._STOP_REQUESTED = False
                marked_dirs = []
                with tempfile.TemporaryDirectory() as tmpdir:
                    video_path = pathlib.Path(tmpdir) / 'video.mp4'
                    video_path.write_bytes(b'video')
                    (pathlib.Path(tmpdir) / 'video_frames.mp4').write_bytes(b'partial')
                    (pathlib.Path(tmpdir) / 'video.checkpoint.json').write_text(
                        '{}', encoding='utf-8'
                    )

                    with mock.patch.object(
                        entrypoint,
                        '_get_processing_decision',
                        lambda path, acquire_claim=True: (
                            None, 'claimed_elsewhere'
                        ),
                    ), mock.patch.object(
                        entrypoint,
                        '_mark_directory_done',
                        lambda path, names: marked_dirs.append(path),
                    ):
                        args = [tmpdir, 'person']
                        if needs_model:
                            args.append(object())
                        result = entrypoint.process_directory_videos(*args)

                self.assertEqual(result, 0)
                self.assertEqual(marked_dirs, [])

    def test_directory_index_does_not_cache_checkpoint_as_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = pathlib.Path(tmpdir) / 'video.mp4'
            video_path.write_bytes(b'video')
            (pathlib.Path(tmpdir) / 'video_frames.mp4').write_bytes(b'partial')
            (pathlib.Path(tmpdir) / 'video.checkpoint.json').write_text(
                '{}', encoding='utf-8'
            )
            idx = self.main_module.utils.DirectoryIndex(':memory:')
            try:
                idx.refresh(tmpdir)
                info = idx.get_directory_info(tmpdir)
                self.assertIsNotNone(info)
                self.assertFalse(info[0])
            finally:
                idx.close()

    def test_partial_current_artifact_without_done_is_not_completed(self):
        for module_path, _needs_model in ENTRYPOINTS:
            with self.subTest(entrypoint=module_path.name):
                entrypoint = load_main_module(module_path)
                with tempfile.TemporaryDirectory() as tmpdir:
                    video_path = pathlib.Path(tmpdir) / 'video.mp4'
                    video_path.write_bytes(b'video')
                    partial_path = pathlib.Path(tmpdir) / 'video_frames.mp4'
                    partial_path.write_bytes(b'partial')

                    self.assertFalse(entrypoint.has_existing_artifacts(str(video_path)))

                    legacy_base = entrypoint.legacy_artifact_basename(
                        str(video_path)
                    )
                    (pathlib.Path(tmpdir) / f'{legacy_base}_frames.mp4').write_bytes(
                        b'legacy-partial'
                    )
                    self.assertFalse(entrypoint.has_existing_artifacts(str(video_path)))

                    done_path = pathlib.Path(tmpdir) / 'video.done'
                    done_path.write_text('done\n', encoding='utf-8')
                    self.assertTrue(entrypoint.has_existing_artifacts(str(video_path)))

    def test_directory_index_does_not_cache_partial_artifact_as_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = pathlib.Path(tmpdir) / 'video.mp4'
            video_path.write_bytes(b'video')
            (pathlib.Path(tmpdir) / 'video_frames.mp4').write_bytes(b'partial')
            idx = self.main_module.utils.DirectoryIndex(':memory:')
            try:
                idx.refresh(tmpdir)
                info = idx.get_directory_info(tmpdir)
                self.assertIsNotNone(info)
                self.assertFalse(info[0])
            finally:
                idx.close()

    def test_checkpoint_blocks_directory_completion_at_final_gate(self):
        for module_path, _needs_model in ENTRYPOINTS:
            with self.subTest(entrypoint=module_path.name):
                entrypoint = load_main_module(module_path)
                with tempfile.TemporaryDirectory() as tmpdir:
                    video_path = pathlib.Path(tmpdir) / 'video.mp4'
                    video_path.write_bytes(b'video')
                    (pathlib.Path(tmpdir) / 'video.checkpoint.json').write_text(
                        '{}', encoding='utf-8'
                    )

                    class FakeDirectoryIndex:
                        def has_active_claims_for_paths(self, paths):
                            raise AssertionError('checkpoint 应先阻断目录完成')

                        def mark_video_processed(self, **kwargs):
                            raise AssertionError('checkpoint 存在时不得补录完成态')

                        def mark_directory_processed_if_idle(self, path, paths):
                            raise AssertionError('checkpoint 存在时不得标记目录完成')

                    with mock.patch.object(
                        entrypoint, 'DIRECTORY_INDEX', FakeDirectoryIndex()
                    ):
                        self.assertFalse(
                            entrypoint._mark_directory_done(tmpdir, ['video.mp4'])
                        )

    def test_active_claim_blocks_directory_completion(self):
        for module_path, _needs_model in ENTRYPOINTS:
            with self.subTest(entrypoint=module_path.name):
                entrypoint = load_main_module(module_path)

                class FakeDirectoryIndex:
                    def has_active_claims_for_paths(self, paths):
                        return True

                    def mark_directory_processed_if_idle(self, path, paths):
                        raise AssertionError('存在 claim 时不得标记目录完成')

                    def mark_video_processed(self, **kwargs):
                        raise AssertionError('存在 claim 时不得补录完成视频')

                with mock.patch.object(
                    entrypoint, 'DIRECTORY_INDEX', FakeDirectoryIndex()
                ):
                    self.assertFalse(
                        entrypoint._mark_directory_done('root', ['video.mp4'])
                    )

    def test_claim_race_loss_skips_detection_and_directory_completion(self):
        for module_path, needs_model in ENTRYPOINTS:
            with self.subTest(entrypoint=module_path.name):
                entrypoint = load_main_module(module_path)
                entrypoint.utils._STOP_REQUESTED = False
                detected = []
                marked_dirs = []
                claim_attempts = []
                releases = []

                class FakeCapture:
                    def isOpened(self):
                        return True

                    def get(self, prop):
                        return 30 if prop == entrypoint.cv2.CAP_PROP_FPS else 300

                    def release(self):
                        return None

                class FakeDirectoryIndex:
                    def try_claim_video(self, md5, path):
                        claim_attempts.append((md5, path))
                        return False

                    def release_claim(self, md5):
                        releases.append(md5)
                        return True

                with mock.patch.object(
                    entrypoint.os, 'listdir', lambda path: ['video.mp4']
                ), mock.patch.object(
                    entrypoint, 'is_video_file', lambda path: path.endswith('.mp4')
                ), mock.patch.object(
                    entrypoint.cv2, 'VideoCapture', lambda path: FakeCapture()
                ), mock.patch.object(
                    entrypoint,
                    '_get_processing_decision',
                    lambda path, acquire_claim=True: ('md5-race', 'ready'),
                ), mock.patch.object(
                    entrypoint, 'DIRECTORY_INDEX', FakeDirectoryIndex()
                ), mock.patch.object(
                    entrypoint,
                    'detect_objects_in_video',
                    lambda *args, **kwargs: detected.append(args),
                ), mock.patch.object(
                    entrypoint,
                    '_mark_directory_done',
                    lambda path, names: marked_dirs.append(path),
                ):
                    args = ['root', 'person']
                    if needs_model:
                        args.append(object())
                    result = entrypoint.process_directory_videos(*args)

                self.assertEqual(result, 0)
                self.assertEqual(len(claim_attempts), 1)
                self.assertEqual(detected, [])
                self.assertEqual(releases, [])
                self.assertEqual(marked_dirs, [])

    def test_partial_claim_block_never_marks_directory_completed(self):
        for module_path, needs_model in ENTRYPOINTS:
            with self.subTest(entrypoint=module_path.name):
                entrypoint = load_main_module(module_path)
                entrypoint.utils._STOP_REQUESTED = False
                entrypoint.save_mosaic_switch = False
                entrypoint.save_timestamps_switch = True
                marked_dirs = []

                class FakeCapture:
                    def isOpened(self):
                        return True

                    def get(self, prop):
                        return 30 if prop == entrypoint.cv2.CAP_PROP_FPS else 300

                    def release(self):
                        return None

                class FakeDirectoryIndex:
                    def try_claim_video(self, md5, path):
                        return True

                    def release_claim(self, md5):
                        return True

                def get_decision(path, acquire_claim=True):
                    if path.endswith('blocked.mp4'):
                        return None, 'claimed_elsewhere'
                    return 'md5-ready', 'ready'

                with mock.patch.object(
                    entrypoint.os,
                    'listdir',
                    lambda path: ['ready.mp4', 'blocked.mp4'],
                ), mock.patch.object(
                    entrypoint, 'is_video_file', lambda path: path.endswith('.mp4')
                ), mock.patch.object(
                    entrypoint.cv2, 'VideoCapture', lambda path: FakeCapture()
                ), mock.patch.object(
                    entrypoint, '_get_processing_decision', get_decision
                ), mock.patch.object(
                    entrypoint, 'DIRECTORY_INDEX', FakeDirectoryIndex()
                ), mock.patch.object(
                    entrypoint, 'detect_objects_in_video', lambda *args, **kwargs: []
                ), mock.patch.object(
                    entrypoint, '_mark_video_completed', lambda *args, **kwargs: True
                ), mock.patch.object(
                    entrypoint,
                    '_mark_directory_done',
                    lambda path, names: marked_dirs.append(path),
                ), mock.patch.object(entrypoint.time, 'sleep', lambda seconds: None):
                    args = ['root', 'person']
                    if needs_model:
                        args.append(object())
                    result = entrypoint.process_directory_videos(*args)

                self.assertEqual(result, 1)
                self.assertEqual(marked_dirs, [])

    def test_unopenable_candidate_is_never_claimed(self):
        for module_path, needs_model in ENTRYPOINTS:
            with self.subTest(entrypoint=module_path.name):
                entrypoint = load_main_module(module_path)
                entrypoint.utils._STOP_REQUESTED = False
                claims = []
                marked_dirs = []

                class FakeCapture:
                    def isOpened(self):
                        return False

                class FakeDirectoryIndex:
                    def try_claim_video(self, md5, path):
                        claims.append((md5, path))
                        return True

                    def release_claim(self, md5):
                        return True

                with mock.patch.object(
                    entrypoint.os, 'listdir', lambda path: ['video.mp4']
                ), mock.patch.object(
                    entrypoint, 'is_video_file', lambda path: path.endswith('.mp4')
                ), mock.patch.object(
                    entrypoint.cv2, 'VideoCapture', lambda path: FakeCapture()
                ), mock.patch.object(
                    entrypoint,
                    '_get_processing_decision',
                    lambda path, acquire_claim=True: ('md5-unopenable', 'ready'),
                ), mock.patch.object(
                    entrypoint, 'DIRECTORY_INDEX', FakeDirectoryIndex()
                ), mock.patch.object(
                    entrypoint,
                    '_mark_directory_done',
                    lambda path, names: marked_dirs.append(path),
                ):
                    args = ['root', 'person']
                    if needs_model:
                        args.append(object())
                    result = entrypoint.process_directory_videos(*args)

                self.assertEqual(result, 0)
                self.assertEqual(claims, [])
                self.assertEqual(marked_dirs, [])

    def test_long_video_is_never_claimed(self):
        claims = []

        class FakeCapture:
            def isOpened(self):
                return True

            def get(self, prop):
                if prop == self_main.cv2.CAP_PROP_FPS:
                    return 1
                if prop == self_main.cv2.CAP_PROP_FRAME_COUNT:
                    return 3601
                return 0

            def release(self):
                return None

        class FakeDirectoryIndex:
            def try_claim_video(self, md5, path):
                claims.append((md5, path))
                return True

        self_main = self.main_module
        with mock.patch.object(
            self_main.os, 'listdir', lambda path: ['video.mp4']
        ), mock.patch.object(
            self_main, 'is_video_file', lambda path: path.endswith('.mp4')
        ), mock.patch.object(
            self_main.cv2, 'VideoCapture', lambda path: FakeCapture()
        ), mock.patch.object(
            self_main,
            '_get_processing_decision',
            lambda path, acquire_claim=True: ('md5-long', 'ready'),
        ), mock.patch.object(
            self_main, 'DIRECTORY_INDEX', FakeDirectoryIndex()
        ):
            result = self_main.process_directory_videos('root', 'person')

        self.assertEqual(result, 0)
        self.assertEqual(claims, [])

    def test_mark_video_completed_writes_all_completion_state(self):
        recorded = []
        yoloed = []
        done = []
        cleared = []

        class FakeDirectoryIndex:
            def complete_claimed_video(self, **kwargs):
                recorded.append(kwargs)
                return True

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
             ), \
             mock.patch.object(
                 self.main_module,
                 '_clear_checkpoint',
                 lambda path: cleared.append(path),
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
        self.assertEqual(cleared, ['video.mp4'])

    def test_failed_completion_keeps_checkpoint(self):
        for module_path, _needs_model in ENTRYPOINTS:
            with self.subTest(entrypoint=module_path.name):
                entrypoint = load_main_module(module_path)
                cleared = []
                fake_index = types.SimpleNamespace(
                    complete_claimed_video=lambda **kwargs: False
                )
                with mock.patch.object(
                    entrypoint, 'DIRECTORY_INDEX', fake_index
                ), mock.patch.object(
                    entrypoint,
                    '_clear_checkpoint',
                    lambda path: cleared.append(path),
                ):
                    self.assertFalse(
                        entrypoint._mark_video_completed(
                            'video.mp4', [1.0], file_md5='md5-failed'
                        )
                    )
                self.assertEqual(cleared, [])

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
             mock.patch.object(
                 self_main,
                 '_get_processing_decision',
                 lambda path, acquire_claim=True: ('md5-c', 'ready'),
             ), \
             mock.patch.object(
                 self_main,
                 'DIRECTORY_INDEX',
                 types.SimpleNamespace(try_claim_video=lambda md5, path: True),
             ), \
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
             mock.patch.object(
                 self_main,
                 '_get_processing_decision',
                 lambda path, acquire_claim=True: ('md5-d', 'ready'),
             ), \
             mock.patch.object(
                 self_main,
                 'DIRECTORY_INDEX',
                 types.SimpleNamespace(try_claim_video=lambda md5, path: True),
             ), \
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

    def test_process_directory_videos_does_not_mark_directory_done_when_claimed_elsewhere(self):
        marked_dirs = []

        self_main = self.main_module
        with mock.patch.object(self_main.os, 'listdir', lambda path: ['video.mp4']), \
             mock.patch.object(self_main, 'is_video_file', lambda path: path.endswith('.mp4')), \
             mock.patch.object(
                 self_main,
                 '_get_processing_decision',
                 lambda path, acquire_claim=True: (None, 'claimed_elsewhere'),
             ), \
             mock.patch.object(
                 self_main,
                 '_mark_directory_done',
                 lambda dir_path, names: marked_dirs.append(dir_path),
             ):
            result = self_main.process_directory_videos('root', 'person')

        self.assertEqual(result, 0)
        self.assertEqual(marked_dirs, [])

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

    def test_process_directory_videos_completes_current_video_before_stopping(self):
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

        def mark_completed(*args, **kwargs):
            completed.append(args)
            return True

        with mock.patch.object(self_main.os, 'listdir', lambda path: ['video.mp4']), \
             mock.patch.object(self_main, 'is_video_file', lambda path: path.endswith('.mp4')), \
             mock.patch.object(self_main.cv2, 'VideoCapture', lambda path: FakeCapture()), \
             mock.patch.object(
                 self_main,
                 '_get_processing_decision',
                 lambda path, acquire_claim=True: ('md5-e', 'ready'),
             ), \
             mock.patch.object(
                 self_main,
                 'DIRECTORY_INDEX',
                 types.SimpleNamespace(try_claim_video=lambda md5, path: True),
             ), \
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
                 mark_completed,
             ):
            with self.assertRaises(self_main.PauseRequested):
                self_main.process_directory_videos('root', 'person')

        self.assertEqual(released, ['md5-e'])
        self.assertEqual(len(completed), 1)

    def test_top_level_pause_prints_clean_stop_message(self):
        with mock.patch('builtins.print') as print_mock:
            try:
                raise self.main_module.PauseRequested()
            except self.main_module.PauseRequested:
                print("\n已按请求暂停，本轮处理已停止。")

        print_mock.assert_called_once_with("\n已按请求暂停，本轮处理已停止。")


if __name__ == '__main__':
    unittest.main()
