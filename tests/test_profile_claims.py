import os
import pathlib
import sqlite3
import sys
import tempfile
import time
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import utils as utils_module


PIPELINE_FACE = 'video:yolov11-face:test'
PIPELINE_CUSTOM = 'video:custom-model:test'


def create_legacy_database(db_path, active_claim=False):
    """创建 processed_videos 仍以 file_md5 为单主键的旧版数据库。"""
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            '''
            CREATE TABLE processed_videos (
                file_md5 TEXT PRIMARY KEY,
                video_path TEXT,
                processed_at REAL,
                detection_count INTEGER,
                model_name TEXT
            )
            '''
        )
        conn.execute(
            '''
            INSERT INTO processed_videos (
                file_md5, video_path, processed_at, detection_count, model_name
            ) VALUES (?, ?, ?, ?, ?)
            ''',
            ('legacy-md5', 'legacy.mp4', 123.5, 7, 'legacy-model'),
        )
        if active_claim:
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
                ('active-md5', 'active.mp4', time.time(), 'OLD-HOST', 123),
            )
        conn.commit()
    finally:
        conn.close()


class ProfileClaimTests(unittest.TestCase):
    def test_legacy_processed_table_migrates_to_pipeline_composite_key(self):
        """旧完成记录必须原子迁移到复合主键，并归入 legacy-v1。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = pathlib.Path(tmpdir) / 'legacy.db'
            create_legacy_database(db_path)

            index = utils_module.DirectoryIndex(str(db_path))
            try:
                table_info = index.conn.execute(
                    'PRAGMA table_info(processed_videos)'
                ).fetchall()
                primary_key = {
                    row['name']: row['pk'] for row in table_info if row['pk']
                }
                self.assertEqual(
                    primary_key,
                    {'file_md5': 1, 'pipeline_id': 2},
                )

                row = index.conn.execute(
                    '''
                    SELECT file_md5, pipeline_id, video_path, processed_at,
                           detection_count, model_name
                    FROM processed_videos
                    WHERE file_md5=?
                    ''',
                    ('legacy-md5',),
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row['pipeline_id'], utils_module.LEGACY_PIPELINE_ID)
                self.assertEqual(row['video_path'], 'legacy.mp4')
                self.assertEqual(row['processed_at'], 123.5)
                self.assertEqual(row['detection_count'], 7)
                self.assertEqual(row['model_name'], 'legacy-model')
            finally:
                index.close()

    def test_legacy_migration_fails_closed_while_claim_is_active(self):
        """旧实例仍有 active claim 时不得迁移或继续启动。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = pathlib.Path(tmpdir) / 'legacy-active.db'
            create_legacy_database(db_path, active_claim=True)

            opened_connections = []
            real_connect = sqlite3.connect

            def tracked_connect(*args, **kwargs):
                connection = real_connect(*args, **kwargs)
                opened_connections.append(connection)
                return connection

            try:
                with mock.patch.object(
                    utils_module.sqlite3, 'connect', side_effect=tracked_connect
                ):
                    with self.assertRaisesRegex(
                        RuntimeError, '旧实例仍持有处理声明'
                    ):
                        utils_module.DirectoryIndex(str(db_path))
            finally:
                for connection in opened_connections:
                    connection.close()

            conn = sqlite3.connect(db_path)
            try:
                columns = {
                    row[1]: row[5]
                    for row in conn.execute(
                        'PRAGMA table_info(processed_videos)'
                    ).fetchall()
                }
                self.assertNotIn('pipeline_id', columns)
                self.assertEqual(columns.get('file_md5'), 1)
                self.assertEqual(
                    conn.execute(
                        'SELECT COUNT(*) FROM processed_videos'
                    ).fetchone()[0],
                    1,
                )
                self.assertEqual(
                    conn.execute(
                        'SELECT COUNT(*) FROM processing_claims'
                    ).fetchone()[0],
                    1,
                )
            finally:
                conn.close()

    def test_same_md5_completion_is_isolated_by_pipeline(self):
        """同一 MD5 在不同 pipeline 下必须分别领取和完成。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            video_path = root / 'video.mp4'
            video_path.write_bytes(b'stable-video')
            index = utils_module.DirectoryIndex(str(root / 'claims.db'))
            try:
                self.assertTrue(
                    index.try_claim_video(
                        'same-md5',
                        str(video_path),
                        pipeline_id=PIPELINE_FACE,
                    )
                )
                self.assertTrue(
                    index.complete_claimed_video(
                        'same-md5',
                        str(video_path),
                        detection_count=2,
                        model_name='face-model',
                        pipeline_id=PIPELINE_FACE,
                    )
                )
                self.assertTrue(
                    index.is_video_processed_by_md5(
                        'same-md5', pipeline_id=PIPELINE_FACE
                    )
                )
                self.assertFalse(
                    index.is_video_processed_by_md5(
                        'same-md5', pipeline_id=PIPELINE_CUSTOM
                    )
                )

                self.assertTrue(
                    index.try_claim_video(
                        'same-md5',
                        str(video_path),
                        pipeline_id=PIPELINE_CUSTOM,
                    )
                )
                self.assertTrue(
                    index.complete_claimed_video(
                        'same-md5',
                        str(video_path),
                        detection_count=5,
                        model_name='custom-model',
                        pipeline_id=PIPELINE_CUSTOM,
                    )
                )

                rows = index.conn.execute(
                    '''
                    SELECT pipeline_id, detection_count, model_name
                    FROM processed_videos
                    WHERE file_md5=?
                    ORDER BY pipeline_id
                    ''',
                    ('same-md5',),
                ).fetchall()
                self.assertEqual(len(rows), 2)
                self.assertEqual(
                    {
                        row['pipeline_id']:
                            (row['detection_count'], row['model_name'])
                        for row in rows
                    },
                    {
                        PIPELINE_FACE: (2, 'face-model'),
                        PIPELINE_CUSTOM: (5, 'custom-model'),
                    },
                )
            finally:
                index.close()

    def test_source_path_key_blocks_second_claim_after_content_changes(self):
        """文件内容和 MD5 变化后，同一源路径仍不得被第二个实例领取。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            db_path = root / 'claims.db'
            video_path = root / 'growing.mp4'
            video_path.write_bytes(b'version-one')
            first = utils_module.DirectoryIndex(
                str(db_path), owner_token='owner-one'
            )
            second = utils_module.DirectoryIndex(
                str(db_path), owner_token='owner-two'
            )
            try:
                self.assertTrue(
                    first.try_claim_video(
                        'md5-version-one',
                        str(video_path),
                        pipeline_id=PIPELINE_FACE,
                    )
                )

                video_path.write_bytes(b'version-two-is-longer')
                self.assertFalse(
                    second.try_claim_video(
                        'md5-version-two',
                        str(video_path),
                        pipeline_id=PIPELINE_FACE,
                    )
                )

                rows = first.conn.execute(
                    '''
                    SELECT file_md5, owner_token, source_path_key
                    FROM processing_claims
                    '''
                ).fetchall()
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]['file_md5'], 'md5-version-one')
                self.assertEqual(rows[0]['owner_token'], 'owner-one')
                self.assertTrue(rows[0]['source_path_key'])
            finally:
                first.close()
                second.close()

    def test_source_size_or_mtime_change_rejects_completion(self):
        """领取后源文件大小或修改时间变化时不得写入完成态。"""
        mutations = {
            'size': lambda path, snapshot: path.write_bytes(b'content-is-longer'),
            'mtime': lambda path, snapshot: os.utime(
                path,
                ns=(snapshot['mtime_ns'], snapshot['mtime_ns'] + 5_000_000_000),
            ),
        }
        for mutation_name, mutate in mutations.items():
            with self.subTest(mutation=mutation_name), tempfile.TemporaryDirectory() as tmpdir:
                root = pathlib.Path(tmpdir)
                video_path = root / 'video.mp4'
                video_path.write_bytes(b'content')
                snapshot = utils_module._source_snapshot(str(video_path))
                index = utils_module.DirectoryIndex(str(root / 'claims.db'))
                try:
                    self.assertTrue(
                        index.try_claim_video(
                            f'md5-{mutation_name}',
                            str(video_path),
                            pipeline_id=PIPELINE_FACE,
                            source_snapshot=snapshot,
                        )
                    )
                    mutate(video_path, snapshot)
                    current = utils_module._source_snapshot(str(video_path))
                    self.assertTrue(
                        current['size'] != snapshot['size']
                        or current['mtime_ns'] != snapshot['mtime_ns']
                    )

                    self.assertFalse(
                        index.complete_claimed_video(
                            f'md5-{mutation_name}',
                            str(video_path),
                            pipeline_id=PIPELINE_FACE,
                        )
                    )
                    self.assertFalse(
                        index.is_video_processed_by_md5(
                            f'md5-{mutation_name}', pipeline_id=PIPELINE_FACE
                        )
                    )
                    self.assertEqual(
                        index.conn.execute(
                            '''
                            SELECT COUNT(*) FROM processing_claims
                            WHERE file_md5=?
                            ''',
                            (f'md5-{mutation_name}',),
                        ).fetchone()[0],
                        1,
                    )
                finally:
                    index.close()

    def test_foreign_owner_or_pipeline_cannot_mutate_claim(self):
        """不同 owner 或 pipeline 不得刷新、释放或完成现有 claim。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            db_path = root / 'claims.db'
            video_path = root / 'video.mp4'
            video_path.write_bytes(b'stable-video')
            owner = utils_module.DirectoryIndex(
                str(db_path), owner_token='claim-owner'
            )
            foreign = utils_module.DirectoryIndex(
                str(db_path), owner_token='foreign-owner'
            )
            try:
                self.assertTrue(
                    owner.try_claim_video(
                        'owned-md5',
                        str(video_path),
                        pipeline_id=PIPELINE_FACE,
                    )
                )

                self.assertFalse(
                    foreign.refresh_claim('owned-md5', pipeline_id=PIPELINE_FACE)
                )
                self.assertFalse(
                    foreign.release_claim('owned-md5', pipeline_id=PIPELINE_FACE)
                )
                self.assertFalse(
                    foreign.complete_claimed_video(
                        'owned-md5',
                        str(video_path),
                        pipeline_id=PIPELINE_FACE,
                    )
                )

                self.assertFalse(
                    owner.refresh_claim('owned-md5', pipeline_id=PIPELINE_CUSTOM)
                )
                self.assertFalse(
                    owner.release_claim('owned-md5', pipeline_id=PIPELINE_CUSTOM)
                )
                self.assertFalse(
                    owner.complete_claimed_video(
                        'owned-md5',
                        str(video_path),
                        pipeline_id=PIPELINE_CUSTOM,
                    )
                )

                row = owner.conn.execute(
                    '''
                    SELECT owner_token, pipeline_id
                    FROM processing_claims
                    WHERE file_md5=?
                    ''',
                    ('owned-md5',),
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row['owner_token'], 'claim-owner')
                self.assertEqual(row['pipeline_id'], PIPELINE_FACE)
                self.assertFalse(
                    owner.is_video_processed_by_md5(
                        'owned-md5', pipeline_id=PIPELINE_FACE
                    )
                )
            finally:
                owner.close()
                foreign.close()

    def test_explicit_pipeline_claim_requires_readable_matching_source(self):
        """显式 pipeline 不得领取不可读源文件或已过期快照。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            video_path = root / 'video.mp4'
            missing_path = root / 'missing.mp4'
            video_path.write_bytes(b'first-version')
            stale_snapshot = utils_module._source_snapshot(str(video_path))
            video_path.write_bytes(b'second-version-is-longer')

            index = utils_module.DirectoryIndex(str(root / 'claims.db'))
            try:
                self.assertFalse(
                    index.try_claim_video(
                        'missing-md5',
                        str(missing_path),
                        pipeline_id=PIPELINE_FACE,
                    )
                )
                with mock.patch.object(
                    utils_module,
                    '_source_snapshot',
                    side_effect=PermissionError('源文件不可读'),
                ):
                    self.assertFalse(
                        index.try_claim_video(
                            'unreadable-md5',
                            str(video_path),
                            pipeline_id=PIPELINE_FACE,
                        )
                    )
                self.assertFalse(
                    index.try_claim_video(
                        'stale-md5',
                        str(video_path),
                        pipeline_id=PIPELINE_FACE,
                        source_snapshot=stale_snapshot,
                    )
                )
                self.assertEqual(
                    index.conn.execute(
                        'SELECT COUNT(*) FROM processing_claims'
                    ).fetchone()[0],
                    0,
                )

                fresh_snapshot = utils_module._source_snapshot(str(video_path))
                self.assertTrue(
                    index.try_claim_video(
                        'fresh-md5',
                        str(video_path),
                        pipeline_id=PIPELINE_FACE,
                        source_snapshot=fresh_snapshot,
                    )
                )
            finally:
                index.close()


if __name__ == '__main__':
    unittest.main()
