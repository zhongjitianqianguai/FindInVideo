"""只读查看 FindInVideo 的持久化处理中断与恢复审计日志。"""

import argparse
import datetime
import json
import os
import sqlite3
import sys


def _parse_args():
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description='查看 directory_index.db 中的处理中断、释放与恢复审计事件',
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        '--root',
        help='处理根目录，例如 G:\\z；数据库位于 <root>\\md5_list\\directory_index.db',
    )
    source_group.add_argument('--db', help='directory_index.db 的完整路径')
    parser.add_argument('--limit', type=int, default=100, help='显示最近事件数，默认 100')
    parser.add_argument('--md5', help='仅显示指定视频 MD5 的事件')
    parser.add_argument('--video', help='仅显示指定视频完整路径的事件')
    parser.add_argument('--pipeline', help='仅显示指定流水线的事件')
    parser.add_argument('--json', action='store_true', help='以 JSON 数组输出')
    return parser.parse_args()


def _resolve_db_path(args):
    """根据参数确定数据库路径。"""
    if args.db:
        return os.path.abspath(args.db)
    return os.path.abspath(os.path.join(args.root, 'md5_list', 'directory_index.db'))


def _open_read_only(db_path):
    """以只读模式打开 SQLite，避免诊断命令修改正在使用的共享数据库。"""
    if not os.path.isfile(db_path):
        raise FileNotFoundError(f'找不到状态数据库: {db_path}')
    uri_path = os.path.abspath(db_path).replace('\\', '/')
    if not uri_path.startswith('/'):
        uri_path = '/' + uri_path
    return sqlite3.connect(f'file:{uri_path}?mode=ro', uri=True)


def _load_events(conn, args):
    """按筛选条件读取最新审计事件。"""
    clauses = []
    params = []
    if args.md5:
        clauses.append('file_md5=?')
        params.append(args.md5)
    if args.video:
        clauses.append('video_path=?')
        params.append(args.video)
    if args.pipeline:
        clauses.append('pipeline_id=?')
        params.append(args.pipeline)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ''
    limit = max(1, min(int(args.limit or 100), 10000))
    rows = conn.execute(
        f"""
        SELECT id, occurred_at, event_type, file_md5, pipeline_id, video_path,
               host_name, pid, details_json
        FROM processing_events{where}
        ORDER BY id DESC
        LIMIT ?
        """,
        (*params, limit),
    ).fetchall()
    events = []
    for row in reversed(rows):
        item = dict(zip(
            ('id', 'occurred_at', 'event_type', 'file_md5', 'pipeline_id',
             'video_path', 'host_name', 'pid', 'details_json'),
            row,
        ))
        try:
            item['details'] = json.loads(item.pop('details_json') or '{}')
        except (TypeError, ValueError, json.JSONDecodeError):
            item['details'] = {}
            item.pop('details_json', None)
        events.append(item)
    return events


def _format_time(timestamp):
    """将 Unix 时间戳格式化为本地可读时间。"""
    try:
        return datetime.datetime.fromtimestamp(float(timestamp)).astimezone().strftime(
            '%Y-%m-%d %H:%M:%S%z'
        )
    except (TypeError, ValueError, OSError, OverflowError):
        return '未知时间'


def _print_events(events):
    """以便于人工核查的形式输出事件链。"""
    if not events:
        print('没有匹配的持久化处理审计事件。')
        return
    for event in events:
        print(
            f"[{_format_time(event['occurred_at'])}] "
            f"#{event['id']} {event['event_type']}"
        )
        if event.get('video_path'):
            print(f"  视频: {event['video_path']}")
        if event.get('file_md5'):
            print(f"  MD5: {event['file_md5']}")
        print(f"  流水线: {event.get('pipeline_id') or 'legacy-v1'}")
        if event.get('details'):
            print(
                '  详情: ' + json.dumps(
                    event['details'], ensure_ascii=False, sort_keys=True,
                )
            )


def main():
    """运行只读审计日志查询。"""
    args = _parse_args()
    db_path = _resolve_db_path(args)
    try:
        with _open_read_only(db_path) as conn:
            events = _load_events(conn, args)
    except FileNotFoundError as e:
        print(f'读取失败: {e}', file=sys.stderr)
        return 2
    except sqlite3.OperationalError as e:
        if 'no such table' in str(e).lower():
            print('读取失败: 当前状态数据库尚未被新版程序写入审计表。', file=sys.stderr)
            return 3
        print(f'读取失败: 无法打开或查询状态数据库: {e}', file=sys.stderr)
        return 4
    if args.json:
        print(json.dumps(events, ensure_ascii=False, indent=2))
    else:
        _print_events(events)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
