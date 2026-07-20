"""用 ffmpeg 剪辑视频片段（无重编码，直接拷贝流）"""

import os
import sys
import subprocess

from media_tool_utils import (
    clean_path,
    looks_like_path,
    parse_position,
    resolve_media_binary,
    seconds_to_display,
    seconds_to_ffmpeg_time,
    time_to_filename_safe,
    validate_time_range,
)


FFMPEG = resolve_media_binary('ffmpeg')


def clip_video(video_path, start_seconds, end_seconds):
    """
    用 ffmpeg 从 video_path 中剪出 [start_seconds, end_seconds) 片段。
    输出文件在同目录下，文件名追加起止时间。
    使用 -c copy 直接拷贝流，速度极快。
    """
    if not os.path.isfile(video_path):
        print(f'错误：文件不存在 → {video_path}')
        return

    try:
        start_seconds, end_seconds = validate_time_range(
            start_seconds, end_seconds
        )
    except ValueError as exc:
        print(f'错误：{exc}')
        return

    duration = end_seconds - start_seconds
    video_dir = os.path.dirname(os.path.abspath(video_path))
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    video_ext = os.path.splitext(os.path.basename(video_path))[1]

    # 输出文件名：原名_开始时间_结束时间.ext
    start_tag = time_to_filename_safe(start_seconds)
    end_tag = time_to_filename_safe(end_seconds)
    out_name = f'{video_stem}_{start_tag}-{end_tag}{video_ext}'
    out_path = os.path.join(video_dir, out_name)

    print(f'剪辑范围：{seconds_to_display(start_seconds)} → {seconds_to_display(end_seconds)}')
    print(f'片段时长：{duration:.3f}s')
    print(f'输出文件：{out_path}')
    print()

    cmd = [
        FFMPEG,
        '-hide_banner',
        '-ss', seconds_to_ffmpeg_time(start_seconds),
        '-i', video_path,
        '-t', str(duration),
        '-c', 'copy',
        '-avoid_negative_ts', 'make_zero',
        out_path,
        '-y'
    ]

    print('正在剪辑...')
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0:
            print(f'ffmpeg 出错：')
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return
    except Exception as e:
        print(f'ffmpeg 执行失败：{e}')
        return

    # 检查输出文件
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f'完成！输出文件 {size_mb:.2f} MB → {out_path}')
    else:
        print('错误：输出文件生成失败')


def main():
    """主入口：交互式获取视频路径、开始时间、结束时间"""
    print('=== 视频剪辑工具 ===')
    print('从视频中剪出指定时间范围的片段')
    print()

    # 检查 ffmpeg
    if not FFMPEG or not os.path.isfile(FFMPEG):
        print(f'错误：找不到 ffmpeg → {FFMPEG}')
        print('请配置 FINDINVIDEO_FFMPEG_BIN 或将 ffmpeg 加入 PATH')
        return

    # 获取视频路径
    if len(sys.argv) > 1:
        raw_path = ' '.join(sys.argv[1:])
    else:
        print('请将视频文件拖入终端，然后按回车：')
        raw_path = input().strip()

    video_path = clean_path(raw_path)
    if not video_path:
        print('错误：未提供视频路径')
        return

    print(f'视频路径：{video_path}')
    print()

    # 循环剪辑，支持对同一视频多次剪辑
    while True:
        print('请输入开始时间（支持格式如下）：')
        print('  时间格式: 01:30:11.467')
        print('  帧格式:   110632,20.444')
        print('  拖入视频: 切换到新视频')
        print('  输入 q 退出')
        print()
        start_input = input('开始> ').strip()

        if start_input.lower() == 'q':
            print('退出')
            break

        # 检测是否拖入了新视频
        if start_input and looks_like_path(start_input):
            new_path = clean_path(start_input)
            if os.path.isfile(new_path):
                video_path = new_path
                print(f'切换视频：{video_path}')
                print()
                continue
            else:
                print(f'文件不存在：{new_path}')
                print()
                continue

        if not start_input:
            print('错误：请输入开始时间')
            print()
            continue

        try:
            start_seconds = parse_position(start_input)
        except ValueError as e:
            print(f'输入格式错误：{e}')
            print()
            continue

        print(f'开始: {seconds_to_display(start_seconds)}')
        print()

        # 获取结束时间
        end_input = input('结束> ').strip()

        if not end_input:
            print('错误：请输入结束时间')
            print()
            continue

        try:
            end_seconds = parse_position(end_input)
        except ValueError as e:
            print(f'输入格式错误：{e}')
            print()
            continue

        print(f'结束: {seconds_to_display(end_seconds)}')
        print()

        clip_video(video_path, start_seconds, end_seconds)
        print()


if __name__ == '__main__':
    main()
