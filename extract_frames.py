"""从视频中按时间定位提取连续帧到文件夹，同时剪辑对应片段（基于 ffmpeg）"""

import os
import sys
import json
import subprocess
import hashlib
import math

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
FFPROBE = resolve_media_binary('ffprobe')


def probe_video(video_path):
    """用 ffprobe 获取视频信息，返回 (fps, total_frames, duration)"""
    cmd = [
        FFPROBE,
        '-v', 'quiet',
        '-print_format', 'json',
        '-show_streams',
        '-show_format',
        video_path
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding='utf-8'
        )
        if result.returncode != 0:
            print(f'ffprobe 出错：{result.stderr}')
            return None, None, None
        info = json.loads(result.stdout)
        if not isinstance(info, dict):
            raise ValueError('ffprobe 返回的 JSON 根节点不是对象')

        streams = info.get('streams', [])
        if not isinstance(streams, list):
            raise ValueError('ffprobe streams 字段不是数组')
        format_info = info.get('format', {})
        if not isinstance(format_info, dict):
            format_info = {}

        for stream in streams:
            if not isinstance(stream, dict) or stream.get('codec_type') != 'video':
                continue

            fps = None
            for rate_key in ('r_frame_rate', 'avg_frame_rate'):
                rate_text = str(stream.get(rate_key, '')).strip()
                try:
                    numerator, denominator = rate_text.split('/', 1)
                    numerator = float(numerator)
                    denominator = float(denominator)
                    candidate = numerator / denominator
                except (TypeError, ValueError, ZeroDivisionError, OverflowError):
                    continue
                if math.isfinite(candidate) and candidate > 0:
                    fps = candidate
                    break
            if fps is None:
                raise ValueError('视频帧率字段无效')

            duration = None
            for raw_duration in (
                stream.get('duration'),
                format_info.get('duration'),
            ):
                try:
                    candidate = float(raw_duration)
                except (TypeError, ValueError, OverflowError):
                    continue
                if math.isfinite(candidate) and candidate > 0:
                    duration = candidate
                    break
            if duration is None:
                raise ValueError('视频时长字段无效')

            try:
                total_frames = int(stream.get('nb_frames', 0))
            except (TypeError, ValueError, OverflowError):
                total_frames = 0
            if total_frames <= 0:
                total_frames = int(duration * fps)
            if total_frames <= 0:
                raise ValueError('视频总帧数字段无效')

            return fps, total_frames, duration
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f'ffprobe 信息解析失败：{exc}')
        return None, None, None

    return None, None, None


def build_frame_output_dir(video_path, start_seconds, end_seconds):
    """按源文件身份和提取区间生成独立输出目录。"""
    start_seconds, end_seconds = validate_time_range(
        start_seconds, end_seconds
    )
    absolute_path = os.path.abspath(video_path)
    normalized_path = os.path.normcase(os.path.realpath(absolute_path))
    try:
        source_stat = os.stat(absolute_path)
        identity_source = (
            f'{normalized_path}\0{source_stat.st_size}\0{source_stat.st_mtime_ns}'
        )
    except OSError:
        identity_source = normalized_path
    source_id = hashlib.sha256(identity_source.encode('utf-8')).hexdigest()[:12]
    source_name = os.path.basename(video_path)
    start_tag = time_to_filename_safe(start_seconds)
    end_tag = time_to_filename_safe(end_seconds)
    directory_name = (
        f'{source_name}_{source_id}_{start_tag}-{end_tag}_frames'
    )
    return os.path.join(os.path.dirname(absolute_path), directory_name)


def parse_end_input(user_input, start_seconds):
    """
    解析结束时间输入。
    支持与开始时间相同的所有格式（时间/帧/秒数）。
    自动判断：如果解析结果 > start_seconds，当作结束时间点；否则当作时长。
    直接回车默认2秒。
    """
    start_seconds = parse_position(str(start_seconds))
    text = user_input.strip()
    if not text:
        return start_seconds + 2.0

    value = parse_position(text)

    # 智能判断：值大于开始时间 → 当作结束时间点；否则当作时长
    if value > start_seconds:
        return value
    else:
        return start_seconds + value


def clip_video(video_path, start_seconds, end_seconds):
    """
    用 ffmpeg 剪辑视频片段，-c copy 不重编码。
    输出文件在视频同目录下，文件名追加起止时间。
    """
    try:
        start_seconds, end_seconds = validate_time_range(
            start_seconds, end_seconds
        )
    except ValueError as exc:
        print(f'剪辑参数错误：{exc}')
        return

    duration = end_seconds - start_seconds
    video_dir = os.path.dirname(os.path.abspath(video_path))
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    video_ext = os.path.splitext(os.path.basename(video_path))[1]

    start_tag = time_to_filename_safe(start_seconds)
    end_tag = time_to_filename_safe(end_seconds)
    out_name = f'{video_stem}_{start_tag}-{end_tag}{video_ext}'
    out_path = os.path.join(video_dir, out_name)

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

    print(f'正在剪辑视频片段...')
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0:
            print(f'剪辑出错：')
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return
    except Exception as e:
        print(f'剪辑执行失败：{e}')
        return

    if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f'剪辑完成！{size_mb:.2f} MB → {out_path}')
    else:
        print('剪辑失败：输出文件为空')


def extract_frames(video_path, start_seconds, end_seconds):
    """
    用 ffmpeg 提取 [start_seconds, end_seconds) 范围内的所有帧。
    同时剪辑输出对应的视频片段。
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

    # 获取视频信息
    fps, total_frames, total_duration = probe_video(video_path)
    if not fps or fps <= 0:
        print('错误：无法获取视频帧率')
        return

    print(f'视频信息：')
    print(f'  帧率: {fps:.3f}')
    print(f'  总帧数: {total_frames}')
    print(f'  总时长: {seconds_to_display(total_duration)}')
    print()

    if start_seconds >= total_duration:
        print(f'错误：起始位置 {seconds_to_display(start_seconds)} 超出视频总时长 {seconds_to_display(total_duration)}')
        return

    if end_seconds > total_duration:
        print(f'注意：结束时间超出视频总时长，截断到 {seconds_to_display(total_duration)}')
        end_seconds = total_duration

    duration = end_seconds - start_seconds
    start_frame = int(start_seconds * fps)
    end_frame = int(end_seconds * fps)
    frames_to_extract = end_frame - start_frame

    print(f'提取范围：{seconds_to_display(start_seconds)} → {seconds_to_display(end_seconds)}')
    print(f'预计帧数：{frames_to_extract} (帧 {start_frame} → {end_frame - 1})')
    print()

    # 按源文件身份和区间隔离，避免同 stem 或不同提取任务互相污染。
    output_dir = build_frame_output_dir(
        video_path, start_seconds, end_seconds
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f'输出目录：{output_dir}')

    # --- 提取帧 ---
    # 上次失败可能残留纯序号临时帧；必须先清理，避免计入本轮。
    pending_frames = [
        name
        for name in os.listdir(output_dir)
        if name.lower().endswith('.png')
        and os.path.splitext(name)[0].isdigit()
    ]
    for pending_name in pending_frames:
        pending_path = os.path.join(output_dir, pending_name)
        try:
            os.remove(pending_path)
        except OSError as exc:
            print(f'错误：无法清理旧临时帧 {pending_path}：{exc}')
            return

    temp_pattern = os.path.join(output_dir, '%08d.png')

    cmd = [
        FFMPEG,
        '-hide_banner',
        '-ss', seconds_to_ffmpeg_time(start_seconds),
        '-i', video_path,
        '-t', str(duration),
        '-vsync', '0',
        '-qscale:v', '2',
        temp_pattern,
        '-y'
    ]

    print(f'正在提取帧...')
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0:
            print(f'ffmpeg 出错：')
            print(result.stderr[-500:] if len(result.stderr) > 500 else result.stderr)
            return
    except Exception as e:
        print(f'ffmpeg 执行失败：{e}')
        return

    # 重命名：将 ffmpeg 序号 → 实际帧号_时间戳
    # ffmpeg 输出的文件名是纯8位数字如 00000001.png，过滤掉之前已重命名的文件（含下划线）
    extracted_files = sorted([
        f for f in os.listdir(output_dir)
        if f.endswith('.png') and os.path.splitext(f)[0].isdigit()
    ])

    saved_count = 0
    for seq_name in extracted_files:
        seq_num = int(os.path.splitext(seq_name)[0])
        actual_frame = start_frame + (seq_num - 1)
        actual_time = actual_frame / fps
        new_name = f'{actual_frame:08d}_{seconds_to_display(actual_time).replace(":", ".")}.png'

        old_path = os.path.join(output_dir, seq_name)
        new_path = os.path.join(output_dir, new_name)
        try:
            os.rename(old_path, new_path)
        except OSError as exc:
            print(f'重命名失败：{old_path} → {new_path}：{exc}')
            continue
        saved_count += 1

    print(f'帧提取完成！共保存 {saved_count} 帧')
    print()

    # --- 剪辑视频片段 ---
    clip_video(video_path, start_seconds, end_seconds)


def main():
    """主入口：交互式获取视频路径和时间范围"""
    print('=== 视频帧提取 + 剪辑工具 ===')
    print('提取指定范围的所有帧，同时剪辑对应视频片段')
    print()

    # 检查 ffmpeg
    if not FFMPEG or not os.path.isfile(FFMPEG):
        print(f'错误：找不到 ffmpeg → {FFMPEG}')
        print('请配置 FINDINVIDEO_FFMPEG_BIN 或将 ffmpeg 加入 PATH')
        return
    if not FFPROBE or not os.path.isfile(FFPROBE):
        print(f'错误：找不到 ffprobe → {FFPROBE}')
        print('请配置 FINDINVIDEO_FFMPEG_BIN 或将 ffprobe 加入 PATH')
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

    # 循环提取
    while True:
        print('请输入开始时间（支持格式如下）：')
        print('  时间格式: 01:30:11.467')
        print('  帧格式:   110632,20.444')
        print('  直接回车: 提取整个视频所有帧')
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

        # 空回车 → 提取全部帧
        if not start_input:
            print('未输入开始时间，将提取整个视频的所有帧并剪辑')
            print()
            _, _, total_duration = probe_video(video_path)
            if not total_duration or total_duration <= 0:
                print('错误：无法获取视频时长')
                print()
                continue
            extract_frames(video_path, 0.0, total_duration)
            print()
            continue

        try:
            start_seconds = parse_position(start_input)
        except ValueError as e:
            print(f'输入格式错误：{e}')
            print()
            continue

        print(f'开始: {seconds_to_display(start_seconds)}')

        # 输入结束时间，支持所有格式，直接回车默认2秒
        print('请输入结束时间（同样格式），直接回车默认往后2秒:')
        end_input = input('结束> ').strip()

        try:
            end_seconds = parse_end_input(end_input, start_seconds)
        except ValueError as e:
            print(f'输入格式错误：{e}')
            print()
            continue

        print(f'结束: {seconds_to_display(end_seconds)} (时长 {end_seconds - start_seconds:.3f}s)')
        print()

        extract_frames(video_path, start_seconds, end_seconds)
        print()


if __name__ == '__main__':
    main()
