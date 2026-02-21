"""从视频中按时间定位提取连续帧到文件夹（基于 ffmpeg）"""

import os
import sys
import json
import subprocess

# ffmpeg / ffprobe 路径
FFMPEG_BIN = r'C:\project\DouyinLiveRecorder\ffmpeg-7.0-essentials_build\bin'
FFMPEG = os.path.join(FFMPEG_BIN, 'ffmpeg.exe')
FFPROBE = os.path.join(FFMPEG_BIN, 'ffprobe.exe')


def clean_path(raw_path):
    """清理拖拽到终端时路径可能带有的引号和空白"""
    p = raw_path.strip()
    # 去掉首尾的单引号或双引号
    if (p.startswith('"') and p.endswith('"')) or \
       (p.startswith("'") and p.endswith("'")):
        p = p[1:-1]
    return p.strip()


def parse_position(user_input):
    """
    解析用户输入的时间定位，返回起始秒数。

    支持两种格式（纯数字，不包含文字）：
      1) 时间格式  HH:MM:SS.mmm   例如 01:30:11.467
         也兼容 MM:SS.mmm 和 SS.mmm
      2) 帧格式    帧数,帧率       例如 110632,20.444
    """
    text = user_input.strip()

    # 帧格式：包含逗号 → 帧数,帧率
    if ',' in text:
        parts = text.split(',')
        if len(parts) != 2:
            raise ValueError(f'帧格式应为 帧数,帧率，收到: {text}')
        frame_num = int(parts[0].strip())
        fps = float(parts[1].strip())
        if fps <= 0:
            raise ValueError(f'帧率必须大于0，收到: {fps}')
        return frame_num / fps

    # 时间格式：包含冒号 → HH:MM:SS.mmm
    if ':' in text:
        parts = text.split(':')
        if len(parts) == 3:
            h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
            return h * 3600 + m * 60 + s
        elif len(parts) == 2:
            m, s = int(parts[0]), float(parts[1])
            return m * 60 + s
        else:
            raise ValueError(f'时间格式应为 HH:MM:SS.mmm 或 MM:SS.mmm，收到: {text}')

    # 纯数字：直接当作秒数
    return float(text)


def seconds_to_display(seconds):
    """将秒数转换为 HH:MM:SS.mmm 的可读字符串"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f'{h:02d}:{m:02d}:{s:06.3f}'


def seconds_to_ffmpeg_time(seconds):
    """将秒数转换为 ffmpeg 的时间格式 HH:MM:SS.mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f'{h:02d}:{m:02d}:{s:06.3f}'


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
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        if result.returncode != 0:
            print(f'ffprobe 出错：{result.stderr}')
            return None, None, None
        info = json.loads(result.stdout)
    except Exception as e:
        print(f'ffprobe 执行失败：{e}')
        return None, None, None

    # 从视频流中提取信息
    for stream in info.get('streams', []):
        if stream.get('codec_type') == 'video':
            # 帧率：优先 r_frame_rate，备选 avg_frame_rate
            fps_str = stream.get('r_frame_rate', stream.get('avg_frame_rate', '0/1'))
            num, den = fps_str.split('/')
            fps = float(num) / float(den) if float(den) != 0 else 0

            # 总帧数
            total_frames = int(stream.get('nb_frames', 0)) if stream.get('nb_frames') else 0

            # 时长：优先流的 duration，备选 format 的 duration
            duration = float(stream.get('duration', 0)) if stream.get('duration') else 0
            if duration <= 0:
                duration = float(info.get('format', {}).get('duration', 0))

            # 如果没有 nb_frames，用 duration * fps 估算
            if total_frames <= 0 and fps > 0 and duration > 0:
                total_frames = int(duration * fps)

            return fps, total_frames, duration

    return None, None, None


def extract_frames(video_path, start_seconds, duration=2.0):
    """
    用 ffmpeg 从视频 video_path 的 start_seconds 位置开始，提取 duration 秒内的所有帧。
    帧保存到以视频文件名命名的文件夹中。
    """
    if not os.path.isfile(video_path):
        print(f'错误：文件不存在 → {video_path}')
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

    end_seconds = start_seconds + duration
    if end_seconds > total_duration:
        actual_duration = total_duration - start_seconds
        print(f'注意：视频剩余不足 {duration}s，实际提取 {actual_duration:.3f}s')
        duration = actual_duration

    start_frame = int(start_seconds * fps)
    end_frame = int((start_seconds + duration) * fps)
    frames_to_extract = end_frame - start_frame

    print(f'提取范围：{seconds_to_display(start_seconds)} → {seconds_to_display(start_seconds + duration)}')
    print(f'预计帧数：{frames_to_extract} (帧 {start_frame} → {end_frame - 1})')
    print()

    # 在视频所在目录下创建输出文件夹，以视频文件名（不含扩展名）命名
    video_dir = os.path.dirname(os.path.abspath(video_path))
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(video_dir, video_stem)
    os.makedirs(output_dir, exist_ok=True)
    print(f'输出目录：{output_dir}')

    # 输出文件名模式：帧序号.png（ffmpeg 从1开始编号，后面重命名）
    temp_pattern = os.path.join(output_dir, '%08d.png')

    # 用 ffmpeg 提取帧
    # -ss 放在 -i 前面实现快速定位（input seeking）
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

    # 重命名：将 ffmpeg 的序号(1,2,3...) 改为 实际帧号_时间戳
    extracted_files = sorted([
        f for f in os.listdir(output_dir)
        if f.endswith('.png') and f[:8].isdigit()
    ])

    saved_count = 0
    for seq_name in extracted_files:
        seq_num = int(os.path.splitext(seq_name)[0])  # ffmpeg从1开始
        actual_frame = start_frame + (seq_num - 1)
        actual_time = actual_frame / fps
        new_name = f'{actual_frame:08d}_{seconds_to_display(actual_time).replace(":", ".")}.png'

        old_path = os.path.join(output_dir, seq_name)
        new_path = os.path.join(output_dir, new_name)
        try:
            os.rename(old_path, new_path)
        except OSError:
            pass  # 重命名失败就保留原名
        saved_count += 1

    print()
    print(f'完成！共保存 {saved_count} 帧到 {output_dir}')


def main():
    """主入口：交互式获取视频路径和时间定位"""
    print('=== 视频帧提取工具 ===')
    print('从指定时间点提取连续帧到文件夹')
    print()

    # 检查 ffmpeg 是否存在
    if not os.path.isfile(FFMPEG):
        print(f'错误：找不到 ffmpeg → {FFMPEG}')
        print('请修改脚本顶部的 FFMPEG_BIN 路径')
        return
    if not os.path.isfile(FFPROBE):
        print(f'错误：找不到 ffprobe → {FFPROBE}')
        print('请修改脚本顶部的 FFMPEG_BIN 路径')
        return

    # 获取视频路径
    if len(sys.argv) > 1:
        # 支持命令行传参
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

    # 循环提取，支持多次输入不同时间点
    while True:
        print('请输入时间定位（支持格式如下）：')
        print('  时间格式: 01:30:11.467')
        print('  帧格式:   110632,20.444')
        print('  输入 q 退出')
        print()
        user_input = input('> ').strip()

        if user_input.lower() == 'q':
            print('退出')
            break

        if not user_input:
            print('错误：请输入时间定位')
            print()
            continue

        try:
            start_seconds = parse_position(user_input)
        except ValueError as e:
            print(f'输入格式错误：{e}')
            print()
            continue

        print(f'定位到: {seconds_to_display(start_seconds)}')

        # 输入提取时长，直接回车默认2秒
        dur_input = input('提取时长（秒），直接回车默认2秒: ').strip()
        if dur_input:
            try:
                duration = float(dur_input)
                if duration <= 0:
                    print('时长必须大于0，使用默认值2秒')
                    duration = 2.0
            except ValueError:
                print(f'无法解析时长 "{dur_input}"，使用默认值2秒')
                duration = 2.0
        else:
            duration = 2.0

        print()
        extract_frames(video_path, start_seconds, duration=duration)
        print()


if __name__ == '__main__':
    main()
