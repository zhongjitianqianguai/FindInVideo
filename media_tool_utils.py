"""媒体命令行工具共享的输入校验与可执行文件解析。"""

import math
import os
import shutil


LEGACY_FFMPEG_BIN = (
    r'C:\project\DouyinLiveRecorder\ffmpeg-7.0-essentials_build\bin'
)


def clean_path(raw_path):
    """清理拖拽到终端时路径可能带有的引号和空白。"""
    path = str(raw_path).strip()
    if (
        (path.startswith('"') and path.endswith('"'))
        or (path.startswith("'") and path.endswith("'"))
    ):
        path = path[1:-1]
    return path.strip()


def looks_like_path(text):
    """判断输入是否像一个文件路径。"""
    cleaned = clean_path(text)
    if os.path.isfile(cleaned):
        return True
    if ('\\' in cleaned or '/' in cleaned) and '.' in cleaned:
        return True
    return len(cleaned) >= 2 and cleaned[1] == ':'


def _binary_file_name(binary_name):
    """返回当前平台常用的可执行文件名。"""
    if os.name == 'nt' and not binary_name.lower().endswith('.exe'):
        return f'{binary_name}.exe'
    return binary_name


def _binary_from_environment(binary_name):
    """从 FINDINVIDEO_FFMPEG_BIN 解析指定工具。"""
    configured = clean_path(os.environ.get('FINDINVIDEO_FFMPEG_BIN', ''))
    if not configured:
        return None

    file_name = _binary_file_name(binary_name)
    if os.path.isdir(configured):
        candidate = os.path.join(configured, file_name)
        return candidate if os.path.isfile(candidate) else None

    if os.path.isfile(configured):
        configured_stem = os.path.splitext(os.path.basename(configured))[0]
        if configured_stem.lower() == binary_name.lower():
            return configured
        sibling = os.path.join(os.path.dirname(configured), file_name)
        return sibling if os.path.isfile(sibling) else None

    return None


def resolve_media_binary(binary_name):
    """按环境变量、PATH、旧版兼容目录的顺序解析媒体工具。"""
    configured = _binary_from_environment(binary_name)
    if configured:
        return configured

    path_binary = shutil.which(binary_name)
    if path_binary:
        return path_binary

    return os.path.join(LEGACY_FFMPEG_BIN, _binary_file_name(binary_name))


def _non_negative_finite(value, label):
    """将数值转为有限非负浮点数。"""
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{label}必须是有效数字，收到: {value}') from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f'{label}必须是有限非负数，收到: {value}')
    return number


def parse_position(user_input):
    """解析时间、秒数或“帧数,帧率”输入并返回秒数。"""
    text = str(user_input).strip()
    if not text:
        raise ValueError('时间输入不能为空')

    if ',' in text:
        parts = text.split(',')
        if len(parts) != 2:
            raise ValueError(f'帧格式应为 帧数,帧率，收到: {text}')
        try:
            frame_num = int(parts[0].strip())
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'帧数必须是非负整数，收到: {parts[0]}') from exc
        if frame_num < 0:
            raise ValueError(f'帧数必须是非负整数，收到: {frame_num}')
        fps = _non_negative_finite(parts[1].strip(), '帧率')
        if fps <= 0:
            raise ValueError(f'帧率必须大于0，收到: {fps}')
        try:
            return _non_negative_finite(frame_num / fps, '时间')
        except OverflowError as exc:
            raise ValueError('换算后的时间超出有效范围') from exc

    if ':' in text:
        parts = text.split(':')
        if len(parts) not in (2, 3):
            raise ValueError(
                f'时间格式应为 HH:MM:SS.mmm 或 MM:SS.mmm，收到: {text}'
            )
        values = [
            _non_negative_finite(part.strip(), '时间分量')
            for part in parts
        ]
        if len(values) == 3:
            hours, minutes, seconds = values
        else:
            hours = 0.0
            minutes, seconds = values
        return _non_negative_finite(
            hours * 3600 + minutes * 60 + seconds,
            '时间',
        )

    return _non_negative_finite(text, '时间')


def validate_time_range(start_seconds, end_seconds):
    """验证起止时间并返回规范化后的浮点数。"""
    start = _non_negative_finite(start_seconds, '开始时间')
    end = _non_negative_finite(end_seconds, '结束时间')
    if end <= start:
        raise ValueError('结束时间必须大于开始时间')
    return start, end


def seconds_to_display(seconds):
    """将秒数转换为 HH:MM:SS.mmm 的可读字符串。"""
    seconds = _non_negative_finite(seconds, '时间')
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining = seconds % 60
    return f'{hours:02d}:{minutes:02d}:{remaining:06.3f}'


def seconds_to_ffmpeg_time(seconds):
    """将秒数转换为 ffmpeg 时间参数格式。"""
    return seconds_to_display(seconds)


def time_to_filename_safe(seconds):
    """将秒数转换为可用于文件名的字符串。"""
    seconds = _non_negative_finite(seconds, '时间')
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    remaining = seconds % 60
    if hours > 0:
        return f'{hours:02d}h{minutes:02d}m{remaining:06.3f}s'
    if minutes > 0:
        return f'{minutes:02d}m{remaining:06.3f}s'
    return f'{remaining:06.3f}s'
