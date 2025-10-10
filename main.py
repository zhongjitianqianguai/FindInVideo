from ultralytics import YOLO
import cv2
import numpy as np
import os
from tqdm import tqdm  # 新增导入tqdm库用于进度条显示
import hashlib
import subprocess
import errno
import ntpath


def is_windows_style_path(path):
    """Return True if the path looks like a Windows drive path (e.g. C:\\)."""
    if not path or not isinstance(path, str):
        return False
    if len(path) < 2:
        return False
    drive, sep = path[0], path[1]
    return drive.isalpha() and sep == ':'


def windows_path_to_wsl(path):
    """Convert a Windows path to a WSL-accessible path."""
    if not path:
        return None
    try:
        result = subprocess.run(['wslpath', '-a', path], check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding='utf-8', errors='ignore')
        converted = result.stdout.strip()
        return converted or None
    except Exception:
        if is_windows_style_path(path):
            drive = path[0].lower()
            remainder = path[2:].replace('\\', '/').lstrip('/')
            return f"/mnt/{drive}/{remainder}" if remainder else f"/mnt/{drive}"
        return None


def wsl_path_to_windows(path):
    """Convert a WSL path back to Windows format."""
    if not path:
        return None
    try:
        result = subprocess.run(['wslpath', '-w', path], check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, encoding='utf-8', errors='ignore')
        converted = result.stdout.strip()
        return converted or None
    except Exception:
        # Best-effort manual conversion for /mnt/<drive>/...
        prefix = '/mnt/'
        if path.startswith(prefix) and len(path) > len(prefix):
            drive = path[len(prefix)]
            if drive.isalpha():
                remainder = path[len(prefix) + 1:].replace('/', '\\')
                return f"{drive.upper()}:\\{remainder}" if remainder else f"{drive.upper()}:\\"
        return None


def normalize_posix_path_with_fs(path):
    """Resolve symlinks and normalize a POSIX path if the filesystem allows."""
    if not path:
        return None
    try:
        resolved = os.path.realpath(path)
        return resolved
    except OSError:
        return os.path.normpath(path)


def safe_artifact_basename(video_path, max_length=80):
    """Generate a filesystem-friendly, bounded-length base name for artifacts."""
    base_name = os.path.basename(os.path.splitext(video_path)[0])
    sanitized = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in base_name)
    if not sanitized:
        sanitized = 'video'
    digest = hashlib.md5(str(video_path).encode('utf-8', 'ignore')).hexdigest()[:8]
    limit = max(8, max_length - len(digest) - 1)
    if len(sanitized) > limit:
        sanitized = sanitized[:limit]
    return f"{sanitized}_{digest}"

# 常见的视频文件扩展名
VIDEO_EXTENSIONS = {
    '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v', 
    '.mpg', '.mpeg', '.3gp', '.f4v', '.ts', '.vob', '.rmvb', '.rm',
    '.asf', '.divx', '.xvid', '.m2ts', '.mts'
}

# 需要排除的路径变量
EXCLUDE_PATHS = [
    os.path.abspath(r"D:\$RECYCLE.BIN"),
    os.path.abspath(r"D:\System Volume Information"),
    # 在这里添加其他需要排除的路径
]

def is_video_file(file_path):
    """检查文件是否为视频文件"""
    _, ext = os.path.splitext(file_path.lower())
    return ext in VIDEO_EXTENSIONS

def is_leaf_directory(dir_path):
    """检查目录是否为叶子节点（不包含子目录）"""
    try:
        for item in os.listdir(dir_path):
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path):
                return False
        return True
    except (PermissionError, FileNotFoundError):
        return False

def count_videos_in_directory(dir_path):
    """统计目录中的视频文件数量"""
    count = 0
    try:
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path) and is_video_file(file_path):
                count += 1
    except (PermissionError, FileNotFoundError):
        pass
    return count

def find_leaf_directories_with_videos(root_path, exclusions=None):
    """
    递归查找所有包含视频文件的叶子节点目录
    返回: [(目录路径, 视频数量), ...] 按视频数量降序排列
    """
    if exclusions is None:
        exclusions = EXCLUDE_PATHS
    
    leaf_dirs = []
    
    # 检查根路径是否被排除
    for ex_path in exclusions:
        if root_path.startswith(ex_path):
            return leaf_dirs
    
    try:
        for root, dirs, files in os.walk(root_path):
            # 检查当前目录是否被排除
            is_excluded = any(root.startswith(ex_path) for ex_path in exclusions)
            if is_excluded:
                continue
            
            # 过滤掉被排除的子目录
            dirs[:] = [d for d in dirs if not any(os.path.join(root, d).startswith(ex_path) for ex_path in exclusions)]
            
            # 检查是否为叶子节点
            if is_leaf_directory(root):
                video_count = count_videos_in_directory(root)
                if video_count > 0:
                    leaf_dirs.append((root, video_count))
    
    except (PermissionError, FileNotFoundError) as e:
        print(f"警告: 无法访问目录 '{root_path}': {e}")
    
    # 按视频数量降序排列
    leaf_dirs.sort(key=lambda x: x[1], reverse=True)
    return leaf_dirs

def detect_objects_in_video(video_path, target_class,
                            show_window=False, save_crops=False,
                            save_training_data=False,
                            all_objects=False):
    # ...existing code...
    # 如果不开启全量检测，则保证 target_class 为列表
    if not all_objects and isinstance(target_class, str):
        target_class = [target_class]

    # 加载模型
    model = YOLO('models/yolov11l-face.pt')

    video_dir = os.path.dirname(video_path)
    artifact_base = safe_artifact_basename(video_path)
    txt_save_path = os.path.join(video_dir, artifact_base + '.txt')
    mosaic_path = os.path.join(video_dir, artifact_base + '_mosaic.jpg')

    # 若需要生成训练数据，则构造保存目录及生成 classes.txt 文件
    if save_training_data:
        training_folder = os.path.join(video_dir, "training_data")
        os.makedirs(training_folder, exist_ok=True)
        classes_txt = os.path.join(training_folder, "classes.txt")
        if not os.path.exists(classes_txt):
            with open(classes_txt, "w") as f:
                if all_objects:
                    # 写入模型中所有的类别，按 key 排序
                    for key in sorted(model.names.keys()):
                        f.write(model.names[key] + "\n")
                else:
                    for cls in target_class:
                        f.write(cls + "\n")
    
    # 视频处理初始化
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # 初始化进度条
    pbar = tqdm(total=total_frames, desc=f"处理视频: {os.path.basename(video_path)}")
    
    frame_count = 0
    last_detected = -5
    detections = []
    
    # 截图存储配置（用于拼接大图）
    crop_size = (160, 160)  # 统一缩放到的小图尺寸
    max_cols = 8  # 拼接大图每行最多显示数量
    crops = []  # 存储所有截取的目标区域
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # 更新进度条
        pbar.update(1)
        
        if save_training_data:
            frame_annotations = []
        
        current_time = frame_count / fps
        results = model.predict(frame, conf=0.5, verbose=False)
        detected = False
        
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                # 如果全量检测开启，或者检测到的类别在 target_class 内则处理
                if all_objects or model.names[cls_id] in target_class:
                    if current_time - last_detected >= 0.1:
                        detections.append(current_time)
                        last_detected = current_time
                        # 注释掉秒数打印输出
                        # print(f"{video_path}: 检测到 {model.names[cls_id]} @ {current_time:.2f}秒")
                        detected = True
                    
                    if save_crops:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:
                            resized = cv2.resize(crop, crop_size)
                            crops.append(resized)
                    
                    if show_window:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    if save_training_data:
                        h, w, _ = frame.shape
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        cx = ((x1 + x2) / 2) / w
                        cy = ((y1 + y2) / 2) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        # 如果全量检测，类别编号直接用 cls_id；否则使用 target_class 的索引
                        if all_objects:
                            class_index = cls_id
                        else:
                            class_index = target_class.index(model.names[cls_id])
                        annotation_line = f"{class_index} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                        frame_annotations.append(annotation_line)
        
        if show_window and detected:
            cv2.imshow('Detection Preview', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        if save_training_data and frame_annotations:
            training_base = artifact_base
            training_image_path = os.path.join(training_folder, f"{training_base}_{frame_count}.jpg")
            training_annotation_path = os.path.splitext(training_image_path)[0] + ".txt"
            cv2.imwrite(training_image_path, frame)
            with open(training_annotation_path, 'w') as f:
                for line in frame_annotations:
                    f.write(line + "\n")
            # print(f"保存训练图片及标注: {training_image_path} 和 {training_annotation_path}")
        
        frame_count += 1
    
    cap.release()
    pbar.close()
    if show_window:
        cv2.destroyAllWindows()
    
    with open(txt_save_path, 'w') as f:
        f.write("检测到目标的时间位置（秒）:\n")
        for t in detections:
            f.write(f"{t:.2f}\n")
    print(f"已保存检测时间戳至: {txt_save_path}")
    
    if save_crops and crops:
        rows = []
        row = []
        for i, crop in enumerate(crops):
            row.append(crop)
            if (i + 1) % max_cols == 0:
                rows.append(np.hstack(row))
                row = []
        if row:
            missing = max_cols - len(row)
            blank = np.zeros_like(row[0])
            row.extend([blank] * missing)
            rows.append(np.hstack(row))
        final_mosaic = np.vstack(rows)
        
        if show_window:
            cv2.imshow('All Detected Faces', final_mosaic)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            cv2.imwrite(mosaic_path, final_mosaic)
            print(f"已保存拼接图片至: {mosaic_path}")
    
    return detections

def should_process(file_path):
    dir_name = os.path.dirname(file_path)
    artifact_base = safe_artifact_basename(file_path)
    mosaic_path_new = os.path.join(dir_name, artifact_base + "_mosaic.jpg")
    legacy_base = os.path.splitext(os.path.basename(file_path))[0]
    mosaic_path_legacy = os.path.join(dir_name, legacy_base + "_mosaic.jpg")
    return not (os.path.exists(mosaic_path_new) or os.path.exists(mosaic_path_legacy))

def process_directory_videos(dir_path, target_item, all_objects_switch=False, skip_long_videos=True):
    """处理目录中的所有视频文件"""
    if os.name == 'posix' and is_windows_style_path(dir_path):
        converted = windows_path_to_wsl(dir_path)
        if converted:
            dir_path = normalize_posix_path_with_fs(converted)
    video_files = []
    
    try:
        for file in os.listdir(dir_path):
            if is_video_file(file):
                file_path = os.path.join(dir_path, file)
                if should_process(file_path):
                    # 检查视频时长
                    cap = cv2.VideoCapture(file_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    cap.release()
                    duration = frame_count / fps if fps > 0 else float('inf')

                    if not skip_long_videos or duration <= 3600:
                        video_files.append((file_path, duration))
                    else:
                        print(f"视频时长 {duration:.2f}秒超过一小时，跳过处理: {file_path}")
                else:
                    print(f"已存在拼接图片，跳过处理: {file_path}")
    except (PermissionError, FileNotFoundError) as e:
        print(f"警告: 无法访问目录 '{dir_path}': {e}")
        return
    
    # 处理视频文件
    for video_file, duration in video_files:
        if duration == float('inf'):
            print(f"提示: 无法获取视频时长，仍尝试处理: {video_file}")
        print(f"开始处理视频文件: {video_file}")
        detect_objects_in_video(video_file, target_item,
                                show_window=False,
                                save_crops=True,
                                save_training_data=False,
                                all_objects=all_objects_switch)


def process_root_directory(root_path, target_item, all_objects_switch, skip_long_videos, use_leaf_node_processing):
    """根据配置处理根目录下的所有视频目录/文件"""
    if os.name == 'posix' and is_windows_style_path(root_path):
        converted_root = windows_path_to_wsl(root_path)
        if converted_root:
            root_path = normalize_posix_path_with_fs(converted_root)
    if not os.path.isdir(root_path):
        print(f"警告: 目录 '{root_path}' 无法通过常规方式访问，尝试继续处理。")
    if use_leaf_node_processing:
        print(f"启用叶子节点处理模式，正在扫描目录: {root_path}")
        print("正在查找包含视频文件的叶子节点目录...")

        leaf_dirs = find_leaf_directories_with_videos(root_path, EXCLUDE_PATHS)

        if not leaf_dirs:
            print("未找到包含视频文件的叶子节点目录")
            return

        print(f"\n找到 {len(leaf_dirs)} 个包含视频文件的叶子节点目录:")
        for i, (dir_path, video_count) in enumerate(leaf_dirs, 1):
            relative_path = os.path.relpath(dir_path, root_path)
            print(f"{i:3d}. {relative_path} ({video_count} 个视频文件)")

        print("\n开始按视频数量从多到少的顺序处理叶子节点目录...")
        for i, (dir_path, video_count) in enumerate(leaf_dirs, 1):
            relative_path = os.path.relpath(dir_path, root_path)
            print(f"\n=== 处理第 {i}/{len(leaf_dirs)} 个目录: {relative_path} ({video_count} 个视频) ===")
            process_directory_videos(dir_path, target_item, all_objects_switch, skip_long_videos)
    else:
        print(f"使用原有目录遍历逻辑处理: {root_path}")
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        for root, dirs, files in os.walk(root_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in video_extensions:
                    file_path = os.path.join(root, file)
                    if should_process(file_path):
                        cap = cv2.VideoCapture(file_path)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        cap.release()
                        duration = frame_count / fps if fps > 0 else float('inf')
                        if skip_long_videos and duration > 3600:
                            print(f"视频时长 {duration:.2f}秒超过一小时，跳过处理: {file_path}")
                            continue
                        if duration == float('inf'):
                            print(f"提示: 无法获取视频时长，仍尝试处理: {file_path}")
                        elif duration > 3600:
                            print(f"提示: 视频时长 {duration:.2f}秒超过一小时，但仍根据配置处理: {file_path}")

                        print(f"开始处理视频文件: {file_path}")
                        detect_objects_in_video(file_path, target_item,
                                                show_window=False,
                                                save_crops=True,
                                                save_training_data=True,
                                                all_objects=all_objects_switch)
                    else:
                        print(f"已存在拼接图片，跳过处理: {file_path}")

def get_file_md5(file_path):
    """计算文件的MD5哈希值，兼容极长的 Windows 路径。"""
    if not file_path:
        return None

    def candidate_paths(original_path):
        seen = set()
        original = str(original_path)
        for variant in (original,):
            if variant and variant not in seen:
                seen.add(variant)
                yield variant
        if os.name == 'posix':
            if is_windows_style_path(original) or '\\' in original:
                converted = windows_path_to_wsl(original)
                for derived in (normalize_posix_path_with_fs(converted) if converted else None, converted):
                    if derived and derived not in seen:
                        seen.add(derived)
                        yield derived
            else:
                normalized = normalize_posix_path_with_fs(original)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    yield normalized

    short_dir = '/tmp/findinvideo_md5'

    def open_stream(path):
        hash_md5 = hashlib.md5()
        with open(path, 'rb') as fh:
            max_bytes = 10 * 1024 * 1024
            read = 0
            for chunk in iter(lambda: fh.read(4096), b''):
                hash_md5.update(chunk)
                read += len(chunk)
                if read >= max_bytes:
                    break
        return hash_md5.hexdigest()

    def attempt_symlink(path):
        os.makedirs(short_dir, exist_ok=True)
        ext = os.path.splitext(path)[1] or ''
        hashed = hashlib.md5(path.encode('utf-8', 'ignore')).hexdigest()
        link_path = os.path.join(short_dir, hashed + ext)
        try:
            if os.path.lexists(link_path):
                if not os.path.islink(link_path) or os.readlink(link_path) != path:
                    os.unlink(link_path)
            if not os.path.lexists(link_path):
                os.symlink(path, link_path)
            digest = open_stream(link_path)
            print(f"[MD5] 通过短路径计算: {link_path}")
            return digest
        except Exception:
            return None

    def windows_copy(path):
        win_src = path if is_windows_style_path(path) else wsl_path_to_windows(path)
        if not win_src:
            return None
        win_src = win_src.replace('/', '\\')
        ext = os.path.splitext(win_src)[1] or '.bin'
        hashed = hashlib.md5(win_src.encode('utf-8', 'ignore')).hexdigest()
        base = os.environ.get('FINDINVIDEO_WIN_TEMP', r'C:\Temp\findinvideo_md5') or r'C:\Temp\findinvideo_md5'
        base = base.replace('/', '\\').rstrip('\\') or r'C:\Temp\findinvideo_md5'
        drive, tail = ntpath.splitdrive(base)
        base = ntpath.join('C:\\', base.lstrip('\\')) if not drive else ntpath.normpath(drive + tail)
        dest_win = ntpath.normpath(ntpath.join(base, hashed + ext))
        dest_dir = ntpath.dirname(dest_win)

        sanitized_dest_dir = dest_dir.replace("'", "''")
        sanitized_src = win_src.replace("'", "''")
        sanitized_dst = dest_win.replace("'", "''")
        ensure_script = (
            "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
            f"$dir='\\\\?\\{sanitized_dest_dir}';"
            "if (-not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }"
        )
        copy_script = (
            "[Console]::OutputEncoding=[System.Text.Encoding]::UTF8;"
            f"$src='\\\\?\\{sanitized_src}';"
            f"$dst='\\\\?\\{sanitized_dst}';"
            "Copy-Item -LiteralPath $src -Destination $dst -Force"
        )
        try:
            subprocess.run(['powershell.exe', '-NoProfile', '-Command', ensure_script],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           text=True, encoding='utf-8', errors='ignore')
            subprocess.run(['powershell.exe', '-NoProfile', '-Command', copy_script],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                           text=True, encoding='utf-8', errors='ignore')
        except subprocess.CalledProcessError as exc:
            msg = exc.stderr.strip() if exc.stderr else str(exc)
            print(f"[MD5] Windows Copy-Item 失败: {msg}")
            return None
        dest_wsl = windows_path_to_wsl(dest_win)
        return dest_wsl

    last_error = None
    for candidate in candidate_paths(file_path):
        try:
            digest = open_stream(candidate)
            if candidate != file_path:
                print(f"[MD5] 使用备用路径计算: {candidate}")
            return digest
        except (FileNotFoundError, NotADirectoryError) as e:
            last_error = e
            continue
        except OSError as e:
            last_error = e
            if os.name != 'posix':
                continue
            need_short = getattr(e, 'errno', None) == errno.ENAMETOOLONG or 'File name too long' in str(e)
            candidate_is_file = False
            try:
                candidate_is_file = os.path.isfile(candidate)
            except OSError as exists_err:
                if getattr(exists_err, 'errno', None) == errno.ENAMETOOLONG:
                    candidate_is_file = True
            if not need_short and len(candidate) >= 240 and candidate_is_file:
                need_short = True
            if need_short and candidate_is_file:
                digest = attempt_symlink(candidate)
                if digest:
                    return digest
            alt = windows_copy(candidate)
            if alt:
                try:
                    digest = open_stream(alt)
                    print(f"[MD5] 通过 Windows 缩短路径计算: {alt}")
                    return digest
                except Exception as copy_err:
                    last_error = copy_err
            continue
        except Exception as e:
            last_error = e
            break

    print(f"计算MD5错误: {file_path}, 错误: {last_error or '未找到可访问路径'}")
    return None

if __name__ == "__main__":
    video_path = r"D:\z\按摩小子 1"  # 可设置为视频文件或目录
    original_video_path = video_path
    if os.name == 'posix' and is_windows_style_path(video_path):
        converted = windows_path_to_wsl(video_path)
        normalized = normalize_posix_path_with_fs(converted) if converted else None
        if normalized:
            video_path = normalized
            print(f"检测到 Windows 路径，已转换为 WSL 路径处理: {video_path}")
        else:
            print(f"警告: 无法转换 Windows 路径 {original_video_path}，将尝试直接使用。")
    # 如要检测所有模型内对象，则将 target_item 设置为任意值并启用全量检测开关
    target_item = "face"  # 当 all_objects 为 True 时，该值不再限制检测
    all_objects_switch = False  # 设置为 True 表示显示所有检测对象
    skip_long_videos = False  # 设置为 True 时跳过超过一小时的视频
    
    # 新增功能：按叶子节点视频数量排序处理
    use_leaf_node_processing = True  # 设置为 True 启用叶子节点处理模式
    
    if os.path.isdir(video_path):
        process_root_directory(video_path, target_item, all_objects_switch, skip_long_videos, use_leaf_node_processing)
    elif os.path.isfile(video_path) and is_video_file(video_path):
        # 处理单个视频文件前检查视频时长
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        duration = frame_count / fps if fps > 0 else float('inf')
        if skip_long_videos and duration > 3600:
            print(f"视频时长 {duration:.2f}秒超过一小时，跳过处理: {video_path}")
        else:
            if duration == float('inf'):
                print(f"提示: 无法获取视频时长，仍尝试处理: {video_path}")
            elif duration > 3600:
                print(f"提示: 视频时长 {duration:.2f}秒超过一小时，但仍根据配置处理: {video_path}")
            if should_process(video_path):
                detect_objects_in_video(video_path, target_item,
                                        show_window=False,
                                        save_crops=True,
                                        save_training_data=True,
                                        all_objects=all_objects_switch)
            else:
                print(f"已存在拼接图片，跳过处理: {video_path}")
    else:
        if not is_video_file(video_path):
            print(f"提示: '{video_path}' 不像是视频文件，按目录进行处理尝试。")
            process_root_directory(video_path, target_item, all_objects_switch, skip_long_videos, use_leaf_node_processing)
        else:
            print(f"路径不存在或无法访问: {video_path}")