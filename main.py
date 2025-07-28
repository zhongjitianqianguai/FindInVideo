from ultralytics import YOLO
import cv2
import numpy as np
import os
from tqdm import tqdm
import gc  # 导入垃圾回收模块
import time

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

def save_mosaic_batch(crops_batch, batch_idx, dir_name, base_name, max_cols=8):
    """保存一批检测到的目标区域为拼接图像"""
    if not crops_batch:
        return
    
    rows = []
    row = []
    for i, crop in enumerate(crops_batch):
        row.append(crop)
        if (i + 1) % max_cols == 0:
            rows.append(np.hstack(row))
            row = []
    
    if row:
        missing = max_cols - len(row)
        blank = np.zeros_like(crops_batch[0])
        row.extend([blank] * missing)
        rows.append(np.hstack(row))
    
    if rows:
        batch_mosaic = np.vstack(rows)
        # 根据批次保存不同的文件名
        mosaic_path = os.path.join(dir_name, f"{base_name}_mosaic_{batch_idx}.jpg")
        cv2.imwrite(mosaic_path, batch_mosaic)
        print(f"已保存第{batch_idx}批拼接图片至: {mosaic_path}")

    # 清理内存
    del crops_batch, rows, row, batch_mosaic
    gc.collect()
    
def detect_objects_in_video(video_path, target_class,
                          show_window=False, save_crops=False,
                          save_training_data=False,
                          all_objects=False):
    # 如果不开启全量检测，则保证 target_class 为列表
    if not all_objects and isinstance(target_class, str):
        target_class = [target_class]

    # 加载模型
    model = YOLO('models/yolov11l-face.pt')

    # 若需要生成训练数据，则构造保存目录及生成 classes.txt 文件
    if save_training_data:
        training_folder = os.path.join(os.path.dirname(video_path), "training_data")
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
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return []
        
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
    crops_batch = []  # 存储当前批次的目标区域
    batch_size = 200  # 每批处理的目标数量
    batch_idx = 1  # 批次计数器
    
    try:
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
                            detected = True
                        
                        if save_crops:
                            xyxy = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = map(int, xyxy)
                            # 边界检查
                            h, w = frame.shape[:2]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            
                            if x2 > x1 and y2 > y1:  # 确保有效区域
                                crop = frame[y1:y2, x1:x2].copy()  # 使用.copy()避免引用原始帧
                                try:
                                    resized = cv2.resize(crop, crop_size)
                                    crops_batch.append(resized)
                                    
                                    # 批处理，避免占用过多内存
                                    if len(crops_batch) >= batch_size:
                                        dir_name = os.path.dirname(video_path)
                                        base_name = os.path.splitext(os.path.basename(video_path))[0]
                                        save_mosaic_batch(crops_batch, batch_idx, dir_name, base_name, max_cols)
                                        batch_idx += 1
                                        crops_batch = []
                                        # 强制垃圾回收
                                        gc.collect()
                                except Exception as e:
                                    print(f"处理裁剪图像时出错: {e}")
                        
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
                video_base = os.path.splitext(os.path.basename(video_path))[0]
                training_image_path = os.path.join(training_folder, f"{video_base}_{frame_count}.jpg")
                training_annotation_path = os.path.splitext(training_image_path)[0] + ".txt"
                cv2.imwrite(training_image_path, frame)
                with open(training_annotation_path, 'w') as f:
                    for line in frame_annotations:
                        f.write(line + "\n")
            
            frame_count += 1
            
            # 每100帧清理一次内存
            if frame_count % 100 == 0:
                # 手动触发垃圾回收
                gc.collect()
                
                # 为减轻内存压力，暂停一小段时间
                if frame_count % 500 == 0:
                    time.sleep(0.1)
                
            # 释放当前帧
            del frame
    
    except Exception as e:
        print(f"处理视频时发生错误: {e}")
    
    finally:
        # 确保资源释放
        cap.release()
        pbar.close()
        if show_window:
            cv2.destroyAllWindows()
    
    # 处理剩余的裁剪图像
    if save_crops and crops_batch:
        dir_name = os.path.dirname(video_path)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        save_mosaic_batch(crops_batch, batch_idx, dir_name, base_name, max_cols)
    
    # 保存检测结果
    txt_save_path = video_path + ".txt"
    with open(txt_save_path, 'w') as f:
        f.write("检测到目标的时间位置（秒）:\n")
        for t in detections:
            f.write(f"{t:.2f}\n")
    print(f"已保存检测时间戳至: {txt_save_path}")
    
    # 创建总拼接图
    if save_crops and batch_idx > 1:
        try:
            dir_name = os.path.dirname(video_path)
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            # 合并所有批次图片
            mosaic_images = []
            for i in range(1, batch_idx + 1):
                mosaic_path = os.path.join(dir_name, f"{base_name}_mosaic_{i}.jpg")
                if os.path.exists(mosaic_path):
                    img = cv2.imread(mosaic_path)
                    if img is not None:
                        mosaic_images.append(img)
            
            # 如果有多个批次图片，创建最终拼接图
            if len(mosaic_images) > 0:
                # 计算总拼接图的高度
                final_height = sum(img.shape[0] for img in mosaic_images)
                final_width = max(img.shape[1] for img in mosaic_images)
                
                # 创建大小合适的画布
                final_mosaic = np.zeros((final_height, final_width, 3), dtype=np.uint8)
                
                # 填充图像
                y_offset = 0
                for img in mosaic_images:
                    h, w = img.shape[:2]
                    final_mosaic[y_offset:y_offset+h, 0:w] = img
                    y_offset += h
                
                # 保存最终拼接图
                final_mosaic_path = os.path.join(dir_name, f"{base_name}_mosaic.jpg")
                cv2.imwrite(final_mosaic_path, final_mosaic)
                print(f"已保存最终拼接图片至: {final_mosaic_path}")
                
                # 删除临时批次图片
                for i in range(1, batch_idx + 1):
                    temp_path = os.path.join(dir_name, f"{base_name}_mosaic_{i}.jpg")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        
        except Exception as e:
            print(f"创建最终拼接图时出错: {e}")
    
    # 清理内存
    gc.collect()
    return detections

def should_process(file_path):
    dir_name = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    mosaic_path = os.path.join(dir_name, base_name + "_mosaic.jpg")
    return not os.path.exists(mosaic_path)

def process_directory_videos(dir_path, target_item, all_objects_switch=False):
    """处理目录中的所有视频文件"""
    video_files = []
    
    try:
        for file in os.listdir(dir_path):
            if is_video_file(file):
                file_path = os.path.join(dir_path, file)
                if should_process(file_path):
                    # 检查视频时长
                    cap = cv2.VideoCapture(file_path)
                    if not cap.isOpened():
                        print(f"无法打开视频: {file_path}")
                        continue
                        
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    cap.release()
                    duration = frame_count / fps if fps > 0 else float('inf')
                    
                    if duration <= 3600:  # 小于等于1小时的视频
                        video_files.append(file_path)
                    else:
                        print(f"视频时长 {duration:.2f}秒超过一小时，跳过处理: {file_path}")
                else:
                    print(f"已存在拼接图片，跳过处理: {file_path}")
    except (PermissionError, FileNotFoundError) as e:
        print(f"警告: 无法访问目录 '{dir_path}': {e}")
        return
    
    # 处理视频文件
    for video_file in video_files:
        print(f"开始处理视频文件: {video_file}")
        detect_objects_in_video(video_file, target_item,
                                show_window=False,
                                save_crops=True,
                                save_training_data=False,
                                all_objects=all_objects_switch)
        # 视频处理完成后强制垃圾回收
        gc.collect()
        # 短暂休眠，让系统有时间释放资源
        time.sleep(1)

if __name__ == "__main__":
    video_path = r"D:\z"  # 可设置为视频文件或目录
    # 如要检测所有模型内对象，则将 target_item 设置为任意值并启用全量检测开关
    target_item = "face"  # 当 all_objects 为 True 时，该值不再限制检测
    all_objects_switch = False  # 设置为 True 表示显示所有检测对象
    
    # 新增功能：按叶子节点视频数量排序处理
    use_leaf_node_processing = True  # 设置为 True 启用叶子节点处理模式
    
    if use_leaf_node_processing and os.path.isdir(video_path):
        print(f"启用叶子节点处理模式，正在扫描目录: {video_path}")
        print("正在查找包含视频文件的叶子节点目录...")
        
        # 查找所有叶子目录
        leaf_dirs = find_leaf_directories_with_videos(video_path, EXCLUDE_PATHS)
        
        if not leaf_dirs:
            print(f"未找到包含视频文件的叶子节点目录")
        else:
            print(f"\n找到 {len(leaf_dirs)} 个包含视频文件的叶子节点目录:")
            for i, (dir_path, video_count) in enumerate(leaf_dirs, 1):
                relative_path = os.path.relpath(dir_path, video_path)
                print(f"{i:3d}. {relative_path} ({video_count} 个视频文件)")
            
            print(f"\n开始按视频数量从多到少的顺序处理叶子节点目录...")
            
            # 按顺序处理每个叶子目录
            for i, (dir_path, video_count) in enumerate(leaf_dirs, 1):
                relative_path = os.path.relpath(dir_path, video_path)
                print(f"\n=== 处理第 {i}/{len(leaf_dirs)} 个目录: {relative_path} ({video_count} 个视频) ===")
                process_directory_videos(dir_path, target_item, all_objects_switch)
                
                # 每处理完一个目录后强制垃圾回收
                gc.collect()
                # 让系统有时间释放资源
                time.sleep(2)
    
    # 原有的处理逻辑（当 use_leaf_node_processing 为 False 时使用）
    elif os.path.isdir(video_path):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        for root, dirs, files in os.walk(video_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in video_extensions:
                    file_path = os.path.join(root, file)
                    if should_process(file_path):
                        # 获取视频时长
                        cap = cv2.VideoCapture(file_path)
                        if not cap.isOpened():
                            print(f"无法打开视频: {file_path}")
                            continue
                            
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        cap.release()
                        duration = frame_count / fps if fps > 0 else float('inf')
                        if duration > 3600:
                            print(f"视频时长 {duration:.2f}秒超过一小时，跳过处理: {file_path}")
                            continue
                        
                        print(f"开始处理视频文件: {file_path}")
                        detect_objects_in_video(file_path, target_item,
                                                show_window=False,
                                                save_crops=True,
                                                save_training_data=True,
                                                all_objects=all_objects_switch)
                        
                        # 强制垃圾回收
                        gc.collect()
                        time.sleep(1)
                    else:
                        print(f"已存在拼接图片，跳过处理: {file_path}")
    else:
        # 处理单个视频文件前检查视频时长
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
        else:
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            duration = frame_count / fps if fps > 0 else float('inf')
            if duration > 3600:
                print(f"视频时长 {duration:.2f}秒超过一小时，跳过处理: {video_path}")
            else:
                if should_process(video_path):
                    detect_objects_in_video(video_path, target_item,
                                           show_window=False,
                                           save_crops=True,
                                           save_training_data=True,
                                           all_objects=all_objects_switch)
                else:
                    print(f"已存在拼接图片，跳过处理: {video_path}")