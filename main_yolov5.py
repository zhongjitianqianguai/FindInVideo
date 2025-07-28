import sys
import os
# 添加yolov5文件夹到Python路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

import torch
import cv2
import numpy as np
from tqdm import tqdm
import subprocess
import json
import re
from pathlib import Path

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

def load_yolov5_model(model_path, device='cpu'):
    """加载YOLOv5模型"""
    try:
        # 直接导入本地YOLOv5模块
        from yolov5.models.experimental import attempt_load
        from yolov5.utils.general import non_max_suppression
        from yolov5.utils.torch_utils import select_device
        
        # 尝试导入新版本的scale_boxes，如果失败则尝试scale_coords
        try:
            from yolov5.utils.general import scale_boxes as scale_coords_func
        except ImportError:
            try:
                from yolov5.utils.general import scale_coords as scale_coords_func
            except ImportError:
                print("警告: 无法导入坐标缩放函数，将使用手动缩放")
                scale_coords_func = None
        
        print(f"成功导入本地YOLOv5模块")
        
        device = select_device(device)
        
        # 尝试不同的参数组合来加载模型
        try:
            # 新版本YOLOv5使用weights参数
            model = attempt_load(weights=model_path, device=device)
        except TypeError:
            try:
                # 旧版本使用map_location参数
                model = attempt_load(model_path, map_location=device)
            except TypeError:
                try:
                    # 最简单的调用方式
                    model = attempt_load(model_path)
                    model = model.to(device)
                except Exception as e:
                    print(f"所有加载方式都失败: {e}")
                    raise e
        
        model.eval()
        
        return model, non_max_suppression, scale_coords_func, device
        
    except ImportError as e:
        print(f"导入YOLOv5模块失败: {e}")
        print("请确保yolov5文件夹在项目目录下，并包含必要的模块文件")
        raise e
    except Exception as e:
        print(f"加载YOLOv5模型失败: {e}")
        raise e

def detect_objects_in_video_yolov5(video_path, target_class,
                                   show_window=False, save_crops=False,
                                   save_training_data=False,
                                   all_objects=False,
                                   model_path='models/breast.pt'):
    """使用YOLOv5进行视频目标检测"""
    
    # 如果不开启全量检测，则保证 target_class 为列表
    if not all_objects and isinstance(target_class, str):
        target_class = [target_class]

    # 加载YOLOv5模型
    try:
        model, nms_func, scale_coords_func, device = load_yolov5_model(model_path)
        print(f"成功加载YOLOv5模型: {model_path}")
    except Exception as e:
        print(f"加载模型失败: {e}")
        return []

    # 获取类别名称
    if hasattr(model, 'names'):
        class_names = model.names
    elif hasattr(model, 'module') and hasattr(model.module, 'names'):
        class_names = model.module.names
    else:
        # 默认COCO类别
        class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
                      6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
                      # ... 可以添加更多COCO类别
                      }

    # 若需要生成训练数据，则构造保存目录及生成 classes.txt 文件
    if save_training_data:
        training_folder = os.path.join(os.path.dirname(video_path), "training_data")
        os.makedirs(training_folder, exist_ok=True)
        classes_txt = os.path.join(training_folder, "classes.txt")
        if not os.path.exists(classes_txt):
            with open(classes_txt, "w") as f:
                if all_objects:
                    # 写入模型中所有的类别，按 key 排序
                    for key in sorted(class_names.keys()):
                        f.write(class_names[key] + "\n")
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
    
    # 输入尺寸
    img_size = 640
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        # 更新进度条
        pbar.update(1)
        
        if save_training_data:
            frame_annotations = []
        
        current_time = frame_count / fps
        
        # 预处理图像
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (img_size, img_size))
        img_tensor = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        
        detected = False
        
        try:
            # YOLOv5推理
            if device is not None:
                img_tensor = img_tensor.to(device)
            
            with torch.no_grad():
                # 使用本地YOLOv5模型进行推理
                pred = model(img_tensor)[0]
                
                # 使用NMS函数
                pred = nms_func(pred, conf_thres=0.5, iou_thres=0.45)[0]
                
                if pred is not None and len(pred):
                    # 克隆张量以避免原地更新错误
                    pred = pred.clone()
                    
                    # 缩放坐标到原始图像尺寸
                    if scale_coords_func is not None:
                        # 新版本使用scale_boxes，参数顺序可能不同
                        try:
                            pred[:, :4] = scale_coords_func(img_tensor.shape[2:], pred[:, :4], frame.shape).round()
                        except:
                            # 如果参数顺序不对，尝试新的参数顺序
                            pred[:, :4] = scale_coords_func(pred[:, :4], frame.shape, img_tensor.shape[2:]).round()
                    else:
                        # 手动缩放坐标
                        h, w = frame.shape[:2]
                        pred[:, 0] *= w / img_size  # x1
                        pred[:, 1] *= h / img_size  # y1
                        pred[:, 2] *= w / img_size  # x2
                        pred[:, 3] *= h / img_size  # y2
                    
                    for det in pred:
                        x1, y1, x2, y2, conf, cls_id = det[:6]
                        cls_id = int(cls_id)
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        
                        # 获取类别名称
                        class_name = class_names.get(cls_id, f'class_{cls_id}')
                        
                        # 如果全量检测开启，或者检测到的类别在 target_class 内则处理
                        if all_objects or class_name in target_class:
                            if current_time - last_detected >= 0.1:
                                detections.append(current_time)
                                last_detected = current_time
                                detected = True
                            
                            if save_crops:
                                crop = frame[y1:y2, x1:x2]
                                if crop.size > 0:
                                    resized = cv2.resize(crop, crop_size)
                                    crops.append(resized)
                            
                            if show_window:
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(frame, f'{class_name} {conf:.2f}', (x1, y1-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            if save_training_data:
                                h, w, _ = frame.shape
                                cx = ((x1 + x2) / 2) / w
                                cy = ((y1 + y2) / 2) / h
                                bw = (x2 - x1) / w
                                bh = (y2 - y1) / h
                                # 如果全量检测，类别编号直接用 cls_id；否则使用 target_class 的索引
                                if all_objects:
                                    class_index = cls_id
                                else:
                                    class_index = target_class.index(class_name)
                                annotation_line = f"{class_index} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                                frame_annotations.append(annotation_line)
        
        except Exception as e:
            print(f"处理帧 {frame_count} 时出错: {e}")
            # 继续处理下一帧，而不是中断
            pass
        
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
    
    cap.release()
    pbar.close()
    if show_window:
        cv2.destroyAllWindows()
    
    txt_save_path = video_path + ".txt"
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
            cv2.imshow('All Detected Objects', final_mosaic)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            dir_name = os.path.dirname(video_path)
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            mosaic_path = os.path.join(dir_name, base_name + "_mosaic.jpg")
            cv2.imwrite(mosaic_path, final_mosaic)
            print(f"已保存拼接图片至: {mosaic_path}")
    
    return detections

def should_process(file_path):
    dir_name = os.path.dirname(file_path)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    mosaic_path = os.path.join(dir_name, base_name + "_mosaic.jpg")
    return not os.path.exists(mosaic_path)

def detect_objects_with_detectpy(video_path, target_class,
                                 show_window=False, save_crops=False,
                                 save_training_data=False,
                                 all_objects=False,
                                 model_path='models/breast.pt'):
    """使用YOLOv5的detect.py进行视频目标检测"""
    
    # 如果不开启全量检测，则保证 target_class 为列表
    if not all_objects and isinstance(target_class, str):
        target_class = [target_class]

    # 构建detect.py命令
    detect_script = os.path.join(os.path.dirname(__file__), 'yolov5', 'detect.py')
    output_dir = os.path.join(os.path.dirname(video_path), "yolov5_output")
    
    # 获取当前Python解释器路径（虚拟环境中的Python）
    python_exe = sys.executable
    
    cmd = [
        python_exe, detect_script,  # 使用当前Python解释器
        '--weights', model_path,
        '--source', video_path,
        '--project', output_dir,
        '--name', 'exp',
        '--exist-ok',
        '--conf-thres', '0.5',
        '--save-txt',  # 保存检测结果为txt文件
        '--save-conf'  # 保存置信度
    ]
    
    if save_crops:
        cmd.append('--save-crop')
    
    print(f"运行YOLOv5 detect.py: {' '.join(cmd)}")
    
    try:
        # 运行detect.py
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode != 0:
            print(f"detect.py运行失败: {result.stderr}")
            return []
        
        print("detect.py运行成功")
        
        # 解析检测结果
        detections = []
        results_dir = os.path.join(output_dir, 'exp')
        labels_dir = os.path.join(results_dir, 'labels')
        
        if os.path.exists(labels_dir):
            # 读取视频帧率用于计算时间戳
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            
            # 解析标签文件
            for label_file in os.listdir(labels_dir):
                if label_file.endswith('.txt'):
                    # 从文件名提取帧号
                    frame_match = re.search(r'frame_(\d+)', label_file)
                    if frame_match:
                        frame_num = int(frame_match.group(1))
                        timestamp = frame_num / fps if fps > 0 else 0
                        
                        # 读取检测结果
                        label_path = os.path.join(labels_dir, label_file)
                        with open(label_path, 'r') as f:
                            lines = f.readlines()
                            
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls_id = int(parts[0])
                                confidence = float(parts[5]) if len(parts) > 5 else 0.5
                                
                                # 这里需要根据模型的类别映射来判断
                                # 简化处理：假设检测到任何对象都记录时间戳
                                if timestamp not in detections:
                                    detections.append(timestamp)
        
        # 保存时间戳到txt文件
        txt_save_path = video_path + ".txt"
        with open(txt_save_path, 'w') as f:
            f.write("检测到目标的时间位置（秒）:\n")
            for t in sorted(detections):
                f.write(f"{t:.2f}\n")
        print(f"已保存检测时间戳至: {txt_save_path}")
        
        # 处理拼接图片
        if save_crops:
            crops_dir = os.path.join(results_dir, 'crops')
            if os.path.exists(crops_dir):
                create_mosaic_from_crops(crops_dir, video_path)
        
        return detections
        
    except Exception as e:
        print(f"使用detect.py检测失败: {e}")
        return []

def create_mosaic_from_crops(crops_dir, video_path):
    """从裁剪图片创建拼接图"""
    crops = []
    crop_size = (160, 160)
    max_cols = 8
    
    # 收集所有裁剪的图片
    for root, dirs, files in os.walk(crops_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is not None:
                    resized = cv2.resize(img, crop_size)
                    crops.append(resized)
    
    if crops:
        # 创建拼接图
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
        
        if rows:
            final_mosaic = np.vstack(rows)
            
            dir_name = os.path.dirname(video_path)
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            mosaic_path = os.path.join(dir_name, base_name + "_mosaic.jpg")
            cv2.imwrite(mosaic_path, final_mosaic)
            print(f"已保存拼接图片至: {mosaic_path}")

def detect_objects_with_frame_analysis(video_path, target_class,
                                       show_window=False, save_crops=False,
                                       save_training_data=False,
                                       all_objects=False,
                                       model_path='models/breast.pt'):
    """通过逐帧分析获取准确的时间戳"""
    
    # 首先使用detect.py进行检测
    detect_script = os.path.join(os.path.dirname(__file__), 'yolov5', 'detect.py')
    output_dir = os.path.join(os.path.dirname(video_path), "yolov5_output")
    
    # 获取当前Python解释器路径（虚拟环境中的Python）
    python_exe = sys.executable
    
    cmd = [
        python_exe, detect_script,  # 使用当前Python解释器
        '--weights', model_path,
        '--source', video_path,
        '--project', output_dir,
        '--name', 'exp',
        '--exist-ok',
        '--conf-thres', '0.5',
        '--save-txt',
        '--save-conf'
    ]
    
    if save_crops:
        cmd.append('--save-crop')
    
    print(f"运行YOLOv5 detect.py进行检测...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode != 0:
            print(f"detect.py运行失败: {result.stderr}")
            return []
        
        # 分析视频帧和检测结果
        detections = []
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        results_dir = os.path.join(output_dir, 'exp')
        labels_dir = os.path.join(results_dir, 'labels')
        
        print(f"分析视频帧以获取准确时间戳...")
        pbar = tqdm(total=total_frames, desc=f"分析视频: {os.path.basename(video_path)}")
        
        frame_count = 0
        last_detected = -5
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            pbar.update(1)
            current_time = frame_count / fps
            
            # 检查对应的标签文件是否存在
            label_filename = f"{os.path.splitext(os.path.basename(video_path))[0]}_{frame_count:06d}.txt"
            label_path = os.path.join(labels_dir, label_filename)
            
            if os.path.exists(label_path):
                # 读取检测结果
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                if lines and current_time - last_detected >= 0.1:
                    detections.append(current_time)
                    last_detected = current_time
            
            frame_count += 1
        
        cap.release()
        pbar.close()
        
        # 保存时间戳
        txt_save_path = video_path + ".txt"
        with open(txt_save_path, 'w') as f:
            f.write("检测到目标的时间位置（秒）:\n")
            for t in detections:
                f.write(f"{t:.2f}\n")
        print(f"已保存检测时间戳至: {txt_save_path}")
        
        # 处理拼接图片
        if save_crops:
            crops_dir = os.path.join(results_dir, 'crops')
            if os.path.exists(crops_dir):
                create_mosaic_from_crops(crops_dir, video_path)
        
        return detections
        
    except Exception as e:
        print(f"检测失败: {e}")
        return []

def process_directory_videos(dir_path, target_item, all_objects_switch=False, model_path='models/yolov5s.pt', use_detectpy=False):
    """处理目录中的所有视频文件"""
    video_files = []
    
    try:
        for file in os.listdir(dir_path):
            if is_video_file(file):
                file_path = os.path.join(dir_path, file)
                if should_process(file_path):
                    video_files.append(file_path)
                else:
                    print(f"已存在拼接图片，跳过处理: {file_path}")
    except (PermissionError, FileNotFoundError) as e:
        print(f"警告: 无法访问目录 '{dir_path}': {e}")
        return
    
    # 处理视频文件
    for video_file in video_files:
        print(f"开始处理视频文件: {video_file}")
        
        if use_detectpy:
            detect_objects_with_frame_analysis(video_file, target_item,
                                             show_window=False,
                                             save_crops=True,
                                             save_training_data=False,
                                             all_objects=all_objects_switch,
                                             model_path=model_path)
        else:
            detect_objects_in_video_yolov5(video_file, target_item,
                                          show_window=False,
                                          save_crops=True,
                                          save_training_data=False,
                                          all_objects=all_objects_switch,
                                          model_path=model_path)

if __name__ == "__main__":
    video_path = r"C:\Users\f1094\Desktop\DouyinLiveRecorder\download\邹邹大王"  # 可设置为视频文件或目录
    # 如要检测所有模型内对象，则将 target_item 设置为任意值并启用全量检测开关
    target_item = "breast"  # 当 all_objects 为 True 时，该值不再限制检测
    all_objects_switch = True  # 设置为 True 表示显示所有检测对象
    
    # YOLOv5模型路径 - 使用相对路径
    yolov5_model_path = "models/breast.pt"  # 您可以根据需要修改模型路径
    
    # 新增选项：是否使用detect.py进行推理
    use_detectpy = True  # 设置为 True 使用detect.py，False 使用原有方法
    
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
                process_directory_videos(dir_path, target_item, all_objects_switch, yolov5_model_path, use_detectpy)
    
    # 原有的处理逻辑（当 use_leaf_node_processing 为 False 时使用）
    elif os.path.isdir(video_path):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        for root, dirs, files in os.walk(video_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in video_extensions:
                    file_path = os.path.join(root, file)
                    if should_process(file_path):
                        print(f"开始处理视频文件: {file_path}")
                        detect_objects_in_video_yolov5(file_path, target_item,
                                                      show_window=False,
                                                      save_crops=True,
                                                      save_training_data=False,
                                                      all_objects=all_objects_switch,
                                                      model_path=yolov5_model_path)
                    else:
                        print(f"已存在拼接图片，跳过处理: {file_path}")
    else:
        # 处理单个视频文件
        if should_process(video_path):
            detect_objects_in_video_yolov5(video_path, target_item,
                                          show_window=False,
                                          save_crops=True,
                                          save_training_data=False,
                                          all_objects=all_objects_switch,
                                          model_path=yolov5_model_path)
        else:
            print(f"已存在拼接图片，跳过处理: {video_path}")