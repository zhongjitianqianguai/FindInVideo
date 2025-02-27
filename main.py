from ultralytics import YOLO
import cv2
import numpy as np
import os
from tqdm import tqdm  # 新增导入tqdm库用于进度条显示

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
            video_base = os.path.splitext(os.path.basename(video_path))[0]
            training_image_path = os.path.join(training_folder, f"{video_base}_{frame_count}.jpg")
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
            cv2.imshow('All Detected Faces', final_mosaic)
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


if __name__ == "__main__":
    video_path = r"C:\Users\f1094\Desktop\20250220_092932.fix_p003.mp4"  # 可设置为视频文件或目录
    # 如要检测所有模型内对象，则将 target_item 设置为任意值并启用全量检测开关
    target_item = "face"  # 当 all_objects 为 True 时，该值不再限制检测
    all_objects_switch = False  # 设置为 True 表示显示所有检测对象
    
    # 如果 video_path 是目录，则递归遍历所有视频文件
    if os.path.isdir(video_path):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        for root, dirs, files in os.walk(video_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in video_extensions:
                    file_path = os.path.join(root, file)
                    if should_process(file_path):
                        # 获取视频时长
                        cap = cv2.VideoCapture(file_path)
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
                    else:
                        print(f"已存在拼接图片，跳过处理: {file_path}")
    else:
        # 处理单个视频文件前检查视频时长
        cap = cv2.VideoCapture(video_path)
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