from ultralytics import YOLO
import cv2
import numpy as np
import os
from tqdm import tqdm  # 新增导入tqdm库用于进度条显示

def detect_objects_in_video(video_path, target_class,
                            show_window=False, save_crops=False,
                            save_training_data=False):
    # 加载模型
    model = YOLO('models/yolov11l-face.pt')

    # 若需要生成训练数据，则构造保存目录
    if save_training_data:
        training_folder = os.path.join(os.path.dirname(video_path), "training_data")
        os.makedirs(training_folder, exist_ok=True)
        # 在训练数据文件夹下生成classes.txt文件（如果不存在），写入目标类别
        classes_txt = os.path.join(training_folder, "classes.txt")
        if not os.path.exists(classes_txt):
            with open(classes_txt, "w") as f:
                f.write(target_class + "\n")

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

        # 如果需要保存训练数据，每帧初始化该帧的标注列表
        if save_training_data:
            frame_annotations = []

        current_time = frame_count / fps
        results = model.predict(frame, conf=0.5, verbose=False)
        detected = False

        # 遍历检测结果
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls)
                if model.names[cls_id] == target_class:
                    # 时间记录逻辑
                    if current_time - last_detected >= 0.1:
                        detections.append(current_time)
                        last_detected = current_time
                        # 注释掉秒数打印输出
                        # print(f"{video_path}: 检测到 {target_class} @ {current_time:.2f}秒")
                        detected = True

                    # 截图处理（用于拼接大图）
                    if save_crops:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:  # 确保有效区域
                            resized = cv2.resize(crop, crop_size)
                            crops.append(resized)

                    # 绘制检测框
                    if show_window:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 生成训练数据标注（YOLO格式：class center_x center_y width height，均归一化）
                    if save_training_data:
                        h, w, _ = frame.shape
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        cx = ((x1 + x2) / 2) / w
                        cy = ((y1 + y2) / 2) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h
                        annotation_line = f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                        frame_annotations.append(annotation_line)

        # 实时显示窗口
        if show_window and detected:
            cv2.imshow('Detection Preview', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # 如果需要保存训练数据且当前帧检测到目标，则保存当前帧及标注文件
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

    # 释放资源和关闭进度条
    cap.release()
    pbar.close()
    if show_window:
        cv2.destroyAllWindows()

    # 保存时间戳
    txt_save_path = video_path + ".txt"
    with open(txt_save_path, 'w') as f:
        f.write("检测到目标的时间位置（秒）:\n")
        for t in detections:
            f.write(f"{t:.2f}\n")
    print(f"已保存检测时间戳至: {txt_save_path}")

    # 拼接并显示或保存截图（仅用于可视化）
    if save_crops and crops:
        rows = []
        row = []
        for i, crop in enumerate(crops):
            row.append(crop)
            if (i + 1) % max_cols == 0:
                rows.append(np.hstack(row))
                row = []
        if row:  # 处理最后一行
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
            # 保存图片到视频文件所在的目录
            dir_name = os.path.dirname(video_path)
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            mosaic_path = os.path.join(dir_name, base_name + "_mosaic.jpg")
            cv2.imwrite(mosaic_path, final_mosaic)
            print(f"已保存拼接图片至: {mosaic_path}")

    return detections