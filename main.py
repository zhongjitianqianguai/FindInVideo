from ultralytics import YOLO
import cv2
import numpy as np
import os


def detect_objects_in_video(video_path, target_class,
                            show_window=False, save_crops=False):
    # 加载模型
    model = YOLO('models/yolov11l-face.pt')

    # 视频处理初始化
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    last_detected = -5
    detections = []

    # 截图存储配置
    crop_size = (160, 160)  # 统一缩放到的小图尺寸
    max_cols = 8  # 拼接大图每行最多显示数量
    crops = []  # 存储所有截取的目标区域

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

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
                        print(f"检测到 {target_class} @ {current_time:.2f}秒")
                        detected = True

                    # 截图处理
                    if save_crops:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        crop = frame[y1:y2, x1:x2]
                        if crop.size > 0:  # 确保有效区域
                            resized = cv2.resize(crop, crop_size)
                            crops.append(resized)

                    # 绘制检测框
                    if show_window:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 实时显示窗口
        if show_window and detected:
            cv2.imshow('Detection Preview', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

    # 释放资源
    cap.release()
    if show_window:
        cv2.destroyAllWindows()

    # 保存时间戳
    txt_save_path = video_path + ".txt"
    with open(txt_save_path, 'w') as f:
        f.write("检测到目标的时间位置（秒）:\n")
        for t in detections:
            f.write(f"{t:.2f}\n")
    print(f"已保存检测时间戳至: {txt_save_path}")

    # 拼接并显示或保存截图
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


if __name__ == "__main__":
    video_path = r"C:\Users\f1094\Desktop\python\images\新建文件夹"  # 可设置为视频文件或目录
    target_item = "face"

    # 如果video_path是目录，则递归遍历所有视频文件
    if os.path.isdir(video_path):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        for root, dirs, files in os.walk(video_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in video_extensions:
                    file_path = os.path.join(root, file)
                    print(f"开始处理视频文件: {file_path}")
                    detect_objects_in_video(file_path, target_item,
                                            show_window=False,
                                            save_crops=True)
    else:
        # 单个视频文件处理（实时显示窗口）
        detect_objects_in_video(video_path, target_item,
                                show_window=True,
                                save_crops=True)
