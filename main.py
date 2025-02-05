from ultralytics import YOLO
import cv2
import numpy as np


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
    with open(video_path+".txt", 'w') as f:
        f.write("检测到目标的时间位置（秒）:\n")
        for t in detections:
            f.write(f"{t:.2f}\n")

    # 拼接并显示截图
    if save_crops and crops:
        rows = []
        row = []
        for i, crop in enumerate(crops):
            row.append(crop)
            if (i + 1) % max_cols == 0:
                rows.append(np.hstack(row))
                row = []
        if row:  # 处理最后一行
            # 填充空白图像填充不足的列
            missing = max_cols - len(row)
            blank = np.zeros_like(row[0])
            row.extend([blank] * missing)
            rows.append(np.hstack(row))
        final_mosaic = np.vstack(rows)
        
        cv2.imshow('All Detected Faces', final_mosaic)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detections


# 使用示例
if __name__ == "__main__": 
    video_path = "C:\\Users\\f1094\\Desktop\\1.mp4"
    target_item = "face"

    # 参数说明：
    # show_window=True 显示实时检测窗口
    # save_crops=True 保存所有脸部截图
    detect_objects_in_video(video_path, target_item,
                            show_window=True,
                            save_crops=True)
