"""main_image.py — 使用 YOLOv11 (ultralytics) 对图片进行对象检测

给定文件夹，递归扫描所有图片，将含有检测目标的图片生成带框副本
放在每个目录的 _detected 子目录下供查看。

用法:
    python main_image.py

替代 main.py 中拼接生成新视频的逻辑，直接对图片进行检测并保存带框结果。
"""

from ultralytics import YOLO
import os
import sys
import cv2
import gc
import time
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 常见的图片文件扩展名
IMAGE_EXTENSIONS = {
    '.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif',
    '.gif', '.ico', '.jfif', '.pjpeg', '.pjp',
}

# 检测结果输出子目录名
DETECTED_DIR_NAME = '_detected'

# 排除路径（回收站、系统卷等不应扫描的目录）
EXCLUDE_PATHS = [
    os.path.abspath(r'D:\$RECYCLE.BIN'),
    os.path.abspath(r'D:\System Volume Information'),
]

# 排除的子目录名（在任何层级遇到都跳过）
_IGNORED_SUBDIRS = {
    DETECTED_DIR_NAME, '__pycache__', '.git', '$RECYCLE.BIN',
    'System Volume Information', 'training_data',
}


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def is_image_file(filename):
    """检查文件是否为图片文件"""
    ext = os.path.splitext(filename)[1].lower()
    return ext in IMAGE_EXTENSIONS


def _safe_relpath(path, start):
    """安全的 os.path.relpath，失败时返回原路径。"""
    try:
        return os.path.relpath(path, start)
    except (ValueError, TypeError):
        return path


# ---------------------------------------------------------------------------
# 核心检测函数
# ---------------------------------------------------------------------------

def detect_and_annotate(image_path, model, target_class=None,
                        all_objects=False, conf=0.5):
    """对单张图片进行 YOLOv11 对象检测。

    返回:
        (annotated_image, detection_count)
        若无检测结果，annotated_image 为 None。
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f'无法读取图片: {image_path}')
        return None, 0

    results = model.predict(img, conf=conf, verbose=False)

    detection_count = 0
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            class_name = model.names[cls_id]

            if all_objects or (target_class and class_name in target_class):
                detection_count += 1
                # 绘制检测框
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = map(int, xyxy)
                confidence = float(box.conf)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f'{class_name} {confidence:.2f}'
                cv2.putText(img, label, (x1, max(y1 - 6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    if detection_count > 0:
        return img, detection_count
    return None, 0


# ---------------------------------------------------------------------------
# 目录级处理
# ---------------------------------------------------------------------------

def process_directory_images(dir_path, model, target_class=None,
                             all_objects=False, conf=0.5):
    """处理一个目录中的所有图片文件。

    对每张图片执行检测:
    - 有检测结果 → 将带框图片保存到 _detected/ 子目录
    - 无检测结果 → 创建 .nodetect 标记文件，下次跳过

    Returns:
        (processed_count, detected_count)
    """
    try:
        all_files = os.listdir(dir_path)
    except (PermissionError, FileNotFoundError) as e:
        print(f'警告: 无法访问目录 \'{dir_path}\': {e}')
        return 0, 0

    image_files = [f for f in all_files if is_image_file(f)]
    if not image_files:
        return 0, 0

    # 输出目录
    detected_dir = os.path.join(dir_path, DETECTED_DIR_NAME)

    # 快速跳过：检查 _detected/ 目录中已有的文件
    existing_lower = set()
    if os.path.isdir(detected_dir):
        try:
            existing_lower = set(f.lower() for f in os.listdir(detected_dir))
        except (PermissionError, FileNotFoundError):
            pass

    # 过滤出需要处理的图片
    to_process = []
    for f in image_files:
        f_lower = f.lower()
        # 带框图片已存在
        if f_lower in existing_lower:
            continue
        # 已标记为无检测结果 (basename.nodetect)
        nodetect_name = os.path.splitext(f)[0].lower() + '.nodetect'
        if nodetect_name in existing_lower:
            continue
        to_process.append(f)

    if not to_process:
        print(f'目录中所有 {len(image_files)} 张图片已处理过，跳过整个目录')
        return 0, 0

    skipped = len(image_files) - len(to_process)
    if skipped > 0:
        print(f'跳过 {skipped}/{len(image_files)} 张已处理的图片')

    # 确保输出目录存在
    os.makedirs(detected_dir, exist_ok=True)

    detected_count = 0
    for f in tqdm(to_process, desc='检测图片', unit='张'):
        image_path = os.path.join(dir_path, f)
        annotated, count = detect_and_annotate(
            image_path, model, target_class, all_objects, conf
        )

        if annotated is not None:
            # 有检测结果 → 保存带框图片
            output_path = os.path.join(detected_dir, f)
            cv2.imwrite(output_path, annotated)
            detected_count += 1
        else:
            # 无检测结果 → 创建标记文件，下次跳过
            nodetect_path = os.path.join(
                detected_dir, os.path.splitext(f)[0] + '.nodetect'
            )
            try:
                with open(nodetect_path, 'w') as nf:
                    nf.write('')
            except Exception:
                pass

        # 释放内存
        del annotated
        if detected_count % 50 == 0:
            gc.collect()

    print(f'检测完成: {detected_count}/{len(to_process)} 张图片含有目标'
          f'，已保存到 {DETECTED_DIR_NAME}/')
    return len(to_process), detected_count


# ---------------------------------------------------------------------------
# 目录扫描
# ---------------------------------------------------------------------------

def find_directories_with_images(root_path, exclusions=None):
    """递归查找所有包含图片文件的目录。

    返回: [(dir_path, image_count), ...] 按图片数量降序排列
    """
    if exclusions is None:
        exclusions = EXCLUDE_PATHS

    dirs_with_images = []

    for root, dirs, files in os.walk(root_path):
        # 过滤特殊子目录
        dirs[:] = [
            d for d in dirs
            if d not in _IGNORED_SUBDIRS
            and not any(os.path.join(root, d).startswith(ex) for ex in exclusions)
        ]

        image_count = sum(1 for f in files if is_image_file(f))
        if image_count > 0:
            dirs_with_images.append((root, image_count))

    dirs_with_images.sort(key=lambda x: x[1], reverse=True)
    return dirs_with_images


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # ---- 配置 ----
    image_root = r'D:\z'           # 图片根目录
    target_item = 'face'           # 检测目标（all_objects=True 时忽略）
    all_objects_switch = False      # True = 检测所有类别
    confidence = 0.5                # 置信度阈值
    model_path = 'models/yolov11l-face.pt'  # 模型路径

    # ---- 加载模型（全局一次，所有目录共用）----
    print(f'加载模型: {model_path}')
    model = YOLO(model_path)

    target_class = [target_item] if isinstance(target_item, str) else target_item

    if os.path.isdir(image_root):
        print(f'正在扫描目录: {image_root}')
        dirs = find_directories_with_images(image_root)

        if not dirs:
            print('未找到包含图片的目录')
        else:
            total_images = sum(c for _, c in dirs)
            print(f'\n找到 {len(dirs)} 个包含图片的目录（共 {total_images} 张图片）:')
            for i, (dp, count) in enumerate(dirs, 1):
                rel = _safe_relpath(dp, image_root)
                print(f'{i:3d}. {rel} ({count} 张图片)')

            print(f'\n开始按图片数量从多到少的顺序处理...')
            total_processed = 0
            total_detected = 0
            for i, (dp, count) in enumerate(dirs, 1):
                rel = _safe_relpath(dp, image_root)
                print(f'\n=== [{i}/{len(dirs)}] {rel} ({count} 张图片) ===')
                processed, detected = process_directory_images(
                    dp, model, target_class, all_objects_switch, confidence
                )
                total_processed += processed
                total_detected += detected
                gc.collect()

            print(f'\n===== 全部完成 =====')
            print(f'共处理 {total_processed} 张图片，其中 {total_detected} 张含有检测目标')

    elif os.path.isfile(image_root):
        # 处理单张图片
        print(f'处理单张图片: {image_root}')
        annotated, count = detect_and_annotate(
            image_root, model, target_class, all_objects_switch, confidence
        )
        if annotated is not None:
            dir_name = os.path.dirname(image_root) or '.'
            detected_dir = os.path.join(dir_name, DETECTED_DIR_NAME)
            os.makedirs(detected_dir, exist_ok=True)
            output = os.path.join(detected_dir, os.path.basename(image_root))
            cv2.imwrite(output, annotated)
            print(f'检测到 {count} 个目标，已保存至: {output}')
        else:
            print('未检测到目标')
    else:
        print(f'路径不存在: {image_root}')
