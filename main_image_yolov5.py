"""main_image_yolov5.py — 使用 YOLOv5 对图片进行对象检测

给定文件夹，递归扫描所有图片，将含有检测目标的图片生成带框副本
放在每个目录的 _detected 子目录下供查看。

用法:
    python main_image_yolov5.py

替代 main_yolov5.py 中拼接生成新视频的逻辑，直接对图片进行检测并保存带框结果。
"""

import sys
import os

# 添加 yolov5 文件夹到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

import cv2
import gc
import time
import torch
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


def _letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """对图片进行 letterbox 缩放：等比缩放 + 灰色填充，保持宽高比不变形。

    返回: (letterboxed_img, ratio, (left_pad, top_pad))
        ratio    — 缩放比例（用于坐标反映射）
        left_pad — 左侧实际填充像素数（整数）
        top_pad  — 顶部实际填充像素数（整数）
    """
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad_w, new_unpad_h = int(round(w * r)), int(round(h * r))
    dw = (new_shape[1] - new_unpad_w) / 2
    dh = (new_shape[0] - new_unpad_h) / 2

    if (w, h) != (new_unpad_w, new_unpad_h):
        img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    # 返回实际整数填充值（left, top），确保与 copyMakeBorder 完全一致
    return img, r, (left, top)


# ---------------------------------------------------------------------------
# YOLOv5 模型加载
# ---------------------------------------------------------------------------

def load_yolov5_model(model_path, device='cpu'):
    """加载 YOLOv5 模型（使用本地 yolov5 子模块）。

    返回: (model, nms_func, scale_coords_func, device, class_names)
    """
    try:
        from yolov5.models.experimental import attempt_load
        from yolov5.utils.general import non_max_suppression
        from yolov5.utils.torch_utils import select_device

        # 尝试导入新版本的 scale_boxes，如果失败则尝试 scale_coords
        try:
            from yolov5.utils.general import scale_boxes as scale_coords_func
        except ImportError:
            try:
                from yolov5.utils.general import scale_coords as scale_coords_func
            except ImportError:
                print('警告: 无法导入坐标缩放函数，将使用手动缩放')
                scale_coords_func = None

        print(f'成功导入本地 YOLOv5 模块')
        device = select_device(device)

        # 尝试不同的参数组合来加载模型（兼容新旧版本）
        try:
            model = attempt_load(weights=model_path, device=device)
        except TypeError:
            try:
                model = attempt_load(model_path, map_location=device)
            except TypeError:
                try:
                    model = attempt_load(model_path)
                    model = model.to(device)
                except Exception as e:
                    print(f'所有加载方式都失败: {e}')
                    raise

        model.eval()

        # 获取类别名称
        if hasattr(model, 'names'):
            class_names = model.names
        elif hasattr(model, 'module') and hasattr(model.module, 'names'):
            class_names = model.module.names
        else:
            # 默认 COCO 前几个类别
            class_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle',
                           4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck'}

        return model, non_max_suppression, scale_coords_func, device, class_names

    except ImportError as e:
        print(f'导入 YOLOv5 模块失败: {e}')
        print('请确保 yolov5 文件夹在项目目录下，并包含必要的模块文件')
        raise
    except Exception as e:
        print(f'加载 YOLOv5 模型失败: {e}')
        raise


# ---------------------------------------------------------------------------
# 核心检测函数
# ---------------------------------------------------------------------------

def detect_and_annotate(image_path, model, nms_func, scale_coords_func,
                        device, class_names, target_class=None,
                        all_objects=False, conf=0.5, iou=0.45,
                        img_size=640):
    """对单张图片进行 YOLOv5 对象检测。

    返回:
        (annotated_image, detection_count)
        若无检测结果，annotated_image 为 None。
    """
    frame = cv2.imread(image_path)
    if frame is None:
        print(f'无法读取图片: {image_path}')
        return None, 0

    # 预处理：BGR → RGB → letterbox（等比缩放+填充）→ tensor
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_letterboxed, ratio, (pad_w, pad_h) = _letterbox(img_rgb, (img_size, img_size))
    img_tensor = (
        torch.from_numpy(img_letterboxed)
        .float()
        .permute(2, 0, 1)
        .unsqueeze(0)
        / 255.0
    )

    detection_count = 0

    try:
        if device is not None:
            img_tensor = img_tensor.to(device)

        with torch.no_grad():
            pred = model(img_tensor)[0]
            pred = nms_func(pred, conf_thres=conf, iou_thres=iou)[0]

            if pred is not None and len(pred):
                pred = pred.clone()

                # 将检测坐标从 letterbox 空间映射回原始图像空间
                # 使用 yolov5 自带的 scale_boxes/scale_coords 函数，
                # 传入实际的 ratio_pad 以确保与 letterbox 预处理完全一致
                if scale_coords_func is not None:
                    ratio_pad_arg = ((ratio, ratio), (pad_w, pad_h))
                    try:
                        pred[:, :4] = scale_coords_func(
                            img_tensor.shape[2:], pred[:, :4], frame.shape,
                            ratio_pad=ratio_pad_arg
                        ).round()
                    except Exception:
                        # 兼容不同版本的参数顺序
                        pred[:, :4] = scale_coords_func(
                            pred[:, :4], frame.shape, img_tensor.shape[2:],
                            ratio_pad=ratio_pad_arg
                        ).round()
                else:
                    # 手动坐标映射（fallback）
                    pred[:, 0] = (pred[:, 0] - pad_w) / ratio  # x1
                    pred[:, 1] = (pred[:, 1] - pad_h) / ratio  # y1
                    pred[:, 2] = (pred[:, 2] - pad_w) / ratio  # x2
                    pred[:, 3] = (pred[:, 3] - pad_h) / ratio  # y2
                    h_orig, w_orig = frame.shape[:2]
                    pred[:, 0].clamp_(0, w_orig)
                    pred[:, 1].clamp_(0, h_orig)
                    pred[:, 2].clamp_(0, w_orig)
                    pred[:, 3].clamp_(0, h_orig)

                for det in pred:
                    x1, y1, x2, y2, det_conf, cls_id = det[:6]
                    cls_id = int(cls_id)
                    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                    class_name = class_names.get(cls_id, f'class_{cls_id}')

                    if all_objects or (target_class and class_name in target_class):
                        detection_count += 1
                        # 绘制检测框
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f'{class_name} {float(det_conf):.2f}'
                        cv2.putText(frame, label, (x1, max(y1 - 6, 0)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    except Exception as e:
        print(f'检测图片时出错: {e}')

    if detection_count > 0:
        return frame, detection_count
    return None, 0


# ---------------------------------------------------------------------------
# 目录级处理
# ---------------------------------------------------------------------------

def process_directory_images(dir_path, model, nms_func, scale_coords_func,
                             device, class_names, target_class=None,
                             all_objects=False, conf=0.5, iou=0.45):
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
            image_path, model, nms_func, scale_coords_func,
            device, class_names, target_class, all_objects, conf, iou
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
    image_root = r'E:\ShareMedia原档'            # 图片根目录
    target_item = 'breast'          # 检测目标（all_objects=True 时忽略）
    all_objects_switch = True        # True = 检测所有类别
    confidence = 0.5                 # 置信度阈值
    iou_threshold = 0.45            # NMS IoU 阈值
    model_path = 'models/breast.pt'  # YOLOv5 模型路径

    # ---- 加载模型（全局一次，所有目录共用）----
    print(f'加载 YOLOv5 模型: {model_path}')
    model, nms_func, scale_coords_func, device, class_names = load_yolov5_model(model_path)
    print(f'模型加载成功，类别数: {len(class_names)}')

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
                    dp, model, nms_func, scale_coords_func,
                    device, class_names, target_class,
                    all_objects_switch, confidence, iou_threshold
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
            image_root, model, nms_func, scale_coords_func,
            device, class_names, target_class, all_objects_switch,
            confidence, iou_threshold
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
