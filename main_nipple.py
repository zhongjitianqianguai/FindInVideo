from ultralytics import YOLO
import cv2
import numpy as np
import os
import hashlib
import json
import subprocess
from tqdm import tqdm
import gc  # 导入垃圾回收模块
import time
import logging
import traceback
import ntpath

import utils
from utils import (
    VIDEO_EXTENSIONS, ARTIFACT_SUFFIXES, DONE_SUFFIX, DIR_ARTIFACT_SKIP_SUFFIXES,
    _IGNORED_SUBDIRS, CHECKPOINT_SUFFIX, get_claim_heartbeat_interval_seconds,
    PauseRequested, ClaimLostError, DIRECTORY_INDEX,
    is_windows_style_path, windows_path_to_wsl, wsl_path_to_windows,
    normalize_posix_path_with_fs, windows_path_to_unc, unc_to_drive_letter,
    _safe_relpath, canonical_video_path,
    _get_env_path, get_shared_state_dir, ensure_shared_state_dir, _truthy_env,
    _atomic_create_file, _with_lockfile,
    safe_artifact_basename, _sanitize_basename, legacy_artifact_basename, _legacy_artifact_basename_v1,
    has_completed_artifact, build_pipeline_id, _source_snapshot,
    write_done_marker as _write_done_marker_state,
    _checkpoint_path, _load_checkpoint, _save_checkpoint, _clear_checkpoint,
    _install_pause_signal_handler, _get_pause_file_path, _pause_requested,
    record_resume_seek, is_video_file, is_tail_eof_within_tolerance,
)

_PROCESSING_ROOT_DIR = None
_YOLOED_MD5_CACHE = None
_YOLOED_MD5_CACHE_MTIME = None
_YOLOED_PATH_CACHE = None
_FILE_MD5_CACHE = {}
_FILE_SOURCE_SNAPSHOT_CACHE = {}
_ACTIVE_PIPELINE_ID = None
_ACTIVE_MODEL_PATH = None
_LOGGER = logging.getLogger(__name__)
MODEL_CONFIDENCE = 0.5
DEFAULT_IMAGE_SIZE = 1280
# 需要排除的路径变量
_EXCLUDE_PATHS_RAW = [
    os.path.abspath(r"D:\$RECYCLE.BIN"),
    os.path.abspath(r"D:\System Volume Information"),
    # 在这里添加其他需要排除的路径
    os.path.abspath(r"D:\z\gtll 10yue"),
]
# 自动生成 UNC 变体，确保排除路径在 UNC 格式扫描时也能匹配
EXCLUDE_PATHS = list(_EXCLUDE_PATHS_RAW)
for _ep in _EXCLUDE_PATHS_RAW:
    _unc = windows_path_to_unc(_ep)
    if _unc and _unc not in EXCLUDE_PATHS:
        EXCLUDE_PATHS.append(_unc)


def _init_processing_root(root_dir):
    """设置处理根目录，并将数据库迁移到 <root_dir>/md5_list/directory_index.db。"""
    global _PROCESSING_ROOT_DIR
    if not root_dir or not os.path.isdir(root_dir):
        return
    _PROCESSING_ROOT_DIR = root_dir
    db_dir = os.path.join(root_dir, "md5_list")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "directory_index.db")
    DIRECTORY_INDEX.reopen(db_path)
    print(f"数据库已打开: {db_path}")


def _resolve_pipeline_id(pipeline_id=None):
    return pipeline_id if pipeline_id is not None else _ACTIVE_PIPELINE_ID


def _get_image_size():
    try:
        return int(os.environ.get('FINDINVIDEO_IMGSZ', DEFAULT_IMAGE_SIZE))
    except (TypeError, ValueError):
        return DEFAULT_IMAGE_SIZE


def _activate_pipeline(model_path, target_class, all_objects=False):
    """根据真实权重摘要和推理配置启用独立完成态。"""
    global _ACTIVE_PIPELINE_ID, _ACTIVE_MODEL_PATH
    _ACTIVE_MODEL_PATH = model_path
    _ACTIVE_PIPELINE_ID = build_pipeline_id(
        engine='ultralytics-yolov11-video',
        model_path=model_path,
        target_class=target_class,
        all_objects=all_objects,
        confidence=MODEL_CONFIDENCE,
        image_size=_get_image_size(),
        extra={'entrypoint': 'main_nipple'},
    )
    return _ACTIVE_PIPELINE_ID


def _pipeline_checkpoint_path(video_path):
    pipeline = _resolve_pipeline_id()
    if pipeline is None:
        return _checkpoint_path(video_path)
    return _checkpoint_path(video_path, pipeline_id=pipeline)


def _load_pipeline_checkpoint(video_path):
    pipeline = _resolve_pipeline_id()
    if pipeline is None:
        return _load_checkpoint(video_path)
    return _load_checkpoint(video_path, pipeline_id=pipeline)


def _save_pipeline_checkpoint(video_path, **kwargs):
    pipeline = _resolve_pipeline_id()
    if pipeline is not None:
        kwargs['pipeline_id'] = pipeline
    return _save_checkpoint(video_path, **kwargs)


def _clear_pipeline_checkpoint(video_path):
    pipeline = _resolve_pipeline_id()
    if pipeline is None:
        return _clear_checkpoint(video_path)
    return _clear_checkpoint(video_path, pipeline_id=pipeline)


def _refresh_claim(file_md5):
    pipeline = _resolve_pipeline_id()
    if pipeline is None:
        return DIRECTORY_INDEX.refresh_claim(file_md5)
    return DIRECTORY_INDEX.refresh_claim(file_md5, pipeline_id=pipeline)


def _source_cache_key(video_path, file_md5):
    return (str(video_path), str(file_md5), _resolve_pipeline_id())


def _minimum_source_age_seconds():
    try:
        return max(
            0.0,
            float(os.environ.get('FINDINVIDEO_SOURCE_STABLE_SECONDS', '5')),
        )
    except (TypeError, ValueError):
        return 5.0


def _hash_stable_source(video_path):
    """在 MD5 前后核对源文件，拒绝刚写入或处理中变化的视频。"""
    try:
        before = _source_snapshot(video_path)
    except OSError:
        return None, None, 'source_unstable'
    if time.time() - before['mtime_ns'] / 1_000_000_000 < _minimum_source_age_seconds():
        return None, None, 'source_unstable'
    md5 = get_file_md5_cached(video_path)
    if not md5:
        return None, None, 'md5_unavailable'
    try:
        after = _source_snapshot(video_path)
    except OSError:
        return None, None, 'source_unstable'
    if before != after:
        return None, None, 'source_unstable'
    _FILE_SOURCE_SNAPSHOT_CACHE[_source_cache_key(video_path, md5)] = after
    return md5, after, 'ready'


def _source_snapshot_for(video_path, file_md5):
    return _FILE_SOURCE_SNAPSHOT_CACHE.get(_source_cache_key(video_path, file_md5))


def _try_claim_video(file_md5, video_path):
    pipeline = _resolve_pipeline_id()
    if pipeline is None:
        return DIRECTORY_INDEX.try_claim_video(file_md5, video_path)
    return DIRECTORY_INDEX.try_claim_video(
        file_md5,
        video_path,
        pipeline_id=pipeline,
        source_snapshot=_source_snapshot_for(video_path, file_md5),
    )


def has_existing_artifacts(video_path, pipeline_id=None):
    """检查当前视频是否已有可靠完成标记。"""
    return has_completed_artifact(
        video_path,
        pipeline_id=_resolve_pipeline_id(pipeline_id),
    )


def write_done_marker(
    video_path,
    pipeline_id=None,
    file_md5=None,
    source_snapshot=None,
):
    """写入绑定源文件指纹和处理流水线的完成标记。"""
    return _write_done_marker_state(
        video_path,
        pipeline_id=_resolve_pipeline_id(pipeline_id),
        file_md5=file_md5,
        source_snapshot=source_snapshot,
    )


def directory_has_artifact_outputs(dir_path):
    if _resolve_pipeline_id() is not None:
        return False
    try:
        for name in os.listdir(dir_path):
            lower = name.lower()
            if lower.endswith(DIR_ARTIFACT_SKIP_SUFFIXES):
                return True
    except (PermissionError, FileNotFoundError, OSError):
        return False
    return False


def is_leaf_directory(dir_path):
    """检查目录是否为叶子节点（不包含子目录，忽略工具生成的输出目录）"""
    try:
        for item in os.listdir(dir_path):
            if item in _IGNORED_SUBDIRS:
                continue
            item_path = os.path.join(dir_path, item)
            if os.path.isdir(item_path):
                return False
        return True
    except (PermissionError, FileNotFoundError):
        return False


def count_videos_in_directory(dir_path):
    """统计目录中的视频文件数量"""
    cached = DIRECTORY_INDEX.get_video_count(dir_path)
    if cached is not None:
        return cached
    count = 0
    try:
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path) and is_video_file(file_path):
                count += 1
    except (PermissionError, FileNotFoundError):
        pass
    return count


def find_leaf_directories_with_videos(root_path, exclusions=None, refresh_index=True):
    """使用目录索引查找包含视频文件的叶子节点目录。"""
    if exclusions is None:
        exclusions = EXCLUDE_PATHS
    try:
        if refresh_index:
            DIRECTORY_INDEX.refresh(root_path, exclusions)
        leaf_dirs = DIRECTORY_INDEX.get_leaf_directories(root_path)
        if _resolve_pipeline_id() is not None:
            leaf_dirs = [(path, count, False) for path, count, _done in leaf_dirs]
    except Exception as exc:
        print(f"索引查找叶子目录失败，退回遍历模式: {exc}")
        leaf_dirs = []
    if not leaf_dirs:
        # 兼容性回退
        try:
            for root, dirs, files in os.walk(root_path):
                is_excluded = any(root.startswith(ex_path) for ex_path in exclusions)
                if is_excluded:
                    continue
                dirs[:] = [
                    d
                    for d in dirs
                    if not any(
                        os.path.join(root, d).startswith(ex_path)
                        for ex_path in exclusions
                    )
                ]
                if is_leaf_directory(root):
                    video_count = count_videos_in_directory(root)
                    if video_count > 0:
                        leaf_dirs.append((root, video_count, False))
        except (PermissionError, FileNotFoundError) as e:
            print(f"警告: 无法访问目录 '{root_path}': {e}")
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


def detect_objects_in_video(
    video_path,
    target_class,
    model,
    claim_md5=None,
    show_window=False,
    save_crops=False,
    save_training_data=False,
    all_objects=False,
    save_mosaic=False,
    save_timestamps=False,
):
    # 如果不开启全量检测，则保证 target_class 为列表
    if not all_objects and isinstance(target_class, str):
        target_class = [target_class]

    pause_file = _get_pause_file_path()
    resume_enabled = _truthy_env("FINDINVIDEO_RESUME", default=True)
    claim_heartbeat_interval = get_claim_heartbeat_interval_seconds()
    if claim_md5 and not _refresh_claim(claim_md5):
        raise ClaimLostError(f"视频声明已失效，无法开始处理: {video_path}")
    last_claim_heartbeat = time.monotonic()
    imgsz = _get_image_size()

    video_dir = os.path.dirname(video_path)
    pipeline = _resolve_pipeline_id()
    artifact_base = (
        safe_artifact_basename(video_path)
        if pipeline is None
        else safe_artifact_basename(video_path, pipeline_id=pipeline)
    )
    txt_save_path = os.path.join(video_dir, artifact_base + ".txt")
    mosaic_path = os.path.join(video_dir, artifact_base + "_mosaic.jpg")
    video_save_path = os.path.join(video_dir, artifact_base + "_frames.mp4")

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

    ckpt = _load_pipeline_checkpoint(video_path) if resume_enabled else None
    start_frame = int(ckpt.get("next_frame", 0)) if ckpt else 0
    detections = list(ckpt.get("detections", [])) if ckpt else []
    last_detected = (
        float(ckpt.get("last_detected", detections[-1] if detections else -5.0))
        if ckpt
        else -5
    )

    # 视频处理初始化
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_safe = fps if fps and fps > 0 else 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if start_frame > 0:
        try:
            seek_ok = cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            reported_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        except Exception as exc:
            record_resume_seek(
                video_path, start_frame, False, error=exc,
                file_md5=claim_md5, pipeline_id=_resolve_pipeline_id(),
            )
            cap.release()
            raise RuntimeError(f"无法恢复视频断点到第 {start_frame} 帧") from exc
        if not seek_ok:
            record_resume_seek(
                video_path, start_frame, False, file_md5=claim_md5,
                pipeline_id=_resolve_pipeline_id(),
            )
            cap.release()
            raise RuntimeError(f"无法恢复视频断点到第 {start_frame} 帧")
        record_resume_seek(
            video_path, start_frame, True, reported_frame=reported_frame,
            file_md5=claim_md5, pipeline_id=_resolve_pipeline_id(),
        )

    # 文件名已在上一行完整输出；进度条使用短描述，避免长文件名挤掉进度信息
    pbar = tqdm(
        total=total_frames,
        initial=min(start_frame, total_frames),
        desc="处理视频",
        dynamic_ncols=True,
    )

    frame_count = start_frame

    # 截图存储配置（用于拼接大图）
    crop_size = (160, 160)  # 统一缩放到的小图尺寸
    max_cols = 8  # 拼接大图每行最多显示数量
    crops_batch = []  # 存储当前批次的目标区域
    batch_size = 200  # 每批处理的目标数量
    batch_idx = 1  # 批次计数器

    # ---- 检测帧视频写入器（懒初始化，首次检测到目标时创建）----
    video_writer = None
    frame_w, frame_h = 0, 0

    last_success_frame = start_frame - 1
    reached_eof = False
    try:
        while cap.isOpened():
            if claim_md5 and (
                time.monotonic() - last_claim_heartbeat
                >= claim_heartbeat_interval
            ):
                if not _refresh_claim(claim_md5):
                    raise ClaimLostError(
                        f"视频声明已失效，停止处理以避免重复写入: {video_path}"
                    )
                last_claim_heartbeat = time.monotonic()
            if _pause_requested(pause_file):
                _save_pipeline_checkpoint(
                    video_path,
                    next_frame=frame_count,
                    detections=detections,
                    last_detected=last_detected,
                    claim_md5=claim_md5,
                    last_success_frame=last_success_frame,
                    reason='pause_requested',
                )
                raise PauseRequested()
            success, frame = cap.read()
            if not success:
                if total_frames <= 0 or frame_count >= total_frames:
                    reached_eof = True
                break

            # 更新进度条
            pbar.update(1)

            if save_training_data:
                frame_annotations = []

            current_time = frame_count / fps_safe
            results = model.predict(
                frame,
                conf=MODEL_CONFIDENCE,
                imgsz=imgsz,
                verbose=False,
            )
            detected = False

            # 训练数据需要无框的原始帧，在画框前保存引用
            if save_training_data:
                training_frame = frame.copy()

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
                                crop = frame[
                                    y1:y2, x1:x2
                                ].copy()  # 使用.copy()避免引用原始帧
                                try:
                                    resized = cv2.resize(crop, crop_size)
                                    crops_batch.append(resized)

                                    # 批处理，避免占用过多内存
                                    if len(crops_batch) >= batch_size:
                                        if save_mosaic:
                                            dir_name = os.path.dirname(video_path)
                                            base_name = artifact_base
                                            save_mosaic_batch(
                                                crops_batch,
                                                batch_idx,
                                                dir_name,
                                                base_name,
                                                max_cols,
                                            )
                                            batch_idx += 1
                                        crops_batch = []
                                        # 强制垃圾回收
                                        gc.collect()
                                except Exception as e:
                                    print(f"处理裁剪图像时出错: {e}")

                        # 在帧上绘制检测框（供视频写入和窗口预览共用）
                        xyxy_draw = box.xyxy[0].cpu().numpy()
                        dx1, dy1, dx2, dy2 = map(int, xyxy_draw)
                        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                        label = model.names[cls_id]
                        cv2.putText(
                            frame,
                            label,
                            (dx1, max(dy1 - 6, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                        )

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
                            annotation_line = (
                                f"{class_index} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                            )
                            frame_annotations.append(annotation_line)

            # 若本帧有检测结果，写入输出视频
            if detected:
                if video_writer is None:
                    frame_h, frame_w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        video_save_path, fourcc, fps_safe, (frame_w, frame_h)
                    )
                video_writer.write(frame)

            if show_window and detected:
                cv2.imshow("Detection Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if save_training_data and frame_annotations:
                video_base = os.path.splitext(os.path.basename(video_path))[0]
                training_image_path = os.path.join(
                    training_folder, f"{video_base}_{frame_count}.jpg"
                )
                training_annotation_path = (
                    os.path.splitext(training_image_path)[0] + ".txt"
                )
                cv2.imwrite(training_image_path, training_frame)
                with open(training_annotation_path, "w") as f:
                    for line in frame_annotations:
                        f.write(line + "\n")

            last_success_frame = frame_count
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

    except KeyboardInterrupt:
        _save_pipeline_checkpoint(
            video_path,
            next_frame=frame_count,
            detections=detections,
            last_detected=last_detected,
            claim_md5=claim_md5,
            last_success_frame=last_success_frame,
            reason='keyboard_interrupt',
        )
        print(f"\nCtrl+C 已保存检查点，正在退出...")
        # 释放资源后重新抛出，让调用方知道是用户中断，不要标记为已完成
        if video_writer is not None:
            video_writer.release()
        cap.release()
        pbar.close()
        if show_window:
            cv2.destroyAllWindows()
        raise
    except PauseRequested:
        print('\n已在当前帧结束后保存检查点，准备退出，不会继续处理后续视频。')
        raise
    except Exception as e:
        print(f"处理视频时发生错误: {e}")
        raise

    finally:
        # 确保资源释放
        if video_writer is not None:
            video_writer.release()
        cap.release()
        pbar.close()
        if show_window:
            cv2.destroyAllWindows()

    if not reached_eof:
        if total_frames > 0 and frame_count < total_frames:
            if is_tail_eof_within_tolerance(frame_count, total_frames):
                missing_frames = total_frames - frame_count
                print(
                    f"提示: 视频尾部元数据比实际可读帧多 {missing_frames} 帧，"
                    f"已按正常结束处理: {frame_count}/{total_frames}"
                )
            else:
                _save_pipeline_checkpoint(
                    video_path,
                    next_frame=frame_count,
                    detections=detections,
                    last_detected=last_detected,
                    claim_md5=claim_md5,
                    last_success_frame=last_success_frame,
                    reason='unexpected_early_eof',
                )
                raise RuntimeError(
                    f"视频读取提前结束: 已处理 {frame_count}/{total_frames} 帧"
                )
        reached_eof = True

    if claim_md5 and not _refresh_claim(claim_md5):
        raise ClaimLostError(f"视频声明已失效，放弃写入最终产物: {video_path}")

    # 处理剩余的裁剪图像
    if save_mosaic and save_crops and crops_batch:
        dir_name = os.path.dirname(video_path)
        base_name = artifact_base
        save_mosaic_batch(crops_batch, batch_idx, dir_name, base_name, max_cols)

    # 输出视频结果提示
    if video_writer is not None:
        print(f"已保存检测帧视频至: {video_save_path}")
    else:
        print(f"未检测到目标，未生成帧视频: {os.path.basename(video_path)}")

    # 保存检测结果
    if save_timestamps:
        with open(txt_save_path, "w") as f:
            f.write("检测到目标的时间位置（秒）:\n")
            for t in detections:
                f.write(f"{t:.2f}\n")
        print(f"已保存检测时间戳至: {txt_save_path}")

    # 创建总拼接图
    if save_mosaic and save_crops and batch_idx > 1:
        try:
            dir_name = os.path.dirname(video_path)
            base_name = artifact_base
            mosaic_images = []
            for i in range(1, batch_idx + 1):
                mosaic_path = os.path.join(dir_name, f"{base_name}_mosaic_{i}.jpg")
                if os.path.exists(mosaic_path):
                    img = cv2.imread(mosaic_path)
                    if img is not None:
                        mosaic_images.append(img)
            if len(mosaic_images) > 0:
                final_height = sum(img.shape[0] for img in mosaic_images)
                final_width = max(img.shape[1] for img in mosaic_images)
                final_mosaic = np.zeros((final_height, final_width, 3), dtype=np.uint8)
                y_offset = 0
                for img in mosaic_images:
                    h, w = img.shape[:2]
                    final_mosaic[y_offset : y_offset + h, 0:w] = img
                    y_offset += h
                final_mosaic_path = os.path.join(dir_name, f"{base_name}_mosaic.jpg")
                cv2.imwrite(final_mosaic_path, final_mosaic)
                print(f"已保存最终拼接图片至: {final_mosaic_path}")
                for i in range(1, batch_idx + 1):
                    temp_path = os.path.join(dir_name, f"{base_name}_mosaic_{i}.jpg")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
        except Exception as e:
            print(f"创建最终拼接图时出错: {e}")

    # 清理内存
    gc.collect()
    return detections


def get_yoloed_md5_path():
    # 支持路径转换；默认使用处理目标目录下的 md5_list/yoloed.txt
    yoloed_path = _get_env_path("FINDINVIDEO_YOLOED_PATH")
    if not yoloed_path:
        base_dir = _PROCESSING_ROOT_DIR
        if not base_dir:
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__))
            except Exception:
                base_dir = os.getcwd()
        yoloed_path = os.path.join(base_dir, "md5_list", "yoloed.txt")
    if os.name == "posix":
        yoloed_path = windows_path_to_wsl(yoloed_path)
        if yoloed_path:
            yoloed_path = normalize_posix_path_with_fs(yoloed_path)
    return yoloed_path


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
        if os.name == "posix":
            win = wsl_path_to_windows(original)
            if win and win not in seen:
                seen.add(win)
                yield win

    short_dir = "/tmp/findinvideo_md5"

    def open_stream(path):
        hash_md5 = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def attempt_symlink(path):
        os.makedirs(short_dir, exist_ok=True)
        ext = os.path.splitext(path)[1] or ""
        hashed = hashlib.md5(path.encode("utf-8", "ignore")).hexdigest()
        link_path = os.path.join(short_dir, hashed + ext)
        try:
            if os.path.exists(link_path):
                return link_path
            os.symlink(path, link_path)
            return link_path
        except Exception:
            return None

    def windows_copy(path):
        win_src = path if is_windows_style_path(path) else wsl_path_to_windows(path)
        if not win_src:
            return None
        win_src = win_src.replace("/", "\\")
        ext = os.path.splitext(win_src)[1] or ".bin"
        hashed = hashlib.md5(win_src.encode("utf-8", "ignore")).hexdigest()
        base = (
            os.environ.get("FINDINVIDEO_WIN_TEMP", r"C:\Temp\findinvideo_md5")
            or r"C:\Temp\findinvideo_md5"
        )
        base = base.replace("/", "\\").rstrip("\\") or r"C:\Temp\findinvideo_md5"
        drive, tail = ntpath.splitdrive(base)
        base = (
            ntpath.join("C:\\", base.lstrip("\\"))
            if not drive
            else ntpath.normpath(drive + tail)
        )
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
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ensure_script],
                check=True,
                capture_output=True,
                text=True,
            )
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", copy_script],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"复制文件用于MD5失败: {exc}")
            return None
        dest_wsl = windows_path_to_wsl(dest_win)
        return dest_wsl

    last_error = None
    for candidate in candidate_paths(file_path):
        try:
            return open_stream(candidate)
        except (FileNotFoundError, NotADirectoryError) as e:
            last_error = e
        except OSError as e:
            last_error = e
            if os.name == "posix":
                link = attempt_symlink(candidate)
                if link:
                    try:
                        return open_stream(link)
                    except Exception as e2:
                        last_error = e2
                copied = windows_copy(candidate)
                if copied:
                    try:
                        return open_stream(copied)
                    except Exception as e3:
                        last_error = e3
        except Exception as e:
            last_error = e

    print(f"计算MD5错误: {file_path}, 错误: {last_error or '未找到可访问路径'}")
    return None


def get_file_md5_cached(file_path):
    """Cached wrapper around get_file_md5 to avoid recompute in one run."""
    if not file_path:
        return None
    try:
        st = os.stat(file_path)
        key = (
            str(file_path),
            int(getattr(st, "st_size", 0) or 0),
            float(getattr(st, "st_mtime", 0.0) or 0.0),
        )
    except Exception:
        key = (str(file_path), None, None)
    cached = _FILE_MD5_CACHE.get(key)
    if cached is not None:
        return cached
    digest = get_file_md5(file_path)
    # Best-effort bound memory
    if len(_FILE_MD5_CACHE) > 2048:
        _FILE_MD5_CACHE.clear()
    _FILE_MD5_CACHE[key] = digest
    return digest


def ensure_yoloed_storage():
    path = get_yoloed_md5_path()
    if not path:
        return None
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        try:
            os.makedirs(folder, exist_ok=True)
        except Exception as e:
            print(f"创建MD5存储目录失败: {e}")
            return None
    if not os.path.exists(path):
        try:
            open(path, "a", encoding="utf-8").close()
        except Exception as e:
            print(f"初始化MD5存储文件失败: {e}")
            return None
    return path


def load_yoloed_md5(reload=False):
    global _YOLOED_MD5_CACHE, _YOLOED_MD5_CACHE_MTIME, _YOLOED_PATH_CACHE
    path = ensure_yoloed_storage()
    if not path:
        _YOLOED_MD5_CACHE = set()
        _YOLOED_MD5_CACHE_MTIME = None
        _YOLOED_PATH_CACHE = set()
        return _YOLOED_MD5_CACHE

    try:
        mtime = os.path.getmtime(path)
    except Exception:
        mtime = None

    if (
        not reload
        and _YOLOED_MD5_CACHE is not None
        and _YOLOED_PATH_CACHE is not None
        and mtime is not None
        and _YOLOED_MD5_CACHE_MTIME == mtime
    ):
        return _YOLOED_MD5_CACHE

    md5_set = set()
    path_set = set()
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                entry = line.strip()
                if not entry:
                    continue
                parts = entry.split("|", 1)
                md5_part = parts[0].strip()
                if md5_part:
                    md5_set.add(md5_part)

                if len(parts) == 2:
                    raw_path = parts[1].strip()
                    if raw_path:
                        path_set.add(raw_path)
                        path_set.add(os.path.normpath(raw_path))
                        canon = canonical_video_path(raw_path)
                        if canon:
                            path_set.add(canon)
                        # 跨平台: 若为 WSL 路径 (/mnt/<drive>/...), 同时生成 Windows 盘符 + UNC 变体
                        if (
                            raw_path.startswith("/mnt/")
                            and len(raw_path) > 5
                            and raw_path[5].isalpha()
                        ):
                            _drv = raw_path[5]
                            _rest = raw_path[6:].lstrip("/").replace("/", "\\")
                            _win = (
                                f"{_drv.upper()}:\\{_rest}"
                                if _rest
                                else f"{_drv.upper()}:\\"
                            )
                            path_set.add(_win)
                            _win_norm = os.path.normpath(_win)
                            path_set.add(_win_norm)
                            _unc = canonical_video_path(_win)
                            if _unc:
                                path_set.add(_unc)
    except Exception as e:
        print(f"读取已识别MD5列表失败: {e}")
    _YOLOED_MD5_CACHE = md5_set
    _YOLOED_PATH_CACHE = path_set
    _YOLOED_MD5_CACHE_MTIME = mtime
    return _YOLOED_MD5_CACHE


def append_yoloed_md5(md5, file_path=None):
    global _YOLOED_MD5_CACHE, _YOLOED_MD5_CACHE_MTIME, _YOLOED_PATH_CACHE
    path = ensure_yoloed_storage()
    try:
        lock_path = path + ".lock"
        release = _with_lockfile(lock_path, timeout_seconds=30, stale_seconds=3600)
        entry = md5 if not file_path else f"{md5}|{file_path}"
        try:
            with open(path, "a", encoding="utf-8", errors="ignore") as f:
                f.write(entry + "\n")
        finally:
            release()

        # refresh cache mtime
        try:
            _YOLOED_MD5_CACHE_MTIME = os.path.getmtime(path)
        except Exception:
            _YOLOED_MD5_CACHE_MTIME = None
        if _YOLOED_MD5_CACHE is not None:
            _YOLOED_MD5_CACHE.add(md5)
        if file_path and _YOLOED_PATH_CACHE is not None:
            _fp = str(file_path)
            _YOLOED_PATH_CACHE.add(_fp)
            _YOLOED_PATH_CACHE.add(os.path.normpath(_fp))
            canon = canonical_video_path(file_path)
            if canon:
                _YOLOED_PATH_CACHE.add(canon)
            # 跨平台: WSL 路径同时添加 Windows 变体
            if _fp.startswith("/mnt/") and len(_fp) > 5 and _fp[5].isalpha():
                _drv = _fp[5]
                _rest = _fp[6:].lstrip("/").replace("/", "\\")
                _win = f"{_drv.upper()}:\\{_rest}" if _rest else f"{_drv.upper()}:\\"
                _YOLOED_PATH_CACHE.add(_win)
                _YOLOED_PATH_CACHE.add(os.path.normpath(_win))
                _unc = canonical_video_path(_win)
                if _unc:
                    _YOLOED_PATH_CACHE.add(_unc)
    except Exception as e:
        print(f"写入已识别MD5失败: {e}")


def _get_processing_decision(file_path, acquire_claim=True):
    try:
        checkpoint_exists = os.path.exists(_pipeline_checkpoint_path(file_path))
    except Exception:
        checkpoint_exists = False
    if has_existing_artifacts(file_path):
        return None, "has_artifacts"
    pipeline = _resolve_pipeline_id()
    legacy_pipeline = pipeline is None
    if legacy_pipeline and not checkpoint_exists and is_path_already_yoloed(file_path):
        return None, "path_already_yoloed"
    if legacy_pipeline:
        md5 = get_file_md5_cached(file_path)
        if not md5:
            return None, "md5_unavailable"
    else:
        md5, _snapshot, hash_reason = _hash_stable_source(file_path)
        if hash_reason != 'ready':
            return None, hash_reason
    # 数据库查询：基于MD5判断是否已处理（跨机器通用）
    if checkpoint_exists:
        processed_kwargs = {'include_directory_backfill': False}
    else:
        processed_kwargs = {}
    if pipeline is not None:
        processed_kwargs['pipeline_id'] = pipeline
    processed = DIRECTORY_INDEX.is_video_processed_by_md5(md5, **processed_kwargs)
    if processed:
        if checkpoint_exists:
            _clear_pipeline_checkpoint(file_path)
        return None, "already_processed_by_md5"
    yoloed_md5 = load_yoloed_md5(reload=False) if legacy_pipeline else set()
    if legacy_pipeline and not checkpoint_exists and md5 in yoloed_md5:
        return None, "md5_already_yoloed"
    if not acquire_claim:
        if DIRECTORY_INDEX.is_video_claimed(md5):
            print(f"视频已被其他实例声明，跳过: {file_path}")
            return None, "claimed_elsewhere"
        return md5, "ready"
    if not _try_claim_video(md5, file_path):
        if DIRECTORY_INDEX.is_video_claimed(md5):
            print(f"视频已被其他实例抢先声明，跳过: {file_path}")
            return None, "claimed_elsewhere"
        print(f"无法声明视频，跳过: {file_path}")
        return None, "claim_failed"
    return md5, "ready"


def should_process(file_path, acquire_claim=True):
    md5, _reason = _get_processing_decision(
        file_path, acquire_claim=acquire_claim
    )
    return md5


def is_path_already_yoloed(file_path):
    """Fast check using yoloed.txt stored paths (no hashing)."""
    load_yoloed_md5(reload=False)
    if not _YOLOED_PATH_CACHE:
        return False
    candidates = []
    p = str(file_path)
    candidates.append(p)
    candidates.append(os.path.normpath(p))
    canon = canonical_video_path(p)
    if canon:
        candidates.append(canon)
    return any(c in _YOLOED_PATH_CACHE for c in candidates)


def _mark_video_completed(
    video_path,
    detections,
    file_md5=None,
    source_snapshot=None,
):
    """统一写入视频完成态，避免 completed/processed/done 多处漂移。"""
    try:
        md5 = file_md5 or get_file_md5_cached(video_path)
        if md5:
            detection_count = len(detections) if detections else 0
            pipeline = _resolve_pipeline_id()
            completion_kwargs = {
                'file_md5': md5,
                'video_path': str(video_path),
                'detection_count': detection_count,
                'model_name': os.path.basename(_ACTIVE_MODEL_PATH or 'models/nipples.pt'),
            }
            final_snapshot = source_snapshot
            if pipeline is not None:
                final_snapshot = final_snapshot or _source_snapshot_for(video_path, md5)
                if final_snapshot is None:
                    print(f"缺少源文件快照，未写入完成状态: {video_path}")
                    return False
                completion_kwargs['pipeline_id'] = pipeline
                completion_kwargs['source_snapshot'] = final_snapshot
            completed = DIRECTORY_INDEX.complete_claimed_video(**completion_kwargs)
            if not completed:
                print(f"视频声明已失效，未写入完成状态: {video_path}")
                return False
            marker_ok = write_done_marker(
                video_path,
                pipeline_id=pipeline,
                file_md5=md5,
                source_snapshot=final_snapshot,
            ) if pipeline is not None else write_done_marker(video_path)
            marker_failed = marker_ok is False if pipeline is None else not marker_ok
            if marker_failed:
                rollback_kwargs = {'pipeline_id': pipeline} if pipeline is not None else {}
                rolled_back = DIRECTORY_INDEX.rollback_video_completion(
                    md5, **rollback_kwargs
                )
                if not rolled_back:
                    print(f"完成标记写入失败且数据库回滚失败: {video_path}")
                else:
                    print(f"完成标记写入失败，已撤销数据库完成态: {video_path}")
                return False
            if pipeline is None:
                append_yoloed_md5(md5, file_path=video_path)
            _clear_pipeline_checkpoint(video_path)
            print(
                f"已写入完成状态（检测数={detection_count}）: {os.path.basename(video_path)}"
            )
            return True
    except Exception as e:
        print(f"写入视频完成状态失败: {e}")
    return False


def _release_claim_safely(file_md5):
    """尽力释放处理声明，避免 Ctrl+C 后长时间占住 claim。"""
    if not file_md5:
        return
    try:
        pipeline = _resolve_pipeline_id()
        if pipeline is None:
            DIRECTORY_INDEX.release_claim(file_md5)
        else:
            DIRECTORY_INDEX.release_claim(file_md5, pipeline_id=pipeline)
    except Exception as e:
        print(f"释放视频声明失败: {e}")


def _mark_directory_done(dir_path, video_file_names, pipeline_id=None):
    """将目录标记为已全部处理，并将MD5缓存中已有的视频批量写入processed_videos表。"""
    pipeline = _resolve_pipeline_id(pipeline_id)
    try:
        video_paths = [os.path.join(dir_path, vf) for vf in video_file_names]
        if any(
            os.path.exists(
                _checkpoint_path(path)
                if pipeline is None
                else _checkpoint_path(path, pipeline_id=pipeline)
            )
            for path in video_paths
        ):
            print("目录中仍有未完成检查点，本轮不标记目录完成")
            return False
        if DIRECTORY_INDEX.has_active_claims_for_paths(video_paths):
            print("目录中仍有视频正在处理，本轮不标记目录完成")
            return False
        # 批量记录MD5已缓存的视频（这些视频在 should_process() 中已计算过MD5，无额外I/O）
        # 注意: _FILE_MD5_CACHE 的键是 (path_str, size, mtime) 元组，需要按此格式查找
        batch_count = 0
        for vf in video_file_names:
            vf_path = os.path.join(dir_path, vf)
            # 构造与 get_file_md5_cached 相同格式的缓存键
            try:
                st = os.stat(vf_path)
                cache_key = (
                    str(vf_path),
                    int(getattr(st, "st_size", 0) or 0),
                    float(getattr(st, "st_mtime", 0.0) or 0.0),
                )
            except (OSError, PermissionError):
                cache_key = (str(vf_path), None, None)
            cached_md5 = _FILE_MD5_CACHE.get(cache_key)
            if cached_md5:
                mark_kwargs = {
                    'file_md5': cached_md5,
                    'video_path': vf_path,
                    'detection_count': -1,
                    'model_name': None,
                }
                if pipeline is not None:
                    mark_kwargs['pipeline_id'] = pipeline
                    try:
                        mark_kwargs['source_snapshot'] = _source_snapshot(vf_path)
                    except OSError:
                        return False
                recorded = DIRECTORY_INDEX.mark_video_processed(**mark_kwargs)
                if not recorded:
                    print(f"视频仍有有效声明，停止标记目录完成: {vf_path}")
                    return False
                batch_count += 1
        if pipeline is not None:
            return True
        if not DIRECTORY_INDEX.mark_directory_processed_if_idle(dir_path, video_paths):
            print("目录完成状态在最终校验时发生变化，本轮不标记目录完成")
            return False
        if batch_count > 0:
            print(f"已将 {batch_count} 个视频的处理记录写入数据库")
        return True
    except Exception as e:
        print(f"标记目录完成状态失败: {e}")
        return False


def process_directory_videos(
    dir_path, target_item, model, all_objects_switch=False, skip_long_videos=True
):
    """处理目录中的所有视频文件"""
    if os.name == "posix" and is_windows_style_path(dir_path):
        converted = windows_path_to_wsl(dir_path)
        if converted:
            dir_path = normalize_posix_path_with_fs(converted)

    # 一次 listdir 获取所有文件，后续判断全部在内存中完成
    try:
        all_files = os.listdir(dir_path)
    except (PermissionError, FileNotFoundError) as e:
        print(f"警告: 无法访问目录 '{dir_path}': {e}")
        return

    video_file_names = [f for f in all_files if is_video_file(f)]

    if not video_file_names:
        return

    # 目录级预检查必须尊重 checkpoint，避免把暂停时的部分产物误判为完成。
    unprocessed_videos = []
    for vf in video_file_names:
        if has_existing_artifacts(os.path.join(dir_path, vf)):
            continue
        unprocessed_videos.append(vf)

    if not unprocessed_videos:
        print(f"目录中所有 {len(video_file_names)} 个视频已有衍生文件，跳过整个目录")
        _mark_directory_done(dir_path, video_file_names)
        return

    skipped_count = len(video_file_names) - len(unprocessed_videos)
    if skipped_count > 0:
        print(f"跳过 {skipped_count}/{len(video_file_names)} 个已有衍生文件的视频")

    # 扫描阶段只收集候选，不提前 claim；真正开工前再原子领取当前视频。
    video_files = []
    claim_blocked_videos = []
    unresolved_videos = []
    for file in unprocessed_videos:
        file_path = os.path.join(dir_path, file)
        md5, reason = _get_processing_decision(file_path, acquire_claim=False)
        if md5:
            # 检查视频时长
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                print(f"无法打开视频: {file_path}")
                unresolved_videos.append(file_path)
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            duration = frame_count / fps if fps > 0 else float("inf")

            video_files.append((file_path, duration, md5))
        elif reason in ("claimed_elsewhere", "claim_failed"):
            claim_blocked_videos.append(file_path)
        elif reason in ("md5_unavailable", "source_unstable"):
            unresolved_videos.append(file_path)

    if not video_files:
        if claim_blocked_videos:
            print(
                f"目录中有 {len(claim_blocked_videos)} 个视频当前被其他实例占用，"
                "本轮不标记目录完成"
            )
            return 0
        if unresolved_videos:
            print(
                f"目录中有 {len(unresolved_videos)} 个视频尚未解决，"
                "本轮不标记目录完成"
            )
            return 0
        if unprocessed_videos:
            print(f"目录中剩余 {len(unprocessed_videos)} 个视频经精确检查后均无需处理")
        _mark_directory_done(dir_path, video_file_names)
        return

    # 处理视频文件
    failed_videos = []
    completed_count = 0
    for video_file, duration, md5 in video_files:
        if _pause_requested():
            raise PauseRequested()
        if not _try_claim_video(md5, video_file):
            print(f"视频已被其他实例抢先声明，跳过: {video_file}")
            claim_blocked_videos.append(video_file)
            continue
        try:
            if duration == float("inf"):
                print(f"提示: 无法获取视频时长，仍尝试处理: {video_file}")
            print(f"开始处理视频文件: {video_file}")
            detections = detect_objects_in_video(
                video_file,
                target_item,
                model,
                claim_md5=md5,
                show_window=False,
                save_crops=True,
                save_training_data=True,
                all_objects=all_objects_switch,
                save_mosaic=save_mosaic_switch,
                save_timestamps=save_timestamps_switch,
            )
            if not _mark_video_completed(video_file, detections, file_md5=md5):
                failed_videos.append(video_file)
                continue
            completed_count += 1
            if _pause_requested():
                raise PauseRequested()
        except (PauseRequested, KeyboardInterrupt):
            raise
        except Exception:
            _LOGGER.error("Video failed: %s\n%s", video_file, traceback.format_exc())
            failed_videos.append(video_file)
            continue
        finally:
            _release_claim_safely(md5)
        # 视频处理完成后强制垃圾回收
        gc.collect()
        # 短暂休眠，让系统有时间释放资源
        time.sleep(1)

    if failed_videos:
        print(f"本轮有 {len(failed_videos)} 个视频处理失败，目录不会标记为已完成")
        return completed_count
    if claim_blocked_videos:
        print(
            f"本轮有 {len(claim_blocked_videos)} 个视频被其他实例占用，"
            "目录不会标记为已完成"
        )
        return completed_count
    if unresolved_videos:
        print(
            f"本轮有 {len(unresolved_videos)} 个视频尚未解决，"
            "目录不会标记为已完成"
        )
        return completed_count

    # 所有视频处理完成后，标记目录为已全部处理
    _mark_directory_done(dir_path, video_file_names)
    return completed_count


def _run_main():
    global all_objects_switch, save_mosaic_switch, save_timestamps_switch
    utils._STOP_REQUESTED = False
    _install_pause_signal_handler()
    video_path = r"Z:\待检测"  # 可设置为视频文件或目录
    # 如要检测所有模型内对象，则将 target_item 设置为任意值并启用全量检测开关
    target_item = "nipple"  # 当 all_objects 为 True 时，该值不再限制检测
    all_objects_switch = False  # 设置为 True 表示显示所有检测对象
    save_mosaic_switch = False  # 设置为 True 启用拼接图片保存
    save_timestamps_switch = False  # 设置为 True 启用检测时间戳txt保存
    model_path = os.environ.get('FINDINVIDEO_NIPPLE_MODEL', 'models/nipples.pt')
    _activate_pipeline(
        model_path,
        target_item,
        all_objects=all_objects_switch,
    )

    # 加载模型（全局一次，所有视频共用）
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    # 初始化处理根目录，数据库和yoloed.txt都放在 <video_path>/md5_list/ 下
    _root = video_path if os.path.isdir(video_path) else os.path.dirname(video_path)
    _init_processing_root(_root)

    # 新增功能：按叶子节点视频数量排序处理
    use_leaf_node_processing = True  # 设置为 True 启用叶子节点处理模式

    if use_leaf_node_processing and os.path.isdir(video_path):
        print(f"启用叶子节点处理模式，正在扫描目录: {video_path}")

        scan_round = 0
        while True:
            scan_round += 1
            if scan_round > 1:
                print(f"\n{'=' * 60}")
                print(f"第 {scan_round} 轮扫描：检查是否有新增视频...")
                print(f"{'=' * 60}")

            print("正在查找包含视频文件的叶子节点目录...")

            processed_this_round = 0
            root_video_count = count_videos_in_directory(video_path)
            if root_video_count > 0:
                print(f"\n=== 处理根目录: {video_path} ({root_video_count} 个视频) ===")
                root_processed_count = process_directory_videos(
                    video_path, target_item, model, all_objects_switch
                )
                processed_this_round += int(root_processed_count or 0)

            leaf_dirs = find_leaf_directories_with_videos(
                video_path, EXCLUDE_PATHS, refresh_index=True
            )

            if not leaf_dirs:
                print(f"未找到包含视频文件的叶子节点目录")
                if processed_this_round == 0:
                    break
            else:
                print(f"\n找到 {len(leaf_dirs)} 个包含视频文件的叶子节点目录:")
                for i, (dir_path, video_count, all_processed) in enumerate(
                    leaf_dirs, 1
                ):
                    relative_path = _safe_relpath(dir_path, video_path)
                    status = " [已处理]" if all_processed else ""
                    print(f"{i:3d}. {relative_path} ({video_count} 个视频文件){status}")

                print(f"\n开始按视频数量从多到少的顺序处理叶子节点目录...")

                # 先筛出本轮未处理队列，避免增量处理时继续显示原始排序位置/总目录数
                db_skipped = 0
                pending_leaf_dirs = []
                for dir_path, video_count, _ in leaf_dirs:
                    relative_path = _safe_relpath(dir_path, video_path)

                    # 数据库快速跳过：has_artifact=True 且目录 mtime 未变 → 无需任何I/O
                    dir_info = (
                        DIRECTORY_INDEX.get_directory_info(dir_path)
                        if _resolve_pipeline_id() is None
                        else None
                    )
                    if dir_info:
                        db_has_artifact, db_mtime = dir_info
                        if db_has_artifact and db_mtime is not None:
                            try:
                                current_mtime = os.stat(dir_path).st_mtime
                                if current_mtime == db_mtime:
                                    db_skipped += 1
                                    continue  # 目录未变且已全部处理，完全跳过
                            except (PermissionError, FileNotFoundError, OSError):
                                pass  # 无法获取mtime，退回到正常处理流程

                    pending_leaf_dirs.append((dir_path, video_count, relative_path))

                pending_count = len(pending_leaf_dirs)
                for queue_index, (dir_path, video_count, relative_path) in enumerate(
                    pending_leaf_dirs, 1
                ):
                    print(
                        f"\n=== [未处理队列 {queue_index}/{pending_count}] {relative_path} ({video_count} 个视频) ==="
                    )
                    processed_count = process_directory_videos(
                        dir_path, target_item, model, all_objects_switch
                    )
                    processed_this_round += int(processed_count or 0)

                    # 每处理完一个目录后强制垃圾回收
                    gc.collect()
                    # 让系统有时间释放资源
                    time.sleep(2)

                if db_skipped > 0:
                    print(
                        f"\n数据库快速跳过了 {db_skipped}/{len(leaf_dirs)} 个已确认处理完成的目录"
                    )

                # 被占用或无法读取时不要无限重扫同一目录。
                if processed_this_round == 0:
                    print(f"\n本轮没有取得处理进展，停止重复扫描")
                    break

            print(f"\n第 {scan_round} 轮处理完成，准备重新扫描检查新增...")

    # 原有的处理逻辑（当 use_leaf_node_processing 为 False 时使用）
    elif os.path.isdir(video_path):
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv"]
        for root, dirs, files in os.walk(video_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in video_extensions:
                    file_path = os.path.join(root, file)
                    md5 = should_process(file_path, acquire_claim=False)
                    if md5:
                        # 获取视频时长
                        cap = cv2.VideoCapture(file_path)
                        if not cap.isOpened():
                            print(f"无法打开视频: {file_path}")
                            _release_claim_safely(md5)
                            continue

                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        cap.release()

                        if not _try_claim_video(md5, file_path):
                            print(f"视频已被其他实例抢先声明，跳过: {file_path}")
                            continue
                        try:
                            print(f"开始处理视频文件: {file_path}")
                            detections = detect_objects_in_video(
                                file_path,
                                target_item,
                                model,
                                claim_md5=md5,
                                show_window=False,
                                save_crops=True,
                                save_training_data=True,
                                all_objects=all_objects_switch,
                                save_mosaic=save_mosaic_switch,
                                save_timestamps=save_timestamps_switch,
                            )
                            if not _mark_video_completed(
                                file_path, detections, file_md5=md5
                            ):
                                raise ClaimLostError(
                                    f"视频完成状态写入失败: {file_path}"
                                )
                            if _pause_requested():
                                raise PauseRequested()
                        except (PauseRequested, KeyboardInterrupt):
                            raise
                        except Exception:
                            _LOGGER.error(
                                "Video failed: %s\n%s",
                                file_path,
                                traceback.format_exc(),
                            )
                            continue
                        finally:
                            _release_claim_safely(md5)

                        # 强制垃圾回收
                        gc.collect()
                        time.sleep(1)
                    else:
                        print(f"视频无需处理或已被占用，跳过: {file_path}")
    else:
        # 处理单个视频文件
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {video_path}")
        else:
            cap.release()
            md5 = should_process(video_path, acquire_claim=False)
            if md5:
                if not _try_claim_video(md5, video_path):
                    print(f"视频已被其他实例抢先声明，跳过: {video_path}")
                else:
                    try:
                        print(f"开始处理视频文件: {video_path}")
                        detections = detect_objects_in_video(
                            video_path,
                            target_item,
                            model,
                            claim_md5=md5,
                            show_window=False,
                            save_crops=True,
                            save_training_data=True,
                            all_objects=all_objects_switch,
                            save_mosaic=save_mosaic_switch,
                            save_timestamps=save_timestamps_switch,
                        )
                        if not _mark_video_completed(
                            video_path, detections, file_md5=md5
                        ):
                            raise ClaimLostError(
                                f"视频完成状态写入失败: {video_path}"
                            )
                        if _pause_requested():
                            raise PauseRequested()
                    except (PauseRequested, KeyboardInterrupt):
                        raise
                    except Exception:
                        _LOGGER.error(
                            "Video failed: %s\n%s",
                            video_path,
                            traceback.format_exc(),
                        )
                    finally:
                        _release_claim_safely(md5)
            else:
                print(f"视频无需处理或已被占用，跳过: {video_path}")


if __name__ == "__main__":
    try:
        _run_main()
    except PauseRequested:
        print("\n已按请求暂停，本轮处理已停止。")
        raise SystemExit(0)
    except KeyboardInterrupt:
        print("\n已按请求停止，本轮处理已终止。")
        raise SystemExit(130)
