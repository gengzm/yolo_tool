#!/usr/bin/env python
"""
Step 4: 推理
- 指定输入路径（图片目录或单张图片）
- 使用最新的训练模型进行推理
- 输出：
  * 推理可视化   — 推理结果叠加标注的图片
  * 推理JSON     — 推理结果 JSON 格式（方便后续叠加训练集）
  * 误差结果     — 预测与真值误差图表（平均误差、中值误差、std）

目录结构（基于 data.yaml 所在目录）:
    {dataset_dir}/推理结果/
    ├── 推理可视化/
    ├── 推理json/
    └── 误差结果/

用法:
    python s4_inference.py \\
        --data ./yolo_dataset/data.yaml \\
        --input ./test_images \\
        --task_type detect \\
        --conf 0.25 \\
        --iou 0.45

# 也可以直接用单张图片：
    python s4_inference.py \\
        --data ./yolo_dataset/data.yaml \\
        --input ./test_images/img001.jpg \\
        --task_type segment
"""
import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from config import CLASS_NAMES, CLASS_COLORS_BGR, POINT_COLORS_BGR, IMAGE_EXTENSIONS
from utils import ensure_dir, log_info, log_warn, log_error, load_info_yaml, get_project_config
import config


def find_latest_model(dataset_dir: str) -> str:
    """
    自动查找 run_out 下最新的 best.pt 模型
    """
    run_out_dir = Path(dataset_dir) / "run_out"
    if not run_out_dir.exists():
        log_error(f"run_out directory not found: {run_out_dir}")
        sys.exit(1)

    # 遍历所有时间戳子目录，找最新
    best_models = list(run_out_dir.rglob("**/weights/best.pt"))
    if not best_models:
        log_error(f"No best.pt found in {run_out_dir}")
        sys.exit(1)

    # 按修改时间排序，取最新的
    best_models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest = str(best_models[0])
    log_info(f"Latest model: {latest}")
    return latest


def get_output_dirs(dataset_dir: str) -> dict:
    """获取推理输出目录"""
    base = Path(dataset_dir) / "推理结果"
    vis_dir = ensure_dir(str(base / "推理可视化"))
    json_dir = ensure_dir(str(base / "推理json"))
    err_dir = ensure_dir(str(base / "误差结果"))
    return {
        "vis": str(vis_dir),
        "json": str(json_dir),
        "error": str(err_dir),
    }


def collect_input_images(input_path: str) -> list:
    """收集输入图片列表"""
    input_p = Path(input_path)
    if input_p.is_file():
        if input_p.suffix.lower() in IMAGE_EXTENSIONS:
            return [str(input_p)]
        else:
            log_error(f"Not an image: {input_path}")
            return []
    elif input_p.is_dir():
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(glob.glob(str(input_p / f"*{ext}")))
            images.extend(glob.glob(str(input_p / f"*{ext.upper()}")))
        return sorted(images)
    else:
        log_error(f"Input path not found: {input_path}")
        return []


def load_ground_truth(label_dir: str, image_stem: str) -> list:
    """
    加载指定图片的真值标注（YOLO 格式）
    返回: list of dicts: [{"class_id": int, "bbox": (cx,cy,w,h), "points": [[],...]}, ...]
    """
    label_path = Path(label_dir) / f"{image_stem}.txt"
    if not label_path.exists():
        return []

    gt_list = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            class_id = int(parts[0])
            values = [float(x) for x in parts[1:]]
            item = {"class_id": class_id}
            if len(values) >= 4:
                item["bbox"] = tuple(values[:4])
            if len(values) > 4:
                pts = np.array(values[4:]).reshape(-1, 2)
                item["points"] = pts.tolist()
            gt_list.append(item)
    return gt_list


def calc_bbox_errors(gt_boxes, pred_boxes, img_w: int, img_h: int) -> list:
    """
    计算 bbox 误差：中心点偏移、宽高误差 (像素)
    使用简单的贪心匹配
    """
    errors = []
    if not gt_boxes or not pred_boxes:
        return errors

    # 简单贪心匹配：每个 GT 匹配最近的预测
    for gt in gt_boxes:
        gt_cx, gt_cy, gt_w, gt_h = gt["bbox"]
        gt_cx_px = gt_cx * img_w
        gt_cy_px = gt_cy * img_h
        gt_w_px = gt_w * img_w
        gt_h_px = gt_h * img_h

        best_dist = float("inf")
        best_pred = None
        for pred in pred_boxes:
            pred_cx, pred_cy, pred_w, pred_h = pred["bbox"]
            # 中心点距离
            dist = np.sqrt((gt_cx_px - pred_cx * img_w)**2 +
                           (gt_cy_px - pred_cy * img_h)**2)
            # IOU 作为辅助
            # 计算 IOU
            x1_gt = gt_cx_px - gt_w_px / 2
            y1_gt = gt_cy_px - gt_h_px / 2
            x2_gt = gt_cx_px + gt_w_px / 2
            y2_gt = gt_cy_px + gt_h_px / 2

            x1_p = (pred_cx - pred_w / 2) * img_w
            y1_p = (pred_cy - pred_h / 2) * img_h
            x2_p = (pred_cx + pred_w / 2) * img_w
            y2_p = (pred_cy + pred_h / 2) * img_h

            # IoU
            ix1 = max(x1_gt, x1_p)
            iy1 = max(y1_gt, y1_p)
            ix2 = min(x2_gt, x2_p)
            iy2 = min(y2_gt, y2_p)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            area_gt = max(0, x2_gt - x1_gt) * max(0, y2_gt - y1_gt)
            area_p = max(0, x2_p - x1_p) * max(0, y2_p - y1_p)
            union = area_gt + area_p - inter
            iou = inter / union if union > 0 else 0

            # 综合考虑距离和 IOU
            if iou > 0:
                score = dist / iou
            else:
                score = dist * 10
            if score < best_dist:
                best_dist = score
                best_pred = pred

        if best_pred is not None:
            pred_cx, pred_cy, pred_w, pred_h = best_pred["bbox"]
            err = {
                "class_id": gt["class_id"],
                "class_name": CLASS_NAMES[gt["class_id"]] if gt["class_id"] < len(CLASS_NAMES) else f"cls{gt['class_id']}",
                "center_error_px": np.sqrt(
                    (gt_cx_px - pred_cx * img_w)**2 +
                    (gt_cy_px - pred_cy * img_h)**2
                ),
                "width_error_px": abs(gt_w_px - pred_w * img_w),
                "height_error_px": abs(gt_h_px - pred_h * img_h),
                "iou": _calc_iou(
                    (gt_cx_px - gt_w_px/2, gt_cy_px - gt_h_px/2,
                     gt_cx_px + gt_w_px/2, gt_cy_px + gt_h_px/2),
                    ((pred_cx - pred_w/2)*img_w, (pred_cy - pred_h/2)*img_h,
                     (pred_cx + pred_w/2)*img_w, (pred_cy + pred_h/2)*img_h),
                )
            }
            errors.append(err)

    return errors


def _calc_iou(box1: tuple, box2: tuple) -> float:
    """计算两个 bbox 的 IoU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def generate_error_report(all_errors: list, err_dir: str):
    """
    生成误差分析报告（文本 + 图表）
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not all_errors:
        log_warn("No errors to report (no ground truth available)")
        return

    # 汇总统计
    center_errors = [e["center_error_px"] for e in all_errors]
    width_errors = [e["width_error_px"] for e in all_errors]
    height_errors = [e["height_error_px"] for e in all_errors]
    ious = [e["iou"] for e in all_errors]

    stats = {
        "total_comparisons": len(all_errors),
        "center_error": {
            "mean": float(np.mean(center_errors)),
            "median": float(np.median(center_errors)),
            "std": float(np.std(center_errors)),
            "min": float(np.min(center_errors)),
            "max": float(np.max(center_errors)),
        },
        "width_error": {
            "mean": float(np.mean(width_errors)),
            "median": float(np.median(width_errors)),
            "std": float(np.std(width_errors)),
        },
        "height_error": {
            "mean": float(np.mean(height_errors)),
            "median": float(np.median(height_errors)),
            "std": float(np.std(height_errors)),
        },
        "iou": {
            "mean": float(np.mean(ious)),
            "median": float(np.median(ious)),
            "std": float(np.std(ious)),
        },
    }

    # 保存统计 JSON
    stats_path = os.path.join(err_dir, "error_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log_info(f"Error stats saved: {stats_path}")

    # 绘制误差分布图
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle("Prediction vs Ground Truth Error Analysis", fontsize=14)

    # 中心点误差直方图
    ax = axes[0, 0]
    ax.hist(center_errors, bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(center_errors), color="red", linestyle="--", label=f"Mean: {stats['center_error']['mean']:.2f}px")
    ax.axvline(np.median(center_errors), color="green", linestyle="--", label=f"Median: {stats['center_error']['median']:.2f}px")
    ax.set_xlabel("Center Error (pixels)")
    ax.set_ylabel("Count")
    ax.set_title(f"Center Point Error\nMean={stats['center_error']['mean']:.2f}px, Median={stats['center_error']['median']:.2f}px, Std={stats['center_error']['std']:.2f}px")
    ax.legend()

    # 宽度误差
    ax = axes[0, 1]
    ax.hist(width_errors, bins=30, color="coral", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(width_errors), color="red", linestyle="--", label=f"Mean: {stats['width_error']['mean']:.2f}px")
    ax.axvline(np.median(width_errors), color="green", linestyle="--", label=f"Median: {stats['width_error']['median']:.2f}px")
    ax.set_xlabel("Width Error (pixels)")
    ax.set_ylabel("Count")
    ax.set_title(f"Width Error\nMean={stats['width_error']['mean']:.2f}px, Median={stats['width_error']['median']:.2f}px, Std={stats['width_error']['std']:.2f}px")
    ax.legend()

    # 高度误差
    ax = axes[1, 0]
    ax.hist(height_errors, bins=30, color="mediumseagreen", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(height_errors), color="red", linestyle="--", label=f"Mean: {stats['height_error']['mean']:.2f}px")
    ax.axvline(np.median(height_errors), color="green", linestyle="--", label=f"Median: {stats['height_error']['median']:.2f}px")
    ax.set_xlabel("Height Error (pixels)")
    ax.set_ylabel("Count")
    ax.set_title(f"Height Error\nMean={stats['height_error']['mean']:.2f}px, Median={stats['height_error']['median']:.2f}px, Std={stats['height_error']['std']:.2f}px")
    ax.legend()

    # IoU 分布
    ax = axes[1, 1]
    ax.hist(ious, bins=30, color="mediumpurple", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(ious), color="red", linestyle="--", label=f"Mean: {stats['iou']['mean']:.3f}")
    ax.axvline(np.median(ious), color="green", linestyle="--", label=f"Median: {stats['iou']['median']:.3f}")
    ax.set_xlabel("IoU")
    ax.set_ylabel("Count")
    ax.set_title(f"IoU Distribution\nMean={stats['iou']['mean']:.3f}, Median={stats['iou']['median']:.3f}, Std={stats['iou']['std']:.3f}")
    ax.legend()

    plt.tight_layout()
    chart_path = os.path.join(err_dir, "error_analysis.png")
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log_info(f"Error chart saved: {chart_path}")

    # 打印统计摘要
    log_info("=" * 50)
    log_info("Error Analysis Summary:")
    log_info(f"  Center Error - Mean: {stats['center_error']['mean']:.2f}px, "
             f"Median: {stats['center_error']['median']:.2f}px, STD: {stats['center_error']['std']:.2f}px")
    log_info(f"  Width Error  - Mean: {stats['width_error']['mean']:.2f}px, "
             f"Median: {stats['width_error']['median']:.2f}px, STD: {stats['width_error']['std']:.2f}px")
    log_info(f"  Height Error - Mean: {stats['height_error']['mean']:.2f}px, "
             f"Median: {stats['height_error']['median']:.2f}px, STD: {stats['height_error']['std']:.2f}px")
    log_info(f"  IoU          - Mean: {stats['iou']['mean']:.3f}, "
             f"Median: {stats['iou']['median']:.3f}, STD: {stats['iou']['std']:.3f}")
    log_info("=" * 50)


def draw_prediction(img: np.ndarray, result, task_type: str,
                    class_colors: list = None) -> np.ndarray:
    """在图片上绘制推理结果，覆盖 detect / segment / obb / pose 四种任务"""
    if class_colors is None:
        class_colors = CLASS_COLORS_BGR

    if result.boxes is None:
        return img

    for i, box in enumerate(result.boxes):
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        color = class_colors[cls_id % len(class_colors)]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # 类别标签
        label = f"{CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else cls_id}: {conf:.2f}"

        # ---- detect: 画 bbox 矩形 ----
        if task_type == "detect":
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

        # ---- segment: 画 mask 轮廓 + 填充 + 点 ----
        elif task_type == "segment" and result.masks is not None:
            mask_data = result.masks.data
            if i < len(mask_data):
                mask = mask_data[i].cpu().numpy()
                h, w = img.shape[:2]
                mask_resized = cv2.resize(mask, (w, h))
                mask_bin = (mask_resized > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                overlay = img.copy()
                cv2.drawContours(overlay, contours, -1, color, -1)
                img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
                cv2.drawContours(img, contours, -1, color, 2)
                # 画点（首个点 5px，其余 3px，连线 1px）
                for contour in contours:
                    if len(contour) >= 1:
                        for j, pt in enumerate(contour):
                            px, py = pt[0]
                            r = 5 if j == 0 else 3
                            pt_color = POINT_COLORS_BGR[j % len(POINT_COLORS_BGR)]
                            cv2.circle(img, (px, py), r, pt_color, -1)
                            if j > 0:
                                p_prev = contour[j - 1][0]
                                cv2.line(img, (p_prev[0], p_prev[1]), (px, py), color, 1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ---- obb: 画旋转框（4 个顶点） ----
        elif task_type == "obb" and result.obb is not None:
            obb_data = result.obb
            if i < len(obb_data.xyxyxyxy):
                obb_pts = obb_data.xyxyxyxy[i].cpu().numpy().astype(np.int32)
                cv2.polylines(img, [obb_pts.reshape((-1, 1, 2))], isClosed=True,
                              color=color, thickness=2)
                # 画旋转框顶点
                for j, pt in enumerate(obb_pts):
                    px, py = int(pt[0]), int(pt[1])
                    r = 5 if j == 0 else 3
                    pt_color = POINT_COLORS_BGR[j % len(POINT_COLORS_BGR)]
                    cv2.circle(img, (px, py), r, pt_color, -1)
                    if j > 0:
                        prev = obb_pts[j - 1]
                        cv2.line(img, (int(prev[0]), int(prev[1])), (px, py), color, 1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ---- pose: 画 bbox + 关键点 + 骨架连线 ----
        elif task_type == "pose" and result.keypoints is not None:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            kpts = result.keypoints
            if i < len(kpts.data):
                kpt_data = kpts.data[i].cpu().numpy()  # shape (17, 3) [x, y, conf]
                # 骨架连接（COCO 17 点标准）
                skeleton = [
                    (0, 1), (0, 2), (1, 3), (2, 4),
                    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                    (5, 11), (6, 12), (11, 12),
                    (11, 13), (13, 15), (12, 14), (14, 16),
                ]
                for a, b in skeleton:
                    if (kpt_data[a][2] > 0.5 and kpt_data[b][2] > 0.5):
                        pa = (int(kpt_data[a][0]), int(kpt_data[a][1]))
                        pb = (int(kpt_data[b][0]), int(kpt_data[b][1]))
                        cv2.line(img, pa, pb, color, 1)
                # 画关键点
                for j, (kx, ky, kc) in enumerate(kpt_data):
                    if kc > 0.5:
                        px, py = int(kx), int(ky)
                        r = 5 if j == 0 else 3
                        pt_color = POINT_COLORS_BGR[j % len(POINT_COLORS_BGR)]
                        cv2.circle(img, (px, py), r, pt_color, -1)
                        cv2.circle(img, (px, py), r, (255, 255, 255), 1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ---- 兜底：当 task_type 已指定但结果中缺少对应属性时，画 bbox ----
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

    return img


def extract_predictions_json(results, img_path: str) -> dict:
    """提取推理结果为 JSON 格式（兼容 detect/segment/obb/pose）"""
    predictions = []
    for result in results:
        if result.boxes is None:
            continue

        boxes = result.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            xyxy = boxes.xyxy[i].tolist()
            pred = {
                "class_id": cls_id,
                "class_name": CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"cls{cls_id}",
                "confidence": round(conf, 4),
                "bbox": {
                    "x1": round(xyxy[0], 2), "y1": round(xyxy[1], 2),
                    "x2": round(xyxy[2], 2), "y2": round(xyxy[3], 2),
                },
            }
            # segment: 保存分割多边形
            if hasattr(result, "masks") and result.masks is not None:
                mask_data = result.masks.xyn
                if len(mask_data) > i:
                    pred["segmentation"] = mask_data[i].tolist()
            # obb: 保存旋转框 4 顶点
            if hasattr(result, "obb") and result.obb is not None:
                obb_data = result.obb
                if len(obb_data.xyxyxyxy) > i:
                    pred["obb"] = obb_data.xyxyxyxy[i].tolist()
            # pose: 保存关键点
            if hasattr(result, "keypoints") and result.keypoints is not None:
                kpts = result.keypoints
                if len(kpts.data) > i:
                    pred["keypoints"] = kpts.data[i].tolist()
            predictions.append(pred)

    return {
        "image": os.path.basename(img_path),
        "image_path": img_path,
        "predictions": predictions,
        "num_predictions": len(predictions),
    }


def run_inference(data_path: str, input_path: str, task_type: str,
                  conf: float = 0.25, iou: float = 0.45,
                  model_path: str = None, label_dir: str = None):
    """
    执行推理主流程
    """
    # 加载 data.yaml
    if not os.path.exists(data_path):
        log_error(f"data.yaml not found: {data_path}")
        sys.exit(1)

    with open(data_path, "r") as f:
        data_config = yaml.safe_load(f)

    # 从 data.yaml 加载类别名到全局配置
    names = data_config.get("names", [])
    if names:
        config.set_class_names(names)
        log_info(f"Loaded {len(names)} class(es) from {data_path}")

    dataset_dir = str(Path(data_path).parent.resolve())

    # 找最新模型：优先用 info.yaml 记录的 best 权重，否则自动扫描 run_out
    if model_path is None:
        info = load_info_yaml(dataset_dir)
        recorded = (info.get("weights") or {}).get("best")
        if recorded and Path(recorded).exists():
            model_path = recorded
            log_info(f"Using recorded model: {model_path}")
        else:
            model_path = find_latest_model(dataset_dir)
    if not os.path.exists(model_path):
        log_error(f"Model not found: {model_path}")
        sys.exit(1)

    # 输出目录
    out_dirs = get_output_dirs(dataset_dir)

    # 收集输入图片
    images = collect_input_images(input_path)
    if not images:
        log_error("No images found")
        sys.exit(1)
    log_info(f"Found {len(images)} image(s) to process")

    # 加载模型
    model = YOLO(model_path)
    log_info(f"Model loaded: {model_path}")

    # 确定 label 目录（用于误差计算）
    if label_dir is None:
        # 尝试从 data.yaml 获取 val label 目录
        val_dir = data_config.get("val", "")
        if val_dir:
            label_dir = str(Path(val_dir) / "labels")

    # 处理每张图
    all_errors = []
    all_json_results = []

    for img_path in images:
        log_info(f"Processing: {os.path.basename(img_path)}")

        # 推理
        results = model(img_path, conf=conf, iou=iou, verbose=False)

        # 读取图片用于可视化
        img = cv2.imread(img_path)
        if img is None:
            log_warn(f"Cannot read image: {img_path}")
            continue

        # 可视化
        vis_img = draw_prediction(img.copy(), results[0], task_type)
        vis_path = os.path.join(out_dirs["vis"], os.path.basename(img_path))
        cv2.imwrite(vis_path, vis_img)

        # 提取 JSON 结果
        json_result = extract_predictions_json(results, img_path)
        json_name = Path(img_path).stem + ".json"
        json_path = os.path.join(out_dirs["json"], json_name)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)
        all_json_results.append(json_result)

        # 误差分析（如果有真值）
        if label_dir and Path(label_dir).exists():
            gt_boxes = load_ground_truth(label_dir, Path(img_path).stem)
            if gt_boxes:
                pred_boxes = []
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        x1, y1, x2, y2 = box.xywh[0].tolist()
                        pred_boxes.append({
                            "class_id": cls_id,
                            "bbox": (x1 / img.shape[1], y1 / img.shape[0],
                                     (x2 - x1) / img.shape[1], (y2 - y1) / img.shape[0]),
                        })
                img_errors = calc_bbox_errors(gt_boxes, pred_boxes, img.shape[1], img.shape[0])
                all_errors.extend(img_errors)

    # 汇总 JSON
    summary_path = os.path.join(out_dirs["json"], "_all_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_json_results, f, indent=2, ensure_ascii=False)
    log_info(f"All results saved: {summary_path}")

    # 误差报告
    generate_error_report(all_errors, out_dirs["error"])

    log_info("=" * 60)
    log_info("Inference completed!")
    log_info(f"  Visualizations: {out_dirs['vis']}")
    log_info(f"  JSON results:   {out_dirs['json']}")
    log_info(f"  Error analysis: {out_dirs['error']}")
    log_info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="YOLO 推理工具")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="YOLO 数据集目录（用于定位项目 info.yaml，默认取 config）")
    parser.add_argument("--data", type=str, default=None,
                        help="data.yaml 路径（默认取项目 info.yaml / 数据集目录）")
    parser.add_argument("--input", type=str, default=None,
                        help="输入图片路径（目录或单张图片，默认取项目 info.yaml / 验证集）")
    parser.add_argument("--task_type", type=str, default=None,
                        choices=["detect", "segment", "obb", "pose"],
                        help="任务类型（默认取项目 info.yaml / config）")
    parser.add_argument("--model", type=str, default=None,
                        help="模型路径（默认使用最新的 best.pt）")
    parser.add_argument("--conf", type=float, default=None,
                        help="置信度阈值（默认取项目 info.yaml，否则 0.25）")
    parser.add_argument("--iou", type=float, default=None,
                        help="IoU 阈值（默认取项目 info.yaml，否则 0.45）")
    parser.add_argument("--label_dir", type=str, default=None,
                        help="真值 label 目录（用于误差分析）")
    args = parser.parse_args()

    # 项目配置解析: 命令行参数 > info.yaml 记录 > config 默认值
    cfg = get_project_config(args.dataset_dir)
    data_path = args.data or cfg["data_yaml"]
    if not data_path or not Path(data_path).exists():
        log_error("data.yaml not found. Please specify --data/--dataset_dir or run step1 first.")
        sys.exit(1)
    task_type = args.task_type or cfg["task_type"]
    input_path = args.input or cfg["infer_input"]
    if not input_path or not Path(input_path).exists():
        log_error(f"Input path not found: {input_path}")
        log_error("请指定 --input，或先运行 step1 生成验证集。")
        sys.exit(1)

    conf = args.conf if args.conf is not None else cfg.get("conf", 0.25)
    iou = args.iou if args.iou is not None else cfg.get("iou", 0.45)

    run_inference(data_path, input_path, task_type,
                  conf, iou, args.model, args.label_dir)


if __name__ == "__main__":
    main()
