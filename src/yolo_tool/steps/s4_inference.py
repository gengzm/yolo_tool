#!/usr/bin/env python
"""
Step 4: 推理
- 指定输入路径（图片目录或单张图片）
- 使用最新的训练模型进行推理
- 输出：
  * 推理可视化   — 推理结果叠加标注的图片
  * 推理JSON     — 推理结果 JSON 格式（方便后续叠加训练集）
  * 误差分析     — 预测与真值误差图表（平均误差、中值误差、标准差），
                   bbox 中心/宽高误差 + 点误差（obb 顶点/segment 轮廓/pose 关键点）

目录结构（统一放数据根目录 {DATA_ROOT}，不在训练集内）:
    {DATA_ROOT}/推理结果/
    ├── 推理可视化/
    ├── 推理json/
    └── 误差分析/

用法:
    yolo-tool s4 \\
        --data ./yolo_dataset/data.yaml \\
        --input ./test_images \\
        --task_type detect \\
        --conf 0.25 \\
        --iou 0.45

# 也可以直接用单张图片：
    yolo-tool s4 \\
        --data ./yolo_dataset/data.yaml \\
        --input ./test_images/img001.jpg \\
        --task_type segment
"""
import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境绘图
import matplotlib.pyplot as plt
import numpy as np
import yaml
from ultralytics import YOLO

from ..core import config
from ..core.config import CLASS_NAMES, CLASS_COLORS_BGR, POINT_COLORS_BGR, IMAGE_EXTENSIONS
from ..core.utils import (
    ensure_dir, log_info, log_warn, log_error,
    get_project_config, load_labelme_json, normalize_points,
)


def _training_complete(train_dir: Path) -> bool:
    """训练是否跑完设定 epoch（results.csv 数据行数 >= args.yaml 的 epochs）"""
    args_p = train_dir / "args.yaml"
    results_p = train_dir / "results.csv"
    if not args_p.exists() or not results_p.exists():
        return False
    try:
        args = yaml.safe_load(args_p.read_text(encoding="utf-8")) or {}
        epochs = int(args.get("epochs") or 0)
        with open(results_p, encoding="utf-8") as f:
            rows = sum(1 for _ in f) - 1  # 减去表头行
        return epochs > 0 and rows >= max(1, int(epochs * 0.95))
    except Exception:
        return False


def find_latest_model(run_out_dir: str) -> str:
    """
    自动查找 run_out 下最新且训练完成的 best.pt 模型。
    优先完整训练（results.csv 已达 args.yaml 设定的 epochs），
    避免选中训到一半就中断的坏模型；无完整训练时才退回最新 best.pt 并警告。
    """
    run_out_path = Path(run_out_dir)
    if not run_out_path.exists():
        log_error(f"run_out directory not found: {run_out_path}")
        sys.exit(1)

    # 遍历所有时间戳子目录，找最新
    best_models = list(run_out_path.rglob("**/weights/best.pt"))
    if not best_models:
        log_error(f"No best.pt found in {run_out_path}")
        sys.exit(1)

    # 优先完整训练，跳过中断/未完成的
    completed = [p for p in best_models if _training_complete(p.parent.parent)]
    pool = completed or best_models
    if not completed:
        log_warn("No completed training found, falling back to latest best.pt "
                 "(may be an interrupted run)")

    # 按修改时间排序，取最新的
    pool.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    latest = str(pool[0])
    log_info(f"Latest model: {latest}")
    return latest


def get_output_dirs(data_root: str) -> dict:
    """获取推理输出目录（统一放数据根目录，不放训练集内）"""
    base = Path(data_root) / "推理结果"
    vis_dir = ensure_dir(str(base / "推理可视化"))
    json_dir = ensure_dir(str(base / "推理json"))
    err_dir = ensure_dir(str(base / "误差分析"))
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


def load_ground_truth(label_dir: str, image_stem: str,
                      json_dir: str = None, task_type: str = "detect") -> list:
    """
    加载指定图片的真值标注
    优先读 YOLO txt（{label_dir}/{stem}.txt，归一化坐标），按任务类型解析：
      detect: class cx cy w h                 → bbox
      pose:   class cx cy w h x y v x y v...  → bbox + 关键点
      segment/obb: class x1 y1 x2 y2 ...      → 点集 + 由点计算的 bbox
    无 txt 时回退读 LabelMe json（{json_dir}/{stem}.json，绝对像素转归一化）
    返回: list of dicts: [{"class_id": int, "bbox": (cx,cy,w,h), "points": [[],...]}, ...]
    """
    # 1) YOLO txt 优先
    label_path = Path(label_dir) / f"{image_stem}.txt"
    if label_path.exists():
        gt_list = []
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = int(parts[0])
                values = [float(x) for x in parts[1:]]
                item = {"class_id": class_id}
                if task_type in ("segment", "obb"):
                    # 全部是点：class x1 y1 x2 y2 ...
                    pts = np.array(values).reshape(-1, 2)
                    item["points"] = pts.tolist()
                    xs, ys = pts[:, 0], pts[:, 1]
                    item["bbox"] = ((xs.min() + xs.max()) / 2,
                                    (ys.min() + ys.max()) / 2,
                                    xs.max() - xs.min(),
                                    ys.max() - ys.min())
                elif task_type == "pose" and len(values) >= 4:
                    item["bbox"] = tuple(values[:4])
                    if len(values) > 4:
                        kpt = np.array(values[4:]).reshape(-1, 3)
                        item["points"] = [(x, y) for x, y, v in kpt if v > 0]
                elif len(values) >= 4:
                    item["bbox"] = tuple(values[:4])
                gt_list.append(item)
        return gt_list

    # 2) 回退 LabelMe json（真值点来源之一）
    if json_dir:
        json_path = Path(json_dir) / f"{image_stem}.json"
        if json_path.exists():
            data = load_labelme_json(str(json_path))
            img_w = data.get("imageWidth", 0)
            img_h = data.get("imageHeight", 0)
            gt_list = []
            for shape in data.get("shapes", []):
                pts = np.array(shape.get("points", []), dtype=np.float64)
                if len(pts) == 0:
                    continue
                label = shape.get("label", "")
                class_id = (config.CLASS_NAMES.index(label)
                            if label in config.CLASS_NAMES else None)
                if class_id is None:
                    continue
                item = {"class_id": class_id}
                if img_w > 0 and img_h > 0:
                    norm = normalize_points(pts, img_w, img_h)
                    item["points"] = norm.tolist()
                    xs, ys = pts[:, 0], pts[:, 1]
                    item["bbox"] = (
                        (xs.min() + xs.max()) / 2 / img_w,
                        (ys.min() + ys.max()) / 2 / img_h,
                        (xs.max() - xs.min()) / img_w,
                        (ys.max() - ys.min()) / img_h,
                    )
                gt_list.append(item)
            return gt_list

    return []


def calc_bbox_errors(gt_boxes, pred_boxes, img_w: int, img_h: int) -> list:
    """
    计算 bbox 误差：中心点偏移、宽高误差 (像素)
    同类别贪心匹配：每个 GT 只与同 class_id 的预测匹配；
    该类未检出时跳过，避免跨类别误匹配产生假大误差。
    """
    errors = []
    if not gt_boxes or not pred_boxes:
        return errors

    # 同类别贪心匹配：每个 GT 只匹配同 class_id 的预测
    for gt in gt_boxes:
        pool = [p for p in pred_boxes if p["class_id"] == gt["class_id"]]
        if not pool:
            continue  # 该类未检出，跳过

        gt_cx, gt_cy, gt_w, gt_h = gt["bbox"]
        gt_cx_px = gt_cx * img_w
        gt_cy_px = gt_cy * img_h
        gt_w_px = gt_w * img_w
        gt_h_px = gt_h * img_h

        best_dist = float("inf")
        best_pred = None
        for pred in pool:
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


def _bbox_corners(cx, cy, w, h, class_id=None):
    """bbox (cx,cy,w,h) 转 4 角点，可选带 class_id"""
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    pts = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    if class_id is not None:
        return [(px, py, class_id) for px, py in pts]
    return pts


def extract_pred_points(result, task_type: str) -> list:
    """
    提取预测实例的点集（像素坐标），每个点带 class_id
    detect: bbox 4 角点；obb: 4 顶点；segment: 轮廓点；pose: 可见关键点
    返回: [[(x, y, class_id), ...], ...] 每个实例一个点集
    """
    sets = []
    if result.boxes is None and (task_type != "obb" or getattr(result, "obb", None) is None):
        return sets
    if task_type == "detect" and result.boxes is not None:
        cls_ids = result.boxes.cls.cpu().numpy()
        for i, (cx, cy, w, h) in enumerate(result.boxes.xywh.cpu().numpy()):
            sets.append(_bbox_corners(cx, cy, w, h, int(cls_ids[i])))
        return sets
    if task_type == "obb" and result.obb is not None:
        cls_ids = result.obb.cls.cpu().numpy()
        for i, pts in enumerate(result.obb.xyxyxyxy):
            cid = int(cls_ids[i])
            sets.append([(float(p[0]), float(p[1]), cid)
                         for p in pts.cpu().numpy()])
    elif task_type == "segment" and result.masks is not None:
        cls_ids = result.boxes.cls.cpu().numpy()
        for i, pts in enumerate(result.masks.xy):
            cid = int(cls_ids[i])
            sets.append([(float(p[0]), float(p[1]), cid) for p in pts])
    elif task_type == "pose" and result.keypoints is not None:
        cls_ids = result.boxes.cls.cpu().numpy()
        for i, kpts in enumerate(result.keypoints.data):
            cid = int(cls_ids[i])
            pts = [(float(x), float(y), cid)
                   for x, y, c in kpts.cpu().numpy() if c > 0.5]
            if pts:
                sets.append(pts)
    return sets


def calc_point_errors(gt_items: list, pred_point_sets: list,
                      img_w: int, img_h: int) -> list:
    """
    计算点误差：真值每个点匹配最近且同类别（class_id）的预测点（贪心）。
    gt_items: load_ground_truth 结果（points 为归一化坐标）
    pred_point_sets: extract_pred_points 结果（像素坐标，点带 class_id）
    某类别目标未检出（无同类别预测点）时跳过，
    避免跨类别误匹配产生假大误差。
    返回: [{"class_id": int, "class_name": str, "error_px": float}, ...]
    """
    errors = []
    # 按类别组织预测点
    pred_by_cls = {}
    for pts in pred_point_sets:
        for (x, y, cid) in pts:
            pred_by_cls.setdefault(int(cid), []).append((float(x), float(y)))
    if not pred_by_cls:
        return errors
    for gt in gt_items:
        gt_pts = gt.get("points")
        if not gt_pts:
            # detect 等无显式点集时，用 bbox 4 角点做点对比较
            if gt.get("bbox"):
                gt_pts = _bbox_corners(*gt["bbox"])
        if not gt_pts:
            continue
        cid = gt["class_id"]
        cls_name = CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else f"cls{cid}"
        pool = pred_by_cls.get(cid)
        if not pool:
            continue  # 该类未检出，不计入误差（避免误匹配）
        for gx_n, gy_n in gt_pts:
            gx, gy = gx_n * img_w, gy_n * img_h
            bx, by = min(pool,
                         key=lambda p: (p[0] - gx) ** 2 + (p[1] - gy) ** 2)
            errors.append({
                "class_id": cid,
                "class_name": cls_name,
                "error_px": float(np.hypot(bx - gx, by - gy)),
            })
    return errors


def _summarize(values) -> dict:
    """均值 / 中值 / 标准差 / min / max 汇总"""
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _plot_hist_with_stats(ax, data, title, xlabel, color, stat):
    """直方图 + 均值/中值竖线（平均误差、中值误差、标准差）"""
    ax.hist(data, bins=30, color=color, edgecolor="white", alpha=0.8)
    ax.axvline(stat["mean"], color="red", linestyle="--",
               label=f"Mean: {stat['mean']:.2f}")
    ax.axvline(stat["median"], color="green", linestyle="--",
               label=f"Median: {stat['median']:.2f}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(f"{title}\n"
                 f"Mean={stat['mean']:.2f}, Median={stat['median']:.2f}, Std={stat['std']:.2f}")
    ax.legend()


def _plot_stats_bar(stats: dict, err_dir: str, task_type: str = "detect"):
    """Mean / Median / Std 汇总条形图（仅像素类误差指标）"""
    keys = [k for k in ("center_error", "width_error", "height_error",
                        "point_error") if k in stats]
    if not keys:
        return
    labels = {"center_error": "Center Error", "width_error": "Width Error",
              "height_error": "Height Error", "point_error": "Point Error"}
    names = [labels[k] for k in keys]
    means = [stats[k]["mean"] for k in keys]
    medians = [stats[k]["median"] for k in keys]
    stds = [stats[k]["std"] for k in keys]

    x = np.arange(len(names))
    width = 0.25
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, means, width, label="Mean", color="steelblue")
    ax.bar(x, medians, width, label="Median", color="coral")
    ax.bar(x + width, stds, width, label="Std", color="mediumseagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("pixels (px)")
    ax.set_title(f"{task_type} - Error Metrics: Mean / Median / Std (px)")
    ax.legend()
    for xi, (m, md, s) in enumerate(zip(means, medians, stds)):
        ax.text(xi - width, m, f"{m:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi, md, f"{md:.1f}", ha="center", va="bottom", fontsize=8)
        ax.text(xi + width, s, f"{s:.1f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    bar_path = os.path.join(err_dir, "error_stats_bar.png")
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log_info(f"Error stats bar chart saved: {bar_path}")


def _plot_class_error(task_type: str, cname: str,
                      grp_all: list, grp_pts: list, save_path: str):
    """
    每个类别一张图：单个 ax 占满整张画布。
    点误差直方图：横轴 = 误差(px)，纵轴 = 点数量，
    Mean / Median 用虚线在图中标出。
    detect 用 bbox 4 角点，obb 用 4 顶点，segment 用轮廓点，pose 用关键点。
    """
    if not grp_pts:
        return
    vals = np.asarray([p["error_px"] for p in grp_pts], dtype=float)
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.hist(vals, bins=30, color="steelblue", edgecolor="white", alpha=0.85)
    ax.axvline(vals.mean(), color="red", linestyle="--", linewidth=1.8,
               label=f"Mean: {vals.mean():.2f}px")
    ax.axvline(np.median(vals), color="green", linestyle="--", linewidth=1.8,
               label=f"Median: {np.median(vals):.2f}px")
    ax.set_xlabel("Error (pixels)")
    ax.set_ylabel("Point Count")
    ax.set_title(f"{task_type} - {cname} Point Error\n"
                 f"Mean={vals.mean():.2f}px, Median={np.median(vals):.2f}px, "
                 f"Std={vals.std():.2f}px (n={len(vals)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log_info(f"Error chart saved: {save_path}")


def _build_class_stats(grp_all: list, grp_pts: list) -> tuple:
    """
    由一组误差数据计算 stats dict 与 plot_specs 列表
    返回: (stats, plot_specs)
    """
    stats = {}
    plot_specs = []
    if grp_all:
        center_errors = [e["center_error_px"] for e in grp_all]
        width_errors = [e["width_error_px"] for e in grp_all]
        height_errors = [e["height_error_px"] for e in grp_all]
        ious = [e["iou"] for e in grp_all]
        stats["total_comparisons"] = len(grp_all)
        stats["center_error"] = _summarize(center_errors)
        stats["width_error"] = _summarize(width_errors)
        stats["height_error"] = _summarize(height_errors)
        stats["iou"] = _summarize(ious)
        plot_specs = [
            ("center_error", center_errors, "Center Point Error",
             "Center Error (pixels)", "steelblue"),
            ("width_error", width_errors, "Width Error",
             "Width Error (pixels)", "coral"),
            ("height_error", height_errors, "Height Error",
             "Height Error (pixels)", "mediumseagreen"),
            ("iou", ious, "IoU Distribution", "IoU", "mediumpurple"),
        ]
    if grp_pts:
        pt_vals = [p["error_px"] for p in grp_pts]
        stats["point_error"] = _summarize(pt_vals)
        plot_specs.append(("point_error", pt_vals, "Point Error",
                           "Point Error (pixels)", "darkorange"))
    return stats, plot_specs


def generate_error_report(all_errors: list, point_errors: list,
                          err_dir: str, task_type: str = "detect"):
    """
    生成误差分析报告（JSON 统计 + 图表，每个类别一张图）
    - error_stats.json         各指标 Mean/Median/Std/Min/Max（整体 + 按类别 per_class）
    - error_analysis.png       全类别总览误差分布直方图
    - error_analysis_{类别}.png 每个类别一张误差分布直方图
    - error_stats_bar.png      Mean / Median / Std 汇总条形图
    每张图 title 均包含任务类型 + 类别名称。
    """
    if not all_errors and not point_errors:
        log_warn("No errors to report (no ground truth available)")
        return

    ensure_dir(err_dir)

    # 按类别分组：class_name -> {"all_errors": [...], "point_errors": [...]}
    classes = {}
    for e in all_errors:
        classes.setdefault(e["class_name"],
                           {"all_errors": [], "point_errors": []})["all_errors"].append(e)
    for p in point_errors:
        classes.setdefault(p["class_name"],
                           {"all_errors": [], "point_errors": []})["point_errors"].append(p)

    # 整体统计 + 图表规格
    overall_stats, overall_specs = _build_class_stats(all_errors, point_errors)
    stats = dict(overall_stats)
    stats["per_class"] = {
        cname: _build_class_stats(grp["all_errors"], grp["point_errors"])[0]
        for cname, grp in classes.items()
    }

    # 保存统计 JSON
    stats_path = os.path.join(err_dir, "error_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    log_info(f"Error stats saved: {stats_path}")

    # 全类别总览图（一张图、一个占满画布的图表）
    if overall_specs:
        _plot_class_error(task_type, "All Classes", all_errors, point_errors,
                          os.path.join(err_dir, "error_analysis.png"))

    # 每个类别一张图（title 含任务类型 + 类别名称）
    for cname, grp in classes.items():
        if not grp["all_errors"] and not grp["point_errors"]:
            continue
        safe_name = re.sub(r'[^\w\u4e00-\u9fff-]+', '_', cname)
        _plot_class_error(task_type, cname, grp["all_errors"], grp["point_errors"],
                          os.path.join(err_dir, f"error_analysis_{safe_name}.png"))

    # 汇总条形图（整体）
    _plot_stats_bar(overall_stats, err_dir, task_type)

    # 打印统计摘要（整体 + 按类别）
    log_info("=" * 60)
    log_info(f"Error Analysis Summary ({task_type}):")
    names = {"center_error": "Center Error", "width_error": "Width Error",
             "height_error": "Height Error", "iou": "IoU",
             "point_error": "Point Error"}
    for key in ("center_error", "width_error", "height_error",
                "point_error", "iou"):
        if key not in stats:
            continue
        s = stats[key]
        if key == "iou":
            log_info(f"  {names[key]:<14} - Mean: {s['mean']:.3f}, "
                     f"Median: {s['median']:.3f}, STD: {s['std']:.3f}")
        else:
            log_info(f"  {names[key]:<14} - Mean: {s['mean']:.2f}px, "
                     f"Median: {s['median']:.2f}px, STD: {s['std']:.2f}px")
    for cname, gstats in stats["per_class"].items():
        parts = []
        for key in ("center_error", "point_error"):
            if key in gstats:
                s = gstats[key]
                parts.append(f"{names[key]} mean={s['mean']:.2f}px "
                             f"median={s['median']:.2f}px (n={s['count']})")
        log_info(f"  [{cname}] " + "; ".join(parts) if parts else f"  [{cname}]")
    log_info("=" * 60)


def draw_prediction(img: np.ndarray, result, task_type: str,
                    class_colors: list = None) -> np.ndarray:
    """在图片上绘制推理结果，覆盖 detect / segment / obb / pose 四种任务"""
    if class_colors is None:
        class_colors = CLASS_COLORS_BGR

    obb_data = getattr(result, "obb", None)
    # obb 模型 result.boxes 为 None，预测数据在 result.obb
    if result.boxes is None and (obb_data is None or len(obb_data) == 0):
        return img

    n = len(result.boxes) if result.boxes is not None else len(obb_data)
    for i in range(n):
        if result.boxes is not None:
            box = result.boxes[i]
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        else:
            cls_id = int(obb_data.cls[i])
            conf = float(obb_data.conf[i])
            xs = obb_data.xyxyxyxy[i][:, 0].cpu().numpy()
            ys = obb_data.xyxyxyxy[i][:, 1].cpu().numpy()
            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        color = class_colors[cls_id % len(class_colors)]

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
        boxes = result.boxes
        obb_data = getattr(result, "obb", None)
        # obb 模型 result.boxes 为 None，预测数据在 result.obb
        if boxes is None and (obb_data is None or len(obb_data) == 0):
            continue
        n = len(boxes) if boxes is not None else len(obb_data)
        for i in range(n):
            if boxes is not None:
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                xyxy = boxes.xyxy[i].tolist()
            else:
                cls_id = int(obb_data.cls[i])
                conf = float(obb_data.conf[i])
                xyxy = obb_data.xyxy[i].tolist()
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
                  model_path: str = None, label_dir: str = None,
                  json_dir: str = None):
    """
    执行推理主流程
    json_dir: LabelMe 真值 json 目录（无 YOLO txt 时用于误差分析）
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
        info = get_project_config(dataset_dir)
        recorded = (info.get("weights") or {}).get("best")
        if recorded and Path(recorded).exists():
            model_path = recorded
            log_info(f"Using recorded model: {model_path}")
        else:
            model_path = find_latest_model(
                info.get("run_out_dir") or str(Path(dataset_dir) / "run_out")
            )
    if not os.path.exists(model_path):
        log_error(f"Model not found: {model_path}")
        sys.exit(1)

    # 输出目录（统一放数据根目录 {DATA_ROOT}，不在训练集内）
    out_dirs = get_output_dirs(get_project_config(dataset_dir)["data_root"])

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
    all_point_errors = []

    for img_path in images:
        log_info(f"Processing: {os.path.basename(img_path)}")

        # 推理
        results = model(img_path, conf=conf, iou=iou, verbose=False)

        # 读取图片用于可视化
        img = cv2.imread(img_path)
        if img is None:
            log_warn(f"Cannot read image: {img_path}")
            continue

        # 可视化（jpg 省空间）
        vis_img = draw_prediction(img.copy(), results[0], task_type)
        vis_path = os.path.join(out_dirs["vis"], Path(img_path).stem + ".jpg")
        cv2.imwrite(vis_path, vis_img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # 提取 JSON 结果
        json_result = extract_predictions_json(results, img_path)
        json_name = Path(img_path).stem + ".json"
        json_path = os.path.join(out_dirs["json"], json_name)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_result, f, indent=2, ensure_ascii=False)

        # 误差分析（如果有真值）
        if label_dir and Path(label_dir).exists():
            gt_items = load_ground_truth(label_dir, Path(img_path).stem,
                                         json_dir=json_dir,
                                         task_type=task_type)
            if gt_items:
                img_w, img_h = img.shape[1], img.shape[0]
                # OBB 模型 result.boxes 为 None，预测框从 result.obb 顶点取
                pred_boxes = []
                obb_data = getattr(results[0], "obb", None)
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        cx, cy, bw, bh = box.xywh[0].tolist()
                        pred_boxes.append({
                            "class_id": cls_id,
                            "bbox": (cx / img_w, cy / img_h,
                                     bw / img_w, bh / img_h),
                        })
                elif obb_data is not None and len(obb_data):
                    cls_ids = obb_data.cls.cpu().numpy()
                    for i, xyxyxyxy in enumerate(obb_data.xyxyxyxy):
                        pts = xyxyxyxy.cpu().numpy()
                        xs, ys = pts[:, 0], pts[:, 1]
                        pred_boxes.append({
                            "class_id": int(cls_ids[i]),
                            "bbox": (xs.mean() / img_w, ys.mean() / img_h,
                                     (xs.max() - xs.min()) / img_w,
                                     (ys.max() - ys.min()) / img_h),
                        })
                img_errors = calc_bbox_errors(gt_items, pred_boxes, img_w, img_h)
                all_errors.extend(img_errors)

                # 点误差（detect bbox 角点 / obb 顶点 / segment 轮廓 / pose 关键点）
                pred_sets = extract_pred_points(results[0], task_type)
                all_point_errors.extend(
                    calc_point_errors(gt_items, pred_sets, img_w, img_h))

    # 误差分析报告（bbox 误差 + 点误差，每个类别一张图）
    generate_error_report(all_errors, all_point_errors,
                          out_dirs["error"], task_type)

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
                        help="真值 label 目录（YOLO txt，用于误差分析）")
    parser.add_argument("--json_dir", type=str, default=None,
                        help="真值 json 目录（LabelMe，无 txt 时用于误差分析，默认取项目 source_dir）")
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

    # 真值 json 目录：显式参数 > 项目 source_dir
    json_dir = args.json_dir
    if json_dir is None:
        src = cfg.get("source_dir")
        if src and Path(src).exists():
            json_dir = src

    run_inference(data_path, input_path, task_type,
                  conf, iou, args.model, args.label_dir, json_dir)


if __name__ == "__main__":
    main()
