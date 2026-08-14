#!/usr/bin/env python
"""
Step 2: 标注可视化
- 读取指定目录的图片及其 YOLO 标注文件
- 将标注绘制到图片上，保存到"可视化标注"目录
- 可视化规则：
  * 20 种不同类别颜色
  * 20 种不同点颜色
  * 每个类别第一个点：半径 5px 实心圆，其余点：半径 3px 实心圆
  * 点之间连线宽度 1px
  * 检测框显示 bbox 矩形

用法:
    python s2_visualize.py \\
        --source_dir ./yolo_dataset/train/images \\
        --label_dir ./yolo_dataset/train/labels \\
        --output_dir ./yolo_dataset/可视化标注 \\
        --task_type segment

# 也可直接使用 step1 输出的数据集目录自动寻找：
    python s2_visualize.py \\
        --dataset_dir ./yolo_dataset \\
        --split train \\
        --task_type segment
"""
import argparse
import os
import yaml
from pathlib import Path

import cv2
import numpy as np

import config
from config import CLASS_NAMES, CLASS_COLORS_BGR, POINT_COLORS_BGR, IMAGE_EXTENSIONS
from utils import (
    ensure_dir, denormalize_points, log_info, log_warn, load_class_names_from_yaml,
    get_project_config,
)

# ---------- 可视化样式参数（可通过命令行覆盖，默认值见下） ----------
CIRCLE_DIAMETER_FIRST = 6   # 第一个点直径 px
CIRCLE_DIAMETER_OTHER = 4   # 其他点直径 px
LINE_WIDTH = 2              # 线宽 px（bbox / 多边形 / 骨架 / 点连线）


def draw_detection(img: np.ndarray, class_id: int, cx: float, cy: float,
                   w: float, h: float) -> np.ndarray:
    """绘制检测框"""
    h_img, w_img = img.shape[:2]
    x1 = int((cx - w / 2) * w_img)
    y1 = int((cy - h / 2) * h_img)
    x2 = int((cx + w / 2) * w_img)
    y2 = int((cy + h / 2) * h_img)

    color = CLASS_COLORS_BGR[class_id % len(CLASS_COLORS_BGR)]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=LINE_WIDTH)

    # 类别标签
    label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"cls{class_id}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 2)

    return img


def draw_segmentation(img: np.ndarray, class_id: int,
                      points: np.ndarray) -> np.ndarray:
    """绘制分割多边形及关键点"""
    if len(points) == 0:
        return img

    h_img, w_img = img.shape[:2]
    pts = denormalize_points(points, w_img, h_img).astype(np.int32)
    class_color = CLASS_COLORS_BGR[class_id % len(CLASS_COLORS_BGR)]

    # 画填充多边形（半透明）
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts.reshape((-1, 1, 2))], class_color)
    img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)

    # 画边线
    cv2.polylines(img, [pts.reshape((-1, 1, 2))], isClosed=True,
                  color=class_color, thickness=LINE_WIDTH)

    # 画点
    _draw_points_on_image(img, pts, class_id)

    # 类别标签
    label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"cls{class_id}"
    cv2.putText(img, label, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, class_color, 2)

    return img


def draw_obb(img: np.ndarray, class_id: int,
             points: np.ndarray) -> np.ndarray:
    """绘制旋转框"""
    if len(points) < 4:
        return img

    h_img, w_img = img.shape[:2]
    pts = denormalize_points(points[:4], w_img, h_img).astype(np.int32)
    class_color = CLASS_COLORS_BGR[class_id % len(CLASS_COLORS_BGR)]

    cv2.polylines(img, [pts.reshape((-1, 1, 2))], isClosed=True,
                  color=class_color, thickness=LINE_WIDTH)

    _draw_points_on_image(img, pts, class_id)

    label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"cls{class_id}"
    cv2.putText(img, label, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, class_color, 2)

    return img


def draw_pose(img: np.ndarray, class_id: int, bbox: tuple,
              keypoints: np.ndarray) -> np.ndarray:
    """
    绘制关键点标注
    keypoints: shape (N, 3) — (x, y, visibility) 归一化
    """
    h_img, w_img = img.shape[:2]
    class_color = CLASS_COLORS_BGR[class_id % len(CLASS_COLORS_BGR)]

    # 画 bbox
    cx, cy, w, h = bbox
    x1 = int((cx - w / 2) * w_img)
    y1 = int((cy - h / 2) * h_img)
    x2 = int((cx + w / 2) * w_img)
    y2 = int((cy + h / 2) * h_img)
    cv2.rectangle(img, (x1, y1), (x2, y2), class_color, thickness=LINE_WIDTH)

    # 画关键点及连线（骨架连接, 基于 COCO 17-keypoint 默认骨架）
    skeleton = [
        (0, 1), (0, 2), (1, 3), (2, 4),  # 头→肩
        (5, 6),                             # 左右肩
        (5, 7), (7, 9), (6, 8), (8, 10),  # 手臂
        (5, 11), (6, 12), (11, 12),        # 髋
        (11, 13), (13, 15), (12, 14), (14, 16),  # 腿
    ]

    # 提取有效关键点
    valid_pts = []
    for i, (kx, ky, vis) in enumerate(keypoints):
        if vis >= 1 and kx > 0 and ky > 0:
            valid_pts.append((i, int(kx * w_img), int(ky * h_img)))

    # 画骨骼连线
    for (a, b) in skeleton:
        pa = next(((px, py) for idx, px, py in valid_pts if idx == a), None)
        pb = next(((px, py) for idx, px, py in valid_pts if idx == b), None)
        if pa and pb:
            color = POINT_COLORS_BGR[a % len(POINT_COLORS_BGR)]
            cv2.line(img, pa, pb, color, thickness=LINE_WIDTH)

    # 画关键点
    for idx, (i, px, py) in enumerate(valid_pts):
        is_first = (idx == 0)
        radius = (CIRCLE_DIAMETER_FIRST if is_first else CIRCLE_DIAMETER_OTHER) // 2
        color = POINT_COLORS_BGR[i % len(POINT_COLORS_BGR)]
        cv2.circle(img, (px, py), radius, color, -1)
        cv2.circle(img, (px, py), radius, (255, 255, 255), 1)

    # 类别标签
    label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"cls{class_id}"
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, class_color, 2)

    return img


def _draw_points_on_image(img: np.ndarray, pts: np.ndarray, class_id: int):
    """
    通用点绘制：第一个点直径 CIRCLE_DIAMETER_FIRST，其余点
    CIRCLE_DIAMETER_OTHER，点颜色 20 种，点间连线宽 LINE_WIDTH
    pts: 绝对像素坐标 (N, 2)
    """
    for i, pt in enumerate(pts):
        px, py = int(pt[0]), int(pt[1])
        is_first = (i == 0)
        radius = (CIRCLE_DIAMETER_FIRST if is_first else CIRCLE_DIAMETER_OTHER) // 2
        point_color = POINT_COLORS_BGR[i % len(POINT_COLORS_BGR)]
        cv2.circle(img, (px, py), radius, point_color, -1)
        # 点之间连线
        if i > 0:
            prev_pt = (int(pts[i - 1][0]), int(pts[i - 1][1]))
            line_color = CLASS_COLORS_BGR[class_id % len(CLASS_COLORS_BGR)]
            cv2.line(img, prev_pt, (px, py), line_color, thickness=LINE_WIDTH)


def parse_yolo_label_line(line: str, task_type: str):
    """
    解析一行 YOLO 标注
    返回: (class_id, data_dict)
    detect:  {"bbox": (cx, cy, w, h)}
    segment: {"points": np.array}
    obb:     {"points": np.array}
    pose:    {"bbox": (cx, cy, w, h), "keypoints": np.array}
    """
    parts = line.strip().split()
    if not parts:
        return None, None

    class_id = int(parts[0])
    values = [float(x) for x in parts[1:]]

    if task_type == "detect":
        if len(values) >= 4:
            return class_id, {"bbox": tuple(values[:4])}
    elif task_type == "segment":
        if len(values) >= 4:
            pts = np.array(values).reshape(-1, 2)
            return class_id, {"points": pts}
    elif task_type == "obb":
        if len(values) >= 8:
            pts = np.array(values[:8]).reshape(-1, 2)
            return class_id, {"points": pts}
    elif task_type == "pose":
        if len(values) >= 5:
            bbox = tuple(values[:4])
            kpt_data = values[4:]
            kpt_count = len(kpt_data) // 3
            kpts = np.array(kpt_data[:kpt_count * 3]).reshape(-1, 3)
            return class_id, {"bbox": bbox, "keypoints": kpts}

    return None, None


def visualize_image(img_path: str, label_path: str, output_path: str,
                    task_type: str):
    """对单张图片进行可视化"""
    img = cv2.imread(img_path)
    if img is None:
        log_warn(f"Cannot read image: {img_path}")
        return

    if not os.path.exists(label_path):
        log_warn(f"No label file: {label_path}, saving original image")
        cv2.imwrite(output_path, img)
        return

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        class_id, data = parse_yolo_label_line(line, task_type)
        if class_id is None or data is None:
            continue

        if task_type == "detect" and "bbox" in data:
            cx, cy, w, h = data["bbox"]
            img = draw_detection(img, class_id, cx, cy, w, h)
        elif task_type == "segment" and "points" in data:
            img = draw_segmentation(img, class_id, data["points"])
        elif task_type == "obb" and "points" in data:
            img = draw_obb(img, class_id, data["points"])
        elif task_type == "pose" and "bbox" in data:
            img = draw_pose(img, class_id, data["bbox"], data["keypoints"])

    cv2.imwrite(output_path, img)


def visualize_dataset_split(dataset_dir: str, split: str, task_type: str):
    """
    可视化 YOLO 数据集的某个子集 (train/val/test)
    输出到 {dataset_dir}同级/可视化标注/{split}/
    自动从 data.yaml 加载类别名
    """
    # 尝试从 data.yaml 加载类别名
    yaml_path = Path(dataset_dir) / "data.yaml"
    if yaml_path.exists():
        names = load_class_names_from_yaml(str(yaml_path))
        if names:
            config.set_class_names(names)
            log_info(f"Loaded {len(names)} class(es) from {yaml_path}")

    img_dir = Path(dataset_dir) / split / "images"
    lbl_dir = Path(dataset_dir) / split / "labels"
    # 可视化输出目录：优先用 config.DEFAULT_VISUALIZE_DIR，未配置则默认数据集同级"可视化标注"
    viz_root = config.DEFAULT_VISUALIZE_DIR
    if not viz_root:
        viz_root = str(Path(dataset_dir).parent / "可视化标注")
    out_dir = Path(viz_root) / split

    if not img_dir.exists():
        log_warn(f"Image directory not found: {img_dir}")
        return

    ensure_dir(str(out_dir))
    log_info(f"Visualizing {split} split: {img_dir}")

    # 支持两种模式：YOLO label 目录，或 LabelMe JSON
    img_files = [f for f in img_dir.iterdir()
                 if f.suffix.lower() in IMAGE_EXTENSIONS]

    for img_file in img_files:
        # 优先使用 YOLO label
        label_file = lbl_dir / f"{img_file.stem}.txt"
        if not label_file.exists():
            log_warn(f"No YOLO label for {img_file.name}, skipping")
            continue

        # 可视化输出统一为 jpg，节省空间
        out_file = out_dir / f"{img_file.stem}.jpg"
        visualize_image(str(img_file), str(label_file), str(out_file), task_type)

    log_info(f"Visualization saved to: {out_dir}")


def visualize_source_dir(source_dir: str, label_dir: str, output_dir: str,
                         task_type: str):
    """
    可视化任意包含图片和 YOLO label 的目录
    """
    ensure_dir(output_dir)
    img_dir = Path(source_dir)
    lbl_dir = Path(label_dir)

    img_files = [f for f in img_dir.iterdir()
                 if f.suffix.lower() in IMAGE_EXTENSIONS]
    log_info(f"Visualizing {len(img_files)} images from {source_dir}")

    for img_file in img_files:
        label_file = lbl_dir / f"{img_file.stem}.txt"
        label_path = str(label_file) if label_file.exists() else None
        # 可视化输出统一为 jpg，节省空间
        out_file = str(Path(output_dir) / f"{img_file.stem}.jpg")

        if label_path is None:
            log_warn(f"No YOLO label for {img_file.name}, skipping")
            continue

        visualize_image(str(img_file), label_path, out_file, task_type)

    log_info(f"Visualization saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="YOLO 标注可视化工具")
    parser.add_argument("--source_dir", type=str, default=None,
                        help="图片目录")
    parser.add_argument("--label_dir", type=str, default=None,
                        help="YOLO label 目录（.txt）")
    parser.add_argument("--output_dir", type=str, default="./可视化标注",
                        help="可视化输出目录")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="YOLO 数据集根目录（含 data.yaml）")
    parser.add_argument("--split", type=str, default="train",
                        choices=["train", "val", "test", "all"],
                        help="可视化哪个数据子集")
    parser.add_argument("--task_type", type=str, default=None,
                        choices=["detect", "segment", "obb", "pose"],
                        help="标注类型（默认取项目 info.yaml / config）")
    parser.add_argument("--yaml", type=str, default=None,
                        help="data.yaml 路径（从中加载类别名，--source_dir 模式下推荐使用）")
    parser.add_argument("--circle_diameter_first", type=int, default=6,
                        help="第一个点直径(px)，默认 6")
    parser.add_argument("--circle_diameter_other", type=int, default=4,
                        help="其他点直径(px)，默认 4")
    parser.add_argument("--line_width", type=int, default=2,
                        help="线宽(px，bbox/多边形/骨架/点连线)，默认 2")
    args = parser.parse_args()

    # 应用可视化样式参数
    global CIRCLE_DIAMETER_FIRST, CIRCLE_DIAMETER_OTHER, LINE_WIDTH
    CIRCLE_DIAMETER_FIRST = max(1, args.circle_diameter_first)
    CIRCLE_DIAMETER_OTHER = max(1, args.circle_diameter_other)
    LINE_WIDTH = max(1, args.line_width)

    # 如果提供了 data.yaml，从中加载类别名
    if args.yaml:
        names = load_class_names_from_yaml(args.yaml)
        if names:
            config.set_class_names(names)
            log_info(f"Loaded {len(names)} class(es) from {args.yaml}")

    # 未指定 dataset_dir / task_type 时，从项目 info.yaml（数据目录内部）读取
    cfg = get_project_config(args.dataset_dir)
    if args.dataset_dir is None:
        args.dataset_dir = cfg["dataset_dir"]
    if args.task_type is None:
        args.task_type = cfg["task_type"]

    if args.dataset_dir:
        if args.split == "all":
            for sp in ["train", "val", "test"]:
                sp_dir = Path(args.dataset_dir) / sp / "images"
                if sp_dir.exists():
                    visualize_dataset_split(args.dataset_dir, sp, args.task_type)
        else:
            visualize_dataset_split(args.dataset_dir, args.split, args.task_type)
    elif args.source_dir and args.label_dir:
        visualize_source_dir(args.source_dir, args.label_dir,
                             args.output_dir, args.task_type)
    elif args.source_dir:
        # 如果只指定 source_dir，尝试在同目录找 labels 子目录
        parent = Path(args.source_dir).parent
        label_dir = str(parent / "labels")
        if Path(label_dir).exists():
            visualize_source_dir(args.source_dir, label_dir,
                                 args.output_dir, args.task_type)
        else:
            log_warn("No label_dir specified and no adjacent labels/ directory found.")
    else:
        log_warn("Please specify --dataset_dir or (--source_dir + --label_dir)")


if __name__ == "__main__":
    main()
