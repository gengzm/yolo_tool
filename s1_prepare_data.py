#!/usr/bin/env python
"""
Step 1: 数据准备
- 读取指定目录的图片和 LabelMe JSON 标注文件
- 转换为 YOLO 格式（兼容 detect / segment / obb / pose）
- 按指定比例切分训练/验证/测试集
- 生成 data.yaml 配置文件

用法:
    python s1_prepare_data.py \\
        --source_dir ./raw_data \\
        --dataset_dir ./yolo_dataset \\
        --task_type detect \\
        --train_ratio 0.8 \\
        --val_ratio 0.2

LabelMe shape_type 映射规则:
    rectangle   → detect  (bbox)
    polygon     → segment (分割) / obb (4点旋转框)
    circle      → segment (按多边形近似)
    point       → pose    (关键点)
    line        → segment (转为分割)
    linestrip   → segment (转为分割)
"""
import argparse
import os
import sys
import yaml
from pathlib import Path

import numpy as np

import config
from config import (
    CLASS_NAMES, CLASS_TO_IDX, IMAGE_EXTENSIONS, LABELME_SUFFIX,
)
from utils import (
    find_image_json_pairs, load_labelme_json, get_class_id,
    collect_class_names_from_json, points_to_yolo_bbox,
    normalize_points, split_dataset, copy_image_and_label,
    ensure_dir, log_info, log_warn, log_error, get_image_size,
    update_info_yaml, get_project_config, load_info_yaml,
)


def labelme_to_yolo_detect(shapes: list, img_w: int, img_h: int) -> str:
    """
    LabelMe → YOLO detect 格式
    每行: class_id cx cy w h
    """
    lines = []
    for shape in shapes:
        label = shape.get("label", "")
        class_id = get_class_id(label)
        if class_id is None:
            log_warn(f"Unknown class '{label}', skipping")
            continue

        shape_type = shape.get("shape_type", "rectangle")
        points = np.array(shape["points"], dtype=np.float64)

        # 所有形状转 bbox
        if shape_type == "rectangle":
            # LabelMe rectangle: 左上 + 右下
            x_min, y_min = points[0]
            x_max, y_max = points[1]
            cx = ((x_min + x_max) / 2.0) / img_w
            cy = ((y_min + y_max) / 2.0) / img_h
            bw = abs(x_max - x_min) / img_w
            bh = abs(y_max - y_min) / img_h
        elif shape_type in ("polygon", "circle", "line", "linestrip", "oriented_rectangle"):
            cx, cy, bw, bh = points_to_yolo_bbox(points, img_w, img_h)
        elif shape_type == "point":
            # 单点 → 极小 bbox (5px 边长)
            x, y = points[0]
            cx = x / img_w
            cy = y / img_h
            bw = 5.0 / img_w
            bh = 5.0 / img_h
        else:
            cx, cy, bw, bh = points_to_yolo_bbox(points, img_w, img_h)

        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return "\n".join(lines) + ("\n" if lines else "")


def labelme_to_yolo_segment(shapes: list, img_w: int, img_h: int) -> str:
    """
    LabelMe → YOLO segment 格式
    每行: class_id x1 y1 x2 y2 ... xn yn
    polygon 保留所有顶点，rectangle/circle 做多边形近似，point 忽略
    """
    lines = []
    for shape in shapes:
        label = shape.get("label", "")
        class_id = get_class_id(label)
        if class_id is None:
            log_warn(f"Unknown class '{label}', skipping")
            continue

        shape_type = shape.get("shape_type", "polygon")
        points = np.array(shape["points"], dtype=np.float64)

        if shape_type == "point":
            # 点不适合 segment，跳过
            continue

        if shape_type == "rectangle":
            # 矩形 → 4 个顶点
            x1, y1 = points[0]
            x2, y2 = points[1]
            points = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ], dtype=np.float64)

        elif shape_type == "circle":
            # 圆 → 用 36 边形近似
            center = points[0]
            edge = points[1]
            radius = np.linalg.norm(edge - center)
            angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
            points = np.array([
                [center[0] + radius * np.cos(a),
                 center[1] + radius * np.sin(a)]
                for a in angles
            ])

        # 归一化
        norm = normalize_points(points, img_w, img_h)
        coords = norm.flatten()
        coord_str = " ".join(f"{v:.6f}" for v in coords)
        lines.append(f"{class_id} {coord_str}")

    return "\n".join(lines) + ("\n" if lines else "")


def labelme_to_yolo_obb(shapes: list, img_w: int, img_h: int) -> str:
    """
    LabelMe → YOLO OBB 格式
    每行: class_id x1 y1 x2 y2 x3 y3 x4 y4
    仅将 4 点多边形（polygon / rectangle / oriented_rectangle）作为 OBB 处理，其他形状跳过
    """
    lines = []
    for shape in shapes:
        label = shape.get("label", "")
        class_id = get_class_id(label)
        if class_id is None:
            log_warn(f"Unknown class '{label}', skipping")
            continue

        shape_type = shape.get("shape_type", "polygon")
        points = np.array(shape["points"], dtype=np.float64)

        if shape_type == "rectangle":
            x1, y1 = points[0]
            x2, y2 = points[1]
            points = np.array([
                [x1, y1], [x2, y1], [x2, y2], [x1, y2]
            ], dtype=np.float64)
        # oriented_rectangle: LabelMe 旋转矩形，本身已是 4 点，直接使用
        elif shape_type not in ("polygon", "oriented_rectangle") or len(points) < 4:
            continue

        # 取前4个点
        if len(points) > 4:
            points = points[:4]

        norm = normalize_points(points, img_w, img_h)
        coords = norm.flatten()
        coord_str = " ".join(f"{v:.6f}" for v in coords)
        lines.append(f"{class_id} {coord_str}")

    return "\n".join(lines) + ("\n" if lines else "")


def labelme_to_yolo_pose(shapes: list, img_w: int, img_h: int) -> str:
    """
    LabelMe → YOLO pose 格式
    每行: class_id cx cy w h px1 py1 vis1 px2 py2 vis2 ...
    point 形状视为关键点(visibility=2)，其他形状转 bbox 但不含关键点
    """
    # 首先收集所有 point 形状作为关键点
    keypoints_per_class = {}
    bbox_shapes_per_class = {}

    for shape in shapes:
        label = shape.get("label", "")
        class_id = get_class_id(label)
        if class_id is None:
            continue

        shape_type = shape.get("shape_type", "")
        points = np.array(shape["points"], dtype=np.float64)

        if shape_type == "point" and len(points) > 0:
            if class_id not in keypoints_per_class:
                keypoints_per_class[class_id] = []
            keypoints_per_class[class_id].append(points[0])
        else:
            if class_id not in bbox_shapes_per_class:
                bbox_shapes_per_class[class_id] = []
            bbox_shapes_per_class[class_id].append(shape)

    lines = []
    # 合并同一类别的 bbox 和关键点
    all_class_ids = set(list(keypoints_per_class.keys()) + list(bbox_shapes_per_class.keys()))

    for class_id in sorted(all_class_ids):
        # 确定 bbox（优先取 bbox 形状，否则用关键点计算）
        bbox_shapes = bbox_shapes_per_class.get(class_id, [])
        kpts = keypoints_per_class.get(class_id, [])

        if bbox_shapes:
            # 取第一个非 point 形状做 bbox
            shape = bbox_shapes[0]
            points = np.array(shape["points"], dtype=np.float64)
            shape_type = shape.get("shape_type", "rectangle")
            if shape_type == "rectangle":
                x_min, y_min = points[0]
                x_max, y_max = points[1]
            else:
                x_min, y_min = points.min(axis=0)
                x_max, y_max = points.max(axis=0)
        elif kpts:
            arr = np.array(kpts)
            x_min, y_min = arr.min(axis=0)
            x_max, y_max = arr.max(axis=0)
            x_min -= 5
            y_min -= 5
            x_max += 5
            y_max += 5
        else:
            continue

        cx = ((x_min + x_max) / 2.0) / img_w
        cy = ((y_min + y_max) / 2.0) / img_h
        bw = max(abs(x_max - x_min), 1) / img_w
        bh = max(abs(y_max - y_min), 1) / img_h

        # 关键点（最多支持 17 个，如 COCO 标准）
        MAX_KPTS = 17
        kpt_str = []
        for i in range(MAX_KPTS):
            if i < len(kpts):
                kx = kpts[i][0] / img_w
                ky = kpts[i][1] / img_h
                vis = 2  # visible
                kpt_str.append(f"{kx:.6f} {ky:.6f} {vis}")
            else:
                kpt_str.append("0 0 0")

        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {' '.join(kpt_str)}")

    return "\n".join(lines) + ("\n" if lines else "")


# 转换器映射
CONVERTERS = {
    "detect": labelme_to_yolo_detect,
    "segment": labelme_to_yolo_segment,
    "obb": labelme_to_yolo_obb,
    "pose": labelme_to_yolo_pose,
}


def generate_data_yaml(dataset_dir: str, task_type: str, train_dir: str,
                       val_dir: str, test_dir: str) -> str:
    """
    生成 YOLO data.yaml
    """
    yaml_path = str(Path(dataset_dir) / "data.yaml")
    nc = len(CLASS_NAMES)

    data = {
        "path": str(Path(dataset_dir).resolve()),
        "train": str(Path(train_dir).resolve()),
        "val": str(Path(val_dir).resolve()),
        "nc": nc,
        "names": CLASS_NAMES,
    }
    if test_dir:
        data["test"] = str(Path(test_dir).resolve())

    # 对 pose 任务添加关键点形状配置
    if task_type == "pose":
        data["kpt_shape"] = [17, 3]  # 17 keypoints, (x, y, visibility)

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    log_info(f"data.yaml generated: {yaml_path}")
    return yaml_path


def process_dataset(source_dir: str, dataset_dir: str, task_type: str = "detect",
                    train_ratio: float = None, val_ratio: float = None):
    """
    主流程: 扫描 → 收集类别 → 转换 → 切分 → 复制 → 生成 yaml
    """
    train_ratio = config.TRAIN_RATIO if train_ratio is None else train_ratio
    val_ratio = config.VAL_RATIO if val_ratio is None else val_ratio
    converter = CONVERTERS.get(task_type)
    if converter is None:
        log_error(f"Unsupported task_type: {task_type}. Choose from {list(CONVERTERS.keys())}")
        sys.exit(1)

    log_info(f"Task type: {task_type}")
    log_info(f"Source dir: {source_dir}")
    log_info(f"Dataset dir: {dataset_dir}")

    # 0. 从 LabelMe JSON 中自动收集类别名
    class_names = collect_class_names_from_json(source_dir)
    if not class_names:
        log_error("No class labels found in LabelMe JSON files!")
        log_error("Please ensure source_dir contains .json files with 'shapes' that have 'label' fields.")
        sys.exit(1)

    # 应用界面保存的忽略标签（info.yaml -> labels.ignore），忽略类不参与训练、不写入 txt
    info_before = load_info_yaml(dataset_dir)
    ignore_set = set((info_before.get("labels") or {}).get("ignore", []) or [])
    if ignore_set:
        removed = [n for n in class_names if n in ignore_set]
        class_names = [n for n in class_names if n not in ignore_set]
        if removed:
            log_warn(f"labels.ignore (from info.yaml): {removed}")

    config.set_class_names(class_names)
    log_info(f"Collected {len(class_names)} class(es) from LabelMe JSONs")
    for i, name in enumerate(config.CLASS_NAMES):
        log_info(f"  [{i}] {name}")

    # 1. 扫描图片-JSON 对
    pairs = find_image_json_pairs(source_dir)
    if not pairs:
        log_error(f"No image files found in {source_dir}")
        sys.exit(1)
    log_info(f"Found {len(pairs)} image(s)")

    # 2. 转换并收集标注
    valid_pairs = []
    converted_count = 0
    for img_path, json_path in pairs:
        label_lines = []
        if json_path is not None:
            try:
                data = load_labelme_json(json_path)
                img_w = data.get("imageWidth", 0)
                img_h = data.get("imageHeight", 0)
                if img_w <= 0 or img_h <= 0:
                    img_w, img_h = get_image_size(img_path)
                shapes = data.get("shapes", [])
                if shapes:
                    label_lines = converter(shapes, img_w, img_h)
                    converted_count += 1
            except Exception as e:
                log_warn(f"Failed to convert {json_path}: {e}")
                continue

        valid_pairs.append((img_path, label_lines if isinstance(label_lines, str) else ""))

    log_info(f"Converted {converted_count}/{len(pairs)} annotations")

    # 3. 切分数据集
    train_set, val_set, test_set = split_dataset(valid_pairs, train_ratio, val_ratio)
    log_info(f"Split: train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    # 4. 创建目录结构
    ds = Path(dataset_dir)
    train_img_dir = str(ds / "train" / "images")
    train_lbl_dir = str(ds / "train" / "labels")
    val_img_dir = str(ds / "val" / "images")
    val_lbl_dir = str(ds / "val" / "labels")

    for pair, img_d, lbl_d in [(train_set, train_img_dir, train_lbl_dir),
                                (val_set, val_img_dir, val_lbl_dir)]:
        ensure_dir(img_d)
        ensure_dir(lbl_d)
        for img_path, label_content in pair:
            # 图片无条件拷贝；txt 按任务类型转换后的内容写入
            # （无标注的图片也拷贝，并生成空 txt，YOLO 视为背景图）
            copy_image_and_label(img_path, label_content, img_d, lbl_d, task_type)

    test_img_dir = None
    test_lbl_dir = None
    if test_set:
        test_img_dir = str(ds / "test" / "images")
        test_lbl_dir = str(ds / "test" / "labels")
        ensure_dir(test_img_dir)
        ensure_dir(test_lbl_dir)
        for img_path, label_content in test_set:
            copy_image_and_label(img_path, label_content, test_img_dir, test_lbl_dir, task_type)

    # 5. 生成 data.yaml
    yaml_path = generate_data_yaml(dataset_dir, task_type,
                                   str(ds / "train"),
                                   str(ds / "val"),
                                   str(ds / "test") if test_set else "")

    log_info("=" * 60)
    log_info(f"Dataset ready at: {dataset_dir}")
    log_info(f"  - {len(train_set)} train images")
    log_info(f"  - {len(val_set)} val images")
    if test_set:
        log_info(f"  - {len(test_set)} test images")
    log_info(f"  - data.yaml: {yaml_path}")
    log_info("=" * 60)

    # 6. 生成/更新项目信息 info.yaml（数据目录内部，供各 step 与界面读取）
    viz_dir = config.DEFAULT_VISUALIZE_DIR
    if not viz_dir:
        viz_dir = str(Path(dataset_dir).parent / "可视化标注")
    update_info_yaml(
        dataset_dir,
        task_type=task_type,
        source_dir=str(Path(source_dir).resolve()),
        data_yaml=str(Path(yaml_path).resolve()),
        run_out_dir=str(Path(dataset_dir).resolve() / "run_out"),
        visualize_dir=viz_dir,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        labels={
            "count": len(config.CLASS_NAMES),
            "names": list(config.CLASS_NAMES),
            "ignore": sorted(ignore_set),
        },
    )


def main():
    parser = argparse.ArgumentParser(
        description="LabelMe → YOLO 数据集转换工具 (兼容 detect/segment/obb/pose)"
    )
    parser.add_argument("--source_dir", type=str, default=None,
                        help="原始图片和 LabelMe JSON 所在目录（默认取项目 info.yaml / config）")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="输出的 YOLO 数据集目录（用于定位项目 info.yaml，默认取 config）")
    parser.add_argument("--task_type", type=str, default=None,
                        choices=["detect", "segment", "obb", "pose"],
                        help="目标任务类型（默认取项目 info.yaml / config.TASK_TYPE）")
    parser.add_argument("--train_ratio", type=float, default=None,
                        help="训练集比例（默认取项目 info.yaml / config）")
    parser.add_argument("--val_ratio", type=float, default=None,
                        help="验证集比例（默认取项目 info.yaml / config）")
    args = parser.parse_args()

    cfg = get_project_config(args.dataset_dir)
    process_dataset(
        args.source_dir or cfg["source_dir"],
        args.dataset_dir or cfg["dataset_dir"],
        args.task_type or cfg["task_type"],
        args.train_ratio if args.train_ratio is not None else cfg["train_ratio"],
        args.val_ratio if args.val_ratio is not None else cfg["val_ratio"],
    )


if __name__ == "__main__":
    main()
