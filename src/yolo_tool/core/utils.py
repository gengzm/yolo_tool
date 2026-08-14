"""
通用工具函数
"""
import os
import sys
import json
import random
import shutil
import yaml
from pathlib import Path
from typing import Optional, Tuple, List

import cv2
import numpy as np

from . import config
from .config import (
    CLASS_NAMES, CLASS_TO_IDX, IMAGE_EXTENSIONS, LABELME_SUFFIX,
    TRAIN_RATIO, VAL_RATIO, CLASS_COLORS_BGR, POINT_COLORS_BGR,
)


def ensure_dir(path: str) -> Path:
    """确保目录存在"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def resolve_model_path(model_ref: str) -> str:
    """
    解析模型引用为可加载路径：
    1) 绝对/相对路径 → 原样返回
    2) 纯文件名 → 优先在 config.MODELS_DIR 下找本地文件（如 yolo26n-obb.pt）
    3) 找不到 → 原样返回，交给 ultralytics 在线下载
    """
    if not model_ref:
        return model_ref
    if os.path.sep in model_ref or (os.path.altsep and os.path.altsep in model_ref):
        return model_ref
    local = Path(config.MODELS_DIR) / model_ref
    if local.exists():
        log_info(f"Using local model: {local}")
        return str(local)
    log_warn(f"Model not in {config.MODELS_DIR}, fallback to ultralytics download: {model_ref}")
    return model_ref


def find_image_json_pairs(source_dir: str) -> List[Tuple[str, Optional[str]]]:
    """
    扫描目录，返回 (图片路径, LabelMe JSON路径) 对
    JSON 不是必须的（可能有无标注图片）
    """
    source = Path(source_dir)
    pairs = []
    for f in sorted(source.iterdir()):
        if f.suffix.lower() in IMAGE_EXTENSIONS:
            json_path = source / (f.stem + LABELME_SUFFIX)
            if json_path.exists():
                pairs.append((str(f), str(json_path)))
            else:
                pairs.append((str(f), None))
    return pairs


def load_labelme_json(json_path: str) -> dict:
    """加载 LabelMe JSON 文件"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_class_names_from_json(source_dir: str) -> List[str]:
    """
    扫描目录中所有 LabelMe JSON 文件，收集所有 shape 的 label 名
    返回排序后的去重类别名列表
    """
    source = Path(source_dir)
    labels_set = set()
    json_files = sorted(source.glob(f"*{LABELME_SUFFIX}"))
    for json_path in json_files:
        try:
            data = load_labelme_json(str(json_path))
            for shape in data.get("shapes", []):
                label = shape.get("label", "").strip()
                if label:
                    labels_set.add(label)
        except Exception as e:
            log_warn(f"Failed to read {json_path.name}: {e}")

    names = sorted(labels_set)
    log_info(f"Collected {len(names)} class(es) from LabelMe JSONs: {names}")
    return names


def load_class_names_from_yaml(yaml_path: str) -> List[str]:
    """从 data.yaml 读取类别名"""
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("names", [])


def get_class_id(label: str) -> Optional[int]:
    """获取类别编号，未知类别返回 None"""
    # 如果 CLASS_TO_IDX 为空但 CLASS_NAMES 也是空的，说明还未初始化
    # 这种情况下未知类别动态添加（仅在转换阶段使用）
    if not CLASS_TO_IDX:
        log_warn(f"CLASS_TO_IDX is empty, treating '{label}' as unknown. "
                 f"Please run class collection first.")
        return None
    return CLASS_TO_IDX.get(label)


def get_class_color(class_idx: int) -> Tuple[int, int, int]:
    """获取类别颜色 (BGR)"""
    return CLASS_COLORS_BGR[class_idx % len(CLASS_COLORS_BGR)]


def get_point_color(idx: int) -> Tuple[int, int, int]:
    """获取关键点颜色 (BGR)"""
    return POINT_COLORS_BGR[idx % len(POINT_COLORS_BGR)]


def normalize_points(points: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """
    将绝对坐标归一化到 [0, 1]
    points: shape (N, 2)
    """
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"Invalid image size: {img_w}x{img_h}")
    normalized = points.copy().astype(np.float64)
    normalized[:, 0] /= img_w
    normalized[:, 1] /= img_h
    return np.clip(normalized, 0.0, 1.0)


def denormalize_points(normalized: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """将归一化坐标转为绝对像素坐标"""
    denorm = normalized.copy()
    denorm[:, 0] *= img_w
    denorm[:, 1] *= img_h
    return denorm


def points_to_yolo_bbox(points: np.ndarray, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """
    将多边形点转为 YOLO bbox 格式 (cx, cy, w, h) 归一化
    """
    norm = normalize_points(points, img_w, img_h)
    x_min, y_min = norm.min(axis=0)
    x_max, y_max = norm.max(axis=0)
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    w = x_max - x_min
    h = y_max - y_min
    return cx, cy, w, h


def split_dataset(pairs: List, train_r: float = TRAIN_RATIO, val_r: float = VAL_RATIO,
                  seed: int = 42) -> Tuple[List, List, List]:
    """
    将数据对按比例分割为训练/验证/测试集
    test_r = 1 - train_r - val_r
    """
    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_r)
    val_end = int(total * (train_r + val_r))
    return shuffled[:train_end], shuffled[train_end:val_end], shuffled[val_end:]


def copy_image_and_label(img_path: str, label_content: str, dst_img_dir: str,
                         dst_label_dir: str, task_type: str = "detect"):
    """
    将图片拷贝到指定目录，并写入对应的 YOLO label 文件
    """
    ensure_dir(dst_img_dir)
    ensure_dir(dst_label_dir)

    stem = Path(img_path).stem
    dst_img = str(Path(dst_img_dir) / Path(img_path).name)
    shutil.copy2(img_path, dst_img)

    dst_label = str(Path(dst_label_dir) / f"{stem}.txt")
    with open(dst_label, "w", encoding="utf-8") as f:
        f.write(label_content)


def log_info(msg: str):
    """打印带前缀的日志"""
    print(f"[INFO] {msg}")


def log_warn(msg: str):
    """打印警告"""
    print(f"[WARN] {msg}", file=sys.stderr)


def log_error(msg: str):
    """打印错误"""
    print(f"[ERROR] {msg}", file=sys.stderr)


def get_image_size(img_path: str) -> Tuple[int, int]:
    """快速获取图片宽高（不完整解码）"""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {img_path}")
    return img.shape[1], img.shape[0]


# ======================== 项目信息文件 (info.yaml) ========================

def get_info_yaml_path(dataset_dir: str) -> Path:
    """info.yaml 保存在数据目录内部（{dataset_dir}/info.yaml），作为该项目的档案"""
    return Path(dataset_dir) / config.INFO_YAML_NAME


def load_info_yaml(dataset_dir: str) -> dict:
    """读取 info.yaml，文件不存在或解析失败时返回空 dict"""
    p = get_info_yaml_path(dataset_dir)
    if not p.exists():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        log_warn(f"Failed to load {p}: {e}")
        return {}


def save_info_yaml(dataset_dir: str, info: dict) -> Path:
    """保存 info.yaml 到数据目录内部（{dataset_dir}/info.yaml），自动补充 updated_at 时间戳"""
    from datetime import datetime
    p = get_info_yaml_path(dataset_dir)
    data = dict(info)
    data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    log_info(f"Info yaml updated: {p}")
    return p


def update_info_yaml(dataset_dir: str, remove_keys: list = None, **fields) -> Path:
    """更新 info.yaml（与已有内容合并，保留各 step 写入的字段，可移除废弃键）"""
    info = load_info_yaml(dataset_dir)
    for k in (remove_keys or []):
        info.pop(k, None)
    info["dataset_dir"] = str(Path(dataset_dir).resolve())  # 始终记录定位目录
    info.update(fields)
    return save_info_yaml(dataset_dir, info)


def resolve_dataset_dir(dataset_dir: str = None) -> str:
    """定位数据集目录：显式参数优先，否则用 config 默认"""
    if dataset_dir:
        return dataset_dir
    return config.DEFAULT_DATASET_DIR


def get_project_config(dataset_dir: str = None) -> dict:
    """
    项目配置解析（供各 step 与界面统一使用，直接用 dataset_dir 定位 info.yaml）
    优先级: 命令行参数(调用方处理) > info.yaml 记录 > config 默认值
    返回键: task_type/source_dir/dataset_dir/data_yaml/run_out_dir/
            visualize_dir/weights_dir/train_ratio/val_ratio/epochs/batch/imgsz
    若 info.yaml 另有 weights/weights_dir 等记录，也会一并带出。
    """
    ds_dir = resolve_dataset_dir(dataset_dir)
    info = load_info_yaml(ds_dir)
    if not dataset_dir:                     # 未显式指定时，以 info.yaml 记录为准
        ds_dir = info.get("dataset_dir") or ds_dir
    ds_dir = str(Path(ds_dir).resolve())
    # 数据根目录：config 指定优先，否则回退数据集目录的父目录
    data_root = str(Path(config.DEFAULT_DATA_ROOT).resolve()) if getattr(config, "DEFAULT_DATA_ROOT", None) else str(Path(ds_dir).parent)
    defaults = {
        "data_root": data_root,
        "theme": "light",       # 界面主题：light 浅色 / dark 深色
        "task_type": config.TASK_TYPE,
        "source_dir_name": config.DEFAULT_DIR_NAMES["source"],
        "dataset_dir": ds_dir,
        "data_yaml": str(Path(ds_dir) / "data.yaml"),
        "run_out_dir_name": config.DEFAULT_DIR_NAMES["run_out"],
        "visualize_dir_name": config.DEFAULT_DIR_NAMES["visualize"],
        "infer_input": config.INFER_INPUT or str(Path(ds_dir) / "val" / "images"),
        "conf": config.CONF,
        "iou": config.IOU,
        "weights_dir_name": config.DEFAULT_DIR_NAMES["weights"],
        # 注：source_dir/run_out_dir/visualize_dir/weights_dir 完整路径
        # 统一由 _unify_dir_cfg 按 data_root + 名称 派生，默认值不在此预置
        "train_ratio": config.TRAIN_RATIO,
        "val_ratio": config.VAL_RATIO,
        "epochs": config.EPOCHS,
        "batch": config.BATCH,
        "imgsz": config.IMGSZ,
        "model_size": getattr(config, "MODEL_SIZE", "n"),
        "export_trt": getattr(config, "EXPORT_TRT", False),
        "trt_lib": getattr(config, "TENSORRT_LIB", ""),
        # s2 可视化样式参数
        "circle_diameter_first": 6,
        "circle_diameter_other": 4,
        "line_width": 2,
    }
    # 数据增强默认值（训练时传给 ultralytics）
    defaults.update(config.AUGMENT_DEFAULTS)
    cfg = dict(defaults)
    cfg.update(info)            # info.yaml 记录覆盖默认值
    cfg["dataset_dir"] = ds_dir  # 但 dataset_dir 始终以定位目录为准
    _unify_dir_cfg(cfg, data_root, set(info.keys()))
    return cfg


def _dir_path_for(cfg: dict, path_key: str, data_root: str, name: str) -> str:
    """按目录类型用 data_root + 名称派生完整路径"""
    if path_key == "source_dir":
        return config.path_for_source(data_root, name)
    if path_key == "weights_dir":
        return config.path_for_weights(data_root, name)
    if path_key == "run_out_dir":
        return config.path_for_run_out(data_root, name)
    return config.path_for_visualize(data_root, name)


def _unify_dir_cfg(cfg: dict, data_root: str, recorded: set) -> None:
    """目录统一管理：
    - info.yaml 显式记录了「名称」 → 以 data_root + 名称 派生完整路径
    - 只记录了完整路径 → 保留路径，名称回填为路径最后一段
    - 两者都缺失 → 用默认名称派生"""
    _DEFAULTS = {
        "source_dir_name": "原始数据",
        "weights_dir_name": "权重",
        "run_out_dir_name": "run_out",
        "visualize_dir_name": "可视化标注",
    }
    for name_key, path_key in (
        ("source_dir_name", "source_dir"),
        ("weights_dir_name", "weights_dir"),
        ("run_out_dir_name", "run_out_dir"),
        ("visualize_dir_name", "visualize_dir"),
    ):
        name = str(cfg.get(name_key) or "").strip().strip("/\\")
        if "/" in name or "\\" in name:
            name = name.replace("/", "_").replace("\\", "_")
        if name_key in recorded and name:
            cfg[path_key] = _dir_path_for(cfg, path_key, data_root, name)
            cfg[name_key] = name
        elif cfg.get(path_key):
            cfg[name_key] = str(Path(str(cfg[path_key])).name)
        else:
            name = name or _DEFAULTS[name_key]
            cfg[path_key] = _dir_path_for(cfg, path_key, data_root, name)
            cfg[name_key] = name
