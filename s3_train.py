#!/usr/bin/env python
"""
Step 3: YOLO 训练
- 使用 ultralytics YOLO 进行训练
- 支持不同任务类型（detect / segment / obb / pose）
- 不同类别可配置不同的训练参数
- 训练结果输出到数据集同目录的 run_out

用法:
    python s3_train.py \\
        --data ./yolo_dataset/data.yaml \\
        --task_type detect \\
        --model yolo26n.pt \\
        --epochs 100 \\
        --batch 16 \\
        --imgsz 640
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml
from ultralytics import YOLO

import config
from config import CLASS_TRAIN_OVERRIDE, CLASS_NAMES, TASK_MODELS, EPOCHS, BATCH, IMGSZ
from utils import (
    ensure_dir, log_info, log_warn, log_error,
    load_info_yaml, update_info_yaml, get_project_config,
)


def load_data_config(data_path: str) -> dict:
    """读取 data.yaml"""
    with open(data_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_task_model(task_type: str) -> str:
    """获取任务对应的 YOLO 模型名称（取自 config.TASK_MODELS）"""
    return TASK_MODELS.get(task_type, "yolo26n.pt")


def get_run_out_dir(data_path: str) -> str:
    """
    确定 run_out 输出目录：
    数据集同目录下的 run_out / 时间戳子目录
    """
    data_dir = Path(data_path).parent.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = data_dir / "run_out" / timestamp
    ensure_dir(str(run_dir))
    return str(run_dir)


def merge_class_params(task_type: str, data_config: dict) -> dict:
    """
    合并全局训练参数和类别级别的覆盖参数
    全局参数统一取自 config 顶层变量，类别覆盖来自 CLASS_TRAIN_OVERRIDE
    """
    params = {
        "model": TASK_MODELS.get(task_type, "yolo26n.pt"),
        "epochs": EPOCHS,
        "batch": BATCH,
        "imgsz": IMGSZ,
        # 学习率等超参不设置，直接用 ultralytics 默认值
    }

    # 读取 data.yaml 中的类别名
    names = data_config.get("names", CLASS_NAMES)

    # 检查是否有类别覆盖
    for cls_name, override in CLASS_TRAIN_OVERRIDE.items():
        if cls_name in names:
            log_info(f"Applying class override for '{cls_name}': {override}")
            params.update(override)

    return params


def train(data_path: str, task_type: str, model_path: str = None,
          epochs: int = None, batch: int = None, imgsz: int = None,
          device: str = ""):
    """
    执行 YOLO 训练
    """
    # 读取 data.yaml
    if not os.path.exists(data_path):
        log_error(f"data.yaml not found: {data_path}")
        sys.exit(1)

    data_config = load_data_config(data_path)
    log_info(f"Data config loaded: {len(data_config.get('names', []))} classes")

    # 从 data.yaml 加载类别名到全局配置
    names = data_config.get("names", [])
    if names:
        config.set_class_names(names)

    # 合并参数
    params = merge_class_params(task_type, data_config)

    # 命令行参数覆盖
    if model_path is None:
        model_path = params.get("model", get_task_model(task_type))
    if epochs is not None:
        params["epochs"] = epochs
    if batch is not None:
        params["batch"] = batch
    if imgsz is not None:
        params["imgsz"] = imgsz

    log_info(f"Training config: task={task_type}, model={model_path}")
    log_info(f"  epochs={params.get('epochs')}, batch={params.get('batch')}, "
             f"imgsz={params.get('imgsz')}")

    # 加载模型
    model = YOLO(model_path)

    # 确定输出目录
    run_dir = get_run_out_dir(data_path)
    log_info(f"Run output dir: {run_dir}")

    # 构建训练参数（学习率等超参不显式传入，用 ultralytics 默认值）
    train_args = {
        "data": data_path,
        "epochs": params.get("epochs", EPOCHS),
        "batch": params.get("batch", BATCH),
        "imgsz": params.get("imgsz", IMGSZ),
        "device": device or params.get("device", ""),
        "project": run_dir,
        "name": "train",
        "exist_ok": True,
        "verbose": True,
        "save": True,
        "save_period": -1,  # 只保存最佳
    }

    # 对 segmentation / pose / obb 任务类型
    if task_type in ("segment", "pose", "obb"):
        train_args["task"] = task_type

    log_info("Starting training...")
    results = model.train(**train_args)

    log_info("=" * 60)
    log_info("Training completed!")
    log_info(f"Best model saved at: {results.save_dir}")
    log_info(f"Run output: {run_dir}")
    log_info("=" * 60)

    # 更新项目信息 info.yaml：记录最新权重（合并保留 onnx/engine 等已有字段）
    dataset_dir = Path(data_path).parent.resolve()
    info = load_info_yaml(str(dataset_dir))
    weights = dict(info.get("weights", {}))
    weights_dir = Path(results.save_dir) / "weights"
    for fname, key in [("best.pt", "best"), ("last.pt", "last")]:
        w = weights_dir / fname
        if w.exists():
            weights[key] = str(w.resolve())
    update_info_yaml(str(dataset_dir), weights=weights)

    return results


def main():
    parser = argparse.ArgumentParser(description="YOLO 训练工具")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="YOLO 数据集目录（用于定位项目 info.yaml，默认取 config）")
    parser.add_argument("--data", type=str, default=None,
                        help="data.yaml 路径（默认取项目 info.yaml / 数据集目录）")
    parser.add_argument("--task_type", type=str, default=None,
                        choices=["detect", "segment", "obb", "pose"],
                        help="任务类型（默认取项目 info.yaml / config）")
    parser.add_argument("--model", type=str, default=None,
                        help="预训练模型路径（默认自动选择）")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数（默认取项目 info.yaml / config）")
    parser.add_argument("--batch", type=int, default=None,
                        help="batch size（默认取项目 info.yaml / config）")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="输入图像尺寸（默认取项目 info.yaml / config）")
    parser.add_argument("--device", type=str, default="",
                        help="设备 (0, 1, cpu, mps)")
    args = parser.parse_args()

    # 项目配置解析: 命令行参数 > info.yaml 记录 > config 默认值
    cfg = get_project_config(args.dataset_dir)
    dataset_dir = args.dataset_dir or cfg["dataset_dir"]
    data_path = args.data or cfg["data_yaml"]
    if not data_path or not Path(data_path).exists():
        log_error("data.yaml not found. Please specify --data/--dataset_dir or run step1 first.")
        sys.exit(1)
    task_type = args.task_type or cfg["task_type"]
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    batch = args.batch if args.batch is not None else cfg["batch"]
    imgsz = args.imgsz if args.imgsz is not None else cfg["imgsz"]

    train(data_path, task_type, args.model, epochs, batch, imgsz, args.device)

    # 回写训练参数到项目 info.yaml（界面切换到该项目时可回填表单）
    update_info_yaml(dataset_dir, epochs=epochs, batch=batch, imgsz=imgsz)


if __name__ == "__main__":
    main()
