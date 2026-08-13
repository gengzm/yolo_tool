#!/usr/bin/env python
"""
Step 6: 拷贝转换后的权重到指定目录
- 将 ONNX / TensorRT 等转换结果拷贝到目标目录
- 支持按类型过滤（只拷贝 ONNX 或只拷贝 TensorRT）

用法:
    python s6_copy_weights.py \\
        --source ./converted_models \\
        --target /path/to/deploy/weights \\
        --types onnx engine

    # 从 data.yaml 自动找转换后的模型
    python s6_copy_weights.py \\
        --data ./yolo_dataset/data.yaml \\
        --target /path/to/deploy/weights \\
        --types onnx
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

import config
from utils import ensure_dir, log_info, log_warn, log_error, update_info_yaml, get_project_config


def find_converted_models(source_dir: str, types: list = None) -> dict:
    """
    扫描转换后的模型文件
    返回: {"onnx": ["/path/to/model.onnx", ...], "engine": [...]}
    """
    if types is None:
        types = ["onnx", "engine"]

    ext_map = {
        "onnx": ".onnx",
        "engine": ".engine",
        "pt": ".pt",
        "trt": ".engine",
    }

    found = {t: [] for t in types}
    source = Path(source_dir)

    if not source.exists():
        log_warn(f"Source directory not found: {source_dir}")
        return found

    for t in types:
        ext = ext_map.get(t, f".{t}")
        for f in sorted(source.rglob(f"*{ext}")):
            found[t].append(str(f))
        # 也检查直接子文件
        for f in sorted(source.glob(f"*{ext}")):
            p = str(f)
            if p not in found[t]:
                found[t].append(p)

    return found


def copy_weights(source_dir: str, target_dir: str, types: list = None):
    """
    将指定类型的权重拷贝到目标目录
    """
    ensure_dir(target_dir)
    log_info(f"Copying weights from {source_dir} → {target_dir}")

    found = find_converted_models(source_dir, types)

    total = sum(len(v) for v in found.values())
    if total == 0:
        log_warn(f"No weight files found in {source_dir}")
        return

    copied = 0
    for model_type, files in found.items():
        for f in files:
            dst = os.path.join(target_dir, os.path.basename(f))
            shutil.copy2(f, dst)
            log_info(f"  [{model_type}] {os.path.basename(f)}")
            copied += 1

    log_info("=" * 60)
    log_info(f"Copied {copied} weight file(s) to: {target_dir}")
    log_info("=" * 60)


def find_source_from_data(data_path: str) -> str:
    """
    从 data.yaml 目录找转换后的模型（run_out 或兄弟目录的 converted_models）
    """
    data_dir = Path(data_path).parent.resolve()

    # 检查常见位置
    candidates = [
        data_dir / "converted_models",
        data_dir / "run_out" / "converted",
        data_dir.parent / "converted_models",
    ]

    for c in candidates:
        if c.exists():
            return str(c)

    # 如果 run_out 存在，检查其子目录
    run_out = data_dir / "run_out"
    if run_out.exists():
        converted_dirs = list(run_out.rglob("converted_models"))
        for cd in converted_dirs:
            if cd.exists():
                return str(cd)

    return str(data_dir)


def main():
    parser = argparse.ArgumentParser(description="拷贝转换后的模型权重到目标目录")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="YOLO 数据集目录（用于定位项目 info.yaml，默认取 config）")
    parser.add_argument("--source", type=str, default=None,
                        help="源目录（包含转换后的模型文件）")
    parser.add_argument("--data", type=str, default=None,
                        help="data.yaml 路径（自动推断 source 目录）")
    parser.add_argument("--target", type=str, default=None,
                        help="目标目录（默认取项目 info.yaml 的 deploy_dir / config）")
    parser.add_argument("--types", type=str, nargs="+",
                        default=["onnx", "engine"],
                        help="要拷贝的文件类型 (onnx engine pt)")
    args = parser.parse_args()

    # 项目配置解析: 命令行参数 > info.yaml 记录 > config 默认值
    cfg = get_project_config(args.dataset_dir)
    dataset_dir = args.dataset_dir or cfg["dataset_dir"]

    # 确定源目录: 显式 --source > --data 推断 > info.yaml convert_dir
    if args.source:
        source_dir = args.source
    elif args.data:
        source_dir = find_source_from_data(args.data)
        log_info(f"Auto-detected source: {source_dir}")
    elif cfg.get("convert_dir") and Path(cfg["convert_dir"]).exists():
        source_dir = cfg["convert_dir"]
        log_info(f"Using convert_dir from info.yaml: {source_dir}")
    else:
        log_error("Please specify --source/--data, or run step5 first.")
        sys.exit(1)

    # 目标目录: 显式 --target > info.yaml deploy_dir > config 默认
    target_dir = args.target or cfg.get("deploy_dir") or config.DEFAULT_DEPLOY_DIR

    copy_weights(source_dir, target_dir, args.types)

    # 更新项目信息 info.yaml：记录部署目录（供界面/后续步骤读取）
    update_info_yaml(dataset_dir, deploy_dir=str(Path(target_dir).resolve()))


if __name__ == "__main__":
    main()
