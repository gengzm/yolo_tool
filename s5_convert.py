#!/usr/bin/env python
"""
Step 5: 模型转换
- PT → ONNX 转换
- ONNX → TensorRT 转换（当前占位，需安装 TensorRT 后启用）
- 转换结果保存到指定目录

用法:
    python s5_convert.py \\
        --model ./yolo_dataset/run_out/20240101_120000/train/weights/best.pt \\
        --output_dir ./converted_models \\
        --imgsz 640 \\
        --half

# 也可自动找最新模型:
    python s5_convert.py \\
        --data ./yolo_dataset/data.yaml \\
        --output_dir ./converted_models
"""
import argparse
import os
import sys
from pathlib import Path

from ultralytics import YOLO
import torch

import config
from utils import (
    ensure_dir, log_info, log_warn, log_error,
    load_info_yaml, update_info_yaml, resolve_dataset_dir, get_project_config,
)


def pt_to_onnx(model_path: str, output_dir: str, imgsz: int = 640,
               half: bool = False, simplify: bool = True,
               opset: int = 12) -> str:
    """
    PyTorch → ONNX 转换
    """
    ensure_dir(output_dir)
    log_info(f"Converting {model_path} to ONNX...")

    model = YOLO(model_path)

    stem = Path(model_path).stem
    onnx_path = str(Path(output_dir) / f"{stem}.onnx")

    # ultralytics 内置 ONNX 导出
    # 注意: workspace 仅 TensorRT(engine) 格式支持，onnx 传了会报
    # "argument 'workspace' is not supported for format='onnx'"
    success = model.export(
        format="onnx",
        imgsz=imgsz,
        half=half,
        simplify=simplify,
        opset=opset,
    )

    if success and os.path.exists(onnx_path):
        log_info(f"ONNX model saved: {onnx_path}")
        return onnx_path
    else:
        log_warn(f"ONNX export returned success={success}, checking output...")

        # 手动查找导出的 onnx 文件
        weight_dir = Path(model_path).parent
        onnx_candidates = list(weight_dir.glob("*.onnx"))
        if onnx_candidates:
            # 移到输出目录
            import shutil
            dst = str(Path(output_dir) / onnx_candidates[0].name)
            shutil.move(str(onnx_candidates[0]), dst)
            log_info(f"ONNX model moved to: {dst}")
            return dst

        log_error("ONNX export failed")
        return ""


def onnx_to_tensorrt(onnx_path: str, output_dir: str, imgsz: int = 640,
                     half: bool = True) -> str:
    """
    ONNX → TensorRT 转换 (占位实现)

    TODO: 需要安装 TensorRT Python API 和 onnx-tensorrt 后才能实际运行。
    实际转换需要:
    1. import tensorrt as trt
    2. 构建 TRT engine
    3. 保存为 .engine 文件

    当前为占位代码，提示用户手动转换。
    """
    ensure_dir(output_dir)
    log_warn("=" * 60)
    log_warn("TensorRT conversion is a PLACEHOLDER.")
    log_warn("To enable TensorRT conversion, please:")
    log_warn("  1. Install TensorRT: pip install tensorrt")
    log_warn("  2. Install onnx-tensorrt: pip install onnx-tensorrt")
    log_warn("  3. Or use trtexec CLI:")
    log_warn(f"     trtexec --onnx={onnx_path} --saveEngine={output_dir}/model.engine")
    log_warn("=" * 60)

    engine_path = str(Path(output_dir) / f"{Path(onnx_path).stem}.engine")

    # 尝试实际转换（如果 TensorRT 已安装）
    try:
        import tensorrt as trt
        log_info("TensorRT found, attempting conversion...")

        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, logger)

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    log_error(f"  ONNX Parse Error: {parser.get_error(i)}")
                log_error("ONNX parsing failed")
                return ""

        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

        if half:
            config.set_flag(trt.BuilderFlag.FP16)

        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        profile.set_shape(
            input_tensor.name,
            (1, 3, imgsz, imgsz),  # min
            (1, 3, imgsz, imgsz),  # opt
            (1, 3, imgsz, imgsz),  # max
        )
        config.add_optimization_profile(profile)

        log_info("Building TensorRT engine (this may take several minutes)...")
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            log_error("TensorRT engine build failed")
            return ""

        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

        log_info(f"TensorRT engine saved: {engine_path}")
        return engine_path

    except ImportError:
        log_warn("TensorRT not installed, skipping TensorRT conversion")
        return ""
    except Exception as e:
        log_error(f"TensorRT conversion failed: {e}")
        return ""


def convert_model(model_path: str, output_dir: str, imgsz: int = 640,
                  half: bool = False):
    """
    完整转换流程: PT → ONNX → TensorRT
    """
    if not os.path.exists(model_path):
        log_error(f"Model not found: {model_path}")
        sys.exit(1)

    ensure_dir(output_dir)
    log_info(f"Converting model: {model_path}")
    log_info(f"Output dir: {output_dir}")

    # 1. PT → ONNX
    onnx_path = pt_to_onnx(model_path, output_dir, imgsz, half)
    if not onnx_path:
        log_error("PT→ONNX conversion failed, stopping")
        return

    # 2. ONNX → TensorRT
    engine_path = onnx_to_tensorrt(onnx_path, output_dir, imgsz, half)

    log_info("=" * 60)
    log_info("Conversion summary:")
    log_info(f"  ONNX:    {onnx_path}")
    log_info(f"  TensorRT: {engine_path if engine_path else 'Not generated (placeholder)'}")
    log_info("=" * 60)

    return onnx_path, engine_path


def find_latest_model_from_data(data_path: str) -> str:
    """从 data.yaml 查找最新训练模型"""
    data_dir = Path(data_path).parent.resolve()
    run_out_dir = data_dir / "run_out"
    if not run_out_dir.exists():
        log_error(f"run_out not found: {run_out_dir}")
        sys.exit(1)

    best_models = list(run_out_dir.rglob("**/weights/best.pt"))
    if not best_models:
        log_error(f"No best.pt found in {run_out_dir}")
        sys.exit(1)

    best_models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(best_models[0])


def resolve_model_path(model_arg: str, data_arg: str, dataset_dir: str = None):
    """
    解析模型路径：显式 --model > --data 自动查找 > info.yaml 记录
    返回 (model_path or None, dataset_dir or None)
    """
    if model_arg:
        return model_arg, None
    if data_arg:
        return find_latest_model_from_data(data_arg), str(Path(data_arg).parent.resolve())

    # 未显式指定：从项目 info.yaml（数据目录同级）读取权重/数据集信息
    ds_dir = resolve_dataset_dir(dataset_dir)
    info = load_info_yaml(ds_dir) if Path(ds_dir).exists() else {}
    if info:
        weights = info.get("weights") or {}
        if weights.get("best") and Path(weights["best"]).exists():
            return weights["best"], ds_dir
        data_yaml = info.get("data_yaml")
        if data_yaml and Path(data_yaml).exists():
            return find_latest_model_from_data(data_yaml), ds_dir
    return None, None


def main():
    parser = argparse.ArgumentParser(description="模型转换工具: PT → ONNX → TensorRT")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="YOLO 数据集目录（用于定位项目 info.yaml，默认取 config）")
    parser.add_argument("--model", type=str, default=None,
                        help="PT 模型路径")
    parser.add_argument("--data", type=str, default=None,
                        help="data.yaml 路径（自动查找最新模型）")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="输出目录（默认取项目 info.yaml / config）")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="输入图像尺寸（默认取项目 info.yaml / config）")
    parser.add_argument("--half", action="store_true",
                        help="使用 FP16 精度")
    args = parser.parse_args()

    # 项目配置解析: 命令行参数 > info.yaml 记录 > config 默认值
    cfg = get_project_config(args.dataset_dir)
    imgsz = args.imgsz if args.imgsz is not None else cfg["imgsz"]

    # 确定模型路径（显式参数 > info.yaml 记录 > 自动查找）
    model_path, dataset_dir = resolve_model_path(args.model, args.data, args.dataset_dir)
    if not model_path:
        log_error("Model not found. Please specify --model/--data, or run step3 first.")
        sys.exit(1)
    log_info(f"Using model: {model_path}")

    output_dir = args.output_dir or cfg.get("convert_dir") or config.DEFAULT_CONVERT_DIR
    onnx_path, engine_path = convert_model(model_path, output_dir, imgsz, args.half)

    # 更新项目信息 info.yaml：记录转换后的权重与转换目录（合并保留已有字段）
    if dataset_dir and onnx_path:
        info = load_info_yaml(dataset_dir)
        weights = dict(info.get("weights", {}))
        weights["onnx"] = onnx_path
        if engine_path:
            weights["engine"] = engine_path
        update_info_yaml(dataset_dir, weights=weights,
                         convert_dir=str(Path(output_dir).resolve()))


if __name__ == "__main__":
    main()
