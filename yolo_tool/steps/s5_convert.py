#!/usr/bin/env python
"""
Step 5: 模型转换（产物即最终部署权重）
- PT → ONNX 转换，输出 {DATA_ROOT}/权重/yolo_<task_type>_detector.onnx
- 附一份 best.pt（yolo_<task_type>_detector.pt）
- ONNX → TensorRT 转换（当前占位，需安装 TensorRT 后启用）

用法:
    yolo-tool s5 \\
        --model ./yolo_dataset/run_out/20240101_120000/train/weights/best.pt \\
        --output_dir ./权重 \\
        --imgsz 640 \\
        --half

# 也可自动找最新模型:
    yolo-tool s5 \\
        --data ./yolo_dataset/data.yaml
"""
import argparse
import os
import sys
from pathlib import Path

from ultralytics import YOLO
import torch

from ..config import config
from ..config.utils import (
    ensure_dir, log_info, log_warn, log_error,
    load_info_yaml, update_info_yaml, resolve_dataset_dir, get_project_config,
)


def pt_to_onnx(model_path: str, output_dir: str, task_type: str, imgsz: int = 640,
               half: bool = False, simplify: bool = True,
               opset: int = 12, output_name: str = "") -> str:
    """
    PyTorch → ONNX 转换
    产物重命名为 <output_name>.onnx（默认 yolo_<task_type>_detector.onnx）
    并统一放到 output_dir
    """
    if not output_name:
        output_name = f"yolo_{task_type}_detector"
    ensure_dir(output_dir)
    log_info(f"Converting {model_path} to ONNX...")

    model = YOLO(model_path)

    # ultralytics 内置 ONNX 导出（默认导出到 .pt 同目录，文件名为 best.onnx）
    # 注意: workspace 仅 TensorRT(engine) 格式支持，onnx 传了会报
    # "argument 'workspace' is not supported for format='onnx'"
    success = model.export(
        format="onnx",
        imgsz=imgsz,
        half=half,
        simplify=simplify,
        opset=opset,
    )

    # 定位实际导出的 onnx 文件（优先用 export 返回值，否则回退扫描 .pt 同目录）
    src_path = ""
    if isinstance(success, (str, Path)) and Path(success).exists():
        src_path = str(success)
    if not src_path:
        candidates = sorted(Path(model_path).parent.glob("*.onnx"))
        if candidates:
            src_path = str(candidates[0])

    if not src_path:
        log_error("ONNX export failed: no .onnx produced")
        return ""

    # 重命名并移动到最终权重目录
    import shutil
    dest_path = str(Path(output_dir) / f"{output_name}.onnx")
    if os.path.abspath(src_path) != os.path.abspath(dest_path):
        shutil.move(src_path, dest_path)
    log_info(f"ONNX model saved: {dest_path}")
    return dest_path


def _find_trtexec() -> str:
    """
    定位 trtexec 可执行文件，优先级：
    1. config.TENSORRT_LIB（trtexec 绝对路径，或包含 trtexec 的目录）
    2. 系统 PATH 中的 trtexec
    返回 "" 表示未找到。
    """
    lib = getattr(config, "TENSORRT_LIB", "") or ""
    candidates = []
    if lib:
        p = Path(lib)
        if p.is_dir():
            candidates += [p / "trtexec", p / "trtexec.exe"]
        elif "trtexec" in p.name.lower():
            candidates.append(p)
    import shutil
    for name in ("trtexec", "trtexec.exe"):
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    for c in candidates:
        if c.exists():
            return str(c)
    return ""


def onnx_to_tensorrt(onnx_path: str, output_dir: str, task_type: str, imgsz: int = 640,
                     half: bool = False, output_name: str = "") -> str:
    """
    ONNX → TensorRT 转换
    优先使用 trtexec（来源: config.TENSORRT_LIB / 系统 PATH），
    否则回退到 TensorRT Python API。失败时返回 ""（跳过，不阻断主流程）。
    """
    ensure_dir(output_dir)
    if not output_name:
        output_name = f"yolo_{task_type}_detector"
    engine_path = str(Path(output_dir) / f"{output_name}.engine")

    # 方式一: trtexec CLI（推荐，配置 config.TENSORRT_LIB 指向 trtexec 或其所在目录）
    trtexec = _find_trtexec()
    if trtexec:
        import subprocess
        cmd = [trtexec, f"--onnx={onnx_path}", f"--saveEngine={engine_path}"]
        if half:
            cmd.append("--fp16")
        log_info("Using trtexec: " + " ".join(cmd))
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            if r.returncode == 0 and Path(engine_path).exists():
                log_info(f"TensorRT engine saved: {engine_path}")
                return engine_path
            log_error("trtexec failed:\n"
                      + (r.stdout[-3000:] or "") + "\n" + (r.stderr[-3000:] or ""))
        except Exception as e:
            log_error(f"trtexec error: {e}")
        return ""

    # 方式二: TensorRT Python API
    try:
        import tensorrt as trt
        log_info("TensorRT Python API found, attempting conversion...")

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

        bconfig = builder.create_builder_config()
        bconfig.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)  # 4GB

        if half:
            bconfig.set_flag(trt.BuilderFlag.FP16)

        profile = builder.create_optimization_profile()
        input_tensor = network.get_input(0)
        profile.set_shape(
            input_tensor.name,
            (1, 3, imgsz, imgsz),  # min
            (1, 3, imgsz, imgsz),  # opt
            (1, 3, imgsz, imgsz),  # max
        )
        bconfig.add_optimization_profile(profile)

        log_info("Building TensorRT engine (this may take several minutes)...")
        serialized_engine = builder.build_serialized_network(network, bconfig)
        if serialized_engine is None:
            log_error("TensorRT engine build failed")
            return ""

        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

        log_info(f"TensorRT engine saved: {engine_path}")
        return engine_path

    except ImportError:
        log_warn("TensorRT 不可用：未安装 tensorrt 且未配置 trtexec"
                 "（可在 config.py 中设置 TENSORRT_LIB 指向 trtexec）")
        return ""
    except Exception as e:
        log_error(f"TensorRT conversion failed: {e}")
        return ""


def copy_best_pt(model_path: str, output_dir: str, task_type: str,
                 output_name: str = "") -> str:
    """复制 best.pt 到最终权重目录，命名为 <output_name>.pt"""
    import shutil
    if not output_name:
        output_name = f"yolo_{task_type}_detector"
    dst_path = str(Path(output_dir) / f"{output_name}.pt")
    shutil.copy2(model_path, dst_path)
    log_info(f"best.pt saved: {dst_path}")
    return dst_path


def convert_model(model_path: str, output_dir: str, task_type: str, imgsz: int = 640,
                  half: bool = False, trt: bool = False, output_name: str = ""):
    """
    完整转换流程: PT → ONNX（可选 → TensorRT），产物统一放到 output_dir，
    命名为 <output_name>.*（默认 yolo_<task_type>_detector.*），并附一份 .pt
    """
    if not os.path.exists(model_path):
        log_error(f"Model not found: {model_path}")
        sys.exit(1)

    ensure_dir(output_dir)
    log_info(f"Converting model: {model_path}")
    log_info(f"Output dir: {output_dir}")

    # 1. PT → ONNX
    onnx_path = pt_to_onnx(model_path, output_dir, task_type, imgsz, half,
                           output_name=output_name)
    if not onnx_path:
        log_error("PT→ONNX conversion failed, stopping")
        return

    # 2. ONNX → TensorRT（可选，--trt 时执行）
    engine_path = ""
    if trt:
        engine_path = onnx_to_tensorrt(onnx_path, output_dir, task_type, imgsz,
                                       half, output_name=output_name)

    # 3. 附一份 best.pt（部署侧可能直接加载 pt）
    pt_path = copy_best_pt(model_path, output_dir, task_type, output_name)

    log_info("=" * 60)
    log_info("Conversion summary:")
    log_info(f"  ONNX:     {onnx_path}")
    log_info(f"  best.pt:  {pt_path}")
    log_info(f"  TensorRT: {engine_path if engine_path else 'Not generated'}")
    log_info("=" * 60)

    return onnx_path, engine_path


def find_latest_model_from_data(data_path: str) -> str:
    """从 data.yaml 查找最新训练模型"""
    data_dir = Path(data_path).parent.resolve()
    cfg = get_project_config(str(data_dir))
    run_out_dir = cfg.get("run_out_dir") or str(data_dir / "run_out")
    if not Path(run_out_dir).exists():
        log_error(f"run_out not found: {run_out_dir}")
        sys.exit(1)

    best_models = list(Path(run_out_dir).rglob("**/weights/best.pt"))
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

    # 未显式指定：从项目 info.yaml（数据目录内部）读取权重/数据集信息
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
                        help="使用 FP16 精度（默认 FP32）")
    parser.add_argument("--trt", action="store_true",
                        help="同时导出 TensorRT engine（需要 NVIDIA GPU）")
    parser.add_argument("--output_name", type=str, default=None,
                        help="输出文件名称（不含扩展名，默认 yolo_<task>_detector）")
    args = parser.parse_args()

    # 项目配置解析: 命令行参数 > info.yaml 记录 > config 默认值
    cfg = get_project_config(args.dataset_dir)
    imgsz = args.imgsz if args.imgsz is not None else cfg["imgsz"]
    task_type = cfg.get("task_type") or config.TASK_TYPE

    # 确定模型路径（显式参数 > info.yaml 记录 > 自动查找）
    model_path, dataset_dir = resolve_model_path(args.model, args.data, args.dataset_dir)
    if not model_path:
        log_error("Model not found. Please specify --model/--data, or run step3 first.")
        sys.exit(1)
    log_info(f"Using model: {model_path}")

    output_dir = (args.output_dir or cfg.get("weights_dir")
                  or config.path_for_weights(cfg.get("data_root") or ""))
    output_name = (args.output_name or cfg.get("output_name")
                   or f"yolo_{task_type}_detector")
    onnx_path, engine_path = convert_model(model_path, output_dir, task_type, imgsz,
                                           args.half, args.trt, output_name)

    # 更新项目信息 info.yaml：记录最终权重（清理废弃的 convert_dir/deploy_dir 旧键）
    if dataset_dir and onnx_path:
        info = load_info_yaml(dataset_dir)
        weights = dict(info.get("weights", {}))
        weights["onnx"] = onnx_path
        weights["pt"] = str(Path(output_dir) / f"{output_name}.pt")
        if engine_path:
            weights["engine"] = engine_path
        update_info_yaml(dataset_dir, remove_keys=["convert_dir", "deploy_dir"],
                         weights=weights,
                         weights_dir=str(Path(output_dir).resolve()))


if __name__ == "__main__":
    main()
