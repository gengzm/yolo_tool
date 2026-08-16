#!/bin/bash
# 一键串联全部步骤（与 GUI 共用同一套「项目配置」）
# 用法: bash run_all.sh [task_type]   (detect / segment / obb / pose，默认读项目 info.yaml)
# 前置: 已执行 pip install -e .（提供 yolo-tool 包与命令）
#
# 配置原则（一个数据项目 = 一份配置）：
#   各 step 统一从「项目 info.yaml」读取训练/推理等参数（GUI 设置即写在这里），
#   run_all.sh 只负责定位项目（SOURCE_DIR / DATASET_DIR）并串联步骤，
#   不再与 GUI 抢配置。参数优先级：
#     显式参数（环境变量 / 位置参数） > info.yaml > user_config.yaml > config.py 内置默认
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ======================== 工具函数 ========================
# 从配置模块读取「定位默认值」（自动加载项目根 user_config.yaml）
cfg_value() {
    (cd "$SCRIPT_DIR" && python -c "from yolo_tool.config import config; print($1)")
}

# ======================== 定位参数 ========================
# 只有定位参数（去哪个项目）默认取配置模块，
# 其余训练/推理参数一律交给 step 从项目 info.yaml 读取（与 GUI 设置一致）。
SOURCE_DIR="${SOURCE_DIR:-$(cfg_value 'config.DEFAULT_SOURCE_DIR')}"
DATASET_DIR="${DATASET_DIR:-$(cfg_value 'config.DEFAULT_DATASET_DIR')}"

# ======================== 可选的显式覆盖 ========================
# 未设置时对应参数从 info.yaml / user_config.yaml 读取；
# 设置后优先级最高（适合一次性实验），用法示例：
#   EPOCHS=50 BATCH=8 CONF=0.5 bash run_all.sh
TASK_TYPE="${1:-${TASK_TYPE:-}}"   # 位置参数 或 环境变量
ARGS_TASK=();  [ -n "$TASK_TYPE" ] && ARGS_TASK=(--task_type "$TASK_TYPE")
ARGS_RATIO=()
if [ -n "${TRAIN_RATIO:-}" ] && [ -n "${VAL_RATIO:-}" ]; then
    ARGS_RATIO+=(--train_ratio "$TRAIN_RATIO" --val_ratio "$VAL_RATIO")
fi
ARGS_TRAIN=()
[ -n "${EPOCHS:-}" ] && ARGS_TRAIN+=(--epochs "$EPOCHS")
[ -n "${BATCH:-}" ]  && ARGS_TRAIN+=(--batch "$BATCH")
[ -n "${IMGSZ:-}" ]  && ARGS_TRAIN+=(--imgsz "$IMGSZ")
[ -n "${DEVICE:-}" ] && ARGS_TRAIN+=(--device "$DEVICE")
ARGS_INFER=()
[ -n "${CONF:-}" ]        && ARGS_INFER+=(--conf "$CONF")
[ -n "${IOU:-}" ]         && ARGS_INFER+=(--iou "$IOU")
[ -n "${INFER_INPUT:-}" ] && ARGS_INFER+=(--input "$INFER_INPUT")
ARGS_CONVERT=()
[ -n "${IMGSZ:-}" ]       && ARGS_CONVERT+=(--imgsz "$IMGSZ")
[ -n "${WEIGHTS_DIR:-}" ] && ARGS_CONVERT+=(--output_dir "$WEIGHTS_DIR")

DATA_YAML="${DATASET_DIR}/data.yaml"

# ======================== Step 0: 依赖 ========================
echo "[Step 0] 检查/安装依赖（已安装的自动跳过）..."
python -m pip install -q -r "${SCRIPT_DIR}/requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "[Step 0] 依赖就绪"

# ======================== Step 0.5: 收集标签信息 ========================
echo "[Step 0.5] 收集标签信息..."
python -m yolo_tool.steps.s0_collect_labels \
    --dataset_dir "$DATASET_DIR" \
    --source_dir "$SOURCE_DIR"

# ======================== Step 1: 数据准备 ========================
python -m yolo_tool.steps.s1_prepare_data \
    --source_dir "$SOURCE_DIR" \
    --dataset_dir "$DATASET_DIR" \
    "${ARGS_TASK[@]}" "${ARGS_RATIO[@]}"

# ======================== Step 2: 可视化 ========================
python -m yolo_tool.steps.s2_visualize \
    --dataset_dir "$DATASET_DIR" \
    --split all \
    "${ARGS_TASK[@]}"

# ======================== Step 3: 训练 ========================
python -m yolo_tool.steps.s3_train \
    --dataset_dir "$DATASET_DIR" \
    --data "$DATA_YAML" \
    "${ARGS_TASK[@]}" "${ARGS_TRAIN[@]}"

# ======================== Step 4: 推理 ========================
# 默认对验证集推理；显式设置 INFER_INPUT 时使用自定义路径
python -m yolo_tool.steps.s4_inference \
    --dataset_dir "$DATASET_DIR" \
    --data "$DATA_YAML" \
    "${ARGS_TASK[@]}" "${ARGS_INFER[@]}"

# ======================== Step 5: 转换（产物即最终部署权重） ========================
python -m yolo_tool.steps.s5_convert \
    --dataset_dir "$DATASET_DIR" \
    --data "$DATA_YAML" \
    "${ARGS_CONVERT[@]}"

echo "全部流程完成"
