#!/bin/bash
# 一键串联全部步骤
# 用法: bash run_all.sh [task_type]   (detect / segment / obb / pose，默认 detect)
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ======================== 工具函数 ========================
# 从 config.py 读取任意配置值（传入 python 表达式）
cfg_value() {
    (cd "$SCRIPT_DIR" && python -c "import config; print($1)")
}

# ======================== 参数 ========================
# 任务类型默认取自 config.py，可用命令行参数临时覆盖
TASK_TYPE="${1:-$(cfg_value 'config.TASK_TYPE')}"
# 路径默认取自 config.py（改 config 即可），也可用环境变量覆盖
SOURCE_DIR="${SOURCE_DIR:-$(cfg_value 'config.DEFAULT_SOURCE_DIR')}"
DATASET_DIR="${DATASET_DIR:-$(cfg_value 'config.DEFAULT_DATASET_DIR')}"

# ---- 训练/推理参数（默认全部取自 config.py，环境变量可临时覆盖） ----
EPOCHS="${EPOCHS:-$(cfg_value 'config.EPOCHS')}"
BATCH="${BATCH:-$(cfg_value 'config.BATCH')}"
IMGSZ="${IMGSZ:-$(cfg_value 'config.IMGSZ')}"
INFER_INPUT="${INFER_INPUT:-$(cfg_value 'config.INFER_INPUT')}"   # None=默认验证集（交给 step4 自行解析）
[ "$INFER_INPUT" = "None" ] && INFER_INPUT=""
CONF="${CONF:-$(cfg_value 'config.CONF')}"
IOU="${IOU:-$(cfg_value 'config.IOU')}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$(cfg_value 'config.DEFAULT_WEIGHTS_DIR')}"   # None=交给 step5 解析({DATA_ROOT}/权重，产物即最终部署权重)
[ "$WEIGHTS_DIR" = "None" ] && WEIGHTS_DIR=""
DEVICE="${DEVICE:-}"

DATA_YAML="${DATASET_DIR}/data.yaml"

# ======================== Step 0: 依赖 ========================
echo "[Step 0] 检查/安装依赖（已安装的自动跳过）..."
python -m pip install -q -r "${SCRIPT_DIR}/requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple
echo "[Step 0] 依赖就绪"

# ======================== Step 0.5: 收集标签信息 ========================
echo "[Step 0.5] 收集标签信息..."
python "${SCRIPT_DIR}/s0_collect_labels.py" \
    --dataset_dir "$DATASET_DIR" \
    --source_dir "$SOURCE_DIR"

# ======================== Step 1: 数据准备 ========================
python "${SCRIPT_DIR}/s1_prepare_data.py" \
    --source_dir "$SOURCE_DIR" \
    --dataset_dir "$DATASET_DIR" \
    --task_type "$TASK_TYPE"

# ======================== Step 2: 可视化 ========================
python "${SCRIPT_DIR}/s2_visualize.py" \
    --dataset_dir "$DATASET_DIR" \
    --split all \
    --task_type "$TASK_TYPE"

# ======================== Step 3: 训练 ========================
DEVICE_ARG=""
[ -n "$DEVICE" ] && DEVICE_ARG="--device $DEVICE"
python "${SCRIPT_DIR}/s3_train.py" \
    --dataset_dir "$DATASET_DIR" \
    --data "$DATA_YAML" \
    --task_type "$TASK_TYPE" \
    --epochs "$EPOCHS" \
    --batch "$BATCH" \
    --imgsz "$IMGSZ" \
    $DEVICE_ARG

# ======================== Step 4: 推理 ========================
# 默认（INFER_INPUT 为空）对验证集推理；设置了 --input 则用自定义路径
INFER_ARG=""
[ -n "$INFER_INPUT" ] && INFER_ARG="--input $INFER_INPUT"
python "${SCRIPT_DIR}/s4_inference.py" \
    --dataset_dir "$DATASET_DIR" \
    --data "$DATA_YAML" \
    --task_type "$TASK_TYPE" \
    --conf "$CONF" \
    --iou "$IOU" \
    $INFER_ARG

# ======================== Step 5: 转换（产物即最终部署权重） ========================
CONVERT_ARG=""
[ -n "$WEIGHTS_DIR" ] && CONVERT_ARG="--output_dir $WEIGHTS_DIR"
python "${SCRIPT_DIR}/s5_convert.py" \
    --dataset_dir "$DATASET_DIR" \
    --data "$DATA_YAML" \
    --imgsz "$IMGSZ" \
    $CONVERT_ARG

echo "全部流程完成"
