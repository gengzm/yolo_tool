"""
内置默认配置与程序常量

本文件一般无需修改：
- 日常参数（路径/训练/增强/推理/TensorRT 等）：编辑项目根 run_yolo_config.yaml，
  启动时由 config.user_config 自动加载覆盖（模板见仓库根目录，所有键默认注释）。
- 类别名等运行时数据由步骤脚本自动维护；可视化颜色表见本文件「常量」节。
"""
import os
from pathlib import Path

# ======================== 类别配置 ========================
# 类别名默认为空，由 step1 从 LabelMe JSON 中自动收集
# 也可在此手动预设（会被 JSON 收集结果覆盖）
CLASS_NAMES = []

# 类别 -> 编号映射（随 CLASS_NAMES 同步更新）
CLASS_TO_IDX = {}

# ======================== 训练参数（所有任务统一） ========================
# 学习率(lr0/lrf)、cos_lr、patience 等超参未配置时
# 直接用 ultralytics 内置默认值，无需在此重复定义
EPOCHS = 100
BATCH = 16
IMGSZ = 640

# ======================== 数据增强（训练时，ultralytics 参数） ========================
# 概率类取值 0.0~1.0；degrees 为旋转角度范围（±degrees，度）
# 界面「项目设置 → 训练与推理参数 → 数据增强」可覆盖，改动自动写入 info.yaml
AUGMENT_DEFAULTS = {
    "fliplr": 0.5,      # 左右翻转概率（fliplr）
    "flipud": 0.0,      # 上下翻转概率（flipud）
    "degrees": 0.0,     # 随机旋转角度范围 ±degrees（度）
    "scale": 0.5,       # 随机缩放增益 ±scale（如 0.5 = 50%）
    "translate": 0.1,   # 随机平移比例（宽/高 的 10%）
    "mosaic": 1.0,      # Mosaic 拼接概率（1.0 = 每轮都用 4 图拼接）
    "mixup": 0.0,       # MixUp 图像混合概率
}


# 不同类别的训练参数差异（可选覆盖）
# 如果某个类别需要特殊的学习率、epoch 等，在此配置
CLASS_TRAIN_OVERRIDE = {
    # 示例：person 类别可能需要更多 epoch
    # "person": {"epochs": 150, "lr0": 0.005},
}

# ======================== 任务类型 ========================
# detect | segment | obb | pose
class CLASS_TASK_TYPE:
    OBB = "obb"
    SEGMENT = "segment"
    DETECT = "detect"
    POSE = "pose"

TASK_TYPE = CLASS_TASK_TYPE.OBB

# ======================== 数据根目录 ========================
# 所有数据资产统一放在该根目录下，子目录名称见 DEFAULT_DIR_NAMES：
#   源标注/训练集_{TASK}/run_out/可视化标注/推理结果/权重
# 代码目录只放代码，不产生数据。
# 各子目录完整路径由 path_for_*() 从 data_root + 名称派生，界面只允许改 data_root 与子目录名。
DEFAULT_DATA_ROOT = "/Users/Mac/Code/TemplateMatch"
# 原始图片 + LabelMe JSON 目录（启动默认值，通常由 run_yolo_config.yaml 覆盖）
DEFAULT_SOURCE_DIR = f"{DEFAULT_DATA_ROOT}/2000中筛选出的异常"
# 生成的 YOLO 数据集目录（跟随任务类型）
DEFAULT_DATASET_DIR = f"{DEFAULT_DATA_ROOT}/训练集_{TASK_TYPE.upper()}"

# ======================== 推理参数 ========================
# 推理输入路径（None = 默认取数据集目录的 val/images，即验证集）
INFER_INPUT = None
# 置信度阈值 / IoU 阈值
CONF = 0.25
IOU = 0.45

# ======================== TensorRT 转换（step5） ========================
# trtexec 可执行文件绝对路径，或包含 trtexec 的目录
# 留空 = 自动探测（系统 PATH 中的 trtexec / import tensorrt）
# 注意：TensorRT 仅支持 NVIDIA GPU 环境
TENSORRT_LIB = ""
# 是否默认同时导出 TensorRT (.engine)
EXPORT_TRT = False

# ======================== 切分比例 ========================
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
TEST_RATIO = 0.0   # 0 表示不划分测试集

# ======================== 常量 ========================
# LabelMe JSON 后缀
LABELME_SUFFIX = ".json"
# 支持的图片格式
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ======================== 可视化颜色表（BGR，OpenCV 使用） ========================
# 程序内部常量，一般无需修改
# 20 种类别颜色 (BGR)
CLASS_COLORS_BGR = [
    (0, 0, 255),     # 红
    (0, 255, 0),     # 绿
    (255, 0, 0),     # 蓝
    (0, 255, 255),   # 黄
    (255, 0, 255),   # 品红
    (255, 255, 0),   # 青
    (128, 0, 0),     # 深蓝
    (0, 128, 0),     # 深绿
    (0, 0, 128),     # 深红
    (128, 128, 0),   # 橄榄
    (128, 0, 128),   # 紫
    (0, 128, 128),   # 深青
    (192, 192, 192), # 银
    (64, 64, 64),    # 深灰
    (0, 69, 255),    # 橙
    (255, 191, 0),   # 天蓝
    (147, 20, 255),  # 粉
    (60, 255, 255),  # 浅黄
    (180, 105, 255), # 浅粉
    (50, 205, 154),  # 浅绿
]

# 20 种关键点颜色 (BGR)
POINT_COLORS_BGR = [
    (0, 0, 255),     # 红
    (0, 255, 0),     # 绿
    (255, 0, 0),     # 蓝
    (0, 255, 255),   # 黄
    (255, 0, 255),   # 品红
    (255, 255, 0),   # 青
    (128, 0, 128),   # 紫
    (255, 128, 0),   # 橙蓝
    (64, 128, 255),  # 浅橙
    (0, 128, 255),   # 橙
    (255, 64, 64),   # 浅蓝
    (0, 200, 100),   # 深绿蓝
    (200, 0, 100),   # 紫红
    (100, 200, 0),   # 黄绿
    (0, 100, 200),   # 土黄
    (200, 100, 0),   # 蓝绿
    (100, 0, 200),   # 粉红
    (200, 200, 200), # 白
    (80, 80, 80),    # 灰
    (255, 255, 255), # 纯白
]

# ======================== 项目信息文件 ========================
# 记录标签/权重等运行状态，保存在数据目录内部（{dataset_dir}/info.yaml），供各 step 与外部界面读取
INFO_YAML_NAME = "info.yaml"

# ======================== 预训练模型 ========================
# 本地模型目录（放入 yolo26n-*.pt / yolo26s-*.pt 等文件，训练时自动优先使用）
# 目录不存在或找不到对应文件时，回退 ultralytics 在线下载
MODELS_DIR = os.environ.get(
    "YOLO_MODELS_DIR",
    os.path.join(Path(__file__).resolve().parents[2], "models"),
)

# 模型规格: n(轻量) / s(标准) / m / l / x，切换规格只需改这里
MODEL_SIZE = "n"

# 任务类型 → 模型文件名（自动按 MODEL_SIZE 生成）
def task_model_name(task_type: str = None) -> str:
    """按任务类型 + MODEL_SIZE 生成模型文件名，如 yolo26n-obb.pt"""
    t = (task_type or TASK_TYPE).lower()
    if t == CLASS_TASK_TYPE.SEGMENT:
        return f"yolo26{MODEL_SIZE}-seg.pt"
    if t == CLASS_TASK_TYPE.POSE:
        return f"yolo26{MODEL_SIZE}-pose.pt"
    if t == CLASS_TASK_TYPE.OBB:
        return f"yolo26{MODEL_SIZE}-obb.pt"
    return f"yolo26{MODEL_SIZE}.pt"   # detect

def set_class_names(names: list):
    """运行时设置类别名并同步 CLASS_TO_IDX。
    必须原地修改（list[:] / dict.clear+update），否则其他模块
    `from config import CLASS_NAMES` 持有的引用仍指向旧空对象。
    """
    global CLASS_NAMES, CLASS_TO_IDX
    CLASS_NAMES[:] = list(names)
    CLASS_TO_IDX.clear()
    CLASS_TO_IDX.update({name: i for i, name in enumerate(names)})


# ======================== 数据子目录（data_root 派生） ========================
# 完整路径 = {data_root}/{名称}，界面只允许改 data_root 与子目录名。
DEFAULT_DIR_NAMES = {
    "source": "原始数据",        # 原始图片 + LabelMe JSON
    "dataset": "训练集_{TASK}",  # YOLO 数据集（含 info.yaml），跟随任务类型
    "weights": "权重",           # step5 最终权重（onnx/engine/best.pt）
    "run_out": "run_out",        # step3 训练输出
    "visualize": "可视化标注",   # step2 可视化输出
    "infer": "推理结果",          # step4 推理结果
}


def path_for_source(data_root: str, name: str = None) -> str:
    """原始数据目录（LabelMe JSON）"""
    return str(Path(data_root) / (name or DEFAULT_DIR_NAMES["source"]))


def path_for_dataset(data_root: str, task_type: str = None) -> str:
    """YOLO 数据集目录（含 info.yaml），名称跟随任务类型"""
    t = (task_type or TASK_TYPE).lower()
    return str(Path(data_root) / f"训练集_{t.upper()}")


def path_for_weights(data_root: str, name: str = None) -> str:
    """权重输出目录（step5）"""
    return str(Path(data_root) / (name or DEFAULT_DIR_NAMES["weights"]))


def path_for_run_out(data_root: str, name: str = None) -> str:
    """训练输出目录（step3）"""
    return str(Path(data_root) / (name or DEFAULT_DIR_NAMES["run_out"]))


def path_for_visualize(data_root: str, name: str = None) -> str:
    """可视化输出目录（step2）"""
    return str(Path(data_root) / (name or DEFAULT_DIR_NAMES["visualize"]))


# ======================== 用户配置加载 ========================
# 日常改参数无需编辑本文件：项目根 run_yolo_config.yaml（模板见仓库根目录），
# 或 ~/.config/yolo_tool/config.yaml，启动时自动加载覆盖上面的默认值。
# 查找顺序与优先级说明见 config/user_config.py。
import sys as _sys
from .user_config import load as _load_user_config  # noqa: E402

_load_user_config(_sys.modules[__name__])