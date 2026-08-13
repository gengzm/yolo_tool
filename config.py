"""
共享配置文件 —— 类别、颜色、路径等全局参数
"""
import os

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
# 所有数据资产统一放在该根目录下：
#   {DATA_ROOT}/原始数据/        源标注（source_dir）
#   {DATA_ROOT}/训练集_{TASK}/   生成的 YOLO 数据集（含 info.yaml）
#   {DATA_ROOT}/run_out/         step3 训练输出（时间戳子目录）
#   {DATA_ROOT}/可视化标注/      step2 可视化输出
#   {DATA_ROOT}/推理结果/        step4 推理可视化 / 推理json / 误差分析
#   {DATA_ROOT}/权重           step5 转换产物（yolo_<task>_detector.onnx / .engine / .pt，即最终部署权重）
# 代码目录只放代码，不产生数据。
DEFAULT_DATA_ROOT = "/Users/Mac/Code/TemplateMatch"

# ======================== 默认路径 ========================
# 原始图片 + LabelMe JSON 目录
DEFAULT_SOURCE_DIR = f"{DEFAULT_DATA_ROOT}/2000中筛选出的异常"
# 生成的 YOLO 数据集目录
DEFAULT_DATASET_DIR = f"{DEFAULT_DATA_ROOT}/训练集_{TASK_TYPE.upper()}"
# 可视化输出根目录（None = 自动 = {DATA_ROOT}/可视化标注）
DEFAULT_VISUALIZE_DIR = None
# 训练输出根目录（None = 自动 = {DATA_ROOT}/run_out）
DEFAULT_RUN_OUT_DIR = None
# 最终权重目录（None = 自动 = {DATA_ROOT}/权重）
# step5 转换的 onnx/engine 与 best.pt 统一放这里，即最终部署产物（不再单独拷贝）
DEFAULT_WEIGHTS_DIR = None

# ======================== 推理参数 ========================
# 推理输入路径（None = 默认取数据集目录的 val/images，即验证集）
INFER_INPUT = None
# 置信度阈值 / IoU 阈值
CONF = 0.25
IOU = 0.45

# ======================== 切分比例 ========================
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2
TEST_RATIO = 0.0   # 0 表示不划分测试集

# ======================== 常量 ========================
# LabelMe JSON 后缀
LABELME_SUFFIX = ".json"
# 支持的图片格式
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ======================== 项目信息文件 ========================
# 记录标签/权重等运行状态，保存在数据目录内部（{dataset_dir}/info.yaml），供各 step 与外部界面读取
INFO_YAML_NAME = "info.yaml"

# ======================== 可视化颜色配置 ========================
# 20 种类别颜色 (BGR 格式，OpenCV 使用)
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

# ======================== 预训练模型 ========================
# 本地模型目录（放入 yolo26n-*.pt / yolo26s-*.pt 等文件，训练时自动优先使用）
# 目录不存在或找不到对应文件时，回退 ultralytics 在线下载
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

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
    """
    运行时设置类别名列表，并同步更新 CLASS_TO_IDX
    注意：必须原地修改（list[:] / dict.clear+update），
    因为其他模块用 `from config import CLASS_NAMES` 持有的是同一对象的引用，
    重新赋值会导致其他模块仍看到旧的空列表/dict。
    """
    global CLASS_NAMES, CLASS_TO_IDX
    CLASS_NAMES[:] = list(names)
    CLASS_TO_IDX.clear()
    CLASS_TO_IDX.update({name: i for i, name in enumerate(names)})


def get_class_names() -> list:
    """获取当前类别名列表"""
    return CLASS_NAMES