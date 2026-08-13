# YOLO 标注转换与训练工具链

将 LabelMe 标注（JSON）一键转换为 YOLO 数据集，并完成 **可视化 → 训练 → 推理 → ONNX 转换 → 权重部署** 的全流程。支持 `detect` / `segment` / `obb` / `pose` 四种任务。

## 特性

- **LabelMe → YOLO 转换**：自动映射 shape_type 到 detect/segment/obb/pose 格式
- **项目档案（info.yaml）**：每个数据集目录自带一份档案，记录该项目的参数与运行结果，切换项目即切换目录，参数自动导入
- **参数优先级**：命令行参数 > 项目 info.yaml > config.py 默认值
- **零参数可运行**：每个 step 均可裸跑（`python s1_prepare_data.py`），默认值自动从 config / info.yaml 解析
- **一键串联**：`run_all.sh` 依次执行全部 6 个步骤

## 目录结构

```
yolo_tool/
├── config.py            # 全局默认配置（唯一需要改参数的地方）
├── utils.py             # 公共工具：info.yaml 读写、项目配置解析、转换函数
├── requirements.txt     # 依赖清单
├── run_all.sh           # 一键串联 s0~s5
├── s0_collect_labels.py # 扫描 LabelMe JSON，统计类别
├── s1_prepare_data.py   # 生成 YOLO 数据集 + 切分 + data.yaml
├── s2_visualize.py      # 标注可视化
├── s3_train.py          # YOLO 训练
├── s4_inference.py      # 推理（可视化 + JSON + 误差分析）
└── s5_convert.py        # 转 ONNX（产物即最终部署权重，含 best.pt）
```

## 环境安装

```bash
# 创建环境（示例）
conda create -n yolo python=3.11 -y
conda activate yolo

# 安装依赖（清华源）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 版本适配说明：仅 `ultralytics` 锁定主版本（8.4+ 才支持 YOLO26）。本机 torch 2.2.x 只支持 numpy 1.x，因此 `numpy < 2.0`、`opencv-python < 4.11`、`onnxruntime < 2.0` 三条约束不可放宽；升级 torch 时需同步调整。

## 快速开始

### 方式一：一键全流程（推荐）

```bash
bash run_all.sh                # 使用 config.py 中的默认任务类型与路径
bash run_all.sh obb            # 或显式指定任务类型
```

> 环境变量可临时覆盖参数，如 `DATASET_DIR=... EPOCHS=100 bash run_all.sh`

### 方式二：分步执行

```bash
# 1. 查看源目录的标签类别
python s0_collect_labels.py --source_dir /path/to/raw

# 2. 生成 YOLO 数据集
python s1_prepare_data.py --source_dir /path/to/raw --dataset_dir /path/to/yolo_dataset --task_type obb

# 3. 可视化检查标注
python s2_visualize.py --dataset_dir /path/to/yolo_dataset --split all

# 4. 训练
python s3_train.py --dataset_dir /path/to/yolo_dataset --epochs 200 --batch 16

# 5. 推理（默认对验证集）
python s4_inference.py --dataset_dir /path/to/yolo_dataset

# 6. 转换（产物即最终部署权重，输出到 {DATA_ROOT}/权重）
python s5_convert.py --dataset_dir /path/to/yolo_dataset
```

所有步骤都可省略参数裸跑；多项目场景只需传 `--dataset_dir` 切换。

## 配置体系

### 1. config.py（全局默认值，随代码仓库）

```python
DEFAULT_DATA_ROOT = "/path/to/data_root"       # 数据根目录：所有数据资产统一放这里
TASK_TYPE = "obb"                              # 默认任务类型
DEFAULT_SOURCE_DIR = f"{DATA_ROOT}/原始数据"    # 原始图片 + LabelMe JSON
DEFAULT_DATASET_DIR = f"{DATA_ROOT}/训练集_OBB" # 生成的 YOLO 数据集
DEFAULT_VISUALIZE_DIR = None                   # None = 自动 = {DATA_ROOT}/可视化标注
DEFAULT_RUN_OUT_DIR = None                     # None = 自动 = {DATA_ROOT}/run_out
DEFAULT_WEIGHTS_DIR = None                     # None = 自动 = {DATA_ROOT}/权重（最终部署权重）
EPOCHS = 200
BATCH = 16
IMGSZ = 640
INFER_INPUT = None                             # None = 推理默认用验证集
CONF = 0.25                                    # 推理置信度
IOU = 0.45                                     # 推理 IoU
CLASS_NAMES = [...]                            # 类别列表（与源标注一致）
```

所有数据产物（原始数据、数据集、可视化、最终权重）统一归于 `DEFAULT_DATA_ROOT` 下，代码目录只放代码。

### 2. info.yaml（项目档案，随数据集目录）

每个数据集目录下自动生成 `info.yaml`，记录该项目运行结果与参数：

```yaml
dataset_dir: /path/to/yolo_dataset   # 定位目录
task_type: obb
source_dir: /path/to/raw
data_yaml: /path/to/yolo_dataset/data.yaml
train_ratio: 0.8
val_ratio: 0.2
epochs: 200
batch: 16
imgsz: 640
weights: /path/to/data_root/run_out/20260813_022239/train/weights/best.pt
weights_dir: {DATA_ROOT}/权重     # 最终部署权重（yolo_<task>_detector.onnx / .engine / .pt）
```

运行各 step 时若省略参数，会自动从该项目档案恢复；界面/脚本切换项目 = 切换 `--dataset_dir`。

### 3. 优先级

```
命令行参数  >  项目 info.yaml  >  config.py 默认值
```

## 各 Step 详解

| Step | 脚本 | 功能 | 关键参数 |
|------|------|------|----------|
| 0 | `s0_collect_labels.py` | 扫描 LabelMe JSON，统计各类别次数 | `--source_dir` |
| 1 | `s1_prepare_data.py` | 转 YOLO 格式、切分 train/val/test、生成 data.yaml | `--source_dir` `--task_type` `--train_ratio` `--val_ratio` |
| 2 | `s2_visualize.py` | 标注可视化到"可视化标注"目录 | `--split`（train/val/all） |
| 3 | `s3_train.py` | YOLO 训练，输出到 `{DATA_ROOT}/run_out/{时间戳}/` | `--epochs` `--batch` `--imgsz` `--device` |
| 4 | `s4_inference.py` | 推理 + 可视化 + JSON + 误差分析（bbox/点误差，Mean/Median/Std 图表） | `--input` `--conf` `--iou` `--model` `--json_dir` |
| 5 | `s5_convert.py` | 转 ONNX + 附 best.pt（产物即最终权重）；TensorRT 需 NVIDIA GPU | `--output_dir` `--imgsz` `--half` |

### LabelMe shape_type 映射（s1）

| shape_type | detect | segment | obb | pose |
|-----------|--------|---------|-----|------|
| rectangle | bbox | 多边形近似 | bbox | - |
| polygon   | 外接框 | 保留顶点 | 4 点旋转框 | - |
| circle    | 外接框 | 多边形近似 | - | - |
| point     | 极小框 | 忽略 | - | 关键点 |
| line/linestrip | 外接框 | 转分割 | - | - |

## 推理输出结构

```
{DATA_ROOT}/推理结果/
├── 推理可视化/    # 叠加检测结果的图片
├── 推理json/      # 结构化结果，便于后续处理
└── 误差分析/      # 预测 vs 真值误差统计与图表
    ├── error_stats.json      # Mean / Median / Std / Min / Max 统计
    ├── error_analysis.png    # 误差分布直方图（bbox 中心/宽高/IoU + 点误差）
    └── error_stats_bar.png   # Mean / Median / Std 汇总条形图
```

## 常见问题

**Q1: 训练输出在哪里？**
每次训练在 `{DATA_ROOT}/run_out/{时间戳}/train/weights/` 下生成 `best.pt` / `last.pt`，推理与转换优先取 info.yaml 记录的模型，否则自动取最新且训练完成的 `best.pt`（跳过中断训练）。

**Q2: 能转 TensorRT engine 吗？**
`s5_convert.py` 内置了 TRT 转换代码，但 TensorRT 仅支持 NVIDIA GPU。Mac/无独显环境请用 ONNX + onnxruntime；需要 engine 时把 `{DATA_ROOT}/权重/yolo_<task>_detector.onnx` 拷贝到 NVIDIA 机器上用 `trtexec --onnx=yolo_obb_detector.onnx --saveEngine=yolo_obb_detector.engine` 转换，产物再放回 `{DATA_ROOT}/权重`。

**Q3: 推理默认输入是什么？**
`INFER_INPUT = None` 时对验证集（`{dataset_dir}/val/images`）推理；传 `--input` 可指定任意目录或单张图片。

**Q4: 报错 "argument 'workspace' is not supported for format='onnx'"？**
旧代码在 onnx 导出时误传了 TRT 专有参数 `workspace`，已移除。若更新 ultralytics 后仍有类似校验错误，检查 `s5_convert.py` 的 `model.export()` 参数是否含 TRT 专有项。

**Q5: 报错 "requirements not found, attempting AutoUpdate"？**
`onnxruntime` / `onnxslim` 未安装。执行 `pip install onnxruntime "onnxruntime<2.0" onnxslim -i https://pypi.tuna.tsinghua.edu.cn/simple`。
