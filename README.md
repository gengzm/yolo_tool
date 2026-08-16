# YOLO 工具箱（YOLO Tool）

基于 Ultralytics YOLO（YOLO26）的目标检测/分割/OBB/姿态估计 数据处理与训练工具链。

提供两条使用路径：
- **图形界面（推荐）**：PySide6 工作台，配置自动保存、可视化操作全流程。
- **命令行**：`yolo-tool s0 .. s5` 单步执行，`run_all.sh` 一键串联。

---

## 1. 安装

### 1.1 创建 conda 环境（推荐）

为避免污染系统 Python 或其他项目环境，建议为本项目创建独立 conda 环境：

```bash
conda create -n yolo python=3.10 -y
conda activate yolo
```

> 说明：
> - 环境名 `yolo` 可自定义，后续所有命令需在该环境激活状态下执行。
> - Python 建议 3.10~3.11（与 torch 2.2~2.4 官方支持完全对齐，最稳）；3.12 亦可，pip 会自动解析到 torch 2.3+。
> - 未安装 conda 可先装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)（轻量）或 Anaconda。

### 1.2 安装依赖

```bash
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 以可编辑方式安装到当前 conda 环境，依赖由 `pyproject.toml` 统一管理，安装后获得 `yolo-tool` 命令。
> `run_all.sh` 内部仍会按 `requirements.txt` 检查/补齐依赖。
>
> 版本约束（见 `requirements.txt` 内注释）：
> `numpy<2.0`、`opencv-python<4.11`、`onnxruntime<2.0`、`torch<2.5` —— torch 2.2.x 只支持 numpy 1.x，四条约束必须同时保留，升版需同步放宽。

### 1.3 验证安装

```bash
yolo-tool --help        # 或 python -m yolo_tool --help，能打印命令说明即安装成功
```

---

## 2. 启动

> 以下命令均需在已激活的 conda 环境中执行（`conda activate yolo`，见 1.1）。

### 图形界面

```bash
yolo-tool          # 等价于 python -m yolo_tool
```

工作台左侧为步骤页：`设置` → `准备数据(s1)` → `可视化(s2)` → `训练(s3)` → `推理(s4)` → `转换导出(s5)`，右上为全局运行日志。

- **配置自动保存**：所有可编辑字段改动即保存（防抖 300ms），无需手动保存；「数据根目录」失焦后自动同步派生目录。
- **数据根目录**：唯一需要手填的根路径，其余目录（数据集、权重、输出等）由它按名称自动派生，名称框与路径框联动显示（只读灰框）。
- **训练/验证比例**：训练集、验证集比例自动互补，总和恒为 1；验证集必须非空（训练评估与 `best.pt` 生成依赖），下限 0.05。
- **s5 TensorRT 导出**：未检测到 NVIDIA GPU（CUDA）或 TensorRT 时，相关两项自动灰显。

### 命令行单步

```bash
yolo-tool s0 .. s5 [参数]      # 等价于 python -m yolo_tool.steps.sX ...
yolo-tool s3 --epochs 50 --batch 8
```

### 一键流程

```bash
bash run_all.sh [task_type]     # detect / segment / obb / pose
```

等价于依次执行 s0 → s5。训练/推理等参数与 GUI **共用同一份项目配置**（见第 5 节），脚本只负责定位项目并串联步骤；位置参数或环境变量可临时覆盖（见脚本内注释）。

---

## 3. 目录结构

```
yolo_tool/                    项目根
├── yolo_tool/                包（安装后为 yolo-tool 命令）
│   ├── __init__.py           版本信息
│   ├── __main__.py           python -m yolo_tool 入口
│   ├── cli.py                yolo-tool 命令：GUI / s0~s5 子命令分发
│   ├── app/                  GUI 工作台
│   │   ├── main.py           主窗口、步骤页装配、保存流程
│   │   ├── tabs.py           各步骤页（设置 / s1~s5）
│   │   ├── widgets.py        基础控件（表单、路径框、预览等）
│   │   ├── step_runner.py    后台执行步骤（python -m 子进程）
│   │   └── theme.py          全局 QSS 主题
│   ├── config/               共享配置与公共函数
│   │   ├── config.py         全局配置（目录、训练、转换参数）+ 用户配置加载
│   │   └── utils.py          路径派生、配置读写、数据集拆分等
│   └── steps/                s0~s5 步骤脚本（python -m 运行）
│       ├── s0_collect_labels.py   收集源目录标签信息（类别/图片数）
│       ├── s1_prepare_data.py     数据准备：拆分训练/验证集 + data.yaml
│       ├── s2_visualize.py        可视化标注（结果图 + 可播放动画）
│       ├── s3_train.py            训练（YOLO26，输出 best.pt / last.pt）
│       ├── s4_inference.py        推理（默认验证集，可指定 --input）
│       └── s5_convert.py          导出 ONNX / TensorRT(.engine)
├── pyproject.toml           打包配置与依赖清单
├── run_all.sh               一键串联全部步骤
├── requirements.txt         run_all.sh 依赖安装清单（与 pyproject 一致）
├── user_config.yaml         用户配置模板（全部键默认注释，见 5.2）
├── README.md
└── models/                  本地预训练权重（yolo26n/s-{obb,seg,pose}.pt，
                             训练时优先使用，缺则在线下载）
```

---

## 4. 各步骤说明

| 步骤 | 模块 | 作用 |
| ---- | ---- | ---- |
| s0 | `s0_collect_labels` | 扫描源目录，收集类别与样本统计，供 s1 标签配置 |
| s1 | `s1_prepare_data` | 按训练/验证比例拆分数据，生成 `data.yaml`；标签 ID 在 GUI 中可配置（参与训练 0..n-1 独占，跳过留空） |
| s2 | `s2_visualize` | 可视化标注与拆分结果（`--split train/val/all`） |
| s3 | `s3_train` | 训练 YOLO26，输出 `best.pt` / `last.pt`（`--epochs --batch --imgsz --device`） |
| s4 | `s4_inference` | 推理（`--conf --iou`，`--input` 缺省用验证集） |
| s5 | `s5_convert` | 导出 ONNX / TensorRT（`--imgsz --output_name`，`--output_dir` 缺省为最终部署权重目录） |

### s5 转换产物

- 默认输出文件名：`yolo_<任务>_detector`（detect/segment/obb/pose 前缀），GUI 中可在「输出文件名称」处自定义。
- 产物命名：`<output_name>.onnx`（ONNX）、`<output_name>.engine`（TensorRT）、`<output_name>.pt`（best 权重拷贝）。

---

## 5. 配置

### 5.1 配置结构（一个数据项目 = 一份配置）

配置分「项目级」与「全局级」两层，所有入口（GUI / `run_all.sh` / 单步命令）统一走同一条优先级链：

```
显式参数（命令行 / 环境变量）  >  项目 info.yaml  >  user_config.yaml  >  config.py 内置默认
```

| 层级 | 位置 | 作用 |
| ---- | ---- | ---- |
| **项目级（随数据走）** | `{数据集目录}/info.yaml` | **单一真相源**。GUI 设置、各 step 的关键参数都读写这里；`run_all.sh` 与单步命令未显式指定的参数均从此读取。切换项目 = 换数据目录，配置自动跟随 |
| **全局默认（随代码走）** | 项目根 `user_config.yaml` | 兜底。仅当项目 info.yaml 未记录对应项时生效（见 5.2） |
| **内置兜底** | `yolo_tool/config/config.py` | 代码默认值，一般无需修改（见 5.3） |
| **临时覆盖** | 命令行参数 / 环境变量 | 优先级最高，适合一次性实验：`EPOCHS=50 BATCH=8 bash run_all.sh` |

使用示例：

```bash
# 1) GUI：设置页填好后自动写入项目 info.yaml，跑哪步用哪步
yolo-tool
# 2) 命令行/脚本：不用 GUI 也能跑，参数自动从 info.yaml 读取
bash run_all.sh                      # 完全沿用项目配置
EPOCHS=200 bash run_all.sh           # 仅临时改训练轮数
yolo-tool s3 --epochs 200            # 单步临时覆盖
# 3) 新项目：指定不同的数据目录即可，配置自动跟随
SOURCE_DIR=/data/项目B/原始 DATASET_DIR=/data/项目B/数据集 bash run_all.sh
```

### 5.2 全局默认配置（user_config.yaml）

项目根目录的 **`user_config.yaml`**（模板已随仓库提供，全部键默认注释），启动时自动加载并覆盖内置默认值。**无需改任何代码、无需重装**，GUI 与 `run_all.sh` 同时生效。

配置文件查找顺序（只加载第一个找到的）：

| 位置 | 说明 |
| ---- | ---- |
| `YOLO_CONFIG` 环境变量指定文件 | 可放任意路径（如数据目录内的项目配置），例：`YOLO_CONFIG=/data/proj/config.yaml bash run_all.sh` |
| 项目根 `user_config.yaml` | 本仓库推荐方式 |
| `~/.config/yolo_tool/config.yaml` | 个人全局配置，对所有项目生效 |

### 5.3 内置默认值（yolo_tool/config/config.py）

以下为未配置用户文件时的默认值，可直接在 `user_config.yaml` 中覆盖：

| 配置 | 说明 |
| ---- | ---- |
| `TASK_TYPE` | 任务类型：detect / segment / obb / pose |
| `DEFAULT_DATA_ROOT` | 数据根目录，其余目录自动派生（改它时源/数据集目录自动重算） |
| `DEFAULT_SOURCE_DIR` | 原始数据目录（s1 输入） |
| `DEFAULT_DATASET_DIR` | 数据集目录（含 data.yaml / info.yaml） |
| `MODELS_DIR` | 本地预训练权重目录（默认项目根 `models/`，可用环境变量 `YOLO_MODELS_DIR` 覆盖） |
| `MODEL_SIZE` | 模型规格：n / s / m / l / x |
| `TRAIN_RATIO` / `VAL_RATIO` | 训练/验证比例（总和 = 1，验证集 ≥ 0.05） |
| `AUGMENT_DEFAULTS` | 数据增强参数（fliplr / flipud / degrees / scale / translate / mosaic / mixup） |
| `EPOCHS` / `BATCH` / `IMGSZ` | 训练参数 |
| `CONF` / `IOU` | 推理阈值 |
| `EXPORT_TRT` / `TENSORRT_LIB` | TensorRT 导出开关与 trtexec 路径 |
