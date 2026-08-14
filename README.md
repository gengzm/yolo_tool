# YOLO 工具箱（YOLO Tool）

基于 Ultralytics YOLO（YOLO26）的目标检测/分割/OBB/姿态估计 数据处理与训练工具链。

提供两条使用路径：
- **图形界面（推荐）**：PySide6 工作台，配置自动保存、可视化操作全流程。
- **命令行**：`yolo-tool s0 .. s5` 单步执行，`run_all.sh` 一键串联。

---

## 1. 安装

```bash
pip install -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 以可编辑方式安装到当前 Python 环境，依赖由 `pyproject.toml` 统一管理，安装后获得 `yolo-tool` 命令。
> `run_all.sh` 内部仍会按 `requirements.txt` 检查/补齐依赖。
>
> 版本约束（见 `requirements.txt` 内注释）：
> `numpy<2.0`、`opencv-python<4.11`、`onnxruntime<2.0`、`torch<2.5` —— 本机 torch 2.2.x 只支持 numpy 1.x，四条约束必须同时保留，升版需同步放宽。

---

## 2. 启动

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
bash run_all.sh [task_type]     # detect / segment / obb / pose，默认取 config.py
```

等价于依次执行 s0 → s5，所有参数默认取自 `config.py`，可用环境变量临时覆盖（见脚本内注释）。

---

## 3. 目录结构

```
yolo_tool/                    项目根
├── src/
│   └── yolo_tool/            包（安装后为 yolo-tool 命令）
│       ├── __init__.py       版本信息
│       ├── __main__.py       python -m yolo_tool 入口
│       ├── cli.py            yolo-tool 命令：GUI / s0~s5 子命令分发
│       ├── app/              GUI 工作台
│       │   ├── main.py       主窗口、步骤页装配、保存流程
│       │   ├── tabs.py       各步骤页（设置 / s1~s5）
│       │   ├── widgets.py    基础控件（表单、路径框、预览等）
│       │   ├── step_runner.py 后台执行步骤（python -m 子进程）
│       │   └── theme.py      全局 QSS 主题
│       ├── core/             共享配置与公共函数
│       │   ├── config.py     全局配置（目录、训练、转换参数）
│       │   └── utils.py      路径派生、配置读写、数据集拆分等
│       └── steps/            s0~s5 步骤脚本（python -m 运行）
│           ├── s0_collect_labels.py   收集源目录标签信息（类别/图片数）
│           ├── s1_prepare_data.py     数据准备：拆分训练/验证集 + data.yaml
│           ├── s2_visualize.py        可视化标注（结果图 + 可播放动画）
│           ├── s3_train.py            训练（YOLO26，输出 best.pt / last.pt）
│           ├── s4_inference.py        推理（默认验证集，可指定 --input）
│           └── s5_convert.py          导出 ONNX / TensorRT(.engine)
├── pyproject.toml           打包配置与依赖清单
├── run_all.sh               一键串联全部步骤
├── requirements.txt         run_all.sh 依赖安装清单（与 pyproject 一致）
├── README.md
│
├── models/                  本地预训练权重（yolo26n-*.pt 等，训练时优先使用，缺则在线下载）
├── converted_models/        早期转换产物目录（现产物统一写入 {DATA_ROOT}/权重，可忽略）
└── yolo26n-obb.pt           OBB 预训练权重样例（位于项目根目录）
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

## 5. 配置（src/yolo_tool/core/config.py）

| 配置 | 说明 |
| ---- | ---- |
| `TASK_TYPE` | 任务类型：detect / segment / obb / pose |
| `DEFAULT_SOURCE_DIR` | 原始数据目录（s1 输入） |
| `DEFAULT_DATA_ROOT` | 数据根目录，其余目录自动派生 |
| `MODELS_DIR` | 本地预训练权重目录（默认项目根 `models/`，可用环境变量 `YOLO_MODELS_DIR` 覆盖） |
| `TRAIN_RATIO` / `VAL_RATIO` | 训练/验证比例（GUI 中联动，总和 = 1，验证集 ≥ 0.05） |
| `EPOCHS` / `BATCH` / `IMGSZ` / `DEVICE` | 训练参数 |
| `CONF` / `IOU` | 推理阈值 |
| `EXPORT_TRT` / `TENSORRT_LIB` | TensorRT 导出开关与 trtexec 路径 |
