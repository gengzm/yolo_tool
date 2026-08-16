"""
用户配置加载（机制代码，与 config.py 解耦）

config.py 末尾调用 load(config)，把用户 YAML 键值覆盖到 config 模块。
配置文件只加载第一个找到的：
  1. 环境变量 YOLO_CONFIG 指向的文件
  2. 项目根目录 run_yolo_config.yaml
  3. 用户主目录 ~/.config/yolo_tool/config.yaml（个人全局配置）

优先级：命令行 > info.yaml 记录 > 用户配置 > config.py 内置默认
"""
import os
import sys
from pathlib import Path


def _apply(config, data: dict, path: str) -> None:
    """应用用户配置：数据根/任务类型变更时自动重算派生路径，其余逐键覆盖。"""
    root = data.get("DEFAULT_DATA_ROOT")
    if root:
        # 数据根目录变更 → 自动重算默认源/数据集目录（未显式指定时）
        root = str(Path(str(root)).expanduser())
        data.setdefault("DEFAULT_SOURCE_DIR",
                        str(Path(root) / Path(config.DEFAULT_SOURCE_DIR).name))
        data.setdefault("DEFAULT_DATASET_DIR",
                        _dataset_dir(root, data.get("TASK_TYPE", config.TASK_TYPE)))
    elif "TASK_TYPE" in data and "DEFAULT_DATASET_DIR" not in data:
        # 仅任务类型变更 → 数据集目录名跟随任务类型
        data["DEFAULT_DATASET_DIR"] = _dataset_dir(
            str(Path(config.DEFAULT_DATASET_DIR).parent), data["TASK_TYPE"])
    # 逐键覆盖（可变容器原地更新，保证 from config import X 的模块也能看到新值）
    for key, value in data.items():
        if key == "CLASS_NAMES":
            config.CLASS_NAMES[:] = list(value)
            config.CLASS_TO_IDX.clear()
            config.CLASS_TO_IDX.update({n: i for i, n in enumerate(value)})
        elif key == "CLASS_TO_IDX":
            config.CLASS_TO_IDX.clear()
            config.CLASS_TO_IDX.update(value or {})
        elif hasattr(config, key):
            setattr(config, key, value)
        else:
            print(f"[WARN] [config] 未知配置键 {key!r}，已忽略（{path}）", file=sys.stderr)


def _dataset_dir(root: str, task_type) -> str:
    """数据集目录 = {root}/训练集_{task.upper()}"""
    return str(Path(root) / f"训练集_{str(task_type).upper()}")


def load(config) -> str:
    """加载用户配置文件并应用到 config 模块，返回生效的路径（无则空串）。"""
    candidates = [os.environ["YOLO_CONFIG"]] if os.environ.get("YOLO_CONFIG") else []
    candidates.append(str(Path(config.__file__).resolve().parents[2] / "run_yolo_config.yaml"))
    candidates.append(str(Path.home() / ".config" / "yolo_tool" / "config.yaml"))
    for path in candidates:
        p = Path(path)
        if not p.exists():
            continue
        try:
            import yaml
            with open(p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[WARN] [config] 读取用户配置失败 {p}: {e}", file=sys.stderr)
            continue
        if not isinstance(data, dict):
            print(f"[WARN] [config] 用户配置应为键值映射，已忽略: {p}", file=sys.stderr)
            continue
        _apply(config, data, str(p))
        # 日志走 stderr：避免被 shell 命令替换 $(...) 捕获混入 stdout 数据
        print(f"[config] 已加载用户配置: {p}", file=sys.stderr)
        return str(p)
    return ""
