"""yolo-tool 命令行入口。

用法:
    yolo-tool                  启动图形界面（等价 python -m yolo_tool）
    yolo-tool s0 .. s5 [参数]  运行对应步骤（等价 python -m yolo_tool.steps.sX ...）
    yolo-tool --help           显示帮助
"""

import subprocess
import sys

STEPS = {
    "s0": "s0_collect_labels",
    "s1": "s1_prepare_data",
    "s2": "s2_visualize",
    "s3": "s3_train",
    "s4": "s4_inference",
    "s5": "s5_convert",
}

USAGE = """用法:
  yolo-tool                   启动图形界面（等同 python -m yolo_tool）
  yolo-tool s0 .. s5 [参数]   运行对应步骤（等同 python -m yolo_tool.steps.sX ...）
  yolo-tool --help            显示帮助

示例:
  yolo-tool s3 --epochs 50 --batch 8
"""


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        _launch_gui()
        return 0
    head = argv[0].lower()
    if head in ("-h", "--help", "help"):
        print(USAGE)
        return 0
    if head in STEPS:
        module = f"yolo_tool.steps.{STEPS[head]}"
        return subprocess.call([sys.executable, "-m", module] + argv[1:])
    # 其它参数（如 Qt 平台选项）交给 GUI
    _launch_gui()
    return 0


def _launch_gui():
    from .app.main import main as gui_main
    gui_main()
