"""
YOLO Tool 工作台 —— PySide6 图形界面

用法:
    /Users/Mac/miniconda3/envs/yolo/bin/python -m app.main

功能:
- Tab 分别对应项目设置与 s1~s5 各 step（s0 收集标签已并入 s1，页面移除）
- 项目设置页 = 目录组（数据根目录+各子目录）+ 参数区 + 转换导出，管控全部设置
- 设置页字段改动自动保存到 info.yaml（界面字段 <-> info.yaml 双向同步）
- 「执行」在后台线程运行对应脚本，日志统一输出到 s3 训练页日志区
"""
import sys
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QVBoxLayout, QWidget,
    QLabel, QGraphicsOpacityEffect,
)

from ..config import config as C
from ..config.utils import get_project_config, update_info_yaml

from . import tabs as T
from .step_runner import StepRunner
from .theme import apply_theme as apply_theme_qss

# 项目根（存放 models/ 等），供子进程 cwd 使用
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# step 脚本名 → 包内模块名（子进程以 python -m 方式运行，不再依赖 cwd）
STEP_MODULES = {
    "s0_collect_labels.py": "yolo_tool.steps.s0_collect_labels",
    "s1_prepare_data.py": "yolo_tool.steps.s1_prepare_data",
    "s2_visualize.py": "yolo_tool.steps.s2_visualize",
    "s3_train.py": "yolo_tool.steps.s3_train",
    "s4_inference.py": "yolo_tool.steps.s4_inference",
    "s5_convert.py": "yolo_tool.steps.s5_convert",
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO Tool 工作台")
        self.resize(1100, 780)
        self.runner = None
        self.cfg = {}
        self._toast = None
        self._toast_timer = None
        self._status_timer = None
        self._status_seq = 0
        self._status_msg_until = 0.0
        self._build_ui()
        self.reload_config()

    # ================= 界面构建 =================
    def _build_ui(self):
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(1, 1, 1, 1)
        root.setSpacing(1)

        self.tabs = QTabWidget()
        self.project_tab = T.ProjectTab(self)
        self.t1 = T.Step1Tab(self)
        self.t2 = T.Step2Tab(self)
        self.t3 = T.Step3Tab(self)
        self.t4 = T.Step4Tab(self)
        self.t5 = T.Step5Tab(self)
        for tab, name in ((self.project_tab, "项目设置"),
                          (self.t1, "s1 准备数据"),
                          (self.t2, "s2 可视化标注"),
                          (self.t3, "s3 训练"),
                          (self.t4, "s4 推理 + 误差分析"),
                          (self.t5, "s5 转换导出")):
            self.tabs.addTab(tab, name)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        root.addWidget(self.tabs, 1)
        self.setCentralWidget(central)

        self.statusBar().showMessage("就绪")

    # ================= 配置读写 =================
    def current_dataset_dir(self) -> str:
        """当前数据集目录（由【项目设置】页配置）"""
        try:
            return self.project_tab.v("dataset_dir") or C.DEFAULT_DATASET_DIR
        except Exception:
            return C.DEFAULT_DATASET_DIR

    def all_tabs(self):
        return (self.project_tab, self.t1, self.t2, self.t3, self.t4, self.t5)

    def _on_tab_changed(self, index: int):
        """切页钩子：页面实现 on_show() 时在切入时调用（如 s4 刷新训练权重列表）"""
        tab = self.tabs.widget(index)
        on_show = getattr(tab, "on_show", None)
        if callable(on_show):
            try:
                on_show()
            except Exception as e:
                self.log(f"[界面] {type(tab).__name__} 页面刷新失败: {e}\n")

    def reload_config(self):
        try:
            cfg = get_project_config(self.current_dataset_dir())
        except Exception as e:
            self.status_message(f"加载配置失败: {e}", warn=True)
            self.log(f"[界面] 加载配置失败: {e}\n")
            return
        self.cfg = cfg
        # 先按配置应用全局主题，再加载各页（避免先浅后深的闪烁）
        self.apply_theme(str(cfg.get("theme", "light")))
        for tab in self.all_tabs():
            try:
                tab.load_cfg(cfg)
            except Exception as e:
                self.log(f"[界面] {type(tab).__name__} 加载失败: {e}\n")
        self.log(f"[界面] 已加载配置 <- {cfg['dataset_dir']}\n")
        self._update_title()

    def _update_title(self):
        ds = self.current_dataset_dir()
        self.setWindowTitle(f"YOLO Tool — {ds}")
        # 提示消息显示期间不覆盖状态栏（避免防抖自动保存打断提示）
        if time.monotonic() < self._status_msg_until:
            return
        self.statusBar().showMessage(f"数据集目录: {ds}")

    def save_config(self, source_tab=None) -> str:
        """收集所有 Tab 参数合并写入 info.yaml，返回写入目录。
        自动保存（source_tab 非空）：失败仅写日志，避免反复弹窗打断输入；
        手动保存（source_tab 为空）：失败弹窗提醒。"""
        merged = {}
        for tab in self.all_tabs():
            try:
                fields = tab.save_cfg()
            except Exception as e:
                self.log(f"[界面] {type(tab).__name__} 保存字段失败: {e}\n")
                continue
            merged.update({k: v for k, v in fields.items()
                           if v not in (None, "")})
        ds = merged.pop("dataset_dir", None) or self.current_dataset_dir()
        try:
            update_info_yaml(ds, **merged)
            self.log(f"[界面] 配置已保存 -> {ds}/info.yaml\n")
        except Exception as e:
            if source_tab is None:       # 手动保存 → 状态栏提示
                self.status_message(f"保存配置失败: {e}", warn=True)
            self.log(f"[界面] 保存配置失败: {e}\n")
            return str(ds)
        # 更新内存配置，供 Tab 读取默认输出目录等
        try:
            self.cfg = get_project_config(ds)
        except Exception:
            self.cfg.setdefault("dataset_dir", ds)
        self._update_title()
        return str(ds)

    # ================= 主题 =================
    def apply_theme(self, name: str = None):
        """应用主题：name 可为 light / dark / 浅色 / 深色，缺省读当前选择"""
        if name is None:
            try:
                name = str(self.project_tab.v("theme"))
            except Exception:
                name = "light"
        app = QApplication.instance()
        if app is not None:
            apply_theme_qss(app, name)

    # ================= step 执行 =================
    def run_step(self, args: list, tab):
        if self.runner is not None and self.runner.isRunning():
            self.log("[界面] 有任务正在运行，请等待其完成\n")
            return
        # 脚本名 → python -m 模块方式运行，不依赖 cwd
        if args and args[0] in STEP_MODULES:
            args = ["-m", STEP_MODULES[args[0]]] + args[1:]
        tab.set_running(True)
        self.statusBar().showMessage("运行中: " + " ".join(args[:4]) + " ...")
        self.runner = StepRunner(args, cwd=str(PROJECT_ROOT))
        self.runner.log.connect(self.log)
        self.runner.done.connect(lambda ok, code: self._on_done(tab, ok, code))
        self.runner.start()

    def _on_done(self, tab, ok: bool, code: int):
        tab.set_running(False)
        if ok:
            self.log(f"\n[界面] 任务完成 (exit {code})\n")
            self.statusBar().showMessage(f"完成 (exit {code})")
        else:
            self.log(f"\n[界面] 任务失败 (exit {code})\n")
            self.statusBar().showMessage(f"失败 (exit {code})")
        try:
            tab.on_done(ok)
        except Exception as e:
            self.log(f"[界面] 任务后处理失败: {e}\n")

    # ================= 杂项 =================
    def status_message(self, text: str, timeout: int = 4000, warn: bool = False):
        """提示：状态栏常驻显示 + 右下角浮动窗（toast），时长一致。

        warn=True 时在消息前加「⚠ 」标记并延长到 6 秒，用于提醒类信息。
        纯 Qt 自绘控件，macOS / Ubuntu 表现一致。
        """
        if warn:
            text = "⚠ " + text
            timeout = 6000
        self._status_msg_until = time.monotonic() + timeout / 1000.0
        self.statusBar().showMessage(text, timeout)
        self.show_toast(text, timeout)
        # 提示结束后恢复状态栏常驻消息（数据集目录）
        self._status_seq += 1
        seq = self._status_seq
        if self._status_timer is not None:
            self._status_timer.stop()
        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.timeout.connect(lambda: self._restore_status(seq))
        self._status_timer.start(timeout)

    def _restore_status(self, seq: int):
        # 期间又有新提示则跳过，由新提示的恢复定时器负责
        if seq != self._status_seq or not self.isVisible():
            return
        self.statusBar().showMessage(
            f"数据集目录: {self.current_dataset_dir()}")

    def show_toast(self, text: str, timeout: int = 4000):
        """浮动提示（toast）：窗口右下角浮出，停留后自动淡出。"""
        if self._toast is None:
            self._toast = QLabel(self)
            self._toast.setAttribute(Qt.WA_TransparentForMouseEvents, True)
            self._toast.setStyleSheet(
                "QLabel { background-color: rgba(48, 52, 60, 238); color: #ffffff;"
                " border: 1px solid rgba(255, 255, 255, 60);"
                " border-radius: 6px; padding: 8px 14px; font-size: 13px; }"
            )
            self._toast_opacity = QGraphicsOpacityEffect(self._toast)
            self._toast.setGraphicsEffect(self._toast_opacity)
            self._toast_anim = QPropertyAnimation(self._toast_opacity,
                                                  b"opacity", self)
            self._toast_anim.finished.connect(self._toast.hide)

        self._toast.setText(text)
        self._toast.adjustSize()
        self._place_toast()
        self._toast.show()
        self._toast.raise_()

        # 重置为不透明，并重启停留计时（先停留，末尾再淡出）
        self._toast_anim.stop()
        self._toast_opacity.setOpacity(1.0)
        if self._toast_timer is not None:
            self._toast_timer.stop()
        self._toast_timer = QTimer(self)
        self._toast_timer.setSingleShot(True)
        self._toast_timer.timeout.connect(self._fade_toast_out)
        self._toast_timer.start(max(timeout - 400, 300))

    def _place_toast(self):
        """把 toast 定位到窗口水平居中、垂直约 60% 高度处，随窗口缩放跟随。"""
        m = 16
        x = (self.width() - self._toast.width()) // 2
        y = int(self.height() * 0.6) - self._toast.height() // 2
        self._toast.move(max(x, m), max(y, m))

    def _fade_toast_out(self):
        self._toast_anim.stop()
        self._toast_anim.setDuration(400)
        self._toast_anim.setStartValue(self._toast_opacity.opacity())
        self._toast_anim.setEndValue(0.0)
        self._toast_anim.start()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._toast is not None and self._toast.isVisible():
            self._place_toast()

    def open_dir(self, path: str):
        if path and Path(path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        else:
            self.status_message(f"目录不存在：{path}", warn=True)

    def log(self, text: str):
        """所有日志统一输出到 s3 训练页日志区（其他页面不显示日志）。"""
        if hasattr(self, "t3") and self.t3 is not None:
            self.t3.append_log(text)


def main():
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("YOLO Tool")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
