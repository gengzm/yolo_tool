"""
各 Step 功能 Tab：
- 项目设置页 = 数据根目录 + 目录区 + 参数区 + 转换导出设置，管控全部设置
- 其他页不再显示数据集目录输入框，当前目录显示在主窗口标题/状态栏
- 数字参数框短、目录路径框长；s2/s4 左右分割（s2 左 30% : 右 70%，s4 左 20% : 右 80%）
- s1 标签表格（标签 / ID / 处理方式）；s0 收集标签功能已合并进 s1，页面移除
- s3 训练：上方左参数、右上动画播放器，下方 40% 为全局运行日志
- s5 转换只读展示，imgsz / TensorRT 等配置在项目设置页统一管理
"""
import shutil
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QFrame, QGroupBox,
    QHeaderView, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit,
    QPushButton, QScrollArea, QSpinBox, QSplitter, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..config import config as C
from ..config.utils import (
    collect_class_names_from_json, load_info_yaml, resolve_model_path,
    update_info_yaml,
)

from .widgets import (
    MediaPlayer, PreviewPanel, add_row, browse_path, make_form, path_row,
    style_primary_button,
)

# 是否有 NVIDIA GPU / TensorRT（用于 s5 TensorRT 导出可用性判断）
try:
    import torch as _torch
    HAS_CUDA = bool(_torch.cuda.is_available())
except Exception:
    HAS_CUDA = False
try:
    import tensorrt as _trt   # noqa: F401
    _HAS_TRT_PY = True
except Exception:
    _HAS_TRT_PY = False
HAS_TENSORRT = _HAS_TRT_PY or shutil.which("trtexec") is not None

TASK_CHOICES = ["detect", "segment", "obb", "pose"]
SPLIT_CHOICES = ["train", "val", "test", "all"]

# 设置页参数区编辑框统一宽度（px）：与目录区名称框对齐，便于横向视觉一致
CTRL_WIDTH = 160


class BaseStepTab(QWidget):
    """step Tab 基类：统一字段管理 / 执行 / 配置读写"""

    task_label = "step"
    use_form = True    # False: 页面完全自定义
    hsplit = False     # True: 左表单 + 右图片预览
    left_frac = 0.25   # hsplit 时左栏默认宽度占比（s2/s4 用 20%）

    def __init__(self, main):
        super().__init__()
        self.main = main
        self.fields = {}   # key -> widget
        self.root = QVBoxLayout(self)
        self.root.setContentsMargins(1, 1, 1, 1)
        self.root.setSpacing(1)
        self.form = make_form()
        self._loading = False
        # 自动保存防抖定时器（所有 tab 共用）
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(300)
        self._save_timer.timeout.connect(self._flush_save)
        self.build_ui()
        self._layout_body()

    # ---------- 布局 ----------
    def _layout_body(self):
        if self.hsplit:
            split = QSplitter(Qt.Orientation.Horizontal)
            split.setHandleWidth(1)
            left = QWidget()
            ll = QVBoxLayout(left)
            ll.setContentsMargins(0, 0, 0, 0)
            ll.setSpacing(1)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.Shape.NoFrame)
            fp = QWidget()
            fp.setLayout(self.form)
            scroll.setWidget(fp)
            ll.addWidget(scroll, 1)
            self._build_actions()
            ll.addLayout(self._actions)
            split.addWidget(left)
            self.preview = self.build_preview()
            split.addWidget(self.preview)
            split.setStretchFactor(0, 0)
            split.setStretchFactor(1, 1)
            frac = max(0.1, min(0.5, self.left_frac))
            total = 1100
            lw = int(total * frac)
            split.setSizes([lw, total - lw])
            self.root.addWidget(split, 1)
        elif self.use_form:
            self.root.addLayout(self.form)
            self.root.addStretch(1)
            self._build_actions()
            self.root.addLayout(self._actions)
            self.preview = self.build_preview()
            if self.preview is not None:
                self.root.addWidget(self.preview, 1)
        else:
            # build_ui 已自行填充 self.root
            self._build_actions()
            self.root.addLayout(self._actions)
            self.preview = self.build_preview()
            if self.preview is not None:
                self.root.addWidget(self.preview, 1)

    # ---------- 字段构建辅助 ----------
    @staticmethod
    def _label(text: str, tip: str = "") -> QLabel:
        """带悬停提示的表单标签"""
        lab = QLabel(text)
        if tip:
            lab.setToolTip(tip)
        return lab

    def add_path(self, key, label, value, mode="dir", ffilter="", tip="",
                 placeholder=None):
        """目录/文件路径框：长；tip 提供悬停提示（替代「留空自动」类字样）"""
        if placeholder is None:
            placeholder = label if value else "留空 = 使用默认"
        w, edit = path_row(str(value or ""), mode, ffilter, placeholder=placeholder)
        self.fields[key] = edit
        if tip:
            label = self._label(label, tip)
        add_row(self.form, label, w)

    def add_text(self, key, label, value="", placeholder=""):
        """短文本框（如 device）"""
        edit = QLineEdit(str(value or ""))
        edit.setMinimumWidth(CTRL_WIDTH)
        edit.setMaximumWidth(CTRL_WIDTH)
        if placeholder:
            edit.setPlaceholderText(placeholder)
        self.fields[key] = edit
        add_row(self.form, label, edit)

    def add_readonly(self, key, label, value="", enabled=True):
        """只读展示（灰色背景），配置在项目设置页统一修改；
        enabled=False 时整行灰掉（如无 NVIDIA GPU / TensorRT）"""
        edit = QLineEdit(str(value or ""))
        edit.setReadOnly(True)   # 样式由全局主题的 QLineEdit:read-only 控制
        edit.setMinimumWidth(CTRL_WIDTH)
        edit.setMaximumWidth(CTRL_WIDTH)
        lab = self._label(label)
        lab.setEnabled(enabled)
        edit.setEnabled(enabled)
        self.fields[key] = edit
        add_row(self.form, lab, edit)

    def add_spin(self, key, label, value, lo=1, hi=10_000, step=1):
        """整数参数框：短"""
        sp = QSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setValue(int(value))
        sp.setMaximumWidth(CTRL_WIDTH)
        self.fields[key] = sp
        add_row(self.form, label, sp)

    def add_dspin(self, key, label, value, lo=0.0, hi=1.0, step=0.05,
                  decimals=3):
        """小数参数框：短"""
        sp = QDoubleSpinBox()
        sp.setRange(lo, hi)
        sp.setSingleStep(step)
        sp.setDecimals(decimals)
        sp.setValue(float(value))
        sp.setMaximumWidth(CTRL_WIDTH)
        self.fields[key] = sp
        add_row(self.form, label, sp)

    def add_combo(self, key, label, items, value=None):
        cb = QComboBox()
        cb.addItems(items)
        cb.setMaximumWidth(CTRL_WIDTH)
        if value is not None and value in items:
            cb.setCurrentText(value)
        self.fields[key] = cb
        add_row(self.form, label, cb)

    def add_check(self, key, label, checked=False):
        chk = QCheckBox(label)
        chk.setChecked(bool(checked))
        self.fields[key] = chk
        add_row(self.form, "", chk)

    # ---------- 执行按钮 ----------
    def _build_actions(self):
        self._actions = QHBoxLayout()
        self.run_btn = QPushButton(f"▶ 执行 {self.task_label}")
        style_primary_button(self.run_btn)
        self.run_btn.clicked.connect(self.on_run)
        self._actions.addWidget(self.run_btn)
        for btn in self.extra_action_buttons():
            self._actions.addWidget(btn)
        self._actions.addStretch(1)

    def set_running(self, running: bool):
        if hasattr(self, "run_btn"):
            self.run_btn.setEnabled(not running)
            self.run_btn.setText("⏳ 运行中..." if running
                                 else f"▶ 执行 {self.task_label}")

    # ---------- 可覆盖的扩展点 ----------
    def extra_action_buttons(self) -> list:
        return []

    def build_preview(self):
        return None

    def on_done(self, ok: bool):
        pass

    # ---------- 子类实现 ----------
    def build_ui(self):
        raise NotImplementedError

    def load_cfg(self, cfg: dict):
        raise NotImplementedError

    def save_cfg(self) -> dict:
        raise NotImplementedError

    def build_args(self) -> list:
        raise NotImplementedError

    # ---------- 通用取值 ----------
    def v(self, key):
        w = self.fields[key]
        if isinstance(w, (QSpinBox, QDoubleSpinBox)):
            return w.value()
        if isinstance(w, QCheckBox):
            return w.isChecked()
        if isinstance(w, QComboBox):
            return w.currentText()
        return w.text().strip()

    def ds(self) -> str:
        """当前数据集目录（全局顶部配置）"""
        return str(self.main.cfg.get("dataset_dir", ""))

    def on_run(self):
        self.main.save_config(self)
        self.main.run_step(self.build_args(), self)

    # ---------- 自动保存（所有 tab 共用，防抖定时器在 __init__ 创建） ----------
    def _auto_save(self):
        if self._loading or not self.main.cfg:
            return
        self._save_timer.start()

    def _flush_save(self):
        try:
            self.main.save_config(self)
        except Exception as e:
            self.main.log(f"[界面] 自动保存失败: {e}\n")


class ProjectTab(BaseStepTab):
    """项目设置：数据根目录 + 目录区 + 参数区 + 转换导出，管控全部设置"""

    task_label = "项目设置"
    _loading = False
    _ratio_linking = False   # 训练/验证比例互相关联锁，防 setValue 死循环

    # ---------- UI ----------
    def build_ui(self):
        # 目录组：全部强制由 data_root 派生，完整路径只读、不给自由选择；
        # 仅「原始数据 / 输出目录」允许修改子目录名称（完整路径 = data_root / 名称）
        dirs_box = QGroupBox("目录（统一由数据根目录管理）")
        dirs_box.setToolTip("所有目录完整路径均固定由「数据根目录」自动派生，\n"
                            "只能修改少数子目录的名称，不允许自定义完整路径")
        df = make_form()
        # 数据根目录：编辑框左边缘与其它行「路径只读灰框」左边缘对齐
        # （前面留出与名称框等宽的位置），右侧拉伸顶到浏览按钮，按钮贴组框右边界
        dw = QWidget()
        dl = QHBoxLayout(dw)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.setSpacing(4)
        place = QWidget()
        place.setFixedWidth(CTRL_WIDTH)   # 与其它行的名称编辑框同宽
        dl.addWidget(place)
        edit = QLineEdit(str(C.DEFAULT_DATA_ROOT))
        dl.addWidget(edit, 1)
        btn = QPushButton("浏览")
        btn.setMaximumWidth(72)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(lambda: browse_path(edit, "dir"))
        dl.addWidget(btn)
        self.fields["data_root"] = edit
        # 文本真正变化才触发保存（textChanged）；失焦同步派生目录并按需补存
        edit.textChanged.connect(self._auto_save)
        edit.editingFinished.connect(self._on_root_edited)
        add_row(df, self._label("数据根目录",
                                "所有数据的根目录（data_root），\n其余目录全部由它自动派生"),
                dw)
        # 原始数据：可改名称 + 限范围浏览（上一级固定为数据根目录）；数据集：固定派生只读
        self._add_dir_name_row(df, "原始数据目录",
                               "source_dir_name", "source_dir", "原始数据",
                               "LabelMe 原始标注数据所在目录（step1 读取），\n"
                               "上一级固定为数据根目录：可直接改名称，\n"
                               "或点「选择」在数据根目录下浏览子目录自动回填",
                               browse=True)
        # 数据集目录：名称跟随任务类型自动生成、灰掉只读，仅展示不对齐不参与编辑
        dw = QWidget()
        dl = QHBoxLayout(dw)
        dl.setContentsMargins(0, 0, 0, 0)
        dl.setSpacing(4)
        dne = QLineEdit("")
        dne.setReadOnly(True)
        dne.setMinimumWidth(CTRL_WIDTH)   # 固定宽度，与其它名称框对齐
        dne.setMaximumWidth(CTRL_WIDTH)
        self.fields["dataset_dir_name"] = dne
        dro = QLineEdit("")
        dro.setReadOnly(True)
        dro.setMinimumWidth(CTRL_WIDTH)
        self.fields["dataset_dir"] = dro
        dl.addWidget(dne)
        dl.addWidget(dro, 1)
        add_row(df, self._label("数据集目录",
                                "生成的 YOLO 数据集目录，\n"
                                "名称跟随任务类型自动生成，不可修改"),
                dw)
        # 输出目录：只能改「名称」，完整路径自动 = data_root / 名称
        self._add_dir_name_row(df, "权重输出目录",
                               "weights_dir_name", "weights_dir", "权重",
                               "step5 转换导出的 ONNX / engine / best.pt 等\n"
                               "最终部署权重存放目录")
        self._add_dir_name_row(df, "训练输出目录",
                               "run_out_dir_name", "run_out_dir", "run_out",
                               "step3 训练输出（run_out）目录，\n"
                               "每次训练自动生成时间戳子目录")
        self._add_dir_name_row(df, "可视化输出目录",
                               "visualize_dir_name", "visualize_dir", "可视化标注",
                               "step2 标注可视化图片的输出目录")
        dirs_box.setLayout(df)
        df.setContentsMargins(2, 2, 2, 2)

        # 训练与推理参数：左侧原有训练/推理参数，右侧数据增强（左右分栏）
        params_box = QGroupBox("训练与推理参数")
        params_box.setToolTip("step3 训练与 step4 推理共用参数；\n"
                              "右侧「数据增强」在训练时生效")
        ph = QHBoxLayout(params_box)
        ph.setContentsMargins(2, 2, 2, 2)
        ph.setSpacing(10)
        pf = make_form()
        cb = QComboBox()
        cb.addItems(TASK_CHOICES)
        cb.setCurrentText(C.TASK_TYPE)
        cb.setMaximumWidth(CTRL_WIDTH)
        cb.currentTextChanged.connect(self._on_task_changed)
        self.fields["task_type"] = cb
        add_row(pf, self._label("任务类型",
                                "目标检测 / 实例分割 / 旋转框检测（OBB），\n"
                                "数据集目录名随之自动变化"),
                cb)

        thm = QComboBox()
        thm.addItems(["浅色", "深色"])
        thm.setMaximumWidth(CTRL_WIDTH)
        thm.currentIndexChanged.connect(self._on_theme_changed)
        self.fields["theme"] = thm
        add_row(pf, self._label("界面主题",
                                "浅色 / 深色，全局即时切换\n（含编辑框、图片框等）"),
                thm)

        ms = QComboBox()
        ms.addItems(["n", "s", "m", "l", "x"])
        ms.setCurrentText(getattr(C, "MODEL_SIZE", "n"))
        ms.setMaximumWidth(CTRL_WIDTH)
        ms.currentIndexChanged.connect(self._auto_save)
        self.fields["model_size"] = ms
        add_row(pf, self._label("模型规格",
                                "YOLO 模型规格 n/s/m/l/x\n（越大越准、越耗显存）"),
                ms)

        tr = QDoubleSpinBox()
        tr.setRange(0.0, 0.95)   # 上限 0.95：保证验证集至少 0.05
        tr.setSingleStep(0.05); tr.setDecimals(3)
        tr.setValue(float(C.TRAIN_RATIO)); tr.setMaximumWidth(CTRL_WIDTH)
        tr.valueChanged.connect(self._on_ratio_changed)
        self.fields["train_ratio"] = tr
        add_row(pf, self._label("训练集比例",
                                "训练集 / 验证集比例自动互补，\n总和恒为 1"),
                tr)

        vr = QDoubleSpinBox()
        vr.setRange(0.05, 1.0)   # 验证集必须非空（训练评估 / best.pt 需要），下限 0.05
        vr.setSingleStep(0.05); vr.setDecimals(3)
        vr.setValue(float(C.VAL_RATIO)); vr.setMaximumWidth(CTRL_WIDTH)
        vr.valueChanged.connect(self._on_ratio_changed)
        self.fields["val_ratio"] = vr
        add_row(pf, self._label("验证集比例",
                                "验证集必须非空：训练评估与 best.pt 生成依赖它，\n"
                                "下限 0.05，与训练集比例自动互补"),
                vr)

        for key, label, tip, val, lo, hi in (
            ("epochs", "训练轮数", "训练轮数（epochs），越多训练越久",
             C.EPOCHS, 1, 100_000),
            ("batch", "batch size", "每批送入训练的图片数（batch）",
             C.BATCH, 1, 10_000),
            ("imgsz", "输入尺寸", "训练 / 推理输入尺寸（imgsz），建议 32 的倍数",
             C.IMGSZ, 32, 8192),
        ):
            sp = QSpinBox()
            sp.setRange(lo, hi)
            sp.setValue(int(val))
            sp.setMaximumWidth(CTRL_WIDTH)
            if key == "imgsz":
                sp.setSingleStep(32)
            sp.valueChanged.connect(self._auto_save)
            self.fields[key] = sp
            add_row(pf, self._label(label, tip), sp)

        for key, label, tip, val in (("conf", "置信度阈值",
                                      "推理过滤阈值（conf），越大越保守", C.CONF),
                                     ("iou", "IoU 阈值",
                                      "NMS 去重 IoU 阈值，越大保留越多重叠框", C.IOU)):
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 1.0); sp.setSingleStep(0.05); sp.setDecimals(3)
            sp.setValue(float(val)); sp.setMaximumWidth(CTRL_WIDTH)
            sp.valueChanged.connect(self._auto_save)
            self.fields[key] = sp
            add_row(pf, self._label(label, tip), sp)

        dev = QLineEdit("")
        dev.setPlaceholderText("如 0 / cpu / mps，留空自动")
        dev.setMaximumWidth(CTRL_WIDTH)
        dev.textChanged.connect(self._auto_save)
        self.fields["device"] = dev
        add_row(pf, self._label("训练设备",
                                "训练/推理设备（device）：GPU 编号 / cpu / mps，\n"
                                "留空自动"),
                dev)
        pf.setContentsMargins(2, 2, 2, 2)
        ph.addLayout(pf, 3)

        # 右侧：数据增强（训练时生效，写回 info.yaml 并传给 step3）
        aug_box = QGroupBox("数据增强")
        aug_box.setToolTip("训练时的数据增强（ultralytics 参数），\n"
                           "改动自动保存并传给 step3 训练")
        af = make_form()
        aug_specs = (
            ("fliplr", "左右翻转", "随机水平翻转概率（fliplr）"),
            ("flipud", "上下翻转", "随机垂直翻转概率（flipud）"),
            ("degrees", "旋转角度", "随机旋转角度范围 ±degrees（度）"),
            ("scale", "缩放", "随机缩放增益 ±scale（如 0.5 = ±50%）"),
            ("translate", "平移", "随机平移比例（宽/高的 10% 默认）"),
            ("mosaic", "马赛克拼接", "Mosaic 概率（1.0 = 每轮都用 4 图拼接）"),
            ("mixup", "混合", "MixUp 图像混合概率"),
        )
        for key, label, tip in aug_specs:
            if key == "degrees":
                sp = QSpinBox()
                sp.setRange(0, 180)
                sp.setValue(int(C.AUGMENT_DEFAULTS[key]))
                sp.setMaximumWidth(CTRL_WIDTH)
            else:
                sp = QDoubleSpinBox()
                sp.setRange(0.0, 1.0); sp.setSingleStep(0.05); sp.setDecimals(2)
                sp.setValue(float(C.AUGMENT_DEFAULTS[key]))
                sp.setMaximumWidth(CTRL_WIDTH)
            sp.valueChanged.connect(self._auto_save)
            self.fields[key] = sp
            add_row(af, self._label(label, tip), sp)
        aug_box.setLayout(af)
        ph.addWidget(aug_box, 2)

        # 转换与导出：路径编辑框贴近标签；CUDA 提示在浏览按钮右侧
        trt_box = QGroupBox("转换与导出")
        trt_box.setToolTip("step5：将 best.pt 转换为 ONNX（可选 TensorRT engine），\n"
                           "产物输出到「权重输出目录」")
        tv = QVBoxLayout(trt_box)
        tv.setContentsMargins(2, 2, 2, 2)
        tv.setSpacing(2)
        row, edit = path_row(str(getattr(C, "TENSORRT_LIB", "")), mode="file",
                             placeholder="留空 = 自动探测（trtexec / import tensorrt）")
        edit.textChanged.connect(self._auto_save)
        self.fields["trt_lib"] = edit
        trt_row = QWidget()
        th = QHBoxLayout(trt_row)
        th.setContentsMargins(0, 0, 0, 0)
        th.setSpacing(4)
        th.addWidget(QLabel("trtexec 路径"))
        th.addWidget(row, 1)
        tv.addWidget(trt_row)
        if not HAS_CUDA:
            tip = QLabel("未检测到 NVIDIA GPU (CUDA)，TensorRT 导出不可用")
            tip.setObjectName("hint")
            row.layout().addWidget(tip)   # 放在浏览按钮右侧
        trt_box.setLayout(tv)

        # 三个组框跨列占满整行（不占用标签列），紧贴表单左边缘
        self.form.addRow(dirs_box)
        self.form.addRow(params_box)
        self.form.addRow(trt_box)

    def _build_actions(self):
        self._actions = QHBoxLayout()
        self._actions.addStretch(1)

    def build_args(self) -> list:
        return []

    # ---------- 目录联动（强制统一派生，不给自由选择） ----------
    def _add_dir_name_row(self, form, label, name_key, path_key, default_name,
                          tooltip: str = "", browse: bool = False):
        """名称目录行：名称编辑框（可改）+ 完整路径只读预览（自动派生）

        browse=True 时附加「选择」按钮：限浏览数据根目录下的子目录并回填名称。
        """
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        ne = QLineEdit(default_name)
        ne.setMinimumWidth(CTRL_WIDTH)   # 固定宽度，保证与数据根目录占位对齐
        ne.setMaximumWidth(CTRL_WIDTH)
        ne.setPlaceholderText("子目录名称")
        # 文本真正变化才触发保存；失焦仅做非法字符清理与目录同步
        ne.textChanged.connect(self._auto_save)
        ne.editingFinished.connect(self._on_name_edited)
        self.fields[name_key] = ne
        pe = QLineEdit("")
        pe.setReadOnly(True)
        pe.setMinimumWidth(CTRL_WIDTH)
        self.fields[path_key] = pe
        lay.addWidget(ne)
        if browse:
            btn = QPushButton("选择")
            btn.setMaximumWidth(56)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)
            btn.setToolTip("在数据根目录下浏览选择子目录，自动回填名称")
            btn.clicked.connect(lambda: self._browse_dir_name(name_key, path_key))
            lay.addWidget(btn)
        lay.addWidget(pe, 1)
        add_row(form, self._label(label, tooltip), w)

    def _browse_dir_name(self, name_key, path_key):
        """限范围浏览：只能在「数据根目录」下选择直接子目录，回填名称

        QFileDialog 本身无法限制导航范围，故选择后强校验：
        所选目录必须是 data_root 的直接子目录（相对路径恰为一级），
        超出范围（父目录之外 / 多级嵌套 / 根目录本身）一律拒绝并提示。
        """
        root = self.v("data_root").strip()
        if not root:
            self.main.status_message(
                "请先设置「数据根目录」，再浏览选择其下的原始数据子目录",
                warn=True)
            return
        root_p = Path(root).expanduser().resolve()
        # 起始目录：当前已选路径在根目录内且存在 → 用它，否则从根目录开始
        start = root_p
        cur = self.v(path_key).strip()
        if cur:
            cp = Path(cur).expanduser().resolve()
            if cp.is_relative_to(root_p) and cp.exists():
                start = cp
        path = QFileDialog.getExistingDirectory(
            self, "选择数据根目录下的子目录", str(start))
        if not path:
            return
        sel = Path(path).expanduser().resolve()
        if sel == root_p:
            self.main.status_message("不能选择数据根目录本身，请选择其下的子目录",
                                     warn=True)
            return
        try:
            rel = sel.relative_to(root_p)
        except ValueError:
            self.main.status_message(
                f"所选目录不在数据根目录内：{sel}（根目录：{root_p}），"
                "原始数据必须放在数据根目录下", warn=True)
            return
        if len(rel.parts) != 1:
            self.main.status_message(
                f"仅支持数据根目录下的直接子目录（一级），"
                f"当前相对路径：{rel}，请选择更上一级", warn=True)
            return
        # 回填名称并同步派生路径（setText 触发 textChanged → 自动保存）
        self.fields[name_key].setText(rel.name)
        self._sync_all_dirs()

    def _on_root_edited(self):
        # 失焦：先同步派生目录，再与上次保存值比较，真变了才补存（点一下不改不保存）
        self._sync_all_dirs()
        try:
            changed = self.v("data_root") != str(self.main.cfg.get("data_root", ""))
        except Exception:
            changed = True
        if changed:
            self._auto_save()

    def _on_name_edited(self):
        # 只允许纯目录名：拒绝路径分隔符、空值回退默认（不给自由指定完整路径）
        for key, default in (("source_dir_name", "原始数据"),
                             ("weights_dir_name", "权重"),
                             ("run_out_dir_name", "run_out"),
                             ("visualize_dir_name", "可视化标注")):
            txt = self.v(key).strip().strip("/\\")
            if "/" in txt or "\\" in txt:
                txt = txt.replace("/", "_").replace("\\", "_")
            if not txt:
                txt = default
            if self.v(key) != txt:
                # setText 会触发 textChanged → 自动保存（值确实变化了才保存）
                self.fields[key].setText(txt)
        self._sync_all_dirs()

    def _on_task_changed(self):
        self._sync_all_dirs()
        self._auto_save()

    def _on_theme_changed(self):
        if self._loading:
            return
        self.main.apply_theme()   # 立即切换全局主题
        self._auto_save()

    def _on_ratio_changed(self):
        """训练/验证比例联动：改一个，另一个自动补成 1 - x，总和恒为 1"""
        if self._loading or self._ratio_linking:
            return
        self._ratio_linking = True
        try:
            if self.sender() is self.fields["train_ratio"]:
                other = round(1.0 - float(self.v("train_ratio")), 3)
                self.fields["val_ratio"].setValue(other)
            else:
                other = round(1.0 - float(self.v("val_ratio")), 3)
                self.fields["train_ratio"].setValue(other)
        finally:
            self._ratio_linking = False
        self._auto_save()

    def _sync_all_dirs(self):
        if self._loading:
            return
        root = self.v("data_root").strip()
        if not root:
            return
        self.fields["source_dir"].setText(
            C.path_for_source(root, self.v("source_dir_name")))
        task = str(self.v("task_type")).lower()
        self.fields["dataset_dir_name"].setText(f"训练集_{task.upper()}")
        self.fields["dataset_dir"].setText(
            C.path_for_dataset(root, self.v("task_type")))
        self.fields["weights_dir"].setText(
            C.path_for_weights(root, self.v("weights_dir_name")))
        self.fields["run_out_dir"].setText(
            C.path_for_run_out(root, self.v("run_out_dir_name")))
        self.fields["visualize_dir"].setText(
            C.path_for_visualize(root, self.v("visualize_dir_name")))

    # ---------- 配置读写 ----------
    def load_cfg(self, cfg: dict):
        self._loading = True
        try:
            for key in ("data_root", "source_dir", "dataset_dir",
                        "weights_dir", "run_out_dir", "visualize_dir"):
                if key in self.fields:
                    self.fields[key].setText(str(cfg.get(key, "")))
            # 目录名称：优先取配置记录，缺失时从完整路径反推最后一段
            for name_key, path_key, default in (
                ("source_dir_name", "source_dir", "原始数据"),
                ("weights_dir_name", "weights_dir", "权重"),
                ("run_out_dir_name", "run_out_dir", "run_out"),
                ("visualize_dir_name", "visualize_dir", "可视化标注"),
            ):
                name = str(cfg.get(name_key) or "")
                if not name and cfg.get(path_key):
                    name = Path(str(cfg[path_key])).name
                self.fields[name_key].setText(name or default)
            theme = str(cfg.get("theme", "light"))
            self.fields["theme"].setCurrentIndex(1 if theme == "dark" else 0)
            task = str(cfg.get("task_type", "detect")).lower()
            self.fields["task_type"].setCurrentText(task)
            self.fields["dataset_dir_name"].setText(f"训练集_{task.upper()}")
            self.fields["model_size"].setCurrentText(
                cfg.get("model_size", "n"))
            tr = float(cfg.get("train_ratio", 0.8))
            tr = min(max(tr, 0.0), 0.95)
            self.fields["train_ratio"].setValue(tr)
            # 验证集由训练集自动补齐，保证加载后总和恒为 1
            self.fields["val_ratio"].setValue(round(1.0 - tr, 3))
            self.fields["epochs"].setValue(int(cfg.get("epochs", 100)))
            self.fields["batch"].setValue(int(cfg.get("batch", 16)))
            self.fields["imgsz"].setValue(int(cfg.get("imgsz", 640)))
            self.fields["conf"].setValue(float(cfg.get("conf", 0.25)))
            self.fields["iou"].setValue(float(cfg.get("iou", 0.45)))
            self.fields["device"].setText(str(cfg.get("device", "")))
            self.fields["trt_lib"].setText(str(cfg.get("trt_lib") or ""))
            # 数据增强：默认值来自 config.AUGMENT_DEFAULTS
            for key in C.AUGMENT_DEFAULTS:
                default = C.AUGMENT_DEFAULTS[key]
                if isinstance(self.fields[key], QSpinBox):
                    self.fields[key].setValue(int(cfg.get(key, default)))
                else:
                    self.fields[key].setValue(float(cfg.get(key, default)))
        finally:
            self._loading = False

    def save_cfg(self) -> dict:
        cfg = {
            "theme": "dark" if self.fields["theme"].currentIndex() == 1
                     else "light",
            "data_root": self.v("data_root"),
            "source_dir": self.v("source_dir"),
            "dataset_dir": self.v("dataset_dir"),
            "dataset_dir_name": self.v("dataset_dir_name"),
            "weights_dir": self.v("weights_dir"),
            "run_out_dir": self.v("run_out_dir"),
            "visualize_dir": self.v("visualize_dir"),
            "source_dir_name": self.v("source_dir_name"),
            "weights_dir_name": self.v("weights_dir_name"),
            "run_out_dir_name": self.v("run_out_dir_name"),
            "visualize_dir_name": self.v("visualize_dir_name"),
            "task_type": self.v("task_type"),
            "model_size": self.v("model_size"),
            "train_ratio": self.v("train_ratio"),
            "val_ratio": self.v("val_ratio"),
            "epochs": self.v("epochs"),
            "batch": self.v("batch"),
            "imgsz": self.v("imgsz"),
            "conf": self.v("conf"),
            "iou": self.v("iou"),
            "device": self.v("device"),
            "trt_lib": self.v("trt_lib"),
        }
        for key in C.AUGMENT_DEFAULTS:      # 数据增强参数一并写入
            cfg[key] = self.v(key)
        return cfg


class Step1Tab(BaseStepTab):
    """s1 准备数据：标签 | ID | 处理方式 表格，执行时读取项目设置参数"""

    task_label = "s1 准备数据"
    use_form = False
    _rows = []      # [(name, QSpinBox(id), QComboBox(参与训练/跳过))]
    _loading = False

    def build_ui(self):
        top = QHBoxLayout()
        top.setSpacing(1)
        self.reload_btn = QPushButton("重新加载标签")
        self.reload_btn.setFixedHeight(36)
        self.reload_btn.clicked.connect(lambda: self.reload_labels(silent=False))
        top.addWidget(self.reload_btn)
        info = QLabel("标签顺序即默认 ID，可修改；参与训练的 ID 独占 0..n-1（冲突自动纠正）；"
                      "「跳过」→ 不参与训练、ID 置空灰掉；改动自动保存")
        info.setObjectName("hint")
        top.addWidget(info)
        top.addStretch(1)
        top_w = QWidget()
        top_w.setLayout(top)
        self.root.addWidget(top_w)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["标签", "ID", "处理方式"])
        # 三列等宽（各占 1/3）
        header = self.table.horizontalHeader()
        for c in range(3):
            header.setSectionResizeMode(c, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.root.addWidget(self.table, 1)

    def _build_actions(self):
        # 「执行 s1 准备数据」按钮左右居中（顶部已有提示文字，此处不加 hint）
        self._actions = QHBoxLayout()
        self._actions.addStretch(1)
        self.run_btn = QPushButton(f"▶ 执行 {self.task_label}")
        style_primary_button(self.run_btn)
        self.run_btn.clicked.connect(self.on_run)
        self._actions.addWidget(self.run_btn)
        self._actions.addStretch(1)

    # ---------- 标签加载 / 保存（全部自动保存） ----------
    def reload_labels(self, silent: bool = True):
        info = load_info_yaml(self.ds())
        labels = info.get("labels") or {}
        names = list(labels.get("names") or [])
        if not names:
            src = str(self.main.cfg.get("source_dir", ""))
            if src and Path(src).is_dir():
                names = collect_class_names_from_json(src)
        if not names:
            if not silent:
                self.main.status_message(
                    "未找到标签：请在【项目设置】配置原始数据目录后"
                    "点「重新加载标签」", warn=True)
            return
        ignore = set(labels.get("ignore", []) or [])
        self._rows = []
        self.table.setRowCount(len(names))
        self._loading = True
        try:
            nxt = 0                             # 参与训练标签的下一个 ID（0..n-1）
            for i, name in enumerate(names):
                name_item = QTableWidgetItem(name)
                name_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                skip = name in ignore
                id_spin = QSpinBox()
                id_spin.setRange(-1, 9999)
                id_spin.setSpecialValueText("")     # ID 为空（跳过行）
                id_spin.setValue(-1 if skip else nxt)
                id_spin.setEnabled(not skip)        # 跳过 → 灰掉不可改
                if not skip:
                    nxt += 1
                id_spin.valueChanged.connect(self._on_labels_changed)
                combo = QComboBox()
                combo.addItems(["参与训练", "跳过"])
                combo.setCurrentText("跳过" if skip else "参与训练")
                combo.currentTextChanged.connect(self._on_labels_changed)
                self.table.setItem(i, 0, name_item)
                self.table.setCellWidget(i, 1, id_spin)
                self.table.setCellWidget(i, 2, combo)
                self._rows.append((name, id_spin, combo))
        finally:
            self._loading = False
        self.main.log(f"[s1] 已加载 {len(names)} 个标签\n")

    def _on_labels_changed(self):
        """标签/ID/处理方式改动：同步跳过行 ID 状态 → 校验独占 → 自动保存。"""
        if self._loading:
            return
        for name, sp, cb in self._rows:
            if cb.currentText() == "跳过":
                if sp.isEnabled() or sp.value() != -1:
                    sp.blockSignals(True)
                    sp.setEnabled(False)        # 灰掉
                    sp.setValue(-1)             # ID 置空
                    sp.blockSignals(False)
            elif not sp.isEnabled():
                sp.blockSignals(True)
                sp.setEnabled(True)             # 恢复参与
                sp.blockSignals(False)
        self._normalize_ids(log=True)
        self.save_labels(silent=True)

    def _normalize_ids(self, log: bool = False):
        """参与训练的标签 ID 必须独占 0..n-1（n=参与数）；重复/越界/空值自动纠正。"""
        parts = [(name, sp) for name, sp, cb in self._rows
                 if cb.currentText() != "跳过"]
        k = len(parts)
        used = set()
        for name, sp in parts:                  # 第一轮：标记合法占用
            v = sp.value()
            if 0 <= v < k and v not in used:
                used.add(v)
        for name, sp in parts:                  # 第二轮：非法 → 分配首个空 ID
            v = sp.value()
            if 0 <= v < k and v not in used:
                used.add(v)
                continue
            nv = next((i for i in range(k) if i not in used), k)
            sp.blockSignals(True)
            sp.setValue(nv)
            sp.blockSignals(False)
            used.add(nv)
            if log and v != nv:
                self.main.log(f"[s1] 标签「{name}」ID 冲突，已自动调整为 {nv}\n")

    def save_labels(self, silent: bool = True):
        if not self._rows:
            if not silent:
                self.main.status_message("请先点击「重新加载标签」", warn=True)
            return
        active = [(sp.value(), name) for name, sp, cb in self._rows
                  if cb.currentText() != "跳过"]
        active.sort(key=lambda x: x[0])         # 参与训练按 ID 排序
        names = [n for _, n in active]
        names.extend(name for name, sp, cb in self._rows
                     if cb.currentText() == "跳过")   # 跳过后置，ID 为空
        ignore = sorted(name for name, sp, cb in self._rows
                        if cb.currentText() == "跳过")
        update_info_yaml(
            self.ds(),
            labels={"count": len(names), "names": names, "ignore": ignore},
        )
        self.main.cfg["labels"] = {"count": len(names), "names": names,
                                   "ignore": ignore}
        if not silent:
            self.main.log(f"[s1] 标签设置已保存：参与训练 {len(active)} 类，"
                          f"跳过 {ignore if ignore else '无'}\n")

    # ---------- BaseStepTab 接口 ----------
    def load_cfg(self, cfg: dict):
        self.reload_labels(silent=True)

    def save_cfg(self) -> dict:
        if self._rows:
            self.save_labels()
        return {}

    def build_args(self) -> list:
        args = ["s1_prepare_data.py", "--dataset_dir", self.ds()]
        for key, cli in (("source_dir", "--source_dir"),
                         ("task_type", "--task_type")):
            if self.main.cfg.get(key):
                args += [cli, str(self.main.cfg.get(key))]
        for key, cli in (("train_ratio", "--train_ratio"),
                         ("val_ratio", "--val_ratio")):
            if self.main.cfg.get(key):
                args += [cli, str(self.main.cfg.get(key))]
        return args


class Step2Tab(BaseStepTab):
    """s2 可视化标注：左参数（30%）+ 右预览（70%），左右 3:7"""

    task_label = "s2 可视化标注"
    hsplit = True
    left_frac = 0.3   # 左右 3:7

    def build_ui(self):
        self.add_combo("split", "数据子集", SPLIT_CHOICES, "train")
        self.add_spin("circle_diameter_first", "第一个点直径(px)", 6, 1, 200)
        self.add_spin("circle_diameter_other", "其他点直径(px)", 4, 1, 200)
        self.add_spin("line_width", "线宽(px)", 2, 1, 50)
        open_btn = QPushButton("打开可视化目录")
        open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_btn.setMaximumWidth(CTRL_WIDTH)   # 与上面 4 个编辑框同宽
        open_btn.clicked.connect(lambda: self.main.open_dir(self.output_dir_path()))
        add_row(self.form, "", open_btn)

    # ---------- 布局：左 30% 表单 + 右 70% 预览，执行按钮横贯底部居中 ----------
    def _layout_body(self):
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setHandleWidth(1)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        fp = QWidget()
        fp.setLayout(self.form)
        scroll.setWidget(fp)
        ll.addWidget(scroll, 1)
        self._build_actions()
        ll.addLayout(self._actions)      # 执行按钮置于左侧栏内底部，居中、不占右侧
        split.addWidget(left)
        self.preview = self.build_preview()
        split.addWidget(self.preview)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        total = 1100
        lw = int(total * max(0.1, min(0.5, self.left_frac)))
        split.setSizes([lw, total - lw])
        self.root.addWidget(split, 1)

    def _build_actions(self):
        # 「执行 s2 可视化标注」按钮左右居中
        self._actions = QHBoxLayout()
        self._actions.addStretch(1)
        self.run_btn = QPushButton(f"▶ 执行 {self.task_label}")
        style_primary_button(self.run_btn)
        self.run_btn.clicked.connect(self.on_run)
        self._actions.addWidget(self.run_btn)
        self._actions.addStretch(1)

    def load_cfg(self, cfg: dict):
        self.fields["circle_diameter_first"].setValue(
            int(cfg.get("circle_diameter_first", 6)))
        self.fields["circle_diameter_other"].setValue(
            int(cfg.get("circle_diameter_other", 4)))
        self.fields["line_width"].setValue(int(cfg.get("line_width", 2)))

    def save_cfg(self) -> dict:
        return {"circle_diameter_first": self.v("circle_diameter_first"),
                "circle_diameter_other": self.v("circle_diameter_other"),
                "line_width": self.v("line_width")}

    def build_args(self) -> list:
        return ["s2_visualize.py", "--dataset_dir", self.ds(),
                "--split", self.v("split"),
                "--circle_diameter_first", str(self.v("circle_diameter_first")),
                "--circle_diameter_other", str(self.v("circle_diameter_other")),
                "--line_width", str(self.v("line_width"))]

    # ---------- 目录打开与预览 ----------
    def output_dir_path(self) -> str:
        return str(self.main.cfg.get("visualize_dir", ""))

    def build_preview(self):
        return PreviewPanel("标注可视化预览", refresh_cb=self._collect_preview)

    def _collect_preview(self) -> list:
        d = self.output_dir_path()
        if not d or not Path(d).is_dir():
            return []
        files = sorted(Path(d).rglob("*.jpg"))
        return [(f.name, str(f)) for f in files][:80]

    def on_done(self, ok: bool):
        if ok:
            self.preview.refresh()


class Step3Tab(BaseStepTab):
    """s3 训练：上方左参数 + 右上动画播放器，下方 45% 为全局运行日志"""

    task_label = "s3 训练"
    left_frac = 0.3   # 上部分左右 3:7，参照 s2

    def build_ui(self):
        self.add_readonly("task_type", "任务类型", "")
        # 预训练模型：只读显示模型/权重名称，不需要浏览；留空训练脚本自动下载默认权重
        lab = self._label("预训练模型",
                          "显示模型/权重名称，留空时训练脚本自动下载默认权重")
        edit = QLineEdit("")
        edit.setReadOnly(True)
        edit.setMinimumWidth(CTRL_WIDTH)
        edit.setMaximumWidth(CTRL_WIDTH)
        edit.setPlaceholderText("默认（自动下载）")
        self.fields["model"] = edit
        add_row(self.form, lab, edit)
        self.add_readonly("epochs", "训练轮数", "")
        self.add_readonly("batch", "batch size", "")
        self.add_readonly("imgsz", "输入尺寸", "")
        self.add_readonly("device", "训练设备", "")

    # ---------- 布局：上（左参数 + 右动画） / 下 40% 日志 ----------
    def _layout_body(self):
        v_split = QSplitter(Qt.Orientation.Vertical)
        v_split.setHandleWidth(1)
        v_split.setChildrenCollapsible(False)

        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(1)
        h_split = QSplitter(Qt.Orientation.Horizontal)
        h_split.setHandleWidth(1)
        h_split.setChildrenCollapsible(False)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)     # 参照 s2：参数区贴边
        ll.setSpacing(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        fp = QWidget()
        fp.setLayout(self.form)
        scroll.setWidget(fp)
        ll.addWidget(scroll, 1)
        self._build_actions()
        ll.addLayout(self._actions)
        h_split.addWidget(left)

        self.media = MediaPlayer()
        h_split.addWidget(self.media)
        h_split.setStretchFactor(0, 0)
        h_split.setStretchFactor(1, 1)
        total = 1100
        lw = int(total * max(0.1, min(0.5, self.left_frac)))
        h_split.setSizes([lw, total - lw])
        tl.addWidget(h_split, 1)
        v_split.addWidget(top)

        v_split.addWidget(self.build_log_panel())
        v_split.setSizes([605, 495])   # 上 55% : 下 45%
        self.root.addWidget(v_split, 1)

    def _build_actions(self):
        # 「执行 s3 训练」按钮居中（参照 s2，置于左侧栏内底部）
        self._actions = QHBoxLayout()
        self._actions.addStretch(1)
        self.run_btn = QPushButton(f"▶ 执行 {self.task_label}")
        style_primary_button(self.run_btn)
        self.run_btn.clicked.connect(self.on_run)
        self._actions.addWidget(self.run_btn)
        self._actions.addStretch(1)

    # ---------- 日志区（去掉标题与「最近训练 / 清空」按钮，只留文本框） ----------
    def build_log_panel(self) -> QWidget:
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(2, 2, 2, 2)
        v.setSpacing(1)
        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMaximumBlockCount(5000)
        self.train_log.setStyleSheet("font-family: Menlo, Consolas, monospace;")
        self.train_log.setPlaceholderText(
            "日志区：所有 step 的运行输出都在这里显示")
        v.addWidget(self.train_log, 1)
        return box

    def append_log(self, text: str):
        """主窗口日志统一路由到这里显示。"""
        if text:
            self.train_log.appendPlainText(text.rstrip("\n"))

    def clear_log(self):
        self.train_log.clear()

    # ---------- 最近训练摘要 ----------
    def _latest_train_dir(self) -> Optional[Path]:
        base = str(self.main.cfg.get("run_out_dir", ""))
        if not base or not Path(base).is_dir():
            return None
        runs = sorted((p for p in Path(base).iterdir() if p.is_dir()),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        for r in runs:
            td = r / "train"
            if (td / "results.csv").exists():
                return td
        return None

    def _append_recent_train(self):
        td = self._latest_train_dir()
        if td is None:
            self.append_log("\n[最近训练] 暂无训练日志（run_out 下还没有 results.csv）")
            return
        lines = ["", "----- 最近训练输出 -----",
                 f"训练目录: {td}"]
        args_yaml = td / "args.yaml"
        if args_yaml.exists():
            lines.append("--- args.yaml ---")
            lines.append(args_yaml.read_text(encoding="utf-8",
                                             errors="replace").strip())
        csv = td / "results.csv"
        if csv.exists():
            rows = csv.read_text(encoding="utf-8",
                                 errors="replace").strip().splitlines()
            tail = max(1, min(6, len(rows)))
            lines.append(f"--- results.csv（最近 {tail} 行）---")
            lines.extend(rows[-tail:])
        lines.append("-------------------------")
        self.append_log("\n".join(lines))

    # ---------- 配置 / 参数 ----------
    def load_cfg(self, cfg: dict):
        self.fields["task_type"].setText(str(cfg.get("task_type", "")))
        self.fields["epochs"].setText(str(cfg.get("epochs", "")))
        self.fields["batch"].setText(str(cfg.get("batch", "")))
        self.fields["imgsz"].setText(str(cfg.get("imgsz", "")))
        self.fields["device"].setText(str(cfg.get("device", "")))
        # 预训练模型：显示配置值；为空则显示按任务计算出的默认模型
        model = str(cfg.get("model") or "")
        if not model:
            try:
                model = resolve_model_path(C.task_model_name(cfg.get("task_type")))
            except Exception:
                model = ""
        self.fields["model"].setText(model)

    def save_cfg(self) -> dict:
        return {}

    def build_args(self) -> list:
        args = ["s3_train.py", "--dataset_dir", self.ds()]
        if self.v("model"):
            args += ["--model", self.v("model")]
        return args

    def on_done(self, ok: bool):
        if ok:
            self._append_recent_train()


class Step4Tab(BaseStepTab):
    """s4 推理 + 误差分析：左参数（20%）+ 右图表预览"""

    task_label = "s4 推理 + 误差分析"
    hsplit = True
    left_frac = 0.2

    def build_ui(self):
        self.add_readonly("task_type", "任务类型", "")
        self.add_readonly("model", "使用模型", "")
        self.add_path("input", "推理输入", "", mode="dir",
                      tip="默认显示数据根目录下的验证集图片目录；"
                          "可改为任意图片目录或单张图片路径")
        self.add_dspin("conf", "置信度阈值", C.CONF, 0.0, 1.0)
        self.add_dspin("iou", "IoU 阈值", C.IOU, 0.0, 1.0)
        open_btn = QPushButton("打开推理结果目录")
        open_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_btn.setMaximumWidth(CTRL_WIDTH)   # 与上方编辑框同宽
        open_btn.clicked.connect(lambda: self.main.open_dir(self.result_root()))
        add_row(self.form, "", open_btn)

    def load_cfg(self, cfg: dict):
        self.fields["task_type"].setText(str(cfg.get("task_type", "")))
        weights = cfg.get("weights") or {}
        model = (weights.get("best") or weights.get("last") or "")
        self.fields["model"].setText(str(model))
        # 推理输入：配置值优先；为空则显示默认验证集目录（数据根目录下）
        inp = str(cfg.get("infer_input") or "")
        if not inp:
            try:
                inp = str(Path(self.ds()) / "val" / "images")
            except Exception:
                inp = ""
        self.fields["input"].setText(inp)
        self.fields["conf"].setValue(float(cfg.get("conf", 0.25)))
        self.fields["iou"].setValue(float(cfg.get("iou", 0.45)))

    def save_cfg(self) -> dict:
        return {"infer_input": self.v("input") or None,
                "conf": self.v("conf"), "iou": self.v("iou")}

    def build_args(self) -> list:
        args = ["s4_inference.py", "--dataset_dir", self.ds(),
                "--conf", str(self.v("conf")),
                "--iou", str(self.v("iou"))]
        if self.v("input"):
            args += ["--input", self.v("input")]
        return args

    # ---------- 目录打开与预览 ----------
    def result_root(self) -> str:
        root = Path(str(self.main.cfg.get("data_root", "")))
        return str(root / "推理结果")

    def build_preview(self):
        return PreviewPanel("推理结果预览（误差图表 / 推理可视化）",
                            refresh_cb=self._collect_preview)

    def _collect_preview(self) -> list:
        root = Path(str(self.main.cfg.get("data_root", "")))
        err = root / "推理结果" / "误差分析"
        vis = root / "推理结果" / "推理可视化"
        out = []
        if err.is_dir():
            out += [(f"[误差] {f.name}", str(f))
                    for f in sorted(err.glob("*.png"))]
        if vis.is_dir():
            out += [(f"[可视化] {f.name}", str(f))
                    for f in sorted(vis.glob("*.jpg"))]
        return out[:100]

    def on_done(self, ok: bool):
        if ok:
            self.preview.refresh()


class Step5Tab(BaseStepTab):
    """s5 转换导出：只读展示，imgsz / TensorRT 等配置在【项目设置】页统一管理"""

    task_label = "s5 转换导出"

    def build_ui(self):
        # 无 NVIDIA GPU 或无 TensorRT 时，TensorRT 相关项不可用，整行灰掉
        self._trt_ok = HAS_CUDA and HAS_TENSORRT
        self.add_readonly("task_type", "任务类型", "")
        self.add_readonly("model", "转换模型", "")
        self.add_readonly("output_dir", "输出目录", "")
        # 输出文件名称：可编辑，留空使用默认 yolo_<任务>_detector
        self.add_text("output_name", "输出文件名称", "",
                      placeholder="yolo_detector（留空用默认）")
        self.fields["output_name"].textChanged.connect(self._auto_save)
        self.add_readonly("imgsz", "输入尺寸 imgsz", "")
        self.add_readonly("trt", "TensorRT 导出",
                          "不可用" if not self._trt_ok else "",
                          enabled=self._trt_ok)
        self.add_readonly("trt_lib", "trtexec 路径", "",
                          enabled=self._trt_ok)

    def load_cfg(self, cfg: dict):
        self.fields["task_type"].setText(str(cfg.get("task_type", "")))
        weights = cfg.get("weights") or {}
        model = (weights.get("best") or weights.get("last") or "")
        self.fields["model"].setText(str(model))
        self.fields["output_dir"].setText(str(cfg.get("weights_dir", "")))
        task = str(cfg.get("task_type", "detect")).lower()
        self.fields["output_name"].setText(
            str(cfg.get("output_name") or f"yolo_{task}_detector"))
        self.fields["imgsz"].setText(str(cfg.get("imgsz", 640)))
        if self._trt_ok:
            trt_on = bool(cfg.get("export_trt"))
            self.fields["trt"].setText("是（同时导出 .engine）" if trt_on
                                       else "否（仅 ONNX）")
            self.fields["trt_lib"].setText(
                str(cfg.get("trt_lib") or getattr(C, "TENSORRT_LIB", "")))
        else:
            self.fields["trt"].setText("不可用（未检测到 NVIDIA GPU / TensorRT）")
            self.fields["trt_lib"].setText("")

    def save_cfg(self) -> dict:
        return {"output_name": self.v("output_name").strip() or None}

    def build_args(self) -> list:
        args = ["s5_convert.py", "--dataset_dir", self.ds()]
        name = self.v("output_name").strip()
        if name and name != f"yolo_{str(self.v('task_type')).lower()}_detector":
            args += ["--output_name", name]
        if self.main.cfg.get("export_trt"):
            args += ["--trt"]
        return args

    def _build_actions(self):
        # 「执行 s5 转换导出」按钮居中（与其他 step 页一致）
        self._actions = QHBoxLayout()
        self._actions.addStretch(1)
        self.run_btn = QPushButton(f"▶ 执行 {self.task_label}")
        style_primary_button(self.run_btn)
        self.run_btn.clicked.connect(self.on_run)
        self._actions.addWidget(self.run_btn)
        self._actions.addStretch(1)
