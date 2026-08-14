"""
全局主题：浅色 / 深色。

所有界面颜色集中在这里管理。切换主题只需重新设置 QPalette + 全局 QSS，
控件级不再硬编码颜色（浏览按钮、只读框、提示文字、预览框等统一由全局 QSS
按 objectName / 伪状态控制），保证深色下编辑框、下拉框、图片预览框全部变深。
"""
from PySide6.QtGui import QColor, QPalette

LIGHT = {
    "window": "#f2f3f5",       # 窗口 / 页面背景
    "panel": "#ffffff",        # 面板（GroupBox / Tab 内容区）
    "panel_alt": "#eceef1",    # 只读框 / 悬浮 / 表头
    "border": "#c9c9c9",       # 边框
    "text": "#24292f",         # 正文
    "text_sub": "#6a7078",     # 提示 / 次要文字
    "input_bg": "#ffffff",     # 编辑框 / 下拉框 / 表格背景
    "tab_active": "#ffffff",
    "tab_inactive": "#e4e7eb",
    "tab_text_active": "#2f6fed",
    "table_alt": "#f4f6f8",
    "scroll": "#c0c5cc",
    "scroll_hover": "#a5abb4",
}

DARK = {
    "window": "#1e2124",
    "panel": "#2b2f33",
    "panel_alt": "#3a3f45",
    "border": "#454b52",
    "text": "#e6e8eb",
    "text_sub": "#9aa0aa",
    "input_bg": "#191c1f",
    "tab_active": "#2b2f33",
    "tab_inactive": "#23272b",
    "tab_text_active": "#6ea1ff",
    "table_alt": "#32373c",
    "scroll": "#565c63",
    "scroll_hover": "#6b7279",
}

_THEMES = {"light": LIGHT, "dark": DARK}

current = "light"      # 全局当前主题名（light / dark）


def normalize(name: str = None) -> str:
    """把各种写法统一成 light / dark（兼容中文项“浅色/深色”）"""
    name = str(name or "").strip().lower()
    if name in ("dark", "深色"):
        return "dark"
    return "light"


def get(name: str = None) -> dict:
    return _THEMES.get(normalize(name), LIGHT)


# ---------------- 全局 QSS（用 ${key} 占位符，主题替换） ----------------
_GLOBAL_QSS = """
QMainWindow, QDialog { background: ${window}; }
QWidget { background: transparent; }

QTabWidget::pane { border: 1px solid ${border}; background: ${panel}; top: -1px; }
QTabBar::tab {
    background: ${tab_inactive}; color: ${text};
    padding: 6px 16px; margin-right: 2px;
    border: 1px solid ${border}; border-bottom: none;
    border-top-left-radius: 4px; border-top-right-radius: 4px;
}
QTabBar::tab:selected { background: ${tab_active}; color: ${tab_text_active}; font-weight: bold; }
QTabBar::tab:hover:!selected { background: ${panel_alt}; }

QGroupBox {
    border: 1px solid ${border}; border-radius: 6px;
    margin-top: 10px; background: ${panel};
}
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: ${text}; }

QLabel { color: ${text}; background: transparent; }
QLabel#hint { color: ${text_sub}; }

/* 编辑框 / 下拉框 / 输入控件：统一稍高 + 圆角 */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {
    background: ${input_bg}; color: ${text}; border: 1px solid ${border};
    border-radius: 4px; padding: 3px 6px; min-height: 22px;
    selection-background-color: #2f6fed; selection-color: #ffffff;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QTextEdit:focus { border: 1px solid #2f6fed; }
QLineEdit:read-only { background: ${panel_alt}; color: ${text_sub}; }
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled,
QComboBox:disabled { color: ${text_sub}; background: ${panel_alt}; }

QComboBox::drop-down { border: none; width: 20px; }
QComboBox::down-arrow {
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid ${text_sub}; margin-right: 6px;
}
QComboBox QAbstractItemView {
    background: ${panel}; color: ${text}; border: 1px solid ${border};
    selection-background-color: #2f6fed; selection-color: #ffffff; outline: none;
}

QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border; right: 2px; width: 16px;
    border-left: 1px solid ${border}; background: ${panel};
    border-top-right-radius: 3px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border; right: 2px; bottom: 2px; width: 16px;
    border-left: 1px solid ${border}; background: ${panel};
    border-bottom-right-radius: 3px;
}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-bottom: 5px solid ${text_sub};
}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid ${text_sub};
}

QPushButton {
    background: ${panel}; color: ${text}; border: 1px solid ${border};
    border-radius: 4px; padding: 3px 10px; min-height: 20px;
}
QPushButton:hover { background: ${panel_alt}; }
QPushButton:pressed { background: ${scroll}; }
QPushButton:disabled { color: ${text_sub}; background: ${panel_alt}; border-color: ${border}; }

QCheckBox, QRadioButton { color: ${text}; spacing: 6px; background: transparent; }
QCheckBox::indicator, QRadioButton::indicator { width: 15px; height: 15px; }

QTableWidget, QTableView, QTreeView, QListView {
    background: ${input_bg}; color: ${text}; border: 1px solid ${border};
    gridline-color: ${border}; alternate-background-color: ${table_alt};
    selection-background-color: #2f6fed; selection-color: #ffffff;
}
QHeaderView::section {
    background: ${panel_alt}; color: ${text}; border: none;
    border-right: 1px solid ${border}; border-bottom: 1px solid ${border};
    padding: 4px 6px;
}
QTableCornerButton::section { background: ${panel_alt}; border: none; }

QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: ${scroll}; border-radius: 5px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: ${scroll_hover}; }
QScrollBar:horizontal { background: transparent; height: 10px; margin: 0; }
QScrollBar::handle:horizontal { background: ${scroll}; border-radius: 5px; min-width: 24px; }
QScrollBar::handle:horizontal:hover { background: ${scroll_hover}; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }

QSplitter::handle { background: ${border}; }
QToolTip { background: ${panel_alt}; color: ${text}; border: 1px solid ${border}; }
QStatusBar { background: ${window}; color: ${text_sub}; }
QMenuBar { background: ${window}; color: ${text}; }
QMenu { background: ${panel}; color: ${text}; border: 1px solid ${border}; }
QMenu::item:selected { background: #2f6fed; color: #ffffff; }

/* 图片 / 动画预览面板（PreviewPanel.img、MediaPlayer.placeholder） */
#previewBox { border: 1px dashed ${border}; background: ${panel_alt}; color: ${text_sub}; }
"""


def qss(name: str = None) -> str:
    """生成指定主题的全局样式表"""
    t = get(name)
    s = _GLOBAL_QSS
    for k, v in t.items():
        s = s.replace("${%s}" % k, v)
    return s


def _palette(name: str = None) -> QPalette:
    t = get(name)
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(t["window"]))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(t["text"]))
    pal.setColor(QPalette.ColorRole.Base, QColor(t["input_bg"]))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(t["table_alt"]))
    pal.setColor(QPalette.ColorRole.Text, QColor(t["text"]))
    pal.setColor(QPalette.ColorRole.Button, QColor(t["panel"]))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(t["text"]))
    pal.setColor(QPalette.ColorRole.BrightText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.Highlight, QColor("#2f6fed"))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(t["panel_alt"]))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(t["text"]))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(t["text_sub"]))
    return pal


def apply_theme(app, name: str = None):
    """把主题应用到整个应用（QPalette + 全局 QSS）"""
    global current
    name = normalize(name)
    current = name
    if app is None:
        return
    app.setPalette(_palette(name))
    app.setStyleSheet(qss(name))
