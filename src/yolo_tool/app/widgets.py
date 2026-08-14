"""
公共控件：路径输入行、图片预览面板、GIF/视频播放器、参数表单辅助。
"""
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QMovie, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

# QtMultimedia 可选（部分精简环境未安装），缺失时仅支持 GIF
try:
    from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
    from PySide6.QtMultimediaWidgets import QVideoWidget
    _MEDIA_AVAILABLE = True
except Exception:
    _MEDIA_AVAILABLE = False


def browse_path(edit: QLineEdit, mode: str = "dir", file_filter: str = ""):
    """打开系统选择框并回填到 QLineEdit"""
    current = edit.text().strip()
    if mode == "file":
        path, _ = QFileDialog.getOpenFileName(
            edit, "选择文件", current or ".", file_filter or "所有文件 (*)")
    elif mode == "save_file":
        path, _ = QFileDialog.getSaveFileName(
            edit, "保存到", current or ".", file_filter or "所有文件 (*)")
    else:
        path = QFileDialog.getExistingDirectory(edit, "选择目录", current or ".")
    if path:
        edit.setText(path)


def path_row(value: str = "", mode: str = "dir", file_filter: str = "",
             placeholder: str = "") -> tuple:
    """
    生成一行 [输入框 | 浏览] 控件。
    mode: dir | file | save_file | none
    返回 (container_widget, line_edit)
    """
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(1)
    edit = QLineEdit(value)
    edit.setMinimumWidth(110)
    edit.setMaximumWidth(420)   # 路径框不要太宽，整体靠左，减少右侧空白
    if placeholder:
        edit.setPlaceholderText(placeholder)
    lay.addWidget(edit, 1)
    if mode != "none":
        btn = QPushButton("浏览")
        btn.setMaximumWidth(72)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.clicked.connect(lambda: browse_path(edit, mode, file_filter))
        lay.addWidget(btn)
    return w, edit


def add_row(form: QFormLayout, label: str, widget: QWidget) -> None:
    """向表单添加一行"""
    form.addRow(label, widget)


def make_form() -> QFormLayout:
    """创建统一表单布局（紧凑：垂直间距 1px、无内边距）"""
    form = QFormLayout()
    form.setContentsMargins(0, 0, 0, 0)
    form.setVerticalSpacing(1)
    form.setHorizontalSpacing(6)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return form


def style_primary_button(btn: QPushButton, big: bool = True) -> QPushButton:
    """统一的「执行当前页」主按钮样式：更大、圆角、主题色。"""
    if big:
        btn.setMinimumWidth(180)
        btn.setMinimumHeight(40)
    btn.setCursor(Qt.CursorShape.PointingHandCursor)
    btn.setStyleSheet("""
        QPushButton {
            background: #2f6fed; color: white; border: none;
            border-radius: 8px; font-size: 14px; font-weight: bold;
            padding: 4px 18px;
        }
        QPushButton:hover { background: #3b7bf5; }
        QPushButton:pressed { background: #2758c4; }
        QPushButton:disabled { background: #9db8ea; }
    """)
    return btn


class PreviewPanel(QWidget):
    """
    图片预览面板：下拉列出目录图片 + 自适应缩放显示。
    refresh_cb: 无参回调，返回 list of (label, path) 或 list of str
    """

    def __init__(self, title: str = "图片预览", refresh_cb=None):
        super().__init__()
        self.refresh_cb = refresh_cb
        self._files = []
        self._raw = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(1)

        head = QHBoxLayout()
        head.setSpacing(1)
        head.addWidget(QLabel(title))
        head.addStretch(1)
        lay.addLayout(head)

        self.img = QLabel("暂无图片")
        self.img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img.setMinimumHeight(300)
        self.img.setObjectName("previewBox")   # 边框/背景由全局主题控制
        lay.addWidget(self.img, 1)

        # 底部操作条：翻页 / 选择 / 刷新
        bottom = QHBoxLayout()
        bottom.setSpacing(1)
        self.prev_btn = QPushButton("◀")
        self.prev_btn.setMaximumWidth(40)
        self.prev_btn.clicked.connect(lambda: self._step(-1))
        bottom.addWidget(self.prev_btn)
        self.combo = QComboBox()
        self.combo.setMaximumWidth(520)
        self.combo.currentIndexChanged.connect(lambda _: self._show())
        bottom.addWidget(self.combo, 1)
        self.next_btn = QPushButton("▶")
        self.next_btn.setMaximumWidth(40)
        self.next_btn.clicked.connect(lambda: self._step(1))
        bottom.addWidget(self.next_btn)
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.refresh)
        bottom.addWidget(self.refresh_btn)
        lay.addLayout(bottom)

    # ---------- 对外接口 ----------
    def refresh(self):
        if self.refresh_cb:
            try:
                self.set_files(self.refresh_cb() or [])
            except Exception as e:
                self.img.setText(f"刷新失败: {e}")

    def set_files(self, files):
        """files: list of (label, path) 或 list of str"""
        self._files = [f if isinstance(f, tuple) else (Path(str(f)).name, str(f))
                       for f in files]
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItems([label for label, _ in self._files])
        self.combo.blockSignals(False)
        self._show()

    # ---------- 内部 ----------
    def _step(self, delta: int):
        i = self.combo.currentIndex() + delta
        if 0 <= i < self.combo.count():
            self.combo.setCurrentIndex(i)

    def _show(self):
        idx = self.combo.currentIndex()
        if not self._files or idx < 0 or idx >= len(self._files):
            self._raw = None
            self.img.setText("暂无图片")
            return
        _, path = self._files[idx]
        pm = QPixmap(path)
        if pm.isNull():
            self._raw = None
            self.img.setText(f"无法加载:\n{path}")
            return
        self._raw = pm
        self._update_pixmap()

    def _update_pixmap(self):
        if self._raw is None:
            return
        w, h = self.img.width(), self.img.height()
        if w > 1 and h > 1:
            self.img.setPixmap(self._raw.scaled(
                w, h, Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()


class MediaPlayer(QWidget):
    """GIF / 视频播放器：选择文件后加载播放，支持播放 / 暂停。"""

    def __init__(self, title: str = "动画 / 视频播放器"):
        super().__init__()
        self._movie = None
        self._player = None
        self._audio = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(1)

        head = QHBoxLayout()
        head.setSpacing(1)
        head.addWidget(QLabel(title))
        head.addStretch(1)
        lay.addLayout(head)

        self.stack = QStackedWidget()
        self.placeholder = QLabel("请选择 GIF 或视频文件播放")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setMinimumHeight(180)
        self.placeholder.setWordWrap(True)
        self.placeholder.setObjectName("previewBox")   # 边框/背景由全局主题控制
        self.stack.addWidget(self.placeholder)
        if _MEDIA_AVAILABLE:
            self.video_widget = QVideoWidget()
            self.video_widget.setStyleSheet("background: #000000;")
            self.stack.addWidget(self.video_widget)
        else:
            self.video_widget = None
        lay.addWidget(self.stack, 1)

        row = QHBoxLayout()
        row.setSpacing(1)
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("动画文件路径（.gif / .mp4 / .avi / .mov 等）")
        row.addWidget(self.path_edit, 1)
        browse_btn = QPushButton("浏览")
        browse_btn.setMaximumWidth(64)
        browse_btn.clicked.connect(self._browse)
        row.addWidget(browse_btn)
        load_btn = QPushButton("加载播放")
        load_btn.setMaximumWidth(96)
        load_btn.clicked.connect(self.load)
        row.addWidget(load_btn)
        self.pause_btn = QPushButton("暂停")
        self.pause_btn.setMaximumWidth(70)
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.toggle_pause)
        row.addWidget(self.pause_btn)
        lay.addLayout(row)

    # ---------- 对外接口 ----------
    def load(self):
        """按当前输入框路径加载并播放。"""
        p = self.path_edit.text().strip()
        if not p:
            return
        self._stop_all()
        if Path(p).suffix.lower() == ".gif":
            self._load_gif(p)
        else:
            self._load_video(p)

    def load_path(self, p: str):
        """直接按给定路径加载并播放。"""
        self.path_edit.setText(p)
        self.load()

    def stop(self):
        self._stop_all()

    def toggle_pause(self):
        if self._movie is not None:
            if self._movie.state() == QMovie.MovieState.Running:
                self._movie.setPaused(True)
                self.pause_btn.setText("继续")
            else:
                self._movie.setPaused(False)
                self.pause_btn.setText("暂停")
        elif self._player is not None:
            if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self._player.pause()
                self.pause_btn.setText("继续")
            else:
                self._player.play()
                self.pause_btn.setText("暂停")

    # ---------- 内部实现 ----------
    def _browse(self):
        d, _ = QFileDialog.getOpenFileName(
            self, "选择动画文件", str(Path.home()),
            "动画 / 视频 (*.gif *.mp4 *.avi *.mov *.mkv *.webm);;所有文件 (*)")
        if d:
            self.load_path(d)

    def _stop_all(self):
        if self._movie is not None:
            self._movie.stop()
            self._movie = None
        if self._player is not None:
            self._player.stop()
            self._player = None
        if self._audio is not None:
            self._audio = None
        self.pause_btn.setText("暂停")
        self.pause_btn.setEnabled(False)

    def _load_gif(self, p):
        try:
            self._movie = QMovie(p)
        except Exception:
            self.placeholder.setText("GIF 加载失败：%s" % p)
            self.stack.setCurrentIndex(0)
            return
        self._movie.setScaledSize(self.placeholder.size())
        self.placeholder.setMovie(self._movie)
        self.stack.setCurrentIndex(0)
        self._movie.start()
        self.pause_btn.setEnabled(True)

    def _load_video(self, p):
        if not _MEDIA_AVAILABLE or self.video_widget is None:
            self.placeholder.setText("当前环境缺少 QtMultimedia，仅支持 GIF 播放")
            self.stack.setCurrentIndex(0)
            return
        try:
            self._audio = QAudioOutput()
            self._audio.setVolume(0.5)
            self._player = QMediaPlayer()
            self._player.setAudioOutput(self._audio)
            self._player.setVideoOutput(self.video_widget)
            self._player.setSource(QUrl.fromLocalFile(p))
            self.stack.setCurrentIndex(1)
            self._player.play()
            self.pause_btn.setEnabled(True)
        except Exception as e:
            self.placeholder.setText("视频加载失败：%s" % e)
            self.stack.setCurrentIndex(0)
            self._player = None
            self._audio = None
