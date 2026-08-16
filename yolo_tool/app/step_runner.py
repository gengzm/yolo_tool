"""
子进程执行器：在后台 QThread 中运行各 step 脚本，
stdout/stderr 实时通过信号转发到界面日志区。
"""
import subprocess
import sys

from PySide6.QtCore import QThread, Signal


class StepRunner(QThread):
    """执行一条 CLI 命令并实时回传输出"""

    log = Signal(str)          # 每行输出
    done = Signal(bool, int)   # (成功与否, returncode)

    def __init__(self, args: list, cwd: str = None,
                 python: str = None, env_extra: dict = None):
        super().__init__()
        self.args = list(args)
        self.cwd = cwd
        # 默认使用启动界面时同一个解释器，保证 cv2/ultralytics 依赖一致
        self.python = python or sys.executable
        self.env_extra = env_extra or {}

    def run(self):
        cmd = [self.python] + self.args
        self.log.emit(f"$ {' '.join(cmd)}\n")
        try:
            import os
            env = dict(os.environ)
            env.update(self.env_extra)
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=self.cwd,
                env=env,
                text=True,
                bufsize=1,
                encoding="utf-8",
                errors="replace",
            )
            for line in iter(proc.stdout.readline, ""):
                self.log.emit(line)
            proc.stdout.close()
            proc.wait()
            self.done.emit(proc.returncode == 0, proc.returncode)
        except Exception as e:
            self.log.emit(f"[ERROR] 启动子进程失败: {e}\n")
            self.done.emit(False, -1)
