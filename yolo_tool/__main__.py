"""
启动 YOLO Tool 工作台: python -m yolo_tool
"""
import sys
from pathlib import Path

# 项目根目录（含 config.py / utils.py / app/ / s*.py）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.main import main

if __name__ == "__main__":
    main()
