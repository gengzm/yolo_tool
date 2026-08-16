"""`python -m yolo_tool` 入口，等价于 `yolo-tool` 命令。"""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
