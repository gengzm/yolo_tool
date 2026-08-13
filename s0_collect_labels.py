#!/usr/bin/env python
"""
Step 0: 收集标签信息
扫描源目录中所有 LabelMe JSON，统计各类别出现次数，展示类别清单。

用法:
    python s0_collect_labels.py                # 使用 config.py 的源目录
    python s0_collect_labels.py --source_dir /path/to/dir
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import config
from utils import load_labelme_json, log_info, log_warn, log_error, get_project_config


def collect_label_stats(source_dir: str):
    """扫描 LabelMe JSON，返回 (类别名列表, label->出现次数, label->涉及图片数)"""
    source = Path(source_dir)
    if not source.is_dir():
        log_error(f"Source dir not found: {source_dir}")
        sys.exit(1)

    json_files = sorted(source.glob(f"*{config.LABELME_SUFFIX}"))
    if not json_files:
        log_error(f"No LabelMe JSON files (*{config.LABELME_SUFFIX}) found in {source_dir}")
        sys.exit(1)

    total_count = Counter()      # label -> shape 出现总次数
    image_count = Counter()      # label -> 涉及的图片数
    for json_path in json_files:
        try:
            data = load_labelme_json(str(json_path))
        except Exception as e:
            log_warn(f"Failed to read {json_path.name}: {e}")
            continue
        seen = set()
        for shape in data.get("shapes", []):
            label = shape.get("label", "").strip()
            if not label:
                continue
            total_count[label] += 1
            seen.add(label)
        for label in seen:
            image_count[label] += 1

    return sorted(total_count.keys()), total_count, image_count


def main():
    parser = argparse.ArgumentParser(description="收集 LabelMe JSON 中的标签信息")
    parser.add_argument("--dataset_dir", type=str, default=None,
                        help="YOLO 数据集目录（用于定位项目 info.yaml，默认取 config）")
    parser.add_argument("--source_dir", type=str, default=None,
                        help="原始图片和 LabelMe JSON 所在目录（默认取项目 info.yaml / config）")
    args = parser.parse_args()

    # 项目配置解析: 命令行参数 > info.yaml 记录 > config 默认值
    cfg = get_project_config(args.dataset_dir)
    source_dir = args.source_dir or cfg["source_dir"]

    log_info(f"Source dir: {source_dir}")
    names, total_count, image_count = collect_label_stats(source_dir)

    print()
    print("=" * 60)
    print(f"共收集到 {len(names)} 个类别:")
    print(f"{'类别ID':<8}{'标签名':<20}{'标注数':<10}{'涉及图片数'}")
    print("-" * 60)
    for i, name in enumerate(names):
        print(f"{i:<8}{name:<20}{total_count[name]:<10}{image_count[name]}")
    print("=" * 60)

    if names:
        log_info(f"类别清单: {names}")
        log_info("转换时 (step1) 会自动使用以上类别清单，无需手动配置。")
    else:
        log_error("没有收集到任何 label，请检查 JSON 的 shapes[].label 字段。")


if __name__ == "__main__":
    main()
