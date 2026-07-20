#!/usr/bin/env python3
"""
数据集导入脚本 — 将 D:\cy\images 数据集导入 wafer-backend gallery

用法:
  1. 确保 wafer-backend 运行中 (port 8080)
  2. python scripts/import_dataset_to_gallery.py

效果:
  - 将 1150 张图像上传至 MinIO
  - 在 gallery_item 表中创建对应记录
  - 可按 DEFECT_ID 分类筛选
"""

import csv
import os
import sys
import requests
import json
from pathlib import Path

DATASET_DIR = r"D:\cy\images"
LABEL_CSV = os.path.join(DATASET_DIR, "label.csv")
API_BASE = "http://localhost:8080/api/v1"

def import_dataset():
    # 读取标注
    with open(LABEL_CSV, "r") as f:
        reader = csv.DictReader(f)
        labels = {row["IMAGE_NAME"]: int(row["DEFECT_ID"]) for row in reader}

    images = sorted([
        f for f in os.listdir(DATASET_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and f != "label.csv"
    ])

    print(f"Total images: {len(images)}, labeled: {len(labels)}")

    # 按 DEFECT_ID 分组统计
    from collections import Counter
    counts = Counter(labels.values())
    print(f"Defect classes: {len(counts)}")
    for cid in sorted(counts.keys()):
        print(f"  Class {cid}: {counts[cid]} images")

    success = 0
    failed = 0

    for i, img_name in enumerate(images):
        img_path = os.path.join(DATASET_DIR, img_name)
        defect_id = labels.get(img_name, 0)

        # 通过后端上传 API
        with open(img_path, "rb") as f:
            files = {"file": (img_name, f, "image/jpeg")}
            data = {
                "category": f"defect_{defect_id}",
                "description": f"Dataset image, defect class {defect_id}",
                "tags": json.dumps(["dataset", f"class_{defect_id}"]),
            }

            try:
                resp = requests.post(
                    f"{API_BASE}/images/upload",
                    files=files,
                    data=data,
                    timeout=30,
                )
                if resp.status_code == 200:
                    success += 1
                else:
                    failed += 1
                    print(f"  [{i+1}/{len(images)}] {img_name} -> FAILED ({resp.status_code})")
            except Exception as e:
                failed += 1
                print(f"  [{i+1}/{len(images)}] {img_name} -> ERROR: {e}")

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(images)} (success: {success}, failed: {failed})")

    print(f"\nImport complete: {success} success, {failed} failed")

if __name__ == "__main__":
    import_dataset()
