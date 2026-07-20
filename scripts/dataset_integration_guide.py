#!/usr/bin/env python3
"""
数据集在 wafer-inspection 中的使用脚本

本数据集为 480×320 灰度分类图像（1150张，105类缺陷）。
由于 wafer-inspection 模型是检测模型（需要边界框），本数据集的用途包括：

1. 编码器无监督预训练
   - 用全部 1150 张图像训练编码器（MAE/MIM 风格自监督）
   - 帮助编码器学习晶圆图像的通用特征表示

2. Clark 特征提取 + t-SNE 可视化
   - 用预训练编码器提取每张图像的特征
   - t-SNE 投影到 2D，按真实 DEFECT_ID 着色
   - 可视化编码器的类别区分能力

3. 分类微调评估
   - 在编码器后加线性分类头，按 DEFECT_ID 分类
   - 评估编码器的表征质量（可作为下游任务的参考基线）
"""

import os
import sys
import csv
import random

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ===== 配置 =====
DATASET_DIR = r"D:\cy\images"
LABEL_CSV = os.path.join(DATASET_DIR, "label.csv")
OUTPUT_DIR = r"D:\cy\wafer-inspection\deploy\dataset_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 32
IMAGE_SIZE = 224  # 输入尺寸（训练时需要 Resize）


class WaferDataset(Dataset):
    """晶圆图像分类数据集"""

    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []

        with open(LABEL_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["IMAGE_NAME"]
                img_path = os.path.join(DATASET_DIR, name)
                if os.path.exists(img_path):
                    self.samples.append((img_path, int(row["DEFECT_ID"])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")  # 灰度图
        if self.transform:
            img = self.transform(img)
        return img, label, os.path.basename(img_path)


def extract_features(model, device="cpu"):
    """用 wafter-inspection 编码器提取特征"""
    dataset = WaferDataset(
        transform=lambda img: torch.tensor(
            np.array(img.resize((IMAGE_SIZE, IMAGE_SIZE)), dtype=np.float32)
        ).unsqueeze(0) / 255.0
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    features = []
    labels = []
    names = []

    with torch.no_grad():
        for imgs, lbls, nms in loader:
            # imgs: [B, 1, H, W]
            # 这里用编码器的前向传播提取特征
            # 具体取决于 wafer-inspection 模型的接口
            # 示例：model.encode(imgs.to(device))
            # feat = model.encode(imgs.to(device))
            # features.append(feat.cpu().numpy())

            labels.extend(lbls.numpy())
            names.extend(nms)

    return np.concatenate(features) if features else None, np.array(labels), names


def visualize_tsne(features, labels, save_path):
    """t-SNE 可视化特征分布"""
    if features is None or len(features) < 10:
        print("Not enough features for t-SNE")
        return

    print(f"Running t-SNE on {features.shape}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(features) - 1))
    xy = tsne.fit_transform(features)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(xy[:, 0], xy[:, 1], c=labels, cmap="tab20", alpha=0.7, s=10)
    plt.colorbar(scatter, label="Defect Class ID")
    plt.title("t-SNE of Wafer Image Features")
    plt.xlabel("t-SNE dim 1")
    plt.ylabel("t-SNE dim 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"t-SNE plot saved to {save_path}")


def main():
    print("=" * 60)
    print("Wafer Inspection - Dataset Integration Script")
    print("=" * 60)

    dataset = WaferDataset()
    print(f"Dataset loaded: {len(dataset)} images")

    # 统计类别分布
    from collections import Counter
    counts = Counter([s[1] for s in dataset.samples])
    print(f"Total classes: {len(counts)}")
    print(f"Top 10 classes: {counts.most_common(10)}")

    # 使用说明
    print("\n" + "=" * 60)
    print("USAGE NOTES")
    print("=" * 60)
    print("""
1. For encoder pretraining:
   from wafer_inspection.models import WaferMultitask
   model = WaferMultitask()

   # Unsupervised pretraining on all 1150 images
   python train_stage1.py --unlabeled_data D:/cy/images/

2. For feature extraction (after pretraining):
   python deploy/dataset_output/extract_features.py --checkpoint runs/best.pth

3. For linear probe evaluation:
   Add a classification head on top of encoder:
       model.classifier = nn.Linear(448, 105)  # 105 classes
       Train only the classifier head (freeze encoder)
    """)


if __name__ == "__main__":
    main()
