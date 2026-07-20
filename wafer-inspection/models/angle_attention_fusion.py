"""
角度注意力融合层

用物理先验（散射增益置信度）+ 学习注意力混合策略，
融合 13 个角度的 193nm 检测分支输出。

融合策略:
  w = 0.7 * 物理置信度 + 0.3 * 学习注意力
  物理置信度: 各角度 PISM 增益图的可靠性
  学习注意力: Conv 网络从预测中学习的跨角度模式
"""

import torch
import torch.nn as nn


class AngleAttentionFusion(nn.Module):
    """
    角度注意力融合层

    Args:
        num_classes: 缺陷类别数
        num_angles: 角度数
        in_channels: 每角度预测通道数 = num_classes + 4(reg)
        gamma: 物理先验混合权重 (0.7 = 偏重物理)
    """

    def __init__(self, num_classes=4, num_angles=13, in_channels=None, gamma=0.7):
        super().__init__()
        in_channels = in_channels or (num_classes + 4)
        self.num_angles = num_angles
        self.gamma = gamma

        # 学习注意力: 从所有角度的预测中学习跨角度权重
        self.angle_attention = nn.Sequential(
            nn.Conv2d(in_channels * num_angles, 64, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_angles, 1, bias=True),
            nn.Softmax(dim=1)  # 各角度权重归一化
        )

    def forward(self, angle_preds, angle_conf, gain_map=None):
        """
        Args:
            angle_preds: list of [B, C, H, W] 每个角度的预测
            angle_conf: [B, num_angles, H, W] 每个角度的置信度图
            gain_map: [B, 1, H, W] PISM 增益图（可选，用于物理修正）

        Returns:
            fused: [B, C, H, W] 融合后的预测
            diagnostics: 诊断信息
        """
        B, C, H, W = angle_preds[0].shape
        num_angles = len(angle_preds)

        # Step 1: 堆叠所有角度预测
        stacked = torch.stack(angle_preds, dim=1)  # [B, N, C, H, W]

        # Step 2: 物理先验权重
        conf_norm = angle_conf / (angle_conf.sum(dim=1, keepdim=True) + 1e-8)
        phys_weights = conf_norm.unsqueeze(2)  # [B, N, 1, H, W]

        # Step 3: 学习注意力权重
        stacked_flat = stacked.reshape(B, C * num_angles, H, W)
        learned_weights = self.angle_attention(stacked_flat)  # [B, N, H, W]
        learned_weights = learned_weights.unsqueeze(2)  # [B, N, 1, H, W]

        # Step 4: 混合权重
        weights = (self.gamma * phys_weights +
                   (1 - self.gamma) * learned_weights)
        # 重新归一化
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # Step 5: 加权融合
        fused = (stacked * weights).sum(dim=1)  # [B, C, H, W]

        return fused, {'weights': weights.detach()}
