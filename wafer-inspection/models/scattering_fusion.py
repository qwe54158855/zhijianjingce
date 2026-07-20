"""
SGF: Scattering-Guided Fusion 散射引导融合层

用 PISM 的物理诊断图（散射增益 + 散射尺度）作为注意力先验，
指导 266nm 和 193nm 双分支的融合权重。

物理先验:
  高增益区域 (gain_map↑) → 193nm 散射增强显著 → 更信赖 193nm 分支
  低增益区域 (gain_map→1) → 无散射增强 → 平衡融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScatteringGuidedFusion(nn.Module):
    """
    散射引导融合层

    Args:
        num_classes: 缺陷类别数
        in_channels: 预测通道数 = num_classes (cls) + 4 (reg)
    """

    def __init__(self, num_classes=4, in_channels=None):
        super().__init__()
        in_channels = in_channels or (num_classes + 4)

        # 物理先验权重生成器
        self.weight_predictor = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 3, padding=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()  # 输出: w_193 ∈ [0,1]
        )

        # 可学习每通道偏置
        # 允许不同缺陷类别有不同的融合倾向
        self.channel_bias = nn.Parameter(torch.zeros(1, in_channels, 1, 1))

    def forward(self, pred_266, pred_193, gain_map, scale_map):
        """
        Args:
            pred_266: 266nm 分支预测 [B, C, H, W]
            pred_193: 193nm 分支预测 [B, C, H, W]
            gain_map: PISM 散射增益图 [B, 1, H, W]
            scale_map: PISM 散射尺度图 [B, 1, H, W]

        Returns:
            fused: 融合后预测 [B, C, H, W]
            diagnostics: 诊断信息
        """
        # Step 1: 对齐空间尺寸
        target_size = pred_266.shape[-2:]
        gain = F.interpolate(gain_map, size=target_size, mode='bilinear', align_corners=False)
        scale = F.interpolate(scale_map, size=target_size, mode='bilinear', align_corners=False)

        # Step 2: 物理先验生成融合权重
        phys = torch.cat([gain, scale], dim=1)  # [B, 2, H, W]
        w_193_base = self.weight_predictor(phys)  # [B, 1, H, W]

        # Step 3: 通道级偏置微调
        # sigmoid 确保偏置在 [0,1] 范围
        bias = torch.sigmoid(self.channel_bias)  # [1, C, 1, 1]
        w_193 = (w_193_base + bias).clamp(0, 1)
        w_266 = 1.0 - w_193

        # Step 4: 加权融合
        fused = w_266 * pred_266 + w_193 * pred_193

        return fused, {'w_193_mean': w_193.mean()}
