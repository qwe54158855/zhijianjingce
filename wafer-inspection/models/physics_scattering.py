"""
PISM: Physics-Informed Scattering Module
可微散射物理模块 — 实现 Rayleigh/Mie 混合模型

物理先验 ≈ 90% 信号 + 可学习残差补偿 ≈ 10% 近似误差
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.mie_lut import precompute_mie_gain_table


class ScatteringPhysics(nn.Module):
    """
    可微散射物理模块 PISM

    输入: 编码器多尺度特征 [F1, F2, F3, F4]
    输出: 物理增强后的虚拟 193nm 特征 + 物理诊断图

    Args:
        channels_per_stage: 每层特征图的通道数
        lambda_in: 输入波长 (nm), 默认 266
        lambda_out: 输出波长 (nm), 默认 193
    """

    def __init__(self,
                 channels_per_stage=(56, 112, 224, 448),
                 lambda_in=266.0,
                 lambda_out=193.0):
        super().__init__()

        self.lambda_in = lambda_in
        self.lambda_out = lambda_out

        # === 1. 局部尺度估计头 ===
        # 估计每像素位置的有效散射尺度 s(x) ∈ [0,1]
        # s→0: Rayleigh 散射主导（小缺陷）
        # s→1: Mie 散射主导（大缺陷）
        self.scale_estimators = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, 16, 3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 3, padding=1, bias=False),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            ) for c in channels_per_stage
        ])

        # === 2. Rayleigh 散射比（常量） ===
        # I_out / I_in = (lambda_in / lambda_out)^4
        rayleigh_ratio = (lambda_in / lambda_out) ** 4
        self.register_buffer('rayleigh_ratio',
                             torch.tensor(rayleigh_ratio, dtype=torch.float32))

        # === 3. Mie 散射查找表（预计算冻结） ===
        gains, x_vals = precompute_mie_gain_table(
            lambda_in=lambda_in,
            lambda_out=lambda_out,
            n_sic_in=2.6+0.1j,
            n_sic_out=2.8+0.15j,
            table_size=512
        )
        # 将增益值 reshape 为 Conv1d 权重格式 [1,1,table_size]
        mie_table = torch.from_numpy(gains).float().view(1, 1, -1)
        self.register_buffer('mie_lut', mie_table)
        # 保存 x_vals 作为辅助信息（不参与梯度）
        self.register_buffer('x_vals',
                             torch.from_numpy(x_vals).float().view(1, 1, -1))

        # === 4. 可学习残差网络 ===
        # 补偿物理近似误差
        self.residual_nets = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, max(16, c // 16), 1, bias=False),
                nn.BatchNorm2d(max(16, c // 16)),
                nn.ReLU(inplace=True),
                nn.Conv2d(max(16, c // 16), c, 1, bias=False),
                nn.BatchNorm2d(c),
            ) for c in channels_per_stage
        ])

        # === 5. 光谱角约束边界 ===
        self.lower_bound = 1.0   # 不弱于 266nm
        self.upper_bound = 8.0   # ~2× Rayleigh（含共振增强裕量）

    def _interpolate_lut(self, lut, indices, H, W):
        """
        从 1D 查找表进行双线性插值

        Args:
            lut: [1, 1, N] 查找表
            indices: [B, 1, H, W] 索引值 [0, N-1]
            H, W: 输出特征图尺寸

        Returns:
            [B, 1, H, W] 插值后的增益图
        """
        # 将 lut 展开为 2D 以便使用 grid_sample
        # lut_2d: [1, 1, 1, N]
        lut_2d = lut.unsqueeze(2)  # [1, 1, 1, N]

        # 将 indices 归一化到 [-1, 1] 作为 grid_sample 的坐标
        N = lut.shape[-1]
        grid = (indices / (N - 1)) * 2.0 - 1.0  # [B, 1, H, W] → [-1, 1]

        # grid_sample 需要 [B, H, W, 2] 的 grid
        grid_expanded = torch.stack([grid.squeeze(1)], dim=-1)  # [B, H, W, 1]
        # 复制到 x,y 两个维度（但 y 维度固定为 0 因为我们是一维 LUT）
        grid_2d = torch.cat([
            grid_expanded,
            torch.zeros_like(grid_expanded)
        ], dim=-1)  # [B, H, W, 2]

        # 网格采样
        sampled = F.grid_sample(
            lut_2d.expand(indices.shape[0], -1, -1, -1),
            grid_2d,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )  # [B, 1, H, W]

        return sampled

    def forward(self, feats_266):
        """
        前向: 266nm 特征 → 虚拟 193nm 特征

        Args:
            feats_266: 多尺度特征列表 [F1, F2, F3, F4]
                      形状: [B, C_i, H_i, W_i]

        Returns:
            feats_193: 物理增强后的虚拟 193nm 特征
            diagnostics: 包含 scale_map, gain_map 等诊断信息
        """
        feats_193 = []
        diagnostics = {}

        for i, (feat, scale_est, res_net) in enumerate(
            zip(feats_266, self.scale_estimators, self.residual_nets)):

            B, C, H, W = feat.shape

            # Step 1: 估计每像素有效散射尺度
            s = scale_est(feat)  # [B, 1, H, W], 范围 [0,1]

            # Step 2: 计算物理散射增强比
            # Rayleigh 部分: 常数 (266/193)^4
            rayleigh_gain = self.rayleigh_ratio  # 标量

            # Mie 部分: 查找表插值
            # 将 s 映射到 LUT 索引范围 [0, N-1]
            s_norm = s * (self.mie_lut.shape[-1] - 1)  # [B, 1, H, W]
            mie_gain = self._interpolate_lut(self.mie_lut, s_norm, H, W)

            # Step 3: 混合增益
            # Rayleigh-Mie 过渡区在 s ≈ 0.3 附近
            # 高斯权重: s→0 时 w_r→1 (Rayleigh 主导)
            #            s→1 时 w_r→0 (Mie 主导)
            sigma = 0.3
            w_r = torch.exp(-s ** 2 / (2 * sigma ** 2))
            w_m = 1.0 - w_r

            combined_gain = w_r * rayleigh_gain + w_m * mie_gain

            # 光谱角约束（投影到物理合理范围）
            combined_gain = combined_gain.clamp(self.lower_bound, self.upper_bound)

            # Step 4: 物理驱动增强（逐元素乘法）
            feat_193_phys = feat * combined_gain

            # Step 5: 可学习残差补偿
            residual = res_net(feat)
            feat_193_i = feat_193_phys + 0.1 * residual  # 残差缩放，防止主导
            feat_193_i = feat_193_i.clamp(-10.0, 10.0)  # 数值稳定性

            feats_193.append(feat_193_i)

            # 收集诊断信息
            diagnostics[f'scale_map_{i}'] = s
            diagnostics[f'gain_map_{i}'] = combined_gain
            if i == 0:
                diagnostics['scale_map'] = s
                diagnostics['gain_map'] = combined_gain
                diagnostics['rayleigh_weight'] = w_r.mean()
                diagnostics['mie_weight'] = w_m.mean()

        return feats_193, diagnostics
