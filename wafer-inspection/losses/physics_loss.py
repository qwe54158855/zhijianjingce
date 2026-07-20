"""
物理约束损失函数

L_scat: 散射一致性损失 — 约束 193nm 特征经过 PISM 后散射响应与 266nm 一致
L_spec: 光谱角损失 — 约束 193nm/266nm 响应比率在 [1.0, 8.0] 物理范围
"""

import torch
import torch.nn.functional as F


def scattering_consistency_loss(feat_266, feat_193, pism_module):
    """
    散射一致性损失

    约束: PISM(feat_193) ≈ feat_266
    含义: 虚拟 193nm 特征经过 PISM 散射响应后，
          应与原始 266nm 的散射响应一致

    Args:
        feat_266: 编码器输出的 266nm 多尺度特征列表
        feat_193: PISM 输出的虚拟 193nm 多尺度特征列表
        pism_module: PISM 模块（共享权重，梯度不传播到其参数）

    Returns:
        loss: 标量损失值
    """
    # 对 193nm 特征计算散射响应（共享 PISM 权重）
    with torch.no_grad():
        feat_193_pism, _ = pism_module([f.detach() for f in feat_193])

    loss = 0.0
    count = 0
    for f_266, f_193_p in zip(feat_266, feat_193_pism):
        # 对齐空间尺寸（降采样到与 f_193_p 一致）
        if f_266.shape[-2:] != f_193_p.shape[-2:]:
            f_266 = F.interpolate(f_266, size=f_193_p.shape[-2:],
                                  mode='bilinear', align_corners=False)

        # MSE 损失
        loss += F.mse_loss(f_266, f_193_p)
        count += 1

    return loss / max(count, 1)


def spectral_angle_loss(feat_266_list, feat_193_list):
    """
    光谱角损失

    约束: 1.0 ≤ I_193 / I_266 ≤ 8.0
    物理含义:
      下限 1.0: 193nm 散射强度不应弱于 266nm（违反 Rayleigh 律）
      上限 8.0: 考虑共振增强效应后留 2× 裕量

    Args:
        feat_266_list: 编码器特征列表
        feat_193_list: PISM 输出特征列表

    Returns:
        loss: 标量损失值
    """
    loss = 0.0
    count = 0

    for f_266, f_193 in zip(feat_266_list, feat_193_list):
        # 对齐空间尺寸
        if f_266.shape[-2:] != f_193.shape[-2:]:
            f_266 = F.interpolate(f_266, size=f_193.shape[-2:],
                                  mode='bilinear', align_corners=False)

        # 安全比率计算
        ratio = (f_193.abs() + 1e-6) / (f_266.abs() + 1e-6)

        # 双边惩罚
        lower_penalty = F.relu(1.0 - ratio).mean()
        upper_penalty = F.relu(ratio - 8.0).mean()

        loss += lower_penalty + upper_penalty
        count += 1

    return loss / max(count, 1)
