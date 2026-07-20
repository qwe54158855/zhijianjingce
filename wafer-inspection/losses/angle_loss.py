"""
角度散射损失函数

L_angle_consistency: 跨角度散射一致性
L_angle_temporal: 相邻角度平滑性
"""

import torch
import torch.nn.functional as F


def angle_scattering_consistency_loss(feat_266_base, feat_266_theta):
    """
    角度散射一致性损失

    约束: ASG 生成的不同角度特征在散射域保持一致。
    方位角变化只改变观测方向，不改变缺陷的散射类型。

    Args:
        feat_266_base: 0° 基准特征列表
        feat_266_theta: θ° 目标特征列表

    Returns:
        loss: 标量
    """
    loss = 0.0
    count = 0

    for f_base, f_theta in zip(feat_266_base, feat_266_theta):
        # 对齐空间尺寸
        if f_base.shape[-2:] != f_theta.shape[-2:]:
            f_base = F.interpolate(f_base, size=f_theta.shape[-2:],
                                   mode='bilinear', align_corners=False)

        # 散射能量谱一致性（功率谱密度应相近）
        spec_base = f_base.pow(2).mean(dim=1, keepdim=True) + 1e-8
        spec_theta = f_theta.pow(2).mean(dim=1, keepdim=True) + 1e-8

        # 归一化为概率分布
        spec_base = spec_base / spec_base.sum(dim=(-2, -1), keepdim=True)
        spec_theta = spec_theta / spec_theta.sum(dim=(-2, -1), keepdim=True)

        # KL 散度
        kl = (spec_base * (spec_base / spec_theta).log()).mean()
        loss += kl
        count += 1

    return loss / max(count, 1)


def angle_smoothness_loss(angle_preds):
    """
    相邻角度平滑性损失

    约束: 相邻角度(θ, θ+5°)的检测结果应平滑变化
    物理原因: 散射强度随角度连续变化，不应有剧烈跳变
    """
    loss = 0.0
    for i in range(len(angle_preds) - 1):
        loss += F.mse_loss(angle_preds[i], angle_preds[i + 1])
    return loss / max(len(angle_preds) - 1, 1)
