"""
Tests for angle loss functions
"""

import torch
from losses.angle_loss import angle_scattering_consistency_loss, angle_smoothness_loss


def test_angle_scattering_consistency_loss():
    """角度散射一致性损失应正常前向"""
    f_base = [torch.randn(2, 56, 16, 16), torch.randn(2, 112, 8, 8)]
    f_theta = [torch.randn(2, 56, 16, 16), torch.randn(2, 112, 8, 8)]
    loss = angle_scattering_consistency_loss(f_base, f_theta)
    assert loss.ndim == 0, f"Expected scalar, got shape {loss.shape}"
    assert loss > 0
    assert torch.isfinite(loss)


def test_angle_scattering_consistency_different_sizes():
    """应处理不同空间尺寸（通过插值对齐）"""
    f_base = [torch.randn(2, 56, 32, 32)]
    f_theta = [torch.randn(2, 56, 16, 16)]
    loss = angle_scattering_consistency_loss(f_base, f_theta)
    assert loss.ndim == 0
    assert loss > 0
    assert torch.isfinite(loss)


def test_angle_scattering_consistency_identical():
    """相同特征应在散射域完全一致 → loss 应为 0"""
    torch.manual_seed(42)
    t = torch.randn(2, 8, 16, 16)
    f_base = [t]
    f_theta = [t.clone()]
    loss = angle_scattering_consistency_loss(f_base, f_theta)
    assert loss < 1e-6, f"Expected ~0 loss for identical features, got {loss}"


def test_angle_scattering_consistency_multi_level():
    """应正确处理多层特征"""
    f_base = [
        torch.randn(1, 56, 32, 32),
        torch.randn(1, 112, 16, 16),
        torch.randn(1, 224, 8, 8),
        torch.randn(1, 448, 4, 4),
    ]
    f_theta = [
        torch.randn(1, 56, 32, 32),
        torch.randn(1, 112, 16, 16),
        torch.randn(1, 224, 8, 8),
        torch.randn(1, 448, 4, 4),
    ]
    loss = angle_scattering_consistency_loss(f_base, f_theta)
    assert loss.ndim == 0
    assert loss > 0
    assert torch.isfinite(loss)


def test_angle_smoothness_loss():
    """相邻角度平滑损失应正常前向"""
    preds = [torch.randn(2, 4, 8, 8) for _ in range(3)]
    loss = angle_smoothness_loss(preds)
    assert loss.ndim == 0
    assert loss > 0
    assert torch.isfinite(loss)


def test_angle_smoothness_identical():
    """完全相同的预测应给出零损失"""
    t = torch.randn(2, 4, 8, 8)
    preds = [t, t, t]
    loss = angle_smoothness_loss(preds)
    assert loss < 1e-6, f"Expected ~0 loss, got {loss}"


def test_angle_smoothness_two_angles():
    """最少 2 个角度也应正常工作"""
    a = torch.randn(1, 8, 16, 16)
    b = torch.randn(1, 8, 16, 16)
    loss = angle_smoothness_loss([a, b])
    assert loss.ndim == 0
    assert loss > 0


def test_angle_smoothness_single_angle():
    """单个角度应返回 0 损失"""
    t = torch.randn(2, 4, 8, 8)
    loss = angle_smoothness_loss([t])
    assert loss == 0.0, f"Single angle should give 0 loss, got {loss}"


def test_angle_smoothness_gradient_flow():
    """平滑损失应可微"""
    preds = [torch.randn(1, 2, 8, 8, requires_grad=True) for _ in range(3)]
    loss = angle_smoothness_loss(preds)
    loss.backward()
    for p in preds:
        assert p.grad is not None, "No gradient flowed"
        assert p.grad.abs().sum() > 0
