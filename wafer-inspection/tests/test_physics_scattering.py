"""
Tests for PISM: Physics-Informed Scattering Module
"""

import torch
import pytest
from models.physics_scattering import ScatteringPhysics


def test_pism_init():
    """PISM 应初始化所有子模块"""
    model = ScatteringPhysics(channels_per_stage=[56, 112, 224, 448])

    # 检查核心组件存在
    assert hasattr(model, 'scale_estimators')
    assert hasattr(model, 'rayleigh_ratio')
    assert hasattr(model, 'mie_lut')
    assert hasattr(model, 'residual_nets')

    # 检查 Rayleigh 常数
    expected_ratio = (266.0 / 193.0) ** 4
    assert abs(model.rayleigh_ratio.item() - expected_ratio) < 1e-5


def test_pism_forward_shape():
    """PISM 前向应输出相同形状的多尺度特征"""
    model = ScatteringPhysics(channels_per_stage=[56, 112, 224, 448])
    B = 2

    feats = [
        torch.randn(B, 56, 128, 128),
        torch.randn(B, 112, 64, 64),
        torch.randn(B, 224, 32, 32),
        torch.randn(B, 448, 16, 16),
    ]

    feats_out, diagnostics = model(feats)

    assert len(feats_out) == 4
    for f_in, f_out in zip(feats, feats_out):
        assert f_out.shape == f_in.shape, f"Shape mismatch: {f_out.shape} vs {f_in.shape}"

    # 诊断输出应包含 scale_maps
    assert 'scale_map_0' in diagnostics
    assert 'gain_map_0' in diagnostics
    assert diagnostics['scale_map_0'].shape == (B, 1, 128, 128)


def test_pism_physical_gain_in_range():
    """散射增益应在物理合理范围 [1.0, 8.0] 内"""
    model = ScatteringPhysics()
    model.eval()

    feats = [torch.randn(1, c, 32, 32) for c in [56, 112, 224, 448]]
    _, diagnostics = model(feats)

    gain = diagnostics['gain_map_0']
    assert gain.min() >= 1.0, f"Gain too low: {gain.min()}"
    assert gain.max() <= 8.0, f"Gain too high: {gain.max()}"


def test_pism_jit_traceable():
    """PISM 应可被 torch.jit.trace"""
    model = ScatteringPhysics(channels_per_stage=[56, 112])
    model.eval()

    feats = [
        torch.randn(1, 56, 64, 64),
        torch.randn(1, 112, 32, 32),
    ]

    traced = torch.jit.trace(model, (feats,), strict=False)

    feats_out, diag = traced(feats)
    assert len(feats_out) == 2
    for f_out in feats_out:
        assert f_out.shape[0] == 1


def test_pism_grad_flows():
    """PISM 的梯度应能回传到编码器输出特征"""
    model = ScatteringPhysics(channels_per_stage=[56])

    feat = torch.randn(1, 56, 32, 32, requires_grad=True)
    feats_out, _ = model([feat])
    loss = feats_out[0].sum()
    loss.backward()

    assert feat.grad is not None
    assert feat.grad.abs().sum() > 0


def test_pism_residual_network_params():
    """残差网络的参数量应控制在 30K 以内"""
    model = ScatteringPhysics(channels_per_stage=[56, 112, 224, 448])
    total_residual_params = sum(
        p.numel() for name, p in model.named_parameters() if 'residual' in name
    )
    assert total_residual_params < 50000, f"Residual params: {total_residual_params}"
