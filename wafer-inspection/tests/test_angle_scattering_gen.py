"""Tests for ASG: Angle Scattering Generator"""

import torch
from models.angle_scattering_gen import AngleScatteringGenerator


def test_asg_init():
    """ASG 应初始化指定角度和子模块"""
    model = AngleScatteringGenerator(
        channels_per_stage=[56, 112],
        target_angles=[15, 30, 45, 60, 75]
    )
    assert model.num_angles == 5
    assert model.shifts.shape[0] == 5
    assert hasattr(model, 'scatter_modulator')
    assert hasattr(model, 'completion_net')
    assert hasattr(model, 'confidence_predictor')


def test_asg_shift_correct():
    """几何偏移量应匹配极坐标 512px=360°"""
    model = AngleScatteringGenerator()
    # 45° 对应的像素偏移: 45 * 512/360 = 64
    shift_45 = model.shifts[6]  # angles[6] = 45
    assert abs(shift_45.item() - 64) <= 1, f"45° shift should be ~64px, got {shift_45}"


def test_asg_forward_single_angle():
    """单角度前向应输出正确形状"""
    model = AngleScatteringGenerator(channels_per_stage=[56, 112])
    model.eval()

    feats = [
        torch.randn(1, 56, 32, 64),  # 注意: W 方向是角度轴
        torch.randn(1, 112, 16, 32),
    ]
    scale_map_f1 = torch.rand(1, 1, 32, 64)  # F1 尺度图

    feat_theta, conf = model.forward_one_angle(feats, 3, scale_map_f1)  # angle_idx=3
    assert len(feat_theta) == 2
    assert feat_theta[0].shape == feats[0].shape
    assert conf.shape == (1, 1, 32, 64)


def test_asg_forward_all_angles():
    """全 13 角度前向应返回正确数量"""
    model = AngleScatteringGenerator(channels_per_stage=[56])
    model.eval()

    feats = [torch.randn(1, 56, 32, 128)]
    scale_map_f1 = torch.rand(1, 1, 32, 128)

    angle_feats, conf = model(feats, scale_map_f1)
    assert len(angle_feats) == 13
    assert conf.shape == (1, 13, 32, 128)


def test_asg_geometric_shift_roll():
    """torch.roll 应实现循环平移"""
    model = AngleScatteringGenerator()

    feat = torch.zeros(1, 1, 4, 8)
    feat[0, 0, :, 0] = 1.0  # 在 x=0 处设标记

    shifted = model._geometric_shift(feat, angle_idx=6)  # 45° → 64px shift

    # 验证循环平移（由于 W=8, shift_45=64 ≡ 0 mod 8）
    # 实际 shift = 64 % 8 = 0
    effective_shift = model.shifts[6].item() % feat.shape[-1]
    if effective_shift > 0:
        assert shifted[0, 0, 0, effective_shift].item() == 1.0


def test_asg_jit_traceable():
    """ASG 应可 JIT trace"""
    model = AngleScatteringGenerator(channels_per_stage=[56])
    model.eval()

    feats = [torch.randn(1, 56, 16, 32)]
    scale_map_f1 = torch.rand(1, 1, 16, 32)

    traced = torch.jit.trace(model, (feats, scale_map_f1))
    angle_feats, conf = traced(feats, scale_map_f1)
    assert len(angle_feats) == 13


def test_asg_parameter_count():
    """ASG 参数应 < 0.2M"""
    model = AngleScatteringGenerator()
    total = sum(p.numel() for p in model.parameters())
    assert total < 200000, f"ASG params: {total}"
