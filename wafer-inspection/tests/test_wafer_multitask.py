"""
Tests for WaferMultiTaskModel — 主模型整合
"""

import torch
import pytest
from models.wafer_multitask import WaferMultiTaskModel


def test_model_init():
    """模型应初始化所有子模块"""
    model = WaferMultiTaskModel(in_channels=1, num_classes=4)

    # 共享模块
    assert hasattr(model, 'encoder')
    assert hasattr(model, 'enhance_decoder')

    # 物理增强模块（enable_physics=True 时）
    assert hasattr(model, 'pism')
    assert hasattr(model, 'asg')
    assert hasattr(model, 'angle_fusion')
    assert hasattr(model, 'detect_193')
    assert hasattr(model, 'sgf')

    # 266nm 检测子模块
    assert hasattr(model, 'msd_266_f2')
    assert hasattr(model, 'msd_266_f3')
    assert hasattr(model, 'msd_266_f4')
    assert hasattr(model, 'cls_266_f2')
    assert hasattr(model, 'cls_266_f3')
    assert hasattr(model, 'cls_266_f4')
    assert hasattr(model, 'reg_266_f2')
    assert hasattr(model, 'reg_266_f3')
    assert hasattr(model, 'reg_266_f4')


def test_model_forward_shape():
    """前向应输出增强图和检测结果"""
    model = WaferMultiTaskModel(in_channels=1, num_classes=4)
    model.eval()

    dummy = torch.randn(1, 1, 512, 512)
    with torch.no_grad():
        enhanced, detections = model(dummy)

    assert enhanced.shape == (1, 1, 512, 512), \
        f"Enhanced shape mismatch: {enhanced.shape}"
    assert detections.shape[0] == 1, f"Batch dim: {detections.shape[0]}"
    assert detections.shape[1] == 8, \
        f"Detection channels (4 reg + 4 cls): {detections.shape[1]}"


def test_model_parameter_count():
    """总参数量应 < 10M"""
    model = WaferMultiTaskModel(in_channels=1, num_classes=4)
    total = sum(p.numel() for p in model.parameters())
    assert total < 10000000, f"Total params: {total} (expected < 10M)"


def test_model_physics_enabled_disabled():
    """enable_physics=False 应回退到原始单分支"""
    model_disabled = WaferMultiTaskModel(
        in_channels=1, num_classes=4, enable_physics=False
    )
    assert not hasattr(model_disabled, 'detect_193'), \
        "Should not have detect_193"
    assert not hasattr(model_disabled, 'sgf'), \
        "Should not have sgf"
    assert not hasattr(model_disabled, 'pism'), \
        "Should not have pism"

    # 前向仍正常
    model_disabled.eval()
    dummy = torch.randn(1, 1, 512, 512)
    with torch.no_grad():
        enhanced, detections = model_disabled(dummy)
    assert enhanced.shape == (1, 1, 512, 512), \
        f"Enhanced shape: {enhanced.shape}"
    assert detections.shape[1] == 8, \
        f"Detection channels: {detections.shape[1]}"


def test_model_forward_grad_flow():
    """完整模型前向应可微分（梯度回传）"""
    model = WaferMultiTaskModel(in_channels=1, num_classes=4)
    model.train()

    dummy = torch.randn(1, 1, 128, 128)  # 小尺寸加速
    enhanced, detections = model(dummy)

    loss = enhanced.sum() + detections.sum()
    loss.backward()

    # 检查编码器梯度
    has_grad = False
    for name, p in model.encoder.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad, "No gradient flowed to encoder!"


def test_model_jit_traceable():
    """完整模型应可 JIT trace"""
    model = WaferMultiTaskModel(in_channels=1, num_classes=4)
    model.eval()

    dummy = torch.randn(1, 1, 64, 64)  # 小尺寸快速验证
    try:
        traced = torch.jit.trace(model, dummy, strict=False)
        # 验证 traced 模型输出与原始模型一致
        with torch.no_grad():
            enhanced_ref, det_ref = model(dummy)
            enhanced_tr, det_tr = traced(dummy)
        assert enhanced_ref.shape == enhanced_tr.shape
        assert det_ref.shape == det_tr.shape
    except Exception as e:
        pytest.fail(f"JIT trace failed: {e}")


def test_model_physics_without_polar():
    """enable_physics=True, enable_polar=False 应跳过 ASG"""
    model = WaferMultiTaskModel(
        in_channels=1, num_classes=4,
        enable_physics=True, enable_polar=False
    )
    assert hasattr(model, 'pism')
    assert hasattr(model, 'detect_193')
    assert hasattr(model, 'sgf')
    assert not hasattr(model, 'asg'), "Should not have asg"
    assert not hasattr(model, 'angle_fusion'), "Should not have angle_fusion"

    model.eval()
    dummy = torch.randn(1, 1, 512, 512)
    with torch.no_grad():
        enhanced, detections = model(dummy)
    assert enhanced.shape == (1, 1, 512, 512)
    assert detections.shape[1] == 8


def test_model_encoder_output_sizes():
    """编码器输出特征图的空间尺寸应符合 strides [4, 8, 16, 32]"""
    model = WaferMultiTaskModel(in_channels=1, num_classes=4)
    model.eval()

    dummy = torch.randn(1, 1, 256, 256)
    with torch.no_grad():
        feats = model.encoder(dummy)

    assert len(feats) == 4
    assert feats[0].shape[-2:] == (64, 64), f"F1 expected 64x64, got {feats[0].shape[-2:]}"
    assert feats[1].shape[-2:] == (32, 32), f"F2 expected 32x32, got {feats[1].shape[-2:]}"
    assert feats[2].shape[-2:] == (16, 16), f"F3 expected 16x16, got {feats[2].shape[-2:]}"
    assert feats[3].shape[-2:] == (8, 8), f"F4 expected 8x8, got {feats[3].shape[-2:]}"

    # 通道数检查
    assert feats[0].shape[1] == 56
    assert feats[1].shape[1] == 112
    assert feats[2].shape[1] == 224
    assert feats[3].shape[1] == 448


def test_asg_flattened_52_output():
    """验证 enable_physics=True 时 ASG 输出 52 扁平张量（13 角度 × 4 层级）"""
    model = WaferMultiTaskModel(in_channels=1, num_classes=4)
    model.eval()

    dummy = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        feats = model.encoder(dummy)
        scale_map = torch.randn(1, 1, feats[0].shape[2], feats[0].shape[3])
        all_angle_feats, stacked_conf = model.asg(feats, scale_map)

    assert len(all_angle_feats) == 52, \
        f"Expected 52 flattened tensors, got {len(all_angle_feats)}"
    assert stacked_conf.shape == (1, 13, feats[0].shape[2], feats[0].shape[3]), \
        f"stacked_conf shape: {stacked_conf.shape}"

    # 验证分组: 每组 4 个张量
    for i in range(13):
        group = all_angle_feats[i * 4: i * 4 + 4]
        assert len(group) == 4, f"Angle {i} group has {len(group)} tensors"
        # 第一层是 F1（56 通道）
        assert group[0].shape[1] == 56, f"Angle {i} F1 channels: {group[0].shape[1]}"
