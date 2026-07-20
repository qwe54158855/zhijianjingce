import torch
from models.scattering_fusion import ScatteringGuidedFusion


def test_sgf_init():
    model = ScatteringGuidedFusion(num_classes=4)
    assert hasattr(model, 'weight_predictor')
    assert hasattr(model, 'channel_bias')
    assert model.channel_bias.shape == (1, 8, 1, 1)  # 4 cls + 4 reg


def test_sgf_forward_shape():
    model = ScatteringGuidedFusion(num_classes=4)
    model.eval()
    B, H, W = 2, 32, 32

    pred_266 = torch.randn(B, 8, H, W)
    pred_193 = torch.randn(B, 8, H, W)
    gain_map = torch.rand(B, 1, H, W) * 3.0 + 1.0
    scale_map = torch.rand(B, 1, H, W)

    fused, diag = model(pred_266, pred_193, gain_map, scale_map)
    assert fused.shape == (B, 8, H, W)
    assert 'w_193_mean' in diag


def test_sgf_physical_prior():
    """增益高的区域应更信赖 193nm 分支"""
    model = ScatteringGuidedFusion(num_classes=4)
    model.eval()
    B, H, W = 1, 16, 16

    pred_266 = torch.zeros(B, 8, H, W)
    pred_193 = torch.ones(B, 8, H, W)

    # 高增益区域 (gain≈3.5) → w_193 应高 → 输出应接近 1
    gain_map = torch.full((B, 1, H, W), 3.5)
    scale_map = torch.zeros(B, 1, H, W)

    fused, diag = model(pred_266, pred_193, gain_map, scale_map)
    assert fused.mean() > 0.5, f"Expected high gain→193nm dominant, got {fused.mean():.3f}"


def test_sgf_jit_traceable():
    model = ScatteringGuidedFusion(num_classes=4)
    model.eval()

    pred_266 = torch.randn(1, 8, 16, 16)
    pred_193 = torch.randn(1, 8, 16, 16)
    gain_map = torch.rand(1, 1, 16, 16) * 2.0 + 1.0
    scale_map = torch.rand(1, 1, 16, 16)

    traced = torch.jit.trace(model, (pred_266, pred_193, gain_map, scale_map), strict=False)
    fused, diag = traced(pred_266, pred_193, gain_map, scale_map)
    assert fused.shape == (1, 8, 16, 16)
