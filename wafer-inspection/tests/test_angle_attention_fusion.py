import torch
from models.angle_attention_fusion import AngleAttentionFusion


def test_aaf_init():
    model = AngleAttentionFusion(num_classes=4, num_angles=13)
    assert hasattr(model, 'angle_attention')
    assert hasattr(model, 'gamma')
    assert model.gamma == 0.7


def test_aaf_forward_shape():
    model = AngleAttentionFusion(num_classes=4, num_angles=5)
    model.eval()
    B, H, W = 2, 16, 16
    num_angles = 5

    angle_preds = [torch.randn(B, 8, H, W) for _ in range(num_angles)]
    angle_conf = torch.rand(B, num_angles, H, W)

    fused, diag = model(angle_preds, angle_conf)
    assert fused.shape == (B, 8, H, W)
    assert 'weights' in diag


def test_aaf_confidence_dominant():
    """高置信度角度应主导融合结果"""
    torch.manual_seed(42)
    model = AngleAttentionFusion(num_classes=4, num_angles=3)
    model.eval()
    B, H, W = 1, 8, 8

    # angle 0: 全 0, angle 1: 全 1, angle 2: 全 2
    angle_preds = [
        torch.zeros(B, 8, H, W),
        torch.ones(B, 8, H, W),
        torch.ones(B, 8, H, W) * 2.0,
    ]
    # angle 2 置信度最高 (使用更极端的置信度分配使物理先验主导)
    angle_conf = torch.zeros(1, 3, H, W)
    angle_conf[:, 0, :, :] = 0.05
    angle_conf[:, 1, :, :] = 0.05
    angle_conf[:, 2, :, :] = 0.9

    fused, _ = model(angle_preds, angle_conf)
    # 结果应接近 2.0 (接近 angle 2)
    assert fused.mean() > 1.5, f"Should lean toward angle 2, got {fused.mean():.3f}"


def test_aaf_jit_traceable():
    model = AngleAttentionFusion(num_classes=4, num_angles=3)
    model.eval()

    angle_preds = [torch.randn(1, 8, 8, 8) for _ in range(3)]
    angle_conf = torch.rand(1, 3, 8, 8)

    traced = torch.jit.trace(model, (angle_preds, angle_conf), strict=False)
    fused, diag = traced(angle_preds, angle_conf)
    assert fused.shape == (1, 8, 8, 8)
