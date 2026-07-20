import torch
from losses.physics_loss import scattering_consistency_loss, spectral_angle_loss

def test_scattering_consistency_forward():
    """散射一致性损失应正常前向"""
    B, C, H, W = 2, 56, 32, 32
    feat_266 = [torch.randn(B, C, H, W)]
    feat_193 = [torch.randn(B, C, H, W)]

    # 创建简单的 PISM mock
    class MockPISM(torch.nn.Module):
        def forward(self, x):
            # 输入就应该是一个列表
            return [f * 0.9 for f in x], {}

    pism = MockPISM()
    loss = scattering_consistency_loss(feat_266, feat_193, pism)
    assert loss > 0
    assert torch.isfinite(loss)

def test_scattering_consistency_identical():
    """如果输入相同，损失应为 0"""
    feat = [torch.ones(1, 8, 16, 16)]
    class IdentityPISM(torch.nn.Module):
        def forward(self, x):
            return x, {}
    pism = IdentityPISM()
    loss = scattering_consistency_loss(feat, feat, pism)
    assert loss < 1e-6, f"Expected ~0, got {loss}"

def test_spectral_angle_loss_forward():
    """光谱角损失应正常前向"""
    f266 = [torch.randn(1, 8, 16, 16) + 0.5]
    f193 = [torch.randn(1, 8, 16, 16) * 2.0 + 1.0]
    loss = spectral_angle_loss(f266, f193)
    assert loss > 0
    assert torch.isfinite(loss)

def test_spectral_angle_loss_penalizes_extremes():
    """
    当比率超出 [1.0, 8.0] 范围时，损失应增大
    """
    # 比率在合理范围内 → 损失小
    f266_in = [torch.ones(1, 4, 8, 8) * 2.0]
    f193_in = [torch.ones(1, 4, 8, 8) * 4.0]  # 比率=2.0
    loss_in = spectral_angle_loss(f266_in, f193_in)
    assert loss_in < 0.01, f"Loss should be near 0 for in-range ratios, got {loss_in}"

    # 比率违反下限 → 损失大
    f266_low = [torch.ones(1, 4, 8, 8) * 2.0]
    f193_low = [torch.ones(1, 4, 8, 8) * 0.5]  # 比率=0.25 < 1.0
    loss_low = spectral_angle_loss(f266_low, f193_low)
    assert loss_low > loss_in + 0.1, f"Lower-bound violation should increase loss"
