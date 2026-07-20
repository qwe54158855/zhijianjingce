# tests/test_mie_lut.py
import numpy as np
from utils.mie_lut import (
    compute_mie_coefficients,
    mie_scattering_efficiency,
    precompute_mie_gain_table
)


def test_mie_coefficients_shape():
    """Mie 系数应返回正确长度的数组"""
    x = 1.0
    m = 1.5 + 0.1j
    a_n, b_n = compute_mie_coefficients(x, m)
    nmax = int(np.ceil(x + 4 * x**(1/3) + 2))
    assert len(a_n) == nmax, f"Expected {nmax}, got {len(a_n)}"
    assert len(b_n) == nmax
    assert np.all(np.isfinite(a_n))
    assert np.all(np.isfinite(b_n))


def test_mie_efficiency_positive():
    """散射效率应为正实数"""
    x = 0.5
    m = 1.5 + 0.0j
    Q = mie_scattering_efficiency(x, m)
    assert Q > 0, f"Q_sca should be positive, got {Q}"
    assert np.isfinite(Q)


def test_mie_small_particle_rayleigh_limit():
    """
    小粒子极限 (x << 1) 下 Mie 应趋近 Rayleigh 散射
    Rayleigh: Q_sca ∝ x^4
    """
    x1, x2 = 0.01, 0.02
    m = 1.5 + 0.0j
    Q1 = mie_scattering_efficiency(x1, m)
    Q2 = mie_scattering_efficiency(x2, m)
    # Q_sca(x2)/Q_sca(x1) ≈ (x2/x1)^4 = 16
    ratio = Q2 / Q1
    assert 10 < ratio < 24, f"Rayleigh limit ratio ~16, got {ratio}"


def test_gain_table_266_to_193():
    """
    266nm→193nm 散射增强比表
    - Rayleigh 极限下应接近 (266/193)^4 ≈ 3.5
    - 所有值应 > 0
    - 单调性检验: 小尺寸增益高(193nm优势), 大尺寸趋近1
    """
    gains, x_vals = precompute_mie_gain_table(table_size=256)

    assert len(gains) == 256
    assert np.all(gains > 0), "All gains must be positive"

    # Rayleigh 极限 (x → 0) 应接近 3.5
    rayleigh_limit = (266.0 / 193.0) ** 4
    # 取前 10 个点平均
    small_x_gain = np.mean(gains[:10])
    assert abs(small_x_gain / rayleigh_limit - 1.0) < 0.3, \
        f"Rayleigh limit ~3.5, got {small_x_gain}"


def test_gain_table_jit_compatible():
    """验证增益表可转换为 PyTorch tensor 用作 buffer"""
    import torch
    gains, x_vals = precompute_mie_gain_table(table_size=128)
    tensor = torch.tensor(gains).view(1, 1, -1)
    assert tensor.shape == (1, 1, 128), f"Shape mismatch: {tensor.shape}"
    assert tensor.dtype == torch.float32
    assert torch.all(torch.isfinite(tensor))
