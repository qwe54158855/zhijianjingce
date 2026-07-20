"""
Mie 散射理论数值计算 — Bohren-Huffman 算法

用于预计算 266nm→193nm 的散射增强比查找表 (LUT)。

物理参数:
  SiC (6H-SiC) 复折射率:
    @ 266nm: n=2.6+0.1i
    @ 193nm: n=2.8+0.15i
  环境介质: 空气 n_medium=1.0
  尺寸范围: d ∈ [0.01λ_266, 100λ_266] → [2.66nm, 26.6μm]
"""

import numpy as np
from scipy.special import spherical_jn, spherical_yn


def compute_mie_coefficients(x, m):
    """
    Mie 散射系数 a_n, b_n (Bohren-Huffman 算法)

    Args:
        x: 尺寸参数 = πd/λ
        m: 相对复折射率 n_particle / n_medium

    Returns:
        a_n, b_n: 复数散射系数数组
    """
    nmax = int(np.ceil(x + 4 * x**(1/3) + 2))
    nmax = max(nmax, 2)

    # Riccati-Bessel 函数递归计算
    # ψ_n(z) = z * j_n(z)
    # ξ_n(z) = z * (j_n(z) - i*y_n(z))

    # 计算 n=0 到 nmax 的球贝塞尔和球诺伊曼函数
    # j_n(x), y_n(x) for n=0..nmax
    jn = spherical_jn(np.arange(nmax + 1), x)
    yn = spherical_yn(np.arange(nmax + 1), x)

    # ψ_n(x) = x * j_n(x)
    psi = x * jn

    # ξ_n(x) = x * (j_n(x) - i*y_n(x))
    xi = x * (jn - 1j * yn)

    # 对 m*x 计算
    mx = m * x
    jn_mx = spherical_jn(np.arange(nmax + 1), mx)
    # ψ_n(mx) = mx * j_n(mx)
    psi_mx = mx * jn_mx

    # 导数: ψ'_n(z) = ψ_{n-1}(z) - n*ψ_n(z)/z
    # 用递推: D_n(mx) = ψ'_n(mx) / ψ_n(mx) — 对数导数
    D = np.zeros(nmax + 1, dtype=complex)
    for n in range(nmax, 0, -1):
        D[n - 1] = n / mx - 1.0 / (D[n] + n / mx)
    D[nmax] = 0.0 + 0.0j
    # 向下递推计算所有 D_n
    for n in range(nmax, 0, -1):
        D[n - 1] = n / mx - 1.0 / (D[n] + n / mx)

    a_n = np.zeros(nmax, dtype=complex)
    b_n = np.zeros(nmax, dtype=complex)

    for n in range(1, nmax + 1):
        n_idx = n - 1

        # 对数导数 D_n(mx)
        D_n = D[n_idx]

        # ψ_n(x) 和 ξ_n(x)
        psi_n = psi[n]
        xi_n = xi[n]

        # ψ_n'(x)
        if n == 0:
            psi_n_prime = np.cos(x)
        else:
            psi_n_prime = psi[n - 1] - n * psi_n / x

        # a_n = (D_n/m + n/x) * ψ_n - ψ_{n-1}  /  (D_n/m + n/x) * ξ_n - ξ_{n-1}
        # b_n = (m*D_n + n/x) * ψ_n - ψ_{n-1}  /  (m*D_n + n/x) * ξ_n - ξ_{n-1}

        xi_n_minus_1 = xi[n - 1] if n > 0 else np.cos(x) + 1j * np.sin(x)
        psi_n_minus_1 = psi[n - 1] if n > 0 else np.sin(x)

        term_a_num = (D_n / m + n / x) * psi_n - psi_n_minus_1
        term_a_den = (D_n / m + n / x) * xi_n - xi_n_minus_1
        a_n[n_idx] = term_a_num / term_a_den

        term_b_num = (m * D_n + n / x) * psi_n - psi_n_minus_1
        term_b_den = (m * D_n + n / x) * xi_n - xi_n_minus_1
        b_n[n_idx] = term_b_num / term_b_den

    return a_n, b_n


def mie_scattering_efficiency(x, m):
    """
    Mie 散射效率 Q_sca

    Args:
        x: 尺寸参数
        m: 相对复折射率

    Returns:
        Q_sca: 散射效率因子
    """
    a_n, b_n = compute_mie_coefficients(x, m)
    n = np.arange(1, len(a_n) + 1)
    factor = (2.0 * n + 1.0) / (x**2)
    Q_sca = np.real(np.sum(factor * (np.abs(a_n)**2 + np.abs(b_n)**2)))
    return Q_sca


def precompute_mie_gain_table(lambda_in=266.0, lambda_out=193.0,
                               n_sic_in=2.6+0.1j, n_sic_out=2.8+0.15j,
                               n_medium=1.0, table_size=512):
    """
    预计算 Mie 散射增强比查找表

    gain(x) = Q_sca_193nm(x) / Q_sca_266nm(x)
    其中尺寸参数 x = πd/λ_in 随缺陷尺寸 d 变化

    Args:
        lambda_in: 输入波长 (nm)
        lambda_out: 输出波长 (nm)
        n_sic_in: SiC 在 lambda_in 的复折射率
        n_sic_out: SiC 在 lambda_out 的复折射率
        n_medium: 环境介质折射率
        table_size: 查找表点数

    Returns:
        gains: [table_size] float32 数组，scattering gain vs x
        x_values: [table_size] 对应的尺寸参数值
    """
    # 归一化尺寸参数 x = πd/λ
    # d ∈ [0.01*λ_in, 100*λ_in] 覆盖 Rayleigh 到 Mie 全范围
    x_values = np.logspace(-2, 2, table_size)

    m_in = n_sic_in / n_medium
    m_out = n_sic_out / n_medium

    gains = np.zeros(table_size, dtype=np.float32)
    for i, x in enumerate(x_values):
        # 在输入波长下的散射效率
        Q_in = mie_scattering_efficiency(x, m_in)

        # 在输出波长下的散射效率 (尺寸参数按波长比例缩放)
        x_out = x * lambda_in / lambda_out
        Q_out = mie_scattering_efficiency(x_out, m_out)

        # 散射增强比
        gains[i] = Q_out / max(Q_in, 1e-10)

    return gains, x_values
