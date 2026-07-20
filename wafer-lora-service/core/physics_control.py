"""
Physics-informed control for LoRA inference.

Provides gain map computation (simulating PISM output) as ControlNet
conditioning for wavelength conversion mode.
"""
import logging

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


def compute_gain_map(image: Image.Image, lambda_in: float = 266.0, lambda_out: float = 193.0) -> Image.Image:
    """
    Compute simplified Rayleigh scattering gain map.

    This approximates PISM's output for ControlNet conditioning.
    The full PISM gain map requires the wafer-inspection model;
    this is a lightweight analytical approximation.

    Args:
        image: Input 266nm wafer image (grayscale compatible)
        lambda_in: Input wavelength in nm (266)
        lambda_out: Output wavelength in nm (193)

    Returns:
        Gain map image (grayscale, 0-255)
    """
    # Rayleigh scattering ratio
    rayleigh_gain = (lambda_in / lambda_out) ** 4  # ~3.5 for 266->193

    # Convert to numpy
    if image.mode != "L":
        gray = image.convert("L")
    else:
        gray = image.copy()
    img_array = np.array(gray, dtype=np.float32)

    # Local contrast as defect proxy (high contrast = defect = higher gain)
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(img_array, size=15)
    local_contrast = np.abs(img_array - local_mean)

    # Normalize contrast to [0, 1]
    contrast_norm = np.clip(local_contrast / 64.0, 0, 1)

    # Combine: base Rayleigh gain + defect modulation
    # Low contrast regions -> near Rayleigh gain
    # High contrast regions -> enhanced gain (simulating Mie resonance)
    gain_array = rayleigh_gain * (1.0 + 0.5 * contrast_norm)

    # Clamp to physical range [1.0, 8.0]
    gain_array = np.clip(gain_array, 1.0, 8.0)

    # Scale to 8-bit for ControlNet
    gain_8bit = ((gain_array - 1.0) / 7.0 * 255).astype(np.uint8)

    return Image.fromarray(gain_8bit, mode="L").convert("RGB")
