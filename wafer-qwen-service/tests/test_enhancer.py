import cv2
import numpy as np

from engine.enhancer import enhance, auto_enhance


def test_enhance_output_shape():
    """增强输出应与输入尺寸一致"""
    dummy = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    result = auto_enhance(dummy)
    assert result.shape == (512, 512), f"Shape mismatch: {result.shape}"
    assert result.dtype == np.uint8


def test_enhance_with_params():
    """带自定义参数的增强"""
    dummy = np.ones((100, 100), dtype=np.uint8) * 50
    params = {
        "clahe_clip": 3.0,
        "clahe_grid": 4,
        "denoise_strength": 5,
        "gamma": 0.8,
        "contrast": 1.0,
        "sharpen": False,
    }
    result = enhance(dummy, params)
    assert result.shape == (100, 100)
    assert result.mean() > 50


def test_enhance_color_input():
    """彩色输入应正确处理"""
    dummy = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    result = auto_enhance(dummy)
    assert len(result.shape) == 2  # 输出应为灰度


if __name__ == "__main__":
    test_enhance_output_shape()
    test_enhance_with_params()
    test_enhance_color_input()
    print("[OK] All enhancer tests passed")
