import cv2
import numpy as np


def enhance(image: np.ndarray, params: dict) -> np.ndarray:
    """
    根据 Qwen 分析的参数执行 OpenCV 增强流水线。

    Args:
        image: 输入灰度图 (H, W) 或彩色图 (H, W, 3)，uint8 [0,255]
        params: 增强参数字典，来自 Qwen 分析输出

    Returns:
        增强后的灰度图 (H, W), uint8 [0,255]
    """
    img = image.copy()

    # 转灰度
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. CLAHE 自适应直方图均衡
    clahe = cv2.createCLAHE(
        clipLimit=params.get("clahe_clip", 2.5),
        tileGridSize=(
            params.get("clahe_grid", 8),
            params.get("clahe_grid", 8),
        ),
    )
    img = clahe.apply(img)

    # 2. NLM 去噪（保留边缘）
    img = cv2.fastNlMeansDenoising(
        img, None,
        params.get("denoise_strength", 10),
        7, 21,
    )

    # 3. Gamma 校正（暗场提亮）
    gamma = params.get("gamma", 1.2)
    img = (img / 255.0) ** gamma * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 4. 对比度拉伸
    contrast = params.get("contrast", 1.3)
    img = cv2.multiply(img, contrast)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 5. 锐化
    if params.get("sharpen", True):
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]], dtype=np.float32)
        img = cv2.filter2D(img, -1, kernel)

    return img.astype(np.uint8)


def auto_enhance(image: np.ndarray) -> np.ndarray:
    """无 Qwen 参数时的默认增强（用于快速测试）。"""
    default_params = {
        "clahe_clip": 2.5,
        "clahe_grid": 8,
        "denoise_strength": 10,
        "gamma": 1.2,
        "contrast": 1.3,
        "sharpen": True,
    }
    return enhance(image, default_params)


def brightfield_enhance(image: np.ndarray, bg_brightness: int = 220) -> np.ndarray:
    """
    暗场→亮场风格转换：缺陷变为黑色，背景变为亮色。

    原理：
    1. 先做标准暗场增强（缺陷变亮，背景变灰）
    2. 反转像素值（255 - x），使缺陷变暗、背景变亮
    3. 调整对比度和 gamma 使背景更亮、缺陷更黑

    Args:
        image: 输入暗场灰度图 (H, W)，uint8 [0,255]
        bg_brightness: 目标背景亮度 (默认 220/255)

    Returns:
        亮场风格灰度图 (H, W), uint8 [0,255]，缺陷呈黑色
    """
    # 1. 标准暗场增强 — 让缺陷信号放大
    enhanced = auto_enhance(image)

    # 2. 反转：暗场（缺陷亮）→ 亮场（缺陷暗）
    inverted = 255 - enhanced

    # 3. 调整对比度：拉伸到目标范围
    #   - 背景（原本的暗部）→ bg_brightness
    #   - 缺陷（原本的亮部）→ 更低值（变黑）
    result = inverted.astype(np.float32)
    result = (result / 255.0) ** 0.85 * bg_brightness
    result = np.clip(result, 0, 255).astype(np.uint8)

    # 4. 轻度锐化
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]], dtype=np.float32)
    result = cv2.filter2D(result, -1, kernel)

    return result.astype(np.uint8)
