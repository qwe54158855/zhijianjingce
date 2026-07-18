import cv2
import numpy as np


def enhance(image: np.ndarray, params: dict) -> np.ndarray:
    img = image.copy()
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=params.get("clahe_clip", 2.5),
        tileGridSize=(params.get("clahe_grid", 8), params.get("clahe_grid", 8)),
    )
    img = clahe.apply(img)
    img = cv2.fastNlMeansDenoising(img, None, params.get("denoise_strength", 10), 7, 21)

    gamma = params.get("gamma", 1.2)
    img = (img / 255.0) ** gamma * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    contrast = params.get("contrast", 1.3)
    img = cv2.multiply(img, contrast)
    img = np.clip(img, 0, 255).astype(np.uint8)

    if params.get("sharpen", True):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32)
        img = cv2.filter2D(img, -1, kernel)

    return img.astype(np.uint8)


def auto_enhance(image: np.ndarray) -> np.ndarray:
    default_params = {
        "clahe_clip": 2.5, "clahe_grid": 8,
        "denoise_strength": 10, "gamma": 1.2,
        "contrast": 1.3, "sharpen": True,
    }
    return enhance(image, default_params)


def brightfield_enhance(image: np.ndarray, preserve_morphology: bool = True) -> np.ndarray:
    """
    暗场→亮场风格转换。保持原始形态学结构，仅调整亮度极性。

    Args:
        image: 暗场灰度图 (H, W), uint8 [0,255]
        preserve_morphology: 保持原始形态学细节（减少平滑）

    Returns:
        亮场风格灰度图，背景≈255，缺陷≈0
    """
    # 1. 轻度CLAHE（保持形态学，只做基本的对比度增强）
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    # 2. 不做NLM去噪（保留原始缺陷形态）
    # 3. 轻度Gamma
    enhanced = ((enhanced / 255.0) ** 1.1 * 255.0).clip(0, 255).astype(np.uint8)

    # 4. 反转：暗场（缺陷亮）→ 亮场（缺陷暗）
    inverted = 255 - enhanced

    # 5. 百分位对比度拉伸保持形态学
    flat = inverted.ravel()
    lo = max(0, int(np.percentile(flat, 2)))
    hi = min(255, int(np.percentile(flat, 98)))
    if hi - lo < 10:
        hi, lo = 255, 0
    result = ((inverted.astype(np.float32) - lo) / (hi - lo) * 255)
    result = result.clip(0, 255).astype(np.uint8)

    # 6. 最小形态学闭运算（仅合并相邻暗点，不改变缺陷形状）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    # 7. 轻微锐化还原细节
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    result = cv2.filter2D(result, -1, kernel)
    result = result.clip(0, 255).astype(np.uint8)

    return result
