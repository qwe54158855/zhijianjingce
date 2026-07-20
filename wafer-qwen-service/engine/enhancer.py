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
    暗场→亮场风格转换。缺陷边缘加深 + 暗区对比度强化，缺陷→纯黑、背景→纯白。

    Args:
        image: 暗场灰度图 (H, W), uint8 [0,255]
        preserve_morphology: 保持原始形态学细节（减少平滑）

    Returns:
        亮场风格灰度图，背景≈255，缺陷≈0（缺陷边缘加深、深度增强）
    """
    # 1. 轻度CLAHE（保持形态学，只做基本的对比度增强）
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)

    # 2. 轻度Gamma
    enhanced = ((enhanced / 255.0) ** 1.1 * 255.0).clip(0, 255).astype(np.uint8)

    # 3. 反转：暗场（缺陷亮）→ 亮场（缺陷暗）
    inverted = 255 - enhanced

    # 4. 百分位对比度拉伸（加强拉伸，让暗区更暗、亮区更亮）
    flat = inverted.ravel()
    lo = max(0, int(np.percentile(flat, 1)))       # 拉至1%分位（更激进）
    hi = min(255, int(np.percentile(flat, 99)))     # 拉至99%分位
    if hi - lo < 10:
        hi, lo = 255, 0
    result = ((inverted.astype(np.float32) - lo) / (hi - lo) * 255)
    result = result.clip(0, 255).astype(np.uint8)

    # 5. 形态学闭运算（合并相邻暗点）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)

    # === 6. 缺陷边缘加深 + 暗区深度增强 ===
    # 6a. 梯度边缘检测：提取暗缺陷的边缘轮廓
    grad_x = cv2.Sobel(result, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(result, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_mag = (grad_mag / grad_mag.max() * 255).clip(0, 255).astype(np.uint8)

    # 6b. 边缘加深：在原始图上减去梯度（缺陷边缘变黑）
    edge_boost = cv2.addWeighted(result.astype(np.float32), 1.0,
                                 grad_mag.astype(np.float32), -0.4, 0)
    edge_boost = edge_boost.clip(0, 255).astype(np.uint8)

    # 6c. 暗区局部对比度增强：对暗像素(<128)做更强拉伸
    dark_mask = (edge_boost < 128).astype(np.uint8) * 255
    # 对暗区做形态学闭运算扩大覆盖
    dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    # 暗区拉暗：暗像素再降低15%
    dark_region = edge_boost.astype(np.float32)
    dark_region = np.where(dark_mask > 0, dark_region * 0.85, dark_region)
    dark_region = dark_region.clip(0, 255).astype(np.uint8)

    # 6d. 柔和锐化还原细节
    sharpen = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    result = cv2.filter2D(dark_region, -1, sharpen)
    result = result.clip(0, 255).astype(np.uint8)

    # 6e. 最终再拉伸一次：确保背景趋近255、缺陷趋近0
    flat_final = result.ravel()
    lo_f = max(0, int(np.percentile(flat_final, 0.5)))
    hi_f = min(255, int(np.percentile(flat_final, 99.5)))
    if hi_f - lo_f < 10:
        hi_f, lo_f = 255, 0
    result = ((result.astype(np.float32) - lo_f) / (hi_f - lo_f) * 255)
    result = result.clip(0, 255).astype(np.uint8)

    return result
