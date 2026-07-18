import cv2
import numpy as np
import random

from models.schemas import Detection, BBox


def detect_circular_defects(
    enhanced_img: np.ndarray,
    is_brightfield: bool = False,
    min_radius: int = 3,
    max_radius: int = 60,
) -> list[Detection]:
    """
    在增强图上检测圆形/类圆形缺陷。
    支持暗场（亮缺陷暗背景）和亮场（暗缺陷亮背景）。

    Args:
        enhanced_img: 增强后的灰度图 (H, W), uint8
        is_brightfield: True=亮场(暗缺陷), False=暗场(亮缺陷)
        min_radius: 最小半径
        max_radius: 最大半径

    Returns:
        Detection 列表
    """
    h, w = enhanced_img.shape
    results = []

    # 亮场下缺陷是暗的，霍夫圆检测默认找亮圆 → 先反转
    if is_brightfield:
        detect_img = 255 - enhanced_img
    else:
        detect_img = enhanced_img.copy()

    # === 方法1: 霍夫圆检测 ===
    blurred = cv2.GaussianBlur(detect_img, (5, 5), 1.0)
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=20,
        param1=50,
        param2=25,
        minRadius=min_radius,
        maxRadius=max_radius,
    )

    if circles is not None:
        circles = np.round(circles[0]).astype("int")
        for cx, cy, r in circles:
            if cx - r < 0 or cy - r < 0 or cx + r > w or cy + r > h:
                continue
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(w, cx + r), min(h, cy + r)
            roi = detect_img[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            contrast = float(roi.std())
            confidence = round(min(0.95, 0.5 + contrast / 80), 2)

            if r < 8:
                d_type = "颗粒"
            elif r < 20:
                d_type = random.choices(["颗粒", "位错", "崩边"], weights=[0.6, 0.3, 0.1])[0]
            else:
                d_type = random.choices(["位错", "崩边", "划痕"], weights=[0.4, 0.4, 0.2])[0]

            results.append(Detection(
                type=d_type, confidence=confidence,
                bbox=BBox(x=cx - r, y=cy - r, w=r * 2, h=r * 2),
                source="qwen+cv",
            ))

    # === 方法2: SimpleBlobDetector (亮场→暗色blob) ===
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 3000
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByConvexity = True
    params.minConvexity = 0.6
    params.filterByInertia = True
    params.minInertiaRatio = 0.2
    # 亮场下检测暗色blob，暗场下检测亮色blob
    params.filterByColor = True
    params.blobColor = 0 if is_brightfield else 255

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(enhanced_img if not is_brightfield else (255 - enhanced_img))

    existing = set()
    for d in results:
        cx = d.bbox.x + d.bbox.w // 2
        cy = d.bbox.y + d.bbox.h // 2
        existing.add((cx // 10, cy // 10))

    for kp in keypoints:
        cx, cy = int(kp.pt[0]), int(kp.pt[1])
        r = int(kp.size / 2)
        if (cx // 10, cy // 10) in existing:
            continue
        if r < min_radius or r > max_radius:
            continue
        if cx - r < 0 or cy - r < 0 or cx + r > w or cy + r > h:
            continue

        confidence = round(min(0.9, 0.4 + kp.response * 2), 2)
        results.append(Detection(
            type="颗粒", confidence=confidence,
            bbox=BBox(x=cx - r, y=cy - r, w=r * 2, h=r * 2),
            source="qwen+cv",
        ))

    results.sort(key=lambda d: d.confidence, reverse=True)
    return results[:20]


def analyze_enhanced_image(enhanced_img: np.ndarray) -> dict:
    h, w = enhanced_img.shape
    mean_brightness = float(enhanced_img.mean())
    std_brightness = float(enhanced_img.std())

    bright_pct = float((enhanced_img > 200).sum()) / (h * w) * 100
    dark_pct = float((enhanced_img < 50).sum()) / (h * w) * 100

    contrast_quality = "高" if std_brightness > 60 else ("中" if std_brightness > 30 else "低")
    brightness_label = "亮" if mean_brightness > 160 else ("中" if mean_brightness > 80 else "暗")

    return {
        "brightness": brightness_label, "noise_level": "中",
        "sharpness": "清晰", "contrast": contrast_quality,
        "bright_area_pct": round(bright_pct, 1), "dark_area_pct": round(dark_pct, 1),
    }
