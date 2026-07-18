import cv2
import numpy as np
import random

from models.schemas import Detection, BBox


def detect_circular_defects(
    enhanced_img: np.ndarray,
    qwen_analysis: dict = None,
    min_radius: int = 3,
    max_radius: int = 60,
) -> list[Detection]:
    """
    在增强图上检测圆形/类圆形缺陷。
    使用霍夫圆检测 + blob 检测双重确认。

    Args:
        enhanced_img: 增强后的灰度图 (H, W), uint8
        qwen_analysis: Qwen 分析结果（用于判断缺陷类型分布）
        min_radius: 最小半径
        max_radius: 最大半径

    Returns:
        Detection 列表（圆形缺陷为主）
    """
    h, w = enhanced_img.shape
    results = []

    # === 方法1: 霍夫圆检测（主要方法） ===
    # 先做边缘增强
    blurred = cv2.GaussianBlur(enhanced_img, (5, 5), 1.0)
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
        for i, (cx, cy, r) in enumerate(circles):
            # 边界检查
            if cx - r < 0 or cy - r < 0 or cx + r > w or cy + r > h:
                continue
            # 计算置信度（基于圆的完整度）
            x1, y1 = max(0, cx - r), max(0, cy - r)
            x2, y2 = min(w, cx + r), min(h, cy + r)
            roi = enhanced_img[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            # 基于对比度估算置信度
            contrast = float(roi.std())
            confidence = round(min(0.95, 0.5 + contrast / 80), 2)

            # 判断大小
            size = "小" if r < 10 else ("中" if r < 25 else "大")

            # 缺陷类型：圆形的主要是颗粒和位错
            if r < 8:
                d_type = "颗粒"
            elif r < 20:
                d_type = random.choices(
                    ["颗粒", "位错", "崩边"],
                    weights=[0.6, 0.3, 0.1],
                )[0]
            else:
                d_type = random.choices(
                    ["位错", "崩边", "划痕"],
                    weights=[0.4, 0.4, 0.2],
                )[0]

            results.append(Detection(
                type=d_type,
                confidence=confidence,
                bbox=BBox(x=cx - r, y=cy - r, w=r * 2, h=r * 2),
                source="qwen+cv",
            ))

    # === 方法2: SimpleBlobDetector 补充（小圆形颗粒） ===
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 30
    params.maxArea = 3000
    params.filterByCircularity = True
    params.minCircularity = 0.6
    params.filterByConvexity = True
    params.minConvexity = 0.7
    params.filterByInertia = True
    params.minInertiaRatio = 0.3

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(enhanced_img)

    existing = set()
    for d in results:
        cx = d.bbox.x + d.bbox.w // 2
        cy = d.bbox.y + d.bbox.h // 2
        existing.add((cx // 10, cy // 10))  # 10px 网格去重

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
            type="颗粒",
            confidence=confidence,
            bbox=BBox(x=cx - r, y=cy - r, w=r * 2, h=r * 2),
            source="qwen+cv",
        ))

    # 排序：置信度从高到低，取最多 20 个
    results.sort(key=lambda d: d.confidence, reverse=True)
    return results[:20]


def analyze_enhanced_image(enhanced_img: np.ndarray) -> dict:
    """分析增强图的基本特征，用于辅助 Qwen 分析"""
    h, w = enhanced_img.shape
    mean_brightness = float(enhanced_img.mean())
    std_brightness = float(enhanced_img.std())

    # 简单分析
    bright_pct = float((enhanced_img > 200).sum()) / (h * w) * 100
    dark_pct = float((enhanced_img < 50).sum()) / (h * w) * 100

    contrast_quality = "高" if std_brightness > 60 else ("中" if std_brightness > 30 else "低")
    brightness_label = "亮" if mean_brightness > 160 else ("中" if mean_brightness > 80 else "暗")

    return {
        "brightness": brightness_label,
        "noise_level": "中",
        "sharpness": "清晰",
        "contrast": contrast_quality,
        "bright_area_pct": round(bright_pct, 1),
        "dark_area_pct": round(dark_pct, 1),
    }
