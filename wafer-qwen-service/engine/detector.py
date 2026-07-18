import cv2
import numpy as np

from models.schemas import Detection, BBox

# Qwen 位置描述 → 归一化 ROI [x, y, w, h] 的映射
REGION_MAP = {
    "右上":      [0.65, 0.0, 0.35, 0.35],
    "右上角":    [0.65, 0.0, 0.35, 0.35],
    "左上":      [0.0, 0.0, 0.35, 0.35],
    "左上角":    [0.0, 0.0, 0.35, 0.35],
    "右下":      [0.65, 0.65, 0.35, 0.35],
    "右下角":    [0.65, 0.65, 0.35, 0.35],
    "左下":      [0.0, 0.65, 0.35, 0.35],
    "左下角":    [0.0, 0.65, 0.35, 0.35],
    "中心":      [0.3, 0.3, 0.4, 0.4],
    "中央":      [0.3, 0.3, 0.4, 0.4],
    "左侧":      [0.0, 0.2, 0.3, 0.6],
    "右侧":      [0.7, 0.2, 0.3, 0.6],
    "顶部":      [0.2, 0.0, 0.6, 0.3],
    "底部":      [0.2, 0.7, 0.6, 0.3],
    "边缘":      [0.0, 0.0, 1.0, 1.0],
    "上边缘":    [0.0, 0.0, 1.0, 0.15],
    "下边缘":    [0.0, 0.85, 1.0, 0.15],
    "左边缘":    [0.0, 0.0, 0.15, 1.0],
    "右边缘":    [0.85, 0.0, 0.15, 1.0],
}


def _resolve_roi(region_desc: str) -> list[float]:
    """将 Qwen 的自然语言位置描述转为归一化 ROI"""
    for keyword, roi in REGION_MAP.items():
        if keyword in region_desc:
            return roi
    return [0.0, 0.0, 1.0, 1.0]  # 默认全图


def detect_defects(
    enhanced_img: np.ndarray,
    suspected_defects: list[dict],
    min_area: int = 20,
    confidence_threshold: float = 0.5,
) -> list[Detection]:
    """
    在 Qwen 标注的疑似区域中，用形态学方法检出缺陷。

    Args:
        enhanced_img: 增强后的灰度图 (H, W)
        suspected_defects: Qwen 输出的缺陷列表
            [{"type": "...", "confidence": 0.85, "region": "位置描述"}, ...]
        min_area: 最小轮廓面积（过滤噪声）
        confidence_threshold: 最低置信度阈值

    Returns:
        Detection 列表
    """
    h, w = enhanced_img.shape
    results = []

    for spec in suspected_defects:
        defect_type = spec.get("type", "未知")
        confidence = spec.get("confidence", 0.6)
        region_desc = spec.get("region", "全图")

        if confidence < confidence_threshold:
            continue

        # ROI 坐标 (归一化 → 像素)
        roi_norm = _resolve_roi(region_desc)
        rx = int(roi_norm[0] * w)
        ry = int(roi_norm[1] * h)
        rw = int(roi_norm[2] * w)
        rh = int(roi_norm[3] * h)

        # 裁剪 ROI
        roi_img = enhanced_img[ry:ry + rh, rx:rx + rw]
        if roi_img.size == 0:
            continue

        # 自适应阈值
        thresh = cv2.adaptiveThreshold(
            roi_img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2,
        )

        # 形态学开运算去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # 轮廓查找
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            # 将局部 ROI 坐标映射回全图坐标
            results.append(Detection(
                type=defect_type,
                confidence=round(min(confidence, 0.5 + area / 5000), 2),
                bbox=BBox(x=rx + bx, y=ry + by, w=bw, h=bh),
                source="qwen+cv",
            ))

    return results
