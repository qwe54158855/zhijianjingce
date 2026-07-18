import cv2
import numpy as np

from models.schemas import Detection

# YOLO 风格类别颜色 (BGR)
CLASS_COLORS = {
    "崩边": (0, 0, 255),       # 红色
    "颗粒": (255, 191, 0),     # 青色
    "划痕": (0, 255, 0),       # 绿色
    "位错": (255, 0, 255),     # 紫色
    "未知": (128, 128, 128),   # 灰色
}


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    """
    在增强图上绘制 YOLO 风格的检测框。

    Args:
        image: 灰度图或 BGR 图
        detections: Detection 列表

    Returns:
        BGR 彩色图（带检测框）
    """
    # 转 BGR
    if len(image.shape) == 2:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_rgb = image.copy()

    for det in detections:
        color = CLASS_COLORS.get(det.type, CLASS_COLORS["未知"])
        b = det.bbox

        # 画矩形框（2px 实线）
        cv2.rectangle(
            img_rgb,
            (b.x, b.y),
            (b.x + b.w, b.y + b.h),
            color, 2,
        )

        # 标签文字
        label = f"{det.type} {det.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        # 标签背景填充
        label_bg_x1 = b.x
        label_bg_y1 = b.y - th - 6
        label_bg_x2 = b.x + tw + 6
        label_bg_y2 = b.y

        # 确保标签不超出图像上边界
        if label_bg_y1 < 0:
            label_bg_y1 = b.y
            label_bg_y2 = b.y + th + 6

        cv2.rectangle(
            img_rgb,
            (label_bg_x1, label_bg_y1),
            (label_bg_x2, label_bg_y2),
            color, -1,  # 填充
        )

        # 标签文字
        label_y = label_bg_y2 - 3 if label_bg_y1 < b.y else label_bg_y1 + th + 4
        cv2.putText(
            img_rgb, label,
            (b.x + 3, label_y),
            font, font_scale,
            (255, 255, 255), thickness, cv2.LINE_AA,
        )

    return img_rgb
