import cv2
import numpy as np


def generate_angle_views(
    brightfield_img: np.ndarray,
    angles: list[int] = None,
) -> dict[int, np.ndarray]:
    """
    生成多角度 193nm 亮场视图（保持形态学细节）。

    原理：极坐标展开 → 方位角滚动（torch.roll）→ 反变换。
    滚动量 Δx = θ × W / 360，在极坐标中等价于旋转 Δθ。

    Args:
        brightfield_img: 亮场增强灰度图 (H, W)
        angles: 角度列表，默认 15° 到 75°，步长 5°

    Returns:
        {角度: 图像} 字典
    """
    if angles is None:
        angles = list(range(15, 76, 5))

    h, w = brightfield_img.shape
    center_x, center_y = w // 2, h // 2
    radius = min(center_x, center_y) - 5

    # 极坐标展开（保留形态学细节，使用 INTER_NEAREST 减少插值模糊）
    polar_w, polar_h = 512, 128
    polar = cv2.warpPolar(
        brightfield_img,
        (polar_w, polar_h),
        (center_x, center_y),
        radius,
        cv2.WARP_FILL_OUTLIERS + cv2.INTER_NEAREST,
    )

    results = {}
    for angle in angles:
        # 方位角滚动（零成本旋转）
        shift = int(angle * polar_w / 360)
        rolled = np.roll(polar, shift, axis=1)

        # 散射强度调制：角度越大散射增益越大，但保持形态学
        gain = 0.9 + 0.2 * (angle - 15) / 60
        rolled = np.clip(rolled.astype(np.float32) * gain, 0, 255).astype(np.uint8)

        # 反变换回笛卡尔（同样用 INTER_NEAREST 保留边缘）
        angle_img = cv2.warpPolar(
            rolled,
            (w, h),
            (center_x, center_y),
            radius,
            cv2.WARP_INVERSE_MAP + cv2.INTER_NEAREST,
        )

        results[angle] = angle_img

    return results
