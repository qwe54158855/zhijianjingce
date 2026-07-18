import numpy as np
from engine.visualizer import draw_detections
from models.schemas import Detection, BBox


def test_draw_detections_grayscale_input():
    """灰度图输入应返回 BGR 彩色图"""
    img = np.ones((100, 100), dtype=np.uint8) * 128
    dets = [
        Detection(type="崩边", confidence=0.85,
                  bbox=BBox(x=10, y=10, w=20, h=15), source="qwen+cv"),
    ]
    result = draw_detections(img, dets)
    assert result.shape == (100, 100, 3)


def test_draw_detections_empty():
    """无检出时返回原始灰度转 BGR"""
    img = np.ones((50, 50), dtype=np.uint8) * 128
    result = draw_detections(img, [])
    assert result.shape == (50, 50, 3)


def test_draw_detections_multiple_types():
    """多类缺陷应正确染色"""
    img = np.ones((200, 200), dtype=np.uint8) * 128
    dets = [
        Detection(type="崩边", confidence=0.85,
                  bbox=BBox(x=10, y=10, w=20, h=15), source="qwen+cv"),
        Detection(type="颗粒", confidence=0.75,
                  bbox=BBox(x=100, y=100, w=25, h=25), source="qwen+cv"),
        Detection(type="划痕", confidence=0.65,
                  bbox=BBox(x=50, y=150, w=80, h=5), source="qwen+cv"),
    ]
    result = draw_detections(img, dets)
    assert result.shape == (200, 200, 3)


if __name__ == "__main__":
    test_draw_detections_grayscale_input()
    test_draw_detections_empty()
    test_draw_detections_multiple_types()
    print("[OK] All visualizer tests passed")
