import numpy as np
from engine.detector import detect_defects
from models.schemas import Detection


def test_detect_defects_empty():
    """无缺陷时返回空列表"""
    img = np.ones((200, 200), dtype=np.uint8) * 128
    result = detect_defects(img, [], min_area=10)
    assert result == []


def test_detect_defects_with_spec():
    """有缺陷描述时返回检出结果"""
    img = np.ones((200, 200), dtype=np.uint8) * 128
    # 在右下角画一个暗色矩形模拟缺陷
    img[150:180, 150:180] = 30

    specs = [
        {"type": "颗粒", "confidence": 0.8, "region": "右下"},
    ]
    result = detect_defects(img, specs, min_area=10)
    assert len(result) > 0
    assert result[0].type == "颗粒"
    assert result[0].confidence > 0.5


def test_confidence_filter():
    """低于阈值的缺陷应过滤"""
    img = np.ones((100, 100), dtype=np.uint8) * 128
    specs = [
        {"type": "划痕", "confidence": 0.3, "region": "右侧"},
    ]
    result = detect_defects(img, specs, min_area=10, confidence_threshold=0.5)
    assert result == []


if __name__ == "__main__":
    test_detect_defects_empty()
    test_detect_defects_with_spec()
    test_confidence_filter()
    print("[OK] All detector tests passed")
