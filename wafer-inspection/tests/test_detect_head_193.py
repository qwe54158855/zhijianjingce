import torch
from models.detect_head_193 import WaferDetectHead193


def test_head_init():
    """检测头应初始化指定结构"""
    head = WaferDetectHead193(num_classes=4, ch_in=[112, 224, 448], ch_hidden=[56, 64, 128])

    assert hasattr(head, 'cls_f2')
    assert hasattr(head, 'cls_f3')
    assert hasattr(head, 'cls_f4')
    assert hasattr(head, 'reg_f2')
    assert hasattr(head, 'reg_f3')
    assert hasattr(head, 'reg_f4')


def test_head_forward_shape():
    """检测头前向应输出 3 层 cls/reg"""
    head = WaferDetectHead193(num_classes=4)
    head.eval()

    feats = [
        torch.randn(2, 112, 64, 64),   # F2 原始 112 (ch_in)
        torch.randn(2, 224, 32, 32),   # F3 原始 224 (ch_in)
        torch.randn(2, 448, 16, 16),  # F4 原始 448 (ch_in)
    ]

    outputs = head(feats)
    assert len(outputs) == 3, f"Expected 3 outputs, got {len(outputs)}"

    for i, (cls, reg) in enumerate(outputs):
        assert cls.shape[1] == 4, f"cls ch should be 4, got {cls.shape[1]}"
        assert reg.shape[1] == 4, f"reg ch should be 4, got {reg.shape[1]}"


def test_head_parameter_count():
    """193nm 检测头参数量应 < 0.5M"""
    head = WaferDetectHead193(num_classes=4)
    total = sum(p.numel() for p in head.parameters())
    assert total < 500000, f"Parameter count: {total}"


def test_head_forward_differentiable():
    """检测头输出应可微分"""
    head = WaferDetectHead193(num_classes=4)
    feats = [
        torch.randn(1, 112, 16, 16, requires_grad=True),
        torch.randn(1, 224, 8, 8, requires_grad=True),
        torch.randn(1, 448, 4, 4, requires_grad=True),
    ]

    outputs = head(feats)
    loss = sum(o[0].sum() + o[1].sum() for o in outputs)
    loss.backward()

    for f in feats:
        assert f.grad is not None, "Gradient did not flow!"
        assert f.grad.abs().sum() > 0


def test_head_output_is_list():
    """检测头前向输出应为 List[Tuple[Tensor, Tensor]] 结构"""
    head = WaferDetectHead193(num_classes=4)
    head.eval()

    feats = [
        torch.randn(1, 112, 32, 32),
        torch.randn(1, 224, 16, 16),
        torch.randn(1, 448, 8, 8),
    ]

    outputs = head(feats)
    assert isinstance(outputs, list), "Output must be list"
    assert len(outputs) == 3
    for i, (cls, reg) in enumerate(outputs):
        assert cls.shape[1] == 4
        assert reg.shape[1] == 4
