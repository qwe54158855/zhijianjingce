"""
Test inference route with mock model.
Run: pytest tests/test_infer.py -v
"""
import pytest
from PIL import Image
from api.routes.infer import generate_control_image, encode_image_to_base64, decode_base64_image


def test_image_roundtrip():
    img = Image.new("RGB", (64, 64), color="gray")
    b64 = encode_image_to_base64(img)
    decoded = decode_base64_image(b64)
    assert decoded.size == (64, 64)


def test_control_image_enhance():
    img = Image.new("RGB", (64, 64), color="white")
    control = generate_control_image(img, "enhance")
    assert control.size == (64, 64)
