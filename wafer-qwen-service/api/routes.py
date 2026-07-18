import base64
import logging
import time

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException

from core.config import settings
from engine import enhancer, detector as det_mod, visualizer, reporter
from models.schemas import (
    Detection,
    EnhanceAnalysis,
    QwenEnhanceRequest,
    QwenEnhanceResponse,
    QwenReportRequest,
    QwenReportResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# 从 main.py 引入全局 llama_client
import main as main_module


def _decode_image(image_b64: str) -> np.ndarray:
    """解码 base64 图片为 OpenCV 格式"""
    try:
        img_bytes = base64.b64decode(image_b64)
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Image decode failed")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


def _encode_image(img: np.ndarray) -> str:
    """编码 OpenCV 图为 base64"""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")


@router.post("/enhance", response_model=QwenEnhanceResponse)
async def enhance_endpoint(request: QwenEnhanceRequest):
    """Qwen 分析 → OpenCV 增强 → 缺陷检出 → 渲染返回。"""
    start = time.time()

    # 1. 解码图像
    img = _decode_image(request.image)
    img_b64 = request.image

    # 2. Qwen 分析
    llama = main_module.llama_client
    if llama is None:
        raise HTTPException(status_code=503, detail="Llama client not initialized")

    analysis_result = await llama.analyze_image(
        image_base64=img_b64,
        system_prompt=reporter.ENHANCE_SYSTEM,
        user_prompt=reporter.ENHANCE_USER,
    )

    if analysis_result is None:
        # Qwen 分析失败，使用默认参数
        logger.warning("Qwen analysis failed, using defaults")
        enhance_params = {
            "clahe_clip": settings.default_clahe_clip,
            "clahe_grid": settings.default_clahe_grid,
            "denoise_strength": settings.default_denoise,
            "gamma": settings.default_gamma,
            "contrast": settings.default_contrast,
            "sharpen": True,
        }
        suspected_defects = []
    else:
        enhance_params = analysis_result.get("enhance_params", {})
        suspected_defects = (
            analysis_result.get("analysis", {}).get("suspected_defects", [])
        )

    # 3. OpenCV 增强
    enhanced = enhancer.enhance(img, enhance_params)

    # 4. 缺陷检出
    detections = det_mod.detect_defects(
        enhanced,
        suspected_defects,
        min_area=settings.min_contour_area,
        confidence_threshold=settings.detection_confidence_threshold,
    )

    # 5. 渲染检测框
    result_img = visualizer.draw_detections(enhanced, detections)

    # 6. 编码返回
    enhanced_b64 = _encode_image(result_img)

    elapsed = int((time.time() - start) * 1000)
    logger.info(f"Enhance complete: {len(detections)} defects, {elapsed}ms")

    return QwenEnhanceResponse(
        success=True,
        enhanced_image=enhanced_b64,
        detections=detections,
        analysis=EnhanceAnalysis(
            brightness=analysis_result.get("analysis", {}).get("brightness", "中")
            if analysis_result else "中",
            noise_level=analysis_result.get("analysis", {}).get("noise_level", "中")
            if analysis_result else "中",
            sharpness=analysis_result.get("analysis", {}).get("sharpness", "清晰")
            if analysis_result else "清晰",
            defect_count=len(detections),
        ),
        inference_time_ms=elapsed,
    )


@router.post("/report", response_model=QwenReportResponse)
async def report_endpoint(request: QwenReportRequest):
    """基于增强图和检出结果生成分析报告"""
    start = time.time()

    llama = main_module.llama_client
    if llama is None:
        raise HTTPException(status_code=503, detail="Llama client not initialized")

    report_text = await reporter.generate_report(
        llama, request.image, request.detections,
    )

    elapsed = int((time.time() - start) * 1000)

    return QwenReportResponse(
        success=True,
        report=report_text or "报告生成失败",
        inference_time_ms=elapsed,
    )


@router.post("/analyze")
async def analyze_only(request: QwenEnhanceRequest):
    """仅做 Qwen 分析，不增强"""
    llama = main_module.llama_client
    if llama is None:
        raise HTTPException(status_code=503, detail="Llama client not initialized")

    result = await llama.analyze_image(
        image_base64=request.image,
        system_prompt=reporter.ENHANCE_SYSTEM,
        user_prompt=reporter.ENHANCE_USER,
    )

    return {
        "success": result is not None,
        "analysis": result,
    }
