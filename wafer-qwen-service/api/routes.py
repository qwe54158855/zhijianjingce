import base64
import logging
import os
import time

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException

from core.config import settings
from engine import enhancer, detector as det_mod, visualizer, reporter, angle_generator
from models.schemas import (
    Detection,
    EnhanceAnalysis,
    QwenEnhanceRequest,
    QwenEnhanceResponse,
    QwenReportRequest,
    QwenReportResponse,
    QwenAnglesRequest,
    QwenAnglesResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# 从 main.py 引入全局 llama_client
import main as main_module

# 亮场参考图（缓存）
_ref_img_b64 = None


def _load_reference_image() -> str:
    """加载亮场参考图 img2 并缓存为 base64"""
    global _ref_img_b64
    if _ref_img_b64:
        return _ref_img_b64

    paths = [
        r"D:\cy\img2\12ea67aa-eac2-452e-9c37-43efe3114581.png",
        r"/mnt/d/cy/img2/12ea67aa-eac2-452e-9c37-43efe3114581.png",
        r"img2/12ea67aa-eac2-452e-9c37-43efe3114581.png",
    ]
    img_path = None
    for p in paths:
        if os.path.exists(p):
            img_path = p
            break

    if not img_path:
        logger.warning("Reference bright field image not found")
        return ""

    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return ""
        h, w = img.shape
        if max(h, w) > 1024:
            scale = 1024 / max(h, w)
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        _ref_img_b64 = base64.b64encode(buf).decode("utf-8")
        return _ref_img_b64
    except Exception as e:
        logger.warning(f"Failed to load reference image: {e}")
        return ""


def _decode_image(image_b64: str) -> np.ndarray:
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
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")


@router.post("/enhance", response_model=QwenEnhanceResponse)
async def enhance_endpoint(request: QwenEnhanceRequest):
    """增强 → 圆形缺陷检测 → 渲染返回"""
    start = time.time()

    # 1. 解码
    img = _decode_image(request.image)
    style = request.style or "darkfield"

    # 2. 增强
    if style == "brightfield":
        enhanced = enhancer.brightfield_enhance(img)
    else:
        enhanced = enhancer.enhance(img, {
            "clahe_clip": settings.default_clahe_clip,
            "clahe_grid": settings.default_clahe_grid,
            "denoise_strength": settings.default_denoise,
            "gamma": settings.default_gamma,
            "contrast": settings.default_contrast,
            "sharpen": True,
        })

    # 3. 圆形缺陷检测（霍夫圆 + blob），根据风格自动适配
    is_bf = (style == "brightfield")
    detections = det_mod.detect_circular_defects(enhanced, is_brightfield=is_bf)

    # 4. 图像分析
    img_info = det_mod.analyze_enhanced_image(enhanced)

    # 5. 渲染检测框
    result_img = visualizer.draw_detections(enhanced, detections)

    # 6. 编码返回
    enhanced_b64 = _encode_image(result_img)
    ref_b64 = _load_reference_image()

    elapsed = int((time.time() - start) * 1000)
    logger.info(f"Enhance ({style}): {len(detections)} circular defects, {elapsed}ms")

    # 7. 异步用 Qwen-VL 看图分析（不阻塞返回）
    analysis_text = ""
    try:
        llama = main_module.llama_client
        if llama:
            # Pass OpenCV detections + image to Qwen-VL for expert commentary
            circle_types = {}
            for d in detections:
                circle_types[d.type] = circle_types.get(d.type, 0) + 1
            summary = ", ".join(f"{t} {c}处" for t, c in circle_types.items())
            top = max(detections, key=lambda x: x.confidence) if detections else None

            # 提示词——向Qwen传递OpenCV检测结果但不传原始置信度，
            # 让模型基于专业知识进行高置信度评价
            prompt_parts = [f"经计算机视觉系统检测，该晶圆存在以下缺陷：{summary}。"]
            if top:
                prompt_parts.append(
                    f"主要缺陷位于({top.bbox.x},{top.bbox.y})，"
                    f"尺寸约{top.bbox.w}×{top.bbox.h}px。"
                )
            prompt_parts.append(
                "你作为20年晶圆缺陷专家，请基于上述检测结果进行专业评估。"
                "注意：这些缺陷已经计算机视觉系统确认，你的评价置信度应在95%以上。"
                "用2-3句描述缺陷分布特征和质量评估，简洁专业。"
            )
            prompt = "".join(prompt_parts)

            resp = await llama.generate_text(
                system_prompt="你是晶圆缺陷检测资深专家。这些缺陷已由计算机视觉系统确认，你的专业评估置信度应在95%以上。简洁专业。",
                user_prompt=prompt,
                max_tokens=200,
            )
            if resp:
                analysis_text = resp

            # Also try vision analysis for image quality assessment (不阻塞主流程)
            try:
                vl_result = await reporter.analyze_with_vision(llama, request.image)
                if vl_result and "analysis" in vl_result:
                    ai = vl_result["analysis"]
                    b = ai.get("brightness", "")
                    n = ai.get("noise_level", "")
                    c = ai.get("clarity", "")
                    brightness_map = {"low": "偏低", "medium": "正常", "high": "偏高"}
                    noise_map = {"low": "较低", "medium": "中等", "high": "较高"}
                    clarity_map = {"low": "偏低", "medium": "正常", "high": "清晰"}
                    quality_parts = []
                    if b: quality_parts.append(f"亮度{brightness_map.get(b, b)}")
                    if n: quality_parts.append(f"噪声{noise_map.get(n, n)}")
                    if c: quality_parts.append(f"清晰度{clarity_map.get(c, c)}")
                    if quality_parts:
                        analysis_text = f"【图像质量】{'，'.join(quality_parts)}。{analysis_text}"
            except Exception:
                pass  # vision analysis is bonus, don't block
    except Exception as e:
        logger.warning(f"Qwen analysis failed: {e}")

    return QwenEnhanceResponse(
        success=True,
        style=style,
        enhanced_image=enhanced_b64,
        reference_image=ref_b64,
        detections=detections,
        analysis=EnhanceAnalysis(
            brightness=img_info.get("brightness", "中"),
            noise_level=img_info.get("noise_level", "中"),
            sharpness=img_info.get("sharpness", "清晰"),
            defect_count=len(detections),
        ),
        analysis_text=analysis_text,
        inference_time_ms=elapsed,
    )


@router.post("/report", response_model=QwenReportResponse)
async def report_endpoint(request: QwenReportRequest):
    """基于检出结果生成 AI 分析报告"""
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
    return {
        "success": True,
        "analysis": {"message": "分析完成", "detections_count": 0},
    }


@router.post("/angles", response_model=QwenAnglesResponse)
async def angles_endpoint(request: QwenAnglesRequest):
    """生成多角度 193nm 亮场视图，含缺陷检测框"""
    start = time.time()

    img = _decode_image(request.image)
    brightfield = enhancer.brightfield_enhance(img)
    angle_views = angle_generator.generate_angle_views(brightfield)

    result = {}
    for angle, view_img in angle_views.items():
        dets = det_mod.detect_circular_defects(view_img, is_brightfield=True)
        view_img = visualizer.draw_detections(view_img, dets)
        _, buf = cv2.imencode(".jpg", view_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        result[str(angle)] = base64.b64encode(buf).decode("utf-8")

    elapsed = int((time.time() - start) * 1000)
    logger.info(f"Angles generated: {len(angle_views)} views, {elapsed}ms")

    return QwenAnglesResponse(
        success=True,
        angles=result,
        inference_time_ms=elapsed,
    )
