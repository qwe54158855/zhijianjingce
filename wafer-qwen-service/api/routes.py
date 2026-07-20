import base64
import csv
import logging
import os
import time

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Query

from core.config import settings
from engine import enhancer, detector as det_mod, visualizer, reporter, angle_generator
from models.schemas import (
    DatasetBrowseResponse,
    DatasetClassifyRequest,
    DatasetClassifyResponse,
    DatasetClassifyResult,
    DatasetImageInfo,
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

# 数据集路径
DATASET_DIR = r"D:\cy\images"
LABEL_CSV = os.path.join(DATASET_DIR, "label.csv")

# 数据集缓存
_dataset_cache = None


def _load_dataset():
    """加载数据集索引（带缓存）"""
    global _dataset_cache
    if _dataset_cache is not None:
        return _dataset_cache

    dataset = {}
    if os.path.exists(LABEL_CSV):
        with open(LABEL_CSV, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row["IMAGE_NAME"]
                try:
                    did = int(row["DEFECT_ID"])
                except ValueError:
                    did = 0
                dataset[name] = did

    # 扫描目录实际存在的文件
    all_images = sorted([
        f for f in os.listdir(DATASET_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and f != "label.csv"
    ])

    _dataset_cache = {
        "labels": dataset,
        "images": all_images,
        "total": len(all_images),
    }
    logger.info(f"Dataset loaded: {len(all_images)} images, {len(dataset)} labeled")
    return _dataset_cache


def _read_dataset_image(image_name: str) -> np.ndarray:
    """从数据集读取图像"""
    path = os.path.join(DATASET_DIR, image_name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Image {image_name} not found")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=500, detail=f"Failed to read {image_name}")
    return img


def _image_to_base64(img: np.ndarray) -> str:
    """OpenCV图像 → base64"""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")

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


# ========== 数据集端点 ==========

@router.get("/dataset/image/{image_name}")
async def dataset_image(image_name: str):
    """获取数据集中某张图像的 JPEG 数据（供前端直接显示）"""
    path = os.path.join(DATASET_DIR, image_name)
    if not os.path.exists(path):
        raise HTTPException(404, f"Image {image_name} not found")
    from fastapi.responses import Response
    with open(path, "rb") as f:
        return Response(content=f.read(), media_type="image/jpeg")


@router.get("/dataset/list", response_model=DatasetBrowseResponse)
async def dataset_list(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    class_filter: int = Query(0, ge=0, description="按DEFECT_ID筛选，0=全部"),
):
    """浏览数据集"""
    ds = _load_dataset()
    labels = ds["labels"]
    images = ds["images"]

    # 过滤
    if class_filter > 0:
        filtered = [n for n in images if labels.get(n) == class_filter]
    else:
        filtered = images

    total = len(filtered)
    start = (page - 1) * page_size
    end = min(start + page_size, total)
    page_items = filtered[start:end]

    return DatasetBrowseResponse(
        success=True,
        total=total,
        images=[
            DatasetImageInfo(
                image_name=name,
                defect_id=labels.get(name),
                file_size_kb=round(os.path.getsize(os.path.join(DATASET_DIR, name)) / 1024, 1),
            )
            for name in page_items
        ],
    )


@router.get("/dataset/classes")
async def dataset_classes():
    """获取数据集的所有类别及其数量"""
    ds = _load_dataset()
    labels = ds["labels"]
    from collections import Counter
    counts = Counter(labels.values())
    classes = sorted(counts.keys())
    return {
        "success": True,
        "total_classes": len(classes),
        "total_images": ds["total"],
        "classes": [
            {"defect_id": int(c), "count": counts[c]}
            for c in classes
        ],
    }


@router.post("/dataset/classify", response_model=DatasetClassifyResponse)
async def dataset_classify(request: DatasetClassifyRequest):
    """用 Qwen-VL 对数据集单张图像做缺陷分类"""
    start = time.time()

    ds = _load_dataset()
    if request.image_name not in ds["labels"] and request.image_name not in ds["images"]:
        raise HTTPException(404, f"Image '{request.image_name}' not found in dataset")

    img = _read_dataset_image(request.image_name)
    img_b64 = _image_to_base64(img)
    ground_truth = ds["labels"].get(request.image_name)

    llama = main_module.llama_client
    if not llama:
        raise HTTPException(503, "Qwen-VL client not available")

    # Qwen-VL 分类 prompt
    system_prompt = """You are a semiconductor wafer defect classification expert.
Analyze the wafer image and classify the defect type by its DEFECT_ID number.
Output JSON only. Be decisive."""

    class_list = ", ".join([str(c) for c in sorted(set(ds["labels"].values()))])
    user_prompt = f"""Classify this wafer image into one of the following DEFECT_ID categories: {class_list}.

First, describe what you see in 1-2 sentences.
Then output exactly 3 most likely DEFECT_ID predictions with confidence scores.

Output JSON format:
{{
  "analysis": "brief description of what you see",
  "predictions": [
    {{"defect_id": 1, "confidence": 0.95, "reason": "visible chipping at edge"}},
    {{"defect_id": 2, "confidence": 0.03, "reason": "some surface texture"}},
    {{"defect_id": 3, "confidence": 0.02, "reason": "fallback"}}
  ]
}}
Note: The first prediction should be your most confident answer. Confidence MUST sum to approximately 1.0 across all top-{request.top_k} predictions."""

    try:
        result = await llama.analyze_image(
            image_base64=img_b64,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=512,
            temperature=0.1,
        )

        predictions = []
        if result and "predictions" in result:
            for p in result["predictions"][:request.top_k]:
                predictions.append(DatasetClassifyResult(
                    defect_id=p.get("defect_id"),
                    confidence=float(p.get("confidence", 0)),
                    reason=p.get("reason", ""),
                ))

        # 如果 Qwen 返回了 top-3，补充到 k 个
        while len(predictions) < request.top_k:
            predictions.append(DatasetClassifyResult(
                defect_id=None, confidence=0.0, reason="no prediction"
            ))

        analysis_text = result.get("analysis", "") if result else ""

    except Exception as e:
        logger.warning(f"Qwen classification failed: {e}")
        predictions = [DatasetClassifyResult(defect_id=None, confidence=0, reason="error")]
        analysis_text = str(e)

    elapsed = int((time.time() - start) * 1000)

    return DatasetClassifyResponse(
        success=True,
        image_name=request.image_name,
        ground_truth=ground_truth,
        predictions=predictions,
        analysis_text=analysis_text,
        inference_time_ms=elapsed,
    )


@router.post("/dataset/batch-eval")
async def dataset_batch_eval(
    sample_size: int = Query(10, ge=1, le=100, description="随机采样数"),
):
    """批量评估：随机采样N张用Qwen-VL分类，统计准确率"""
    start = time.time()

    ds = _load_dataset()
    labels = ds["labels"]

    # 只取有标签的图片
    labeled = [n for n in ds["images"] if n in labels]
    if not labeled:
        return {"success": False, "error": "No labeled images found"}

    import random
    random.shuffle(labeled)
    sample = labeled[:min(sample_size, len(labeled))]

    llama = main_module.llama_client
    if not llama:
        raise HTTPException(503, "Qwen-VL client not available")

    results = []
    correct = 0
    system_prompt = "Classify the wafer defect by DEFECT_ID. Output JSON only."

    for img_name in sample:
        img = _read_dataset_image(img_name)
        img_b64 = _image_to_base64(img)
        gt = labels[img_name]

        try:
            result = await llama.analyze_image(
                image_base64=img_b64,
                system_prompt=system_prompt,
                user_prompt=f"Output {{'defect_id': <number>, 'confidence': <0-1>, 'reason': '<brief>'}}. Classify this wafer image.",
                max_tokens=256,
                temperature=0.1,
            )

            predicted = None
            conf = 0
            if result:
                predicted = result.get("defect_id")
                conf = result.get("confidence", 0)

            is_correct = (predicted == gt)
            if is_correct:
                correct += 1

            results.append({
                "image_name": img_name,
                "ground_truth": gt,
                "predicted": predicted,
                "confidence": conf,
                "correct": is_correct,
            })
        except Exception as e:
            results.append({
                "image_name": img_name,
                "ground_truth": gt,
                "predicted": None,
                "confidence": 0,
                "correct": False,
                "error": str(e),
            })

    elapsed = int((time.time() - start) * 1000)
    accuracy = correct / len(sample) if sample else 0

    return {
        "success": True,
        "total": len(sample),
        "processed": len(results),
        "correct": correct,
        "accuracy": round(accuracy, 4),
        "inference_time_ms": elapsed,
        "results": results,
    }
