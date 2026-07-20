import json
import logging
from typing import Optional

from core.model_manager import LlamaClient
from models.schemas import Detection

logger = logging.getLogger(__name__)

# Vision prompts for Qwen-VL (仅用于 analyze_with_vision，报告不再用看图)
VISION_SYSTEM = """You are a chief semiconductor wafer defect inspection expert with 20 years experience.
Analyze the dark-field wafer image and output JSON only. You MUST be extremely decisive.

RULES (non-negotiable):
- If you can identify ANY specific defect type → confidence MUST be >= 0.95
- Confidence 1.0 means you are absolutely certain
- Confidence values below 0.95 are ONLY for features too ambiguous to classify
- A blank image or one with no visible features should have total_defects_estimated = 0 and empty suspected_defects
- Trust your 20 years of expertise — be confident in your judgment"""

VISION_USER = """As a chief wafer inspection expert, analyze this dark-field wafer image decisively.

1. Assess brightness, noise level, clarity
2. Identify defects: chipping (V-edge crack), particle (surface contamination), scratch (linear mark), dislocation (dark lattice defect line)
3. For each identified defect → confidence >= 0.95 (you are an expert, be confident!)
4. Only leave defects out if truly nothing visible

Output JSON:
{
  "analysis": {
    "brightness": "low|medium|high",
    "noise_level": "low|medium|high",
    "clarity": "low|medium|high",
    "total_defects_estimated": 0,
    "suspected_defects": [
      {"type": "chipping|particle|scratch|dislocation", "confidence": 0.95, "region": "description", "size": "small|medium|large"}
    ]
  }
}"""

# 报告提示词：不传图像，完全基于检测明细表格数据
REPORT_SYSTEM = """你是半导体晶圆缺陷检测专家。请基于以下检测数据生成专业中文报告。
报告必须严格基于数据，不能凭空增加或改变内容。简洁专业，3-5句话。"""


async def generate_report(
    llama_client: LlamaClient,
    image_base64: str,
    detections: list[Detection],
) -> Optional[str]:
    """基于检测明细表格数据生成 AI 报告。

    注意：报告完全依据 detection 数据（类型/置信度/坐标/尺寸），
    不依赖 Qwen-VL 视觉分析，确保报告与前端表格一一对应。
    """
    if not detections:
        return "未检测到缺陷，晶圆质量良好。"

    from collections import Counter, defaultdict

    counts = Counter(d.type for d in detections)
    by_type = defaultdict(list)
    for d in detections:
        by_type[d.type].append(d)

    # 构建完整的检测数据文本（和前端表格一致）
    detail_lines = []
    detail_lines.append(f"共检测到 {len(detections)} 个缺陷：")
    for defect_type in ["崩边", "颗粒", "划痕", "位错"]:
        items = by_type.get(defect_type, [])
        if not items:
            continue
        detail_lines.append(f"\n【{defect_type}】{len(items)} 处：")
        for d in items[:5]:  # 每种类型最多列5个
            detail_lines.append(
                f"  - 置信度 {d.confidence*100:.1f}% | "
                f"坐标({d.bbox.x},{d.bbox.y}) | "
                f"尺寸{d.bbox.w}×{d.bbox.h}px"
            )
        if len(items) > 5:
            detail_lines.append(f"  ... 还有 {len(items)-5} 处")

    prompt = "\n".join(detail_lines)
    prompt += (
        "\n\n请以专家身份生成3-5句中文检测报告："
        "总结缺陷类型分布、重点关注区域、质量评估结论。"
    )

    # 用 Qwen 文本生成（不传图像），基于数据写报告
    try:
        report_text = await llama_client.generate_text(
            system_prompt=REPORT_SYSTEM,
            user_prompt=prompt,
            max_tokens=512,
            temperature=0.3,
        )

        if report_text and len(report_text) > 20:
            logger.info(f"Qwen data-driven report generated ({len(report_text)} chars)")
            return report_text.strip()

        logger.warning(f"Qwen report too short ({len(report_text) if report_text else 0} chars), using template")
    except Exception as e:
        logger.warning(f"Qwen report failed: {e}, using template")

    # 保底：基于数据的结构化模板
    lines = [f"晶圆检测报告 - 共发现 {len(detections)} 处缺陷\n"]
    for defect_type in ["崩边", "颗粒", "划痕", "位错"]:
        items = by_type.get(defect_type, [])
        if not items:
            continue
        max_conf = max(items, key=lambda d: d.confidence)
        min_conf = min(items, key=lambda d: d.confidence)
        avg_conf = sum(d.confidence for d in items) / len(items)
        lines.append(
            f"【{defect_type}】{len(items)}处 | "
            f"置信度范围 {min_conf*100:.1f}%~{max_conf*100:.1f}% | "
            f"平均 {avg_conf*100:.1f}%"
        )
        for d in sorted(items, key=lambda x: x.confidence, reverse=True)[:3]:
            lines.append(f"  位置 ({d.bbox.x},{d.bbox.y}) 尺寸 {d.bbox.w}×{d.bbox.h}px")

    lines.append(f"\n重点关注：置信度最高的缺陷为{max(detections, key=lambda d: d.confidence).type}"
                 f"({max(detections, key=lambda d: d.confidence).confidence*100:.1f}%)"
                 f"位于({max(detections, key=lambda d: d.confidence).bbox.x},"
                 f"{max(detections, key=lambda d: d.confidence).bbox.y})，"
                 f"尺寸{max(detections, key=lambda d: d.confidence).bbox.w}×"
                 f"{max(detections, key=lambda d: d.confidence).bbox.h}px。")
    return "\n".join(lines)


async def analyze_with_vision(
    llama_client: LlamaClient,
    image_base64: str,
) -> Optional[dict]:
    """Analyze wafer image with Qwen-VL and return structured JSON.

    Returns parsed JSON from the vision model, or None on failure.
    The frontend can use this to supplement OpenCV detections.
    """
    try:
        result = await llama_client.analyze_image(
            image_base64=image_base64,
            system_prompt=VISION_SYSTEM,
            user_prompt=VISION_USER,
            max_tokens=1024,
            temperature=0.1,
        )
        return result
    except Exception as e:
        logger.warning(f"Qwen-VL vision analysis failed: {e}")
        return None
