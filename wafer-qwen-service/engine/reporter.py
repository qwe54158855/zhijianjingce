import logging
from typing import Optional

from core.model_manager import LlamaClient
from models.schemas import Detection

logger = logging.getLogger(__name__)

# Prompts
ENHANCE_SYSTEM = """你是半导体晶圆缺陷检测专家。
分析这张暗场晶圆图像，输出 JSON 格式的分析结果。"""

ENHANCE_USER = """分析这张暗场晶圆图像，特别注意：
1. 图像整体亮度、噪声水平、清晰度
2. 是否存在疑似缺陷，包括崩边（边缘碎裂V形缺口）、颗粒（表面附着）、划痕（细长）、位错（晶格缺陷暗色线条）
3. 缺陷的位置（用上下左右/中心/边缘+角描述）、大小（小<10px/中10-50px/大>50px）、形状

输出 JSON，格式如下：
{
  "analysis": {
    "brightness": "低|中|高",
    "noise_level": "低|中|高",
    "sharpness": "模糊|清晰",
    "suspected_defects": [
      {"type": "崩边|颗粒|划痕|位错", "confidence": 0.0-1.0,
       "region": "位置描述", "size": "小|中|大"}
    ]
  },
  "enhance_params": {
    "clahe_clip": 2.0-4.0,
    "clahe_grid": 4|8,
    "denoise_strength": 5-20,
    "gamma": 0.8-2.0,
    "contrast": 1.0-2.0,
    "sharpen": true|false
  }
}

只输出 JSON，不要额外文字。"""

REPORT_SYSTEM = """You are a semiconductor wafer defect inspection expert. Provide concise, professional analysis."""


async def generate_report(
    llama_client: LlamaClient,
    image_base64: str,
    detections: list[Detection],
) -> Optional[str]:
    """Generate defect analysis report (text-only mode with template fallback)"""
    if not detections:
        return "No defects detected. Wafer quality appears good."

    from collections import Counter, defaultdict

    # Build structured report
    counts = Counter(d.type for d in detections)
    by_type = defaultdict(list)
    for d in detections:
        by_type[d.type].append(d)

    lines = ["=== Wafer Defect Inspection Report ===", ""]
    lines.append(f"Total defects found: {len(detections)}")
    lines.append(f"Types detected: {', '.join(f'{t} x{c}' for t, c in counts.most_common())}")
    lines.append("")

    for defect_type, items in by_type.items():
        max_conf = max(items, key=lambda d: d.confidence)
        lines.append(f"[{defect_type}] {len(items)} defect(s)")
        lines.append(f"  Highest confidence: {max_conf.confidence:.0%}")
        lines.append(f"  Location: ({max_conf.bbox.x}, {max_conf.bbox.y})")
        lines.append("")

    lines.append("Recommendations:")
    for d in sorted(detections, key=lambda x: x.confidence, reverse=True):
        if d.confidence >= 0.7:
            lines.append(f"- Review {d.type} at ({d.bbox.x}, {d.bbox.y}) - high confidence")
        elif d.confidence >= 0.5:
            lines.append(f"- Monitor {d.type} at ({d.bbox.x}, {d.bbox.y}) - medium confidence")

    return "\n".join(lines)
