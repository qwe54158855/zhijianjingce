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

REPORT_SYSTEM = """你是半导体晶圆缺陷检测质量分析专家。
基于增强后的晶圆图像和已检出的缺陷信息，生成一份专业的检测报告。"""


async def generate_report(
    llama_client: LlamaClient,
    image_base64: str,
    detections: list[Detection],
) -> Optional[str]:
    """生成缺陷分析报告"""
    detection_text = "\n".join(
        f"- {d.type}：置信度 {d.confidence}，位于坐标 ({d.bbox.x}, {d.bbox.y})"
        for d in detections
    )

    report_user = f"""已检出以下缺陷：
{detection_text if detection_text else "（未检出明显缺陷）"}

请生成一份检测报告，包含：
1. 图像质量评估（增强效果、清晰度等）
2. 缺陷汇总（各类缺陷数量、位置分布）
3. 重点关注（高置信度缺陷的处理建议）
4. 总体结论（晶圆质量初步判断）

报告要求：专业、简洁、中文。"""

    return await llama_client.generate_report(
        image_base64=image_base64,
        system_prompt=REPORT_SYSTEM,
        user_prompt=report_user,
    )
