from pydantic import BaseModel, Field
from typing import Optional


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class Detection(BaseModel):
    type: str = Field(description="缺陷类型: 崩边|颗粒|划痕|位错")
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox
    source: str = "qwen+cv"


class EnhanceAnalysis(BaseModel):
    brightness: str = "中"
    noise_level: str = "中"
    sharpness: str = "清晰"
    defect_count: int = 0


class QwenEnhanceRequest(BaseModel):
    image: str = Field(description="Base64 编码的图片数据")
    format: str = "jpg"
    style: str = "darkfield"  # "darkfield" | "brightfield"


class QwenEnhanceResponse(BaseModel):
    success: bool = True
    style: str = "darkfield"
    enhanced_image: str = Field(description="Base64 编码的增强图（含检测框）")
    reference_image: str = Field(description="Base64 编码的亮场参考图", default="")
    detections: list[Detection] = []
    analysis: EnhanceAnalysis = EnhanceAnalysis()
    analysis_text: str = Field(description="Qwen-VL 视觉分析文本", default="")
    inference_time_ms: int = 0


class QwenReportRequest(BaseModel):
    image: str = Field(description="Base64 编码的增强图片")
    detections: list[Detection] = []


class QwenReportResponse(BaseModel):
    success: bool = True
    report: str = ""
    inference_time_ms: int = 0


class QwenHealthResponse(BaseModel):
    status: str = "UP"
    llama_available: bool = False
    device: str = "cpu"


class QwenAnglesRequest(BaseModel):
    image: str = Field(description="Base64 编码的增强图片")
    format: str = "jpg"


class QwenAnglesResponse(BaseModel):
    success: bool = True
    angles: dict[str, str] = Field(description="{角度: base64图片} 字典")
    inference_time_ms: int = 0
