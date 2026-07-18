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


class QwenEnhanceResponse(BaseModel):
    success: bool = True
    enhanced_image: str = Field(description="Base64 编码的增强图（含检测框）")
    detections: list[Detection] = []
    analysis: EnhanceAnalysis = EnhanceAnalysis()
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
