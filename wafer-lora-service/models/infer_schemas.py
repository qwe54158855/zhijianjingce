from pydantic import BaseModel, Field
from typing import Optional


class InferRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded input image")
    mode: str = Field(
        ..., pattern=r"^(enhance|wavelength|defect)(,(enhance|wavelength|defect))*$"
    )
    strength: float = Field(default=0.75, ge=0.1, le=1.0)
    prompt: Optional[str] = None
    control_image_base64: Optional[str] = None  # ControlNet conditioning
    combined_mode: bool = False  # When True, mode is comma-separated list


class InferResponse(BaseModel):
    result_base64: str = Field(description="Base64 encoded result image")
    duration_ms: int = Field(description="Inference duration in ms")


class LoraInfo(BaseModel):
    name: str
    active: bool
    weight: float


class HealthResponse(BaseModel):
    status: str
    device: str
    gpu_available: bool
    active_loras: list[LoraInfo]
