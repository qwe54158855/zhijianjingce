from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Service
    service_name: str = "wafer-qwen-service"
    host: str = "0.0.0.0"
    port: int = 8001

    # llama-server
    llama_server_url: str = "http://localhost:8002/v1"
    llama_model_name: str = "qwen2.5-vl-7b"
    llama_timeout: int = 120

    # Enhancement defaults
    default_clahe_clip: float = 2.5
    default_clahe_grid: int = 8
    default_denoise: int = 10
    default_gamma: float = 1.2
    default_contrast: float = 1.3

    # Detection
    min_contour_area: int = 20
    detection_confidence_threshold: float = 0.5

    class Config:
        env_prefix = "QWEN_"


settings = Settings()
