from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Service
    service_name: str = "wafer-lora-service"
    host: str = "0.0.0.0"
    port: int = 8000

    # Model
    model_id: str = "runwayml/stable-diffusion-v1-5"
    torch_dtype: str = "float16"
    device: str = "cuda"
    offload: bool = True
    vae_slicing: bool = True

    # LoRA
    lora_dir: Path = Path(__file__).parent.parent / "loras"
    num_inference_steps: int = 20

    class Config:
        env_prefix = "LORA_"


settings = Settings()
