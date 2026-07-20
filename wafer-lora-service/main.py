import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from models.infer_schemas import HealthResponse, LoraInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model manager (lazy init)
model_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init model manager, load default LoRA"""
    global model_manager
    from core.model_manager import ModelManager
    logger.info("Initializing ModelManager...")
    model_manager = ModelManager()
    try:
        model_manager.load_lora("enhance")
    except FileNotFoundError:
        logger.warning("LoRA weights not found — service running without active LoRA")
    logger.info(f"ModelManager ready. GPU: {torch.cuda.is_available()}")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Wafer LoRA Inference Service",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    global model_manager
    active = []
    if model_manager:
        active = [
            LoraInfo(name=name, active=True, weight=w)
            for name, w in model_manager.active_loras.items()
        ]
    return HealthResponse(
        status="UP",
        device=settings.device,
        gpu_available=torch.cuda.is_available(),
        active_loras=active,
    )


# Import and register routers
from api.routes import infer, lora
app.include_router(infer.router, prefix="/api/v1/lora")
app.include_router(lora.router, prefix="/api/v1/lora")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
