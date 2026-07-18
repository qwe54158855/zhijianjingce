import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from models.schemas import QwenHealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
llama_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llama_client
    from core.model_manager import LlamaClient
    llama_client = LlamaClient()
    logger.info(f"Llama client initialized, server: {settings.llama_server_url}")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Wafer Qwen-VL Enhancement Service",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", response_model=QwenHealthResponse)
async def health():
    global llama_client
    llama_ok = False
    if llama_client:
        llama_ok = await llama_client.check_health()
    return QwenHealthResponse(
        status="UP",
        llama_available=llama_ok,
        device="cpu",
    )


from api.routes import router
app.include_router(router, prefix="/api/v1/qwen")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
