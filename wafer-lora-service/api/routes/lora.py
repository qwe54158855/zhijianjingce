import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from models.infer_schemas import LoraInfo

logger = logging.getLogger(__name__)
router = APIRouter(tags=["LoRA Management"])


class SwitchLoraRequest(BaseModel):
    name: str
    weight: float = 1.0


class SetLorasRequest(BaseModel):
    loras: dict[str, float]


@router.post("/set", summary="Set multiple active LoRAs simultaneously")
async def set_loras(req: SetLorasRequest):
    """Activate multiple LoRAs with individual weights for combined inference."""
    from main import model_manager
    if not req.loras:
        raise HTTPException(status_code=400, detail="At least one LoRA required")
    try:
        model_manager.set_loras(req.loras)
        return {"status": "ok", "active": model_manager.active_loras}
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/switch")
async def switch_lora(req: SwitchLoraRequest):
    """Hot-switch to a specific LoRA adapter."""
    from main import model_manager
    try:
        model_manager.switch_lora(req.name, req.weight)
        return {"status": "ok", "active": req.name, "weight": req.weight}
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/active", response_model=list[LoraInfo])
async def get_active_loras():
    """List currently active LoRA adapters."""
    from main import model_manager
    return [
        LoraInfo(name=name, active=True, weight=w)
        for name, w in model_manager.active_loras.items()
    ]
