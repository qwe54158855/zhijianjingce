import base64
import io
import logging
import time

from fastapi import APIRouter, HTTPException
from PIL import Image, ImageFilter, ImageOps

from core.physics_control import compute_gain_map
from models.infer_schemas import InferRequest, InferResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Inference"])

# Default prompts for each mode
DEFAULT_PROMPTS = {
    "enhance":    "wafer inspection dark field to bright field enhancement, "
                  "high contrast defect visibility, sharp edges, clean background",
    "wavelength": "deep UV 193nm wafer inspection, enhanced Rayleigh scattering, "
                  "high resolution defect detection, sharp nanostructures",
    "defect":     "wafer defect, realistic scratch and particle contamination, "
                  "semiconductor manufacturing defect, high detail",
}


def decode_base64_image(b64: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    try:
        image_bytes = base64.b64decode(b64)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string (JPEG, quality 95)."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_control_image(image: Image.Image, mode: str) -> Image.Image:
    """Generate ControlNet conditioning image based on mode."""
    if mode == "defect":
        # For defect generation, create an edge map
        return image.filter(ImageFilter.FIND_EDGES)
    else:
        # For enhance/wavelength, use Canny-like edge detection
        gray = image.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        return edges.convert("RGB")


@router.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    """Run LoRA inference on input image."""
    from main import model_manager

    t0 = time.time()

    # Decode input image
    image = decode_base64_image(req.image_base64)
    image = image.resize((512, 512), Image.LANCZOS)

    if req.combined_mode:
        # --- Multi-LoRA combined mode ---
        modes = [m.strip() for m in req.mode.split(",") if m.strip()]
        loras = {m: 1.0 for m in modes}
        try:
            model_manager.set_loras(loras)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(status_code=400, detail=str(e))

        # Build combined prompt from individual mode prompts
        parts = [
            DEFAULT_PROMPTS.get(m) or f"wafer inspection {m}"
            for m in modes
        ]
        prompt = req.prompt or "; ".join(parts)

        # ControlNet condition (use the first mode's logic for now)
        control_image = None
        if req.control_image_base64:
            control_image = decode_base64_image(req.control_image_base64)
        elif modes[0] == "wavelength":
            control_image = compute_gain_map(image)
            logger.info("Using physics-informed gain map for wavelength conversion")
        elif modes[0] == "enhance":
            control_image = generate_control_image(image, modes[0])

        try:
            result = model_manager.infer_multi(
                image=image,
                prompt=prompt,
                control_image=control_image,
                strength=req.strength,
            )
        except Exception as e:
            logger.error(f"Combined inference failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    else:
        # --- Single-LoRA mode ---
        if req.mode not in model_manager.active_loras:
            try:
                model_manager.switch_lora(req.mode)
            except (ValueError, FileNotFoundError) as e:
                raise HTTPException(status_code=400, detail=str(e))

        # Prepare prompt
        prompt = req.prompt or DEFAULT_PROMPTS.get(req.mode, "wafer inspection")

        # Prepare ControlNet condition
        control_image = None
        if req.control_image_base64:
            control_image = decode_base64_image(req.control_image_base64)
        elif req.mode == "wavelength":
            control_image = compute_gain_map(image)
            logger.info("Using physics-informed gain map for wavelength conversion")
        elif req.mode == "enhance":
            control_image = generate_control_image(image, req.mode)

        # Run inference
        try:
            result = model_manager.infer(
                image=image,
                prompt=prompt,
                control_image=control_image,
                strength=req.strength,
            )
        except Exception as e:
            logger.error(f"Inference failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # Encode result
    result_b64 = encode_image_to_base64(result)
    duration_ms = int((time.time() - t0) * 1000)

    return InferResponse(result_base64=result_b64, duration_ms=duration_ms)
