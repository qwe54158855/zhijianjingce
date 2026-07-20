import logging
from pathlib import Path
from typing import Optional

import torch
from diffusers import StableDiffusionImg2ImgPipeline, ControlNetModel
from diffusers.utils import load_image
from PIL import Image

from core.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """SD model lifecycle + LoRA hot-swap manager."""

    def __init__(self):
        self.device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if settings.torch_dtype == "float16" else torch.float32

        logger.info(f"Loading base model: {settings.model_id} ({dtype})")
        self.base = StableDiffusionImg2ImgPipeline.from_pretrained(
            settings.model_id,
            torch_dtype=dtype,
            variant="fp16" if settings.torch_dtype == "float16" else None,
            safety_checker=None,  # Disable NSFW filter for wafer images
        )
        self.base.to(self.device)

        if settings.offload:
            self.base.enable_model_cpu_offload()
        if settings.vae_slicing:
            self.base.enable_vae_slicing()

        if hasattr(torch, 'compile'):
            logger.info("Applying torch.compile to UNet...")
            self.base.unet = torch.compile(
                self.base.unet,
                mode="reduce-overhead",
                fullgraph=True,
            )

        # LoRA registry: name -> safetensors path
        self.lora_dir = Path(settings.lora_dir)
        self.lora_registry = {
            "enhance":    str(self.lora_dir / "enhance_lora.safetensors"),
            "wavelength": str(self.lora_dir / "wavelength_lora.safetensors"),
            "defect":     str(self.lora_dir / "defect_lora.safetensors"),
        }
        self.active_loras: dict[str, float] = {}

        # ControlNet (lazy load)
        self._controlnet: Optional[ControlNetModel] = None

        logger.info(f"Model loaded on {self.device}")

    @property
    def controlnet(self) -> ControlNetModel:
        if self._controlnet is None:
            logger.info("Loading ControlNet (Canny)...")
            self._controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=torch.float16,
            ).to(self.device)
        return self._controlnet

    def set_loras(self, loras: dict[str, float]):
        """Set multiple LoRAs simultaneously. loras = {name: weight, ...}."""
        if self.active_loras:
            names = list(self.active_loras.keys())
            self.base.delete_adapters(names)
            self.active_loras = {}
        for name, weight in loras.items():
            path = self.lora_registry.get(name)
            if not path:
                raise ValueError(f"Unknown LoRA: {name}. Available: {list(self.lora_registry.keys())}")
            if not Path(path).exists():
                logger.warning(f"LoRA weight not found: {path} — skipping")
                continue
            self.base.load_lora_weights(path, adapter_name=name)
            self.active_loras[name] = weight
        logger.info(f"Active LoRAs: {self.active_loras}")

    def load_lora(self, name: str, weight: float = 1.0):
        """Load a LoRA adapter. Multiple LoRAs can be active simultaneously."""
        path = self.lora_registry.get(name)
        if not path:
            raise ValueError(f"Unknown LoRA: {name}. Available: {list(self.lora_registry.keys())}")
        if not Path(path).exists():
            raise FileNotFoundError(f"LoRA weight not found: {path}")

        self.base.load_lora_weights(path, adapter_name=name)
        self.active_loras[name] = weight
        logger.info(f"Loaded LoRA: {name} (weight={weight})")

    def switch_lora(self, name: str, weight: float = 1.0):
        """Switch to a single LoRA (remove all others)."""
        if self.active_loras:
            self.base.delete_adapters(list(self.active_loras.keys()))
            self.active_loras = {}
        self.load_lora(name, weight)

    @torch.inference_mode()
    def infer(
        self,
        image: Image.Image,
        prompt: str,
        control_image: Optional[Image.Image] = None,
        strength: float = 0.75,
    ) -> Image.Image:
        """Run SD img2img inference with active LoRA adapters."""
        kwargs = {
            "image": image,
            "prompt": prompt,
            "strength": strength,
            "num_inference_steps": settings.num_inference_steps,
            "output_type": "pil",
        }

        if self.active_loras:
            kwargs["cross_attention_kwargs"] = {
                "scale": list(self.active_loras.values())
            }

        if control_image is not None:
            kwargs["control_image"] = control_image
            kwargs["controlnet_conditioning_scale"] = 0.8

        result = self.base(**kwargs)
        return result.images[0]

    @torch.inference_mode()
    def infer_multi(
        self,
        image: Image.Image,
        prompt: str = "",
        control_image: Optional[Image.Image] = None,
        strength: float = 0.75,
        adapter_weights: Optional[list[float]] = None,
    ) -> Image.Image:
        """Run inference with multiple active LoRAs and explicit adapter weights."""
        kwargs = {
            "image": image,
            "prompt": prompt,
            "strength": strength,
            "num_inference_steps": settings.num_inference_steps,
            "output_type": "pil",
        }
        if self.active_loras:
            kwargs["cross_attention_kwargs"] = {
                "scale": adapter_weights or list(self.active_loras.values())
            }
        if control_image is not None:
            kwargs["control_image"] = control_image
            kwargs["controlnet_conditioning_scale"] = 0.8
        result = self.base(**kwargs)
        return result.images[0]
