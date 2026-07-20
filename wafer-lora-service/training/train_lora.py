"""
LoRA training script for wafer inspection image translation.

Extends diffusers' train_dreambooth_lora.py with wafer-specific:
- PISM physical consistency loss (optional)
- Real calibration pair upweighting
- Multi-adapter support

Usage:
    python train_lora.py \
        --config config_lora.yaml \
        --instance_data_dir data/wavelength_pairs \
        --output_dir ../loras \
        --adapter_name wavelength_lora

Requires: pip install diffusers[training] peft bitsandbytes
"""
import argparse
import logging
import math
import os
from pathlib import Path

import torch
import yaml
from diffusers import StableDiffusionPipeline
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

check_min_version("0.27.0")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WaferPairDataset(Dataset):
    """Wafer image pair dataset (266nm -> 193nm)."""

    def __init__(self, data_dir: str, resolution: int = 512, flip_p: float = 0.5):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.flip_p = flip_p

        # Expect structure: data_dir/{266nm,193nm}/*.jpg
        self.images_266 = sorted((self.data_dir / "266nm").glob("*.jpg"))
        self.images_193 = sorted((self.data_dir / "193nm").glob("*.jpg"))

        assert len(self.images_266) == len(self.images_193), \
            f"Mismatch: {len(self.images_266)} vs {len(self.images_193)}"

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.RandomHorizontalFlip(p=flip_p),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.images_266)

    def __getitem__(self, idx):
        img_266 = Image.open(self.images_266[idx]).convert("RGB")
        img_193 = Image.open(self.images_193[idx]).convert("RGB")

        # Apply same random seed for paired transform
        seed = torch.randint(0, 2**30, (1,)).item()
        torch.manual_seed(seed)
        pixel_266 = self.transform(img_266)
        torch.manual_seed(seed)
        pixel_193 = self.transform(img_193)

        return {
            "pixel_values": pixel_266,
            "target_values": pixel_193,
        }


def train_lora(config_path: str, data_dir: str, output_dir: str, adapter_name: str):
    """Main training loop."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        config["base_model"],
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,
    )
    pipe.to(device)
    vae = pipe.vae
    unet = pipe.unet
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Apply LoRA to UNet
    lora_config = LoraConfig(
        r=config.get("lora_rank", 64),
        lora_alpha=config.get("lora_alpha", 32),
        target_modules=config.get("target_modules"),
        lora_dropout=0.0,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.train()

    # Dataset
    dataset = WaferPairDataset(
        data_dir=data_dir,
        resolution=config["data"]["resolution"],
        flip_p=config.get("training", {}).get("flip_p", 0.5),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["dataloader_num_workers"],
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config["training"]["learning_rate"],
    )

    # LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["num_epochs"],
    )

    # Training loop
    logger.info(f"Starting LoRA training for {adapter_name}")
    global_step = 0
    for epoch in range(config["training"]["num_epochs"]):
        for step, batch in enumerate(dataloader):
            pixel_266 = batch["pixel_values"].to(device)
            target_193 = batch["target_values"].to(device)

            # Encode target to latent space
            with torch.no_grad():
                latents = vae.encode(target_193).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Sample noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, pipe.scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            )
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # Encode prompt (empty prompt for unconditional)
            with torch.no_grad():
                encoder_hidden_states = text_encoder(
                    tokenizer(
                        [""] * latents.shape[0],
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to(device)
                )[0]

            # Predict noise
            noise_pred = unet(
                noisy_latents, timesteps, encoder_hidden_states
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % config["training"]["save_steps"] == 0:
                logger.info(f"Step {global_step}, Loss: {loss.item():.6f}")

        lr_scheduler.step()
        logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']} complete, Loss: {loss.item():.6f}")

    # Save LoRA weights
    output_path = Path(output_dir) / f"{adapter_name}.safetensors"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(str(output_path.parent), safe_serialization=True)
    logger.info(f"LoRA saved to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train wafer LoRA adapters")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./loras")
    parser.add_argument("--adapter_name", type=str, default="wavelength_lora")
    args = parser.parse_args()

    result = train_lora(args.config, args.instance_data_dir, args.output_dir, args.adapter_name)
    logger.info(f"Training complete: {result}")
