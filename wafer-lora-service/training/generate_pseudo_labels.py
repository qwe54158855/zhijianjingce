"""
PISM pseudo-label generation pipeline.

Uses wafer-inspection PISM module to generate 266nm→193nm training pairs.
Run after wafer-inspection is installed as a package.

Usage:
    python generate_pseudo_labels.py \
        --input_dir /path/to/266nm/images \
        --output_dir training/data/wavelength_pairs \
        --num_pairs 1000
"""
import argparse
import logging
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_wafer_inspection():
    """Check wafer-inspection is importable."""
    try:
        from wafer_inspection.models.physics_scattering import ScatteringPhysics
        from wafer_inspection.models.wafer_multitask import (
            RepViTEncoder, EnhanceDecoder
        )
        return True
    except ImportError:
        logger.error(
            "wafer-inspection not installed. "
            "Run: cd /path/to/wafer-inspection && pip install -e ."
        )
        return False


@torch.inference_mode()
def generate_pairs(input_dir: Path, output_dir: Path, num_pairs: int):
    """Generate 266nm → pseudo-193nm image pairs using PISM."""
    from wafer_inspection.models.physics_scattering import ScatteringPhysics
    from wafer_inspection.models.wafer_multitask import (
        RepViTEncoder, EnhanceDecoder
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "266nm").mkdir(exist_ok=True)
    (output_dir / "193nm").mkdir(exist_ok=True)

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    encoder = RepViTEncoder(in_channels=1).to(device).eval()
    pism = ScatteringPhysics(
        channels_per_stage=[56, 112, 224, 448],
        lambda_in=266.0,
        lambda_out=193.0,
    ).to(device).eval()
    decoder = EnhanceDecoder().to(device).eval()

    # Find input images
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif")
    images = []
    for ext in extensions:
        images.extend(input_dir.glob(ext))
        images.extend(input_dir.glob(ext.upper()))

    if not images:
        logger.warning(f"No images found in {input_dir}")
        return

    logger.info(f"Found {len(images)} source images, generating {num_pairs} pairs")
    to_tensor = transforms.ToTensor()

    count = 0
    while count < num_pairs:
        for img_path in images:
            if count >= num_pairs:
                break

            # Load and preprocess
            pil_img = Image.open(img_path).convert("L")
            pil_img = pil_img.resize((512, 512), Image.LANCZOS)
            tensor = to_tensor(pil_img).unsqueeze(0).to(device)  # [1,1,512,512]

            # Forward through encoder → PISM → decoder
            feats = encoder(tensor)
            feats_193, pism_diag = pism(feats)
            enhanced = decoder(feats_193)  # [1,1,512,512]

            # Save pair
            stem = f"pair_{count:05d}"
            Image.fromarray(
                (tensor[0, 0].cpu().numpy() * 255).astype("uint8")
            ).save(str(output_dir / "266nm" / f"{stem}_266.jpg"))

            Image.fromarray(
                (enhanced[0, 0].cpu().numpy() * 255).astype("uint8")
            ).save(str(output_dir / "193nm" / f"{stem}_193.jpg"))

            count += 1
            if count % 100 == 0:
                logger.info(f"Generated {count}/{num_pairs} pairs")

    logger.info(f"Done: {count} pairs saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PISM pseudo-labels")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with 266nm wafer images")
    parser.add_argument("--output_dir", type=str, default="training/data/wavelength_pairs",
                        help="Output directory for pairs")
    parser.add_argument("--num_pairs", type=int, default=1000,
                        help="Number of pairs to generate")
    args = parser.parse_args()

    if not ensure_wafer_inspection():
        sys.exit(1)

    generate_pairs(
        Path(args.input_dir),
        Path(args.output_dir),
        args.num_pairs,
    )
