# Wafer LoRA Inference Service

Stable Diffusion 1.5 inference microservice with 3 LoRA modes and ControlNet physical constraints for wafer image generation and enhancement.

## Tech Stack
- FastAPI (Python)
- Stable Diffusion 1.5 (HuggingFace diffusers)
- LoRA adapters + ControlNet
- NVIDIA GPU (CUDA)

## Quick Start
```bash
pip install -r requirements.txt
python main.py  # Port 8000
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check (GPU status) |
| `/api/v1/lora/infer` | POST | Single-mode inference |
| `/api/v1/lora/switch` | POST | Switch/activate LoRA adapter |
| `/api/v1/lora/active` | GET | Query active LoRA |

## LoRA Modes

| Mode | Weight File | Effect |
|------|------------|--------|
| `enhance` | enhance_lora.safetensors | Dark-field → bright-field enhancement |
| `wavelength` | wavelength_lora.safetensors | 266nm → 193nm wavelength conversion |
| `defect` | defect_lora.safetensors | Defect generation for data augmentation |

## Training

```bash
python training/train_lora.py \
  --config training/config_lora.yaml \
  --instance_data_dir training/data/wavelength_pairs \
  --output_dir ./loras \
  --adapter_name wavelength_lora
```

## LoRA Weight Files
LoRA `.safetensors` files are not included in git (generated via training pipeline above).

## Docker
```bash
docker build -t wafer-lora-service .
docker run --gpus all -p 8000:8000 wafer-lora-service
```
