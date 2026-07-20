# Wafer Qwen Service — Qwen-VL Defect Analysis

## Overview
Qwen-VL powered wafer defect detection service using llama.cpp for pure CPU inference. Provides enhancement, detection, and natural language analysis of wafer images.

## Tech Stack
- FastAPI (Python)
- Qwen2.5-VL-7B-Instruct (GGUF Q4_K_M)
- llama.cpp (CPU inference)
- OpenCV (image processing)

## Quick Start
```bash
pip install -r requirements.txt

# 1. Start llama.cpp server (requires Qwen GGUF model)
# See docker-compose.yml for configuration

# 2. Start Qwen service
python main.py  # Port 8001
```

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/enhance` | POST | Enhance wafer image (OpenCV pipeline) |
| `/api/v1/detect` | POST | Detect defects (CV + Qwen analysis) |
| `/api/v1/analyze` | POST | Qwen-VL text analysis |
| `/api/v1/report` | POST | Generate defect report |
| `/api/v1/angular-views` | POST | Multi-angle view generation |

## Docker
```dockerfile
docker build -t wafer-qwen-service .
# Requires llama.cpp container with Qwen GGUF model
```

## Model Download
The Qwen GGUF model (~4.6GB) is not included in git. Download:
```bash
# From Hugging Face
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct-GGUF qwen2.5-vl-7b-instruct-q4_k_m.gguf --local-dir /models/
```

## Architecture
```
Input Image → OpenCV Enhancer → Qwen-VL Analysis → Reporter → Structured Output
                 ↓                     ↓
         Circular Detection      Text Analysis
```
