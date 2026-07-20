# Changelog

## [Unreleased]

### Added
- Open-source release preparation: LICENSE, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY
- Document compression & reorganization for public release
- Bilingual README (English + Chinese)
- Service-level README for all 5 microservices
- GitHub issue/PR templates

## [0.2.0] — 2026-07-19

### Added
- Qwen-VL wafer detection Phase A: Qwen2.5-VL-7B integration with llama.cpp
- OpenCV enhancement engine (denoising + contrast enhancement)
- Circular defect detection with Hough Circle + Qwen text analysis
- Brightfield morphology preservation in enhancement pipeline
- Multi-angle view generation with detection overlay
- React-based Qwen detection page in wafer-showcase

### Changed
- Enhanced reporter with structured JSON output + YAML prompt system
- API routes refined for Qwen service endpoints

## [0.1.0] — 2026-07-11

### Added
- Spring Boot 3.2 backend (wafer-backend): API gateway, gallery, inference, metrics
- FastAPI LoRA inference service (wafer-lora-service): SD 1.5 + ControlNet
- React frontend showcase (wafer-showcase): workbench, gallery, metrics dashboard
- Docker Compose orchestration (PostgreSQL, Redis, MinIO, Nginx)
- GitHub Actions CI/CD workflows for all services
- Full integration: backend ↔ LoRA service ↔ frontend

## [0.0.1] — 2026-07-06

### Added
- wafer-inspection PyTorch model: shared encoder (RepViT) + dual-branch architecture
- Optical physics injection v2.0: PISM + ASG + dual-wavelength detection
- Three-stage semi-supervised training pipeline
- Model compression: reparameterization fusion + INT8 quantization + TorchScript JIT
- LibTorch C++ edge inference SDK with C API
- Polar coordinate transform for wafer edge optimization
- Online adaptive calibration for light source drift compensation
- Complete technical documentation and design specs
