# Wafer Inspection — Core Detection Model

## Overview
PyTorch-based wafer edge defect detection model with shared-encoder dual-branch architecture. Features RepViT lightweight backbone, optical physics injection (PISM + ASG), and three-stage semi-supervised training.

## Tech Stack
- PyTorch 2.x + TorchScript JIT
- RepViT-M0.9 / M0.6 backbone
- LibTorch C++ inference SDK

## Quick Start
```bash
pip install -e .

# Stage 1: CycleGAN unsupervised pretraining
python train_stage1.py --config configs/default.yaml

# Verify model after training
python deploy/verify_physics.py
python deploy/benchmark.py
```

## Key Modules
| Module | File | Description |
|--------|------|-------------|
| RepViT Backbone | `models/` | Multi-scale shared encoder |
| PISM | `models/physics_scattering.py` | Differentiable physical scattering |
| ASG | `models/angle_scattering_gen.py` | Multi-angle scattering generator |
| Wafer Multitask | `models/wafer_multitask.py` | Full model assembly |

## Model Export
```python
# After training: fuse + quantize + export
python -c "from deploy import *; export_jit('checkpoint.pt')"
# Output: deploy/wafer_detector_jit_int8.pt (< 8MB)
```

## Directory
```
models/         # Model architectures
losses/         # Loss functions
configs/        # Training configs
utils/          # Utilities
deploy/         # Verification & export scripts
tests/          # Unit tests
```
