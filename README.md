# Wafer Inspection — 半导体晶圆缺陷检测一体化 AI 解决方案

<p align="center">
  <img src="docs/assets/wafer-banner.png" alt="Wafer Inspection" width="600">
</p>

<p align="center">
  <strong>🇬🇧 An integrated AI solution for semiconductor wafer edge defect detection.</strong><br>
  <strong>🇨🇳 一体化半导体晶圆边缘缺陷检测 AI 解决方案。</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <a href="https://github.com/your-org/wafer-inspection"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://github.com/your-org/wafer-inspection"><img src="https://img.shields.io/badge/Spring%20Boot-3.2-brightgreen.svg" alt="Spring Boot 3.2"></a>
  <a href="https://github.com/your-org/wafer-inspection"><img src="https://img.shields.io/badge/React-18-61dafb.svg" alt="React 18"></a>
</p>

---

## 🇬🇧 Overview / 🇨🇳 项目概览

**EN:** This project targets the hardest problem in semiconductor manufacturing — detecting wafer edge defects (chipping, particles, scratches, dislocations) on **ARM edge devices with only 30–50 labeled samples**. The core innovation is a **shared-encoder dual-branch architecture** (RepViT backbone) with **optical physics prior injection** (PISM + ASG) — achieving < 80ms inference per frame on ARM Cortex-A76, INT8 model < 8MB.

**ZH:** 本项目解决半导体制造中最前沿的难题——在 **ARM 边缘设备上仅用 30-50 张标注样本**实现晶圆边缘缺陷检测（崩边、颗粒、划痕、位错）。核心创新为**共享编码器双分支架构**（RepViT 骨干）结合**光学物理先验注入**（PISM + ASG）—— ARM Cortex-A76 上单帧推理 < 80ms，INT8 模型 < 8MB。

---

## 🇬🇧 System Architecture / 🇨🇳 系统架构

```
                   暗场灰度图 (512×512)
                          │
                          ▼
              ┌───────────────────────────┐
              │   RepViT-M0.9 共享编码器    │
              └────────┬─────────┬────────┘
                       │         │
              ┌────────▼┐  ┌─────▼──────────┐
              │ 增强解码  │  │  检测输出分支    │
              │ (U型降噪)  │  │  Anchor-free    │
              │ → 明场图  │  │  PISM+ASG+SGF   │
              └─────────┘  └──────┬──────────┘
                                  │
              ┌───────────────────▼────────────┐
              │ TorchScript JIT + INT8 量化     │
              │ → ARM C++ SDK < 80ms/帧         │
              └────────────────────────────────┘
```

### 5 Microservices / 五大微服务

| Service | Stack | Role |
|---------|-------|------|
| [wafer-inspection](wafer-inspection/) | PyTorch | Core detection model + training |
| [wafer-backend](wafer-backend/) | Spring Boot 3.2 | API gateway, business logic |
| [wafer-lora-service](wafer-lora-service/) | FastAPI + SD 1.5 | LoRA-based image generation |
| [wafer-qwen-service](wafer-qwen-service/) | FastAPI + llama.cpp | Qwen-VL defect analysis |
| [wafer-showcase](wafer-showcase/) | React 18 + Vite 5 | Frontend dashboard |

---

## 🇬🇧 Quick Start / 🇨🇳 快速开始

### Docker (Full Stack)
```bash
git clone https://github.com/your-org/wafer-inspection.git
cd wafer-inspection
docker-compose up -d
# Visit http://localhost:80
```

### Local Development
See individual service READMEs for detailed instructions.

---

## 🇬🇧 Key Innovations / 🇨🇳 核心创新

### 1. 光学物理约束注入 v2.0 / Optical Physics Injection

| Module | Params | Function |
|--------|--------|----------|
| **PISM** | 0.16M | Differentiable Rayleigh-Mie scattering: virtual 266nm→193nm |
| **ASG** | 0.01M | 13-angle scatter generation via `torch.roll` (zero-cost) |
| **Dual-branch** | 0.37M | 266nm (precision) + virtual 193nm (recall) parallel detection |
| **SGF** | 0.008M | Scattering-guided fusion for wavelength-aware merging |

**Insight:** Replace a \$500K 193nm DUV light source with AI — Rayleigh gain `(266/193)⁴ ≈ 3.5×` for small defects.

### 2. 三阶段半监督训练 / Semi-supervised Training

```
50 labeled samples → Copy-Paste + Mosaic + MixUp → 10,000+ effective samples
    ↓
Stage 1: CycleGAN unsupervised pretraining (0 labels needed)
Stage 2: Joint fine-tuning (50 labels)
Stage 3: Knowledge distillation (7.5M → 4.2M, < 1% accuracy loss)
```

### 3. 边缘部署优化 / Edge Deployment

| Constraint | Target | Solution |
|-----------|--------|----------|
| Parameters | < 10M | RepViT-M0.9 + reparameterization |
| INT8 model | < 8MB | Per-channel quantization + sensitive-layer FP32 skip |
| ARM latency | < 80ms/frame | Polar transform + reparameterization fusion |
| 12" wafer | < 3min | 2000-4000 sub-images batched pipeline |

---

## 🇬🇧 Documentation / 🇨🇳 文档索引

| Doc | 说明 |
|-----|------|
| [Technical Reference](docs/technical-reference.md) | 完整技术路线（压缩版） |
| [Optical Physics](docs/optical-physics.md) | 光学物理约束注入详解 |
| [Project Manager Guide](docs/project-manager-guide.md) | 非技术负责人导读 |
| [Deployment Guide](docs/deployment-guide.md) | 部署运维指南 |
| [Qwen-VL Detection](docs/qwen-vl-detection.md) | Qwen-VL 晶圆检测方案 |
| [Showcase Design](docs/showcase-design.md) | 前端展厅设计文档 |
| [Competition Guide](docs/competition-guide.md) | 光电竞赛策划指导 |

---

## 🇬🇧 License / 🇨🇳 许可证

MIT License — see [LICENSE](LICENSE) for details.

## 🇬🇧 Contact / 🇨🇳 联系

- **School**: 东北大学秦皇岛分校 (Northeastern University at Qinhuangdao)
- **Project**: 晶圆缺陷检测·一体化 AI 解决方案
- **Year**: 2026

---

<p align="center">
  <strong>全链路自主可控 · 不依赖进口光学硬件 · 算法全栈自主</strong>
</p>
