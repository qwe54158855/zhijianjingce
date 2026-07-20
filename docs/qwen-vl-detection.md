# 基于 Qwen-VL 的晶圆缺陷检测方案

> **版本**：v1.0 | **模型**：Qwen2.5-VL-7B-Instruct-Q4_K_M (GGUF, ~4.6GB) | **推理**：llama.cpp (纯 CPU)

---

## 1. 方案定位

Qwen-VL 方案与 ARM 边缘部署方案互为补充：

| 维度 | 原方案（ARM 边缘） | Qwen-VL 方案（服务器） |
|------|-------------------|----------------------|
| 硬件 | ARM Cortex-A76 | Intel Ultra 9 CPU / 云 GPU |
| 模型规模 | < 10M 参数 | ~7B 参数 |
| 推理速度 | < 80ms/帧 | ~10-30s/帧 |
| 智能程度 | 固定检测输出 | 可理解、可对话、可解释 |
| 报告能力 | CSV 数据 | 自然语言 + 结构化报告 |
| 适应性 | 需重新训练 | 改 prompt 即可适配 |

## 2. 方案演进路线

```
Phase A（立即可行）                  Phase B（后续升级）
┌─────────────────┐                ┌─────────────────┐
│ Qwen-VL 分析     │                │ Qwen-VL 分析     │
│       ↓          │                │       ↓          │
│ OpenCV 增强引擎  │  ───→ 过渡     │ 轻量 CNN 增强    │
│       ↓          │                │       ↓          │
│ Qwen 复审+报告   │                │ Qwen 复审+报告   │
│       ↓          │                │       ↓          │
│ YOLO 风格展示框  │                │ YOLO 风格展示框  │
└─────────────────┘                └─────────────────┘
```

## 3. 系统架构

```
用户上传 → wafer-qwen-service (FastAPI)
  ├── OpenCV Enhancer：去噪 + 对比度增强 + 自适应直方图均衡
  ├── Circular Detector：Hough 圆检测 + 参数筛选
  ├── Qwen-VL Analyzer：llama.cpp HTTP 客户端 → 结构化 JSON 输出
  └── Reporter：YAML prompt + Qwen 自然语言 → 完整报告 + 可视化

wafer-backend Feign Client → 存储结果 + SSE 进度推送
wafer-showcase React Page → 展示增强图 + 检测结果 + 分析报告
```

## 4. API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/enhance` | POST | OpenCV 增强流水线 |
| `/api/v1/detect` | POST | 圆形检测 + Qwen 分析 |
| `/api/v1/analyze` | POST | Qwen-VL 文本分析 |
| `/api/v1/report` | POST | 完整检测报告生成 |
| `/api/v1/angular-views` | POST | 多角度视图生成 |

## 5. 部署

```bash
# 需要先下载 Qwen GGUF 模型
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct-GGUF \
  qwen2.5-vl-7b-instruct-q4_k_m.gguf --local-dir /models/

# Docker Compose 一键部署
docker-compose up -d wafer-qwen-service llama-server
```
