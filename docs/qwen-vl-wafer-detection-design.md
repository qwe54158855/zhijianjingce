# 基于 Qwen-VL 的晶圆缺陷检测方案 — 设计文档

> **文档版本**：v1.0  
> **日期**：2026-07-18  
> **适用场景**：SiC/GaN 第三代半导体边缘检测、晶圆暗场→明场增强、AI 辅助缺陷分析报告  
> **核心思想**：用 Qwen-VL 的视觉理解能力替代原案复杂的光学物理注入方案，实现更灵活的增强+分析流水线  
> **模型**：Qwen2.5-VL-7B-Instruct-Q4_K_M (GGUF, ~4.6GB)  
> **推理引擎**：llama.cpp (纯 CPU 推理，无 GPU 依赖)  

---

## 1. 项目背景与设计动机

### 1.1 为什么要换路线

原技术路线（`COMPLETE_TECHNICAL_REFERENCE.md`）设计了一套极为精密的方案：

- RepViT 轻量骨干 + 双分支（增强+检测）
- 光学物理约束注入（PISM + ASG 多角度散射）
- 三阶段半监督训练
- INT8 量化 + TorchScript JIT → ARM 边缘部署

这套方案的**核心约束**是 ARM 边缘部署（< 80ms/帧，< 10M 参数），适用于量产产线的嵌入式设备。

新的需求场景不同：**在服务器/工作站上部署大模型，用视觉语言模型理解图像，实现更智能、更灵活的增强与分析**。

### 1.2 新路线的核心优势

| 维度 | 原方案 (ARM 边缘) | 新方案 (Qwen-VL 服务器) |
|------|-------------------|------------------------|
| 硬件 | ARM Cortex-A76 | Intel Ultra 9 CPU / 云 GPU |
| 模型规模 | < 10M 参数 | ~7B 参数 |
| 推理速度 | < 80ms/帧 | ~10-30s/帧 |
| 智能程度 | 固定检测输出 | 可理解、可对话、可解释 |
| 报告能力 | CSV 数据 | 自然语言 + 结构化报告 |
| 适应性 | 需重新训练 | 改 prompt 即可适配 |
| 部署复杂度 | 交叉编译 + INT8 量化 | Docker Compose 一键部署 |

### 1.3 方案演进路线

```
Phase A (立即可行)                    Phase B (后续升级)
┌─────────────────┐                  ┌─────────────────┐
│ Qwen-VL 分析     │                  │ Qwen-VL 分析     │
│       ↓          │                  │       ↓          │
│ OpenCV 增强引擎  │   ───→ 过渡      │ 轻量 CNN 增强    │
│       ↓          │                  │       ↓          │
│ Qwen 复审+报告   │                  │ Qwen 复审+报告   │
│       ↓          │                  │       ↓          │
│ YOLO 风格展示框  │                  │ YOLO 风格展示框  │
└─────────────────┘                  └─────────────────┘
```

Phase A 是所有后续步骤的基础：验证 Qwen-VL 的晶圆图像理解能力 + 建立完整的服务架构。  
Phase B 在增强质量上做升级，复用原 wafer-inspection 项目的部分能力。

---

## 2. 系统架构

### 2.1 整体架构

```
wafer-showcase (React 前端, Vite + Tailwind)
    │ 上传暗场晶圆图
    ▼
wafer-backend (Spring Boot, 端口 8080)
    │ POST /api/v1/qwen/enhance    ← 新增路由
    │ POST /api/v1/qwen/report     ← 新增路由
    ▼
wafer-qwen-service (FastAPI, 端口 8001)   ← 新增
    │
    ├── Qwen 分析流水线
    │   └──→ 调用 llama-server (端口 8002)
    │
    ├── OpenCV 增强引擎
    │   └── CLAHE + 去噪 + Gamma + 对比度
    │
    └── 检出框生成
        └── ROI 形态学检测 + YOLO 风格渲染
                │
                ▼
        返回: {增强图base64, 检测框列表, 分析报告}
                │
                ▼
wafer-showcase 展示:
    ┌──────────────────────────────┐
    │ 左右对比: 暗场原图 / 增强图    │
    │ 检测框叠加 (YOLO 风格)        │
    │ AI 分析报告面板               │
    └──────────────────────────────┘
```

### 2.2 服务组件

| 服务 | 技术栈 | 端口 | 说明 |
|------|--------|------|------|
| wafer-showcase | React 18 + Vite + Tailwind | 5173 (dev) | 前端展厅 |
| wafer-backend | Spring Boot + Java | 8080 | 后端 API |
| wafer-qwen-service | FastAPI + Python 3.10+ | 8001 | 增强+报告编排 |
| llama-server | llama.cpp C++ | 8002 | Qwen-VL 推理后端 |
| postgres | PostgreSQL 15 | 5432 | 持久化 |
| redis | Redis 7 | 6379 | 缓存 |
| minio | MinIO | 9000 | 图像存储 |

### 2.3 wafer-qwen-service 内部结构

```
wafer-qwen-service/
├── main.py                  # FastAPI 应用入口
├── core/
│   ├── config.py            # 配置（模型路径、端口等）
│   └── model_manager.py     # llama.cpp 客户端（HTTP 调用）
├── engine/
│   ├── enhancer.py          # OpenCV 增强流水线
│   ├── detector.py          # 缺陷检出（形态学 + ROI）
│   ├── reporter.py          # Qwen 报告生成
│   └── visualizer.py        # 检测框渲染（YOLO 风格）
├── prompts/
│   ├── enhance.yaml         # Qwen 增强分析 prompt
│   ├── detect.yaml          # Qwen 缺陷识别 prompt
│   └── report.yaml          # Qwen 报告生成 prompt
├── api/
│   └── routes.py            # API 路由
├── models/
│   └── schemas.py           # 请求/响应模型
├── Dockerfile
└── requirements.txt
```

---

## 3. 核心流水线设计

### 3.1 全流程

```
[输入] 暗场晶圆图 (JPG/PNG)
    │
    ├── Step 1: Qwen 图像分析
    │   ├── 分析亮度/噪声/对比度
    │   ├── 识别疑似缺陷区域+类型
    │   └── 输出结构化参数 + 缺陷标注
    │
    ├── Step 2: OpenCV 增强 (根据 Qwen 参数)
    │   ├── CLAHE 自适应直方图均衡
    │   ├── NLM 去噪 (保留边缘)
    │   ├── Gamma 校正
    │   └── 对比度拉伸 + 锐化
    │
    ├── Step 3: 缺陷检出 (在 Qwen ROI 内)
    │   ├── 自适应阈值分割
    │   ├── 形态学开运算
    │   ├── 轮廓查找 → 最小外接矩形
    │   └── 置信度评分
    │
    ├── Step 4: Qwen 复审
    │   ├── 看增强后图像
    │   ├── 验证检测框是否合理
    │   └── 生成缺陷分析报告
    │
    └── [输出] {增强图, 检测框列表, 分析报告}
```

### 3.2 Step 1: Qwen 图像分析 (enhance prompt)

```yaml
# prompts/enhance.yaml
system: >
  你是半导体晶圆缺陷检测专家。
  分析这张暗场晶圆图像，输出 JSON 格式的分析结果。

user: >
  分析这张暗场晶圆图像，特别注意：
  1. 图像整体亮度、噪声水平、清晰度
  2. 是否存在疑似缺陷（崩边/颗粒/划痕/位错）
  3. 缺陷的位置（用上下左右描述）、大小、形状
  
  输出 JSON，格式如下：
  {
    "analysis": {
      "brightness": "低|中|高",
      "noise_level": "低|中|高",
      "sharpness": "模糊|清晰",
      "suspected_defects": [
        {"type": "崩边|颗粒|划痕|位错", "confidence": 0.0-1.0,
         "region": "位置描述", "size": "小|中|大"}
      ]
    },
    "enhance_params": {
      "clahe_clip": 2.0-4.0,
      "clahe_grid": 4或8,
      "denoise_strength": 5-20,
      "gamma": 0.8-2.0,
      "contrast": 1.0-2.0,
      "sharpen": true或false
    }
  }
  
  只输出 JSON，不要额外文字。
```

### 3.3 Step 2: OpenCV 增强引擎

```python
# engine/enhancer.py - 核心增强流水线

def enhance(image: np.ndarray, params: dict) -> np.ndarray:
    """根据 Qwen 分析的参数执行 OpenCV 增强"""
    img = image.copy()

    # 如果是彩色图转灰度
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. CLAHE - 自适应直方图均衡
    clahe = cv2.createCLAHE(
        clipLimit=params.get('clahe_clip', 2.5),
        tileGridSize=(params.get('clahe_grid', 8),
                      params.get('clahe_grid', 8))
    )
    img = clahe.apply(img)

    # 2. Non-Local Means 去噪（保留边缘）
    img = cv2.fastNlMeansDenoising(
        img, None,
        params.get('denoise_strength', 10),
        7, 21
    )

    # 3. Gamma 校正
    gamma = params.get('gamma', 1.2)
    img = (img / 255.0) ** gamma * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 4. 对比度拉伸
    contrast = params.get('contrast', 1.3)
    img = cv2.multiply(img, contrast)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 5. 锐化
    if params.get('sharpen', True):
        kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]]) / 1.0
        img = cv2.filter2D(img, -1, kernel)

    return img
```

### 3.4 Step 3: 缺陷检出 (detector.py)

在 Qwen 标注的 ROI 区域内执行轻量形态学检测：

```python
# engine/detector.py

def detect_in_roi(enhanced_img, roi, defect_type, confidence):
    """
    在 ROI 区域内用 CV 方法检出缺陷轮廓
    
    Args:
        enhanced_img: 增强后的灰度图
        roi: (x, y, w, h) 归一化坐标
        defect_type: 缺陷类型
        confidence: Qwen 给出的置信度
    
    Returns:
        list of Detection
    """
    h, w = enhanced_img.shape
    x, y, rw, rh = [int(v * d) for v, d in zip(roi, [w, h, w, h])]
    roi_img = enhanced_img[y:y+rh, x:x+rw]

    # 自适应阈值
    thresh = cv2.adaptiveThreshold(
        roi_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    # 形态学开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 轮廓查找
    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:  # 过滤噪声
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        boxes.append(Detection(
            type=defect_type,
            confidence=min(confidence, 0.5 + area / 5000),
            bbox={
                "x": x + bx, "y": y + by,
                "w": bw, "h": bh
            }
        ))

    return boxes
```

### 3.5 Step 4: YOLO 风格检测框渲染 (visualizer.py)

```python
# engine/visualizer.py

# YOLO 风格的类别颜色
CLASS_COLORS = {
    "崩边":   (0, 0, 255),     # 红色
    "颗粒":   (255, 191, 0),   # 青色
    "划痕":   (0, 255, 0),     # 绿色
    "位错":   (255, 0, 255),   # 紫色
}

def draw_detections(image, detections):
    """在增强图上绘制 YOLO 风格的检测框"""
    img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    for det in detections:
        color = CLASS_COLORS.get(det.type, (255, 255, 255))
        b = det.bbox

        # 画框（2px 实线）
        cv2.rectangle(img_rgb,
            (b['x'], b['y']),
            (b['x'] + b['w'], b['y'] + b['h']),
            color, 2)

        # 标签背景（YOLO 风格半透明填充）
        label = f"{det.type} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label,
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_rgb,
            (b['x'], b['y'] - th - 4),
            (b['x'] + tw + 4, b['y']),
            color, -1)

        # 标签文字
        cv2.putText(img_rgb, label,
            (b['x'] + 2, b['y'] - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (255, 255, 255), 1, cv2.LINE_AA)

    return img_rgb
```

---

## 4. API 设计

### 4.1 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/qwen/enhance` | POST | Qwen 分析 + 增强 + 检出框，返回增强图和检测列表 |
| `/api/v1/qwen/report` | POST | 基于增强图生成分析报告 |
| `/api/v1/qwen/analyze` | POST | 仅做分析（不增强），返回 Qwen 原始分析 JSON |

### 4.2 请求/响应格式

**POST /api/v1/qwen/enhance**

请求：
```json
{
  "image": "base64_encoded_image_data",
  "format": "jpg|png"
}
```

响应：
```json
{
  "success": true,
  "enhanced_image": "base64_encoded_enhanced_image",  // 带检测框
  "detections": [
    {
      "type": "崩边",
      "confidence": 0.85,
      "bbox": {"x": 320, "y": 50, "w": 42, "h": 18},
      "source": "qwen+cv"
    }
  ],
  "analysis": {
    "brightness": "低",
    "noise_level": "中",
    "defect_count": 2
  },
  "inference_time_ms": 18500
}
```

---

## 5. Docker 部署

### 5.1 新增服务

```yaml
# docker-compose.yml 新增部分

  wafer-qwen-service:
    build: ./wafer-qwen-service
    ports: ["8001:8001"]
    environment:
      LLAMA_SERVER_URL: http://llama-server:8002/v1
      LOG_LEVEL: INFO
    volumes:
      - ./wafer-qwen-service:/app
    depends_on:
      - llama-server
    restart: unless-stopped

  llama-server:
    image: ghcr.io/ggerganov/llama.cpp:full
    ports: ["8002:8002"]
    volumes:
      - type: bind
        source: /mnt/d/models
        target: /models
    command:
      - --model
      - /models/qwen2.5-vl-7b-instruct-q4_k_m.gguf
      - --port
      - "8002"
      - --host
      - "0.0.0.0"
      - --ctx-size
      - "8192"
      - --batch-size
      - "512"
      - --n-gpu-layers
      - "0"
    restart: unless-stopped
```

### 5.2 Spring Boot 新增路由

在 wafer-backend 中新增 Qwen 服务代理路由，方向与现有 `lora-service` 一致：

```java
// 在 wafer-backend 中新增 QwenController
@RestController
@RequestMapping("/api/v1/qwen")
public class QwenController {

    private final RestTemplate restTemplate;

    @Value("${qwen.service.url:http://wafer-qwen-service:8001}")
    private String qwenServiceUrl;

    @PostMapping("/enhance")
    public ResponseEntity<?> enhance(@RequestBody Map<String, Object> request) {
        // 上传图像到 MinIO
        // 调用 wafer-qwen-service 的 /api/v1/qwen/enhance
        // 保存增强结果
        // 返回增强图 base64 + 检测列表
    }

    @PostMapping("/report")
    public ResponseEntity<?> report(@RequestBody Map<String, Object> request) {
        // 基于增强结果生成分析报告
    }
}
```

---

## 6. 检出效果预期

### 6.1 检出方式对比

| 方案 | 检出方式 | 精度 | 速度 | 适用阶段 |
|------|---------|------|------|---------|
| Qwen 直接识别 | 视觉语言模型理解 | ★★☆ | 10-30s | Phase A |
| Qwen + CV 形态学 | Qwen 语义引导 + CV 检测 | ★★★ | 10-20s | Phase A |
| 轻量 CNN 增强+检测 | 训练专用模型 | ★★★★ | < 1s | Phase B |

### 6.2 YOLO 展示要求

- **框样式**：2px 实线 + 半透明标签背景，每类缺陷固定颜色
- **标签**：`缺陷类型 + 置信度`（如 `崩边 0.85`）
- **置信度≥0.5 才展示**，低于此的标注为"可疑区域"

---

## 7. 风险与应对

| 风险 | 概率 | 应对 |
|------|------|------|
| Qwen 看图分析不准 | 中 | 提供 5-10 张标注样例做 few-shot；调 prompt 迭代 |
| CPU 推理超 30s/张 | 中 | 降低 ctx-size；换更小模型如 Qwen2.5-VL-3B |
| CV 形态学检出误报高 | 高 | Qwen 复审过滤；提高面积阈值 |
| llama.cpp 下载慢 | 低 | 使用 ModelScope 国内源；或预下载 |
| 前端渲染大图慢 | 低 | 返回 1024px 缩放图；渐进式加载 |

---

## 8. 与原有项目的关系

| 原有项目 | 角色 | 变更 |
|---------|------|------|
| `wafer-showcase` | 前端展示 | 新增增强对比+检测框展示面板 |
| `wafer-backend` | 后端 API | 新增 /api/v1/qwen/* 路由 |
| `wafer-lora-service` | SD LoRA 推理 | 保持，可选留作对照 |
| `wafer-inspection` | 训练框架 | Phase B 复用增强训练数据 |
| `docs/COMPLETE_TECHNICAL_REFERENCE.md` | 原技术路线 | 留存，作为参考 |
| `ppt-master-main/...` | 原方案 PPT | 留存，作为参考 |
