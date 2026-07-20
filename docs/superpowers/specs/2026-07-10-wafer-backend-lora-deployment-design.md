# 晶圆检测后端 + LoRA 推理部署系统 — 设计文档

> **版本**：v1.0  
> **日期**：2026-07-10  
> **项目路径**：  
> - 后端：`D:/cy/wafer-backend/` (新项目)  
> - LoRA 服务：`D:/cy/wafer-lora-service/` (新项目)  
> - 前端：`D:/cy/wafer-showcase/` (已有)  
> - 模型：`D:/cy/wafer-inspection/` (已有)  
> - 标定数据：`D:/cy/193nm/` (已有 5 张真实 193nm 图像)  
>
> **引用文档**：
> - `docs/optical-physics-technical-deepdive.md` (物理散射方案 v2.0)
> - `docs/wafer-showcase-design.md` (前端展示设计)
> - `docs/superpowers/specs/2026-07-06-optical-physics-injection-design.md` (物理注入规格)

---

## 一、项目定位

本项目为**半导体晶圆缺陷检测一体化 AI 解决方案**的后端配套系统，旨在：

1. 用 Spring Boot 构建生产级后端（API 网关 / 业务逻辑 / 存储 / 任务编排）
2. 部署三路 LoRA 微调 Stable Diffusion 模型，模拟 `wafer-inspection` PyTorch 模型管线的三个核心效果
3. 为 `wafer-showcase` 前端展厅提供**混合模式**内容——静态预生成素材 + 实时在线推理

---

## 二、总体架构

```
┌──────────────────────────────────────────────────────────────┐
│                    前端层 (wafer-showcase)                    │
│  React 18 + Vite 5 + Tailwind CSS 3 + Framer Motion         │
│                                                              │
│  展厅模式 → 静态预生成素材 (从后端 gallery API 获取)         │
│  工作台模式 → 上传原图 → 选 LoRA 模式 → 实时推理展示         │
└──────────────────────┬───────────────────────────────────────┘
                       │ HTTPS / REST + SSE 进度推送
                       ▼
┌──────────────────────────────────────────────────────────────┐
│                  API 网关层 (Spring Boot 3.2)                 │
│                                                              │
│  控制器层:                                                    │
│    GalleryController   /api/v1/gallery/*   素材 CRUD          │
│    InferenceController /api/v1/inference/* 推理编排           │
│    ImageController     /api/v1/images/*   文件上传/管理      │
│    MetricsController   /api/v1/metrics/*  指标展示            │
│                                                              │
│  服务层:                                                      │
│    InferenceService → 异步任务编排 → Feign 调用 FastAPI      │
│    GalleryService   → 素材查询/排序/缓存                      │
│    StorageService   → MinIO 上传/下载/缩略图                  │
│    CacheService     → Redis 结果缓存                          │
│                                                              │
│  数据层: JPA + PostgreSQL 15 / Redis 7 / MinIO               │
└──────┬──────────────────────────────┬────────────────────────┘
       │ HTTP REST                     │ HTTP REST
       ▼                               ▼
┌──────────────────┐     ┌──────────────────────────────────────┐
│  PostgreSQL 15   │     │  FastAPI LoRA 推理微服务             │
│  + MinIO (S3)    │     │  Port 8000 │ GPU 6GB+ VRAM           │
│                  │     │                                      │
│  · gallery_item  │     │  /api/v1/lora/infer   执行推理       │
│  · inference_task│     │  /api/v1/lora/switch  热切换 LoRA   │
│  · 用户(可选)    │     │  /api/v1/lora/active  当前 LoRA      │
│                  │     │  /api/v1/health       健康检查        │
└──────────────────┘     │                                      │
                         │  底座: SD 1.5 (FP16, torch.compile)  │
                         │  LoRA ×3: enhance/wavelength/defect  │
                         │  ControlNet: Canny 边缘 + 增益图条件  │
                         └──────────────────────────────────────┘
```

### 核心架构原则

1. **三层解耦**：前端展示 / 后端业务 / AI 推理 各自独立部署扩缩容
2. **异步推理**：所有 AI 推理通过任务队列异步执行，防止 HTTP 阻塞
3. **混合数据流**：展厅模式走缓存素材（零延迟），工作台模式走实时推理（SSE 推送）
4. **物理约束**：LoRA 推理通过 ControlNet 注入 PISM 增益图保持物理合理性

---

## 三、Spring Boot 后端设计

### 3.1 项目结构

```
wafer-backend/
├── pom.xml                              # Spring Boot 3.2 + JDK 17
│
├── src/main/java/com/wafer/
│   ├── WaferApplication.java
│   │
│   ├── config/
│   │   ├── WebConfig.java               # CORS + 静态资源
│   │   ├── SecurityConfig.java          # JWT 认证 (可选)
│   │   ├── RedisConfig.java             # Redis 连接 + 序列化
│   │   ├── MinIOConfig.java             # MinIO 客户端配置
│   │   └── AsyncConfig.java             # 异步任务线程池
│   │
│   ├── controller/
│   │   ├── GalleryController.java       # 展厅素材 CRUD API
│   │   ├── InferenceController.java     # 推理任务提交/查询
│   │   ├── ImageController.java         # 图片上传/管理
│   │   └── MetricsController.java       # 指标统计 API
│   │
│   ├── service/
│   │   ├── InferenceService.java        # 推理编排 (异步)
│   │   ├── GalleryService.java          # 素材管理
│   │   ├── StorageService.java          # MinIO 文件存储
│   │   └── CacheService.java            # Redis 缓存
│   │
│   ├── client/
│   │   └── LoraInferenceClient.java     # Feign → FastAPI
│   │
│   ├── model/
│   │   ├── entity/
│   │   │   ├── GalleryItem.java         # 展厅素材
│   │   │   └── InferenceTask.java       # 推理任务记录
│   │   ├── dto/
│   │   │   ├── InferenceRequest.java
│   │   │   ├── InferenceResponse.java
│   │   │   └── GalleryItemDTO.java
│   │   └── enums/
│   │       ├── TaskStatus.java          # PENDING/RUNNING/DONE/FAILED
│   │       └── InferenceType.java       # ENHANCE/WAVELENGTH/DEFECT
│   │
│   ├── repository/
│   │   ├── GalleryRepository.java
│   │   └── InferenceTaskRepository.java
│   │
│   └── exception/
│       └── GlobalExceptionHandler.java  # @ControllerAdvice
│
└── src/main/resources/
    ├── application.yml
    ├── application-dev.yml
    └── application-prod.yml
```

### 3.2 API 端点

| 方法 | 路径 | 功能 | 说明 |
|------|------|------|------|
| `GET` | `/api/v1/gallery` | 展厅素材列表 | 分页，按 category/tags 筛选 |
| `GET` | `/api/v1/gallery/{id}` | 素材详情 | 含原图/结果图/指标 URL |
| `GET` | `/api/v1/gallery/stats` | 展厅统计 | 推理总量/平均增益等 |
| `POST` | `/api/v1/inference` | 提交推理任务 | multipart: file + type + params |
| `GET` | `/api/v1/inference/{id}` | 查询任务结果 | 返回 status/output_url/metrics |
| `GET` | `/api/v1/inference/{id}/stream` | SSE 进度推送 | 实时推理进度 |
| `POST` | `/api/v1/inference/batch` | 批量推理 | 用于预生成素材 |
| `POST` | `/api/v1/images/upload` | 上传图片 | 自动生成缩略图 |
| `GET` | `/api/v1/images/{filename}` | 获取图片 | 支持 ?thumbnail=true |
| `GET` | `/api/v1/metrics/overview` | 指标总览 | 响应延迟/推理次数等 |
| `GET` | `/api/v1/health` | 健康检查 | 含外部服务状态 |

### 3.3 异步推理流程

```
1. 用户 POST /api/v1/inference
   └→ 图片上传 → MinIO
   └→ InferenceTask 创建 (status=PENDING)
   └→ 返回 { taskId, status }

2. @Async 线程池消费
   └→ status → RUNNING
   └→ Redis 写入缓存结果
   └→ LoraInferenceClient.infer(type, image_url, params)
       └→ FastAPI POST /api/v1/lora/infer

3. FastAPI 推理完成
   └→ 结果图片 → MinIO
   └→ status → DONE
   └→ 写入 InferenceTask 数据库
   └→ SSE 推送完成通知

4. 前端轮询 /inference/{id} 或 SSE 接收
```

### 3.4 数据库设计

```sql
-- 展厅素材表
CREATE TABLE gallery_item (
    id              BIGSERIAL    PRIMARY KEY,
    title           VARCHAR(200) NOT NULL,
    description     TEXT,
    category        VARCHAR(50)  NOT NULL,    -- enhance/wavelength/defect
    tags            JSONB        DEFAULT '[]',
    original_url    VARCHAR(500) NOT NULL,    -- 原图 MinIO URL
    result_url      VARCHAR(500) NOT NULL,    -- LoRA 结果图 URL
    diff_url        VARCHAR(500),             -- 差异对比图
    thumbnail_url   VARCHAR(500),
    metrics         JSONB        DEFAULT '{}', -- {gain: 3.5, psnr:...}
    display_order   INT          DEFAULT 0,
    created_at      TIMESTAMP    DEFAULT NOW(),
    updated_at      TIMESTAMP    DEFAULT NOW()
);

CREATE INDEX idx_gallery_category ON gallery_item(category);
CREATE INDEX idx_gallery_order   ON gallery_item(display_order);

-- 推理任务表
CREATE TABLE inference_task (
    id              BIGSERIAL    PRIMARY KEY,
    type            VARCHAR(20)  NOT NULL,     -- enhance/wavelength/defect
    status          VARCHAR(20)  NOT NULL,     -- PENDING/RUNNING/DONE/FAILED
    input_url       VARCHAR(500) NOT NULL,
    output_url      VARCHAR(500),
    thumbnail_url   VARCHAR(500),
    params          JSONB        DEFAULT '{}', -- {strength, steps, ...}
    error_message   TEXT,
    duration_ms     INT,
    created_at      TIMESTAMP    DEFAULT NOW(),
    completed_at    TIMESTAMP,
    client_ip       VARCHAR(45),
    user_agent      VARCHAR(300)
);

CREATE INDEX idx_task_status     ON inference_task(status);
CREATE INDEX idx_task_created    ON inference_task(created_at DESC);
```

### 3.5 依赖 (pom.xml 核心)

```xml
<parent>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-parent</artifactId>
    <version>3.2.5</version>
</parent>

<properties>
    <java.version>17</java.version>
    <spring-cloud.version>2023.0.1</spring-cloud.version>
</properties>

<dependencies>
    <!-- Web -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
    <!-- JPA + PostgreSQL -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-data-jpa</artifactId>
    </dependency>
    <dependency>
        <groupId>org.postgresql</groupId>
        <artifactId>postgresql</artifactId>
    </dependency>
    <!-- Redis -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-data-redis</artifactId>
    </dependency>
    <!-- Feign 客户端 → FastAPI -->
    <dependency>
        <groupId>org.springframework.cloud</groupId>
        <artifactId>spring-cloud-starter-openfeign</artifactId>
    </dependency>
    <!-- MinIO -->
    <dependency>
        <groupId>io.minio</groupId>
        <artifactId>minio</artifactId>
        <version>8.5.10</version>
    </dependency>
    <!-- Validation -->
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-validation</artifactId>
    </dependency>
    <!-- Lombok -->
    <dependency>
        <groupId>org.projectlombok</groupId>
        <artifactId>lombok</artifactId>
        <optional>true</optional>
    </dependency>
</dependencies>
```

---

## 四、FastAPI LoRA 推理微服务

### 4.1 项目结构

```
wafer-lora-service/
├── main.py                            # FastAPI 入口 + 生命周期
├── requirements.txt                   # torch, diffusers, controlnet, etc.
├── Dockerfile                         # CUDA 12.x + PyTorch 2.x
│
├── api/
│   └── routes/
│       ├── __init__.py
│       ├── infer.py                   # 推理路由
│       └── lora.py                    # LoRA 管理路由
│
├── core/
│   ├── __init__.py
│   ├── model_manager.py               # SD + LoRA 加载/热切换
│   ├── scheduler.py                   # 推理队列调度
│   └── config.py                      # YAML 配置
│
├── loras/                             # LoRA 权重文件
│   ├── enhance_lora.safetensors
│   ├── wavelength_lora.safetensors
│   └── defect_lora.safetensors
│
└── models/
    ├── __init__.py
    └── infer_schemas.py               # Pydantic 请求/响应
```

### 4.2 三种 LoRA 推理模式

| 模式 | 路径 | LoRA 权重 | ControlNet 条件 | 输入 | 输出 | 模拟目标 |
|------|------|-----------|----------------|------|------|---------|
| 暗场→明场 | `/infer/enhance` | `enhance_lora` | Canny 边缘 | 暗场晶圆图 | 增强明场图 | CycleGAN 解码器 |
| 波长转换 | `/infer/wavelength` | `wavelength_lora` | PISM 增益图 + Canny | 266nm 图 | 虚拟 193nm 图 | PISM 散射增强 |
| 缺陷生成 | `/infer/defect` | `defect_lora` | 缺陷掩码 + Canny | 无缺陷图 | 带缺陷合成图 | 训练数据增强 |

### 4.3 模型管理器核心设计

```python
class ModelManager:
    """SD 模型生命周期 + LoRA 热切换管理器"""

    MODELS_DIR = Path(__file__).parent.parent / "loras"

    def __init__(self):
        # 加载 SD 1.5 底座 (FP16, CPU offload)
        self.base = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.base.to("cuda")
        self.base.enable_model_cpu_offload()
        self.base.enable_vae_slicing()

        # LoRA 注册表
        self.lora_registry = {
            "enhance":    self.MODELS_DIR / "enhance_lora.safetensors",
            "wavelength": self.MODELS_DIR / "wavelength_lora.safetensors",
            "defect":     self.MODELS_DIR / "defect_lora.safetensors",
        }
        self.active_loras: dict[str, float] = {}

        # ControlNet
        self.controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-canny",
            torch_dtype=torch.float16,
        ).to("cuda")

    def load_lora(self, name: str, weight: float = 1.0):
        path = self.lora_registry[name]
        self.base.load_lora_weights(path, adapter_name=name)
        self.active_loras[name] = weight

    def switch_lora(self, name: str):
        self.base.delete_adapters(list(self.active_loras.keys()))
        self.active_loras = {}
        self.load_lora(name)

    @torch.inference_mode()
    def infer(self, image: Image, prompt: str,
              control_image: Image | None = None,
              strength: float = 0.75) -> Image:
        return self.base(
            image=image, prompt=prompt,
            control_image=control_image,
            strength=strength,
            num_inference_steps=20,
            cross_attention_kwargs={
                "scale": list(self.active_loras.values())
            },
        ).images[0]
```

### 4.4 LoRA 训练方案

#### 4.4.1 wavelength LoRA (266nm → 193nm) — 核心

利用 `wafer-inspection` 的 PISM 模块解决小样本问题：

```
阶段 1 — PISM 伪标签生成
  wafer-inspection PISM 模块前向:
    266nm 图像 → feats_193 + gain_map
  → 用 EnhanceDecoder 渲染为可视化图像
  → 获得 ~1000 对 (266nm, pseudo_193nm) 训练数据

阶段 2 — 真实标定片权重上调
  5 张 D:/cy/193nm/ 真实图像 loss × λ=5.0
  确保模型优先拟合真实物理散射

阶段 3 — 物理约束 Loss (可选增强)
  L_total = L_SD + 0.05 × ||PISM(gen_193) - PISM(feat_266)||²
```

#### 4.4.2 LoRA 超参数

```yaml
base_model: runwayml/stable-diffusion-v1-5
resolution: 512
lora_rank: 64
lora_alpha: 32
target_modules:
  - to_q  - to_k  - to_v  - to_out.0
  - proj_in  - proj_out
  - ff.net.0.proj  - ff.net.2

training:
  batch_size: 4
  learning_rate: 1e-4
  lr_scheduler: cosine
  num_epochs: 50
  mixed_precision: fp16
  gradient_accumulation_steps: 2
  use_8bit_adam: true
  dataloader_num_workers: 4
  flip_p: 0.5
  random_crop: true
  random_brightness: 0.1
```

#### 4.4.3 LoRA 权重汇总

| LoRA | Rank | 文件大小 | 加载耗时 | 训练数据来源 | 训练数据量 |
|------|------|----------|---------|-------------|-----------|
| `wavelength_lora` | 64 | ~35MB | ~2s | PISM 伪标签 + 5 张真实标定片 | ~1000 对 |
| `enhance_lora` | 64 | ~35MB | ~2s | CycleGAN 训练管线产出 | ~100 对 |
| `defect_lora` | 64 | ~35MB | ~2s | 检测标注提取缺陷区域 + 合成 | ~200 对 |

总显存占用：SD 底座 ~1.5GB + 三 LoRA ~100MB + ControlNet ~400MB ≈ **2GB**，推荐 6GB+ VRAM。

---

## 五、文件存储设计 (MinIO)

```
wafer-images/                        # Bucket
├── uploads/{YYYY}/{MM}/{DD}/
│   ├── {uuid}_original.jpg          # 用户上传原图
│   └── {uuid}_thumb.jpg             # 缩略图 (256×256)
├── results/{task_id}/
│   ├── result.jpg                   # LoRA 推理结果
│   └── diff.jpg                     # 差异对比热力图
├── gallery/
│   ├── enhance/                     # 增强类预生成素材
│   ├── wavelength/                  # 波长转换类预生成素材
│   └── defect/                      # 缺陷生成类预生成素材
└── lora_training/
    └── wavelength_pairs/            # 266nm↔193nm 配对训练数据
```

---

## 六、前端扩展设计

### 6.1 新增页面

```diff
 wafer-showcase/src/
+ ├── pages/
+ │   ├── Gallery.jsx               # 展厅浏览 (数据来自后端 API)
+ │   ├── Workbench.jsx             # 推理工作台 (上传+推理+对比)
+ │   └── Metrics.jsx               # 实时指标面板
  ├── components/
+ │   ├── ImageUploader.jsx         # 拖拽/点击上传
+ │   ├── InferenceResult.jsx       # 并排对比原图/结果/差异
+ │   ├── ModelSelector.jsx         # LoRA 模式切换卡片
+ │   └── ProgressTracker.jsx       # 推理进度动画
+ ├── hooks/
+ │   ├── useInference.js           # 推理 API hook
+ │   └── useGallery.js             # 素材 API hook
+ └── api/
+     └── index.js                  # axios 封装
```

### 6.2 API 客户端

```javascript
const BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8080/api/v1'

export const api = {
  getGallery:  (params) => axios.get(`${BASE}/gallery`, { params }),
  getGalleryStats: ()     => axios.get(`${BASE}/gallery/stats`),

  submitInference: (type, file, params) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('type', type)
    if (params) fd.append('params', JSON.stringify(params))
    return axios.post(`${BASE}/inference`, fd)
  },

  getTaskStatus: (id) => axios.get(`${BASE}/inference/${id}`),
  getTaskStream: (id) => new EventSource(`${BASE}/inference/${id}/stream`),

  getMetrics: () => axios.get(`${BASE}/metrics/overview`),
}
```

### 6.3 工作台交互流程

```
用户进入工作台
  → 选择 LoRA 模式 (三张卡片: 增强/波长转换/缺陷生成)
  → 拖拽/点击上传晶圆图
  → 参数配置 (推理强度滑块 0.5~1.0)
  → 点击"开始推理"

前端:
  POST /inference → 返回 { taskId }
  GET /inference/{taskId} 轮询 (每 1s)
  或 SSE /inference/{taskId}/stream

完成后:
  三栏并排展示: 原图 | LoRA 结果 | 差异对比
  显示指标: 推理耗时 / 增益值 / PSNR
  按钮: 重新推理 / 下载 / 加入展厅素材
```

---

## 七、部署方案

### 7.1 Docker Compose

```yaml
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports: ["80:80", "443:443"]
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on: [spring-api, wafer-showcase]

  wafer-showcase:
    build: ./wafer-showcase
    ports: ["3000:80"]

  spring-api:
    build: ./wafer-backend
    ports: ["8080:8080"]
    environment:
      - SPRING_PROFILES_ACTIVE=prod
      - DB_URL=jdbc:postgresql://postgres:5432/wafer
      - REDIS_HOST=redis
      - MINIO_ENDPOINT=http://minio:9000
      - LORA_SERVICE_URL=http://lora-service:8000
    depends_on: [postgres, redis, minio]

  lora-service:
    build: ./wafer-lora-service
    ports: ["8000:8000"]
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./loras:/app/loras          # LoRA 权重热挂载
    depends_on: [minio]

  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: wafer
      POSTGRES_USER: wafer
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: ${MINIO_USER}
      MINIO_ROOT_PASSWORD: ${MINIO_PASS}
    volumes:
      - minio_data:/data

volumes:
  pgdata:
  minio_data:
```

### 7.2 Nginx 路由

```nginx
# /etc/nginx/nginx.conf
server {
    listen 80;

    # 前端静态站
    location / {
        proxy_pass http://wafer-showcase:80;
    }

    # 后端 API
    location /api/ {
        proxy_pass http://spring-api:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # SSE 支持
        proxy_buffering off;
        proxy_cache off;
        chunked_transfer_encoding on;
    }

    # MinIO 图片直链 (可选)
    location /images/ {
        proxy_pass http://minio:9000/wafer-images/;
    }
}
```

---

## 八、迭代路线图

### 里程碑 1：后端骨架 (2周)

| 时间 | 内容 | 产出 |
|------|------|------|
| 第 1-2 天 | Spring Boot 脚手架 + pom 依赖 | 可启动项目 |
| 第 3-4 天 | PostgreSQL + MinIO + JPA 实体 + Repository | DB 层完成 |
| 第 5-6 天 | GalleryController + service CRUD | 素材 API 可用 |
| 第 7-8 天 | 文件上传 + 缩略图 + 静态资源 | 图片管理完成 |
| 第 9-10 天 | Nginx 配置 + Docker Compose 编排 | **MS1 交付** |

### 里程碑 2：LoRA 训练 + 推理服务 (3周)

| 时间 | 内容 | 产出 |
|------|------|------|
| 第 1-2 天 | FastAPI 脚手架 + GPU 容器化 | 推理服务可启动 |
| 第 3-4 天 | SD 1.5 加载 + LoRA 热切换机制 | 底座就绪 |
| 第 5-7 天 | PISM 伪标签生成管线 | ~1000 对训练数据 |
| 第 8-11 天 | wavelength LoRA 训练 + 验证 | 波长转换可用 |
| 第 12-13 天 | enhance LoRA 训练 | 增强转换可用 |
| 第 14-15 天 | defect LoRA 训练 | 缺陷生成可用 |
| 第 16-17 天 | ControlNet 物理约束集成 | 质量提升 |
| 第 18-21 天 | 全面测试 + 效果验证 | **MS2 交付** |

### 里程碑 3：后端↔LoRA 联调 (2周)

| 时间 | 内容 | 产出 |
|------|------|------|
| 第 1-2 天 | LoraInferenceClient Feign 客户端 | HTTP 通信 |
| 第 3-4 天 | InferenceService 异步编排 + Redis 缓存 | 任务队列 |
| 第 5-6 天 | SSE 进度推送 | 实时反馈 |
| 第 7-8 天 | 工作台前端页面 (上传 + 选择 + 结果) | UI 可用 |
| 第 9-10 天 | 展厅接入后端 API | 全端到端 |
| 第 11-14 天 | 错误处理 + 重试 + 边界测试 | **MS3 交付** |

### 里程碑 4：优化 + 部署 (2周)

| 时间 | 内容 | 产出 |
|------|------|------|
| 第 1-2 天 | torch.compile + FP16 + VAE slicing | 推理 ~1s |
| 第 3-4 天 | 多 LoRA 叠加推理 | 复合效果 |
| 第 5-6 天 | Prometheus + Grafana 监控 | 可观测 |
| 第 7-8 天 | CI/CD + 集成测试 | CI 流水线 |
| 第 9-10 天 | 性能压测 + 显存优化 | SLO 达标 |
| 第 11-14 天 | 文档 + 部署手册 | **MS4 交付** |

---

## 九、关键决策记录

| 决策 | 选择 | 备选方案 | 理由 |
|------|------|---------|------|
| 后端框架 | Spring Boot 3.2 | FastAPI 单体 | 微服务分工 + Java 生态优势 |
| AI 底座 | SD 1.5 | SDXL / SwinIR | 1.5 推理快 VRAM 友好，LoRA 生态成熟 |
| LoRA 数量 | 3 独立权重 | 单 LoRA 含全部 | 独立权重可热切换/组合 |
| 物理约束 | ControlNet Canny | 直接 PISM loss | ControlNet 是推理解，无需训练 |
| 推理方式 | 异步任务队列 | 同步直调 | 不阻塞 HTTP worker 线程 |
| 存储 | MinIO | 本地文件系统 | S3 兼容，Docker 卷分离 |
| 缓存 | Redis | Caffeine | 多个后端实例共享缓存 |
| 推理加速 | torch.compile | xFormers | PyTorch 官方支持，稳定 |
| 展馆模式 | 后端 API 素材 | 纯前端静态 | 可管理/排序/动态更新 |

---

## 十、验收标准

### P0 硬约束

| 编号 | 指标 | 目标值 | 测量方法 |
|------|------|--------|---------|
| P0-1 | LoRA 推理延迟 | < 1.5s (A100) / < 3s (3060) | 100 次推理取平均 |
| P0-2 | LoRA 热切换时间 | < 3s | 计时 switch_lora() |
| P0-3 | 异步任务成功率 | > 99% | 任务完成数/总数 |
| P0-4 | API 响应时间 (缓存) | < 50ms | JMeter 压测 |
| P0-5 | API 响应时间 (DB) | < 200ms | JMeter 压测 |
| P0-6 | SSE 推送延迟 | < 500ms | 端到端测量 |
| P0-7 | 无显著视觉 artifact | 用户验收 | LoRA 结果目视检查 |
| P0-8 | 三模式均可用 | 所有路由 200 | 集成测试 |

### P1 指标

| 编号 | 指标 | 目标值 |
|------|------|--------|
| P1-1 | wavelength LoRA 散射增益 | > 2.5× (接近理论 3.5×) |
| P1-2 | 展厅素材加载 | < 200ms (含缩略图) |
| P1-3 | GPU 显存占用 | < 8GB |
| P1-4 | 容器启动时间 | < 30s (不含 GPU 模型加载) |

---

## 附：系统边界与扩展性

### 不包含在 v1 范围

- 用户认证/权限系统 (v1 可选 JWT，默认无鉴权)
- 模型 A/B 测试平台
- 批量离线推理调度器
- LoRA 在线训练 (训练是离线 pipeline)
- wafer-inspection 模型直接调用 (仅用 PISM 做伪标签)

### 未来扩展

- 增加 SDXL 底座支持 (画质优先)
- 多 GPU 推理负载均衡
- LoRA 训练界面 (UI 触发训练)
- wafer-inspection 模型直接作为推理后端 (跳过 LoRA)
- 批量产线集成模式 (而非交互式工作台)
