# 部署与运维指南

> Docker Compose 全栈部署，支持本地开发模式

---

## 1. 前置要求

| 组件 | 版本要求 | 用途 |
|------|---------|------|
| Docker & Docker Compose | Docker 24+ / Compose 2+ | 容器编排 |
| NVIDIA Container Toolkit | 可选 | GPU 推理（LoRA 服务） |
| JDK | 17+ | 本地开发 wafer-backend |
| Node.js | 20+ | 本地开发 wafer-showcase |

---

## 2. 服务架构

```
                    ┌──────────────┐
                    │   Nginx :80  │
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐
                    │  wafer-backend│  Spring Boot :8080
                    └──┬───┬───┬───┘
                       │   │   │
              ┌────────┘   │   └──────────┐
              ▼            ▼              ▼
        ┌──────────┐ ┌────────┐ ┌──────────────┐
        │PostgreSQL│ │ Redis  │ │    MinIO     │
        │  :5432   │ │ :6379  │ │   :9000/9001 │
        └──────────┘ └────────┘ └──────────────┘
              │                           │
              ▼                           ▼
        ┌──────────────┐         ┌──────────────┐
        │lora-service  │         │qwen-service  │
        │ FastAPI :8000│         │ FastAPI :8001│
        │  (GPU)       │         │      │       │
        └──────────────┘         └──────▼───────┘
                                    ┌──────────────┐
                                    │llama-server  │
                                    │ llama.cpp    │
                                    │   :8002      │
                                    └──────────────┘
```

## 3. Docker Compose 全栈启动

```bash
# 克隆仓库
git clone https://github.com/your-org/wafer-inspection.git
cd wafer-inspection

# 启动所有服务
docker-compose up -d

# 查看状态
docker-compose ps

# 查看日志
docker-compose logs -f

# 停止所有服务
docker-compose down
```

### 服务端口

| 服务 | 端口 | 访问 |
|------|------|------|
| Nginx（前端） | 80 | http://localhost |
| Spring API | 8080 | http://localhost:8080 |
| LoRA 服务 | 8000 | http://localhost:8000 |
| Qwen 服务 | 8001 | http://localhost:8001 |
| llama.cpp | 8002 | HTTP API |
| PostgreSQL | 5432 | jdbc:postgresql://localhost:5432/wafer |
| Redis | 6379 | redis://localhost:6379 |
| MinIO API | 9000 | http://localhost:9000 |
| MinIO Console | 9001 | http://localhost:9001 |

---

## 4. 环境变量

| 变量 | 默认值 | 服务 | 说明 |
|------|--------|------|------|
| `DB_HOST` | localhost | backend | PostgreSQL 主机 |
| `DB_PORT` | 5432 | backend | PostgreSQL 端口 |
| `REDIS_HOST` | localhost | backend | Redis 主机 |
| `REDIS_PORT` | 6379 | backend | Redis 端口 |
| `MINIO_ENDPOINT` | http://localhost:9000 | backend | MinIO 端点 |
| `LORA_SERVICE_URL` | http://localhost:8000 | backend | LoRA 服务地址 |
| `QWEN_LLAMA_SERVER_URL` | http://llama-server:8002/v1 | qwen | llama.cpp 地址 |
| `VITE_API_BASE` | http://localhost:8080 | showcase | API 基础路径 |

---

## 5. 本地开发模式（无 Docker）

### 启动依赖服务
```bash
docker-compose up -d postgres redis minio
```

### 各服务单独启动

**wafer-backend**（Spring Boot）：
```bash
cd wafer-backend
./mvnw spring-boot:run
```

**wafer-lora-service**（FastAPI + GPU）：
```bash
cd wafer-lora-service
pip install -r requirements.txt
python main.py
```

**wafer-showcase**（React）：
```bash
cd wafer-showcase
npm install
VITE_API_BASE=http://localhost:8080 npm run dev
```

**wafer-qwen-service**（FastAPI + llama.cpp）：
```bash
cd wafer-qwen-service
pip install -r requirements.txt
# 先启动 llama-server（见 docker-compose.yml）
python -m uvicorn main:app --port 8001
```

---

## 6. Qwen 模型模型部署

Qwen2.5-VL-7B GGUF 模型 (~4.6GB) 需单独下载：
```bash
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct-GGUF \
  qwen2.5-vl-7b-instruct-q4_k_m.gguf \
  --local-dir /models/
```

然后在 `docker-compose.yml` 中配置 `source` 路径指向模型文件所在目录。

---

## 7. 运维检查

### 健康检查端点
```bash
# 后端 API
curl http://localhost:8080/api/v1/health

# LoRA 服务
curl http://localhost:8000/api/v1/health

# Qwen 服务
curl http://localhost:8001/health
```

### 日志查看
```bash
# 所有服务
docker-compose logs -f

# 特定服务
docker-compose logs -f spring-api
docker-compose logs -f lora-service
```

### 数据备份
```bash
# PostgreSQL
docker exec -t wafer_postgres pg_dump -U wafer wafer > backup_$(date +%Y%m%d).sql

# MinIO
docker exec wafer_minio mc cp --recursive /data backup/
```

### Prometheus 指标
```bash
curl http://localhost:8080/api/v1/actuator/prometheus
```
