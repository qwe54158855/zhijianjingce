# Wafer Inspection — 晶圆检测后端 + LoRA 推理部署系统

## 项目架构

```
wafer-backend/      Spring Boot 3.2 后端 — API 网关/业务/存储
wafer-lora-service/ FastAPI LoRA 推理微服务 — SD 1.5 + LoRA 热切换
wafer-showcase/     React 前端展厅 — 工作台 + 展厅 + 指标面板
wafer-inspection/   PyTorch 晶圆检测模型 — 多任务学习 + 物理约束
```

## 快速开始

### 前置要求
- Docker & Docker Compose (NVIDIA Container Toolkit for GPU)
- JDK 17+ (本地开发), Node.js 20+ (前端开发)

### 全栈启动
```bash
docker-compose up -d
# 访问 http://localhost:80
```

### 本地开发 (无 Docker)
```bash
# 1. 启动依赖服务
docker-compose up -d postgres redis minio

# 2. 启动后端
cd wafer-backend
./mvnw spring-boot:run

# 3. 启动 LoRA 服务
cd wafer-lora-service
pip install -r requirements.txt
python main.py

# 4. 启动前端
cd wafer-showcase
npm install
VITE_API_BASE=http://localhost:8080 npm run dev
```

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DB_HOST` | localhost | PostgreSQL 主机 |
| `DB_PORT` | 5432 | PostgreSQL 端口 |
| `REDIS_HOST` | localhost | Redis 主机 |
| `REDIS_PORT` | 6379 | Redis 端口 |
| `MINIO_ENDPOINT` | http://localhost:9000 | MinIO 端点 |
| `LORA_SERVICE_URL` | http://localhost:8000 | LoRA 服务地址 |
| `VITE_API_BASE` | http://localhost:8080/api/v1 | 前端 API 基础路径 |

## API 概览

| 端点 | 方法 | 说明 |
|------|------|------|
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/gallery` | GET | 展厅素材列表 (分页) |
| `/api/v1/inference` | POST | 提交推理任务 |
| `/api/v1/inference/{id}` | GET | 查询推理结果 |
| `/api/v1/inference/{id}/stream` | GET | SSE 进度推送 |
| `/api/v1/images/upload` | POST | 上传图片 |
| `/api/v1/metrics/overview` | GET | 指标概览 |
| `/api/v1/actuator/prometheus` | GET | Prometheus 指标 |

## 压测
```bash
# k6
k6 run scripts/benchmark.js

# Locust
locust -f scripts/locustfile.py --host http://localhost:8080
```

## 许可证
MIT
