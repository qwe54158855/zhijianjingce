# Wafer Backend — Spring Boot API Gateway

## Overview
Spring Boot 3.2 backend providing REST API gateway for wafer inspection services. Manages user gallery, inference tasks, metrics, and SSE streaming.

## Tech Stack
- Spring Boot 3.2 + Java 17
- PostgreSQL (main DB)
- Redis (caching, SSE)
- MinIO (image storage)
- Prometheus (metrics)

## Quick Start
```bash
# Local development (requires PostgreSQL, Redis, MinIO running)
./mvnw spring-boot:run
```

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/gallery` | GET | Gallery images (paginated) |
| `/api/v1/inference` | POST | Submit inference task |
| `/api/v1/inference/{id}` | GET | Get inference result |
| `/api/v1/inference/{id}/stream` | GET | SSE progress stream |
| `/api/v1/images/upload` | POST | Upload image |
| `/api/v1/metrics/overview` | GET | Metrics dashboard data |
| `/api/v1/actuator/prometheus` | GET | Prometheus metrics |

## Docker
```dockerfile
docker build -t wafer-backend .
```

## Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `DB_HOST` | localhost | PostgreSQL host |
| `DB_PORT` | 5432 | PostgreSQL port |
| `REDIS_HOST` | localhost | Redis host |
| `MINIO_ENDPOINT` | http://localhost:9000 | MinIO endpoint |
| `LORA_SERVICE_URL` | http://localhost:8000 | LoRA service URL |
