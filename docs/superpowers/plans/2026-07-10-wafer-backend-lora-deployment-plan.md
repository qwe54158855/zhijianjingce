# 晶圆检测后端 + LoRA 推理部署 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建 Spring Boot 后端 + FastAPI LoRA 推理微服务，为 wafer-showcase 前端提供混合模式（静态素材 + 实时推理）数据支持。

**Architecture:** Spring Boot 3.2 (API 网关/业务/存储) + FastAPI (SD 1.5 LoRA 推理) 微服务架构。三层解耦：前端/后端/AI 各自独立部署。异步任务队列 + SSE 进度推送。

**Tech Stack:** Spring Boot 3.2 / JDK 17 / PostgreSQL 15 / Redis 7 / MinIO / FastAPI / PyTorch 2.x / diffusers / SD 1.5 / ControlNet / Docker Compose

## Global Constraints

- Spring Boot 3.2+ / JDK 17+ / Maven wrapper
- Python 3.10+ / PyTorch 2.1+ / CUDA 12.x
- SD 1.5 FP16 底座，VRAM 需求 < 6GB
- 所有推理通过异步任务队列，HTTP 不阻塞
- 三路 LoRA 权重文件 < 40MB 每个
- API 路径前缀统一 `/api/v1/`
- MinIO bucket 名 `wafer-images`
- 前端通过 `VITE_API_BASE` 环境变量配置后端地址

---
## File Structure

```
wafer-backend/                          # Spring Boot (新建)
├── pom.xml
├── Dockerfile
└── src/main/java/com/wafer/
    ├── WaferApplication.java
    ├── config/
    │   ├── WebConfig.java
    │   ├── MinIOConfig.java
    │   ├── RedisConfig.java
    │   └── AsyncConfig.java
    ├── controller/
    │   ├── GalleryController.java
    │   ├── InferenceController.java
    │   └── ImageController.java
    ├── service/
    │   ├── GalleryService.java
    │   ├── InferenceService.java
    │   ├── StorageService.java
    │   └── CacheService.java
    ├── client/
    │   └── LoraInferenceClient.java
    ├── model/
    │   ├── entity/
    │   │   ├── GalleryItem.java
    │   │   └── InferenceTask.java
    │   ├── dto/
    │   │   ├── InferenceRequest.java
    │   │   ├── InferenceResponse.java
    │   │   └── GalleryItemDTO.java
    │   └── enums/
    │       ├── TaskStatus.java
    │       └── InferenceType.java
    ├── repository/
    │   ├── GalleryRepository.java
    │   └── InferenceTaskRepository.java
    └── exception/
        └── GlobalExceptionHandler.java

wafer-lora-service/                     # FastAPI (新建)
├── main.py
├── requirements.txt
├── Dockerfile
├── api/routes/
│   ├── __init__.py
│   ├── infer.py
│   └── lora.py
├── core/
│   ├── __init__.py
│   ├── model_manager.py
│   ├── scheduler.py
│   └── config.py
├── models/
│   ├── __init__.py
│   └── infer_schemas.py
└── loras/
    ├── enhance_lora.safetensors
    ├── wavelength_lora.safetensors
    └── defect_lora.safetensors

wafer-showcase/                         # 前端扩展 (已有项目)
├── src/
│   ├── pages/
│   │   ├── Gallery.jsx
│   │   ├── Workbench.jsx
│   │   └── Metrics.jsx
│   ├── components/
│   │   ├── ImageUploader.jsx
│   │   ├── InferenceResult.jsx
│   │   ├── ModelSelector.jsx
│   │   └── ProgressTracker.jsx
│   ├── hooks/
│   │   ├── useInference.js
│   │   └── useGallery.js
│   └── api/
│       └── index.js

docker-compose.yml                      # 根目录编排
nginx.conf                              # 反向代理
```

---

## 里程碑 1: Spring Boot 后端骨架

### Task 1: Spring Boot 项目脚手架

**Files:**
- Create: `wafer-backend/pom.xml`
- Create: `wafer-backend/src/main/java/com/wafer/WaferApplication.java`
- Create: `wafer-backend/src/main/resources/application.yml`
- Create: `wafer-backend/src/main/resources/application-dev.yml`
- Create: `wafer-backend/src/main/resources/application-prod.yml`

**Interfaces:**
- Produces: 可启动的 Spring Boot 3.2 项目，监听 8080 端口，健康检查 `/api/v1/health` 返回 200

- [ ] **Step 1: Create `wafer-backend/pom.xml`**

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0
         https://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>3.2.5</version>
        <relativePath/>
    </parent>

    <groupId>com.wafer</groupId>
    <artifactId>wafer-backend</artifactId>
    <version>1.0.0</version>
    <name>wafer-backend</name>
    <description>晶圆检测后端服务</description>

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
        <!-- JPA -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-data-jpa</artifactId>
        </dependency>
        <!-- PostgreSQL -->
        <dependency>
            <groupId>org.postgresql</groupId>
            <artifactId>postgresql</artifactId>
            <scope>runtime</scope>
        </dependency>
        <!-- Redis -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-data-redis</artifactId>
        </dependency>
        <!-- OpenFeign -->
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
        <!-- Test -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>
    </dependencies>

    <dependencyManagement>
        <dependencies>
            <dependency>
                <groupId>org.springframework.cloud</groupId>
                <artifactId>spring-cloud-dependencies</artifactId>
                <version>${spring-cloud.version}</version>
                <type>pom</type>
                <scope>import</scope>
            </dependency>
        </dependencies>
    </dependencyManagement>

    <build>
        <plugins>
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
                <configuration>
                    <excludes>
                        <exclude>
                            <groupId>org.projectlombok</groupId>
                            <artifactId>lombok</artifactId>
                        </exclude>
                    </excludes>
                </configuration>
            </plugin>
        </plugins>
    </build>
</project>
```

- [ ] **Step 2: Create `WaferApplication.java`**

```java
package com.wafer;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.cloud.openfeign.EnableFeignClients;
import org.springframework.scheduling.annotation.EnableAsync;

@SpringBootApplication
@EnableAsync
@EnableFeignClients
public class WaferApplication {
    public static void main(String[] args) {
        SpringApplication.run(WaferApplication.class, args);
    }
}
```

- [ ] **Step 3: Create `application.yml`**

```yaml
server:
  port: 8080

spring:
  application:
    name: wafer-backend
  profiles:
    active: dev
  servlet:
    multipart:
      max-file-size: 50MB
      max-request-size: 100MB

  datasource:
    url: jdbc:postgresql://${DB_HOST:localhost}:${DB_PORT:5432}/${DB_NAME:wafer}
    username: ${DB_USER:wafer}
    password: ${DB_PASSWORD:wafer123}
    hikari:
      maximum-pool-size: 20

  jpa:
    hibernate:
      ddl-auto: update
    show-sql: false
    properties:
      hibernate:
        dialect: org.hibernate.dialect.PostgreSQLDialect
        format_sql: true

  data:
    redis:
      host: ${REDIS_HOST:localhost}
      port: ${REDIS_PORT:6379}

minio:
  endpoint: ${MINIO_ENDPOINT:http://localhost:9000}
  access-key: ${MINIO_ACCESS_KEY:minioadmin}
  secret-key: ${MINIO_SECRET_KEY:minioadmin}
  bucket: wafer-images

lora:
  service:
    url: ${LORA_SERVICE_URL:http://localhost:8000}

app:
  async:
    core-pool-size: 4
    max-pool-size: 8
    queue-capacity: 100
```

- [ ] **Step 4: Create health check endpoint in `GalleryController.java` (placeholder)**

```java
package com.wafer.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import java.util.Map;

@RestController
@RequestMapping("/api/v1")
public class GalleryController {

    @GetMapping("/health")
    public Map<String, Object> health() {
        return Map.of("status", "UP", "service", "wafer-backend");
    }
}
```

- [ ] **Step 5: Verify project starts**

```bash
cd wafer-backend
mvn clean compile
# Expected: BUILD SUCCESS
```

- [ ] **Step 6: Commit**

```bash
git init wafer-backend
cd wafer-backend
git add .
git commit -m "feat: init Spring Boot 3.2 project scaffold"
```

---

### Task 2: 配置层 — CORS / MinIO / Redis / Async

**Files:**
- Create: `wafer-backend/src/main/java/com/wafer/config/WebConfig.java`
- Create: `wafer-backend/src/main/java/com/wafer/config/MinIOConfig.java`
- Create: `wafer-backend/src/main/java/com/wafer/config/RedisConfig.java`
- Create: `wafer-backend/src/main/java/com/wafer/config/AsyncConfig.java`

**Interfaces:**
- Produces: 4 个 @Configuration 类，Spring 启动时自动注入

- [ ] **Step 1: Create `WebConfig.java`**

```java
package com.wafer.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.CorsRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebConfig implements WebMvcConfigurer {
    @Override
    public void addCorsMappings(CorsRegistry registry) {
        registry.addMapping("/api/**")
                .allowedOrigins("*")
                .allowedMethods("GET", "POST", "PUT", "DELETE", "OPTIONS")
                .allowedHeaders("*")
                .maxAge(3600);
    }
}
```

- [ ] **Step 2: Create `MinIOConfig.java`**

```java
package com.wafer.config;

import io.minio.MinioClient;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class MinIOConfig {

    @Value("${minio.endpoint}")
    private String endpoint;

    @Value("${minio.access-key}")
    private String accessKey;

    @Value("${minio.secret-key}")
    private String secretKey;

    @Value("${minio.bucket}")
    private String bucket;

    @Bean
    public MinioClient minioClient() {
        return MinioClient.builder()
                .endpoint(endpoint)
                .credentials(accessKey, secretKey)
                .build();
    }

    public String getBucket() {
        return bucket;
    }
}
```

- [ ] **Step 3: Create `RedisConfig.java`**

```java
package com.wafer.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.connection.RedisConnectionFactory;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.serializer.GenericJackson2JsonRedisSerializer;
import org.springframework.data.redis.serializer.StringRedisSerializer;

@Configuration
public class RedisConfig {

    @Bean
    public RedisTemplate<String, Object> redisTemplate(RedisConnectionFactory factory) {
        RedisTemplate<String, Object> template = new RedisTemplate<>();
        template.setConnectionFactory(factory);
        template.setKeySerializer(new StringRedisSerializer());
        template.setValueSerializer(new GenericJackson2JsonRedisSerializer());
        template.setHashKeySerializer(new StringRedisSerializer());
        template.setHashValueSerializer(new GenericJackson2JsonRedisSerializer());
        return template;
    }
}
```

- [ ] **Step 4: Create `AsyncConfig.java`**

```java
package com.wafer.config;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.scheduling.concurrent.ThreadPoolTaskExecutor;

import java.util.concurrent.Executor;

@Configuration
public class AsyncConfig {

    @Value("${app.async.core-pool-size:4}")
    private int corePoolSize;

    @Value("${app.async.max-pool-size:8}")
    private int maxPoolSize;

    @Value("${app.async.queue-capacity:100}")
    private int queueCapacity;

    @Bean(name = "taskExecutor")
    public Executor taskExecutor() {
        ThreadPoolTaskExecutor executor = new ThreadPoolTaskExecutor();
        executor.setCorePoolSize(corePoolSize);
        executor.setMaxPoolSize(maxPoolSize);
        executor.setQueueCapacity(queueCapacity);
        executor.setThreadNamePrefix("wafer-async-");
        executor.setWaitForTasksToCompleteOnShutdown(true);
        executor.initialize();
        return executor;
    }
}
```

- [ ] **Step 5: Verify compilation**

```bash
cd wafer-backend
mvn clean compile
```

- [ ] **Step 6: Commit**

```bash
git add .
git commit -m "feat: add CORS, MinIO, Redis, Async config"
```

---

### Task 3: JPA 实体 + Repository

**Files:**
- Create: `wafer-backend/src/main/java/com/wafer/model/entity/GalleryItem.java`
- Create: `wafer-backend/src/main/java/com/wafer/model/entity/InferenceTask.java`
- Create: `wafer-backend/src/main/java/com/wafer/model/enums/TaskStatus.java`
- Create: `wafer-backend/src/main/java/com/wafer/model/enums/InferenceType.java`
- Create: `wafer-backend/src/main/java/com/wafer/repository/GalleryRepository.java`
- Create: `wafer-backend/src/main/java/com/wafer/repository/InferenceTaskRepository.java`

**Interfaces:**
- Produces: JPA 实体 + Repository，自动建表

- [ ] **Step 1: Create `TaskStatus.java`**

```java
package com.wafer.model.enums;

public enum TaskStatus {
    PENDING,
    RUNNING,
    DONE,
    FAILED
}
```

- [ ] **Step 2: Create `InferenceType.java`**

```java
package com.wafer.model.enums;

public enum InferenceType {
    ENHANCE,
    WAVELENGTH,
    DEFECT
}
```

- [ ] **Step 3: Create `GalleryItem.java`**

```java
package com.wafer.model.entity;

import jakarta.persistence.*;
import lombok.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "gallery_item", indexes = {
    @Index(name = "idx_gallery_category", columnList = "category"),
    @Index(name = "idx_gallery_order", columnList = "displayOrder")
})
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class GalleryItem {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Column(nullable = false, length = 200)
    private String title;

    @Column(columnDefinition = "TEXT")
    private String description;

    @Column(nullable = false, length = 50)
    private String category;

    @Column(columnDefinition = "JSONB DEFAULT '[]'")
    private String tags;

    @Column(name = "original_url", nullable = false, length = 500)
    private String originalUrl;

    @Column(name = "result_url", nullable = false, length = 500)
    private String resultUrl;

    @Column(name = "diff_url", length = 500)
    private String diffUrl;

    @Column(name = "thumbnail_url", length = 500)
    private String thumbnailUrl;

    @Column(columnDefinition = "JSONB DEFAULT '{}'")
    private String metrics;

    @Column(name = "display_order")
    @Builder.Default
    private Integer displayOrder = 0;

    @Column(name = "created_at", updatable = false)
    @Builder.Default
    private LocalDateTime createdAt = LocalDateTime.now();

    @Column(name = "updated_at")
    @Builder.Default
    private LocalDateTime updatedAt = LocalDateTime.now();

    @PreUpdate
    protected void onUpdate() {
        updatedAt = LocalDateTime.now();
    }
}
```

- [ ] **Step 4: Create `InferenceTask.java`**

```java
package com.wafer.model.entity;

import com.wafer.model.enums.InferenceType;
import com.wafer.model.enums.TaskStatus;
import jakarta.persistence.*;
import lombok.*;
import java.time.LocalDateTime;

@Entity
@Table(name = "inference_task", indexes = {
    @Index(name = "idx_task_status", columnList = "status"),
    @Index(name = "idx_task_created", columnList = "createdAt")
})
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class InferenceTask {

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 20)
    private InferenceType type;

    @Enumerated(EnumType.STRING)
    @Column(nullable = false, length = 20)
    private TaskStatus status;

    @Column(name = "input_url", nullable = false, length = 500)
    private String inputUrl;

    @Column(name = "output_url", length = 500)
    private String outputUrl;

    @Column(name = "thumbnail_url", length = 500)
    private String thumbnailUrl;

    @Column(columnDefinition = "JSONB DEFAULT '{}'")
    private String params;

    @Column(name = "error_message", columnDefinition = "TEXT")
    private String errorMessage;

    @Column(name = "duration_ms")
    private Integer durationMs;

    @Column(name = "created_at", updatable = false)
    @Builder.Default
    private LocalDateTime createdAt = LocalDateTime.now();

    @Column(name = "completed_at")
    private LocalDateTime completedAt;
}
```

- [ ] **Step 5: Create `GalleryRepository.java`**

```java
package com.wafer.repository;

import com.wafer.model.entity.GalleryItem;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.Pageable;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface GalleryRepository extends JpaRepository<GalleryItem, Long> {
    Page<GalleryItem> findByCategory(String category, Pageable pageable);
    List<GalleryItem> findAllByOrderByDisplayOrderAsc();
}
```

- [ ] **Step 6: Create `InferenceTaskRepository.java`**

```java
package com.wafer.repository;

import com.wafer.model.entity.InferenceTask;
import com.wafer.model.enums.TaskStatus;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface InferenceTaskRepository extends JpaRepository<InferenceTask, Long> {
    List<InferenceTask> findByStatusOrderByCreatedAtDesc(TaskStatus status);
    long countByStatus(TaskStatus status);
}
```

- [ ] **Step 7: Verify compilation**

```bash
cd wafer-backend
mvn clean compile
```

- [ ] **Step 8: Commit**

```bash
git add .
git commit -m "feat: add JPA entities and repositories"
```

---

### Task 4: 全局异常处理 + DTO

**Files:**
- Create: `wafer-backend/src/main/java/com/wafer/exception/GlobalExceptionHandler.java`
- Create: `wafer-backend/src/main/java/com/wafer/model/dto/GalleryItemDTO.java`
- Create: `wafer-backend/src/main/java/com/wafer/model/dto/InferenceRequest.java`
- Create: `wafer-backend/src/main/java/com/wafer/model/dto/InferenceResponse.java`

**Interfaces:**
- Produces: 统一异常处理 + 请求/响应 DTO 类

- [ ] **Step 1: Create `GlobalExceptionHandler.java`**

```java
package com.wafer.exception;

import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.multipart.MaxUploadSizeExceededException;

import java.time.LocalDateTime;
import java.util.Map;

@Slf4j
@RestControllerAdvice
public class GlobalExceptionHandler {

    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public ResponseEntity<Map<String, Object>> handleMaxUploadSize(MaxUploadSizeExceededException e) {
        return ResponseEntity.status(HttpStatus.PAYLOAD_TOO_LARGE).body(Map.of(
                "error", "文件过大，最大支持 50MB",
                "timestamp", LocalDateTime.now().toString()
        ));
    }

    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<Map<String, Object>> handleBadRequest(IllegalArgumentException e) {
        return ResponseEntity.badRequest().body(Map.of(
                "error", e.getMessage(),
                "timestamp", LocalDateTime.now().toString()
        ));
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<Map<String, Object>> handleGeneral(Exception e) {
        log.error("Unexpected error", e);
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(Map.of(
                "error", "服务器内部错误",
                "timestamp", LocalDateTime.now().toString()
        ));
    }
}
```

- [ ] **Step 2: Create `GalleryItemDTO.java`**

```java
package com.wafer.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class GalleryItemDTO {
    private Long id;
    private String title;
    private String description;
    private String category;
    private String[] tags;
    private String originalUrl;
    private String resultUrl;
    private String diffUrl;
    private String thumbnailUrl;
    private String metrics;
    private Integer displayOrder;
}
```

- [ ] **Step 3: Create `InferenceRequest.java`**

```java
package com.wafer.model.dto;

import com.wafer.model.enums.InferenceType;
import jakarta.validation.constraints.NotNull;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class InferenceRequest {
    @NotNull
    private InferenceType type;

    private String params;  // JSON string: {strength: 0.75, steps: 20}
}
```

- [ ] **Step 4: Create `InferenceResponse.java`**

```java
package com.wafer.model.dto;

import com.wafer.model.enums.InferenceType;
import com.wafer.model.enums.TaskStatus;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class InferenceResponse {
    private Long taskId;
    private InferenceType type;
    private TaskStatus status;
    private String inputUrl;
    private String outputUrl;
    private String thumbnailUrl;
    private Integer durationMs;
    private String errorMessage;
}
```

- [ ] **Step 5: Verify compilation**

```bash
cd wafer-backend
mvn clean compile
```

- [ ] **Step 6: Commit**

```bash
git add .
git commit -m "feat: add exception handler and DTOs"
```

---

### Task 5: StorageService — MinIO 文件上传/下载

**Files:**
- Create: `wafer-backend/src/main/java/com/wafer/service/StorageService.java`

**Interfaces:**
- Produces: `StorageService.upload(MultipartFile, path)` → URL string
- Produces: `StorageService.delete(path)` → void
- Produces: `StorageService.getUrl(path)` → URL string

- [ ] **Step 1: Create `StorageService.java`**

```java
package com.wafer.service;

import io.minio.*;
import io.minio.errors.MinioException;
import io.minio.http.Method;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.InputStream;
import java.util.UUID;
import java.util.concurrent.TimeUnit;

@Slf4j
@Service
@RequiredArgsConstructor
public class StorageService {

    private final MinioClient minioClient;
    private final com.wafer.config.MinIOConfig minioConfig;

    @PostConstruct
    public void init() {
        try {
            String bucket = minioConfig.getBucket();
            boolean exists = minioClient.bucketExists(
                    BucketExistsArgs.builder().bucket(bucket).build());
            if (!exists) {
                minioClient.makeBucket(
                        MakeBucketArgs.builder().bucket(bucket).build());
                log.info("Created MinIO bucket: {}", bucket);
            }
        } catch (Exception e) {
            log.error("Failed to initialize MinIO bucket", e);
        }
    }

    /**
     * 上传文件到 MinIO
     * @param file 上传的文件
     * @param dir  目标目录 (如 "uploads/2026/07/10")
     * @return 文件在 MinIO 中的存储路径
     */
    public String upload(MultipartFile file, String dir) {
        try {
            String filename = UUID.randomUUID().toString() + "_" + file.getOriginalFilename();
            String objectName = dir + "/" + filename;

            minioClient.putObject(
                    PutObjectArgs.builder()
                            .bucket(minioConfig.getBucket())
                            .object(objectName)
                            .stream(file.getInputStream(), file.getSize(), -1)
                            .contentType(file.getContentType())
                            .build());

            log.info("Uploaded: {}", objectName);
            return objectName;
        } catch (Exception e) {
            throw new RuntimeException("MinIO upload failed", e);
        }
    }

    /**
     * 从 MinIO 获取文件的可访问 URL
     */
    public String getUrl(String objectName) {
        try {
            return minioClient.getPresignedObjectUrl(
                    GetPresignedObjectUrlArgs.builder()
                            .bucket(minioConfig.getBucket())
                            .object(objectName)
                            .method(Method.GET)
                            .expiry(24, TimeUnit.HOURS)
                            .build());
        } catch (Exception e) {
            log.warn("Failed to get presigned URL, fallback to path: {}", objectName);
            return "/api/v1/images/" + objectName;
        }
    }

    /**
     * 从 MinIO 获取文件流
     */
    public InputStream download(String objectName) {
        try {
            return minioClient.getObject(
                    GetObjectArgs.builder()
                            .bucket(minioConfig.getBucket())
                            .object(objectName)
                            .build());
        } catch (Exception e) {
            throw new RuntimeException("MinIO download failed: " + objectName, e);
        }
    }

    /**
     * 删除文件
     */
    public void delete(String objectName) {
        try {
            minioClient.removeObject(
                    RemoveObjectArgs.builder()
                            .bucket(minioConfig.getBucket())
                            .object(objectName)
                            .build());
            log.info("Deleted: {}", objectName);
        } catch (Exception e) {
            log.error("Failed to delete: {}", objectName, e);
        }
    }

    /**
     * 生成按日期分类的存储路径
     */
    public static String generatePath(String prefix) {
        return prefix + "/" + java.time.LocalDate.now().toString().replace("-", "/");
    }
}
```

- [ ] **Step 2: Verify compilation**

```bash
cd wafer-backend
mvn clean compile
```

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "feat: add MinIO storage service (upload/download/delete)"
```

---

### Task 6: CacheService — Redis 缓存

**Files:**
- Create: `wafer-backend/src/main/java/com/wafer/service/CacheService.java`

**Interfaces:**
- Produces: `CacheService.set(key, value, ttlSeconds)` → void
- Produces: `CacheService.get(key)` → Object
- Produces: `CacheService.delete(key)` → void

- [ ] **Step 1: Create `CacheService.java`**

```java
package com.wafer.service;

import lombok.RequiredArgsConstructor;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.stereotype.Service;

import java.util.concurrent.TimeUnit;

@Service
@RequiredArgsConstructor
public class CacheService {

    private final RedisTemplate<String, Object> redisTemplate;

    public void set(String key, Object value, long ttlSeconds) {
        redisTemplate.opsForValue().set(key, value, ttlSeconds, TimeUnit.SECONDS);
    }

    public Object get(String key) {
        return redisTemplate.opsForValue().get(key);
    }

    public void delete(String key) {
        redisTemplate.delete(key);
    }

    public boolean hasKey(String key) {
        return Boolean.TRUE.equals(redisTemplate.hasKey(key));
    }

    // 推理结果缓存专用
    public void cacheTaskResult(Long taskId, Object result) {
        set("task:result:" + taskId, result, 3600);  // 1 hour
    }

    public Object getTaskResult(Long taskId) {
        return get("task:result:" + taskId);
    }
}
```

- [ ] **Step 2: Verify compilation**

```bash
cd wafer-backend
mvn clean compile
```

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "feat: add Redis cache service"
```

---

### Task 7: GalleryService + GalleryController — 展厅素材 API

**Files:**
- Create: `wafer-backend/src/main/java/com/wafer/service/GalleryService.java`
- Modify: `wafer-backend/src/main/java/com/wafer/controller/GalleryController.java` (添加上下文)

**Interfaces:**
- Produces: GET `/api/v1/gallery` — 分页素材列表
- Produces: GET `/api/v1/gallery/{id}` — 素材详情
- Produces: GET `/api/v1/gallery/stats` — 展厅统计

- [ ] **Step 1: Create `GalleryService.java`**

```java
package com.wafer.service;

import com.wafer.model.dto.GalleryItemDTO;
import com.wafer.model.entity.GalleryItem;
import com.wafer.repository.GalleryRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.data.domain.PageRequest;
import org.springframework.data.domain.Sort;
import org.springframework.stereotype.Service;

import java.util.Map;
import java.util.Optional;

@Service
@RequiredArgsConstructor
public class GalleryService {

    private final GalleryRepository galleryRepository;

    public Page<GalleryItemDTO> getGallery(String category, int page, int size) {
        PageRequest pr = PageRequest.of(page, size, Sort.by(Sort.Direction.ASC, "displayOrder"));
        Page<GalleryItem> items;

        if (category != null && !category.isEmpty()) {
            items = galleryRepository.findByCategory(category, pr);
        } else {
            items = galleryRepository.findAll(pr);
        }

        return items.map(this::toDTO);
    }

    public Optional<GalleryItemDTO> getById(Long id) {
        return galleryRepository.findById(id).map(this::toDTO);
    }

    public Map<String, Object> getStats() {
        long total = galleryRepository.count();
        return Map.of("totalItems", total);
    }

    public GalleryItemDTO create(GalleryItem item) {
        GalleryItem saved = galleryRepository.save(item);
        return toDTO(saved);
    }

    private GalleryItemDTO toDTO(GalleryItem item) {
        return GalleryItemDTO.builder()
                .id(item.getId())
                .title(item.getTitle())
                .description(item.getDescription())
                .category(item.getCategory())
                .tags(item.getTags() != null ? item.getTags().replaceAll("[\\[\\]\" ]", "").split(",") : new String[0])
                .originalUrl(item.getOriginalUrl())
                .resultUrl(item.getResultUrl())
                .diffUrl(item.getDiffUrl())
                .thumbnailUrl(item.getThumbnailUrl())
                .metrics(item.getMetrics())
                .displayOrder(item.getDisplayOrder())
                .build();
    }
}
```

- [ ] **Step 2: Update `GalleryController.java`**

```java
package com.wafer.controller;

import com.wafer.model.dto.GalleryItemDTO;
import com.wafer.service.GalleryService;
import lombok.RequiredArgsConstructor;
import org.springframework.data.domain.Page;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/v1/gallery")
@RequiredArgsConstructor
public class GalleryController {

    private final GalleryService galleryService;

    @GetMapping
    public ResponseEntity<Page<GalleryItemDTO>> getGallery(
            @RequestParam(required = false) String category,
            @RequestParam(defaultValue = "0") int page,
            @RequestParam(defaultValue = "20") int size) {
        return ResponseEntity.ok(galleryService.getGallery(category, page, size));
    }

    @GetMapping("/{id}")
    public ResponseEntity<GalleryItemDTO> getById(@PathVariable Long id) {
        return galleryService.getById(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping("/stats")
    public ResponseEntity<Map<String, Object>> getStats() {
        return ResponseEntity.ok(galleryService.getStats());
    }
}
```

- [ ] **Step 3: Accept old health endpoint into `GalleryController`**

保留之前的 `/api/v1/health` 端点不动。

- [ ] **Step 4: Verify compilation**

```bash
cd wafer-backend
mvn clean compile
```

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "feat: add gallery service and controller"
```

---

### Task 8: ImageController — 图片上传/查看

**Files:**
- Create: `wafer-backend/src/main/java/com/wafer/controller/ImageController.java`

**Interfaces:**
- Produces: POST `/api/v1/images/upload` — 上传图片，返回 URL
- Produces: GET `/api/v1/images/{filename}` — 获取图片流

- [ ] **Step 1: Create `ImageController.java`**

```java
package com.wafer.controller;

import com.wafer.service.StorageService;
import jakarta.servlet.http.HttpServletResponse;
import lombok.RequiredArgsConstructor;
import org.apache.tomcat.util.http.fileupload.IOUtils;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.InputStream;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/images")
@RequiredArgsConstructor
public class ImageController {

    private final StorageService storageService;

    @PostMapping("/upload")
    public ResponseEntity<Map<String, String>> upload(
            @RequestParam("file") MultipartFile file,
            @RequestParam(defaultValue = "uploads") String dir) {
        if (file.isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("error", "文件为空"));
        }
        String path = storageService.upload(file, StorageService.generatePath(dir));
        String url = storageService.getUrl(path);
        return ResponseEntity.ok(Map.of("path", path, "url", url));
    }

    @GetMapping("/**")
    public void getImage(HttpServletResponse response) {
        // Extract path from request: /api/v1/images/uploads/2026/07/10/xxx.jpg
        String path = ""; // In real impl, extract from request
        // Proxying MinIO images; for simplicity, return presigned URL redirect
        response.setStatus(302);
        response.setHeader("Location", storageService.getUrl(path));
    }
}
```

- [ ] **Step 2: Verify compilation**

```bash
cd wafer-backend
mvn clean compile
```

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "feat: add image upload controller"
```

---

### Task 9: Docker Compose 基础编排

**Files:**
- Create: `docker-compose.yml`
- Create: `nginx.conf`
- Create: `wafer-backend/Dockerfile`

**Interfaces:**
- Produces: `docker-compose up` 启动 PostgreSQL + Redis + MinIO + Spring Boot

- [ ] **Step 1: Create `wafer-backend/Dockerfile`**

```dockerfile
FROM maven:3.9-eclipse-temurin-17 AS builder
WORKDIR /app
COPY pom.xml .
RUN mvn dependency:go-offline -B
COPY src ./src
RUN mvn package -DskipTests -B

FROM eclipse-temurin:17-jre
WORKDIR /app
COPY --from=builder /app/target/*.jar app.jar
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar", "--spring.profiles.active=prod"]
```

- [ ] **Step 2: Create `docker-compose.yml`**

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: wafer
      POSTGRES_USER: wafer
      POSTGRES_PASSWORD: wafer123
    ports: ["5432:5432"]
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U wafer"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    ports: ["9000:9000", "9001:9001"]
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio_data:/data
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/minio/health/live"]
      interval: 10s
      timeout: 5s
      retries: 5

  spring-api:
    build: ./wafer-backend
    ports: ["8080:8080"]
    environment:
      SPRING_PROFILES_ACTIVE: prod
      DB_HOST: postgres
      DB_PORT: 5432
      DB_NAME: wafer
      DB_USER: wafer
      DB_PASSWORD: wafer123
      REDIS_HOST: redis
      REDIS_PORT: 6379
      MINIO_ENDPOINT: http://minio:9000
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
      LORA_SERVICE_URL: http://lora-service:8000
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
      minio:
        condition: service_healthy

  nginx:
    image: nginx:alpine
    ports: ["80:80"]
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - spring-api

volumes:
  pgdata:
  minio_data:
```

- [ ] **Step 3: Create `nginx.conf`**

```nginx
events {
    worker_connections 1024;
}

http {
    include /etc/nginx/mime.types;
    default_type application/octet-stream;

    upstream spring_api {
        server spring-api:8080;
    }

    server {
        listen 80;

        location /api/ {
            proxy_pass http://spring_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

            # SSE support
            proxy_buffering off;
            proxy_cache off;
            chunked_transfer_encoding on;
        }

        location /health {
            proxy_pass http://spring_api/api/v1/health;
        }
    }
}
```

- [ ] **Step 4: Verify docker-compose starts**

```bash
docker-compose up -d postgres redis minio
docker-compose logs spring-api
# Expected: Spring Boot started, port 8080
```

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "feat: add Docker Compose orchestration (PG/Redis/MinIO/Spring)"
```

---

## 里程碑 2: LoRA 训练 + FastAPI 推理服务

### Task 10: FastAPI 项目脚手架 + GPU 容器化

**Files:**
- Create: `wafer-lora-service/main.py`
- Create: `wafer-lora-service/requirements.txt`
- Create: `wafer-lora-service/Dockerfile`
- Create: `wafer-lora-service/core/__init__.py`
- Create: `wafer-lora-service/core/config.py`
- Create: `wafer-lora-service/models/__init__.py`
- Create: `wafer-lora-service/models/infer_schemas.py`

**Interfaces:**
- Produces: `docker-compose up` 启动 FastAPI，`/api/v1/health` 返回 GPU 状态

- [ ] **Step 1: Create `requirements.txt`**

```
fastapi==0.110.0
uvicorn[standard]==0.27.0
torch>=2.1.0
torchvision>=0.16.0
diffusers[torch]==0.27.0
transformers>=4.38.0
accelerate>=0.27.0
pillow>=10.0.0
numpy>=1.24.0
pyyaml>=6.0
pydantic>=2.0.0
python-multipart>=0.0.6
```

- [ ] **Step 2: Create `core/config.py`**

```python
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Service
    service_name: str = "wafer-lora-service"
    host: str = "0.0.0.0"
    port: int = 8000

    # Model
    model_id: str = "runwayml/stable-diffusion-v1-5"
    torch_dtype: str = "float16"
    device: str = "cuda"
    offload: bool = True
    vae_slicing: bool = True

    # LoRA
    lora_dir: Path = Path(__file__).parent.parent / "loras"
    num_inference_steps: int = 20

    class Config:
        env_prefix = "LORA_"


settings = Settings()
```

- [ ] **Step 3: Create `models/infer_schemas.py`**

```python
from pydantic import BaseModel, Field
from typing import Optional


class InferRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded input image")
    mode: str = Field(..., pattern=r"^(enhance|wavelength|defect)$")
    strength: float = Field(default=0.75, ge=0.1, le=1.0)
    prompt: Optional[str] = None
    control_image_base64: Optional[str] = None  # ControlNet conditioning


class InferResponse(BaseModel):
    result_base64: str = Field(description="Base64 encoded result image")
    duration_ms: int = Field(description="Inference duration in ms")


class LoraInfo(BaseModel):
    name: str
    active: bool
    weight: float


class HealthResponse(BaseModel):
    status: str
    device: str
    gpu_available: bool
    active_loras: list[LoraInfo]
```

- [ ] **Step 4: Create `main.py`**

```python
import logging
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from models.infer_schemas import HealthResponse, LoraInfo

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model manager (lazy init)
model_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init model manager, load default LoRA"""
    global model_manager
    from core.model_manager import ModelManager
    logger.info("Initializing ModelManager...")
    model_manager = ModelManager()
    model_manager.load_lora("enhance")
    logger.info(f"ModelManager ready. GPU: {torch.cuda.is_available()}")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Wafer LoRA Inference Service",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", response_model=HealthResponse)
async def health():
    global model_manager
    active = []
    if model_manager:
        active = [
            LoraInfo(name=name, active=True, weight=w)
            for name, w in model_manager.active_loras.items()
        ]
    return HealthResponse(
        status="UP",
        device=settings.device,
        gpu_available=torch.cuda.is_available(),
        active_loras=active,
    )


# Import and register routers
from api.routes import infer, lora
app.include_router(infer.router, prefix="/api/v1/lora")
app.include_router(lora.router, prefix="/api/v1/lora")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
```

- [ ] **Step 5: Create `Dockerfile`**

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 6: Verify starts**

```bash
cd wafer-lora-service
pip install -r requirements.txt
python main.py
# Expected: Uvicorn running on http://0.0.0.0:8000
```

- [ ] **Step 7: Commit**

```bash
git add .
git commit -m "feat: init FastAPI project scaffold with GPU Docker"
```

---

### Task 11: ModelManager — SD 1.5 底座 + LoRA 热切换

**Files:**
- Create: `wafer-lora-service/core/model_manager.py`
- Create: `wafer-lora-service/core/__init__.py`
- Create: `wafer-lora-service/api/__init__.py`
- Create: `wafer-lora-service/api/routes/__init__.py`
- Create: `wafer-lora-service/api/routes/lora.py`

**Interfaces:**
- Produces: `ModelManager(base_model_id, device)` 初始化 SD 1.5
- Produces: `model_manager.load_lora(name)` 加载 LoRA 权重
- Produces: `model_manager.switch_lora(name)` 热切换
- Produces: `model_manager.infer(image, prompt, control_image, strength)` → PIL Image
- Produces: POST `/api/v1/lora/switch` 切换 LoRA
- Produces: GET `/api/v1/lora/active` 查询活跃 LoRA

- [ ] **Step 1: Create `core/model_manager.py`**

```python
import logging
from pathlib import Path
from typing import Optional

import torch
from diffusers import StableDiffusionImg2ImgPipeline, ControlNetModel
from diffusers.utils import load_image
from PIL import Image

from core.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """SD model lifecycle + LoRA hot-swap manager."""

    def __init__(self):
        self.device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if settings.torch_dtype == "float16" else torch.float32

        logger.info(f"Loading base model: {settings.model_id} ({dtype})")
        self.base = StableDiffusionImg2ImgPipeline.from_pretrained(
            settings.model_id,
            torch_dtype=dtype,
            variant="fp16" if settings.torch_dtype == "float16" else None,
            safety_checker=None,  # Disable NSFW filter for wafer images
        )
        self.base.to(self.device)

        if settings.offload:
            self.base.enable_model_cpu_offload()
        if settings.vae_slicing:
            self.base.enable_vae_slicing()

        # LoRA registry: name -> safetensors path
        self.lora_dir = Path(settings.lora_dir)
        self.lora_registry = {
            "enhance":    str(self.lora_dir / "enhance_lora.safetensors"),
            "wavelength": str(self.lora_dir / "wavelength_lora.safetensors"),
            "defect":     str(self.lora_dir / "defect_lora.safetensors"),
        }
        self.active_loras: dict[str, float] = {}

        # ControlNet (lazy load)
        self._controlnet: Optional[ControlNetModel] = None

        logger.info(f"Model loaded on {self.device}")

    @property
    def controlnet(self) -> ControlNetModel:
        if self._controlnet is None:
            logger.info("Loading ControlNet (Canny)...")
            self._controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/sd-controlnet-canny",
                torch_dtype=torch.float16,
            ).to(self.device)
        return self._controlnet

    def load_lora(self, name: str, weight: float = 1.0):
        """Load a LoRA adapter. Multiple LoRAs can be active simultaneously."""
        path = self.lora_registry.get(name)
        if not path:
            raise ValueError(f"Unknown LoRA: {name}. Available: {list(self.lora_registry.keys())}")
        if not Path(path).exists():
            raise FileNotFoundError(f"LoRA weight not found: {path}")

        self.base.load_lora_weights(path, adapter_name=name)
        self.active_loras[name] = weight
        logger.info(f"Loaded LoRA: {name} (weight={weight})")

    def switch_lora(self, name: str, weight: float = 1.0):
        """Switch to a single LoRA (remove all others)."""
        if self.active_loras:
            self.base.delete_adapters(list(self.active_loras.keys()))
            self.active_loras = {}
        self.load_lora(name, weight)

    @torch.inference_mode()
    def infer(
        self,
        image: Image.Image,
        prompt: str,
        control_image: Optional[Image.Image] = None,
        strength: float = 0.75,
    ) -> Image.Image:
        """Run SD img2img inference with active LoRA adapters."""
        kwargs = {
            "image": image,
            "prompt": prompt,
            "strength": strength,
            "num_inference_steps": settings.num_inference_steps,
            "output_type": "pil",
        }

        if self.active_loras:
            kwargs["cross_attention_kwargs"] = {
                "scale": list(self.active_loras.values())
            }

        if control_image is not None:
            kwargs["control_image"] = control_image
            kwargs["controlnet_conditioning_scale"] = 0.8

        result = self.base(**kwargs)
        return result.images[0]
```

- [ ] **Step 2: Create `api/routes/__init__.py`**

```python
# Empty __init__ for routes package
```

- [ ] **Step 3: Create `api/routes/lora.py`**

```python
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from models.infer_schemas import LoraInfo

logger = logging.getLogger(__name__)
router = APIRouter(tags=["LoRA Management"])


class SwitchLoraRequest(BaseModel):
    name: str
    weight: float = 1.0


@router.post("/switch")
async def switch_lora(req: SwitchLoraRequest):
    """Hot-switch to a specific LoRA adapter."""
    from main import model_manager
    try:
        model_manager.switch_lora(req.name, req.weight)
        return {"status": "ok", "active": req.name, "weight": req.weight}
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/active", response_model=list[LoraInfo])
async def get_active_loras():
    """List currently active LoRA adapters."""
    from main import model_manager
    return [
        LoraInfo(name=name, active=True, weight=w)
        for name, w in model_manager.active_loras.items()
    ]
```

- [ ] **Step 4: Verify import**

```bash
cd wafer-lora-service
python -c "from core.model_manager import ModelManager; print('OK')"
# Expected: OK
```

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "feat: add ModelManager with SD 1.5 + LoRA hot-swap"
```

---

### Task 12: 推理路由 — LOAD (infer.py)

**Files:**
- Create: `wafer-lora-service/api/routes/infer.py`

**Interfaces:**
- Produces: POST `/api/v1/lora/infer` — 执行推理

- [ ] **Step 1: Create `api/routes/infer.py`**

```python
import base64
import io
import logging
import time

from fastapi import APIRouter, HTTPException
from PIL import Image, ImageFilter, ImageOps

from models.infer_schemas import InferRequest, InferResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Inference"])

# Default prompts for each mode
DEFAULT_PROMPTS = {
    "enhance":    "wafer inspection dark field to bright field enhancement, "
                  "high contrast defect visibility, sharp edges, clean background",
    "wavelength": "deep UV 193nm wafer inspection, enhanced Rayleigh scattering, "
                  "high resolution defect detection, sharp nanostructures",
    "defect":     "wafer defect, realistic scratch and particle contamination, "
                  "semiconductor manufacturing defect, high detail",
}


def decode_base64_image(b64: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    try:
        image_bytes = base64.b64decode(b64)
        return Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


def encode_image_to_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64 string (JPEG, quality 95)."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_control_image(image: Image.Image, mode: str) -> Image.Image:
    """Generate ControlNet conditioning image based on mode."""
    if mode == "defect":
        # For defect generation, create an edge map
        return image.filter(ImageFilter.FIND_EDGES)
    else:
        # For enhance/wavelength, use Canny-like edge detection
        gray = image.convert("L")
        edges = gray.filter(ImageFilter.FIND_EDGES)
        return edges.convert("RGB")


@router.post("/infer", response_model=InferResponse)
async def infer(req: InferRequest):
    """Run LoRA inference on input image."""
    from main import model_manager

    t0 = time.time()

    # Decode input image
    image = decode_base64_image(req.image_base64)
    image = image.resize((512, 512), Image.LANCZOS)

    # Switch to requested LoRA if needed
    if req.mode not in model_manager.active_loras:
        try:
            model_manager.switch_lora(req.mode)
        except (ValueError, FileNotFoundError) as e:
            raise HTTPException(status_code=400, detail=str(e))

    # Prepare prompt
    prompt = req.prompt or DEFAULT_PROMPTS.get(req.mode, "wafer inspection")

    # Prepare ControlNet condition
    control_image = None
    if req.control_image_base64:
        control_image = decode_base64_image(req.control_image_base64)
    elif req.mode in ("enhance", "wavelength"):
        control_image = generate_control_image(image, req.mode)

    # Run inference
    try:
        result = model_manager.infer(
            image=image,
            prompt=prompt,
            control_image=control_image,
            strength=req.strength,
        )
    except Exception as e:
        logger.error(f"Inference failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # Encode result
    result_b64 = encode_image_to_base64(result)
    duration_ms = int((time.time() - t0) * 1000)

    return InferResponse(result_base64=result_b64, duration_ms=duration_ms)
```

- [ ] **Step 2: Write test for `infer.py`**

```python
# Place this as a test file at a location of choice
# wafer-lora-service/tests/test_infer.py
"""
Test inference route with mock model.
Run: pytest tests/test_infer.py -v
"""
import pytest
from PIL import Image
from api.routes.infer import generate_control_image, encode_image_to_base64, decode_base64_image


def test_image_roundtrip():
    img = Image.new("RGB", (64, 64), color="gray")
    b64 = encode_image_to_base64(img)
    decoded = decode_base64_image(b64)
    assert decoded.size == (64, 64)


def test_control_image_enhance():
    img = Image.new("RGB", (64, 64), color="white")
    control = generate_control_image(img, "enhance")
    assert control.size == (64, 64)
```

```bash
cd wafer-lora-service
pytest tests/ -v
# Expected: tests pass
```

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "feat: add inference route with LoRA mode dispatch"
```

---

### Task 13: PISM 伪标签生成管线

**Files:**
- Create: `wafer-lora-service/training/generate_pseudo_labels.py`

**Interfaces:**
- Produces: 调用 wafer-inspection 的 PISM 模块生成 266nm→193nm 伪标签对
- Produces: 输出到 `wafer-lora-service/training/data/wavelength_pairs/`

- [ ] **Step 1: Create `generate_pseudo_labels.py`**

```python
"""
PISM pseudo-label generation pipeline.

Uses wafer-inspection PISM module to generate 266nm→193nm training pairs.
Run after wafer-inspection is installed as a package.

Usage:
    python generate_pseudo_labels.py \
        --input_dir /path/to/266nm/images \
        --output_dir training/data/wavelength_pairs \
        --num_pairs 1000
"""
import argparse
import logging
import sys
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_wafer_inspection():
    """Check wafer-inspection is importable."""
    try:
        from wafer_inspection.models.physics_scattering import ScatteringPhysics
        from wafer_inspection.models.wafer_multitask import (
            RepViTEncoder, EnhanceDecoder
        )
        return True
    except ImportError:
        logger.error(
            "wafer-inspection not installed. "
            "Run: cd /path/to/wafer-inspection && pip install -e ."
        )
        return False


@torch.inference_mode()
def generate_pairs(input_dir: Path, output_dir: Path, num_pairs: int):
    """Generate 266nm → pseudo-193nm image pairs using PISM."""
    from wafer_inspection.models.physics_scattering import ScatteringPhysics
    from wafer_inspection.models.wafer_multitask import (
        RepViTEncoder, EnhanceDecoder
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "266nm").mkdir(exist_ok=True)
    (output_dir / "193nm").mkdir(exist_ok=True)

    # Load models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    encoder = RepViTEncoder(in_channels=1).to(device).eval()
    pism = ScatteringPhysics(
        channels_per_stage=[56, 112, 224, 448],
        lambda_in=266.0,
        lambda_out=193.0,
    ).to(device).eval()
    decoder = EnhanceDecoder().to(device).eval()

    # Find input images
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif")
    images = []
    for ext in extensions:
        images.extend(input_dir.glob(ext))
        images.extend(input_dir.glob(ext.upper()))

    if not images:
        logger.warning(f"No images found in {input_dir}")
        return

    logger.info(f"Found {len(images)} source images, generating {num_pairs} pairs")
    to_tensor = transforms.ToTensor()

    count = 0
    while count < num_pairs:
        for img_path in images:
            if count >= num_pairs:
                break

            # Load and preprocess
            pil_img = Image.open(img_path).convert("L")
            pil_img = pil_img.resize((512, 512), Image.LANCZOS)
            tensor = to_tensor(pil_img).unsqueeze(0).to(device)  # [1,1,512,512]

            # Forward through encoder → PISM → decoder
            feats = encoder(tensor)
            feats_193, pism_diag = pism(feats)
            enhanced = decoder(feats_193)  # [1,1,512,512]

            # Save pair
            stem = f"pair_{count:05d}"
            Image.fromarray(
                (tensor[0, 0].cpu().numpy() * 255).astype("uint8")
            ).save(str(output_dir / "266nm" / f"{stem}_266.jpg"))

            Image.fromarray(
                (enhanced[0, 0].cpu().numpy() * 255).astype("uint8")
            ).save(str(output_dir / "193nm" / f"{stem}_193.jpg"))

            count += 1
            if count % 100 == 0:
                logger.info(f"Generated {count}/{num_pairs} pairs")

    logger.info(f"Done: {count} pairs saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate PISM pseudo-labels")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with 266nm wafer images")
    parser.add_argument("--output_dir", type=str, default="training/data/wavelength_pairs",
                        help="Output directory for pairs")
    parser.add_argument("--num_pairs", type=int, default=1000,
                        help="Number of pairs to generate")
    args = parser.parse_args()

    if not ensure_wafer_inspection():
        sys.exit(1)

    generate_pairs(
        Path(args.input_dir),
        Path(args.output_dir),
        args.num_pairs,
    )
```

- [ ] **Step 2: Verify import (without running full pipeline)**

```bash
python -c "import sys; sys.path.insert(0, '.'); from training.generate_pseudo_labels import ensure_wafer_inspection; print('Import OK')"
```

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "feat: add PISM pseudo-label generation pipeline"
```

---

### Task 14: LoRA 训练脚本

**Files:**
- Create: `wafer-lora-service/training/train_lora.py`
- Create: `wafer-lora-service/training/config_lora.yaml`

**Interfaces:**
- Produces: `python train_lora.py --config config_lora.yaml` 训练 LoRA，输出 `.safetensors`

- [ ] **Step 1: Create `training/config_lora.yaml`**

```yaml
# LoRA training configuration for wafer LoRA adapters
base_model: runwayml/stable-diffusion-v1-5
resolution: 512
lora_rank: 64
lora_alpha: 32
target_modules:
  - to_q
  - to_k
  - to_v
  - to_out.0
  - proj_in
  - proj_out
  - ff.net.0.proj
  - ff.net.2

training:
  batch_size: 4
  learning_rate: 1e-4
  lr_scheduler: cosine
  num_epochs: 50
  save_steps: 200
  max_grad_norm: 1.0
  mixed_precision: fp16
  gradient_accumulation_steps: 2
  use_8bit_adam: true
  dataloader_num_workers: 4
  flip_p: 0.5
  random_crop: true
  random_brightness: 0.1

data:
  image_column: image
  caption_column: text
  resolution: 512
  center_crop: true
  random_flip: true
```

- [ ] **Step 2: Create `training/train_lora.py`**

```python
"""
LoRA training script for wafer inspection image translation.

Extends diffusers' train_dreambooth_lora.py with wafer-specific:
- PISM physical consistency loss (optional)
- Real calibration pair upweighting
- Multi-adapter support

Usage:
    python train_lora.py \
        --config config_lora.yaml \
        --instance_data_dir data/wavelength_pairs \
        --output_dir ../loras \
        --adapter_name wavelength_lora

Requires: pip install diffusers[training] peft bitsandbytes
"""
import argparse
import logging
import math
import os
from pathlib import Path

import torch
import yaml
from diffusers import StableDiffusionPipeline
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

check_min_version("0.27.0")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WaferPairDataset(Dataset):
    """Wafer image pair dataset (266nm → 193nm)."""

    def __init__(self, data_dir: str, resolution: int = 512, flip_p: float = 0.5):
        self.data_dir = Path(data_dir)
        self.resolution = resolution
        self.flip_p = flip_p

        # Expect structure: data_dir/{266nm,193nm}/*.jpg
        self.images_266 = sorted((self.data_dir / "266nm").glob("*.jpg"))
        self.images_193 = sorted((self.data_dir / "193nm").glob("*.jpg"))

        assert len(self.images_266) == len(self.images_193), \
            f"Mismatch: {len(self.images_266)} vs {len(self.images_193)}"

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.RandomHorizontalFlip(p=flip_p),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.images_266)

    def __getitem__(self, idx):
        img_266 = Image.open(self.images_266[idx]).convert("RGB")
        img_193 = Image.open(self.images_193[idx]).convert("RGB")

        # Apply same random seed for paired transform
        seed = torch.randint(0, 2**30, (1,)).item()
        torch.manual_seed(seed)
        pixel_266 = self.transform(img_266)
        torch.manual_seed(seed)
        pixel_193 = self.transform(img_193)

        return {
            "pixel_values": pixel_266,
            "target_values": pixel_193,
        }


def train_lora(config_path: str, data_dir: str, output_dir: str, adapter_name: str):
    """Main training loop."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        config["base_model"],
        torch_dtype=torch.float16,
        variant="fp16",
        safety_checker=None,
    )
    pipe.to(device)
    vae = pipe.vae
    unet = pipe.unet
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Apply LoRA to UNet
    lora_config = LoraConfig(
        r=config.get("lora_rank", 64),
        lora_alpha=config.get("lora_alpha", 32),
        target_modules=config.get("target_modules"),
        lora_dropout=0.0,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.train()

    # Dataset
    dataset = WaferPairDataset(
        data_dir=data_dir,
        resolution=config["data"]["resolution"],
        flip_p=config.get("training", {}).get("flip_p", 0.5),
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["dataloader_num_workers"],
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config["training"]["learning_rate"],
    )

    # LR scheduler
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["num_epochs"],
    )

    # Training loop
    logger.info(f"Starting LoRA training for {adapter_name}")
    global_step = 0
    for epoch in range(config["training"]["num_epochs"]):
        for step, batch in enumerate(dataloader):
            pixel_266 = batch["pixel_values"].to(device)
            target_193 = batch["target_values"].to(device)

            # Encode target to latent space
            with torch.no_grad():
                latents = vae.encode(target_193).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

            # Sample noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, pipe.scheduler.config.num_train_timesteps,
                (latents.shape[0],), device=device
            )
            noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)

            # Encode prompt (empty prompt for unconditional)
            with torch.no_grad():
                encoder_hidden_states = text_encoder(
                    tokenizer(
                        [""] * latents.shape[0],
                        padding="max_length",
                        max_length=tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to(device)
                )[0]

            # Predict noise
            noise_pred = unet(
                noisy_latents, timesteps, encoder_hidden_states
            ).sample

            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % config["training"]["save_steps"] == 0:
                logger.info(f"Step {global_step}, Loss: {loss.item():.6f}")

        lr_scheduler.step()
        logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']} complete, Loss: {loss.item():.6f}")

    # Save LoRA weights
    output_path = Path(output_dir) / f"{adapter_name}.safetensors"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unet.save_pretrained(str(output_path.parent), safe_serialization=True)
    logger.info(f"LoRA saved to {output_path}")
    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train wafer LoRA adapters")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./loras")
    parser.add_argument("--adapter_name", type=str, default="wavelength_lora")
    args = parser.parse_args()

    result = train_lora(args.config, args.instance_data_dir, args.output_dir, args.adapter_name)
    logger.info(f"Training complete: {result}")
```

- [ ] **Step 3: Verify training script imports**

```bash
cd wafer-lora-service
pip install peft bitsandbytes
python -c "from training.train_lora import train_lora, WaferPairDataset; print('Import OK')"
```

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "feat: add LoRA training script with wafer dataset"
```

---

### Task 15: ControlNet 物理约束集成

**Files:**
- Modify: `wafer-lora-service/core/model_manager.py` (已集成，新增 PISM gain map 路径)
- Create: `wafer-lora-service/core/physics_control.py`

**Interfaces:**
- Produces: PISM 增益图作为 ControlNet 额外条件
- Produces: `compute_gain_map(image)` → 灰度增益图

- [ ] **Step 1: Create `core/physics_control.py`**

```python
"""
Physics-informed control for LoRA inference.

Provides gain map computation (simulating PISM output) as ControlNet
conditioning for wavelength conversion mode.
"""
import logging

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


def compute_gain_map(image: Image.Image, lambda_in: float = 266.0, lambda_out: float = 193.0) -> Image.Image:
    """
    Compute simplified Rayleigh scattering gain map.

    This approximates PISM's output for ControlNet conditioning.
    The full PISM gain map requires the wafer-inspection model;
    this is a lightweight analytical approximation.

    Args:
        image: Input 266nm wafer image (grayscale compatible)
        lambda_in: Input wavelength in nm (266)
        lambda_out: Output wavelength in nm (193)

    Returns:
        Gain map image (grayscale, 0-255)
    """
    # Rayleigh scattering ratio
    rayleigh_gain = (lambda_in / lambda_out) ** 4  # ~3.5 for 266→193

    # Convert to numpy
    if image.mode != "L":
        gray = image.convert("L")
    else:
        gray = image.copy()
    img_array = np.array(gray, dtype=np.float32)

    # Local contrast as defect proxy (high contrast = defect = higher gain)
    from scipy.ndimage import uniform_filter
    local_mean = uniform_filter(img_array, size=15)
    local_contrast = np.abs(img_array - local_mean)

    # Normalize contrast to [0, 1]
    contrast_norm = np.clip(local_contrast / 64.0, 0, 1)

    # Combine: base Rayleigh gain + defect modulation
    # Low contrast regions → near Rayleigh gain
    # High contrast regions → enhanced gain (simulating Mie resonance)
    gain_array = rayleigh_gain * (1.0 + 0.5 * contrast_norm)

    # Clamp to physical range [1.0, 8.0]
    gain_array = np.clip(gain_array, 1.0, 8.0)

    # Scale to 8-bit for ControlNet
    gain_8bit = ((gain_array - 1.0) / 7.0 * 255).astype(np.uint8)

    return Image.fromarray(gain_8bit, mode="L").convert("RGB")
```

- [ ] **Step 2: Update infer route to use gain map for wavelength mode**

Edit `wafer-lora-service/api/routes/infer.py`, add gain map logic in the infer function:

```python
# Add import at top
from core.physics_control import compute_gain_map

# In infer() function, after image resize, add:
if req.mode == "wavelength" and not req.control_image_base64:
    # Use physics-informed gain map as ControlNet condition
    control_image = compute_gain_map(image)
    logger.info("Using physics-informed gain map for wavelength conversion")
```

- [ ] **Step 3: Verify import**

```bash
cd wafer-lora-service
python -c "from core.physics_control import compute_gain_map; print('OK')"
```

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "feat: add physics-informed ControlNet gain map"
```

---

## 里程碑 3: 后端 ↔ LoRA 联调

### Task 16: LoraInferenceClient — Feign 客户端

**Files:**
- Create: `wafer-backend/src/main/java/com/wafer/client/LoraInferenceClient.java`
- Create: `wafer-backend/src/main/java/com/wafer/model/dto/LoraInferRequest.java`
- Create: `wafer-backend/src/main/java/com/wafer/model/dto/LoraInferResponse.java`

**Interfaces:**
- Produces: `LoraInferenceClient.infer(InferRequest)` → `InferResponse`

- [ ] **Step 1: Create `LoraInferRequest.java`**

```java
package com.wafer.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class LoraInferRequest {
    private String imageBase64;
    private String mode;         // enhance / wavelength / defect
    private Double strength;     // 0.1 ~ 1.0
    private String prompt;
    private String controlImageBase64;
}
```

- [ ] **Step 2: Create `LoraInferResponse.java`**

```java
package com.wafer.model.dto;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;

@Data
@Builder
@NoArgsConstructor
@AllArgsConstructor
public class LoraInferResponse {
    private String resultBase64;
    private Integer durationMs;
}
```

- [ ] **Step 3: Create `LoraInferenceClient.java`**

```java
package com.wafer.client;

import com.wafer.model.dto.LoraInferRequest;
import com.wafer.model.dto.LoraInferResponse;
import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;

@FeignClient(
    name = "lora-service",
    url = "${lora.service.url}",
    path = "/api/v1/lora"
)
public interface LoraInferenceClient {

    @PostMapping("/infer")
    LoraInferResponse infer(@RequestBody LoraInferRequest request);

    @PostMapping("/switch")
    void switchLora(@RequestBody LoraInferRequest request);
}
```

- [ ] **Step 4: Verify compilation**

```bash
cd wafer-backend
mvn clean compile
```

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "feat: add Feign client for LoRA inference service"
```

---

### Task 17: InferenceService — 异步推理编排

**Files:**
- Create: `wafer-backend/src/main/java/com/wafer/service/InferenceService.java`
- Create: `wafer-backend/src/main/java/com/wafer/controller/InferenceController.java`
- Create: `wafer-backend/src/main/java/com/wafer/service/SseService.java`

**Interfaces:**
- Produces: POST `/api/v1/inference` — 提交推理任务
- Produces: GET `/api/v1/inference/{id}` — 查询任务状态

- [ ] **Step 1: Create `SseService.java`**

```java
package com.wafer.service;

import org.springframework.stereotype.Service;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.io.IOException;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

@Service
public class SseService {

    private final Map<Long, SseEmitter> emitters = new ConcurrentHashMap<>();

    public SseEmitter createEmitter(Long taskId) {
        SseEmitter emitter = new SseEmitter(300_000L); // 5 min timeout
        emitters.put(taskId, emitter);
        emitter.onCompletion(() -> emitters.remove(taskId));
        emitter.onTimeout(() -> emitters.remove(taskId));
        return emitter;
    }

    public void sendProgress(Long taskId, String stage, int progress) {
        SseEmitter emitter = emitters.get(taskId);
        if (emitter != null) {
            try {
                emitter.send(SseEmitter.event()
                        .name("progress")
                        .data(Map.of("stage", stage, "progress", progress)));
            } catch (IOException e) {
                emitters.remove(taskId);
            }
        }
    }

    public void complete(Long taskId, Object result) {
        SseEmitter emitter = emitters.get(taskId);
        if (emitter != null) {
            try {
                emitter.send(SseEmitter.event()
                        .name("complete")
                        .data(result));
                emitter.complete();
            } catch (IOException e) {
                // ignore
            }
            emitters.remove(taskId);
        }
    }

    public void error(Long taskId, String error) {
        SseEmitter emitter = emitters.get(taskId);
        if (emitter != null) {
            try {
                emitter.send(SseEmitter.event()
                        .name("error")
                        .data(Map.of("error", error)));
                emitter.completeWithError(new RuntimeException(error));
            } catch (IOException e) {
                // ignore
            }
            emitters.remove(taskId);
        }
    }
}
```

- [ ] **Step 2: Create `InferenceService.java`**

```java
package com.wafer.service;

import com.wafer.client.LoraInferenceClient;
import com.wafer.model.dto.InferenceResponse;
import com.wafer.model.dto.LoraInferRequest;
import com.wafer.model.dto.LoraInferResponse;
import com.wafer.model.entity.InferenceTask;
import com.wafer.model.enums.InferenceType;
import com.wafer.model.enums.TaskStatus;
import com.wafer.repository.InferenceTaskRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.time.LocalDateTime;
import java.util.Base64;
import java.util.Optional;

@Slf4j
@Service
@RequiredArgsConstructor
public class InferenceService {

    private final InferenceTaskRepository taskRepository;
    private final LoraInferenceClient loraClient;
    private final SseService sseService;
    private final CacheService cacheService;
    private final StorageService storageService;

    /**
     * Submit a new inference task (synchronous, returns taskId immediately).
     */
    public InferenceResponse submitTask(MultipartFile file, InferenceType type, String paramsJson) {
        // Upload file to MinIO
        String path = storageService.upload(file, StorageService.generatePath("uploads"));
        String inputUrl = storageService.getUrl(path);

        // Create task record
        InferenceTask task = InferenceTask.builder()
                .type(type)
                .status(TaskStatus.PENDING)
                .inputUrl(inputUrl)
                .params(paramsJson)
                .build();
        task = taskRepository.save(task);

        // Start async inference
        Long taskId = task.getId();
        processInference(taskId);

        return toResponse(task);
    }

    /**
     * Async inference processing.
     */
    @Async("taskExecutor")
    public void processInference(Long taskId) {
        InferenceTask task = taskRepository.findById(taskId)
                .orElseThrow(() -> new RuntimeException("Task not found: " + taskId));

        try {
            // Update status
            task.setStatus(TaskStatus.RUNNING);
            taskRepository.save(task);
            sseService.sendProgress(taskId, "running", 10);

            // Build LoRA inference request
            LoraInferRequest loraReq = LoraInferRequest.builder()
                    .mode(task.getType().name().toLowerCase())
                    .imageBase64(encodeImageToBase64(task.getInputUrl()))
                    .strength(0.75)
                    .build();

            sseService.sendProgress(taskId, "inferring", 50);

            // Call FastAPI
            LoraInferResponse loraResp = loraClient.infer(loraReq);

            sseService.sendProgress(taskId, "saving", 80);

            // Save result to MinIO
            byte[] resultBytes = Base64.getDecoder().decode(loraResp.getResultBase64());
            // In production: upload to MinIO and get URL
            // Here we store in cache for simplicity
            cacheService.cacheTaskResult(taskId, loraResp);

            // Update task
            task.setStatus(TaskStatus.DONE);
            task.setDurationMs(loraResp.getDurationMs());
            task.setCompletedAt(LocalDateTime.now());
            taskRepository.save(task);

            sseService.complete(taskId, toResponse(task));
            log.info("Task {} completed in {}ms", taskId, loraResp.getDurationMs());

        } catch (Exception e) {
            log.error("Task {} failed", taskId, e);
            task.setStatus(TaskStatus.FAILED);
            task.setErrorMessage(e.getMessage());
            task.setCompletedAt(LocalDateTime.now());
            taskRepository.save(task);
            sseService.error(taskId, e.getMessage());
        }
    }

    /**
     * Get task status and result.
     */
    public Optional<InferenceResponse> getTask(Long taskId) {
        return taskRepository.findById(taskId).map(this::toResponse);
    }

    private String encodeImageToBase64(String url) {
        // In production: download from MinIO and encode
        // Placeholder: return a dummy base64 string
        return "";
    }

    private InferenceResponse toResponse(InferenceTask task) {
        return InferenceResponse.builder()
                .taskId(task.getId())
                .type(task.getType())
                .status(task.getStatus())
                .inputUrl(task.getInputUrl())
                .outputUrl(task.getOutputUrl())
                .thumbnailUrl(task.getThumbnailUrl())
                .durationMs(task.getDurationMs())
                .errorMessage(task.getErrorMessage())
                .build();
    }
}
```

- [ ] **Step 3: Create `InferenceController.java`**

```java
package com.wafer.controller;

import com.wafer.model.dto.InferenceResponse;
import com.wafer.model.enums.InferenceType;
import com.wafer.service.InferenceService;
import com.wafer.service.SseService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.servlet.mvc.method.annotation.SseEmitter;

import java.util.Map;

@RestController
@RequestMapping("/api/v1/inference")
@RequiredArgsConstructor
public class InferenceController {

    private final InferenceService inferenceService;
    private final SseService sseService;

    @PostMapping
    public ResponseEntity<InferenceResponse> submit(
            @RequestParam("file") MultipartFile file,
            @RequestParam("type") String type,
            @RequestParam(value = "params", required = false) String params) {

        InferenceType inferenceType;
        try {
            inferenceType = InferenceType.valueOf(type.toUpperCase());
        } catch (IllegalArgumentException e) {
            return ResponseEntity.badRequest().build();
        }

        InferenceResponse response = inferenceService.submitTask(file, inferenceType, params);
        return ResponseEntity.accepted().body(response);
    }

    @GetMapping("/{id}")
    public ResponseEntity<InferenceResponse> getTask(@PathVariable Long id) {
        return inferenceService.getTask(id)
                .map(ResponseEntity::ok)
                .orElse(ResponseEntity.notFound().build());
    }

    @GetMapping(value = "/{id}/stream", produces = MediaType.TEXT_EVENT_STREAM_VALUE)
    public SseEmitter streamTask(@PathVariable Long id) {
        return sseService.createEmitter(id);
    }

    @PostMapping("/batch")
    public ResponseEntity<Map<String, Object>> batch(
            @RequestBody Map<String, Object> request) {
        // Batch inference for gallery pre-generation
        // Processes multiple images with specified LoRA mode
        return ResponseEntity.ok(Map.of("status", "not implemented"));
    }
}
```

- [ ] **Step 4: Verify compilation**

```bash
cd wafer-backend
mvn clean compile
```

- [ ] **Step 5: Commit**

```bash
git add .
git commit -m "feat: add async inference service + SSE push"
```

---

### Task 18: 前端工作台页面 — 上传 + 选择 + 推理展示

**Files:**
- Create: `wafer-showcase/src/api/index.js`
- Create: `wafer-showcase/src/hooks/useInference.js`
- Create: `wafer-showcase/src/hooks/useGallery.js`
- Create: `wafer-showcase/src/components/ImageUploader.jsx`
- Create: `wafer-showcase/src/components/ModelSelector.jsx`
- Create: `wafer-showcase/src/components/InferenceResult.jsx`
- Create: `wafer-showcase/src/components/ProgressTracker.jsx`
- Create: `wafer-showcase/src/pages/Workbench.jsx`
- Create: `wafer-showcase/src/pages/Gallery.jsx`

**Interfaces:**
- Produces: 工作台页面，用户上传图片 → 选模式 → 推理展示对比图

- [ ] **Step 1: Create `wafer-showcase/src/api/index.js`**

```javascript
import axios from 'axios'

const BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8080/api/v1'

export const api = {
  // Gallery
  getGallery: (params) => axios.get(`${BASE}/gallery`, { params }),
  getGalleryStats: () => axios.get(`${BASE}/gallery/stats`),

  // Inference
  submitInference: async (type, file, params = {}) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('type', type)
    if (Object.keys(params).length) {
      fd.append('params', JSON.stringify(params))
    }
    const res = await axios.post(`${BASE}/inference`, fd)
    return res.data
  },

  getTaskStatus: async (id) => {
    const res = await axios.get(`${BASE}/inference/${id}`)
    return res.data
  },

  getTaskStream: (id) => {
    return new EventSource(`${BASE}/inference/${id}/stream`)
  },

  // Metrics
  getMetrics: async () => {
    const res = await axios.get(`${BASE}/metrics/overview`)
    return res.data
  },
}
```

- [ ] **Step 2: Create `wafer-showcase/src/hooks/useInference.js`**

```javascript
import { useState, useCallback, useRef } from 'react'
import { api } from '../api'

export function useInference() {
  const [taskId, setTaskId] = useState(null)
  const [status, setStatus] = useState('idle') // idle | uploading | running | done | error
  const [result, setResult] = useState(null)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState(null)
  const eventSourceRef = useRef(null)

  const submit = useCallback(async (type, file, params) => {
    setStatus('uploading')
    setError(null)
    setResult(null)
    setProgress(0)

    try {
      const task = await api.submitInference(type, file, params)
      setTaskId(task.taskId)
      setStatus('running')

      // Connect SSE for progress
      const es = api.getTaskStream(task.taskId)
      eventSourceRef.current = es

      es.addEventListener('progress', (e) => {
        const data = JSON.parse(e.data)
        setProgress(data.progress)
      })

      es.addEventListener('complete', (e) => {
        const data = JSON.parse(e.data)
        setResult(data)
        setStatus('done')
        setProgress(100)
        es.close()
      })

      es.addEventListener('error', (e) => {
        const data = JSON.parse(e.data)
        setError(data.error || '推理失败')
        setStatus('error')
        es.close()
      })

      // Fallback: poll if SSE fails
      const pollInterval = setInterval(async () => {
        const taskStatus = await api.getTaskStatus(task.taskId)
        if (taskStatus.status === 'DONE') {
          setResult(taskStatus)
          setStatus('done')
          setProgress(100)
          clearInterval(pollInterval)
          es.close()
        } else if (taskStatus.status === 'FAILED') {
          setError(taskStatus.errorMessage || '推理失败')
          setStatus('error')
          clearInterval(pollInterval)
          es.close()
        }
      }, 2000)

    } catch (err) {
      setError(err.message || '提交失败')
      setStatus('error')
    }
  }, [])

  const reset = useCallback(() => {
    setTaskId(null)
    setStatus('idle')
    setResult(null)
    setProgress(0)
    setError(null)
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
    }
  }, [])

  return { taskId, status, result, progress, error, submit, reset }
}
```

- [ ] **Step 3: Create `wafer-showcase/src/components/ImageUploader.jsx`**

```jsx
import { useState, useRef } from 'react'
import { Upload, X } from 'lucide-react'
import { cn } from '../utils/cn'

export default function ImageUploader({ onFileSelect, disabled }) {
  const [preview, setPreview] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef(null)

  const handleFile = (file) => {
    if (!file || !file.type.startsWith('image/')) return
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(file)
    onFileSelect(file)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files[0])
  }

  const clearImage = () => {
    setPreview(null)
    onFileSelect(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div
      className={cn(
        'relative border-2 border-dashed rounded-xl p-8 text-center transition-all cursor-pointer',
        'border-white/10 hover:border-tech-cyan/50',
        dragOver && 'border-tech-cyan bg-tech-cyan/5',
        disabled && 'opacity-50 pointer-events-none',
      )}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      {preview ? (
        <div className="relative inline-block">
          <img
            src={preview}
            alt="预览"
            className="max-h-64 rounded-lg object-contain"
          />
          <button
            onClick={(e) => { e.stopPropagation(); clearImage() }}
            className="absolute -top-2 -right-2 p-1 bg-red-500 rounded-full"
          >
            <X size={14} />
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          <Upload className="mx-auto text-zinc-400" size={40} />
          <p className="text-zinc-400">
            拖拽晶圆图片到此处，或<strong className="text-tech-cyan">点击选择</strong>
          </p>
          <p className="text-zinc-600 text-sm">支持 JPG / PNG / BMP，最大 50MB</p>
        </div>
      )}
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => handleFile(e.target.files[0])}
      />
    </div>
  )
}
```

- [ ] **Step 4: Create `wafer-showcase/src/components/ModelSelector.jsx`**

```jsx
import { Sparkles, Waves, BugPlay } from 'lucide-react'
import { cn } from '../utils/cn'

const MODELS = [
  {
    id: 'enhance',
    label: '暗场 → 明场增强',
    desc: '提高缺陷可见度，模拟高倍明场检测效果',
    icon: Sparkles,
    color: 'text-tech-cyan',
    gradient: 'from-tech-cyan/20 to-transparent',
  },
  {
    id: 'wavelength',
    label: '266nm → 193nm 波长转换',
    desc: '虚拟深紫外增强，Rayleigh 散射增益 ~3.5×',
    icon: Waves,
    color: 'text-purple-400',
    gradient: 'from-purple-500/20 to-transparent',
  },
  {
    id: 'defect',
    label: '缺陷生成',
    desc: '在无缺陷晶圆上合成逼真的缺陷图像',
    icon: BugPlay,
    color: 'text-emerald-400',
    gradient: 'from-emerald-500/20 to-transparent',
  },
]

export default function ModelSelector({ selected, onSelect, disabled }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {MODELS.map((model) => {
        const Icon = model.icon
        const isActive = selected === model.id
        return (
          <button
            key={model.id}
            disabled={disabled}
            onClick={() => onSelect(model.id)}
            className={cn(
              'relative p-5 rounded-xl border text-left transition-all',
              'border-white/10 bg-white/5 backdrop-blur-sm',
              isActive
                ? 'border-tech-cyan/50 shadow-[0_0_20px_rgba(0,229,255,0.1)]'
                : 'hover:border-white/20 hover:-translate-y-0.5',
              disabled && 'opacity-50 cursor-not-allowed',
            )}
          >
            <Icon className={cn('mb-3', model.color)} size={28} />
            <h3 className={cn('font-semibold mb-1', isActive ? 'text-white' : 'text-zinc-300')}>
              {model.label}
            </h3>
            <p className="text-sm text-zinc-500">{model.desc}</p>
            {isActive && (
              <div className={cn(
                'absolute inset-0 rounded-xl bg-gradient-to-b opacity-20 pointer-events-none',
                model.gradient,
              )} />
            )}
          </button>
        )
      })}
    </div>
  )
}
```

- [ ] **Step 5: Create `wafer-showcase/src/components/InferenceResult.jsx`**

```jsx
import { Clock, Activity } from 'lucide-react'

export default function InferenceResult({ originalFile, result, progress, status }) {
  if (status === 'idle') return null

  return (
    <div className="space-y-4 mt-8">
      <h3 className="text-lg font-semibold text-white">推理结果</h3>

      {status === 'uploading' && (
        <div className="p-8 text-center text-zinc-400">
          <p>上传中...</p>
        </div>
      )}

      {status === 'running' && (
        <div className="p-8 text-center">
          <div className="w-full bg-white/5 rounded-full h-2 mb-4">
            <div
              className="h-full bg-gradient-to-r from-tech-cyan to-tech-purple rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-zinc-400">推理中... {progress}%</p>
        </div>
      )}

      {status === 'done' && result && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-2">
            <p className="text-sm text-zinc-500">原图</p>
            <img
              src={originalFile ? URL.createObjectURL(originalFile) : ''}
              alt="原图"
              className="rounded-lg border border-white/10 w-full"
            />
          </div>
          <div className="space-y-2">
            <p className="text-sm text-tech-cyan">LoRA 结果</p>
            <div className="rounded-lg border border-tech-cyan/30 bg-white/5 p-4 h-full flex items-center justify-center">
              <p className="text-zinc-500 text-sm">结果图片将在此显示</p>
            </div>
          </div>
          <div className="space-y-2">
            <p className="text-sm text-zinc-500">差异对比</p>
            <div className="rounded-lg border border-white/10 bg-white/5 p-4 h-full flex items-center justify-center">
              <p className="text-zinc-500 text-sm">差异热力图将在此显示</p>
            </div>
          </div>
        </div>
      )}

      {status === 'error' && (
        <div className="p-6 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400">
          <p>推理失败：{result?.errorMessage || '未知错误'}</p>
        </div>
      )}
    </div>
  )
}
```

- [ ] **Step 6: Create `wafer-showcase/src/components/ProgressTracker.jsx`**

```jsx
import { cn } from '../utils/cn'

const STEPS = [
  { key: 'upload', label: '上传图片' },
  { key: 'infer', label: 'AI 推理' },
  { key: 'result', label: '查看结果' },
]

export default function ProgressTracker({ currentStep }) {
  const stepIndex = { idle: 0, uploading: 0, running: 1, done: 2, error: 1 }

  return (
    <div className="flex items-center justify-center gap-2 mb-6">
      {STEPS.map((step, i) => {
        const isActive = i <= stepIndex[currentStep]
        const isLast = i === STEPS.length - 1
        return (
          <div key={step.key} className="flex items-center gap-2">
            <div className={cn(
              'flex items-center gap-2 px-3 py-1.5 rounded-full text-sm transition-all',
              isActive ? 'bg-tech-cyan/20 text-tech-cyan' : 'bg-white/5 text-zinc-600',
            )}>
              <div className={cn(
                'w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold',
                isActive ? 'bg-tech-cyan text-white' : 'bg-zinc-700 text-zinc-500',
              )}>
                {i + 1}
              </div>
              <span>{step.label}</span>
            </div>
            {!isLast && (
              <div className={cn(
                'w-8 h-0.5',
                i < stepIndex[currentStep] ? 'bg-tech-cyan' : 'bg-white/10',
              )} />
            )}
          </div>
        )
      })}
    </div>
  )
}
```

- [ ] **Step 7: Create `wafer-showcase/src/pages/Workbench.jsx`**

```jsx
import { useState } from 'react'
import { motion } from 'framer-motion'
import ImageUploader from '../components/ImageUploader'
import ModelSelector from '../components/ModelSelector'
import InferenceResult from '../components/InferenceResult'
import ProgressTracker from '../components/ProgressTracker'
import { useInference } from '../hooks/useInference'

export default function Workbench() {
  const [selectedModel, setSelectedModel] = useState(null)
  const [file, setFile] = useState(null)
  const { status, result, progress, error, submit, reset } = useInference()

  const handleInfer = () => {
    if (!file || !selectedModel) return
    submit(selectedModel, file)
  }

  return (
    <section className="min-h-screen bg-tech-deep pt-24 pb-16 px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-content mx-auto space-y-8"
      >
        <div>
          <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-tech-cyan to-tech-purple mb-2">
            AI 推理工作台
          </h1>
          <p className="text-zinc-500">
            上传晶圆图像，选择 LoRA 模式，体验 AI 驱动的晶圆图像增强
          </p>
        </div>

        <ProgressTracker currentStep={status} />

        <ModelSelector
          selected={selectedModel}
          onSelect={setSelectedModel}
          disabled={status === 'running'}
        />

        <ImageUploader
          onFileSelect={setFile}
          disabled={status === 'running'}
        />

        {file && selectedModel && status === 'idle' && (
          <motion.button
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            onClick={handleInfer}
            className="w-full py-3 rounded-xl font-semibold bg-gradient-to-r from-tech-cyan to-tech-purple text-white hover:opacity-90 transition-all"
          >
            开始推理
          </motion.button>
        )}

        {status !== 'idle' && (
          <motion.button
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            onClick={reset}
            className="w-full py-2 rounded-xl border border-white/10 text-zinc-400 hover:text-white transition-all"
          >
            重新开始
          </motion.button>
        )}

        <InferenceResult
          originalFile={file}
          result={result}
          progress={progress}
          status={status}
          error={error}
        />
      </motion.div>
    </section>
  )
}
```

- [ ] **Step 8: Create `wafer-showcase/src/pages/Gallery.jsx`**

```jsx
import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { api } from '../api'

const CATEGORIES = [
  { id: 'all', label: '全部' },
  { id: 'enhance', label: '暗场→明场增强' },
  { id: 'wavelength', label: '266→193nm 转换' },
  { id: 'defect', label: '缺陷生成' },
]

export default function Gallery() {
  const [items, setItems] = useState([])
  const [category, setCategory] = useState('all')
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    api.getGallery({ category: category === 'all' ? undefined : category })
      .then(res => setItems(res.data.content || []))
      .finally(() => setLoading(false))
  }, [category])

  return (
    <section className="min-h-screen bg-tech-deep pt-24 pb-16 px-4">
      <div className="max-w-content mx-auto">
        <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-tech-cyan to-tech-purple mb-8">
          技术展厅
        </h1>

        {/* Category tabs */}
        <div className="flex gap-2 mb-8 flex-wrap">
          {CATEGORIES.map((cat) => (
            <button
              key={cat.id}
              onClick={() => setCategory(cat.id)}
              className={`px-4 py-2 rounded-full text-sm transition-all ${
                category === cat.id
                  ? 'bg-tech-cyan/20 text-tech-cyan border border-tech-cyan/30'
                  : 'bg-white/5 text-zinc-400 border border-white/10 hover:border-white/20'
              }`}
            >
              {cat.label}
            </button>
          ))}
        </div>

        {/* Gallery grid */}
        {loading ? (
          <div className="text-center text-zinc-500 py-20">加载中...</div>
        ) : items.length === 0 ? (
          <div className="text-center text-zinc-500 py-20">暂无素材</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {items.map((item, i) => (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                className="rounded-xl border border-white/10 bg-white/5 overflow-hidden group hover:border-tech-cyan/30 transition-all"
              >
                <div className="aspect-video bg-zinc-900 relative overflow-hidden">
                  {item.thumbnailUrl ? (
                    <img
                      src={item.thumbnailUrl}
                      alt={item.title}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="flex items-center justify-center h-full text-zinc-700">
                      预览图
                    </div>
                  )}
                </div>
                <div className="p-4">
                  <h3 className="text-white font-medium mb-1">{item.title}</h3>
                  <p className="text-zinc-500 text-sm line-clamp-2">{item.description}</p>
                  {item.metrics && (
                    <div className="mt-2 flex gap-2 text-xs text-zinc-600">
                      <span className="px-2 py-0.5 rounded bg-white/5">{item.metrics}</span>
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </section>
  )
}
```

- [ ] **Step 9: Commit**

```bash
git add .
git commit -m "feat: add Workbench and Gallery pages with inference flow"
```

---

## 里程碑 4: 优化 + 部署

### Task 19: 推理优化 — torch.compile + VAE slicing

**Files:**
- Modify: `wafer-lora-service/core/model_manager.py` (启用 compile)

- [ ] **Step 1: In `ModelManager.__init__`, add torch.compile to UNet**

```python
# After self.base.to(self.device), add:
if hasattr(torch, 'compile'):
    logger.info("Applying torch.compile to UNet...")
    self.base.unet = torch.compile(
        self.base.unet,
        mode="reduce-overhead",  # Best for inference latency
        fullgraph=True,
    )
```

- [ ] **Step 2: Verify benchmark**

```bash
cd wafer-lora-service
python -c "
from core.model_manager import ModelManager
import torch
m = ModelManager()
# Warmup
dummy = torch.randn(1, 3, 512, 512).half().cuda()
for _ in range(3):
    m.base.unet(dummy, torch.randint(0, 1000, (1,)).cuda(), torch.randn(1, 77, 768).half().cuda())
# Benchmark
import time
t0 = time.time()
for _ in range(10):
    m.base.unet(dummy, torch.randint(0, 1000, (1,)).cuda(), torch.randn(1, 77, 768).half().cuda())
torch.cuda.synchronize()
print(f'Avg: {(time.time()-t0)/10*1000:.1f}ms')
"
# Expected: ~15-25ms per UNet forward
```

- [ ] **Step 3: Commit**

```bash
git add .
git commit -m "perf: enable torch.compile for UNet inference"
```

---

### Task 20: Docker Compose 集成 LoRA 服务

**Files:**
- Modify: `docker-compose.yml` (添加 lora-service)
- Create: `wafer-lora-service/Dockerfile`

- [ ] **Step 1: Update `docker-compose.yml` with LoRA service**

```yaml
  lora-service:
    build: ./wafer-lora-service
    ports: ["8000:8000"]
    runtime: nvidia
    environment:
      CUDA_VISIBLE_DEVICES: "0"
      LORA_MODEL_ID: runwayml/stable-diffusion-v1-5
      LORA_TORCH_DTYPE: float16
      LORA_DEVICE: cuda
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    volumes:
      - ./wafer-lora-service/loras:/app/loras
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

- [ ] **Step 2: Update `nginx.conf` to proxy LoRA service**

```nginx
upstream lora_service {
    server lora-service:8000;
}

location /api/v1/lora/ {
    proxy_pass http://lora_service;
    proxy_set_header Host $host;
    client_max_body_size 100M;
}
```

- [ ] **Step 3: Verify full stack starts**

```bash
docker-compose up -d
docker-compose ps
# Expected: all 6 services running
curl http://localhost/api/v1/health
# Expected: {"status":"UP","service":"wafer-backend"}
curl http://localhost/api/v1/lora/active
# Expected: [{"name":"enhance","active":true,"weight":1.0}]
```

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "feat: integrate LoRA service into Docker Compose"
```

---

### Task 21: MetricsController + 监控

**Files:**
- Create: `wafer-backend/src/main/java/com/wafer/controller/MetricsController.java`
- Create: `wafer-showcase/src/pages/Metrics.jsx`

- [ ] **Step 1: Create `MetricsController.java`**

```java
package com.wafer.controller;

import com.wafer.model.enums.TaskStatus;
import com.wafer.repository.InferenceTaskRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

@RestController
@RequestMapping("/api/v1/metrics")
@RequiredArgsConstructor
public class MetricsController {

    private final InferenceTaskRepository taskRepository;

    @GetMapping("/overview")
    public ResponseEntity<Map<String, Object>> overview() {
        return ResponseEntity.ok(Map.of(
            "totalTasks", taskRepository.count(),
            "pendingTasks", taskRepository.countByStatus(TaskStatus.PENDING),
            "runningTasks", taskRepository.countByStatus(TaskStatus.RUNNING),
            "doneTasks", taskRepository.countByStatus(TaskStatus.DONE),
            "failedTasks", taskRepository.countByStatus(TaskStatus.FAILED)
        ));
    }
}
```

- [ ] **Step 2: Create `wafer-showcase/src/pages/Metrics.jsx`**

```jsx
import { useState, useEffect } from 'react'
import { api } from '../api'
import { Activity, CheckCircle, Clock, AlertCircle, Loader } from 'lucide-react'

const STAT_CARDS = [
  { key: 'totalTasks', label: '推理总量', icon: Activity, color: 'text-tech-cyan' },
  { key: 'doneTasks', label: '已完成', icon: CheckCircle, color: 'text-emerald-400' },
  { key: 'runningTasks', label: '运行中', icon: Loader, color: 'text-purple-400' },
  { key: 'pendingTasks', label: '等待中', icon: Clock, color: 'text-yellow-400' },
  { key: 'failedTasks', label: '失败', icon: AlertCircle, color: 'text-red-400' },
]

export default function Metrics() {
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.getMetrics()
      .then(setMetrics)
      .finally(() => setLoading(false))
  }, [])

  return (
    <section className="min-h-screen bg-tech-deep pt-24 pb-16 px-4">
      <div className="max-w-content mx-auto">
        <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-tech-cyan to-tech-purple mb-8">
          系统指标
        </h1>

        {loading ? (
          <div className="text-center text-zinc-500 py-20">加载中...</div>
        ) : !metrics ? (
          <div className="text-center text-zinc-500 py-20">暂无数据</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
            {STAT_CARDS.map(({ key, label, icon: Icon, color }) => (
              <div
                key={key}
                className="rounded-xl border border-white/10 bg-white/5 p-6 hover:border-tech-cyan/30 transition-all"
              >
                <Icon className={color} size={32} />
                <p className="text-3xl font-bold text-white mt-2">
                  {metrics[key] ?? 0}
                </p>
                <p className="text-zinc-500 text-sm mt-1">{label}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  )
}
```

- [ ] **Step 3: Verify compilation**

```bash
cd wafer-backend
mvn clean compile
```

- [ ] **Step 4: Commit**

```bash
git add .
git commit -m "feat: add metrics endpoints and page"
```

---

## Spec Coverage Check

| Spec Section | Task |
|-------------|------|
| Spring Boot 项目结构 | Task 1 |
| 配置层 (CORS/MinIO/Redis/Async) | Task 2 |
| JPA 实体 + Repository | Task 3 |
| 异常处理 + DTO | Task 4 |
| StorageService (MinIO) | Task 5 |
| CacheService (Redis) | Task 6 |
| GalleryService + Controller | Task 7 |
| ImageController | Task 8 |
| Docker Compose 基础 | Task 9 |
| FastAPI 脚手架 | Task 10 |
| ModelManager + LoRA 热切换 | Task 11 |
| 推理路由 | Task 12 |
| PISM 伪标签管线 | Task 13 |
| LoRA 训练脚本 | Task 14 |
| ControlNet 物理约束 | Task 15 |
| Feign 客户端 | Task 16 |
| 异步推理编排 + SSE | Task 17 |
| 前端工作台 | Task 18 |
| torch.compile 优化 | Task 19 |
| Docker Compose 集成 LoRA | Task 20 |
| 监控指标 | Task 21 |
| 性能压测 | (Task 19 benchmark) |
| 文档完善 | (本计划 + 设计文档) |

**未覆盖的 spec 要求**: 文档中提到 CI/CD 和性能压测（SLO 达标）。这些依赖具体工具链（GitLab CI/GitHub Actions），会在 MS4 实施时根据基础设施补充。

---
