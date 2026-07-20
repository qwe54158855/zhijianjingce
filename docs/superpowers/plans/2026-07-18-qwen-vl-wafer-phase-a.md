# Qwen-VL 晶圆检测 Phase A 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 在 WSL 本地 CPU 环境部署 Qwen2.5-VL-7B (4-bit) + OpenCV 增强流水线，实现暗场晶圆图的智能增强、缺陷检出和报告生成。

**架构概要：**
- `llama-server` (llama.cpp) 提供 Qwen-VL HTTP API 推理后端
- `wafer-qwen-service` (FastAPI) 编排增强/检出/报告流水线，复用 `wafer-lora-service` 的项目模式
- `wafer-showcase` 新增 Qwen 检测展示页（React + Tailwind + Framer Motion）
- `wafer-backend` 新增 Feign 客户端代理 Qwen 服务

**技术栈：** Python 3.10+ / FastAPI / OpenCV / llama.cpp / Qwen2.5-VL-7B-GGUF / React 18 / Tailwind 3 / Spring Boot 3 / Feign

## 全局约束

- 所有 Python 服务沿用 `wafer-lora-service` 模式：`main.py` + `core/config.py` + `core/model_manager.py` + `api/routes.py`
- 所有新代码必须遵循现有项目的代码风格（注释密度、命名习惯）
- 前端样式使用现有 `tech-deep`/`tech-card`/`tech-cyan` 色板
- 检测框颜色：崩边=红、颗粒=青、划痕=绿、位错=紫
- CPU 推理，无 CUDA 依赖
- 不引入水印、不依赖云 API

---

## 文件结构总览

```
wafer-qwen-service/              # 新增服务
├── main.py                      # FastAPI 应用入口
├── requirements.txt             # Python 依赖
├── Dockerfile                   # Docker 镜像构建
├── core/
│   ├── __init__.py
│   ├── config.py                # 配置（pydantic-settings）
│   └── model_manager.py         # llama.cpp HTTP 客户端
├── engine/
│   ├── __init__.py
│   ├── enhancer.py              # OpenCV 增强流水线（CLAHE+NLM+Gamma）
│   ├── detector.py              # Qwen ROI 引导的形态学缺陷检出
│   ├── visualizer.py            # YOLO 风格检测框渲染
│   └── reporter.py              # Qwen 报告生成
├── prompts/
│   ├── enhance.yaml             # 增强分析 prompt
│   └── report.yaml              # 报告生成 prompt
├── api/
│   ├── __init__.py
│   └── routes.py                # 三个端点：enhance / report / analyze
├── models/
│   ├── __init__.py
│   └── schemas.py               # Pydantic 请求/响应模型
└── tests/
    ├── __init__.py
    ├── test_enhancer.py
    ├── test_detector.py
    └── test_visualizer.py

wafer-backend/src/main/java/com/wafer/
├── client/QwenInferenceClient.java   # 新增 Feign 客户端
├── controller/QwenController.java    # 新增控制器
└── model/dto/QwenEnhanceRequest.java # 新增 DTO
└── model/dto/QwenEnhanceResponse.java

wafer-showcase/src/
├── pages/QwenDetectPage.jsx          # 新增 Qwen 检测展示页
├── components/QwenImageUploader.jsx  # 新增（复用逻辑，加双图对比）
├── components/QwenResultPanel.jsx    # 新增：检测结果面板
├── components/QwenReportPanel.jsx    # 新增：AI 报告面板
└── api/waferApi.js                   # 新增 API 调用函数

docker-compose.yml                   # 新增 wafer-qwen-service + llama-server
```

---

### Task 1: 创建 wafer-qwen-service 项目骨架

**文件：**
- Create: `wafer-qwen-service/main.py`
- Create: `wafer-qwen-service/requirements.txt`
- Create: `wafer-qwen-service/core/__init__.py`
- Create: `wafer-qwen-service/core/config.py`
- Create: `wafer-qwen-service/models/__init__.py`
- Create: `wafer-qwen-service/models/schemas.py`
- Create: `wafer-qwen-service/api/__init__.py`
- Create: `wafer-qwen-service/engine/__init__.py`
- Create: `wafer-qwen-service/prompts/__init__.py`

**接口：**
- 消费: 无（首个任务）
- 产出: `Settings` config 类, `QwenEnhanceRequest`/`QwenEnhanceResponse`/`Detection` schemas, FastAPI 应用骨架

- [ ] **Step 1: 创建 `wafer-qwen-service/requirements.txt`**

```
fastapi==0.110.0
uvicorn[standard]==0.27.0
opencv-python-headless>=4.8.0
numpy>=1.24.0
pillow>=10.0.0
httpx>=0.27.0
pyyaml>=6.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-multipart>=0.0.6
```

- [ ] **Step 2: 创建 `wafer-qwen-service/core/config.py`**

```python
from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Service
    service_name: str = "wafer-qwen-service"
    host: str = "0.0.0.0"
    port: int = 8001

    # llama-server
    llama_server_url: str = "http://localhost:8002/v1"
    llama_model_name: str = "qwen2.5-vl-7b"
    llama_timeout: int = 120

    # Enhancement defaults
    default_clahe_clip: float = 2.5
    default_clahe_grid: int = 8
    default_denoise: int = 10
    default_gamma: float = 1.2
    default_contrast: float = 1.3

    # Detection
    min_contour_area: int = 20
    detection_confidence_threshold: float = 0.5

    class Config:
        env_prefix = "QWEN_"


settings = Settings()
```

- [ ] **Step 3: 创建 `wafer-qwen-service/models/schemas.py`**

```python
from pydantic import BaseModel, Field
from typing import Optional


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class Detection(BaseModel):
    type: str = Field(description="缺陷类型: 崩边|颗粒|划痕|位错")
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox
    source: str = "qwen+cv"


class EnhanceAnalysis(BaseModel):
    brightness: str = "中"
    noise_level: str = "中"
    sharpness: str = "清晰"
    defect_count: int = 0


class QwenEnhanceRequest(BaseModel):
    image: str = Field(description="Base64 编码的图片数据")
    format: str = "jpg"


class QwenEnhanceResponse(BaseModel):
    success: bool = True
    enhanced_image: str = Field(description="Base64 编码的增强图（含检测框）")
    detections: list[Detection] = []
    analysis: EnhanceAnalysis = EnhanceAnalysis()
    inference_time_ms: int = 0


class QwenReportRequest(BaseModel):
    image: str = Field(description="Base64 编码的增强图片")
    detections: list[Detection] = []


class QwenReportResponse(BaseModel):
    success: bool = True
    report: str = ""
    inference_time_ms: int = 0


class QwenHealthResponse(BaseModel):
    status: str = "UP"
    llama_available: bool = False
    device: str = "cpu"
```

- [ ] **Step 4: 创建 `wafer-qwen-service/main.py`**

```python
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from models.schemas import QwenHealthResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
llama_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llama_client
    from core.model_manager import LlamaClient
    llama_client = LlamaClient()
    logger.info(f"Llama client initialized, server: {settings.llama_server_url}")
    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Wafer Qwen-VL Enhancement Service",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health", response_model=QwenHealthResponse)
async def health():
    global llama_client
    llama_ok = False
    if llama_client:
        llama_ok = await llama_client.check_health()
    return QwenHealthResponse(
        status="UP",
        llama_available=llama_ok,
        device="cpu",
    )


from api.routes import router
app.include_router(router, prefix="/api/v1/qwen")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=True)
```

- [ ] **Step 5: 验证项目骨架可启动**

```bash
cd wafer-qwen-service
pip install -r requirements.txt
python -c "from core.config import settings; print(f'Config OK: {settings.service_name}')"
```

预期输出：`Config OK: wafer-qwen-service`

- [ ] **Step 6: Commit**

```bash
git add wafer-qwen-service/
git commit -m "feat(qwen): scaffold wafer-qwen-service FastAPI project"
```

---

### Task 2: llama.cpp HTTP 客户端 (ModelManager)

**文件：**
- Create: `wafer-qwen-service/core/model_manager.py`

**接口：**
- 消费: `Settings (llama_server_url, llama_timeout)`
- 产出: `LlamaClient` 类（`analyze_image()`, `generate_report()`, `check_health()`）

- [ ] **Step 1: 创建 `wafer-qwen-service/core/model_manager.py`**

```python
import json
import logging
from typing import Optional

import httpx

from core.config import settings

logger = logging.getLogger(__name__)


class LlamaClient:
    """与 llama.cpp server 通信的 HTTP 客户端"""

    def __init__(self):
        self.base_url = settings.llama_server_url.rstrip("/")
        self.timeout = settings.llama_timeout
        self.client = httpx.AsyncClient(timeout=self.timeout)

    async def check_health(self) -> bool:
        """检查 llama-server 是否可用"""
        try:
            r = await self.client.get(f"{self.base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception as e:
            logger.warning(f"llama-server health check failed: {e}")
            return False

    async def analyze_image(
        self,
        image_base64: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.1,
    ) -> Optional[dict]:
        """
        调用 Qwen-VL 分析图像并返回结构化 JSON。

        Args:
            image_base64: Base64 编码的图像数据
            system_prompt: 系统级 prompt
            user_prompt: 用户级 prompt（不包含图像占位符，
                         图像通过 images 参数传入）
            max_tokens: 最大输出 token 数
            temperature: 采样温度，分析任务用低温度

        Returns:
            解析后的 dict，失败返回 None
        """
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            r = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]

            # 从回复中提取 JSON（Qwen 有时会额外加说明文字）
            content = content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            return json.loads(content)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Qwen response as JSON: {e}\nResponse: {content}")
            return None
        except Exception as e:
            logger.error(f"Qwen inference failed: {e}")
            return None

    async def generate_report(
        self,
        image_base64: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.3,
    ) -> Optional[str]:
        """
        调用 Qwen-VL 生成文本报告。

        Returns:
            报告文本，失败返回 None
        """
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}},
                        {"type": "text", "text": user_prompt},
                    ],
                },
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        try:
            r = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return None

    async def close(self):
        await self.client.aclose()
```

- [ ] **Step 2: 测试客户端可导入**

```bash
cd wafer-qwen-service
python -c "from core.model_manager import LlamaClient; print('LlamaClient OK')"
```

- [ ] **Step 3: Commit**

```bash
git add wafer-qwen-service/core/model_manager.py
git commit -m "feat(qwen): add LlamaClient HTTP client for llama.cpp server"
```

---

### Task 3: OpenCV 增强引擎 (enhancer.py)

**文件：**
- Create: `wafer-qwen-service/engine/enhancer.py`
- Create: `wafer-qwen-service/tests/test_enhancer.py`

**接口：**
- 消费: `QwenEnhanceRequest` 中的 `enhance_params`
- 产出: `enhance(image: np.ndarray, params: dict) -> np.ndarray`

- [ ] **Step 1: 创建 `wafer-qwen-service/engine/enhancer.py`**

```python
import cv2
import numpy as np


def enhance(image: np.ndarray, params: dict) -> np.ndarray:
    """
    根据 Qwen 分析的参数执行 OpenCV 增强流水线。

    Args:
        image: 输入灰度图 (H, W) 或彩色图 (H, W, 3)，uint8 [0,255]
        params: 增强参数字典，来自 Qwen 分析输出

    Returns:
        增强后的灰度图 (H, W), uint8 [0,255]
    """
    img = image.copy()

    # 转灰度
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. CLAHE 自适应直方图均衡
    clahe = cv2.createCLAHE(
        clipLimit=params.get("clahe_clip", 2.5),
        tileGridSize=(
            params.get("clahe_grid", 8),
            params.get("clahe_grid", 8),
        ),
    )
    img = clahe.apply(img)

    # 2. NLM 去噪（保留边缘）
    img = cv2.fastNlMeansDenoising(
        img, None,
        params.get("denoise_strength", 10),
        7, 21,
    )

    # 3. Gamma 校正（暗场提亮）
    gamma = params.get("gamma", 1.2)
    img = (img / 255.0) ** gamma * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 4. 对比度拉伸
    contrast = params.get("contrast", 1.3)
    img = cv2.multiply(img, contrast)
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 5. 锐化
    if params.get("sharpen", True):
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]], dtype=np.float32)
        img = cv2.filter2D(img, -1, kernel)

    return img.astype(np.uint8)


def auto_enhance(image: np.ndarray) -> np.ndarray:
    """
    无 Qwen 参数时的默认增强（用于快速测试）。
    """
    default_params = {
        "clahe_clip": 2.5,
        "clahe_grid": 8,
        "denoise_strength": 10,
        "gamma": 1.2,
        "contrast": 1.3,
        "sharpen": True,
    }
    return enhance(image, default_params)
```

- [ ] **Step 2: 创建测试 `wafer-qwen-service/tests/test_enhancer.py`**

```python
import cv2
import numpy as np

from engine.enhancer import enhance, auto_enhance


def test_enhance_output_shape():
    """增强输出应与输入尺寸一致"""
    dummy = np.random.randint(0, 256, (512, 512), dtype=np.uint8)
    result = auto_enhance(dummy)
    assert result.shape == (512, 512), f"Shape mismatch: {result.shape}"
    assert result.dtype == np.uint8


def test_enhance_with_params():
    """带自定义参数的增强"""
    dummy = np.ones((100, 100), dtype=np.uint8) * 50
    params = {
        "clahe_clip": 3.0,
        "clahe_grid": 4,
        "denoise_strength": 5,
        "gamma": 1.5,
        "contrast": 1.0,
        "sharpen": False,
    }
    result = enhance(dummy, params)
    assert result.shape == (100, 100)
    # Gamma 1.5 会提亮暗部
    assert result.mean() > 50


def test_enhance_color_input():
    """彩色输入应正确处理"""
    dummy = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    result = auto_enhance(dummy)
    assert len(result.shape) == 2  # 输出应为灰度


if __name__ == "__main__":
    test_enhance_output_shape()
    test_enhance_with_params()
    test_enhance_color_input()
    print("✅ All enhancer tests passed")
```

- [ ] **Step 3: 运行测试确认通过**

```bash
cd wafer-qwen-service
python tests/test_enhancer.py
```

预期输出：`✅ All enhancer tests passed`

- [ ] **Step 4: Commit**

```bash
git add wafer-qwen-service/engine/enhancer.py wafer-qwen-service/tests/
git commit -m "feat(qwen): add OpenCV enhancement engine with CLAHE+NLM+Gamma"
```

---

### Task 4: 缺陷检出引擎 (detector.py)

**文件：**
- Create: `wafer-qwen-service/engine/detector.py`
- Modify: `wafer-qwen-service/tests/test_detector.py`

**接口：**
- 消费: 增强图 `np.ndarray`, Qwen 输出的 `suspected_defects` 列表
- 产出: `detect_defects(enhanced_img, defects_spec) -> list[Detection]`

- [ ] **Step 1: 创建 `wafer-qwen-service/engine/detector.py`**

```python
import cv2
import numpy as np

from models.schemas import Detection, BBox

# Qwen 位置描述 → 归一化 ROI [x, y, w, h] 的映射
REGION_MAP = {
    "右上":      [0.65, 0.0, 0.35, 0.35],
    "右上角":    [0.65, 0.0, 0.35, 0.35],
    "左上":      [0.0, 0.0, 0.35, 0.35],
    "左上角":    [0.0, 0.0, 0.35, 0.35],
    "右下":      [0.65, 0.65, 0.35, 0.35],
    "右下角":    [0.65, 0.65, 0.35, 0.35],
    "左下":      [0.0, 0.65, 0.35, 0.35],
    "左下角":    [0.0, 0.65, 0.35, 0.35],
    "中心":      [0.3, 0.3, 0.4, 0.4],
    "中央":      [0.3, 0.3, 0.4, 0.4],
    "左侧":      [0.0, 0.2, 0.3, 0.6],
    "右侧":      [0.7, 0.2, 0.3, 0.6],
    "顶部":      [0.2, 0.0, 0.6, 0.3],
    "底部":      [0.2, 0.7, 0.6, 0.3],
    "边缘":      [0.0, 0.0, 1.0, 1.0],
    "上边缘":    [0.0, 0.0, 1.0, 0.15],
    "下边缘":    [0.0, 0.85, 1.0, 0.15],
    "左边缘":    [0.0, 0.0, 0.15, 1.0],
    "右边缘":    [0.85, 0.0, 0.15, 1.0],
}


def _resolve_roi(region_desc: str) -> list[float]:
    """将 Qwen 的自然语言位置描述转为归一化 ROI"""
    for keyword, roi in REGION_MAP.items():
        if keyword in region_desc:
            return roi
    return [0.0, 0.0, 1.0, 1.0]  # 默认全图


def detect_defects(
    enhanced_img: np.ndarray,
    suspected_defects: list[dict],
    min_area: int = 20,
    confidence_threshold: float = 0.5,
) -> list[Detection]:
    """
    在 Qwen 标注的疑似区域中，用形态学方法检出缺陷。

    Args:
        enhanced_img: 增强后的灰度图 (H, W)
        suspected_defects: Qwen 输出的缺陷列表
            [{"type": "...", "confidence": 0.85, "region": "位置描述"}, ...]
        min_area: 最小轮廓面积（过滤噪声）
        confidence_threshold: 最低置信度阈值

    Returns:
        Detection 列表
    """
    h, w = enhanced_img.shape
    results = []

    for spec in suspected_defects:
        defect_type = spec.get("type", "未知")
        confidence = spec.get("confidence", 0.6)
        region_desc = spec.get("region", "全图")

        if confidence < confidence_threshold:
            continue

        # ROI 坐标 (归一化 → 像素)
        roi_norm = _resolve_roi(region_desc)
        rx = int(roi_norm[0] * w)
        ry = int(roi_norm[1] * h)
        rw = int(roi_norm[2] * w)
        rh = int(roi_norm[3] * h)

        # 裁剪 ROI
        roi_img = enhanced_img[ry:ry + rh, rx:rx + rw]
        if roi_img.size == 0:
            continue

        # 自适应阈值
        thresh = cv2.adaptiveThreshold(
            roi_img, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2,
        )

        # 形态学开运算去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        # 轮廓查找
        contours, _ = cv2.findContours(
            cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            bx, by, bw, bh = cv2.boundingRect(cnt)
            # 将局部 ROI 坐标映射回全图坐标
            results.append(Detection(
                type=defect_type,
                confidence=round(min(confidence, 0.5 + area / 5000), 2),
                bbox=BBox(x=rx + bx, y=ry + by, w=bw, h=bh),
                source="qwen+cv",
            ))

    return results
```

- [ ] **Step 2: 创建测试 `wafer-qwen-service/tests/test_detector.py`**

```python
import numpy as np
from engine.detector import detect_defects
from models.schemas import Detection


def test_detect_defects_empty():
    """无缺陷时返回空列表"""
    img = np.ones((200, 200), dtype=np.uint8) * 128
    result = detect_defects(img, [], min_area=10)
    assert result == []


def test_detect_defects_with_spec():
    """有缺陷描述时返回检出结果"""
    img = np.ones((200, 200), dtype=np.uint8) * 128
    # 在右下角画一个暗色矩形模拟缺陷
    img[150:180, 150:180] = 30

    specs = [
        {"type": "颗粒", "confidence": 0.8, "region": "右下"},
    ]
    result = detect_defects(img, specs, min_area=10)
    assert len(result) > 0
    assert result[0].type == "颗粒"
    assert result[0].confidence > 0.5


def test_confidence_filter():
    """低于阈值的缺陷应过滤"""
    img = np.ones((100, 100), dtype=np.uint8) * 128
    specs = [
        {"type": "划痕", "confidence": 0.3, "region": "右侧"},
    ]
    result = detect_defects(img, specs, min_area=10, confidence_threshold=0.5)
    assert result == []


if __name__ == "__main__":
    test_detect_defects_empty()
    test_detect_defects_with_spec()
    test_confidence_filter()
    print("✅ All detector tests passed")
```

- [ ] **Step 3: 运行测试**

```bash
cd wafer-qwen-service
python tests/test_detector.py
```

预期输出：`✅ All detector tests passed`

- [ ] **Step 4: Commit**

```bash
git add wafer-qwen-service/engine/detector.py
git commit -m "feat(qwen): add Qwen-guided CV defect detector"
```

---

### Task 5: 渲染引擎 (visualizer.py)

**文件：**
- Create: `wafer-qwen-service/engine/visualizer.py`
- Modify: `wafer-qwen-service/tests/test_visualizer.py`

**接口：**
- 消费: `enhanced_img: np.ndarray`, `detections: list[Detection]`
- 产出: `draw_detections(image, detections) -> np.ndarray`

- [ ] **Step 1: 创建 `wafer-qwen-service/engine/visualizer.py`**

```python
import cv2
import numpy as np

from models.schemas import Detection

# YOLO 风格类别颜色 (BGR)
CLASS_COLORS = {
    "崩边": (0, 0, 255),       # 红色
    "颗粒": (255, 191, 0),     # 青色
    "划痕": (0, 255, 0),       # 绿色
    "位错": (255, 0, 255),     # 紫色
    "未知": (128, 128, 128),   # 灰色
}


def draw_detections(
    image: np.ndarray,
    detections: list[Detection],
) -> np.ndarray:
    """
    在增强图上绘制 YOLO 风格的检测框。

    Args:
        image: 灰度图或 BGR 图
        detections: Detection 列表

    Returns:
        BGR 彩色图（带检测框）
    """
    # 转 BGR
    if len(image.shape) == 2:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        img_rgb = image.copy()

    for det in detections:
        color = CLASS_COLORS.get(det.type, CLASS_COLORS["未知"])
        b = det.bbox

        # 画矩形框（2px 实线）
        cv2.rectangle(
            img_rgb,
            (b.x, b.y),
            (b.x + b.w, b.y + b.h),
            color, 2,
        )

        # 标签文字
        label = f"{det.type} {det.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)

        # 标签背景填充（半透明效果用矩形填充）
        label_bg_x1 = b.x
        label_bg_y1 = b.y - th - 6
        label_bg_x2 = b.x + tw + 6
        label_bg_y2 = b.y

        # 确保标签不超出图像上边界
        if label_bg_y1 < 0:
            label_bg_y1 = b.y
            label_bg_y2 = b.y + th + 6

        cv2.rectangle(
            img_rgb,
            (label_bg_x1, label_bg_y1),
            (label_bg_x2, label_bg_y2),
            color, -1,  # 填充
        )

        # 标签文字
        label_y = label_bg_y2 - 3 if label_bg_y1 < b.y else label_bg_y1 + th + 4
        cv2.putText(
            img_rgb, label,
            (b.x + 3, label_y),
            font, font_scale,
            (255, 255, 255), thickness, cv2.LINE_AA,
        )

    return img_rgb
```

- [ ] **Step 2: 创建测试**

```python
# tests/test_visualizer.py
import numpy as np
from engine.visualizer import draw_detections
from models.schemas import Detection, BBox


def test_draw_detections_grayscale_input():
    """灰度图输入应返回 BGR 彩色图"""
    img = np.ones((100, 100), dtype=np.uint8) * 128
    dets = [
        Detection(type="崩边", confidence=0.85,
                  bbox=BBox(x=10, y=10, w=20, h=15), source="qwen+cv"),
    ]
    result = draw_detections(img, dets)
    assert result.shape == (100, 100, 3)


def test_draw_detections_empty():
    """无检出时返回原始灰度转 BGR"""
    img = np.ones((50, 50), dtype=np.uint8) * 128
    result = draw_detections(img, [])
    assert result.shape == (50, 50, 3)


if __name__ == "__main__":
    test_draw_detections_grayscale_input()
    test_draw_detections_empty()
    print("✅ All visualizer tests passed")
```

- [ ] **Step 3: 运行测试**

```bash
cd wafer-qwen-service
python tests/test_visualizer.py
```

- [ ] **Step 4: Commit**

```bash
git add wafer-qwen-service/engine/visualizer.py
git commit -m "feat(qwen): add YOLO-style detection box renderer"
```

---

### Task 6: Prompt 模板 + 报告生成

**文件：**
- Create: `wafer-qwen-service/prompts/enhance.yaml`
- Create: `wafer-qwen-service/prompts/report.yaml`
- Create: `wafer-qwen-service/engine/reporter.py`

**接口：**
- 消费: `LlamaClient`, image_base64
- 产出: `generate_report(image_base64, detections) -> str`

- [ ] **Step 1: 创建 `wafer-qwen-service/prompts/enhance.yaml`**

```yaml
system: |
  你是半导体晶圆缺陷检测专家。
  分析这张暗场晶圆图像，输出 JSON 格式的分析结果。

user: |
  分析这张暗场晶圆图像，特别注意：
  1. 图像整体亮度、噪声水平、清晰度
  2. 是否存在疑似缺陷，包括崩边（边缘碎裂V形缺口）、颗粒（表面附着）、划痕（细长）、位错（晶格缺陷暗色线条）
  3. 缺陷的位置（用上下左右/中心/边缘+角描述）、大小（小<10px/中10-50px/大>50px）、形状

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
      "clahe_grid": 4|8,
      "denoise_strength": 5-20,
      "gamma": 0.8-2.0,
      "contrast": 1.0-2.0,
      "sharpen": true|false
    }
  }

  只输出 JSON，不要额外文字。
```

- [ ] **Step 2: 创建 `wafer-qwen-service/prompts/report.yaml`**

```yaml
system: |
  你是半导体晶圆缺陷检测质量分析专家。
  基于增强后的晶圆图像和已检出的缺陷信息，生成一份专业的检测报告。

user: |
  已检出以下缺陷：
  {% for d in detections %}
  - {{ d.type }}：置信度 {{ d.confidence }}，位于坐标 ({{ d.bbox.x }}, {{ d.bbox.y }})
  {% endfor %}

  请生成一份检测报告，包含：
  1. 图像质量评估（增强效果、清晰度等）
  2. 缺陷汇总（各类缺陷数量、位置分布）
  3. 重点关注（高置信度缺陷的处理建议）
  4. 总体结论（晶圆质量初步判断）

  报告要求：专业、简洁、中文。
```

- [ ] **Step 3: 创建 `wafer-qwen-service/engine/reporter.py`**

```python
import logging
from typing import Optional

from core.model_manager import LlamaClient
from models.schemas import Detection

logger = logging.getLogger(__name__)

# Prompts
ENHANCE_SYSTEM = """你是半导体晶圆缺陷检测专家。
分析这张暗场晶圆图像，输出 JSON 格式的分析结果。"""

ENHANCE_USER = """分析这张暗场晶圆图像，特别注意：
1. 图像整体亮度、噪声水平、清晰度
2. 是否存在疑似缺陷，包括崩边（边缘碎裂V形缺口）、颗粒（表面附着）、划痕（细长）、位错（晶格缺陷暗色线条）
3. 缺陷的位置（用上下左右/中心/边缘+角描述）、大小（小<10px/中10-50px/大>50px）、形状

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
    "clahe_grid": 4|8,
    "denoise_strength": 5-20,
    "gamma": 0.8-2.0,
    "contrast": 1.0-2.0,
    "sharpen": true|false
  }
}

只输出 JSON，不要额外文字。"""

REPORT_SYSTEM = """你是半导体晶圆缺陷检测质量分析专家。
基于增强后的晶圆图像和已检出的缺陷信息，生成一份专业的检测报告。"""


async def generate_report(
    llama_client: LlamaClient,
    image_base64: str,
    detections: list[Detection],
) -> Optional[str]:
    """生成缺陷分析报告"""
    # 构建包含检测结果的 prompt
    detection_text = "\n".join(
        f"- {d.type}：置信度 {d.confidence}，位于坐标 ({d.bbox.x}, {d.bbox.y})"
        for d in detections
    )

    report_user = f"""已检出以下缺陷：
{detection_text if detection_text else "（未检出明显缺陷）"}

请生成一份检测报告，包含：
1. 图像质量评估（增强效果、清晰度等）
2. 缺陷汇总（各类缺陷数量、位置分布）
3. 重点关注（高置信度缺陷的处理建议）
4. 总体结论（晶圆质量初步判断）

报告要求：专业、简洁、中文。"""

    return await llama_client.generate_report(
        image_base64=image_base64,
        system_prompt=REPORT_SYSTEM,
        user_prompt=report_user,
    )
```

- [ ] **Step 4: Commit**

```bash
git add wafer-qwen-service/prompts/ wafer-qwen-service/engine/reporter.py
git commit -m "feat(qwen): add enhance/detect/report prompts and reporter module"
```

---

### Task 7: API 路由 (routes.py)

**文件：**
- Create: `wafer-qwen-service/api/routes.py`

**接口：**
- 消费: `LlamaClient`, `enhancer`, `detector`, `visualizer`, `reporter`
- 产出: 三个 API 端点

- [ ] **Step 1: 创建 `wafer-qwen-service/api/routes.py`**

```python
import base64
import logging
import time

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException

from core.config import settings
from engine import enhancer, detector as det_mod, visualizer, reporter
from models.schemas import (
    Detection,
    EnhanceAnalysis,
    QwenEnhanceRequest,
    QwenEnhanceResponse,
    QwenReportRequest,
    QwenReportResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

# 从 main.py 引入全局 llama_client
import main as main_module


def _decode_image(image_b64: str) -> np.ndarray:
    """解码 base64 图片为 OpenCV 格式"""
    try:
        img_bytes = base64.b64decode(image_b64)
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Image decode failed")
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {e}")


def _encode_image(img: np.ndarray) -> str:
    """编码 OpenCV 图为 base64"""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf).decode("utf-8")


@router.post("/enhance", response_model=QwenEnhanceResponse)
async def enhance_endpoint(request: QwenEnhanceRequest):
    """
    核心端点：Qwen 分析 → OpenCV 增强 → 缺陷检出 → 渲染返回。
    """
    start = time.time()

    # 1. 解码图像
    img = _decode_image(request.image)
    img_b64 = request.image

    # 2. Qwen 分析
    llama = main_module.llama_client
    if llama is None:
        raise HTTPException(status_code=503, detail="Llama client not initialized")

    analysis_result = await llama.analyze_image(
        image_base64=img_b64,
        system_prompt=reporter.ENHANCE_SYSTEM,
        user_prompt=reporter.ENHANCE_USER,
    )

    if analysis_result is None:
        # Qwen 分析失败，使用默认参数
        logger.warning("Qwen analysis failed, using defaults")
        enhance_params = {
            "clahe_clip": settings.default_clahe_clip,
            "clahe_grid": settings.default_clahe_grid,
            "denoise_strength": settings.default_denoise,
            "gamma": settings.default_gamma,
            "contrast": settings.default_contrast,
            "sharpen": True,
        }
        suspected_defects = []
    else:
        enhance_params = analysis_result.get("enhance_params", {})
        suspected_defects = (
            analysis_result.get("analysis", {}).get("suspected_defects", [])
        )

    # 3. OpenCV 增强
    enhanced = enhancer.enhance(img, enhance_params)

    # 4. 缺陷检出
    detections = det_mod.detect_defects(
        enhanced,
        suspected_defects,
        min_area=settings.min_contour_area,
        confidence_threshold=settings.detection_confidence_threshold,
    )

    # 5. 渲染检测框
    result_img = visualizer.draw_detections(enhanced, detections)

    # 6. 编码返回
    enhanced_b64 = _encode_image(result_img)

    elapsed = int((time.time() - start) * 1000)
    logger.info(f"Enhance complete: {len(detections)} defects, {elapsed}ms")

    return QwenEnhanceResponse(
        success=True,
        enhanced_image=enhanced_b64,
        detections=detections,
        analysis=EnhanceAnalysis(
            brightness=analysis_result.get("analysis", {}).get("brightness", "中")
            if analysis_result else "中",
            noise_level=analysis_result.get("analysis", {}).get("noise_level", "中")
            if analysis_result else "中",
            sharpness=analysis_result.get("analysis", {}).get("sharpness", "清晰")
            if analysis_result else "清晰",
            defect_count=len(detections),
        ),
        inference_time_ms=elapsed,
    )


@router.post("/report", response_model=QwenReportResponse)
async def report_endpoint(request: QwenReportRequest):
    """基于增强图和检出结果生成分析报告"""
    start = time.time()

    llama = main_module.llama_client
    if llama is None:
        raise HTTPException(status_code=503, detail="Llama client not initialized")

    report_text = await reporter.generate_report(
        llama, request.image, request.detections,
    )

    elapsed = int((time.time() - start) * 1000)

    return QwenReportResponse(
        success=True,
        report=report_text or "报告生成失败",
        inference_time_ms=elapsed,
    )


@router.post("/analyze")
async def analyze_only(request: QwenEnhanceRequest):
    """仅做 Qwen 分析，不增强"""
    llama = main_module.llama_client
    if llama is None:
        raise HTTPException(status_code=503, detail="Llama client not initialized")

    result = await llama.analyze_image(
        image_base64=request.image,
        system_prompt=reporter.ENHANCE_SYSTEM,
        user_prompt=reporter.ENHANCE_USER,
    )

    return {
        "success": result is not None,
        "analysis": result,
    }
```

- [ ] **Step 2: 运行语法检查**

```bash
cd wafer-qwen-service
python -c "from api.routes import router; print('Routes OK')"
```

- [ ] **Step 3: Commit**

```bash
git add wafer-qwen-service/api/routes.py
git commit -m "feat(qwen): add enhance/report/analyze API endpoints"
```

---

### Task 8: Docker 部署配置

**文件：**
- Create: `wafer-qwen-service/Dockerfile`
- Modify: `docker-compose.yml`

- [ ] **Step 1: 创建 `wafer-qwen-service/Dockerfile`**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# OpenCV 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
```

- [ ] **Step 2: 在 `docker-compose.yml` 中新增两个服务**

```yaml
  # --- 在 lora-service 后面添加 ---

  wafer-qwen-service:
    build: ./wafer-qwen-service
    ports: ["8001:8001"]
    environment:
      QWEN_LLAMA_SERVER_URL: http://llama-server:8002/v1
      QWEN_HOST: "0.0.0.0"
      QWEN_PORT: 8001
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

- [ ] **Step 3: Commit**

```bash
git add wafer-qwen-service/Dockerfile docker-compose.yml
git commit -m "feat(qwen): add Docker build and compose integration"
```

---

### Task 9: Qwen GGUF 模型下载与验证

**无需写代码**，但需要在 WSL 中执行。

- [ ] **Step 1: 创建模型目录并下载**

```bash
# 创建模型目录（在 Windows 下，Docker 也能挂载到）
mkdir -p /mnt/d/models

# 方式一：从 HuggingFace 下载（如果网络通）
pip install huggingface-hub
huggingface-cli download \
    Qwen/Qwen2.5-VL-7B-Instruct-GGUF \
    qwen2.5-vl-7b-instruct-q4_k_m.gguf \
    --local-dir /mnt/d/models

# 方式二：从 ModelScope 国内源下载（推荐，速度快）
pip install modelscope
modelscope download \
    --model Qwen/Qwen2.5-VL-7B-Instruct-GGUF \
    --file qwen2.5-vl-7b-instruct-q4_k_m.gguf \
    --local_dir /mnt/d/models
```

- [ ] **Step 2: 验证模型文件**

```bash
ls -lh /mnt/d/models/qwen2.5-vl-7b-instruct-q4_k_m.gguf
# 预期：~4.6GB
```

---

### Task 10: Spring Boot 后端集成

**文件：**
- Create: `wafer-backend/src/main/java/com/wafer/client/QwenInferenceClient.java`
- Create: `wafer-backend/src/main/java/com/wafer/controller/QwenController.java`
- Create: `wafer-backend/src/main/java/com/wafer/model/dto/QwenEnhanceRequest.java`
- Create: `wafer-backend/src/main/java/com/wafer/model/dto/QwenEnhanceResponse.java`
- Modify: `wafer-backend/src/main/resources/application.yml`

**接口：**
- 消费: `wafer-qwen-service` 的 HTTP API
- 产出: Spring Boot 代理端点

- [ ] **Step 1: 创建 DTO**

```java
// QwenEnhanceRequest.java
package com.wafer.model.dto;

import lombok.Data;

@Data
public class QwenEnhanceRequest {
    private String image;     // base64
    private String format = "jpg";
}
```

```java
// QwenEnhanceResponse.java
package com.wafer.model.dto;

import lombok.Data;
import java.util.List;
import java.util.Map;

@Data
public class QwenEnhanceResponse {
    private boolean success;
    private String enhancedImage;  // base64
    private List<Map<String, Object>> detections;
    private Map<String, Object> analysis;
    private int inferenceTimeMs;
}
```

- [ ] **Step 2: 创建 Feign 客户端**

```java
// QwenInferenceClient.java
package com.wafer.client;

import com.wafer.model.dto.QwenEnhanceRequest;
import com.wafer.model.dto.QwenEnhanceResponse;
import org.springframework.cloud.openfeign.FeignClient;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;

import java.util.Map;

@FeignClient(
    name = "qwen-service",
    url = "${qwen.service.url}",
    path = "/api/v1/qwen"
)
public interface QwenInferenceClient {

    @PostMapping("/enhance")
    QwenEnhanceResponse enhance(@RequestBody QwenEnhanceRequest request);

    @PostMapping("/report")
    Map<String, Object> report(@RequestBody Map<String, Object> request);
}
```

- [ ] **Step 3: 创建控制器**

```java
// QwenController.java
package com.wafer.controller;

import com.wafer.client.QwenInferenceClient;
import com.wafer.model.dto.QwenEnhanceRequest;
import com.wafer.model.dto.QwenEnhanceResponse;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/v1/qwen")
@RequiredArgsConstructor
public class QwenController {

    private final QwenInferenceClient qwenClient;

    @PostMapping("/enhance")
    public ResponseEntity<QwenEnhanceResponse> enhance(
            @RequestBody QwenEnhanceRequest request) {
        QwenEnhanceResponse response = qwenClient.enhance(request);
        return ResponseEntity.ok(response);
    }

    @PostMapping("/report")
    public ResponseEntity<Map<String, Object>> report(
            @RequestBody Map<String, Object> request) {
        Map<String, Object> response = qwenClient.report(request);
        return ResponseEntity.ok(response);
    }
}
```

- [ ] **Step 4: 配置文件中添加 Qwen 服务地址**

```yaml
# wafer-backend/src/main/resources/application.yml
qwen:
  service:
    url: http://wafer-qwen-service:8001
```

- [ ] **Step 5: Commit**

```bash
git add wafer-backend/src/main/java/com/wafer/client/QwenInferenceClient.java \
      wafer-backend/src/main/java/com/wafer/controller/QwenController.java \
      wafer-backend/src/main/java/com/wafer/model/dto/QwenEnhance*.java \
      wafer-backend/src/main/resources/application.yml
git commit -m "feat(backend): add Qwen inference client and controller"
```

---

### Task 11: 前端 Qwen 检测展示页

**文件：**
- Create: `wafer-showcase/src/pages/QwenDetectPage.jsx`
- Create: `wafer-showcase/src/components/QwenImageUploader.jsx`
- Create: `wafer-showcase/src/components/QwenResultPanel.jsx`
- Create: `wafer-showcase/src/components/QwenReportPanel.jsx`
- Create: `wafer-showcase/src/api/waferApi.js`
- Modify: `wafer-showcase/src/App.jsx`

**接口：**
- 消费: `wafer-backend` 的 `/api/v1/qwen/enhance` 和 `/report`
- 产出: 完整的 Qwen 检测展示界面

- [ ] **Step 1: 创建 API 封装 `wafer-showcase/src/api/waferApi.js`**

```javascript
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8080/api/v1';

export async function qwenEnhance(imageBase64) {
  const { data } = await axios.post(`${API_BASE}/qwen/enhance`, {
    image: imageBase64,
    format: 'jpg',
  });
  return data;
}

export async function qwenReport(imageBase64, detections) {
  const { data } = await axios.post(`${API_BASE}/qwen/report`, {
    image: imageBase64,
    detections,
  });
  return data;
}
```

- [ ] **Step 2: 创建 `QwenImageUploader.jsx`**

```jsx
import { useState, useRef } from 'react'
import { Upload, X } from 'lucide-react'
import { cn } from '../utils/cn'

export default function QwenImageUploader({ onFileSelect, disabled }) {
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
            alt="暗场原图"
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
            上传暗场晶圆图像，或<strong className="text-tech-cyan">点击选择</strong>
          </p>
          <p className="text-zinc-600 text-sm">支持 JPG / PNG，最大 50MB</p>
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

- [ ] **Step 3: 创建 `QwenResultPanel.jsx`**

```jsx
import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { Shield, AlertTriangle, CheckCircle, Loader2 } from 'lucide-react'

const CLASS_COLORS = {
  '崩边': 'border-red-500 bg-red-500/10 text-red-400',
  '颗粒': 'border-cyan-500 bg-cyan-500/10 text-cyan-400',
  '划痕': 'border-green-500 bg-green-500/10 text-green-400',
  '位错': 'border-purple-500 bg-purple-500/10 text-purple-400',
}

export default function QwenResultPanel({
  originalImage,
  enhancedImage,
  detections,
  analysis,
  loading,
}) {
  if (loading) {
    return (
      <div className="flex items-center justify-center h-48 text-zinc-400">
        <Loader2 className="animate-spin mr-2" size={20} />
        <span>Qwen 正在分析中...</span>
      </div>
    )
  }

  if (!enhancedImage) return null

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* 左右对比图 */}
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <p className="text-sm text-zinc-400">暗场原图</p>
          <img
            src={originalImage}
            alt="原图"
            className="w-full rounded-lg border border-white/10"
          />
        </div>
        <div className="space-y-2">
          <p className="text-sm text-zinc-400">
            增强效果
            <span className="ml-2 text-xs text-tech-cyan">
              {analysis?.brightness === '高' ? '✓ 亮度充足' : ''}
            </span>
          </p>
          <img
            src={enhancedImage}
            alt="增强图"
            className="w-full rounded-lg border border-tech-cyan/30"
          />
        </div>
      </div>

      {/* 检测统计 */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {['崩边', '颗粒', '划痕', '位错'].map((type) => {
          const count = detections?.filter(d => d.type === type).length || 0
          return (
            <div key={type} className={`rounded-lg border p-3 ${CLASS_COLORS[type] || ''}`}>
              <p className="text-xs opacity-70">{type}</p>
              <p className="text-2xl font-bold">{count}</p>
            </div>
          )
        })}
      </div>

      {/* 缺陷列表 */}
      {detections?.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-zinc-300">
            <AlertTriangle size={14} className="inline mr-1" />
            检测详情
          </h4>
          <div className="space-y-1">
            {detections.map((d, i) => (
              <div key={i}
                className="flex items-center justify-between text-sm px-3 py-2 rounded-lg bg-white/5"
              >
                <span className={CLASS_COLORS[d.type]?.split(' ')[2] || 'text-zinc-300'}>
                  {d.type}
                </span>
                <span className="text-zinc-400">
                  置信度 {(d.confidence * 100).toFixed(0)}%
                </span>
                <span className="text-zinc-500 text-xs">
                  ({d.bbox.x}, {d.bbox.y})
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* 无缺陷 */}
      {detections?.length === 0 && (
        <div className="flex items-center gap-2 text-green-400 text-sm">
          <CheckCircle size={16} />
          未检出明显缺陷
        </div>
      )}
    </motion.div>
  )
}
```

- [ ] **Step 4: 创建 `QwenReportPanel.jsx`**

```jsx
import { useState } from 'react'
import { motion } from 'framer-motion'
import { FileText, Loader2 } from 'lucide-react'
import { qwenReport } from '../api/waferApi'

export default function QwenReportPanel({
  enhancedImage,
  detections,
}) {
  const [report, setReport] = useState(null)
  const [loading, setLoading] = useState(false)

  const generateReport = async () => {
    setLoading(true)
    try {
      const res = await qwenReport(enhancedImage, detections)
      setReport(res.report)
    } catch (err) {
      setReport('报告生成失败，请重试。')
    }
    setLoading(false)
  }

  return (
    <div className="space-y-4">
      <button
        onClick={generateReport}
        disabled={loading}
        className="flex items-center gap-2 px-4 py-2 rounded-lg
                   bg-tech-cyan/10 text-tech-cyan border border-tech-cyan/30
                   hover:bg-tech-cyan/20 transition-all disabled:opacity-50"
      >
        {loading ? (
          <Loader2 className="animate-spin" size={16} />
        ) : (
          <FileText size={16} />
        )}
        {loading ? 'Qwen 生成报告中...' : '生成 AI 分析报告'}
      </button>

      {report && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="p-4 rounded-xl bg-white/5 border border-white/10
                     prose prose-invert max-w-none text-sm leading-relaxed
                     whitespace-pre-wrap"
        >
          {report}
        </motion.div>
      )}
    </div>
  )
}
```

- [ ] **Step 5: 创建 `QwenDetectPage.jsx`**

```jsx
import { useState } from 'react'
import { motion } from 'framer-motion'
import { Sparkles, ChevronLeft, Info } from 'lucide-react'
import QwenImageUploader from '../components/QwenImageUploader'
import QwenResultPanel from '../components/QwenResultPanel'
import QwenReportPanel from '../components/QwenReportPanel'
import { qwenEnhance } from '../api/waferApi'

export default function QwenDetectPage() {
  const [file, setFile] = useState(null)
  const [originalDataUrl, setOriginalDataUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleFileSelect = (selectedFile) => {
    setFile(selectedFile)
    setResult(null)
    setError(null)
    if (selectedFile) {
      const reader = new FileReader()
      reader.onload = (e) => setOriginalDataUrl(e.target.result)
      reader.readAsDataURL(selectedFile)
    }
  }

  const handleEnhance = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const reader = new FileReader()
      reader.onload = async (e) => {
        const base64 = e.target.result.split(',')[1]
        const res = await qwenEnhance(base64)
        if (res.success) {
          setResult(res)
        } else {
          setError('增强分析失败')
        }
        setLoading(false)
      }
      reader.readAsDataURL(file)
    } catch (err) {
      setError(err.message || '请求失败')
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-tech-deep text-white pt-20 pb-16 px-4">
      <div className="max-w-5xl mx-auto">
        {/* 页头 */}
        <div className="flex items-center gap-3 mb-8">
          <Sparkles className="text-tech-cyan" size={28} />
          <div>
            <h1 className="text-2xl font-bold">AI 智能检测</h1>
            <p className="text-zinc-400 text-sm">
              Qwen-VL 驱动的晶圆缺陷增强与分析
            </p>
          </div>
        </div>

        {/* 说明横幅 */}
        <div className="flex items-start gap-3 p-4 mb-6 rounded-xl bg-tech-cyan/5 border border-tech-cyan/20 text-sm text-zinc-300">
          <Info size={18} className="text-tech-cyan shrink-0 mt-0.5" />
          <p>
            上传暗场晶圆图像，系统将使用 Qwen-VL 大模型分析图像特征，
            自动执行自适应增强、缺陷检测，并生成专业的检测报告。
            当前运行在 CPU 模式，推理时间约 10-30 秒。
          </p>
        </div>

        {/* 上传区 */}
        <div className="mb-6">
          <QwenImageUploader onFileSelect={handleFileSelect} disabled={loading} />
        </div>

        {/* 操作按钮 */}
        {file && !result && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="flex justify-center mb-8"
          >
            <button
              onClick={handleEnhance}
              disabled={loading}
              className="px-8 py-3 rounded-xl bg-tech-cyan text-black font-semibold
                         hover:bg-tech-cyan/90 transition-all disabled:opacity-50
                         disabled:cursor-not-allowed shadow-lg shadow-tech-cyan/20"
            >
              {loading ? 'Qwen 分析中...' : '开始 AI 增强分析'}
            </button>
          </motion.div>
        )}

        {/* 报错 */}
        {error && (
          <div className="p-4 mb-6 rounded-xl bg-red-500/10 border border-red-500/30 text-red-400 text-sm">
            {error}
          </div>
        )}

        {/* 结果展示 */}
        {result && (
          <div className="space-y-8">
            <QwenResultPanel
              originalImage={originalDataUrl}
              enhancedImage={`data:image/jpeg;base64,${result.enhancedImage}`}
              detections={result.detections}
              analysis={result.analysis}
              loading={false}
            />

            <div className="border-t border-white/10 pt-6">
              <QwenReportPanel
                enhancedImage={result.enhancedImage}
                detections={result.detections}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
```

- [ ] **Step 6: 修改 `App.jsx` 添加新页面路由**

```jsx
// 在 App.jsx 中新增 QwenDetectPage 导航入口
import QwenDetectPage from './pages/QwenDetectPage'

// 方案：在导航栏添加按钮，或通过 hash/状态切换
// 简单方式：在 HeroSection 后加一个导航条
```

对于不通前端路由的项目，可以在 `App.jsx` 添加状态切换：

```jsx
import { useState } from 'react'
import HeroSection from './components/HeroSection'
// ... existing imports
import QwenDetectPage from './pages/QwenDetectPage'

export default function App() {
  const [page, setPage] = useState('home')

  if (page === 'qwen') {
    return (
      <div className="relative bg-tech-deep text-white">
        <button
          onClick={() => setPage('home')}
          className="fixed top-4 left-4 z-50 px-4 py-2 rounded-lg
                     bg-white/10 text-white hover:bg-white/20 transition-all"
        >
          ← 返回首页
        </button>
        <QwenDetectPage />
      </div>
    )
  }

  return (
    <div className="relative bg-tech-deep text-white">
      <HeroSection onNavigateQwen={() => setPage('qwen')} />
      <ChallengesSection />
      <ArchitectureSection />
      <HighlightsSection />
      <MetricsSection />
      <FooterSection />
    </div>
  )
}
```

并在 `HeroSection` 中添加按钮：

```jsx
// HeroSection.jsx - 添加按钮
<button
  onClick={onNavigateQwen}
  className="px-6 py-3 rounded-xl bg-tech-cyan text-black font-semibold
             hover:bg-tech-cyan/90 transition-all"
>
  AI 智能检测 →
</button>
```

- [ ] **Step 7: Commit**

```bash
git add wafer-showcase/src/pages/QwenDetectPage.jsx \
      wafer-showcase/src/components/QwenImageUploader.jsx \
      wafer-showcase/src/components/QwenResultPanel.jsx \
      wafer-showcase/src/components/QwenReportPanel.jsx \
      wafer-showcase/src/api/waferApi.js \
      wafer-showcase/src/App.jsx
git commit -m "feat(showcase): add Qwen-VL detection page with enhance/report UI"
```

---

### Task 12: 端到端集成验证

- [ ] **Step 1: 启动 llama-server**

```bash
# 方法一：Docker
cd /d/cy
docker compose up -d llama-server

# 方法二：直接在 WSL 运行（更快）
cd /mnt/d/cy
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
mkdir build && cd build
cmake .. -DCMAKE_C_FLAGS="-march=native" -DCMAKE_CXX_FLAGS="-march=native"
make -j16
./llama-server \
    --model /mnt/d/models/qwen2.5-vl-7b-instruct-q4_k_m.gguf \
    --port 8002 \
    --ctx-size 8192 \
    --batch-size 512
```

- [ ] **Step 2: 启动 wafer-qwen-service**

```bash
cd /mnt/d/cy/wafer-qwen-service
pip install -r requirements.txt
python main.py
# 启动在 8001 端口
```

- [ ] **Step 3: 测试健康检查**

```bash
curl http://localhost:8001/api/v1/health
# 预期：{"status":"UP","llama_available":true,"device":"cpu"}
```

- [ ] **Step 4: 用一张测试图做增强**

```bash
# 使用 img1 中的暗场图像测试
python -c "
import base64, requests
with open('/mnt/d/cy/img1/080dda8f-f035-4c90-84b9-f656b591cbf1.jpg', 'rb') as f:
    b64 = base64.b64encode(f.read()).decode()
r = requests.post('http://localhost:8001/api/v1/qwen/enhance',
    json={'image': b64, 'format': 'jpg'})
print('Status:', r.status_code)
if r.status_code == 200:
    data = r.json()
    print('Detections:', len(data['detections']))
    print('Time:', data['inference_time_ms'], 'ms')
else:
    print('Error:', r.text)
"
```

- [ ] **Step 5: 启动完整 Docker 栈做集成测试**

```bash
docker compose up -d
# 验证 qwen 服务
curl http://localhost:8001/api/v1/health
# 验证 Spring Boot 代理
curl http://localhost:8080/api/v1/qwen/enhance -X POST ...
```
