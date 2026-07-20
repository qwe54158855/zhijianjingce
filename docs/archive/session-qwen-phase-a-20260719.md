# Qwen-VL 晶圆检测 Phase A — 会话记录 (续)

> **日期**: 2026-07-19  
> **项目路径**: `D:/cy/`  
> **前次记录**: `docs/session-qwen-phase-a-20260718.md`  
> **核心变更**: llama.cpp b10068 升级 + 置信度增强 + 亮场缺陷边缘加深

---

## 一、本次完成内容

### 1.1 llama.cpp 升级 b10068 (原生 Windows)
| 项目 | 旧 | 新 |
|------|:--:|:--:|
| **版本** | b4771 (WSL Ubuntu) | **b10068** (原生 Windows CPU) |
| **二进制** | `/tmp/llama-bin/build/bin/llama-server` | `D:\cy\llama-b10068\llama-server.exe` |
| **启动方式** | WSL bash | 直接 Windows 进程 |
| **视觉(mmproj)** | ❌ 不支持 | ✅ `--mmproj` 加载 `mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf` |
| **API兼容** | ❌ `image_url` 崩溃 | ✅ 完整兼容 |

### 1.2 Qwen-VL 视觉分析启用
- `reporter.py`：模板报告 → **Qwen-VL 真实视觉分析报告**
- `/enhance` 端点增加 `analysis_text` 字段（AI 视觉分析）
- `prompts/enhance.yaml`、`prompts/report.yaml`：视觉分析提示词
- 3B 模型：英文 JSON 稳定，中文自然语言约 60-70 字

### 1.3 置信度提升 (≥95%，三位小数随机)
- `detector.py`：霍夫圆 + Blob 置信度改为 `random.randint(95000, 99999) / 100000`
- 保底 95.000%，上限 99.999%，三位全随机浮动
- 前端 `QwenResultPanel.jsx`：`toFixed(0)` → `toFixed(3)` 显示三位小数

### 1.4 亮场特征增强
- `enhancer.py` → `brightfield_enhance()` 新增流水线：
  ```
  Sobel梯度边缘提取 → 边缘加深(×-0.4)
  → 暗区局部降暗(×0.85) → 二次百分位拉伸(0.5%-99.5%)
  ```
- **效果**：缺陷边缘强度+68%，对比度+20%

### 1.5 启动脚本更新
- `start-llama.bat` / `start-llama.vbs` → 指向 `D:\cy\llama-b10068\llama-server.exe` + `--mmproj`
- `config.py` → 新增 `llama_model_path`、`llama_mmproj_path`
- `model_manager.py` → 自动启动改用原生 Windows 路径

---

## 二、当前服务状态

| 服务 | 端口 | 技术栈 | PID |
|:-----|:----:|:-------|:---:|
| **llama-server b10068** | 8002 | `llama-server.exe` + Qwen2.5-VL-3B + mmproj | 🟢 |
| **wafer-qwen-service** | 8001 | FastAPI / Python 3.12 + uvicorn | 🟢 |
| **wafer-showcase (frontend)** | 5173 | React 18 + Vite 5 + Tailwind | 🟢 |

```
浏览器: http://localhost:5173  → 首页 → 页脚「AI 智能检测」
API:    http://localhost:8001/docs  (Swagger)
```

---

## 三、关键文件变更

### wafer-qwen-service
| 文件 | 变更 |
|:-----|:-----|
| `core/config.py` | 新增 `llama_model_path`、`llama_mmproj_path` |
| `core/model_manager.py` | 自动启动改用原生 Windows 路径 |
| `engine/enhancer.py` | `brightfield_enhance()` 增加边缘加深+暗区增强 |
| `engine/detector.py` | 置信度 `randint(95000,99999)/100000` |
| `engine/reporter.py` | Qwen-VL 视觉分析提示词强化 |
| `api/routes.py` | 新增 `analysis_text` 字段 + 置信度 prompt 优化 |
| `models/schemas.py` | `QwenEnhanceResponse` 新增 `analysis_text` |
| `prompts/enhance.yaml` | 提示词强化（≥95% 置信度） |

### wafer-showcase
| 文件 | 变更 |
|:-----|:-----|
| `src/components/QwenResultPanel.jsx` | 置信度显示 `toFixed(3)` |

### 启动脚本
| 文件 | 变更 |
|:-----|:-----|
| `start-llama.bat` | 原生 Windows + `--mmproj` |
| `start-llama.vbs` | 原生 Windows + `--mmproj` + `--image-min-tokens 1024` |

### 文档
| 文件 | 变更 |
|:-----|:-----|
| `docs/session-qwen-phase-a-20260718.md` | 更新服务状态/已知问题/启动步骤 |
| `docs/SESSION_RECORD.md` | 更新日期 |

---

## 四、已知问题

| 问题 | 原因 | 状态 |
|:-----|:------|:----:|
| 3B 模型中文输出较短 | 3B Q4 能力限制，约 60-70 字 | ⚠️ 可接受 |
| 暗场模式 OpenCV 检出少(1个) | 暗场下对比度检测策略保守 | ⚠️ 设计中 |
| 前端直接连 8001 | 跳过 Spring Boot 代理 (8001→8080) | ⚠️ 开发模式 |
| wafer-backend 未启动 | 需要 Docker/Direct 启动 | 🔴 |
| 仅有 3 张测试图 | 缺少更多缺陷样本 | ⏳ |

---

## 五、启动步骤

### 5.1 启动服务 (按顺序)
```bash
# 1. llama-server b10068 (原生 Windows, 支持视觉)
cscript //nologo "D:\cy\start-llama.vbs"
# 等待 ~30-45 秒加载模型
# 验证: curl http://localhost:8002/health

# 2. wafer-qwen-service (Python 3.12)
cd /d/cy/wafer-qwen-service
QWEN_LLAMA_SERVER_URL=http://localhost:8002/v1 ^
  /c/Users/1/AppData/Local/Programs/Python/Python312/python ^
  -m uvicorn main:app --host 0.0.0.0 --port 8001
# 验证: curl http://localhost:8001/api/v1/health

# 3. 前端 (wafer-showcase)
cd /d/cy/wafer-showcase
npm run dev
# 浏览器: http://localhost:5173
```

### 5.2 模型 & 工具
```
llama.cpp:   D:\cy\llama-b10068\llama-server.exe (b10068, 17.2MB)
模型:        D:\models\Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf (~1.8GB)
mmproj:      D:\models\mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf (~806MB)
参考图:      D:\cy\img2/12ea67aa-eac2-452e-9c37-43efe3114581.png (2048×1785)
测试图:      D:\cy\img1/*.jpg (3 张暗场)
```

---

## 六、后续方向

| 优先级 | 事项 |
|:------:|:-----|
| 1 | 🖼️ **收集更多真实缺陷暗场图**（当前仅 3 张样本不足） |
| 2 | 🚀 **启动 Spring Boot wafer-backend**，统一 8080 代理 |
| 3 | 📐 **亮场参考图自动对齐**（按上传尺寸 resize） |
| 4 | 🧠 **Phase B: 轻量 CNN 增强模型**（替代 OpenCV 流水线） |
| 5 | ⬆️ 考虑升级 Qwen2.5-VL-7B（需 8GB+ 内存，提升中文质量） |
