# Qwen-VL 晶圆检测 Phase A — 会话记录

> **日期**: 2026-07-18  
> **项目路径**: `D:/cy/`  
> **核心分支**: master (root), main (wafer-showcase)  
> **设计文档**: `docs/qwen-vl-wafer-detection-design.md`  
> **实施计划**: `docs/superpowers/plans/2026-07-18-qwen-vl-wafer-phase-a.md`  

---

## 一、项目概述

用 Qwen-VL 大模型 + OpenCV 替代原有的 ARM 边缘部署方案，实现晶圆暗场→亮场增强、圆形缺陷检测、多角度散射视图。

**架构**: 用户上传暗场图 → OpenCV 增强 → 霍夫圆+Blob 检测 → YOLO 框渲染 → Qwen 文字分析 → 极坐标多角度视图

---

## 二、当前服务状态（运行中）

| 服务 | 端口 | 启动方式 | 状态 |
|------|------|---------|------|
| **frontend** (Vite) | 5173 | PowerShell 隐藏窗口 | 🟢 |
| **wafer-qwen-service** (FastAPI) | 8001 | PowerShell 隐藏窗口 | 🟢 |
| **llama-server** (llama.cpp b10068) | 8002 | `start-llama.vbs` (原生Windows, 支持视觉) | 🟢 |
| wafer-backend (Spring Boot) | 8080 | ⛔ 未启动 | 🔴 |

### 浏览器访问
```
http://localhost:5173  → 首页 → 页脚「AI 智能检测」→ 检测页面
```

### API 接口 (8001)
```
POST /api/v1/qwen/enhance   — 增强 + 圆形缺陷检测
POST /api/v1/qwen/report    — Qwen 生成分析报告
POST /api/v1/qwen/angles    — 13角度193nm亮场视图
POST /api/v1/qwen/analyze   — 仅分析
GET  /api/v1/health         — 健康检查
```

---

## 三、代码仓库与提交

### Root repo (D:/cy) — 7 commits
```
e3260b4 feat(qwen): brightfield preserve morphology, angle views with detections
905054a feat(qwen): circular defect detection + Qwen text analysis
96d21ee feat(qwen): add Docker deployment for Qwen services
27b8edf feat(qwen): add enhance/report/analyze API endpoints
6e0c3f8 feat(qwen): add Qwen prompts and report generator
da877ae feat(qwen): add YOLO-style detection box renderer with tests
20c974f feat(qwen): add Qwen-guided CV defect detector with tests
01762cb feat(qwen): add OpenCV enhancement engine with tests
fcaedf5 feat(qwen): add LlamaClient HTTP client for llama.cpp
2b42fae feat(qwen): scaffold wafer-qwen-service FastAPI project
```

### wafer-showcase repo — 5 commits
```
878a5ef feat: add AngleSlider, fix Chinese type key matching
fa7135e fix: update UI text to reflect actual tech pipeline
b71303a fix: strip data URL prefix before sending to API
eedd749 feat: add bright field style conversion with 3-column comparison
befa6a8 feat: move AI detection button to footer section
ac9b084 feat(qwen): add Qwen-VL detection page with enhance/report UI
```

### wafer-backend repo — 1 commit
```
efab777 feat(qwen): add Qwen inference Feign client and controller
```

---

## 四、wafer-qwen-service 完整结构

```
wafer-qwen-service/
├── main.py                      # FastAPI 入口 (GlobalConfig → LlamaClient)
├── requirements.txt             # Python 依赖
├── Dockerfile                   # Docker 构建
├── core/
│   ├── config.py                # Settings (QWEN_ 环境变量前缀)
│   └── model_manager.py         # LlamaClient: llama.cpp HTTP 客户端
│                                #   - analyze_image(): Qwen-VL 看图 JSON 分析
│                                #   - generate_text(): 纯文本聊天
│                                #   - generate_report(): VL 报告
│                                #   - check_health()
├── engine/
│   ├── enhancer.py              # 增强引擎
│   │   ├── enhance()            # 暗场增强 (CLAHE+NLM+Gamma+锐化)
│   │   ├── auto_enhance()       # 默认增强
│   │   └── brightfield_enhance() # 亮场风格: 反转+百分位拉伸, 保持形态学
│   ├── detector.py              # 缺陷检测
│   │   ├── detect_circular_defects()  # 霍夫圆 + SimpleBlobDetector
│   │   │   - is_brightfield 参数切换亮暗场检测模式
│   │   │   - 亮场: 先反转再检测, blobColor=0
│   │   └── analyze_enhanced_image()   # 图像特征分析
│   ├── visualizer.py            # YOLO 风格框渲染 (中文类别+颜色)
│   ├── angle_generator.py       # 多角度193nm视图
│   │   └── generate_angle_views()
│   │       极坐标展开 → torch.roll → 反变换
│   │       15°~75°, 步长5°, 13张
│   └── reporter.py              # 报告生成 (结构化模板)
├── api/
│   └── routes.py                # 5 个端点
├── models/
│   └── schemas.py               # Pydantic 模型
├── prompts/
│   ├── enhance.yaml             # 分析 prompt
│   └── report.yaml              # 报告 prompt
└── tests/
    ├── test_enhancer.py
    ├── test_detector.py
    └── test_visualizer.py
```

---

## 五、核心流水线详解

### 5.1 亮场风格转化 (`enhancer.py → brightfield_enhance`)
```
暗场原图 → 轻度CLAHE(clipLimit=1.5) → Gamma 1.1 → 反转
→ 百分位拉伸(2%-98%) → 形态学闭运算(2x2) → 锐化
```
特点：保持原始形态学结构（不 NLM 去噪），背景→纯白255，缺陷→纯黑0

### 5.2 圆形缺陷检测 (`detector.py → detect_circular_defects`)
```
方法1: 霍夫圆检测 (HOUGH_GRADIENT, dp=1.2, minDist=20, param1=50, param2=25)
  - 亮场下先翻转图像再检测
  - 半径分类: <8px=颗粒, 8-20px=颗粒/位错/崩边, >20px=位错/崩边/划痕
  - 置信度 = contrast/80

方法2: SimpleBlobDetector (补充小圆形)
  - filterByCircularity=0.5, filterByColor
  - 亮场 blobColor=0 (暗色), 暗场 blobColor=255 (亮色)

去重后取 top20
```

### 5.3 多角度视图 (`angle_generator.py → generate_angle_views`)
```
极坐标展开(INTER_NEAREST保持边缘)
→ 13个角度滚动: 15°~75° 步长5°
→ 散射增益: 0.9~1.1
→ 反变换回笛卡尔(INTER_NEAREST)
→ 每张图独立做圆形检测+YOLO框
```

### 5.4 前端展示
```
FooterSection → 点击「AI 智能检测」→ QwenDetectPage
  3栏对比: 暗场原图 | 亮场转化(含检测框) | 亮场参考(img2)
  缺陷统计: 崩边/颗粒/划痕/位错 数量
  检测明细: 类型+置信度+坐标+尺寸 表格
  角度滑块: 15°~75° 滚轮切换13个角度视图
  AI报告: 结构化报告面板
```

---

## 六、已知问题

| 问题 | 原因 | 状态 |
|------|------|------|
| Qwen-VL 看图分析可用 ✅ | llama.cpp 升级至 b10068 (原生Windows, --mmproj) | <2026-07-19> |
| ~~Qwen-VL 看图分析不可用~~ | ~~llama.cpp b4771 不支持 server API 传图, 需 b10066+~~ | ~~⏳~~ |
| Qwen 文本报告生成质量差 | 3B Q4 CPU 中文重复问题, 已改为模板 | ⚠️ 临时方案 |
| 前端直接连 8001 | 跳过了 Spring Boot 代理 (8001→8080) | ⚠️ 开发模式 |
| wafer-backend 未启动 | 需要 Docker/Direct 启动 | 🔴 |
| 只有3张测试图(img1) | 缺少更多缺陷样本 | ⏳ |

---

## 七、下次启动步骤

### 7.1 启动服务 (按顺序)
```bash
# 1. llama-server b10068 (原生Windows, 支持视觉)
cscript //nologo "D:\cy\start-llama.vbs"
# 或手动: "D:\cy\llama-b10068\llama-server.exe" -m "D:\models\Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf" --mmproj "D:\models\mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf" --port 8002 --host 0.0.0.0 --ctx-size 4096 --batch-size 256 --n-gpu-layers 0
# 等待模型加载 (~30-45秒), 验证: curl http://localhost:8002/health

# 2. wafer-qwen-service (Python 3.12)
QWEN_LLAMA_SERVER_URL=http://localhost:8002/v1 /c/Users/1/AppData/Local/Programs/Python/Python312/python -m uvicorn main:app --host 0.0.0.0 --port 8001
# 验证: curl http://localhost:8001/api/v1/health

# 3. 前端 (wafer-showcase)
cd /d/cy/wafer-showcase
npm run dev
# 浏览器: http://localhost:5173
```

### 7.2 关键环境变量
```
QWEN_LLAMA_SERVER_URL=http://localhost:8002/v1
QWEN_HOST=0.0.0.0
QWEN_PORT=8001
```

### 7.3 模型文件和工具
```
llama.cpp:   /d/cy/llama-b10068/llama-server.exe (b10068, 原生Windows CPU)
模型:        /d/models/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf (~1.8GB)
mmproj:      /d/models/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf (~806MB)
参考图:      /d/cy/img2/12ea67aa-eac2-452e-9c37-43efe3114581.png (2048x1785)
测试图:      /d/cy/img1/*.jpg (3张暗场)
```

---

## 八、后续改进方向

1. **升级 llama.cpp** ✅ → 已升级至 b10068 (`D:\\cy\\llama-b10068\\llama-server.exe`), Qwen-VL 视觉分析已启用
2. **Qwen 文字分析接入** → 已改用 vision-based analyze (`reporter.py`), 给 Qwen 传递图像+缺陷统计
3. **更多测试图像** → 收集包含真实缺陷的暗场图
4. **Spring Boot 集成** → 启动 wafer-backend, 统一走 8080 代理
5. **亮场参考图自动对齐** → 根据上传图像尺寸自动 resize 参考图
6. **Phase B: 轻量 CNN 增强** → 训练专用增强模型替代 OpenCV
