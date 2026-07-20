# 开源打包实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 D:/cy 全部内容压缩重构，补充开源资产，准备 GitHub 发布

**Architecture:** 单仓库 Monorepo，docs/ 全面压缩去重，根目录新增 LICENSE/CONTRIBUTING/CODE_OF_CONDUCT/SECURITY/CHANGELOG + GitHub 模板 + 服务 README

**Tech Stack:** Markdown, git

## Global Constraints

- 中英双语：核心文档（README、CONTRIBUTING）有 EN/ZH 版本
- 保留原始版本：COMPLETE_TECHNICAL_REFERENCE.md 等保留在原位置，压缩版放 docs/ 下
- MIT 许可证
- 所有文档在 `D:\cy\` 根目录

---

### Task 1: 创建目录结构

**Files:**
- Create: `D:\cy\docs\archive\.gitkeep`
- Create: `D:\cy\.github\ISSUE_TEMPLATE\.gitkeep`
- Create: `D:\cy\.github\PULL_REQUEST_TEMPLATE.md`

- [x] **Step 1: 创建目录**

```bash
mkdir -p /d/cy/docs/archive
mkdir -p /d/cy/.github/ISSUE_TEMPLATE
touch /d/cy/docs/archive/.gitkeep
```

---

### Task 2: 压缩 COMPLETE_TECHNICAL_REFERENCE → docs/technical-reference.md

**Files:**
- Create: `D:\cy\docs\technical-reference.md`
- Reference: `D:\cy\docs\COMPLETE_TECHNICAL_REFERENCE.md`

从 52KB → ~30KB，去掉冗余描述、保留关键架构图和表格。

- [ ] **Step 1: 写压缩版技术参考**

内容要点（中英文双语标题 + 中文正文）：
- 项目定位（一句话）
- 三大核心难题（精简至每段 1-2 句话）
- 一骨干双分支架构图 + 参数预算表
- RepViT 骨干（关键选择理由表 + 三层结构）
- 增强解码分支（架构图 + MSEFBlock + Denoiser 核心设计）
- 检测输出分支（Anchor-free 设计 + 四项精简优化 + 缺陷类别表）
- 光学物理注入 v2.0 概览（五模块表格 + PISM 三步流程 + ASG/双光路/SGF 一句话介绍）
- 三阶段训练策略（流程图 + 损失权重表）
- 模型压缩与 JIT 导出（融合顺序 + INT8 配置 + 校验标准）
- LibTorch C++ SDK（类设计 + C API + 交叉编译）
- 晶圆优化（极坐标展开 + 在线校准）
- 验收标准（P0/P1 表格 + 四阶段流程）
- 风险矩阵（精简关键行）
- 端到端流水线图 + 延迟分解

- [ ] **Step 2: 写入文件**

---

### Task 3: 合并压缩光学物理文档 → docs/optical-physics.md

**Files:**
- Create: `D:\cy\docs\optical-physics.md`
- Reference: `D:\cy\docs\optical-physics-introduction.md`, `D:\cy\docs\optical-physics-technical-deepdive.md`

合并两个 physics 文档（31KB + 54KB）→ ~15KB。

- [ ] **Step 1: 写光学物理专题**

内容要点：
- 为什么需要光学物理（v1.0 四个问题 + 核心洞察：用算法代偿硬件）
- 物理基础（Rayleigh 散射公式 + Mie 散射公式 + Rayleigh-Mie 混合模型 + 各向异性散射表）
- 极坐标变换原理（同构：方位角旋转 = 特征图水平平移）
- PISM 可微散射物理（三步流程 + Scale Estimator + 增益计算 + 残差补偿 + 类比解释）
- ASG 多方位角度生成器（三层结构 + torch.roll 零成本实现）
- 双光路并行检测（权重不共享原因 + F2 全通道物理依据 + batch 13 优化）
- SGF + 角度注意力融合（w_193 计算 + 完整融合流程）
- 物理约束损失函数（L_scat + L_spec + 三阶段权重表）
- 实现状态与代码结构

---

### Task 4: 压缩 FOR_PROJECT_MANAGER → docs/project-manager-guide.md

**Files:**
- Create: `D:\cy\docs\project-manager-guide.md`
- Reference: `D:\cy\docs\FOR_PROJECT_MANAGER.md`

从 71KB → ~20KB。

- [ ] **Step 1: 写精简版项目导读**

内容要点：
- 一句话概括（保留原始）
- 三个现实问题（每个 1-2 段，保留核心类比）
- 总体架构（保留架构图）
- 技术决策问答形式（为什么 RepViT、为什么共享编码器等）
- 验收标准（白话版表格）
- 风险与应对（精简矩阵）

---

### Task 5: 精简 Qwen / Showcase / 竞赛文档

**Files:**
- Create: `D:\cy\docs\qwen-vl-detection.md` (Move + compress)
- Create: `D:\cy\docs\showcase-design.md` (Move + compress)
- Keep: `D:\cy\docs\competition-guide.md` (Rename from competition-paper-tech-revision-guide.md)
- Source: `D:\cy\docs\qwen-vl-wafer-detection-design.md`, `D:\cy\docs\wafer-showcase-design.md`, `D:\cy\docs\competition-paper-tech-revision-guide.md`

- [ ] **Step 1: 精简 Qwen 方案文档（16KB → 8KB）**

保留核心：方案演进路线图、Phase A/B 对比表、系统架构、Qwen-VL 的独特价值

- [ ] **Step 2: 精简 Showcase 设计文档（6.5KB → 4KB）**

保留核心：色板、字体、页面结构模块清单、组件文件结构

- [ ] **Step 3: 保留竞赛文档原内容，改名为 competition-guide.md**

---

### Task 6: 创建部署指南

**Files:**
- Create: `D:\cy\docs\deployment-guide.md`
- Reference: `D:\cy\docker-compose.yml`, `D:\cy\nginx.conf`, `D:\cy\OPERATION_GUIDE.md`

- [ ] **Step 1: 写部署指南**

内容要点：
- 前置要求（Docker, JDK, Node.js, GPU）
- Docker Compose 全栈启动（services 概览表）
- 环境变量表（合并所有服务）
- 本地开发模式（无 Docker）
- 生产部署建议
- 运维检查（健康检查端点、日志查看、数据备份）

---

### Task 7: 重写根 README.md（双语）

**Files:**
- Modify: `D:\cy\README.md`

当前 README 只有英文内容。重写为中英双语。

- [ ] **Step 1: 写新 README**

```
# Wafer Inspection — 半导体晶圆缺陷检测一体化方案

[![MIT License](badge)]
[![Docker](badge)]

(EN) An integrated AI solution for semiconductor wafer edge defect detection...
(ZH) 一体化半导体晶圆边缘缺陷检测 AI 解决方案...

## Architecture 架构

## Quick Start 快速开始

## Services Overview 服务概览

## Documentation 文档索引

## Contributing 贡献指南

## License 许可证
```

---

### Task 8: 创建 LICENSE / CONTRIBUTING / CODE_OF_CONDUCT / SECURITY / CHANGELOG

**Files:**
- Create: `D:\cy\LICENSE`
- Create: `D:\cy\CONTRIBUTING.md`
- Create: `D:\cy\CODE_OF_CONDUCT.md`
- Create: `D:\cy\SECURITY.md`
- Create: `D:\cy\CHANGELOG.md`

- [ ] **Step 1: LICENSE (MIT)**

```markdown
MIT License

Copyright (c) 2026 Wafer Inspection Team

Permission is hereby granted...
```

- [ ] **Step 2: CONTRIBUTING.md**（中英双语）

内容：Fork 工作流 / 开发环境 / 代码风格 / PR 流程 / 测试要求

- [ ] **Step 3: CODE_OF_CONDUCT.md**（标准 Contributor Covenant）

- [ ] **Step 4: SECURITY.md**

报告安全问题的流程、PGP 加密首选、响应时间

- [ ] **Step 5: CHANGELOG.md**

基于 git log 提取的主要版本日志

---

### Task 9: 创建 GitHub 模板

**Files:**
- Create: `D:\cy\.github\ISSUE_TEMPLATE\bug_report.md`
- Create: `D:\cy\.github\ISSUE_TEMPLATE\feature_request.md`
- Create: `D:\cy\.github\PULL_REQUEST_TEMPLATE.md`

- [ ] **Step 1: Bug report 模板**
- [ ] **Step 2: Feature request 模板**
- [ ] **Step 3: PR 模板**

---

### Task 10: 创建 5 个服务 README

**Files:**
- Create: `D:\cy\wafer-inspection\README.md`
- Create: `D:\cy\wafer-showcase\README.md`
- Create: `D:\cy\wafer-qwen-service\README.md`
- Modify: `D:\cy\wafer-lora-service\README.md`（完善现有）
- Create: `D:\cy\wafer-backend\README.md`（Spring Boot，用 POM 信息）

每个 README 包含：一句话定位、技术栈、API（如有）、本地开发、Docker 构建

- [ ] **Step 1: wafer-inspection README**（PyTorch 模型核心）
- [ ] **Step 2: wafer-backend README**（Spring Boot API）
- [ ] **Step 3: wafer-showcase README**（React 前端）
- [ ] **Step 4: wafer-qwen-service README**（FastAPI + Qwen-VL）
- [ ] **Step 5: wafer-lora-service README**（完善现有）

---

### Task 11: 创建 .gitignore + docs/README.md 索引

**Files:**
- Create: `D:\cy\.gitignore`
- Create: `D:\cy\docs\README.md`

- [ ] **Step 1: 根 .gitignore**

```gitignore
# Python
__pycache__/
*.pyc
.venv/
*.egg-info/

# Node
node_modules/
dist/

# Java
target/
*.class
*.jar
!mvnw

# IDE
.idea/
.vscode/
*.swp

# OS
Thumbs.db
.DS_Store

# Docker
.env

# Models
*.gguf
*.pt
*.pth
*.safetensors
!**/*.py

# Data
*.jpg
*.png
!result*.jpg
!result*.png

# Logs
*.log
```

- [ ] **Step 2: docs/README.md**

文档目录索引，双语标题，每个文档一句话说明用途

---

### Task 12: 归档会话记录 + 移除大附件

**Files:**
- Move: `D:\cy\docs\SESSION_RECORD.md` → `D:\cy\docs\archive\SESSION_RECORD.md`
- Move: `D:\cy\docs\session-qwen-phase-a-20260718.md` → `D:\cy\docs\archive\`
- Move: `D:\cy\docs\session-qwen-phase-a-20260719.md` → `D:\cy\docs\archive\`
- Move: `D:\cy\docs\session-backend-progress.md` → `D:\cy\docs\archive\`
- Remove: `D:\cy\docs\5.13光电竞赛-王子轩-佟诗茜-冯健率(4).docx`
- Remove: `D:\cy\docs\附件3：第十四届全国大学生光电设计竞赛创意组创意策划书（华北区赛）.pdf`

- [ ] **Step 1: 移动会话文件**
- [ ] **Step 2: git rm 大附件**
- [ ] **Step 3: 添加 .gitkeep 到 archive**

---

### Task 13: 最终检查和提交

- [ ] **Step 1: 检查所有文件存在**
- [ ] **Step 2: git status 确认**
- [ ] **Step 3: 分批次 git add + git commit**
