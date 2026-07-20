# 晶圆检测项目开源打包方案

> **文档版本**：v1.0
> **日期**：2026-07-20
> **目标**：将 D:/cy 完整项目内容压缩重构，补充开源标准文件，发布到 GitHub
> **许可证**：MIT
> **仓库结构**：单仓库 Monorepo
> **文档语言**：中英双语

---

## 一、现状与问题

### 1.1 当前文档 (15 文件, ~500KB)

| 文件 | 大小 | 处理策略 |
|------|------|---------|
| `COMPLETE_TECHNICAL_REFERENCE.md` | 52KB | 保留主线，压缩去重 |
| `TECHNICAL_ROADMAP.md` | 102KB | **废弃**（内容已被 COMPLETE v2.0 完全覆盖） |
| `FOR_PROJECT_MANAGER.md` | 71KB | 压缩为项目导读 |
| `optical-physics-introduction.md` | 31KB | 合并压缩为光学物理专题 |
| `optical-physics-technical-deepdive.md` | 54KB | 合并到光学物理专题 |
| `qwen-vl-wafer-detection-design.md` | 16KB | 精简保留 |
| `competition-paper-tech-revision-guide.md` | 25KB | 保留（竞赛专用） |
| `wafer-showcase-design.md` | 6.5KB | 精简保留 |
| 会话记录 (4 文件) | 24KB | **归档至 `archive/`** |
| `.docx` + `.pdf` 附件 | 9.6MB | **移除**（不适合版本控制） |

### 1.2 缺失的开源资产

- LICENSE 文件
- CONTRIBUTING.md
- CODE_OF_CONDUCT.md
- CHANGELOG.md
- SECURITY.md
- GitHub Issue/PR 模板
- 服务级 README（wafer-inspection / wafer-qwen-service / wafer-showcase）
- 根目录 .gitignore

---

## 二、目标目录结构

```
wafer-inspection/          # ← 不改动源码
wafer-backend/             # ← 不改动源码
wafer-lora-service/        # ← 不改动源码
wafer-qwen-service/        # ← 不改动源码
wafer-showcase/            # ← 不改动源码

docs/
├── README.md              # 文档索引 (bilingual)
├── technical-reference.md # 核心技术参考 (ZH, 从 COMPLETE 压缩)
├── optical-physics.md     # 光学物理注入专题 (ZH, 合并两个 physics doc)
├── project-manager-guide.md # 项目负责人导读 (ZH, 从 FOR_PM 压缩)
├── qwen-vl-detection.md   # Qwen-VL 检测方案 (ZH)
├── deployment-guide.md    # 部署指南 (ZH, 整合 docker-compose + OPERATION_GUIDE)
├── showcase-design.md     # 前端展厅设计 (ZH)
├── competition-guide.md   # 竞赛修改指导 (ZH, 保留)
└── archive/               # 归档
    ├── SESSION_RECORD.md
    ├── session-qwen-phase-a-20260718.md
    └── session-qwen-phase-a-20260719.md

# 新增开源资产
LICENSE                     # MIT
CONTRIBUTING.md             # 贡献指南 (bilingual)
CODE_OF_CONDUCT.md          # 行为准则 (EN)
CHANGELOG.md                # 变更日志
SECURITY.md                 # 安全策略
.gitignore                  # 根目录 gitignore

.github/
├── ISSUE_TEMPLATE/
│   ├── bug_report.md
│   └── feature_request.md
└── PULL_REQUEST_TEMPLATE.md

# 新增/完善服务 README
wafer-inspection/README.md     # 新建
wafer-backend/README.md        # 新建
wafer-showcase/README.md       # 新建
wafer-qwen-service/README.md   # 新建
wafer-lora-service/README.md   # 已有，需完善
```

---

## 三、文档压缩策略

### 3.1 主线文档 `technical-reference.md`

从 `COMPLETE_TECHNICAL_REFERENCE.md` (52KB) 压缩：

| 原章节 | 处理 | 目标大小 |
|--------|------|---------|
| Ch.1 项目背景与核心难题 | 精简 40% | ~3KB |
| Ch.2 技术架构 | 保留架构图，精简文字 | ~2KB |
| Ch.3 RepViT 骨干 | 保留关键设计，精简代码 | ~3KB |
| Ch.4 增强解码分支 | 保留架构和关键机制 | ~3KB |
| Ch.5 检测输出分支 | 保留设计和代码 | ~2KB |
| Ch.6 光学物理 v2.0 | 保留五模块概览，详细并入 physics 专题 | ~4KB |
| Ch.7 三阶段训练 | 保留表格和流程图 | ~2KB |
| Ch.8 模型压缩/JIT | 保留关键步骤和校验 | ~2KB |
| Ch.9 C++ SDK | 保留类设计和 C API | ~2KB |
| Ch.10 晶圆优化 | 保留极坐标和自适应校准 | ~2KB |
| Ch.11 验收标准 | 保留表格 | ~2KB |
| Ch.12 约束扩展 | 保留降级预案 | ~1KB |
| Ch.13 风险矩阵 | 保留核心矩阵 | ~1KB |
| Ch.14 参考库 | 保留表格 | ~1KB |
| Ch.15 端到端流水线 | 保留完整图和延迟分解 | ~2KB |
| **合计** | | **~30KB** |

### 3.2 光学物理专题 `optical-physics.md`

从 `optical-physics-introduction.md` (31KB) + `technical-deepdive.md` (54KB) 压缩：
- 保留完整物理基础（Rayleigh/Mie 公式推导）
- 保留 PISM/ASG/SGF 架构设计
- 删除与 COMPLETE 重复的描述
- 目标：~15KB

### 3.3 项目导读 `project-manager-guide.md`

从 `FOR_PROJECT_MANAGER.md` (71KB) 压缩：
- 保留核心比喻和类比解释
- 保留三大问题 & 对策
- 保留验收标准白话版
- 删除重复技术细节
- 目标：~20KB

---

## 四、现有文件处理

### 4.1 保留原位置但内容精简
- `docs/COMPLETE_TECHNICAL_REFERENCE.md` → 保留原始版本，新增压缩版

### 4.2 移动到 archive/
- 会话记录：SESSION_RECORD.md, session-qwen-*.md

### 4.3 从版本控制中移除
- `*.docx` — 竞赛策划书
- `*.pdf` — 附件

### 4.4 废弃但保留在 git 历史
- `TECHNICAL_ROADMAP.md` — v1.0，被 COMPLETE v2.0 取代
- `optical-physics-introduction.md`、`optical-physics-technical-deepdive.md` — 合并进专题

---

## 五、新增文件内容要点

### 5.1 根 README.md (重写)
- 项目标题 + 徽章（构建/许可证/版本）
- 一句话定位（中英双语）
- 系统架构图（ASCII）
- 服务目录快速导航
- Docker Compose 快速启动
- 文档索引链接
- 贡献指引链接

### 5.2 服务 README 统一格式
每个服务 README 包含：
- 服务定位（一句话）
- API 端点表（如有）
- 本地开发快速开始
- 环境变量
- Docker 构建
- 依赖关系

### 5.3 CONTRIBUTING.md
- Fork & PR 工作流
- 开发环境搭建
- 代码风格
- 测试要求
- 双语版本

---

## 六、实施步骤

| 步骤 | 内容 | 产出 |
|------|------|------|
| 1 | 创建目录结构 | `archive/`, `.github/ISSUE_TEMPLATE/` |
| 2 | 压缩核心技术参考 | `docs/technical-reference.md` |
| 3 | 压缩光学物理专题 | `docs/optical-physics.md` |
| 4 | 压缩项目导读 | `docs/project-manager-guide.md` |
| 5 | 精简 Qwen / Showcase / 竞赛文档 | 各自目标文件 |
| 6 | 创建部署指南 | `docs/deployment-guide.md` |
| 7 | 重写根 README | `README.md` |
| 8 | 创建 LICENSE | `LICENSE` |
| 9 | 创建 CONTRIBUTING / CODE_OF_CONDUCT / SECURITY / CHANGELOG | 各自文件 |
| 10 | 创建 GitHub 模板 | Issue + PR 模板 |
| 11 | 创建服务 README（5 个） | 各自 README.md |
| 12 | 创建 .gitignore | `.gitignore` |
| 13 | 归档会话记录 | `docs/archive/` |
| 14 | 移除大附件 | git rm `.docx` `.pdf` |
| 15 | 写 docs/README.md 索引 | `docs/README.md` |
| 16 | 最终检查和提交 | git commit |
