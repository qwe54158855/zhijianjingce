# 晶圆检测项目 — 完整交流记录速查

> 生成日期：2026-07-02
> 最后更新：2026-07-19
> 用途：下次启动会话时快速恢复上下文

---

## 📂 关键文件清单

| 文件 | 说明 | 行数 |
|------|------|------|
| `docs/TECHNICAL_ROADMAP.md` | 完整技术路线文档（已修正全部参数错误 + 新增 RL 章节） | 1347 行 |
| `docs/FOR_PROJECT_MANAGER.md` | 项目负责人导读（面向非技术背景） | ~720 行 / ~47,000 字 |
| `.claude/skills/wafer-multitask-model.md` | 多任务模型架构技能 |
| `.claude/skills/wafer-training-pipeline.md` | 三阶段训练技能 |
| `.claude/skills/wafer-jit-export.md` | JIT 导出 + 量化技能 |
| `.claude/skills/wafer-cpp-sdk.md` | C++ 推理 SDK 技能 |
| `.claude/skills/wafer-specific-adaptation.md` | 晶圆场景优化技能 |
| `.claude/skills/wafer-troubleshooting.md` | 问题排查技能 |
| `.claude/skills/wafer-init-project.md` | 项目初始化技能 |
| **`docs/session-qwen-phase-a-20260718.md`** | 🌟 Qwen-VL Phase A 基础会话记录（2026-07-18） | 
| **`docs/session-qwen-phase-a-20260719.md`** | **🌟 Qwen-VL Phase A 升级记录（b10068+视觉+置信度+亮场增强）** | 

---

## 🔧 已修正的 5 个关键问题

### 1. RepViT-M1.0 参数错误（4.8M → 6.8M）
- 文档写 M1.0=4.8M，官方数据 **6.8M**
- **修正：** 骨干改为推荐 **M0.9（5.1M）**，总参数 ~7.5M，余量 25%
- 技能文件全部同步：repvit_m1_0 → repvit_m0_9

### 2. INT8 文件大小 < 3MB 矛盾
- 7.5M 参数 INT8 = 7.5MB + 元数据 ≈ **8MB**，< 3MB 不可能
- **修正：** M0.9 方案 → < 8MB；M0.6 极致剪裁 → < 3MB

### 3. 检测缺 F2 小目标层
- 仅 F3(stride=16)/F4(stride=32)，<10px 缺陷特征被淹没
- **修正：** 增加 F2(stride=8)，三层多尺度预测

### 4. CycleGAN G_B 缺编码器
- G_B 描述为「独立解码器」逻辑不通（无编码器无法逆映射）
- **修正：** G_B = 共享编码器 + 轻量反向解码分支

### 5. Denoiser + 蒸馏优化
- MHSA 在 ARM 上效率低 → 增加 DWConv7×7 替代，默认 DWConv
- 单层蒸馏 → 改为 stage2/3/4 多层特征对齐 + 框蒸馏

### 6.（新增）多层强化学习分析
- **结论：** 不适合作为主线，但 Contextual Bandit 可锦上添花
- **推荐加入（2个）：** 方向三（在线自适应校准 Bandit）+ 方向四（动态阈值调节 Bandit）
- **不采用（3个）：** RL 检测网络、网络内可微 RL 模块、数据增强策略搜索（暂缓）
- 新章节已写入 TECHNICAL_ROADMAP.md 第 11 节

---

## 📖 项目经理导读文档结构（FOR_PROJECT_MANAGER.md）

```
一、一句话概括
二、三个要命的现实问题（展开说）
三、总体架构图
四、十大技术点详解（每个含：解决了什么 + 原理 + 验证方法）
  4.1  共享编码器 + 双分支
  4.2  RepViT 轻量骨干
  4.3  增强解码分支
  4.4  检测输出分支（含 F2 小目标层）
  4.5  极坐标边缘展开
  4.6  多波长通道融合
  4.7  CycleGAN 无监督预训练（Stage 1）
  4.8  少量样本联合微调（Stage 2）
  4.9  知识蒸馏（Stage 3）
  4.10 INT8 量化 + TorchScript 导出
  4.11 在线自适应校准
五、验收标准全解（6 个 P0 + 4 个 P1）
六、四阶段验收流程
七、技术选型权衡故事
八、风险评估
九、关键里程碑
```

---

## 🎨 技术展厅网站（wafer-showcase）

| 项目 | 说明 |
|------|------|
| 路径 | `D:/cy/wafer-showcase/` |
| 框架 | React 18 + Vite 5 + Tailwind CSS 3 + Framer Motion |
| 3D 背景 | UnicornStudio (`unicornstudio-react`) |
| 配色 | 暗色系 `#06060b` / 青色 `#00e5ff` / 紫色 `#7c3aed` |
| 交互 | Hero 全屏导航卡片 → 点击跳转各模块 |
| 品牌 | 东北大学秦皇岛分校 / © 2026 |
| 设计文档 | `docs/wafer-showcase-design.md` |

```bash
cd D:/cy/wafer-showcase && npm run dev
```

## 🚀 下次启动建议

```bash
# 快速恢复上下文：
cat docs/FOR_PROJECT_MANAGER.md

# 查看技术细节：
cat docs/TECHNICAL_ROADMAP.md | less

# 查看所有技能：
cat .claude/skills/wafer-*.md | less
```

---

## 🔬 2026-07-06 光学物理约束注入 (v2.0 架构升级)

### 核心变更

| 模块 | 说明 | 参数量 |
|------|------|--------|
| PISM | 可微 Rayleigh/Mie 散射模块，266nm→193nm 虚拟转换 | 0.04M |
| 193nm 检测分支 | 并行光路，通道裁剪至 57% | 0.42M |
| SGF | 散射引导融合层，物理先验指导融合权重 | 0.003M |
| **ASG** | **多方位角度生成器，15°-75° 每5° 13角度** | **0.15M** |
| **角度注意力融合** | **13 角度特征层面融合** | **0.02M** |

### 关键参数

- **总参数**: 8.13M / 10M (余量 18.7%)
- **推理时间**: ~71ms / 80ms (余量 11%)
- **核心物理**: (266/193)⁴ ≈ 3.5× Rayleigh 散射增益
- **ASG 效率**: 13 角度仅 +5ms（极坐标 `torch.roll` 零成本几何变换）
- **数据**: ~10 组多角度套图训练 ASG，~20 对配对标定片校准 PISM 残差

### 设计文档
- `docs/superpowers/specs/2026-07-06-optical-physics-injection-design.md` (v2.0, 完整版)

### 关键决策
1. 方位角 15°-75° 每 5° 离散生成（非连续），13 角度 batch 处理
2. ASG = 几何平移(0参数) + 散射调制(物理+学习) + 轻量补全(F1 only)
3. 极坐标空间变换 = `torch.roll`，近乎零成本
4. 角度注意力融合 = 物理置信度先验(γ=0.7) + 学习注意力(γ=0.3) 混合
5. SGF 融合: 266nm(0°基准) + Angle-Fused 193nm
