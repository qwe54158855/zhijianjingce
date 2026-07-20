# 晶圆缺陷检测 · 一体化AI解决方案 — 技术展厅设计文档

> **项目**：半导体晶圆边缘缺陷检测一体化模型 — 技术展示网站  
> **学校**：东北大学秦皇岛分校  
> **日期**：2026-07-03  
> **框架**：React + Vite + Tailwind CSS + Framer Motion + Lucide React  
> **视觉**：暗色系 / 科技感 / 高级克制 / 版心 1700px

---

## 一、设计总览

### 1.1 定位转型

从「个人简历」转型为 **技术项目展厅**，展示晶圆缺陷检测一体化 AI 方案的核心技术创新点与解决的实际问题。

### 1.2 页面结构（共 6 屏）

```
① 全屏 Hero   →   ② 三大难题   →   ③ 技术架构总览
 →   ④ 核心技术亮点   →   ⑤ 关键指标面板   →   ⑥ 全屏收尾
```

### 1.3 设计规范

| 项目 | 值 |
|------|-----|
| 底色 | `#06060b` (极深黑，微蓝调) |
| 强调色 | `#00e5ff` (科技青) |
| 辅助色 | `#7c3aed` (渐变紫) |
| 字体 | Inter (正文) / JetBrains Mono (代码/数字) |
| 版心 | `max-w-[1700px] mx-auto` |
| Hero 高度 | `h-screen` |
| 卡片边框 | `1px solid rgba(255,255,255,0.06)` |
| 卡片 Hover | 边框发光渐变动画 + translateY(-4px) |
| 导航栏 | fixed top-0，透明→滚动后 backdrop-blur 毛玻璃 |

---

## 二、逐屏详细设计

### 2.1 第①屏 · 全屏 Hero

**背景：** Canvas 粒子网络动画 + 高斯模糊暗色遮罩（渐变 `0.6 → 0.4`）

**布局（居中）：**
```
导航栏: fixed top-0, 左右布局
  [Logo: WaferAI]  [三大难题] [技术架构] [创新亮点] [指标]

主视觉:
  大标题: "半导体晶圆缺陷检测" (text-7xl, font-light, tracking-tight)
  强调副标题: "一体化 AI 解决方案" (text-5xl, font-bold, bg-gradient-to-r from-cyan-400 to-purple-500 bg-clip-text text-transparent)
  描述行: "国产光源 · ARM 边缘部署 · 小样本训练" (text-lg, tracking-widest, text-zinc-400)
  CTA 按钮: [探索技术细节 →]  描边样式, hover 时填充青色
  
底部:
  滚动指示动画 (三条线节奏缩放)
```

### 2.2 第②屏 · 三大核心难题

**标题：** "三大核心挑战" / "国产化半导体检测必须跨越的三座大山"

**三张并排卡片：**

| 卡片 | 图标 | 问题 | 数据痛点 |
|------|------|------|---------|
| 信噪比不足 | 🔬 | 国产光源暗场 SNR 仅 15-20dB，小缺陷淹没在噪声中 | `15-20dB` |
| 算力受限 | ⚡ | ARM 单核 <80ms 硬约束，12 寸晶圆需 3min 扫完 | `<80ms` |
| 数据稀缺 | 📊 | SiC/GaN 标注样本仅 30-50 张，传统需 5000+ | `30-50张` |

**交互：** Hover 边框发光渐变 (cyan→purple)，卡片上浮 4px

### 2.3 第③屏 · 技术架构总览

**标题：** "一体式架构：一骨干双分支" / 共享编码器 + 增强解码 + 检测输出 = 单次推理双输出

**架构图（CSS 绘制或交互式 SVG）：**
```
暗场灰度图 (512×512×1)
       │
       ▼
共享编码器 RepViT-M0.9  ~5.1M
  ├─ F1: 128×128×56
  ├─ F2: 64×64×112
  ├─ F3: 32×32×224
  └─ F4: 16×16×448
       │              │
       ▼              ▼
增强解码分支        检测输出分支
MSEFBlock×2        C2f_MSD + Anchor-free
Denoiser 降噪      三层多尺度 (F2/F3/F4)
~1.2M 参数         ~1.2M 参数
       │              │
       ▼              ▼
  增强明场图        缺陷检测结果
  (512×512×1)       N×(4+4)
       │              │
       └──────┬───────┘
              ▼
   TorchScript JIT 单文件 (INT8 < 8MB)
              ▼
   LibTorch C++ 推理 SDK — ARM < 80ms/帧
```

**底部技术标签：** [极坐标展开] [多波长融合] [在线校准] [INT8量化]

**动效：** 滚动进入时逐层淡入上移 (stagger 0.15s)，数据流线条发光动画

### 2.4 第④屏 · 核心技术亮点

**标题：** "核心技术突破" / 七项创新，一个目标：让国产检测真正落地

**Bento 网格布局：**

| 卡片 | 标题 | 一句话亮点 | 关键数据 |
|------|------|-----------|---------|
| ① | 共享编码器+双分支 | 一次编码，双任务并行 | 参数节省 30%，推理省 40% |
| ② | RepViT 轻量骨干 | 原生重参数化，推理融合单路卷积 | M0.9: 5.1M 参数 |
| ③ | 增强解码 MSEFBlock+Denoiser | 暗场→明场域迁移，噪声与缺陷分离 | PSNR > 25dB |
| ④ | CycleGAN 无监督预训练 | 无需配对数据，学暗→明映射 | FID < 30 |
| ⑤ | 知识蒸馏压缩 | 教师→学生，保持 99%+ 精度 | 参数减少 44% |
| ⑥ | 极坐标边缘展开 | ROI 聚焦边缘，跳过无效背景 | 计算量减 60% |
| ⑦ | 在线自适应校准 (横跨全宽) | Bandit RL 自动补偿光源漂移 | 6月误检率仅升 2% |

**动效：** 滚动入场逐张翻转/缩放进场 (stagger 0.1s)，hover 跟随光晕 + 数据高亮

### 2.5 第⑤屏 · 关键指标面板

**标题：** "硬核指标 · 量产级标准" / 六项 P0 硬约束，全部在 ARM 边缘端达标

**六格指标：**
```
<10M   <8MB    <80ms   <3min   >0.80   >25dB
参数    INT8    推理    全片    mAP     增强
```

**动效：** 数字从 0 滚动计数到目标值 (ease-out 1.5s)，边框渐变生长

**技术栈标签（底部）：**
[RepViT] [CycleGAN] [LPIPS] [INT8] [TorchScript] [LibTorch] [Cortex-A76]

### 2.6 第⑥屏 · 全屏收尾

**背景：** 与 Hero 呼应的粒子/渐变精简版

**居中内容：**
```
"从标注 50 张到量产部署"

国产光源 · ARM 边缘推理 · 全链路自主可控

[查看完整技术路线文档 →]  按钮

[GitHub] [技术博客] [联系]   三个图标链接

© 2026 东北大学秦皇岛分校 · 晶圆缺陷检测项目
```

---

## 三、组件树结构

```
src/
├── App.jsx                  # 主入口，6 屏滚动组合
├── main.jsx                 # Vite 入口
├── index.css                # Tailwind 指令 + 自定义样式
│
├── components/
│   ├── Navbar.jsx           # 固定导航栏（透明→毛玻璃过渡）
│   ├── HeroSection.jsx      # ① 全屏 Hero + 粒子背景
│   ├── ParticleBackground.jsx # Canvas 粒子网络
│   ├── ChallengesSection.jsx # ② 三大核心难题
│   ├── ChallengeCard.jsx    # 单张难题卡片
│   ├── ArchitectureSection.jsx # ③ 技术架构总览
│   ├── ArchitectureDiagram.jsx # 交互式架构图
│   ├── HighlightsSection.jsx # ④ 核心技术亮点
│   ├── HighlightCard.jsx    # 单张创新卡片
│   ├── MetricsSection.jsx   # ⑤ 关键指标面板
│   ├── MetricItem.jsx       # 单个指标（含计数动画）
│   ├── FooterSection.jsx    # ⑥ 全屏收尾页
│   └── ScrollIndicator.jsx  # 滚动提示
│
├── hooks/
│   └── useScrollAnimation.js # 滚动触发动画 hook
│
└── utils/
    └── cn.js                # tailwind-merge (class 合并)
```

---

## 四、动效矩阵

| 元素 | 入场动效 | Hover 动效 | 滚动相关 |
|------|---------|-----------|---------|
| Hero 标题 | fadeIn + slideUp (1s) | — | — |
| 架构图各层 | fadeIn + slideUp, stagger 0.15s | tooltip | ScrollTrigger |
| 难题卡片 | fadeIn + slideUp | 边框发光 + 上浮 | 滚动入视口触发 |
| 创新卡片 | scaleIn + rotateIn, stagger 0.1s | 光晕跟随鼠标 | 滚动入视口触发 |
| 指标数字 | countUp 0→N, 1.5s ease-out | — | 滚动入视口触发 |
| 导航栏 | — | 链接文字发光 | 滚动改变透明度|

---

## 五、交互设计决策

- **无复杂状态管理**：纯展示页面，无需 React Context / Redux
- **滚动驱动动画**：使用 Framer Motion 的 `useInView` 或 `whileInView`
- **粒子背景性能**：Canvas 粒子数控制在 80-120 个，带 requestAnimationFrame 节流
- **响应式降级**：PC first，移动端保持可读性（减小字体/改为单列）
