# 晶圆检测技术展厅 — 项目设计文档

> **项目路径**：`D:/cy/wafer-showcase/`  
> **框架**：React 18 + Vite 5 + Tailwind CSS 3 + Framer Motion 11 + Lucide React  
> **3D 背景**：UnicornStudio React (`unicornstudio-react`)  
> **包管理**：npm  
> **版本控制**：git (main 分支)

---

## 一、项目定位

半导体晶圆缺陷检测一体化 AI 解决方案的 **技术展示网站**。  
非个人简历，而是以 **技术展厅** 形态呈现项目的核心创新点与解决的问题。

---

## 二、视觉设计规范

### 2.1 色板

| 用途 | Tailwind 变量 | 色值 | 说明 |
|------|--------------|------|------|
| 页面底色 | `tech-deep` | `#06060b` | 极深黑（微蓝调） |
| 卡片底色 | `tech-card` | `#0d0d14` | 深灰黑 |
| 卡片边框 | `tech-border` | `rgba(255,255,255,0.06)` | 微光边框 |
| 强调色（青） | `tech-cyan` | `#00e5ff` | 科技青 |
| 辅助色（紫） | `tech-purple` | `#7c3aed` | 渐变紫 |

### 2.2 标题渐变

**所有大标题** 统一使用：
```
text-transparent bg-clip-text bg-gradient-to-r from-tech-cyan to-tech-purple
```

涵盖：
- Hero h1：半导体晶圆缺陷检测 · 一体化AI 解决方案
- Challenges h2：三大核心难题
- Architecture h2：一体式架构：一骨干双分支
- Highlights h2：核心技术亮点
- Metrics h2：关键指标
- Footer h2：从标注 50 张到量产部署

### 2.3 字体

| 用途 | 字体 |
|------|------|
| 正文 | Inter (Google Fonts) |
| 代码/数字 | JetBrains Mono (Google Fonts) |

### 2.4 布局

- 版心：`max-w-content: 1700px` + `mx-auto`
- 背景：纯暗色 `bg-tech-deep`
- 文字主色：`text-white`
- 文字次级：`text-zinc-400` / `text-zinc-500`
- 所有卡片：`border border-tech-border` + hover 发光 `shadow-[0_0_30px_rgba(0,229,255,0.08)]`

---

## 三、页面结构

### 3.1 交互方式

**主界面导航台模式** — 非上下滚动长页面：
1. **Hero 主屏**：全屏 3D 背景 + 标题 + 4 张导航卡片
2. **点击卡片** → 锚点跳转对应的 section
3. 各模块内容（三大难题/技术架构/亮点/指标/Footer）仍在页面中，通过导航卡片进入

### 3.2 模块清单

| 模块 | 文件 | 说明 |
|------|------|------|
| Hero | `HeroSection.jsx` | 全屏，3D 背景 + 标题 + 4 导航卡片 |
| 三大难题 | `ChallengesSection.jsx` + `ChallengeCard.jsx` | 3 张卡片展示核心技术挑战 |
| 技术架构 | `ArchitectureSection.jsx` + `ArchitectureDiagram.jsx` | 6 层架构图 + 7 个技术标签 |
| 核心技术亮点 | `HighlightsSection.jsx` + `HighlightCard.jsx` | 7 张 Bento 网格卡片 |
| 关键指标 | `MetricsSection.jsx` + `MetricItem.jsx` | 6 个数字动效指标 |
| Footer | `FooterSection.jsx` | 全屏收尾，CTA + 链接 + 版权 |

### 3.3 Hero 导航卡片

4 张卡片，2×2 网格（lg: 4 列）：

| 卡片 | 图标色 | 描述 | 锚点 |
|------|--------|------|------|
| 三大核心难题 | `text-rose-400` | 信噪比不足、算力受限、数据稀缺 | `#challenges` |
| 技术架构总览 | `text-tech-cyan` | 共享编码器 + 双解码分支 | `#architecture` |
| 核心技术亮点 | `text-purple-400` | 七项创新，从预训练到在线校准 | `#highlights` |
| 关键指标面板 | `text-emerald-400` | 六项 P0 硬约束量产级达标 | `#metrics` |

卡片样式：`border border-white/10 bg-white/5 backdrop-blur-sm`  
hover：上浮 `-translate-y-1` + 边框 `border-tech-cyan/30`

---

## 四、组件文件结构

```
src/
├── App.jsx                    # 根组件，串联所有 section
├── main.jsx                   # Vite 入口
├── index.css                  # Tailwind 指令 + 滚动条样式
│
├── utils/
│   └── cn.js                  # clsx + tailwind-merge 工具
│
└── components/
    ├── Navbar.jsx             # 固定导航（透明→毛玻璃过渡）
    ├── HeroSection.jsx        # 全屏主界面 + 3D + 导航卡片
    ├── ChallengesSection.jsx  # 三大核心难题
    ├── ChallengeCard.jsx      # 单张难题卡片
    ├── ArchitectureSection.jsx # 技术架构
    ├── ArchitectureDiagram.jsx # 架构图（6 层交错动画）
    ├── HighlightsSection.jsx  # 核心技术亮点
    ├── HighlightCard.jsx      # 单张亮点卡片
    ├── MetricsSection.jsx     # 关键指标面板
    ├── MetricItem.jsx         # 单个指标（useMotionValue 计数动画）
    └── FooterSection.jsx      # 全屏收尾
```

---

## 五、配置与依赖

### 5.1 tailwind.config.js 关键配置

```js
colors: {
  tech: {
    cyan: '#00e5ff',
    purple: '#7c3aed',
    deep: '#06060b',
    card: '#0d0d14',
    border: 'rgba(255,255,255,0.06)',
  },
},
maxWidth: {
  content: '1700px',
},
fontFamily: {
  sans: ['Inter', 'system-ui', 'sans-serif'],
  mono: ['JetBrains Mono', 'monospace'],
},
```

### 5.2 依赖

```json
{
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "framer-motion": "^11.0.0",
    "lucide-react": "^0.400.0",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.3.0",
    "unicornstudio-react": "^latest"
  },
  "devDependencies": {
    "@vitejs/plugin-react": "^4.2.0",
    "vite": "^5.4.0",
    "tailwindcss": "^3.4.0",
    "postcss": "^8.4.0",
    "autoprefixer": "^10.4.0"
  }
}
```

---

## 六、品牌信息

- **学校**：东北大学秦皇岛分校
- **版权年份**：2026
- **项目名**：晶圆缺陷检测 · 一体化AI 解决方案
- **Logo**：`WAFER_AI`（Navbar 左上角，青色圆点 + 文字）

---

## 七、3D 背景

使用 UnicornStudio React SDK：
```jsx
import UnicornScene from 'unicornstudio-react'

<UnicornScene
  projectId="GB3z8XtyG0Cbyy6QSoDI"
  width="..."
  height="..."
  scale={1}
  dpi={1.5}
  sdkUrl="https://cdn.jsdelivr.net/gh/hiunicornstudio/unicornstudio.js@v2.2.6/dist/unicornStudio.umd.js"
/>
```

- 容器自适应：通过 `useRef` + `useEffect` 动态测量宽高
- 遮罩层：`bg-gradient-to-b from-black/60 via-black/30 to-black/70`

---

## 八、动效规范

| 元素 | 动效 |
|------|------|
| 标题入场 | `fadeIn + slideUp`，stagger 0.2s 延迟 |
| 导航卡片 | `fadeIn + slideUp`，delay 0.9s |
| 难题/亮点卡片 | `whileInView` + stagger `index * 0.15s` |
| 架构图 | `whileInView` + stagger 0.15s 逐层 |
| 指标数字 | `useMotionValue` + `animate()` 计数 0→N |
| 导航栏 | 滚动 50px 后 `bg-tech-deep/80 backdrop-blur-xl` |
| 卡片 hover | 上浮 4px + 边框发光 + 光晕渐变 |
