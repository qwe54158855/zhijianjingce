# 晶圆缺陷检测技术展厅 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建一个暗色系、科技感、高级克制的 PC 端技术展厅网站，展示晶圆缺陷检测项目的核心创新点与解决的实际问题。

**Architecture:** 6 屏全屏滚动 SPA，React + Vite + Tailwind CSS + Framer Motion + Lucide React。Canvas 粒子背景，1700px 版心，滚动驱动的交互动效。无路由/无数据层，纯展示页面。

**Tech Stack:** React 18, Vite 5, Tailwind CSS 3, Framer Motion 11, Lucide React 0.400+

## Global Constraints

- 版心严格控制在 `max-w-[1700px] mx-auto`
- 暗色色板：底色 `#06060b`，强调色 `#00e5ff`（科技青），辅助色 `#7c3aed`（紫）
- 字体：Inter（正文），JetBrains Mono（数字/代码）
- 所有卡片统一样式：1px `rgba(255,255,255,0.06)` 边框，hover 发光渐变动效
- 导航栏 fixed top-0，透明 → 滚动后 backdrop-blur 毛玻璃
- 学校名称：东北大学秦皇岛分校
- 年份：2026
- 所有样式使用 Tailwind CSS 类实现，不得使用 CSS modules 或 styled-components

---

### Task 1: 脚手架搭建 + 依赖安装

**Files:**
- Create: `D:/cy/wafer-showcase/package.json`
- Create: `D:/cy/wafer-showcase/vite.config.js`
- Create: `D:/cy/wafer-showcase/index.html`
- Create: `D:/cy/wafer-showcase/postcss.config.js`
- Create: `D:/cy/wafer-showcase/tailwind.config.js`
- Create: `D:/cy/wafer-showcase/src/main.jsx`
- Create: `D:/cy/wafer-showcase/src/index.css`
- Create: `D:/cy/wafer-showcase/src/App.jsx`
- Create: `D:/cy/wafer-showcase/src/utils/cn.js`

**Interfaces:**
- Produces: A runnable Vite dev server with Tailwind CSS + Framer Motion configured

- [ ] **Step 1: Create project directory and package.json**

```bash
mkdir -p "D:/cy/wafer-showcase/src/components"
mkdir -p "D:/cy/wafer-showcase/src/hooks"
mkdir -p "D:/cy/wafer-showcase/src/utils"
```

```json
{
  "name": "wafer-tech-showcase",
  "private": true,
  "version": "1.0.0",
  "type": "module",
  "scripts": {
    "dev": "vite",
    "build": "vite build",
    "preview": "vite preview"
  },
  "dependencies": {
    "react": "^18.3.1",
    "react-dom": "^18.3.1",
    "framer-motion": "^11.0.0",
    "lucide-react": "^0.400.0",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.3.0"
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

- [ ] **Step 2: Create vite.config.js**

```javascript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
})
```

- [ ] **Step 3: Create postcss.config.js**

```javascript
export default {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
```

- [ ] **Step 4: Create tailwind.config.js**

```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
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
    },
  },
  plugins: [],
}
```

- [ ] **Step 5: Create index.html**

```html
<!DOCTYPE html>
<html lang="zh-CN">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>晶圆缺陷检测 · 一体化AI解决方案</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@200;300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet" />
  </head>
  <body class="bg-tech-deep text-white">
    <div id="root"></div>
    <script type="module" src="/src/main.jsx"></script>
  </body>
</html>
```

- [ ] **Step 6: Create src/index.css**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

html {
  scroll-behavior: smooth;
}

body {
  font-family: 'Inter', system-ui, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* Custom scrollbar */
::-webkit-scrollbar {
  width: 6px;
}
::-webkit-scrollbar-track {
  background: #06060b;
}
::-webkit-scrollbar-thumb {
  background: rgba(0, 229, 255, 0.2);
  border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
  background: rgba(0, 229, 255, 0.4);
}
```

- [ ] **Step 7: Create src/main.jsx**

```jsx
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
```

- [ ] **Step 8: Create src/utils/cn.js**

```javascript
import { clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function cn(...inputs) {
  return twMerge(clsx(inputs))
}
```

- [ ] **Step 9: Create minimal App.jsx scaffold**

```jsx
export default function App() {
  return (
    <div className="relative bg-tech-deep text-white">
      <p className="text-center pt-20 text-tech-cyan">WaferAI 展厅加载中...</p>
    </div>
  )
}
```

- [ ] **Step 10: Install and verify**

```bash
cd "D:/cy/wafer-showcase"
npm install
npm run dev
```

Expected: Vite dev server starts at http://localhost:5173, page shows "WaferAI 展厅加载中..."

---

### Task 2: Navbar + ScrollIndicator + 工具 Hook

**Files:**
- Create: `D:/cy/wafer-showcase/src/components/Navbar.jsx`
- Create: `D:/cy/wafer-showcase/src/components/ScrollIndicator.jsx`
- Create: `D:/cy/wafer-showcase/src/hooks/useScrollAnimation.js`

**Interfaces:**
- `Navbar` — fixed top, 透明→毛玻璃过渡, 导航链接锚点
- `ScrollIndicator` — 底部三线动画, 提示向下滚动
- `useScrollAnimation(options?)` — 基于 useInView 的动画控制 hook

- [ ] **Step 1: Create src/hooks/useScrollAnimation.js**

```javascript
import { useInView } from 'framer-motion'
import { useRef } from 'react'

export function useScrollAnimation(options = {}) {
  const ref = useRef(null)
  const isInView = useInView(ref, {
    once: true,
    margin: '-100px',
    ...options,
  })
  return [ref, isInView]
}
```

- [ ] **Step 2: Create src/components/Navbar.jsx**

```jsx
import { useState, useEffect } from 'react'
import { cn } from '../utils/cn'

const NAV_ITEMS = [
  { label: '三大难题', href: '#challenges' },
  { label: '技术架构', href: '#architecture' },
  { label: '创新亮点', href: '#highlights' },
  { label: '关键指标', href: '#metrics' },
]

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50)
    window.addEventListener('scroll', handleScroll, { passive: true })
    return () => window.removeEventListener('scroll', handleScroll)
  }, [])

  return (
    <nav
      className={cn(
        'fixed top-0 left-0 right-0 z-50 transition-all duration-500',
        scrolled
          ? 'bg-tech-deep/80 backdrop-blur-xl border-b border-white/5'
          : 'bg-transparent'
      )}
    >
      <div className="max-w-content mx-auto px-8 h-16 flex items-center justify-between">
        <a href="#" className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-tech-cyan shadow-[0_0_8px_rgba(0,229,255,0.5)]" />
          <span className="font-mono text-sm font-semibold tracking-wider text-white/90">
            WAFER_AI
          </span>
        </a>

        <div className="flex items-center gap-8">
          {NAV_ITEMS.map((item) => (
            <a
              key={item.href}
              href={item.href}
              className="text-sm text-zinc-400 hover:text-tech-cyan transition-colors duration-300 tracking-wide"
            >
              {item.label}
            </a>
          ))}
        </div>
      </div>
    </nav>
  )
}
```

- [ ] **Step 3: Create src/components/ScrollIndicator.jsx**

```jsx
import { motion } from 'framer-motion'

export default function ScrollIndicator() {
  return (
    <motion.div
      className="absolute bottom-10 left-1/2 -translate-x-1/2 flex flex-col items-center gap-2"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ delay: 1.5, duration: 1 }}
    >
      <span className="text-xs text-zinc-500 tracking-widest">SCROLL</span>
      <div className="flex flex-col items-center gap-1">
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            className="w-[1px] h-4 bg-tech-cyan/40"
            animate={{ opacity: [0.2, 0.8, 0.2], scaleY: [1, 1.5, 1] }}
            transition={{ duration: 1.5, repeat: Infinity, delay: i * 0.2 }}
          />
        ))}
      </div>
    </motion.div>
  )
}
```

---

### Task 3: HeroSection + ParticleBackground

**Files:**
- Create: `D:/cy/wafer-showcase/src/components/HeroSection.jsx`
- Create: `D:/cy/wafer-showcase/src/components/ParticleBackground.jsx`

**Interfaces:**
- `HeroSection` → 全屏 Hero，包含 Navbar + 标题 + 按钮 + ScrollIndicator + 粒子背景
- `ParticleBackground` → Canvas 粒子网络动画，80-120 粒子，青色连接线

- [ ] **Step 1: Create src/components/ParticleBackground.jsx**

```jsx
import { useEffect, useRef } from 'react'

export default function ParticleBackground() {
  const canvasRef = useRef(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    let animationId
    let particles = []

    const resize = () => {
      canvas.width = window.innerWidth
      canvas.height = window.innerHeight
    }
    resize()
    window.addEventListener('resize', resize)

    const COUNT = 100
    const CONNECTION_DIST = 140

    class Particle {
      constructor() {
        this.x = Math.random() * canvas.width
        this.y = Math.random() * canvas.height
        this.vx = (Math.random() - 0.5) * 0.8
        this.vy = (Math.random() - 0.5) * 0.8
        this.radius = Math.random() * 1.5 + 0.5
      }
      update() {
        this.x += this.vx
        this.y += this.vy
        if (this.x < 0 || this.x > canvas.width) this.vx *= -1
        if (this.y < 0 || this.y > canvas.height) this.vy *= -1
      }
      draw() {
        if (!ctx) return
        ctx.beginPath()
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2)
        ctx.fillStyle = 'rgba(0, 229, 255, 0.4)'
        ctx.fill()
      }
    }

    particles = Array.from({ length: COUNT }, () => new Particle())

    const animate = () => {
      if (!ctx) return
      ctx.clearRect(0, 0, canvas.width, canvas.height)

      particles.forEach((p) => {
        p.update()
        p.draw()
      })

      // Draw connections
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const dx = particles[i].x - particles[j].x
          const dy = particles[i].y - particles[j].y
          const dist = Math.sqrt(dx * dx + dy * dy)
          if (dist < CONNECTION_DIST) {
            const opacity = (1 - dist / CONNECTION_DIST) * 0.15
            ctx.beginPath()
            ctx.moveTo(particles[i].x, particles[i].y)
            ctx.lineTo(particles[j].x, particles[j].y)
            ctx.strokeStyle = `rgba(0, 229, 255, ${opacity})`
            ctx.lineWidth = 0.5
            ctx.stroke()
          }
        }
      }

      animationId = requestAnimationFrame(animate)
    }
    animate()

    return () => {
      cancelAnimationFrame(animationId)
      window.removeEventListener('resize', resize)
    }
  }, [])

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full"
      style={{ filter: 'blur(0.5px)' }}
    />
  )
}
```

- [ ] **Step 2: Create src/components/HeroSection.jsx**

```jsx
import { motion } from 'framer-motion'
import ParticleBackground from './ParticleBackground'
import Navbar from './Navbar'
import ScrollIndicator from './ScrollIndicator'
import { ArrowDownRight } from 'lucide-react'

export default function HeroSection() {
  return (
    <section className="relative h-screen w-full overflow-hidden">
      {/* Background layers */}
      <div className="absolute inset-0 bg-gradient-to-b from-tech-deep via-tech-deep to-tech-deep/95" />
      <ParticleBackground />
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-tech-deep/30 to-tech-deep/80" />

      {/* Navbar */}
      <Navbar />

      {/* Main content */}
      <div className="relative z-10 h-full flex flex-col items-center justify-center px-8">
        <div className="max-w-content mx-auto text-center">
          <motion.p
            className="text-xs tracking-[0.3em] text-tech-cyan/60 mb-6 font-mono"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            东北大学秦皇岛分校 · 半导体检测项目
          </motion.p>

          <motion.h1
            className="text-6xl md:text-7xl lg:text-8xl font-light tracking-tight text-white mb-4"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            半导体晶圆缺陷检测
          </motion.h1>

          <motion.h2
            className="text-4xl md:text-5xl font-bold mb-8"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            <span className="bg-gradient-to-r from-tech-cyan to-tech-purple bg-clip-text text-transparent">
              一体化 AI 解决方案
            </span>
          </motion.h2>

          <motion.p
            className="text-base md:text-lg text-zinc-400 tracking-widest mb-12"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.6 }}
          >
            国产光源 · ARM 边缘部署 · 小样本训练
          </motion.p>

          <motion.a
            href="#challenges"
            className="inline-flex items-center gap-2 px-8 py-3 border border-white/10 rounded-full
                       text-sm text-zinc-300 hover:text-tech-cyan hover:border-tech-cyan/30
                       transition-all duration-500 group"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.8 }}
          >
            探索技术细节
            <ArrowDownRight size={16} className="group-hover:translate-x-1 group-hover:translate-y-1 transition-transform duration-300" />
          </motion.a>
        </div>
      </div>

      <ScrollIndicator />
    </section>
  )
}
```

---

### Task 4: ChallengesSection — 三大核心难题

**Files:**
- Create: `D:/cy/wafer-showcase/src/components/ChallengesSection.jsx`
- Create: `D:/cy/wafer-showcase/src/components/ChallengeCard.jsx`

**Interfaces:**
- `ChallengesSection` → 整屏，标题 + 三张并排卡片
- `ChallengeCard({ icon, title, description, data })` → 单张难题卡片

- [ ] **Step 1: Create src/components/ChallengeCard.jsx**

```jsx
import { motion } from 'framer-motion'
import { cn } from '../utils/cn'

export default function ChallengeCard({ icon, title, problem, metric, unit, detail, index }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 40 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-80px' }}
      transition={{ duration: 0.6, delay: index * 0.15 }}
      className="group relative p-8 rounded-2xl border border-white/[0.06] bg-tech-card/50
                 hover:border-tech-cyan/30 transition-all duration-500
                 hover:-translate-y-1"
    >
      {/* Glow effect on hover */}
      <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500
                      bg-gradient-to-br from-tech-cyan/[0.03] to-tech-purple/[0.03]" />

      <div className="relative z-10">
        <span className="text-3xl mb-4 block">{icon}</span>
        <h3 className="text-xl font-semibold text-white mb-3">{title}</h3>
        <p className="text-zinc-400 text-sm leading-relaxed mb-6">{problem}</p>
        <div className="flex items-baseline gap-1.5">
          <span className="text-4xl font-bold font-mono text-tech-cyan">{metric}</span>
          <span className="text-sm text-zinc-500 font-mono">{unit}</span>
        </div>
        <p className="text-xs text-zinc-600 mt-2">{detail}</p>
      </div>
    </motion.div>
  )
}
```

- [ ] **Step 2: Create src/components/ChallengesSection.jsx**

```jsx
import { motion } from 'framer-motion'
import ChallengeCard from './ChallengeCard'

const CHALLENGES = [
  {
    icon: '🔬',
    title: '信噪比不足',
    problem: '国产光源暗场信噪比偏低，小于 10 像素的微小缺陷完全淹没在背景噪声中。传统图像增强方法在低信噪比下会放大噪声，检测反而更差。',
    metric: '15-20',
    unit: 'dB SNR',
    detail: '明场通常 30dB+，差距 2 倍以上',
  },
  {
    icon: '⚡',
    title: '算力严重受限',
    problem: '边缘设备采用 ARM 嵌入式平台（Cortex-A76），单帧推理必须 <80ms。整片 12 寸晶圆步进扫描 2000-4000 张子图，总耗时需 <3 分钟。',
    metric: '<80',
    unit: 'ms / 帧',
    detail: '普通模型在 ARM 上通常 200-500ms',
  },
  {
    icon: '📊',
    title: '样本极端稀缺',
    problem: 'SiC/GaN 第三代半导体工艺快速迭代，缺陷形态多变。可用标注样本通常仅 30-50 张，传统深度学习方法需要 5000+ 张标注数据。',
    metric: '30-50',
    unit: '张标注',
    detail: '仅为传统需求量的 1%',
  },
]

export default function ChallengesSection() {
  return (
    <section id="challenges" className="min-h-screen flex items-center py-32">
      <div className="max-w-content mx-auto px-8 w-full">
        <motion.div
          className="text-center mb-20"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <p className="text-xs tracking-[0.3em] text-tech-cyan/40 font-mono mb-4">PROBLEM</p>
          <h2 className="text-4xl md:text-5xl font-light text-white mb-4">
            三大核心挑战
          </h2>
          <p className="text-zinc-500 text-sm tracking-wide">
            国产化半导体检测必须跨越的三座大山
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {CHALLENGES.map((c, i) => (
            <ChallengeCard key={c.title} {...c} index={i} />
          ))}
        </div>
      </div>
    </section>
  )
}
```

---

### Task 5: ArchitectureSection — 技术架构总览

**Files:**
- Create: `D:/cy/wafer-showcase/src/components/ArchitectureSection.jsx`
- Create: `D:/cy/wafer-showcase/src/components/ArchitectureDiagram.jsx`

**Interfaces:**
- `ArchitectureSection` → 整屏，标题 + 架构图 + 底部技术标签
- `ArchitectureDiagram` → 交互式 SVG/CSS 架构图，逐层淡入

- [ ] **Step 1: Create src/components/ArchitectureDiagram.jsx**

```jsx
import { motion } from 'framer-motion'

const layers = [
  { label: '暗场灰度图 (512×512×1)', sub: '输入', color: 'from-zinc-500/20 to-zinc-500/5', delay: 0 },
  { label: '共享编码器 RepViT-M0.9', sub: '~5.1M 参数 · 4 层多尺度特征', color: 'from-tech-cyan/20 to-tech-cyan/5', delay: 0.15 },
  { label: '增强解码分支', sub: 'MSEFBlock × 2 · Denoiser · ~1.2M', color: 'from-blue-500/20 to-blue-500/5', delay: 0.3, right: true },
  { label: '检测输出分支', sub: 'C2f_MSD · Anchor-free · ~1.2M', color: 'from-purple-500/20 to-purple-500/5', delay: 0.3, right: false },
  { label: 'TorchScript JIT 单文件', sub: 'INT8 量化 < 8MB', color: 'from-tech-cyan/20 to-tech-purple/20', delay: 0.45 },
  { label: 'LibTorch C++ 推理 SDK', sub: 'ARM Cortex-A76 < 80ms / 帧', color: 'from-emerald-500/20 to-emerald-500/5', delay: 0.6 },
]

export default function ArchitectureDiagram() {
  return (
    <div className="relative flex flex-col items-center gap-3 py-12">
      {/* Vertical connecting line */}
      <div className="absolute top-0 bottom-0 left-1/2 w-[1px] bg-gradient-to-b from-tech-cyan/40 via-tech-purple/40 to-emerald-500/40 -translate-x-1/2" />

      {layers.map((layer, i) => (
        <motion.div
          key={layer.label}
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5, delay: layer.delay }}
          className="relative z-10 w-full max-w-[600px]"
        >
          {/* Branch indicator for dual paths */}
          {layer.right !== undefined && (
            <div className="absolute top-1/2 -translate-y-1/2 w-8 h-[1px] bg-white/10"
                 style={{ [layer.right ? 'left' : 'right']: '-32px' }} />
          )}

          <div className={`rounded-xl p-5 border border-white/[0.06] bg-gradient-to-r ${layer.color}
                          hover:border-tech-cyan/20 transition-all duration-300`}>
            <p className="text-sm font-medium text-white">{layer.label}</p>
            <p className="text-xs text-zinc-500 font-mono mt-1">{layer.sub}</p>
          </div>
        </motion.div>
      ))}

      {/* Dual branch merge indicator */}
      <div className="flex items-center gap-2 text-[10px] text-zinc-600 font-mono tracking-wider">
        <span>增强输出</span>
        <span className="w-8 h-[1px] bg-zinc-700" />
        <span className="text-tech-cyan">●</span>
        <span className="w-8 h-[1px] bg-zinc-700" />
        <span>检测输出</span>
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Create src/components/ArchitectureSection.jsx**

```jsx
import { motion } from 'framer-motion'
import ArchitectureDiagram from './ArchitectureDiagram'

const TAGS = ['RepViT', 'CycleGAN', 'LPIPS', 'INT8', 'TorchScript', 'LibTorch', 'Cortex-A76']

export default function ArchitectureSection() {
  return (
    <section id="architecture" className="min-h-screen flex items-center py-32">
      <div className="max-w-content mx-auto px-8 w-full">
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <p className="text-xs tracking-[0.3em] text-tech-cyan/40 font-mono mb-4">ARCHITECTURE</p>
          <h2 className="text-4xl md:text-5xl font-light text-white mb-4">
            一体式架构：一骨干双分支
          </h2>
          <p className="text-zinc-500 text-sm tracking-wide">
            共享编码器 + 增强解码 + 检测输出 = 单次推理双输出
          </p>
        </motion.div>

        <ArchitectureDiagram />

        <motion.div
          className="flex flex-wrap justify-center gap-3 mt-16"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.6 }}
        >
          {TAGS.map((tag) => (
            <span
              key={tag}
              className="px-4 py-1.5 rounded-full text-xs font-mono
                         border border-white/[0.06] text-zinc-400
                         hover:border-tech-cyan/30 hover:text-tech-cyan
                         transition-all duration-300"
            >
              {tag}
            </span>
          ))}
        </motion.div>
      </div>
    </section>
  )
}
```

---

### Task 6: HighlightsSection — 核心技术亮点

**Files:**
- Create: `D:/cy/wafer-showcase/src/components/HighlightsSection.jsx`
- Create: `D:/cy/wafer-showcase/src/components/HighlightCard.jsx`

**Interfaces:**
- `HighlightsSection` → Bento 网格布局，7 张卡片（前 6 张 2×3 网格 + 第 7 张全宽）
- `HighlightCard({ icon, title, tagline, description, metrics, index, wide? })` → 单张亮点卡片

- [ ] **Step 1: Create src/components/HighlightCard.jsx**

```jsx
import { motion } from 'framer-motion'

export default function HighlightCard({ icon, title, tagline, description, metrics, index, wide }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration: 0.5, delay: index * 0.08 }}
      className={`group relative p-6 md:p-8 rounded-2xl border border-white/[0.06] bg-tech-card/50
                  hover:border-tech-cyan/20 transition-all duration-500 overflow-hidden
                  ${wide ? 'md:col-span-3' : ''}`}
    >
      {/* Hover glow */}
      <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500
                      bg-gradient-to-br from-tech-cyan/[0.02] to-tech-purple/[0.02]" />

      <div className="relative z-10">
        <div className="flex items-start gap-3 mb-4">
          <span className="text-2xl">{icon}</span>
          <div>
            <h3 className="text-lg font-semibold text-white">{title}</h3>
            <p className="text-xs text-tech-cyan/60 font-mono mt-0.5">{tagline}</p>
          </div>
        </div>

        <p className="text-sm text-zinc-400 leading-relaxed mb-4">{description}</p>

        {metrics && (
          <div className="flex flex-wrap gap-3">
            {metrics.map((m) => (
              <span
                key={m.label}
                className="inline-flex items-center gap-1.5 px-3 py-1 rounded-md bg-white/[0.03]
                           border border-white/[0.05] text-xs"
              >
                <span className="text-tech-cyan font-mono font-semibold">{m.value}</span>
                <span className="text-zinc-500">{m.label}</span>
              </span>
            ))}
          </div>
        )}
      </div>
    </motion.div>
  )
}
```

- [ ] **Step 2: Create src/components/HighlightsSection.jsx**

```jsx
import { motion } from 'framer-motion'
import HighlightCard from './HighlightCard'

const HIGHLIGHTS = [
  {
    icon: '🧠',
    title: '共享编码器 + 双分支',
    tagline: '一次编码，双任务并行',
    description: '编码器只需一次前向计算，增强和检测两个分支共享底层特征。相比独立部署两个模型，参数节省 30%，推理时间节省近 40%。',
    metrics: [
      { value: '-30%', label: '参数' },
      { value: '-40%', label: '推理时间' },
    ],
  },
  {
    icon: '🔧',
    title: 'RepViT 轻量骨干',
    tagline: '原生重参数化架构',
    description: '训练时三分支并行学习，推理前 fuse() 合并为单路 3×3 卷积。M0.9 仅 5.1M 参数，ARM NEON 指令集高度优化。',
    metrics: [
      { value: '5.1M', label: '参数' },
      { value: '78.7%', label: 'ImageNet Top-1' },
    ],
  },
  {
    icon: '🎨',
    title: '增强解码 MSEFBlock',
    tagline: '暗场→明场域迁移',
    description: 'MSEFBlock 解耦空间特征提取与通道注意力，Denoiser 用 U-Net 残差降噪。CycleGAN + LPIPS 感知损失确保内容不丢失。',
    metrics: [
      { value: '>25', label: 'dB PSNR' },
      { value: '<0.1', label: 'LPIPS' },
    ],
  },
  {
    icon: '🔄',
    title: 'CycleGAN 无监督预训练',
    tagline: '无需配对数据',
    description: '循环一致性损失约束暗→明→暗映射，确保内容结构不变。仅需大量无标注暗场和明场图像，即可建立域迁移能力。',
    metrics: [
      { value: '<30', label: 'FID' },
      { value: '100', label: 'epochs' },
    ],
  },
  {
    icon: '📉',
    title: '知识蒸馏压缩',
    tagline: '44% 参数压缩，>99% 精度保留',
    description: '教师模型(M0.9, 7.5M)→学生模型(M0.6, 4.2M)。多层特征蒸馏(stage2/3/4) + 框蒸馏，精度损失 < 1%。',
    metrics: [
      { value: '-44%', label: '参数' },
      { value: '>99%', label: '精度保留' },
    ],
  },
  {
    icon: '🎯',
    title: '极坐标边缘展开',
    tagline: '跳过 60% 无效背景',
    description: '缺陷集中在晶圆边缘环形区域(占图 35-40%)。极坐标展开将 ROI 从 512×512 压缩到 512×128，计算量减少 60%。',
    metrics: [
      { value: '-60%', label: '计算量' },
      { value: '~30%', label: '速度提升' },
    ],
  },
  {
    icon: '🤖',
    title: '在线自适应校准',
    tagline: 'Bandit RL 自动补偿光源漂移',
    description: '设备运行 3-6 月后光源衰减、传感器漂移。Contextual Bandit 自动调整归一化参数补偿分布偏移，6 月后误检率仅上升 2%。',
    metrics: [
      { value: '+2%', label: '6月误检率变化' },
      { value: '<5s', label: '校准耗时' },
    ],
    wide: true,
  },
]

export default function HighlightsSection() {
  return (
    <section id="highlights" className="min-h-screen flex items-center py-32">
      <div className="max-w-content mx-auto px-8 w-full">
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <p className="text-xs tracking-[0.3em] text-tech-cyan/40 font-mono mb-4">INNOVATIONS</p>
          <h2 className="text-4xl md:text-5xl font-light text-white mb-4">
            核心技术突破
          </h2>
          <p className="text-zinc-500 text-sm tracking-wide">
            七项创新，一个目标：让国产检测真正落地
          </p>
        </motion.div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {HIGHLIGHTS.map((h, i) => (
            <HighlightCard key={h.title} {...h} index={i} />
          ))}
        </div>
      </div>
    </section>
  )
}
```

---

### Task 7: MetricsSection — 关键指标面板

**Files:**
- Create: `D:/cy/wafer-showcase/src/components/MetricsSection.jsx`
- Create: `D:/cy/wafer-showcase/src/components/MetricItem.jsx`

**Interfaces:**
- `MetricsSection` → 整屏，标题 + 六格指标 + 技术栈标签
- `MetricItem({ value, unit, label, suffix? })` → 单个指标，含计数动画

- [ ] **Step 1: Create src/components/MetricItem.jsx**

```jsx
import { motion, useMotionValue, useTransform, animate } from 'framer-motion'
import { useEffect } from 'react'
import { useInView } from 'framer-motion'
import { useRef } from 'react'

export default function MetricItem({ value, unit, label, suffix, decimals = 0 }) {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: '-60px' })
  const count = useMotionValue(0)
  const rounded = useTransform(count, (latest) => latest.toFixed(decimals))

  useEffect(() => {
    if (isInView) {
      const numericValue = parseFloat(value)
      animate(count, numericValue, { duration: 1.5, ease: 'easeOut' })
    }
  }, [isInView, count, value])

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.5 }}
      className="flex flex-col items-center p-8 rounded-2xl border border-white/[0.06]
                 hover:border-tech-cyan/20 transition-all duration-500"
    >
      <div className="flex items-baseline gap-1 mb-2">
        {value.startsWith('<') || value.startsWith('>') ? (
          <>
            <span className="text-lg text-zinc-500 font-mono">{value[0]}</span>
            <motion.span className="text-5xl md:text-6xl font-bold font-mono text-tech-cyan">
              {value.slice(1)}
            </motion.span>
          </>
        ) : (
          <>
            <motion.span className="text-5xl md:text-6xl font-bold font-mono text-tech-cyan">
              {rounded}
            </motion.span>
            {suffix && <span className="text-lg text-zinc-400 font-mono">{suffix}</span>}
          </>
        )}
      </div>
      <span className="text-sm text-zinc-500 font-mono">{unit}</span>
      <span className="text-xs text-zinc-600 mt-1">{label}</span>
    </motion.div>
  )
}
```

- [ ] **Step 2: Create src/components/MetricsSection.jsx**

```jsx
import { motion } from 'framer-motion'
import MetricItem from './MetricItem'

const METRICS = [
  { value: '<10', unit: 'M', label: '模型参数量' },
  { value: '<8', unit: 'MB', label: 'INT8 文件体积' },
  { value: '<80', unit: 'ms', label: 'ARM 单核推理' },
  { value: '<3', unit: 'min', label: '12寸晶圆总耗时' },
  { value: '>0.80', unit: 'mAP@0.5', label: '检测精度' },
  { value: '>25', unit: 'dB', label: '增强 PSNR' },
]

export default function MetricsSection() {
  return (
    <section id="metrics" className="min-h-screen flex items-center py-32">
      <div className="max-w-content mx-auto px-8 w-full">
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <p className="text-xs tracking-[0.3em] text-tech-cyan/40 font-mono mb-4">BENCHMARKS</p>
          <h2 className="text-4xl md:text-5xl font-light text-white mb-4">
            硬核指标 · 量产级标准
          </h2>
          <p className="text-zinc-500 text-sm tracking-wide">
            六项 P0 硬约束，全部在 ARM 边缘端达标
          </p>
        </motion.div>

        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
          {METRICS.map((m) => (
            <MetricItem key={m.label} {...m} />
          ))}
        </div>
      </div>
    </section>
  )
}
```

---

### Task 8: FooterSection — 全屏收尾

**Files:**
- Create: `D:/cy/wafer-showcase/src/components/FooterSection.jsx`

**Interfaces:**
- `FooterSection` → 全屏收尾，呼应对应 Hero，含总结 + 按钮 + 版权

- [ ] **Step 1: Create src/components/FooterSection.jsx**

```jsx
import { motion } from 'framer-motion'
import { ArrowUpRight, Github, FileText, Mail } from 'lucide-react'

export default function FooterSection() {
  return (
    <section className="relative h-screen flex items-center justify-center overflow-hidden">
      {/* Subtle background */}
      <div className="absolute inset-0 bg-gradient-to-t from-tech-deep via-tech-deep/95 to-tech-deep" />
      <div className="absolute inset-0 opacity-[0.03]">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 rounded-full bg-tech-cyan blur-[120px]" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 rounded-full bg-tech-purple blur-[120px]" />
      </div>

      <div className="relative z-10 max-w-content mx-auto px-8 text-center">
        <motion.p
          className="text-xs tracking-[0.3em] text-tech-cyan/40 font-mono mb-6"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          FROM LAB TO PRODUCTION
        </motion.p>

        <motion.h2
          className="text-4xl md:text-6xl font-light text-white mb-6"
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.15 }}
        >
          从标注 <span className="text-tech-cyan font-semibold">50 张</span>
          <br className="md:hidden" /> 到量产部署
        </motion.h2>

        <motion.p
          className="text-zinc-500 text-sm tracking-wider mb-12"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.3 }}
        >
          国产光源 · ARM 边缘推理 · 全链路自主可控
        </motion.p>

        <motion.a
          href="#"
          className="inline-flex items-center gap-2 px-8 py-3 rounded-full border border-white/10
                     text-sm text-zinc-300 hover:text-tech-cyan hover:border-tech-cyan/30
                     transition-all duration-500 group mb-16"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.45 }}
        >
          查看完整技术路线文档
          <ArrowUpRight size={16} className="group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform duration-300" />
        </motion.a>

        <motion.div
          className="flex justify-center gap-6 mb-16"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.6 }}
        >
          {[
            { icon: Github, href: '#', label: 'GitHub' },
            { icon: FileText, href: '#', label: '技术博客' },
            { icon: Mail, href: '#', label: '联系' },
          ].map(({ icon: Icon, href, label }) => (
            <a
              key={label}
              href={href}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-zinc-500 hover:text-tech-cyan
                         border border-white/[0.06] hover:border-tech-cyan/20 transition-all duration-300 text-sm"
            >
              <Icon size={16} />
              {label}
            </a>
          ))}
        </motion.div>

        <motion.p
          className="text-xs text-zinc-700 font-mono"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.75 }}
        >
          © 2026 东北大学秦皇岛分校 · 晶圆缺陷检测项目
        </motion.p>
      </div>
    </section>
  )
}
```

---

### Task 9: 组装 App.jsx — 串联所有组件

**Files:**
- Modify: `D:/cy/wafer-showcase/src/App.jsx`

**Interfaces:**
- Consumes: HeroSection, ChallengesSection, ArchitectureSection, HighlightsSection, MetricsSection, FooterSection

- [ ] **Step 1: Rewrite src/App.jsx**

```jsx
import HeroSection from './components/HeroSection'
import ChallengesSection from './components/ChallengesSection'
import ArchitectureSection from './components/ArchitectureSection'
import HighlightsSection from './components/HighlightsSection'
import MetricsSection from './components/MetricsSection'
import FooterSection from './components/FooterSection'

export default function App() {
  return (
    <main className="bg-tech-deep text-white selection:bg-tech-cyan/20 selection:text-white">
      <HeroSection />
      <ChallengesSection />
      <ArchitectureSection />
      <HighlightsSection />
      <MetricsSection />
      <FooterSection />
    </main>
  )
}
```

- [ ] **Step 2: Run dev server to verify**

```bash
cd "D:/cy/wafer-showcase"
npm run dev
```

Expected: Browser opens at localhost:5173, 6 屏流畅滚动展示，所有动画正常触发。

- [ ] **Step 3: Final build check**

```bash
npm run build
```

Expected: `vite build` 成功，输出到 `dist/` 目录，无报错。
