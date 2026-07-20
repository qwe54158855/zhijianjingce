import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import { AlertTriangle, Cpu, Sparkles, BarChart3, ArrowRight } from 'lucide-react'
import UnicornScene from 'unicornstudio-react'
import Navbar from './Navbar'

const NAV_CARDS = [
  {
    icon: AlertTriangle,
    title: '三大核心难题',
    desc: '信噪比不足、算力受限、数据稀缺',
    href: '#challenges',
    color: 'from-rose-500/20 to-rose-500/5',
    iconBg: 'bg-rose-500/10',
    iconColor: 'text-rose-400',
  },
  {
    icon: Cpu,
    title: '技术架构总览',
    desc: '共享编码器 + 双解码分支',
    href: '#architecture',
    color: 'from-tech-cyan/20 to-tech-cyan/5',
    iconBg: 'bg-tech-cyan/10',
    iconColor: 'text-tech-cyan',
  },
  {
    icon: Sparkles,
    title: '核心技术亮点',
    desc: '七项创新，从预训练到在线校准',
    href: '#highlights',
    color: 'from-purple-500/20 to-purple-500/5',
    iconBg: 'bg-purple-500/10',
    iconColor: 'text-purple-400',
  },
  {
    icon: BarChart3,
    title: '关键指标面板',
    desc: '六项 P0 硬约束量产级达标',
    href: '#metrics',
    color: 'from-emerald-500/20 to-emerald-500/5',
    iconBg: 'bg-emerald-500/10',
    iconColor: 'text-emerald-400',
  },
]

export default function HeroSection({ onNavigate }) {
  const containerRef = useRef(null)
  const [dimensions, setDimensions] = useState({ width: 1440, height: 900 })

  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.offsetWidth,
          height: containerRef.current.offsetHeight,
        })
      }
    }
    updateSize()
    window.addEventListener('resize', updateSize)
    return () => window.removeEventListener('resize', updateSize)
  }, [])

  return (
    <section id="hero" className="relative min-h-screen overflow-hidden">
      {/* Unicorn 3D dynamic background */}
      <div ref={containerRef} className="absolute inset-0 w-full h-full">
        <UnicornScene
          projectId="GB3z8XtyG0Cbyy6QSoDI"
          width={`${dimensions.width}px`}
          height={`${dimensions.height}px`}
          scale={1}
          dpi={1.5}
          sdkUrl="https://cdn.jsdelivr.net/gh/hiunicornstudio/unicornstudio.js@v2.2.6/dist/unicornStudio.umd.js"
        />
      </div>

      {/* Dark overlay for readability on 3D background */}
      <div className="absolute inset-0 bg-gradient-to-b from-black/60 via-black/30 to-black/70" />

      <Navbar />

      <div className="relative z-10 flex flex-col items-center justify-center min-h-screen px-6 py-20">
        <div className="max-w-content mx-auto w-full">
          {/* Top content area */}
          <div className="text-center mb-12">
            {/* tagline */}
            <motion.span
              className="inline-block px-4 py-1.5 mb-6 text-xs tracking-[0.2em] text-tech-cyan border border-tech-cyan/20 rounded-full bg-tech-cyan/5 backdrop-blur-sm"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              东北大学秦皇岛分校 · 半导体检测项目
            </motion.span>

            {/* main heading */}
            <motion.h1
              className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold leading-tight tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-tech-cyan to-tech-purple"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.4 }}
            >
              半导体晶圆缺陷检测 · 一体化AI 解决方案
            </motion.h1>

            {/* subtitle */}
            <motion.p
              className="max-w-3xl mx-auto mt-6 text-base sm:text-lg text-gray-200 leading-relaxed"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.8, delay: 0.6 }}
            >
              国产光源暗场增强 · ARM 边缘端实时推理 · 三阶段半监督训练
            </motion.p>
          </div>

          {/* Navigation card grid — the main interface */}
          <motion.div
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 max-w-5xl mx-auto"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.9 }}
          >
            {NAV_CARDS.map((card) => (
              <a
                key={card.href}
                href={card.href}
                className="group relative flex flex-col items-start p-6 rounded-2xl border border-white/10 bg-white/5
                           hover:bg-white/10 backdrop-blur-sm transition-all duration-500
                           hover:border-tech-cyan/30 hover:-translate-y-1"
              >
                {/* Icon */}
                <div className={`mb-4 p-3 rounded-xl ${card.iconBg} group-hover:scale-110 transition-transform duration-300`}>
                  <card.icon className={`w-6 h-6 ${card.iconColor}`} />
                </div>

                {/* Title */}
                <h3 className="text-lg font-semibold text-white mb-1.5 group-hover:text-tech-cyan transition-colors duration-300">
                  {card.title}
                </h3>

                {/* Description */}
                <p className="text-sm text-zinc-400 leading-relaxed mb-4">
                  {card.desc}
                </p>

                {/* Arrow indicator */}
                <div className="mt-auto flex items-center gap-1 text-xs text-zinc-500 group-hover:text-tech-cyan transition-colors duration-300">
                  <span>进入</span>
                  <ArrowRight size={14} className="group-hover:translate-x-1 transition-transform duration-300" />
                </div>
              </a>
            ))}
          </motion.div>

        </div>
      </div>
    </section>
  )
}
