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
          <h2 className="text-4xl md:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-tech-cyan to-tech-purple mb-4">
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
