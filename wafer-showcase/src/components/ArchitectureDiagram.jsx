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
