import { Sparkles, Waves, BugPlay } from 'lucide-react'
import { cn } from '../utils/cn'

const MODELS = [
  {
    id: 'enhance',
    label: '暗场 → 明场增强',
    desc: '提高缺陷可见度，模拟高倍明场检测效果',
    icon: Sparkles,
    color: 'text-tech-cyan',
    gradient: 'from-tech-cyan/20 to-transparent',
  },
  {
    id: 'wavelength',
    label: '266nm → 193nm 波长转换',
    desc: '虚拟深紫外增强，Rayleigh 散射增益 ~3.5×',
    icon: Waves,
    color: 'text-purple-400',
    gradient: 'from-purple-500/20 to-transparent',
  },
  {
    id: 'defect',
    label: '缺陷生成',
    desc: '在无缺陷晶圆上合成逼真的缺陷图像',
    icon: BugPlay,
    color: 'text-emerald-400',
    gradient: 'from-emerald-500/20 to-transparent',
  },
]

export default function ModelSelector({ selected, onSelect, disabled }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      {MODELS.map((model) => {
        const Icon = model.icon
        const isActive = selected === model.id
        return (
          <button
            key={model.id}
            disabled={disabled}
            onClick={() => onSelect(model.id)}
            className={cn(
              'relative p-5 rounded-xl border text-left transition-all',
              'border-white/10 bg-white/5 backdrop-blur-sm',
              isActive
                ? 'border-tech-cyan/50 shadow-[0_0_20px_rgba(0,229,255,0.1)]'
                : 'hover:border-white/20 hover:-translate-y-0.5',
              disabled && 'opacity-50 cursor-not-allowed',
            )}
          >
            <Icon className={cn('mb-3', model.color)} size={28} />
            <h3 className={cn('font-semibold mb-1', isActive ? 'text-white' : 'text-zinc-300')}>
              {model.label}
            </h3>
            <p className="text-sm text-zinc-500">{model.desc}</p>
            {isActive && (
              <div className={cn(
                'absolute inset-0 rounded-xl bg-gradient-to-b opacity-20 pointer-events-none',
                model.gradient,
              )} />
            )}
          </button>
        )
      })}
    </div>
  )
}
