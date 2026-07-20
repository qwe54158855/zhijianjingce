import { cn } from '../utils/cn'

const STEPS = [
  { key: 'upload', label: '上传图片' },
  { key: 'infer', label: 'AI 推理' },
  { key: 'result', label: '查看结果' },
]

export default function ProgressTracker({ currentStep }) {
  const stepIndex = { idle: 0, uploading: 0, running: 1, done: 2, error: 1 }

  return (
    <div className="flex items-center justify-center gap-2 mb-6">
      {STEPS.map((step, i) => {
        const isActive = i <= stepIndex[currentStep]
        const isLast = i === STEPS.length - 1
        return (
          <div key={step.key} className="flex items-center gap-2">
            <div className={cn(
              'flex items-center gap-2 px-3 py-1.5 rounded-full text-sm transition-all',
              isActive ? 'bg-tech-cyan/20 text-tech-cyan' : 'bg-white/5 text-zinc-600',
            )}>
              <div className={cn(
                'w-5 h-5 rounded-full flex items-center justify-center text-xs font-bold',
                isActive ? 'bg-tech-cyan text-white' : 'bg-zinc-700 text-zinc-500',
              )}>
                {i + 1}
              </div>
              <span>{step.label}</span>
            </div>
            {!isLast && (
              <div className={cn(
                'w-8 h-0.5',
                i < stepIndex[currentStep] ? 'bg-tech-cyan' : 'bg-white/10',
              )} />
            )}
          </div>
        )
      })}
    </div>
  )
}
