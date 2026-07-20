import { Cpu, FileText, Zap, Clock, Target, Eye } from 'lucide-react'
import MetricItem from './MetricItem'

const METRICS = [
  {
    value: 7.8,
    label: '参数数量',
    sublabel: 'Parameters',
    unit: 'B',
    decimals: 1,
    icon: Cpu,
  },
  {
    value: 12847,
    label: '文件处理',
    sublabel: 'Files Processed',
    decimals: 0,
    icon: FileText,
  },
  {
    value: 45,
    label: '推理速度',
    sublabel: 'Inference Speed',
    unit: 'ms',
    decimals: 0,
    icon: Zap,
  },
  {
    value: 72,
    label: '训练耗时',
    sublabel: 'Training Time',
    unit: 'h',
    decimals: 0,
    icon: Clock,
  },
  {
    value: 89.6,
    label: 'mAP',
    sublabel: 'Mean Average Precision',
    unit: '%',
    decimals: 1,
    icon: Target,
  },
  {
    value: 38.2,
    label: 'PSNR',
    sublabel: 'Peak Signal-to-Noise Ratio',
    unit: 'dB',
    decimals: 1,
    icon: Eye,
  },
]

export default function MetricsSection() {
  return (
    <section id="metrics" className="relative py-24">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-tech-deep via-tech-card to-tech-deep" />

      <div className="relative mx-auto max-w-content px-8">
        {/* Section header */}
        <div className="mb-16 text-center">
          <span className="inline-block rounded-full border border-tech-cyan/20 bg-tech-cyan/5 px-4 py-1 text-xs font-medium tracking-widest text-tech-cyan uppercase">
            Key Metrics
          </span>
          <h2 className="mt-6 text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-tech-cyan to-tech-purple sm:text-4xl lg:text-5xl">
            关键指标
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-zinc-400">
            经过大量实验验证，WaferAI
            在多个维度上达到了行业领先水平
          </p>
        </div>

        {/* Metrics grid */}
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {METRICS.map((metric) => (
            <MetricItem key={metric.label} {...metric} />
          ))}
        </div>
      </div>
    </section>
  )
}
