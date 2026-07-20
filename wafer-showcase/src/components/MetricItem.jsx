import { motion, useMotionValue, useTransform, animate, useInView } from 'framer-motion'
import { useEffect, useRef } from 'react'
import { cn } from '../utils/cn'

export default function MetricItem({
  value,
  label,
  sublabel = '',
  unit = '',
  prefix = '',
  decimals = 0,
  icon: Icon,
  className,
}) {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: '-50px' })
  const count = useMotionValue(0)

  const displayValue = useTransform(count, (latest) => {
    // Format with locale-aware separators for whole numbers, fixed decimals otherwise
    const formatted =
      decimals > 0
        ? latest.toFixed(decimals)
        : Math.round(latest).toLocaleString()
    return prefix + formatted + unit
  })

  useEffect(() => {
    if (isInView) {
      const controls = animate(count, value, {
        duration: 2,
        ease: 'easeOut',
      })
      return () => controls.stop()
    }
  }, [isInView, count, value])

  return (
    <div
      ref={ref}
      className={cn(
        'group relative overflow-hidden rounded-xl border border-tech-border bg-tech-card p-6 transition-all duration-700',
        isInView ? 'translate-y-0 opacity-100' : 'translate-y-8 opacity-0',
        'hover:border-tech-cyan/30 hover:shadow-[0_0_30px_rgba(0,229,255,0.05)]',
        className,
      )}
    >
      {/* Hover glow gradient */}
      <div className="pointer-events-none absolute -inset-0.5 rounded-xl opacity-0 transition-opacity duration-300 group-hover:opacity-100">
        <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-tech-cyan/5 via-transparent to-tech-purple/5" />
      </div>

      {/* Icon */}
      {Icon && (
        <div className="relative mb-4 flex h-10 w-10 items-center justify-center rounded-lg bg-tech-cyan/10">
          <Icon className="h-5 w-5 text-tech-cyan" />
        </div>
      )}

      {/* Animated value */}
      <div className="relative">
        <motion.span className="text-3xl font-bold tracking-tight text-white tabular-nums sm:text-4xl">
          {displayValue}
        </motion.span>
      </div>

      {/* Label */}
      <p className="relative mt-2 text-sm text-zinc-400">{label}</p>

      {/* Sublabel */}
      {sublabel && (
        <p className="relative mt-0.5 text-xs text-zinc-500">{sublabel}</p>
      )}
    </div>
  )
}
