import { motion } from 'framer-motion'
import { cn } from '../utils/cn'

export default function HighlightCard({
  icon: Icon,
  title,
  description,
  metrics = [],
  index = 0,
  horizontal = false,
  className,
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ duration: 0.5, delay: index * 0.08 }}
      className={cn(
        'group relative overflow-hidden rounded-2xl border border-tech-border bg-tech-card p-6 transition-all duration-300 hover:border-tech-cyan/30 hover:shadow-[0_0_30px_rgba(0,229,255,0.06)]',
        horizontal && 'md:flex md:items-start md:gap-6 md:p-8',
        className
      )}
    >
      {/* Hover glow */}
      <div className="pointer-events-none absolute -inset-1 bg-gradient-to-r from-tech-cyan/0 via-tech-cyan/5 to-tech-purple/0 opacity-0 blur-xl transition-opacity duration-500 group-hover:opacity-100" />

      <div className="relative z-10">
        {/* Icon */}
        <div
          className={cn(
            'mb-4 inline-flex h-10 w-10 items-center justify-center rounded-lg bg-tech-cyan/10',
            horizontal && 'md:mb-0 md:shrink-0'
          )}
        >
          <Icon className="h-5 w-5 text-tech-cyan" />
        </div>

        {/* Content */}
        <div className={cn(horizontal && 'md:flex md:flex-1 md:items-start md:gap-6')}>
          <div className={cn('flex-1', horizontal && 'md:pt-1')}>
            <h3 className="mb-2 text-lg font-semibold text-white">{title}</h3>
            <p className="mb-4 text-sm leading-relaxed text-zinc-400">{description}</p>
          </div>

          {/* Metrics tags */}
          {metrics.length > 0 && (
            <div
              className={cn(
                'flex flex-wrap gap-2',
                horizontal && 'md:shrink-0 md:pt-1'
              )}
            >
              {metrics.map((m, i) => (
                <span
                  key={i}
                  className="inline-flex items-center rounded-full border border-tech-cyan/20 bg-tech-cyan/5 px-2.5 py-0.5 text-xs font-medium text-tech-cyan"
                >
                  {m}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>
    </motion.div>
  )
}
