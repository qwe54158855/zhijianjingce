import { motion } from 'framer-motion'

export default function ChallengeCard({ icon: Icon, title, description, index = 0 }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 40 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-50px' }}
      transition={{ duration: 0.6, delay: index * 0.15, ease: 'easeOut' }}
      className="group relative bg-tech-card border border-tech-border rounded-2xl p-8 transition-all duration-500 hover:border-tech-cyan/30 hover:shadow-[0_0_30px_rgba(0,229,255,0.08)]"
    >
      {/* Hover glow overlay */}
      <div className="absolute inset-0 rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none bg-gradient-to-br from-tech-cyan/[0.03] to-transparent" />

      {/* Icon */}
      <div className="relative mb-6 w-12 h-12 rounded-xl bg-tech-cyan/10 flex items-center justify-center group-hover:bg-tech-cyan/20 group-hover:shadow-[0_0_20px_rgba(0,229,255,0.15)] transition-all duration-300">
        <Icon className="w-6 h-6 text-tech-cyan" />
      </div>

      {/* Title */}
      <h3 className="relative text-xl font-semibold text-white/90 mb-3 group-hover:text-white transition-colors duration-300">
        {title}
      </h3>

      {/* Description */}
      <p className="relative text-sm text-zinc-400 leading-relaxed group-hover:text-zinc-300 transition-colors duration-300">
        {description}
      </p>

      {/* Animated bottom accent line */}
      <div className="relative mt-6 h-px bg-gradient-to-r from-transparent via-tech-cyan/20 to-transparent scale-x-0 group-hover:scale-x-100 transition-transform duration-500" />
    </motion.div>
  )
}
