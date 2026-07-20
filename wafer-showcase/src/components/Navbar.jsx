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
