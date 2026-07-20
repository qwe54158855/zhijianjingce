import { ArrowUpRight, Github, FileText, Mail, Sparkles, Database, ArrowRight } from 'lucide-react'

export default function FooterSection({ onNavigate }) {
  return (
    <footer className="relative min-h-screen flex flex-col items-center justify-center overflow-hidden bg-tech-deep">
      {/* Subtle gradient background blobs */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none" aria-hidden="true">
        <div className="absolute -top-40 -left-40 w-[500px] h-[500px] rounded-full bg-tech-cyan/5 blur-[120px]" />
        <div className="absolute -bottom-40 -right-40 w-[600px] h-[600px] rounded-full bg-tech-purple/5 blur-[120px]" />
      </div>

      {/* Main content */}
      <div className="relative z-10 flex flex-col items-center gap-10 px-6 text-center">
        {/* Heading */}
        <h2 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-tech-cyan to-tech-purple">
          从标注 50 张到量产部署
        </h2>

        {/* Subtitle */}
        <p className="text-lg md:text-xl text-zinc-400 max-w-2xl leading-relaxed tracking-wide">
          国产光源 · ARM 边缘推理 · 全链路自主可控
        </p>

        {/* CTA button */}
        <a
          href="#"
          className="inline-flex items-center gap-2 px-8 py-4 rounded-full bg-tech-cyan/10 border border-tech-cyan/30 text-tech-cyan hover:bg-tech-cyan/20 transition-all duration-300 group text-base font-medium tracking-wide"
        >
          查看完整技术路线文档
          <ArrowUpRight className="w-5 h-5 transition-transform group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
        </a>

        {/* Link buttons row */}
        <div className="flex items-center gap-4">
          <a
            href="#"
            className="p-3 rounded-full border border-white/10 text-zinc-400 hover:text-tech-cyan hover:border-tech-cyan/30 transition-all duration-300"
            aria-label="GitHub"
          >
            <Github className="w-5 h-5" />
          </a>
          <a
            href="#"
            className="p-3 rounded-full border border-white/10 text-zinc-400 hover:text-tech-cyan hover:border-tech-cyan/30 transition-all duration-300"
            aria-label="Documentation"
          >
            <FileText className="w-5 h-5" />
          </a>
          <a
            href="#"
            className="p-3 rounded-full border border-white/10 text-zinc-400 hover:text-tech-cyan hover:border-tech-cyan/30 transition-all duration-300"
            aria-label="Contact"
          >
            <Mail className="w-5 h-5" />
          </a>
        </div>

        {/* Dataset Browser entry */}
        <button
          onClick={() => onNavigate?.('dataset')}
          className="group w-full max-w-lg flex items-center justify-between px-6 py-4 rounded-2xl border border-emerald-500/20
                     bg-gradient-to-r from-emerald-500/10 to-cyan-500/10 backdrop-blur-sm
                     hover:from-emerald-500/20 hover:to-cyan-500/20 hover:border-emerald-500/40
                     transition-all duration-500"
        >
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-xl bg-emerald-500/10 group-hover:scale-110 transition-transform duration-300">
              <Database className="w-6 h-6 text-emerald-400" />
            </div>
            <div className="text-left">
              <h3 className="text-base sm:text-lg font-semibold text-white group-hover:text-emerald-400 transition-colors duration-300">
                数据集浏览器
              </h3>
              <p className="text-xs sm:text-sm text-zinc-400 mt-0.5">
                浏览 1150 张标注图像 · Qwen-VL AI分类分析 · 批量评估
              </p>
            </div>
          </div>
          <div className="flex items-center gap-1 text-sm text-emerald-400 group-hover:translate-x-1 transition-transform duration-300">
            <span>浏览数据</span>
            <ArrowRight size={16} />
          </div>
        </button>

        {/* AI Navigation entry */}
        <button
          onClick={() => onNavigate?.('qwen')}
          className="group w-full max-w-lg flex items-center justify-between px-6 py-4 rounded-2xl border border-tech-cyan/20
                     bg-gradient-to-r from-tech-cyan/10 to-purple-500/10 backdrop-blur-sm
                     hover:from-tech-cyan/20 hover:to-purple-500/20 hover:border-tech-cyan/40
                     transition-all duration-500"
        >
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-xl bg-tech-cyan/10 group-hover:scale-110 transition-transform duration-300">
              <Sparkles className="w-6 h-6 text-tech-cyan" />
            </div>
            <div className="text-left">
              <h3 className="text-base sm:text-lg font-semibold text-white group-hover:text-tech-cyan transition-colors duration-300">
                AI 智能检测
              </h3>
              <p className="text-xs sm:text-sm text-zinc-400 mt-0.5">
                AI智能增强 · OpenCV形态学检测 · YOLO风格可视化
              </p>
            </div>
          </div>
          <div className="flex items-center gap-1 text-sm text-tech-cyan group-hover:translate-x-1 transition-transform duration-300">
            <span>开始体验</span>
            <ArrowRight size={16} />
          </div>
        </button>
      </div>

      {/* Copyright */}
      <div className="absolute bottom-8 left-0 right-0 text-center">
        <p className="text-xs text-zinc-600 tracking-wide">
          &copy; 2026 东北大学秦皇岛分校 &middot; 晶圆缺陷检测项目
        </p>
      </div>
    </footer>
  )
}
