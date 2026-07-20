import { useState } from 'react'
import { motion } from 'framer-motion'
import { FileText, Loader2, AlertCircle } from 'lucide-react'

export default function QwenReportPanel({ onGenerate, loading, report, error }) {
  const [hasClicked, setHasClicked] = useState(false)

  const handleClick = () => {
    setHasClicked(true)
    onGenerate?.()
  }

  if (!hasClicked && !report) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="rounded-xl border border-white/10 bg-tech-card p-6"
      >
        <div className="flex items-center gap-3 mb-4">
          <FileText className="w-5 h-5 text-tech-cyan" />
          <h3 className="text-sm font-medium text-zinc-300">AI 检测报告</h3>
        </div>
        <p className="text-xs text-zinc-500 mb-5 leading-relaxed">
          基于增强分析结果，自动生成结构化缺陷检测报告，包含缺陷类型、位置分布与严重等级评估。
        </p>
        <button
          onClick={handleClick}
          disabled={loading}
          className="w-full py-2.5 rounded-xl font-medium text-sm bg-gradient-to-r from-tech-cyan to-purple-500 text-white hover:opacity-90 disabled:opacity-50 transition-all flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              报告生成中...
            </>
          ) : (
            <>
              <FileText className="w-4 h-4" />
              生成 AI 检测报告
            </>
          )}
        </button>
      </motion.div>
    )
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="rounded-xl border border-white/10 bg-tech-card overflow-hidden"
    >
      <div className="flex items-center justify-between px-5 py-3 border-b border-white/5">
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-tech-cyan" />
          <h3 className="text-sm font-medium text-zinc-300">AI 检测报告</h3>
        </div>
        {report && (
          <button
            onClick={handleClick}
            disabled={loading}
            className="text-xs text-tech-cyan hover:text-white transition-colors disabled:opacity-50"
          >
            {loading ? '重新生成中...' : '重新生成'}
          </button>
        )}
      </div>

      {loading && (
        <div className="flex flex-col items-center justify-center py-12 gap-3 text-zinc-500">
          <Loader2 className="w-6 h-6 animate-spin text-tech-cyan" />
          <p className="text-sm">AI 正在生成报告...</p>
          <p className="text-xs text-zinc-600">结构化报告生成中</p>
        </div>
      )}

      {error && (
        <div className="flex items-start gap-3 p-5">
          <AlertCircle className="w-5 h-5 text-red-400 mt-0.5 shrink-0" />
          <div>
            <p className="text-sm text-red-400 font-medium mb-1">报告生成失败</p>
            <p className="text-xs text-zinc-500">{error}</p>
          </div>
        </div>
      )}

      {report && !loading && (
        <div className="p-5">
          <div className="bg-black/30 rounded-lg p-4 max-h-[400px] overflow-y-auto">
            <pre className="text-sm text-zinc-300 leading-relaxed whitespace-pre-wrap font-sans">
              {report}
            </pre>
          </div>
        </div>
      )}
    </motion.div>
  )
}
