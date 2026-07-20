import { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { AlertTriangle, CheckCircle, Loader2, Square } from 'lucide-react'

const TYPE_CONFIG = {
  '崩边': { label: '崩边', color: 'red',    hex: '#ef4444', bg: 'bg-red-500/10', text: 'text-red-400', border: 'border-red-500' },
  '颗粒': { label: '颗粒', color: 'cyan',   hex: '#00e5ff', bg: 'bg-tech-cyan/10', text: 'text-tech-cyan', border: 'border-tech-cyan' },
  '划痕': { label: '划痕', color: 'green',  hex: '#22c55e', bg: 'bg-green-500/10', text: 'text-green-400', border: 'border-green-500' },
  '位错': { label: '位错', color: 'purple', hex: '#a855f7', bg: 'bg-purple-500/10', text: 'text-purple-400', border: 'border-purple-500' },
}

const TYPE_ICONS = {
  '崩边': AlertTriangle,
  '颗粒': Square,
  '划痕': AlertTriangle,
  '位错': AlertTriangle,
}

function DetectionBboxOverlay({ imageUrl, detections, imageNaturalSize }) {
  const imgRef = useRef(null)
  const [displaySize, setDisplaySize] = useState({ w: 1, h: 1 })

  useEffect(() => {
    const img = imgRef.current
    if (!img) return
    const update = () => setDisplaySize({ w: img.offsetWidth, h: img.offsetHeight })
    update()
    window.addEventListener('resize', update)
    img.addEventListener('load', update)
    return () => {
      window.removeEventListener('resize', update)
      img.removeEventListener('load', update)
    }
  }, [imageUrl])

  const scaleX = displaySize.w / (imageNaturalSize?.w || 1)
  const scaleY = displaySize.h / (imageNaturalSize?.h || 1)

  return (
    <div className="relative inline-block w-full">
      <img
        ref={imgRef}
        src={imageUrl}
        alt="增强检测结果"
        className="w-full rounded-lg object-contain"
        draggable={false}
      />
      {detections.map((d, i) => {
        const cfg = TYPE_CONFIG[d.type] || TYPE_CONFIG['崩边']
        return (
          <div
            key={i}
            className="absolute border-2 rounded-sm pointer-events-none"
            style={{
              left: d.bbox.x * scaleX,
              top: d.bbox.y * scaleY,
              width: d.bbox.w * scaleX,
              height: d.bbox.h * scaleY,
              borderColor: cfg.hex,
            }}
          >
            <span
              className="absolute -top-5 left-0 text-[10px] leading-tight px-1 rounded-t"
              style={{ backgroundColor: cfg.hex, color: '#fff' }}
            >
              {cfg.label} {(d.confidence * 100).toFixed(3)}%
            </span>
          </div>
        )
      })}
    </div>
  )
}

function StatCard({ label, count, config, icon: Icon }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className={`rounded-xl border ${config.border} ${config.bg} p-4 flex flex-col items-center gap-2`}
    >
      <Icon className={`w-5 h-5 ${config.text}`} />
      <span className={`text-xs font-medium tracking-wide ${config.text}`}>
        {label}
      </span>
      <span className="text-2xl font-bold text-white">{count}</span>
    </motion.div>
  )
}

export default function QwenResultPanel({ originalImage, result, loading }) {
  if (loading) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="flex flex-col items-center justify-center py-16 gap-4 text-zinc-500"
      >
        <Loader2 className="w-8 h-8 animate-spin text-tech-cyan" />
        <p className="text-sm">AI 模型分析中，请稍候...</p>
        <p className="text-xs text-zinc-600">CPU 推理模式，约需 10-30 秒</p>
      </motion.div>
    )
  }

  if (!result) return null

  const { enhancedImage, detections = [] } = result
  // 从 detections 实时统计
  const stats = {}
  detections.forEach(d => {
    stats[d.type] = (stats[d.type] || 0) + 1
  })

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
      className="space-y-6"
    >
      {/* Three-column comparison */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <motion.div
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          className="rounded-xl border border-white/10 bg-tech-card p-4"
        >
          <h3 className="text-sm font-medium text-zinc-400 mb-3">
            <span className="inline-block w-2 h-2 rounded-full bg-amber-500 mr-2" />
            暗场原图
          </h3>
          <div className="relative flex items-center justify-center min-h-[180px] bg-black/40 rounded-lg overflow-hidden">
            {originalImage ? (
              <img
                src={originalImage}
                alt="原始暗场图像"
                className="max-w-full max-h-[280px] object-contain"
                draggable={false}
              />
            ) : (
              <span className="text-zinc-600 text-sm">无图像数据</span>
            )}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="rounded-xl border border-white/10 bg-tech-card p-4"
        >
          <h3 className="text-sm font-medium text-zinc-400 mb-3">
            <span className="inline-block w-2 h-2 rounded-full bg-tech-cyan mr-2" />
            亮场转化
            <span className="ml-1.5 text-[10px] text-tech-cyan/60">
              193nm图像下的高清晶圆转化图
            </span>
            {detections.length > 0 && (
              <span className="ml-2 text-tech-cyan text-xs">({detections.length} 处)</span>
            )}
          </h3>
          <div className="relative flex items-center justify-center min-h-[180px] bg-black/40 rounded-lg overflow-hidden">
            {enhancedImage ? (
              <DetectionBboxOverlay
                imageUrl={enhancedImage}
                detections={detections}
                imageNaturalSize={result?.imageSize}
              />
            ) : (
              <span className="text-zinc-600 text-sm">等待分析结果</span>
            )}
          </div>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="rounded-xl border border-tech-cyan/20 bg-tech-card p-4"
        >
          <h3 className="text-sm font-medium text-zinc-400 mb-3">
            <span className="inline-block w-2 h-2 rounded-full bg-green-400 mr-2" />
            亮场参考
          </h3>
          <div className="relative flex items-center justify-center min-h-[180px] bg-black/40 rounded-lg overflow-hidden">
            {result?.referenceImage ? (
              <img
                src={result.referenceImage}
                alt="亮场参考图"
                className="max-w-full max-h-[280px] object-contain"
                draggable={false}
              />
            ) : (
              <span className="text-zinc-600 text-sm">参考图加载中</span>
            )}
          </div>
        </motion.div>
      </div>

      {/* Statistics grid */}
      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="grid grid-cols-2 md:grid-cols-4 gap-3"
      >
        {Object.entries(TYPE_CONFIG).map(([key, cfg]) => (
          <StatCard
            key={key}
            label={cfg.label}
            count={stats[key] ?? 0}
            config={cfg}
            icon={TYPE_ICONS[key] || Square}
          />
        ))}
      </motion.div>

      {/* Detection detail list */}
      {detections.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="rounded-xl border border-white/10 bg-tech-card overflow-hidden"
        >
          <div className="px-5 py-3 border-b border-white/5">
            <h3 className="text-sm font-medium text-zinc-300">检测明细</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-white/5 text-zinc-500 text-xs uppercase tracking-wider">
                  <th className="text-left px-5 py-3 font-medium">类型</th>
                  <th className="text-left px-5 py-3 font-medium">置信度</th>
                  <th className="text-left px-5 py-3 font-medium">坐标 (x, y)</th>
                  <th className="text-left px-5 py-3 font-medium">尺寸 (w × h)</th>
                </tr>
              </thead>
              <tbody>
                {detections.map((d, i) => {
                  const cfg = TYPE_CONFIG[d.type] || TYPE_CONFIG['崩边']
                  return (
                    <tr
                      key={i}
                      className="border-b border-white/5 last:border-0 hover:bg-white/[0.02] transition-colors"
                    >
                      <td className="px-5 py-3">
                        <span className={`inline-flex items-center gap-1.5 ${cfg.text}`}>
                          <span
                            className="w-2 h-2 rounded-full inline-block"
                            style={{ backgroundColor: cfg.hex }}
                          />
                          {cfg.label}
                        </span>
                      </td>
                      <td className="px-5 py-3 text-zinc-300">
                        {(d.confidence * 100).toFixed(3)}%
                      </td>
                      <td className="px-5 py-3 text-zinc-400 font-mono text-xs">
                        ({Math.round(d.bbox.x)}, {Math.round(d.bbox.y)})
                      </td>
                      <td className="px-5 py-3 text-zinc-400 font-mono text-xs">
                        {Math.round(d.bbox.w)} × {Math.round(d.bbox.h)}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </motion.div>
      )}

      {/* No detections message */}
      {detections.length === 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="flex items-center justify-center gap-2 py-6 text-zinc-500 rounded-xl border border-white/5 bg-tech-card"
        >
          <CheckCircle className="w-5 h-5 text-green-400" />
          <span className="text-sm">未检测到明显缺陷</span>
        </motion.div>
      )}
    </motion.div>
  )
}
