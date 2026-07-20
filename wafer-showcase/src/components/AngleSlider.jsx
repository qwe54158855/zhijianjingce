import { useCallback, useRef } from 'react'
import { motion } from 'framer-motion'
import { RotateCcw } from 'lucide-react'

const ANGLES = Array.from({ length: 13 }, (_, i) => 15 + i * 5) // 15,20,...,75

export default function AngleSlider({
  angleViews,
  currentAngle,
  onAngleChange,
  loading,
}) {
  const sliderRef = useRef(null)

  const handleSliderChange = useCallback((e) => {
    const val = parseInt(e.target.value, 10)
    onAngleChange(val)
  }, [onAngleChange])

  if (!angleViews && !loading) return null

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="rounded-xl border border-tech-cyan/20 bg-tech-card p-5"
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <RotateCcw className="w-4 h-4 text-tech-cyan" />
          <h3 className="text-sm font-medium text-zinc-300">多角度 193nm 亮场视图</h3>
        </div>
        <span className="text-xs text-tech-cyan font-mono">
          当前: {currentAngle}°
        </span>
      </div>

      {/* Angle image display */}
      <div className="flex items-center justify-center min-h-[160px] bg-black/40 rounded-lg mb-4 overflow-hidden">
        {loading ? (
          <div className="flex flex-col items-center gap-2 text-zinc-500">
            <div className="w-6 h-6 border-2 border-tech-cyan/40 border-t-tech-cyan rounded-full animate-spin" />
            <span className="text-xs">生成角度视图...</span>
          </div>
        ) : angleViews && angleViews[String(currentAngle)] ? (
          <img
            src={angleViews[String(currentAngle)]}
            alt={`${currentAngle}° 角度视图`}
            className="max-w-full max-h-[280px] object-contain"
            draggable={false}
          />
        ) : (
          <span className="text-zinc-600 text-xs">拖拽滑块查看不同角度</span>
        )}
      </div>

      {/* Slider */}
      <div className="flex items-center gap-3">
        <span className="text-xs text-zinc-500 w-6 text-right">15°</span>
        <div className="relative flex-1">
          <input
            ref={sliderRef}
            type="range"
            min="15"
            max="75"
            step="5"
            value={currentAngle}
            onChange={handleSliderChange}
            className="w-full h-2 rounded-full appearance-none cursor-pointer
                       bg-gradient-to-r from-tech-cyan/30 via-tech-cyan to-purple-500
                       [&::-webkit-slider-thumb]:appearance-none
                       [&::-webkit-slider-thumb]:w-5
                       [&::-webkit-slider-thumb]:h-5
                       [&::-webkit-slider-thumb]:rounded-full
                       [&::-webkit-slider-thumb]:bg-white
                       [&::-webkit-slider-thumb]:shadow-lg
                       [&::-webkit-slider-thumb]:shadow-tech-cyan/30
                       [&::-webkit-slider-thumb]:border-2
                       [&::-webkit-slider-thumb]:border-tech-cyan
                       [&::-webkit-slider-thumb]:cursor-pointer"
          />
          {/* Tick marks */}
          <div className="flex justify-between mt-1 px-0.5">
            {ANGLES.filter(a => a % 10 === 0).map(a => (
              <span
                key={a}
                className={`text-[10px] ${currentAngle === a ? 'text-tech-cyan' : 'text-zinc-600'}`}
              >
                {a}°
              </span>
            ))}
          </div>
        </div>
        <span className="text-xs text-zinc-500 w-6">75°</span>
      </div>

      {/* Description */}
      <p className="text-[11px] text-zinc-600 mt-3 leading-relaxed">
        极坐标角度散射模拟 — 滚动滑块切换 15°-75° 不同方位角的 193nm 亮场增强效果。
        角度越大，散射增益越强，微小缺陷信号越明显。
      </p>
    </motion.div>
  )
}
