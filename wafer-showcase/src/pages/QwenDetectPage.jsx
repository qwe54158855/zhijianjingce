import { useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import { Sparkles, ArrowLeft, Info, AlertCircle, ChevronRight, Loader2 } from 'lucide-react'
import QwenImageUploader from '../components/QwenImageUploader'
import QwenResultPanel from '../components/QwenResultPanel'
import QwenReportPanel from '../components/QwenReportPanel'
import AngleSlider from '../components/AngleSlider'
import { qwenEnhance, qwenReport, qwenAngles } from '../api/waferApi'

function readFileAsBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result)
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

export default function QwenDetectPage({ onBack }) {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [enhanceStyle, setEnhanceStyle] = useState('brightfield')
  const [enhancing, setEnhancing] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [generatingReport, setGeneratingReport] = useState(false)
  const [report, setReport] = useState(null)
  const [reportError, setReportError] = useState(null)
  const [angleViews, setAngleViews] = useState(null)
  const [currentAngle, setCurrentAngle] = useState(45)
  const [loadingAngles, setLoadingAngles] = useState(false)

  const handleFileSelect = useCallback((f) => {
    setFile(f)
    setResult(null)
    setError(null)
    setReport(null)
    setReportError(null)
    setAngleViews(null)
    setCurrentAngle(45)
    if (f) {
      setPreviewUrl(URL.createObjectURL(f))
    } else {
      if (previewUrl) URL.revokeObjectURL(previewUrl)
      setPreviewUrl(null)
    }
  }, [previewUrl])

  const loadAngleViews = async (base64) => {
    setLoadingAngles(true)
    try {
      const res = await qwenAngles(base64)
      if (res.success && res.angles) {
        const prefix = 'data:image/jpeg;base64,'
        const processed = {}
        Object.entries(res.angles).forEach(([angle, b64]) => {
          processed[angle] = `${prefix}${b64}`
        })
        setAngleViews(processed)
      }
    } catch (err) {
      console.warn('Failed to load angle views:', err)
    } finally {
      setLoadingAngles(false)
    }
  }

  const handleEnhance = async () => {
    if (!file) return
    setEnhancing(true)
    setError(null)
    setResult(null)
    setReport(null)
    setReportError(null)
    setAngleViews(null)
    setCurrentAngle(45)
    try {
      const dataUrl = await readFileAsBase64(file)
      const base64 = dataUrl.split(',')[1] || dataUrl
      const res = await qwenEnhance(base64, enhanceStyle)
      const prefix = 'data:image/jpeg;base64,'
      if (res.enhancedImage) {
        res.enhancedImage = `${prefix}${res.enhancedImage}`
      }
      if (res.referenceImage) {
        res.referenceImage = `${prefix}${res.referenceImage}`
      }
      res.originalBase64 = base64
      setResult(res)

      // Auto-load angle views for brightfield style
      if (enhanceStyle === 'brightfield') {
        loadAngleViews(base64)
      }
    } catch (err) {
      const message = err?.response?.data?.message || err?.message || 'AI 增强分析失败，请稍后重试'
      setError(message)
    } finally {
      setEnhancing(false)
    }
  }

  const handleGenerateReport = async () => {
    if (!file || !result) return
    setGeneratingReport(true)
    setReportError(null)
    try {
      const dataUrl = await readFileAsBase64(file)
      const base64 = dataUrl.split(',')[1] || dataUrl
      const res = await qwenReport(base64, result.detections || [])
      setReport(typeof res === 'string' ? res : res?.report || JSON.stringify(res, null, 2))
    } catch (err) {
      const message = err?.response?.data?.message || err?.message || '报告生成失败，请稍后重试'
      setReportError(message)
    } finally {
      setGeneratingReport(false)
    }
  }

  return (
    <section className="min-h-screen bg-tech-deep">
      {/* Header */}
      <div className="border-b border-white/5 bg-tech-card/50 backdrop-blur-sm sticky top-0 z-40">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 h-14 flex items-center justify-between">
          <button
            onClick={onBack}
            className="flex items-center gap-1.5 text-sm text-zinc-400 hover:text-tech-cyan transition-colors"
          >
            <ArrowLeft className="w-4 h-4" />
            返回首页
          </button>
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-tech-cyan" />
            <span className="text-sm font-medium text-white/80">AI 智能检测</span>
          </div>
        </div>
      </div>

      <div className="max-w-5xl mx-auto px-4 sm:px-6 py-8 space-y-6">
        {/* Page title */}
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
        >
          <h1 className="text-2xl sm:text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-tech-cyan to-purple-400">
            AI 智能检测
          </h1>
          <p className="text-sm text-zinc-500 mt-1">
            AI驱动增强 · OpenCV形态学检出 · 亮场风格转化
          </p>
        </motion.div>

        {/* Info banner */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="flex items-start gap-3 rounded-xl border border-tech-cyan/20 bg-tech-cyan/5 p-4"
        >
          <Info className="w-5 h-5 text-tech-cyan shrink-0 mt-0.5" />
          <div className="text-xs text-zinc-400 leading-relaxed">
            <span className="text-tech-cyan font-medium">注意：</span>
            当前使用 AI 增强 + OpenCV 形态学缺陷检测流水线。推荐使用暗场光照条件下拍摄的晶圆图像以获得最佳检测效果。
          </div>
        </motion.div>

        {/* Image uploader */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.15 }}
        >
          <QwenImageUploader
            onFileSelect={handleFileSelect}
            disabled={enhancing}
          />
        </motion.div>

        {/* Style toggle */}
        {file && !result && !enhancing && !error && (
          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-center justify-center gap-3"
          >
            <span className="text-xs text-zinc-500">输出风格</span>
            <button
              onClick={() => setEnhanceStyle('darkfield')}
              className={`px-4 py-2 rounded-lg text-xs font-medium transition-all ${
                enhanceStyle === 'darkfield'
                  ? 'bg-tech-cyan/20 text-tech-cyan border border-tech-cyan/40'
                  : 'bg-white/5 text-zinc-400 border border-white/10 hover:border-white/20'
              }`}
            >
              暗场增强
            </button>
            <button
              onClick={() => setEnhanceStyle('brightfield')}
              className={`px-4 py-2 rounded-lg text-xs font-medium transition-all ${
                enhanceStyle === 'brightfield'
                  ? 'bg-tech-cyan/20 text-tech-cyan border border-tech-cyan/40'
                  : 'bg-white/5 text-zinc-400 border border-white/10 hover:border-white/20'
              }`}
            >
              亮场风格转化
            </button>
          </motion.div>
        )}

        {/* Enhance button */}
        {file && !result && !enhancing && !error && (
          <motion.button
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            onClick={handleEnhance}
            className="w-full py-3 rounded-xl font-semibold text-sm bg-gradient-to-r from-tech-cyan to-purple-500 text-white hover:opacity-90 transition-all flex items-center justify-center gap-2 group"
          >
            {enhanceStyle === 'brightfield' ? '开始亮场风格转化' : '开始 AI 增强分析'}
            <ChevronRight className="w-4 h-4 group-hover:translate-x-0.5 transition-transform" />
          </motion.button>
        )}

        {/* Enhancing state */}
        {enhancing && (
          <div className="flex flex-col items-center justify-center py-8 gap-3 text-zinc-500">
            <Loader2 className="w-6 h-6 animate-spin text-tech-cyan" />
            <p className="text-sm">AI 增强分析中...</p>
            <p className="text-xs text-zinc-600">AI 增强分析 + 形态学缺陷检测</p>
          </div>
        )}

        {/* Error display */}
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex items-start gap-3 rounded-xl border border-red-500/20 bg-red-500/5 p-4"
          >
            <AlertCircle className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
            <div>
              <p className="text-sm text-red-400 font-medium mb-1">分析失败</p>
              <p className="text-xs text-zinc-500">{error}</p>
            </div>
          </motion.div>
        )}

        {/* Result panel */}
        <QwenResultPanel
          originalImage={previewUrl}
          result={result}
          loading={enhancing}
        />

        {/* Angle slider - only for brightfield */}
        {result && !enhancing && enhanceStyle === 'brightfield' && (
          <AngleSlider
            angleViews={angleViews}
            currentAngle={currentAngle}
            onAngleChange={setCurrentAngle}
            loading={loadingAngles}
          />
        )}

        {/* Report panel */}
        {result && !enhancing && (
          <QwenReportPanel
            onGenerate={handleGenerateReport}
            loading={generatingReport}
            report={report}
            error={reportError}
          />
        )}
      </div>
    </section>
  )
}
