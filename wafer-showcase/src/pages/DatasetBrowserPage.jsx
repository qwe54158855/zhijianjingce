import { useState, useEffect, useCallback } from 'react'
import { motion } from 'framer-motion'
import { ArrowLeft, Database, Filter, Search, Layers, Beaker, AlertCircle, Loader2, ChevronLeft, ChevronRight } from 'lucide-react'
import { getDatasetClasses, getDatasetList, classifyDatasetImage, batchEvalDataset } from '../api/waferApi'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8001/api/v1'

export default function DatasetBrowserPage({ onBack }) {
  // Stats
  const [stats, setStats] = useState(null)
  const [classes, setClasses] = useState([])
  const [loading, setLoading] = useState(true)

  // List
  const [images, setImages] = useState([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [pageSize] = useState(48)
  const [classFilter, setClassFilter] = useState(0)

  // Classification
  const [selectedImage, setSelectedImage] = useState(null)
  const [classifying, setClassifying] = useState(false)
  const [classifyResult, setClassifyResult] = useState(null)

  // Batch eval
  const [evalRunning, setEvalRunning] = useState(false)
  const [evalResult, setEvalResult] = useState(null)

  // Load stats & classes on mount
  useEffect(() => {
    (async () => {
      try {
        const d = await getDatasetClasses()
        setStats(d)
        setClasses(d.classes || [])
      } catch (e) {
        console.error('Failed to load dataset stats', e)
      }
    })()
  }, [])

  // Load image list
  const loadList = useCallback(async () => {
    setLoading(true)
    try {
      const d = await getDatasetList(page, pageSize, classFilter)
      setImages(d.images || [])
      setTotal(d.total || 0)
    } catch (e) {
      console.error('Failed to load dataset list', e)
    }
    setLoading(false)
  }, [page, pageSize, classFilter])

  useEffect(() => { loadList() }, [loadList])

  const totalPages = Math.ceil(total / pageSize)

  // Classify image
  const handleClassify = async (imageName) => {
    setSelectedImage(imageName)
    setClassifying(true)
    setClassifyResult(null)
    try {
      const result = await classifyDatasetImage(imageName, 3)
      setClassifyResult(result)
    } catch (e) {
      console.error('Classification failed', e)
    }
    setClassifying(false)
  }

  // Batch eval
  const handleBatchEval = async () => {
    if (evalRunning) return
    setEvalRunning(true)
    setEvalResult(null)
    try {
      const result = await batchEvalDataset(20)
      setEvalResult(result)
    } catch (e) {
      console.error('Batch eval failed', e)
    }
    setEvalRunning(false)
  }

  return (
    <div className="min-h-screen bg-[#06060b] text-white">
      {/* Navbar */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-[#06060b]/80 backdrop-blur-xl border-b border-white/5">
        <div className="max-w-[1700px] mx-auto px-6 h-16 flex items-center justify-between">
          <button onClick={onBack} className="flex items-center gap-2 text-zinc-400 hover:text-tech-cyan transition-colors">
            <ArrowLeft size={20} /> 返回首页
          </button>
          <span className="text-sm font-mono text-zinc-500">Dataset Browser v1.0</span>
        </div>
      </nav>

      <div className="pt-20 pb-12 px-6 max-w-[1700px] mx-auto">
        {/* Header */}
        <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="mb-8">
          <h1 className="text-3xl font-bold bg-gradient-to-r from-tech-cyan to-tech-purple text-transparent bg-clip-text">
            <Database className="inline mr-3 mb-1" size={28} />
            缺陷数据集
          </h1>
          <p className="text-zinc-400 mt-2">浏览 1150 张晶圆缺陷图像，使用 Qwen-VL AI 进行分析</p>
        </motion.div>

        {/* Stats Bar */}
        {stats && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            {[
              { label: '总图像数', value: stats.totalImages, icon: <Database size={18} />, color: 'text-tech-cyan' },
              { label: '缺陷类别', value: stats.totalClasses, icon: <Layers size={18} />, color: 'text-purple-400' },
              { label: '已标注', value: stats.totalImages, icon: <Beaker size={18} />, color: 'text-emerald-400' },
              { label: '每页显示', value: `${pageSize}张`, icon: <Filter size={18} />, color: 'text-rose-400' },
            ].map((s, i) => (
              <div key={i} className="border border-white/5 bg-white/5 rounded-xl p-4 backdrop-blur-sm">
                <div className={`${s.color} mb-1`}>{s.icon}</div>
                <div className="text-2xl font-bold font-mono">{s.value}</div>
                <div className="text-xs text-zinc-500 mt-1">{s.label}</div>
              </div>
            ))}
          </motion.div>
        )}

        {/* Batch Eval Button */}
        <div className="flex gap-4 mb-6">
          <button
            onClick={handleBatchEval}
            disabled={evalRunning}
            className="px-4 py-2 bg-tech-cyan/10 border border-tech-cyan/30 rounded-lg text-tech-cyan hover:bg-tech-cyan/20 transition-all flex items-center gap-2 disabled:opacity-50"
          >
            {evalRunning ? <Loader2 className="animate-spin" size={16} /> : <Beaker size={16} />}
            {evalRunning ? '评估中...' : '随机抽检 20 张'}
          </button>

          {/* Class filter */}
          <div className="flex items-center gap-2">
            <Filter size={16} className="text-zinc-500" />
            <select
              value={classFilter}
              onChange={e => { setClassFilter(Number(e.target.value)); setPage(1) }}
              className="bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-tech-cyan/50"
            >
              <option value={0}>全部类别</option>
              {classes.map(c => (
                <option key={c.defectId} value={c.defectId}>
                  类别 {c.defectId} ({c.count} 张)
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Eval Results */}
        {evalResult && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} className="mb-6 border border-emerald-500/20 bg-emerald-500/5 rounded-xl p-4">
            <div className="flex items-center gap-2 text-emerald-400 text-sm font-mono">
              <Beaker size={14} /> 抽检结果
            </div>
            <div className="mt-2 flex gap-6">
              <div><span className="text-zinc-400">样本数:</span> <span className="font-mono text-white">{evalResult.total}</span></div>
              <div><span className="text-zinc-400">正确:</span> <span className="font-mono text-emerald-400">{evalResult.correct}</span></div>
              <div><span className="text-zinc-400">准确率:</span> <span className="font-mono text-tech-cyan">{(evalResult.accuracy * 100).toFixed(1)}%</span></div>
              <div><span className="text-zinc-400">耗时:</span> <span className="font-mono text-zinc-300">{evalResult.inferenceTimeMs}ms</span></div>
            </div>
          </motion.div>
        )}

        {/* Image Grid */}
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="animate-spin text-tech-cyan" size={32} />
          </div>
        ) : images.length === 0 ? (
          <div className="text-center py-20 text-zinc-500">
            <AlertCircle className="mx-auto mb-4" size={40} />
            <p>该类别暂无图像</p>
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
            {images.map((img, i) => (
              <motion.div
                key={img.imageName}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.02 }}
                onClick={() => handleClassify(img.imageName)}
                className={`group relative border border-white/5 bg-white/[0.02] rounded-xl overflow-hidden cursor-pointer hover:border-tech-cyan/30 hover:shadow-[0_0_20px_rgba(0,229,255,0.08)] transition-all ${
                  selectedImage === img.imageName ? 'ring-2 ring-tech-cyan' : ''
                }`}
              >
                {/* Thumbnail */}
                <div className="aspect-[3/2] bg-zinc-900 flex items-center justify-center overflow-hidden">
                  <img
                    src={`${API_BASE}/qwen/dataset/image/${img.imageName}`}
                    alt={img.imageName}
                    className="w-full h-full object-cover opacity-80 group-hover:opacity-100 transition-opacity"
                    loading="lazy"
                  />
                </div>
                {/* Info */}
                <div className="p-2">
                  <div className="text-xs font-mono text-zinc-500 truncate">{img.imageName}</div>
                  <div className="text-xs font-mono mt-1">
                    <span className="text-tech-cyan">Class {img.defectId}</span>
                    <span className="text-zinc-600 ml-2">{img.fileSizeKb}KB</span>
                  </div>
                </div>
                {/* Analyze overlay */}
                {selectedImage === img.imageName && classifying && (
                  <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
                    <Loader2 className="animate-spin text-tech-cyan" size={24} />
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-center gap-3 mt-8">
            <button
              disabled={page <= 1}
              onClick={() => setPage(p => Math.max(1, p - 1))}
              className="p-2 border border-white/10 rounded-lg disabled:opacity-30 hover:border-tech-cyan/30 transition-all"
            >
              <ChevronLeft size={18} />
            </button>
            <span className="text-sm font-mono text-zinc-400">
              {page} / {totalPages}
            </span>
            <button
              disabled={page >= totalPages}
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              className="p-2 border border-white/10 rounded-lg disabled:opacity-30 hover:border-tech-cyan/30 transition-all"
            >
              <ChevronRight size={18} />
            </button>
          </div>
        )}

        {/* Classification Result Panel */}
        {classifyResult && !classifying && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-8 border border-tech-cyan/20 bg-tech-cyan/5 rounded-xl p-6"
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold flex items-center gap-2">
                <Search size={18} className="text-tech-cyan" />
                Qwen-VL 分析结果
              </h3>
              <button onClick={() => setClassifyResult(null)} className="text-zinc-500 hover:text-white transition-colors text-sm">
                关闭
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Image preview */}
              <div className="border border-white/5 rounded-lg overflow-hidden bg-black/30">
                <img
                  src={`${API_BASE}/qwen/dataset/image/${classifyResult.imageName}`}
                  alt={classifyResult.imageName}
                  className="w-full object-contain max-h-64"
                />
                <div className="p-2 text-xs font-mono text-zinc-500 text-center">
                  {classifyResult.imageName}
                </div>
              </div>

              {/* Predictions */}
              <div>
                <div className="mb-3">
                  <span className="text-zinc-400 text-sm">真实标签: </span>
                  <span className="font-mono text-emerald-400">Class {classifyResult.groundTruth}</span>
                </div>

                <div className="space-y-2">
                  {classifyResult.predictions && classifyResult.predictions.map((p, i) => (
                    <div
                      key={i}
                      className={`border rounded-lg p-3 ${
                        p.defectId === classifyResult.groundTruth
                          ? 'border-emerald-500/30 bg-emerald-500/10'
                          : 'border-white/5 bg-white/[0.02]'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-mono text-sm">
                          {i === 0 ? '🥇' : i === 1 ? '🥈' : '🥉'} Class {p.defectId ?? '?'}
                        </span>
                        <span className={`font-mono text-sm ${p.confidence > 0.5 ? 'text-tech-cyan' : 'text-zinc-500'}`}>
                          {(p.confidence * 100).toFixed(1)}%
                        </span>
                      </div>
                      {p.reason && (
                        <div className="text-xs text-zinc-500 mt-1">{p.reason}</div>
                      )}
                    </div>
                  ))}
                </div>

                {classifyResult.analysisText && (
                  <div className="mt-3 p-3 border border-white/5 rounded-lg bg-white/[0.02]">
                    <div className="text-xs text-zinc-500 mb-1">AI 分析:</div>
                    <div className="text-sm text-zinc-300">{classifyResult.analysisText}</div>
                  </div>
                )}

                <div className="mt-3 text-xs text-zinc-600 font-mono">
                  推理耗时: {classifyResult.inferenceTimeMs}ms
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </div>
    </div>
  )
}
