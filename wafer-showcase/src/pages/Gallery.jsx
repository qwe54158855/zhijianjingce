import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { api } from '../api'

const CATEGORIES = [
  { id: 'all', label: '全部' },
  { id: 'enhance', label: '暗场→明场增强' },
  { id: 'wavelength', label: '266→193nm 转换' },
  { id: 'defect', label: '缺陷生成' },
]

export default function Gallery() {
  const [items, setItems] = useState([])
  const [category, setCategory] = useState('all')
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    api.getGallery({ category: category === 'all' ? undefined : category })
      .then(res => setItems(res.data.content || []))
      .catch(err => setError(err.message || '加载失败'))
      .finally(() => setLoading(false))
  }, [category])

  return (
    <section className="min-h-screen bg-tech-deep pt-24 pb-16 px-4">
      <div className="max-w-content mx-auto">
        <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-tech-cyan to-tech-purple mb-8">
          技术展厅
        </h1>

        <div className="flex gap-2 mb-8 flex-wrap">
          {CATEGORIES.map((cat) => (
            <button
              key={cat.id}
              onClick={() => setCategory(cat.id)}
              className={`px-4 py-2 rounded-full text-sm transition-all ${
                category === cat.id
                  ? 'bg-tech-cyan/20 text-tech-cyan border border-tech-cyan/30'
                  : 'bg-white/5 text-zinc-400 border border-white/10 hover:border-white/20'
              }`}
            >
              {cat.label}
            </button>
          ))}
        </div>

        {loading ? (
          <div className="text-center text-zinc-500 py-20">加载中...</div>
        ) : error ? (
          <div className="text-center py-20">
            <p className="text-red-400 mb-2">加载失败</p>
            <p className="text-zinc-500 text-sm">{error}</p>
          </div>
        ) : items.length === 0 ? (
          <div className="text-center text-zinc-500 py-20">暂无素材</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {items.map((item, i) => (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: i * 0.1 }}
                className="rounded-xl border border-white/10 bg-white/5 overflow-hidden group hover:border-tech-cyan/30 transition-all"
              >
                <div className="aspect-video bg-zinc-900 relative overflow-hidden">
                  {item.thumbnailUrl ? (
                    <img
                      src={item.thumbnailUrl}
                      alt={item.title}
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="flex items-center justify-center h-full text-zinc-700">
                      预览图
                    </div>
                  )}
                </div>
                <div className="p-4">
                  <h3 className="text-white font-medium mb-1">{item.title}</h3>
                  <p className="text-zinc-500 text-sm line-clamp-2">{item.description}</p>
                  {item.metrics && (
                    <div className="mt-2 flex gap-2 text-xs text-zinc-600">
                      <span className="px-2 py-0.5 rounded bg-white/5">{item.metrics}</span>
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </section>
  )
}
