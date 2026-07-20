import { useState, useEffect } from 'react'
import { api } from '../api'
import { Activity, CheckCircle, Clock, AlertCircle, Loader } from 'lucide-react'

const STAT_CARDS = [
  { key: 'totalTasks', label: '推理总量', icon: Activity, color: 'text-tech-cyan' },
  { key: 'doneTasks', label: '已完成', icon: CheckCircle, color: 'text-emerald-400' },
  { key: 'runningTasks', label: '运行中', icon: Loader, color: 'text-purple-400' },
  { key: 'pendingTasks', label: '等待中', icon: Clock, color: 'text-yellow-400' },
  { key: 'failedTasks', label: '失败', icon: AlertCircle, color: 'text-red-400' },
]

export default function Metrics() {
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  useEffect(() => {
    api.getMetrics()
      .then(setMetrics)
      .catch(err => setError(err.message || '加载失败'))
      .finally(() => setLoading(false))
  }, [])

  return (
    <section className="min-h-screen bg-tech-deep pt-24 pb-16 px-4">
      <div className="max-w-content mx-auto">
        <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-tech-cyan to-tech-purple mb-8">
          系统指标
        </h1>

        {loading ? (
          <div className="text-center text-zinc-500 py-20">加载中...</div>
        ) : error ? (
          <div className="text-center py-20">
            <p className="text-red-400">加载失败：{error}</p>
            <button
              onClick={() => window.location.reload()}
              className="mt-4 px-4 py-2 rounded-lg border border-white/10 text-zinc-400 hover:text-white"
            >
              重试
            </button>
          </div>
        ) : !metrics ? (
          <div className="text-center text-zinc-500 py-20">暂无数据</div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-4">
            {STAT_CARDS.map(({ key, label, icon: Icon, color }) => (
              <div
                key={key}
                className="rounded-xl border border-white/10 bg-white/5 p-6 hover:border-tech-cyan/30 transition-all"
              >
                <Icon className={color} size={32} />
                <p className="text-3xl font-bold text-white mt-2">
                  {metrics[key] ?? 0}
                </p>
                <p className="text-zinc-500 text-sm mt-1">{label}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  )
}
