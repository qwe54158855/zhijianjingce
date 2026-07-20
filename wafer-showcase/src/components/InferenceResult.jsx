import { useMemo, useEffect } from 'react'

export default function InferenceResult({ originalFile, result, progress, status, error }) {
  const blobUrl = useMemo(() => {
    if (originalFile) return URL.createObjectURL(originalFile)
    return ''
  }, [originalFile])

  useEffect(() => {
    return () => {
      if (blobUrl) URL.revokeObjectURL(blobUrl)
    }
  }, [blobUrl])

  if (status === 'idle') return null

  return (
    <div className="space-y-4 mt-8">
      <h3 className="text-lg font-semibold text-white">推理结果</h3>

      {status === 'uploading' && (
        <div className="p-8 text-center text-zinc-400">
          <p>上传中...</p>
        </div>
      )}

      {status === 'running' && (
        <div className="p-8 text-center">
          <div className="w-full bg-white/5 rounded-full h-2 mb-4">
            <div
              className="h-full bg-gradient-to-r from-tech-cyan to-tech-purple rounded-full transition-all duration-500"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p className="text-zinc-400">推理中... {progress}%</p>
        </div>
      )}

      {status === 'done' && result && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="space-y-2">
            <p className="text-sm text-zinc-500">原图</p>
            <img
              src={blobUrl}
              alt="原图"
              className="rounded-lg border border-white/10 w-full"
            />
          </div>
          <div className="space-y-2">
            <p className="text-sm text-tech-cyan">LoRA 结果</p>
            <div className="rounded-lg border border-tech-cyan/30 bg-white/5 p-4 h-full flex items-center justify-center">
              <p className="text-zinc-500 text-sm">结果图片将在此显示</p>
            </div>
          </div>
          <div className="space-y-2">
            <p className="text-sm text-zinc-500">差异对比</p>
            <div className="rounded-lg border border-white/10 bg-white/5 p-4 h-full flex items-center justify-center">
              <p className="text-zinc-500 text-sm">差异热力图将在此显示</p>
            </div>
          </div>
        </div>
      )}

      {status === 'error' && (
        <div className="p-6 rounded-lg bg-red-500/10 border border-red-500/30 text-red-400">
          <p>推理失败：{error || result?.errorMessage || '未知错误'}</p>
        </div>
      )}
    </div>
  )
}
