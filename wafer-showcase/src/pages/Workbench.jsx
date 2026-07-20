import { useState } from 'react'
import { motion } from 'framer-motion'
import ImageUploader from '../components/ImageUploader'
import ModelSelector from '../components/ModelSelector'
import InferenceResult from '../components/InferenceResult'
import ProgressTracker from '../components/ProgressTracker'
import { useInference } from '../hooks/useInference'

export default function Workbench() {
  const [selectedModel, setSelectedModel] = useState(null)
  const [file, setFile] = useState(null)
  const { status, result, progress, error, submit, reset } = useInference()

  const handleInfer = () => {
    if (!file || !selectedModel) return
    submit(selectedModel, file)
  }

  return (
    <section className="min-h-screen bg-tech-deep pt-24 pb-16 px-4">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="max-w-content mx-auto space-y-8"
      >
        <div>
          <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-tech-cyan to-tech-purple mb-2">
            AI 推理工作台
          </h1>
          <p className="text-zinc-500">
            上传晶圆图像，选择 LoRA 模式，体验 AI 驱动的晶圆图像增强
          </p>
        </div>

        <ProgressTracker currentStep={status} />

        <ModelSelector
          selected={selectedModel}
          onSelect={setSelectedModel}
          disabled={status === 'running'}
        />

        <ImageUploader
          onFileSelect={setFile}
          disabled={status === 'running'}
        />

        {file && selectedModel && status === 'idle' && (
          <motion.button
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            onClick={handleInfer}
            className="w-full py-3 rounded-xl font-semibold bg-gradient-to-r from-tech-cyan to-tech-purple text-white hover:opacity-90 transition-all"
          >
            开始推理
          </motion.button>
        )}

        {status !== 'idle' && (
          <motion.button
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            onClick={reset}
            className="w-full py-2 rounded-xl border border-white/10 text-zinc-400 hover:text-white transition-all"
          >
            重新开始
          </motion.button>
        )}

        <InferenceResult
          originalFile={file}
          result={result}
          progress={progress}
          status={status}
          error={error}
        />
      </motion.div>
    </section>
  )
}
