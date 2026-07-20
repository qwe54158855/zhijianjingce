import { useState, useRef } from 'react'
import { Upload, X } from 'lucide-react'
import { cn } from '../utils/cn'

export default function ImageUploader({ onFileSelect, disabled }) {
  const [preview, setPreview] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef(null)

  const handleFile = (file) => {
    if (!file || !file.type.startsWith('image/')) return
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(file)
    onFileSelect(file)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    handleFile(e.dataTransfer.files[0])
  }

  const clearImage = () => {
    setPreview(null)
    onFileSelect(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  return (
    <div
      className={cn(
        'relative border-2 border-dashed rounded-xl p-8 text-center transition-all cursor-pointer',
        'border-white/10 hover:border-tech-cyan/50',
        dragOver && 'border-tech-cyan bg-tech-cyan/5',
        disabled && 'opacity-50 pointer-events-none',
      )}
      onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
    >
      {preview ? (
        <div className="relative inline-block">
          <img
            src={preview}
            alt="预览"
            className="max-h-64 rounded-lg object-contain"
          />
          <button
            onClick={(e) => { e.stopPropagation(); clearImage() }}
            className="absolute -top-2 -right-2 p-1 bg-red-500 rounded-full"
          >
            <X size={14} />
          </button>
        </div>
      ) : (
        <div className="space-y-3">
          <Upload className="mx-auto text-zinc-400" size={40} />
          <p className="text-zinc-400">
            拖拽晶圆图片到此处，或<strong className="text-tech-cyan">点击选择</strong>
          </p>
          <p className="text-zinc-600 text-sm">支持 JPG / PNG / BMP，最大 50MB</p>
        </div>
      )}
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => handleFile(e.target.files[0])}
      />
    </div>
  )
}
