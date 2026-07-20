import { useState, useCallback, useRef } from 'react'
import { api } from '../api'

export function useInference() {
  const [taskId, setTaskId] = useState(null)
  const [status, setStatus] = useState('idle') // idle | uploading | running | done | error
  const [result, setResult] = useState(null)
  const [progress, setProgress] = useState(0)
  const [error, setError] = useState(null)
  const eventSourceRef = useRef(null)
  const pollRef = useRef(null)

  const submit = useCallback(async (type, file, params) => {
    setStatus('uploading')
    setError(null)
    setResult(null)
    setProgress(0)

    // Clear any previous polling interval
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }

    try {
      const task = await api.submitInference(type, file, params)
      setTaskId(task.taskId)
      setStatus('running')

      // Connect SSE for progress
      const es = api.getTaskStream(task.taskId)
      eventSourceRef.current = es

      es.addEventListener('progress', (e) => {
        const data = JSON.parse(e.data)
        setProgress(data.progress)
      })

      es.addEventListener('complete', (e) => {
        const data = JSON.parse(e.data)
        setResult(data)
        setStatus('done')
        setProgress(100)
        es.close()
        clearInterval(pollRef.current)
        pollRef.current = null
      })

      es.addEventListener('error', (e) => {
        const data = JSON.parse(e.data)
        setError(data.error || '推理失败')
        setStatus('error')
        es.close()
        clearInterval(pollRef.current)
        pollRef.current = null
      })

      // Fallback: poll if SSE fails
      pollRef.current = setInterval(async () => {
        const taskStatus = await api.getTaskStatus(task.taskId)
        if (taskStatus.status === 'DONE') {
          setResult(taskStatus)
          setStatus('done')
          setProgress(100)
          clearInterval(pollRef.current)
          pollRef.current = null
          es.close()
        } else if (taskStatus.status === 'FAILED') {
          setError(taskStatus.errorMessage || '推理失败')
          setStatus('error')
          clearInterval(pollRef.current)
          pollRef.current = null
          es.close()
        }
      }, 2000)

    } catch (err) {
      setError(err.message || '提交失败')
      setStatus('error')
    }
  }, [])

  const reset = useCallback(() => {
    setTaskId(null)
    setStatus('idle')
    setResult(null)
    setProgress(0)
    setError(null)
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }, [])

  return { taskId, status, result, progress, error, submit, reset }
}
