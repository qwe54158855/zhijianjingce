import { useState, useCallback } from 'react'

export function useGallery() {
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const fetchGallery = useCallback(async (apiCall) => {
    setLoading(true)
    setError(null)
    try {
      const res = await apiCall()
      setItems(res.data.content || [])
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }, [])

  return { items, loading, error, fetchGallery, setItems }
}
