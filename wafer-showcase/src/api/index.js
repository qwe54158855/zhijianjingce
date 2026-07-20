import axios from 'axios'

const BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8080/api/v1'

export const api = {
  // Gallery
  getGallery: (params) => axios.get(`${BASE}/gallery`, { params }),
  getGalleryStats: () => axios.get(`${BASE}/gallery/stats`),

  // Inference
  submitInference: async (type, file, params = {}) => {
    const fd = new FormData()
    fd.append('file', file)
    fd.append('type', type)
    if (Object.keys(params).length) {
      fd.append('params', JSON.stringify(params))
    }
    const res = await axios.post(`${BASE}/inference`, fd)
    return res.data
  },

  getTaskStatus: async (id) => {
    const res = await axios.get(`${BASE}/inference/${id}`)
    return res.data
  },

  getTaskStream: (id) => {
    return new EventSource(`${BASE}/inference/${id}/stream`)
  },

  // Metrics
  getMetrics: async () => {
    const res = await axios.get(`${BASE}/metrics/overview`)
    return res.data
  },
}
