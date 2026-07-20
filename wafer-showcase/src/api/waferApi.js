import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8001/api/v1';

function toCamelCase(obj) {
  if (obj === null || typeof obj !== 'object') return obj
  if (Array.isArray(obj)) return obj.map(toCamelCase)
  return Object.fromEntries(
    Object.entries(obj).map(([k, v]) => [
      k.replace(/_([a-z])/g, (_, c) => c.toUpperCase()),
      toCamelCase(v),
    ])
  )
}

export async function qwenEnhance(imageBase64, style = 'brightfield') {
  const { data } = await axios.post(`${API_BASE}/qwen/enhance`, {
    image: imageBase64,
    format: 'jpg',
    style,
  });
  return toCamelCase(data);
}

export async function qwenReport(imageBase64, detections) {
  const { data } = await axios.post(`${API_BASE}/qwen/report`, {
    image: imageBase64,
    detections,
  });
  return toCamelCase(data);
}

export async function qwenAngles(imageBase64) {
  const { data } = await axios.post(`${API_BASE}/qwen/angles`, {
    image: imageBase64,
    format: 'jpg',
  }, { timeout: 60000 });
  return toCamelCase(data);
}
