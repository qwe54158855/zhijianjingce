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

// ===== Dataset API =====

export async function getDatasetClasses() {
  const { data } = await axios.get(`${API_BASE}/qwen/dataset/classes`);
  return toCamelCase(data);
}

export async function getDatasetList(page = 1, pageSize = 50, classFilter = 0) {
  const { data } = await axios.get(`${API_BASE}/qwen/dataset/list`, {
    params: { page, page_size: pageSize, class_filter: classFilter },
  });
  return toCamelCase(data);
}

export async function classifyDatasetImage(imageName, topK = 3) {
  const { data } = await axios.post(`${API_BASE}/qwen/dataset/classify`, {
    image_name: imageName,
    top_k: topK,
  }, { timeout: 60000 });
  return toCamelCase(data);
}

export async function batchEvalDataset(sampleSize = 10) {
  const { data } = await axios.post(`${API_BASE}/qwen/dataset/batch-eval`, null, {
    params: { sample_size: sampleSize },
    timeout: 300000,
  });
  return toCamelCase(data);
}
