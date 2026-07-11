import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const BASE_URL = __ENV.API_BASE || 'http://localhost:8080/api/v1';

const inferenceDuration = new Trend('inference_duration_ms');
const errorRate = new Rate('errors');

export const options = {
  stages: [
    { duration: '30s', target: 5 },    // Ramp up to 5 VUs
    { duration: '1m', target: 10 },     // Stay at 10 VUs
    { duration: '30s', target: 0 },     // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<2000'],  // 95% of requests under 2s
    errors: ['rate<0.1'],               // Error rate < 10%
  },
};

export default function () {
  // Health check
  const healthRes = http.get(`${BASE_URL}/health`);
  check(healthRes, { 'health is UP': (r) => r.json('status') === 'UP' });

  // Gallery list
  const galleryRes = http.get(`${BASE_URL}/gallery?page=0&size=10`);
  check(galleryRes, { 'gallery returns 200': (r) => r.status === 200 });

  // Metrics
  const metricsRes = http.get(`${BASE_URL}/metrics/overview`);
  check(metricsRes, { 'metrics returns 200': (r) => r.status === 200 });

  sleep(1);
}
