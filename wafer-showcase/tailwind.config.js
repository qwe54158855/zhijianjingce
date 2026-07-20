/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      colors: {
        tech: {
          cyan: '#00e5ff',
          purple: '#7c3aed',
          deep: '#06060b',
          card: '#0d0d14',
          border: 'rgba(255,255,255,0.06)',
        },
      },
      maxWidth: {
        content: '1700px',
      },
    },
  },
  plugins: [],
}
