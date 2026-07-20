# Wafer Showcase — React Frontend Dashboard

## Overview
Modern React-based technical showcase website for the wafer inspection AI solution. Features workbench page, gallery browser, metrics dashboard, and Qwen-VL analysis view.

## Tech Stack
- React 18 + Vite 5
- Tailwind CSS 3 + Framer Motion 11
- Lucide React Icons
- UnicornStudio 3D (Hero background)

## Quick Start
```bash
npm install
VITE_API_BASE=http://localhost:8080 npm run dev
# Opens at http://localhost:5173
```

## Pages
| Page | Route | Description |
|------|-------|-------------|
| Home | `/` | Tech showcase with 3D background |
| Workbench | `/workbench` | Inference & analysis interface |
| Gallery | `/gallery` | Image gallery browser |
| Metrics | `/metrics` | Performance dashboard |
| Qwen Detect | `/qwen-detect` | Qwen-VL analysis page |

## Build
```bash
npm run build   # Output: dist/
npm run preview # Preview production build
```

## Design System
- Color: `tech-deep (#06060b)`, `tech-cyan (#00e5ff)`, `tech-purple (#7c3aed)`
- Font: Inter (body), JetBrains Mono (code)
- Animations: staggered fadeIn + slideUp, counting number effects

## Directory
```
src/
├── components/   # React components
├── hooks/        # Custom hooks
├── utils/        # Utilities
└── pages/        # Route pages
```
