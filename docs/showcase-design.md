# 晶圆检测技术展厅 — 设计文档

> **框架**：React 18 + Vite 5 + Tailwind CSS 3 + Framer Motion 11 | **路径**：`wafer-showcase/`

---

## 一、项目定位

半导体晶圆缺陷检测一体化 AI 解决方案的**技术展示网站**。以技术展厅形态呈现项目核心创新点。

## 二、视觉设计规范

### 色板
| 用途 | 色值 | Tailwind 变量 |
|------|------|--------------|
| 页面底色 | `#06060b` | `tech-deep` |
| 卡片底色 | `#0d0d14` | `tech-card` |
| 边框 | `rgba(255,255,255,0.06)` | `tech-border` |
| 强调色(青) | `#00e5ff` | `tech-cyan` |
| 辅助色(紫) | `#7c3aed` | `tech-purple` |

### 字体：Inter（正文）+ JetBrains Mono（代码）
### 布局：版心 1700px，纯暗色背景，hover 上浮 4px + 发光

## 三、页面结构

| 模块 | 组件 | 说明 |
|------|------|------|
| Hero | `HeroSection` | 全屏 3D 背景 + 4 导航卡片 |
| 三大难题 | `ChallengesSection` + `ChallengeCard` | 信噪比/算力/数据 |
| 技术架构 | `ArchitectureSection` + `ArchitectureDiagram` | 6 层交错动画架构图 |
| 亮点 | `HighlightsSection` + `HighlightCard` | Bento 网格 7 项创新 |
| 指标 | `MetricsSection` + `MetricItem` | 6 个数字动效 P0 指标 |
| Footer | `FooterSection` | 全屏收尾 + CTA |

## 四、组件文件结构

```
src/
├── App.jsx               # 根组件
├── components/
│   ├── Navbar.jsx        # 固定导航（透明→毛玻璃）
│   ├── HeroSection.jsx   # 主屏
│   ├── ChallengesSection.jsx / ChallengeCard.jsx
│   ├── ArchitectureSection.jsx / ArchitectureDiagram.jsx
│   ├── HighlightsSection.jsx / HighlightCard.jsx
│   ├── MetricsSection.jsx / MetricItem.jsx
│   └── FooterSection.jsx
└── utils/cn.js           # clsx + tailwind-merge
```

## 五、动效规范
| 元素 | 动效 |
|------|------|
| 标题入场 | fadeIn + slideUp，stagger 0.2s |
| 卡片 hover | 上浮 4px + 边框发光 |
| 指标数字 | 0→N 计数动效 |
| 导航栏 | 滚动 50px 后 backdrop-blur |

## 六、品牌信息
- **学校**：东北大学秦皇岛分校 | **年份**：2026
- **项目名**：晶圆缺陷检测·一体化 AI 解决方案
- **Logo**：`WAFER_AI`（青色圆点 + 文字）
