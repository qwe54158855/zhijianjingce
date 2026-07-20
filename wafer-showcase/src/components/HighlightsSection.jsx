import { useRef } from 'react'
import {
  Cpu,
  GitBranch,
  Crosshair,
  RotateCcw,
  Layers,
  RefreshCcw,
  Sliders,
} from 'lucide-react'

import HighlightCard from './HighlightCard'

const HIGHLIGHTS = [
  {
    icon: Cpu,
    title: 'RepViT 轻量骨干网络',
    description:
      '基于 RepViT-M0.9 的重参数化轻量骨干，5.1M 参数在 ARM 处理器上实现实时推理，同时支持多尺度特征提取，兼顾速度与精度。',
    metrics: ['5.1M 参数', 'ARM 实时推理', '重参数化融合'],
  },
  {
    icon: GitBranch,
    title: '共享编码器 + 双分支架构',
    description:
      '单一编码器同时驱动分割与检测双任务分支，参数共享率超过 60%，实现表征协同与计算效率的完美平衡，显著降低部署体积。',
    metrics: ['参数共享 >60%', '多任务协同'],
  },
  {
    icon: Crosshair,
    title: 'F2 小目标检测层',
    description:
      '新增 stride=8 的 F2 浅层检测头，与原 F3/F4 形成三层多尺度预测架构，确保 <10px 微缺陷不被特征下采样淹没，小目标召回率大幅提升。',
    metrics: ['<10px 缺陷检测', '三层多尺度'],
  },
  {
    icon: RotateCcw,
    title: '极坐标边缘展开',
    description:
      '将晶圆边缘环形区域通过极坐标变换展开为矩形图像，消除旋转不变性需求，降低计算量 60%，边缘缺陷检出率显著提升。',
    metrics: ['计算量减 60%', '边缘缺陷提升'],
  },
  {
    icon: Layers,
    title: '多波长通道融合',
    description:
      '自适应融合 266nm 深紫外与 532nm 可见光双波长成像，通过可学习权重动态调整各通道贡献，捕获不同深度类型的晶圆缺陷。',
    metrics: ['266nm + 532nm', '自适应加权'],
  },
  {
    icon: RefreshCcw,
    title: 'CycleGAN 无监督预训练',
    description:
      '利用循环一致性生成对抗网络在无配对数据上预训练，充分利用海量无标注晶圆图像预训练编码器，大幅降低人工标注成本。',
    metrics: ['无需配对标注', '循环一致性'],
  },
  {
    icon: Sliders,
    title: '在线自适应校准',
    description:
      '基于 Contextual Bandit 强化学习策略，实时感知环境上下文，动态调整预处理归一化参数和检测置信度阈值，自动补偿光源漂移与传感器衰减，实现零人工介入的持续稳定运行。',
    metrics: ['< 0.5ms 推理增量', '零人工介入', '光源漂移补偿'],
    fullWidth: true,
  },
]

export default function HighlightsSection() {
  const ref = useRef(null)

  return (
    <section
      id="highlights"
      ref={ref}
      className="relative py-24"
    >
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-tech-deep via-tech-deep to-tech-deep/95" />

      <div className="relative z-10 mx-auto max-w-content px-8">
        {/* Section header */}
        <div className="mb-16 text-center">
          <h2 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-tech-cyan to-tech-purple sm:text-4xl">
            核心技术亮点
          </h2>
          <p className="mx-auto mt-4 max-w-2xl text-zinc-400">
            七项关键技术突破，构建从晶圆成像到缺陷识别的完整技术闭环
          </p>
        </div>

        {/* Bento grid */}
        <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
          {HIGHLIGHTS.map((item, index) => (
            <div
              key={item.title}
              className={
                item.fullWidth
                  ? 'md:col-span-3'
                  : undefined
              }
            >
              <HighlightCard
                icon={item.icon}
                title={item.title}
                description={item.description}
                metrics={item.metrics}
                index={index}
                horizontal={item.fullWidth}
              />
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
