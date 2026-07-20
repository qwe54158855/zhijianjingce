import { motion } from 'framer-motion'
import { Signal, Cpu, Database } from 'lucide-react'
import ChallengeCard from './ChallengeCard'

const CHALLENGES = [
  {
    icon: Signal,
    title: '信噪比不足',
    description:
      '晶圆缺陷信号微弱，背景噪声干扰严重。传统算法在低信噪比条件下误检率攀升，极端工艺节点下信噪比可低至 0.5 dB，近乎淹没目标特征。',
  },
  {
    icon: Cpu,
    title: '算力受限',
    description:
      '产线检测需在毫秒级完成单晶圆处理，而高分辨率图像数据量达 GB 级别。边缘端算力与检测精度之间存在难以调和的矛盾。',
  },
  {
    icon: Database,
    title: '数据稀缺',
    description:
      '实际产线缺陷样本极少，且缺陷类型分布极度不均衡。正负样本比可达 1:10000，传统监督学习方法在此场景下难以有效训练。',
  },
]

export default function ChallengesSection() {
  return (
    <section id="challenges" className="relative py-32 px-8">
      {/* Background gradient */}
      <div className="absolute inset-0 bg-gradient-to-b from-tech-deep via-tech-deep to-tech-card/50 pointer-events-none" />

      <div className="relative max-w-content mx-auto">
        {/* Section header */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: '-50px' }}
          transition={{ duration: 0.6 }}
          className="text-center mb-20"
        >
          <span className="inline-block text-xs font-mono tracking-[0.2em] text-tech-cyan/60 mb-4">
            CHALLENGES
          </span>
          <h2 className="text-4xl md:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-tech-cyan to-tech-purple mb-4">
            三大核心难题
          </h2>
          <p className="text-zinc-400 max-w-2xl mx-auto text-base">
            晶圆缺陷检测领域长期未能攻克的三大技术壁垒
          </p>
        </motion.div>

        {/* Challenge cards grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {CHALLENGES.map((challenge, index) => (
            <ChallengeCard
              key={challenge.title}
              icon={challenge.icon}
              title={challenge.title}
              description={challenge.description}
              index={index}
            />
          ))}
        </div>
      </div>
    </section>
  )
}
