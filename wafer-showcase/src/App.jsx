import { useState } from 'react'
import HeroSection from './components/HeroSection'
import ChallengesSection from './components/ChallengesSection'
import ArchitectureSection from './components/ArchitectureSection'
import HighlightsSection from './components/HighlightsSection'
import MetricsSection from './components/MetricsSection'
import FooterSection from './components/FooterSection'
import QwenDetectPage from './pages/QwenDetectPage'

export default function App() {
  const [page, setPage] = useState('home')

  if (page === 'qwen') {
    return (
      <div className="relative bg-tech-deep text-white min-h-screen">
        <QwenDetectPage onBack={() => setPage('home')} />
      </div>
    )
  }

  return (
    <div className="relative bg-tech-deep text-white">
      <HeroSection onNavigate={setPage} />
      <ChallengesSection />
      <ArchitectureSection />
      <HighlightsSection />
      <MetricsSection />
      <FooterSection onNavigate={setPage} />
    </div>
  )
}
