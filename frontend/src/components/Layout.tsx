import { useState, useCallback, useEffect, useMemo } from 'react'
import { Header } from './Header'
import { Sidebar, type ViewType } from './Sidebar'
import type { ServiceStatus } from '../types/layout'
import { getHealth, getReady } from '../services/api'

interface LayoutProps {
  children: React.ReactNode
  currentView?: ViewType
  onNavigate?: (view: ViewType) => void
}

// Fallback version if API is unavailable
const FALLBACK_VERSION = '1.1.0'

// Get initial language from localStorage (runs once during module load)
const getInitialLanguage = (): string => {
  if (typeof window !== 'undefined') {
    return localStorage.getItem('preferredLanguage') || 'en-US'
  }
  return 'en-US'
}

// Get initial endpoint info from environment
const getInitialEndpointInfo = (): { endpoint?: string; region?: string } => {
  if (typeof window !== 'undefined' && import.meta.env.VITE_API_BASE_URL) {
    const apiUrl = import.meta.env.VITE_API_BASE_URL as string
    const regionMatch = apiUrl.match(/(\w+)\.(?:azure|speech)/)
    return {
      endpoint: apiUrl,
      region: regionMatch ? regionMatch[1] : undefined,
    }
  }
  return {}
}

export function Layout({ children, currentView, onNavigate }: LayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [selectedLanguage, setSelectedLanguage] = useState(getInitialLanguage)
  const [version, setVersion] = useState(FALLBACK_VERSION)
  const [speechStatus, setSpeechStatus] = useState<ServiceStatus>({
    name: 'Azure Speech Service',
    connected: false,
    message: 'Checking...',
  })
  const [openAIStatus, setOpenAIStatus] = useState<ServiceStatus>({
    name: 'Azure OpenAI',
    connected: false,
    message: 'Checking...',
  })

  // Compute endpoint info once
  const { endpoint, region } = useMemo(() => getInitialEndpointInfo(), [])

  const toggleSidebar = useCallback(() => {
    setSidebarOpen((prev) => !prev)
  }, [])

  const closeSidebar = useCallback(() => {
    setSidebarOpen(false)
  }, [])

  const handleLanguageChange = useCallback((code: string) => {
    setSelectedLanguage(code)
    localStorage.setItem('preferredLanguage', code)
  }, [])

  // Fetch version from backend on mount
  useEffect(() => {
    const fetchVersion = async () => {
      try {
        const healthResponse = await getHealth()
        if (healthResponse.version) {
          setVersion(healthResponse.version)
        }
      } catch {
        // Use fallback version if API unavailable
        console.warn('Could not fetch version from API, using fallback')
      }
    }
    fetchVersion()
  }, [])

  // Check service status on mount
  useEffect(() => {
    const checkServices = async () => {
      try {
        const readyResponse = await getReady()

        // Parse dependencies
        const speechDep = readyResponse.dependencies.find(
          (d) => d.name.toLowerCase().includes('speech') || d.name.toLowerCase().includes('azure')
        )
        const openAIDep = readyResponse.dependencies.find(
          (d) => d.name.toLowerCase().includes('openai') || d.name.toLowerCase().includes('gpt')
        )

        if (speechDep) {
          setSpeechStatus({
            name: 'Azure Speech Service',
            connected: speechDep.healthy,
            message: speechDep.healthy
              ? 'Azure Government Cloud'
              : speechDep.message || 'Not configured',
          })
        }

        if (openAIDep) {
          setOpenAIStatus({
            name: 'Azure OpenAI',
            connected: openAIDep.healthy,
            message: openAIDep.healthy ? 'Connected' : openAIDep.message || 'Not available',
          })
        }
      } catch {
        // API not available - set default status
        setSpeechStatus({
          name: 'Azure Speech Service',
          connected: false,
          message: 'API unavailable',
        })
        setOpenAIStatus({
          name: 'Azure OpenAI',
          connected: false,
          message: 'API unavailable',
        })
      }
    }

    checkServices()
  }, [])

  const handleTestSpeech = useCallback(async () => {
    setSpeechStatus((prev) => ({ ...prev, message: 'Testing...' }))
    try {
      const response = await getReady()
      const speechDep = response.dependencies.find(
        (d) => d.name.toLowerCase().includes('speech')
      )
      setSpeechStatus({
        name: 'Azure Speech Service',
        connected: speechDep?.healthy ?? false,
        message: speechDep?.healthy
          ? 'Connection successful'
          : speechDep?.message || 'Connection failed',
      })
    } catch {
      setSpeechStatus({
        name: 'Azure Speech Service',
        connected: false,
        message: 'Connection failed',
      })
    }
  }, [])

  const handleTestOpenAI = useCallback(async () => {
    setOpenAIStatus((prev) => ({ ...prev, message: 'Testing...' }))
    try {
      const response = await getReady()
      const openAIDep = response.dependencies.find(
        (d) => d.name.toLowerCase().includes('openai')
      )
      setOpenAIStatus({
        name: 'Azure OpenAI',
        connected: openAIDep?.healthy ?? false,
        message: openAIDep?.healthy
          ? 'Connection successful'
          : openAIDep?.message || 'Connection failed',
      })
    } catch {
      setOpenAIStatus({
        name: 'Azure OpenAI',
        connected: false,
        message: 'Connection failed',
      })
    }
  }, [])

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Header */}
      <Header onMenuClick={toggleSidebar} />

      {/* Main area with sidebar */}
      <div className="flex flex-1">
        {/* Sidebar */}
        <Sidebar
          isOpen={sidebarOpen}
          onClose={closeSidebar}
          selectedLanguage={selectedLanguage}
          onLanguageChange={handleLanguageChange}
          speechStatus={speechStatus}
          openAIStatus={openAIStatus}
          region={region}
          endpoint={endpoint}
          version={version}
          onTestSpeech={handleTestSpeech}
          onTestOpenAI={handleTestOpenAI}
          currentView={currentView}
          onNavigate={onNavigate}
        />

        {/* Main content */}
        <main className="flex-1 p-4 lg:p-8">{children}</main>
      </div>

      {/* Footer */}
      <footer className="bg-gray-100 border-t border-gray-200 py-2">
        <div className="container mx-auto px-4 text-center text-sm text-gray-500">
          Captain&apos;s Log v{version} | React + TypeScript + Tailwind CSS
        </div>
      </footer>
    </div>
  )
}
