// Layout types

export interface Language {
  code: string
  name: string
}

export const SUPPORTED_LANGUAGES: Language[] = [
  { code: 'en-US', name: 'English (US)' },
  { code: 'en-GB', name: 'English (UK)' },
  { code: 'es-ES', name: 'Spanish' },
  { code: 'fr-FR', name: 'French' },
  { code: 'de-DE', name: 'German' },
  { code: 'it-IT', name: 'Italian' },
  { code: 'pt-BR', name: 'Portuguese' },
  { code: 'zh-CN', name: 'Chinese (Mandarin)' },
  { code: 'ja-JP', name: 'Japanese' },
  { code: 'ko-KR', name: 'Korean' },
]

export interface ServiceStatus {
  name: string
  connected: boolean
  message?: string
}

export interface SidebarProps {
  isOpen: boolean
  onClose: () => void
  selectedLanguage: string
  onLanguageChange: (code: string) => void
  speechStatus: ServiceStatus
  openAIStatus: ServiceStatus
  region?: string
  endpoint?: string
  version: string
  onTestSpeech?: () => void
  onTestOpenAI?: () => void
}

export interface HeaderProps {
  onMenuClick: () => void
}

export interface LayoutProps {
  children: React.ReactNode
}
