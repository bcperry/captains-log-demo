import { describe, it, expect, vi, beforeEach } from 'vitest'
import { SUPPORTED_LANGUAGES } from '../types/layout'
import type { ServiceStatus, SidebarProps } from '../types/layout'

// Test layout types
describe('SUPPORTED_LANGUAGES', () => {
  it('contains English (US) as first language', () => {
    expect(SUPPORTED_LANGUAGES[0]).toEqual({ code: 'en-US', name: 'English (US)' })
  })

  it('contains all 10 supported languages', () => {
    expect(SUPPORTED_LANGUAGES).toHaveLength(10)
  })

  it('each language has code and name', () => {
    SUPPORTED_LANGUAGES.forEach((lang) => {
      expect(lang.code).toBeDefined()
      expect(lang.name).toBeDefined()
      expect(lang.code.length).toBeGreaterThan(0)
      expect(lang.name.length).toBeGreaterThan(0)
    })
  })

  it('includes major world languages', () => {
    const codes = SUPPORTED_LANGUAGES.map((l) => l.code)
    expect(codes).toContain('en-US')
    expect(codes).toContain('es-ES')
    expect(codes).toContain('fr-FR')
    expect(codes).toContain('de-DE')
    expect(codes).toContain('zh-CN')
    expect(codes).toContain('ja-JP')
  })
})

// Test Header component structure
describe('Header component structure', () => {
  it('module exports Header', async () => {
    const { Header } = await import('../components/Header')
    expect(Header).toBeDefined()
    expect(typeof Header).toBe('function')
  })
})

// Test Sidebar component structure
describe('Sidebar component structure', () => {
  it('module exports Sidebar', async () => {
    const { Sidebar } = await import('../components/Sidebar')
    expect(Sidebar).toBeDefined()
    expect(typeof Sidebar).toBe('function')
  })
})

// Test Layout component structure
describe('Layout component structure', () => {
  it('module exports Layout', async () => {
    const { Layout } = await import('../components/Layout')
    expect(Layout).toBeDefined()
    expect(typeof Layout).toBe('function')
  })
})

// Test components index exports new components
describe('components index exports layout components', () => {
  it('exports Header component', async () => {
    const components = await import('../components')
    expect(components.Header).toBeDefined()
  })

  it('exports Sidebar component', async () => {
    const components = await import('../components')
    expect(components.Sidebar).toBeDefined()
  })

  it('exports Layout component', async () => {
    const components = await import('../components')
    expect(components.Layout).toBeDefined()
  })
})

// Test types index exports layout types
describe('types index exports layout types', () => {
  it('exports SUPPORTED_LANGUAGES', async () => {
    const types = await import('../types')
    expect(types.SUPPORTED_LANGUAGES).toBeDefined()
    expect(Array.isArray(types.SUPPORTED_LANGUAGES)).toBe(true)
  })
})

// Test ServiceStatus type usage
describe('ServiceStatus type', () => {
  it('can create connected status', () => {
    const status: ServiceStatus = {
      name: 'Azure Speech Service',
      connected: true,
      message: 'Connected successfully',
    }
    expect(status.connected).toBe(true)
    expect(status.name).toBe('Azure Speech Service')
  })

  it('can create disconnected status', () => {
    const status: ServiceStatus = {
      name: 'Azure OpenAI',
      connected: false,
      message: 'Not configured',
    }
    expect(status.connected).toBe(false)
  })

  it('message is optional', () => {
    const status: ServiceStatus = {
      name: 'Test Service',
      connected: true,
    }
    expect(status.message).toBeUndefined()
  })
})

// Test SidebarProps type usage
describe('SidebarProps type', () => {
  it('defines required sidebar props', () => {
    const mockProps: SidebarProps = {
      isOpen: true,
      onClose: vi.fn(),
      selectedLanguage: 'en-US',
      onLanguageChange: vi.fn(),
      speechStatus: { name: 'Speech', connected: true },
      openAIStatus: { name: 'OpenAI', connected: false },
      version: '1.0.0',
    }
    expect(mockProps.isOpen).toBe(true)
    expect(mockProps.selectedLanguage).toBe('en-US')
    expect(mockProps.version).toBe('1.0.0')
  })

  it('supports optional props', () => {
    const mockProps: SidebarProps = {
      isOpen: false,
      onClose: vi.fn(),
      selectedLanguage: 'es-ES',
      onLanguageChange: vi.fn(),
      speechStatus: { name: 'Speech', connected: true },
      openAIStatus: { name: 'OpenAI', connected: true },
      region: 'usgovvirginia',
      endpoint: 'https://example.speech.azure.us',
      version: '1.0.0',
      onTestSpeech: vi.fn(),
      onTestOpenAI: vi.fn(),
    }
    expect(mockProps.region).toBe('usgovvirginia')
    expect(mockProps.endpoint).toBeDefined()
    expect(mockProps.onTestSpeech).toBeDefined()
    expect(mockProps.onTestOpenAI).toBeDefined()
  })
})

// Test API getReady is available for Layout
describe('API functions for Layout', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('getReady function exists', async () => {
    const api = await import('../services/api')
    expect(api.getReady).toBeDefined()
    expect(typeof api.getReady).toBe('function')
  })
})
