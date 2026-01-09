import { describe, it, expect, vi, beforeEach } from 'vitest'
import { msalConfig, loginRequest, apiScopes } from '../auth/msalConfig'
import { config } from '../config'

// Test MSAL configuration
describe('msalConfig', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it('has auth configuration with clientId', () => {
    expect(msalConfig.auth).toBeDefined()
    expect(msalConfig.auth.clientId).toBe(config.azure.clientId)
  })

  it('has cache configuration with localStorage', () => {
    expect(msalConfig.cache).toBeDefined()
    expect(msalConfig.cache?.cacheLocation).toBe('localStorage')
  })

  it('has system configuration with logger', () => {
    expect(msalConfig.system).toBeDefined()
    expect(msalConfig.system?.loggerOptions).toBeDefined()
  })

  it('has correct authority for commercial cloud', () => {
    expect(msalConfig.auth.authority).toContain(config.azure.tenantId)
  })

  it('has navigateToLoginRequestUrl enabled', () => {
    expect(msalConfig.auth.navigateToLoginRequestUrl).toBe(true)
  })
})

describe('loginRequest', () => {
  it('has scopes array', () => {
    expect(loginRequest.scopes).toBeDefined()
    expect(Array.isArray(loginRequest.scopes)).toBe(true)
  })

  it('scopes match apiScopes', () => {
    expect(loginRequest.scopes).toEqual(apiScopes)
  })
})

describe('apiScopes', () => {
  it('is an array', () => {
    expect(Array.isArray(apiScopes)).toBe(true)
  })

  it('generates api scope from clientId when configured', () => {
    if (config.azure.clientId) {
      expect(apiScopes).toContain(`api://${config.azure.clientId}/.default`)
    } else {
      expect(apiScopes).toHaveLength(0)
    }
  })
})

// Test component rendering without MSAL context
describe('LoginButton component structure', () => {
  it('module exports LoginButton', async () => {
    const { LoginButton } = await import('../components/LoginButton')
    expect(LoginButton).toBeDefined()
    expect(typeof LoginButton).toBe('function')
  })
})

describe('UserInfo component structure', () => {
  it('module exports UserInfo', async () => {
    const { UserInfo } = await import('../components/UserInfo')
    expect(UserInfo).toBeDefined()
    expect(typeof UserInfo).toBe('function')
  })
})

describe('ProtectedRoute component structure', () => {
  it('module exports ProtectedRoute', async () => {
    const { ProtectedRoute } = await import('../components/ProtectedRoute')
    expect(ProtectedRoute).toBeDefined()
    expect(typeof ProtectedRoute).toBe('function')
  })
})

describe('useAuth hook structure', () => {
  it('module exports useAuth', async () => {
    const { useAuth } = await import('../auth/useAuth')
    expect(useAuth).toBeDefined()
    expect(typeof useAuth).toBe('function')
  })
})

describe('AuthProvider component structure', () => {
  it('module exports AuthProvider', async () => {
    const { AuthProvider } = await import('../auth/AuthProvider')
    expect(AuthProvider).toBeDefined()
    expect(typeof AuthProvider).toBe('function')
  })

  it('module exports AuthenticatedTemplate', async () => {
    const { AuthenticatedTemplate } = await import('../auth/AuthProvider')
    expect(AuthenticatedTemplate).toBeDefined()
  })

  it('module exports UnauthenticatedTemplate', async () => {
    const { UnauthenticatedTemplate } = await import('../auth/AuthProvider')
    expect(UnauthenticatedTemplate).toBeDefined()
  })
})

describe('auth index exports', () => {
  it('exports all required auth modules', async () => {
    const auth = await import('../auth')
    expect(auth.msalConfig).toBeDefined()
    expect(auth.loginRequest).toBeDefined()
    expect(auth.silentRequest).toBeDefined()
    expect(auth.apiScopes).toBeDefined()
    expect(auth.msalInstance).toBeDefined()
    expect(auth.useAuth).toBeDefined()
    expect(auth.AuthProvider).toBeDefined()
    expect(auth.AuthenticatedTemplate).toBeDefined()
    expect(auth.UnauthenticatedTemplate).toBeDefined()
  })
})

describe('components index exports', () => {
  it('exports all required components', async () => {
    const components = await import('../components')
    expect(components.ProtectedRoute).toBeDefined()
    expect(components.UserInfo).toBeDefined()
    expect(components.LoginButton).toBeDefined()
  })
})

describe('hooks index exports', () => {
  it('exports useAuth hook', async () => {
    const hooks = await import('../hooks')
    expect(hooks.useAuth).toBeDefined()
  })
})
