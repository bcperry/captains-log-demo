export type AzureCloud = 'commercial' | 'government'

export interface AppConfig {
  apiBaseUrl: string
  azure: {
    clientId: string
    tenantId: string
    redirectUri: string
    cloud: AzureCloud
    authority: string
  }
}

const getAzureAuthority = (tenantId: string, cloud: AzureCloud): string => {
  const domain = cloud === 'government' ? 'login.microsoftonline.us' : 'login.microsoftonline.com'
  return `https://${domain}/${tenantId}`
}

// In production (when served from FastAPI), use same origin for API
// In development (Vite dev server), use the configured API URL
const getApiBaseUrl = (): string => {
  // If VITE_API_BASE_URL is set, use it
  if (import.meta.env.VITE_API_BASE_URL) {
    return import.meta.env.VITE_API_BASE_URL
  }
  // In production, API is served from same origin (no base URL needed)
  if (import.meta.env.PROD) {
    return ''
  }
  // Default for development
  return 'http://localhost:8001'
}

// Get redirect URI - use current origin in production
const getRedirectUri = (): string => {
  if (import.meta.env.VITE_AZURE_REDIRECT_URI) {
    return import.meta.env.VITE_AZURE_REDIRECT_URI
  }
  if (import.meta.env.PROD && typeof window !== 'undefined') {
    return window.location.origin
  }
  return 'http://localhost:3000'
}

export const config: AppConfig = {
  apiBaseUrl: getApiBaseUrl(),
  azure: {
    clientId: import.meta.env.VITE_AZURE_CLIENT_ID || '',
    tenantId: import.meta.env.VITE_AZURE_TENANT_ID || '',
    redirectUri: getRedirectUri(),
    cloud: (import.meta.env.VITE_AZURE_CLOUD as AzureCloud) || 'commercial',
    get authority() {
      return getAzureAuthority(this.tenantId, this.cloud)
    },
  },
}
