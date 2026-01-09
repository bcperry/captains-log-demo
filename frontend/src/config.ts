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

export const config: AppConfig = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8001',
  azure: {
    clientId: import.meta.env.VITE_AZURE_CLIENT_ID || '',
    tenantId: import.meta.env.VITE_AZURE_TENANT_ID || '',
    redirectUri: import.meta.env.VITE_AZURE_REDIRECT_URI || 'http://localhost:3000',
    cloud: (import.meta.env.VITE_AZURE_CLOUD as AzureCloud) || 'commercial',
    get authority() {
      return getAzureAuthority(this.tenantId, this.cloud)
    },
  },
}
