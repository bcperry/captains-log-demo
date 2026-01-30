import { PublicClientApplication } from '@azure/msal-browser'
import { msalConfig } from './msalConfig'

// Create MSAL instance
export const msalInstance = new PublicClientApplication(msalConfig)
