import type { Configuration } from '@azure/msal-browser'
import { LogLevel, BrowserCacheLocation } from '@azure/msal-browser'
import { config, type AzureCloud } from '../config'

const getCloudInstance = (cloud: AzureCloud): string => {
  return cloud === 'government'
    ? 'https://login.microsoftonline.us'
    : 'https://login.microsoftonline.com'
}

export const msalConfig: Configuration = {
  auth: {
    clientId: config.azure.clientId,
    authority: config.azure.authority,
    redirectUri: config.azure.redirectUri,
    postLogoutRedirectUri: config.azure.redirectUri,
    navigateToLoginRequestUrl: true,
    cloudDiscoveryMetadata: undefined,
    knownAuthorities: [getCloudInstance(config.azure.cloud)],
  },
  cache: {
    cacheLocation: BrowserCacheLocation.LocalStorage,
    storeAuthStateInCookie: false,
  },
  system: {
    loggerOptions: {
      loggerCallback: (level, message, containsPii) => {
        if (containsPii) return
        switch (level) {
          case LogLevel.Error:
            console.error(message)
            break
          case LogLevel.Warning:
            console.warn(message)
            break
          case LogLevel.Info:
            // Only log in development
            if (import.meta.env.DEV) {
              console.info(message)
            }
            break
          case LogLevel.Verbose:
          case LogLevel.Trace:
            break
        }
      },
      logLevel: import.meta.env.DEV ? LogLevel.Info : LogLevel.Error,
    },
  },
}

// Scopes for the API access
export const apiScopes = config.azure.clientId
  ? [`api://${config.azure.clientId}/.default`]
  : []

// Login request configuration
export const loginRequest = {
  scopes: apiScopes,
}

// Silent token request configuration
export const silentRequest = {
  scopes: apiScopes,
  forceRefresh: false,
}
