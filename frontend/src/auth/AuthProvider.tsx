import type { ReactNode } from 'react'
import { useEffect, useState } from 'react'
import {
  MsalProvider,
  AuthenticatedTemplate,
  UnauthenticatedTemplate,
} from '@azure/msal-react'
import type { AccountInfo } from '@azure/msal-browser'
import { EventType } from '@azure/msal-browser'
import { msalInstance } from './msalInstance'
import { loginRequest } from './msalConfig'
import { setAccessToken } from '../services/api'

interface AuthProviderProps {
  children: ReactNode
}

export function AuthProvider({ children }: AuthProviderProps): ReactNode {
  const [isInitialized, setIsInitialized] = useState(false)

  useEffect(() => {
    const initializeMsal = async (): Promise<void> => {
      try {
        await msalInstance.initialize()

        // Handle redirect promise
        const response = await msalInstance.handleRedirectPromise()
        if (response) {
          msalInstance.setActiveAccount(response.account)
          setAccessToken(response.accessToken)
        } else {
          // If no redirect response, check for active account
          const accounts = msalInstance.getAllAccounts()
          if (accounts.length > 0) {
            const activeAccount = accounts[0]
            msalInstance.setActiveAccount(activeAccount)
            // Acquire token silently for existing session
            try {
              const tokenResponse = await msalInstance.acquireTokenSilent({
                ...loginRequest,
                account: activeAccount,
              })
              setAccessToken(tokenResponse.accessToken)
            } catch (tokenError) {
              console.warn('Silent token acquisition failed:', tokenError)
              // Token will be acquired on next API call or user action
            }
          }
        }

        // Register event callback for account changes
        msalInstance.addEventCallback((event) => {
          if (event.eventType === EventType.LOGIN_SUCCESS && event.payload) {
            const payload = event.payload as { account: AccountInfo; accessToken: string }
            msalInstance.setActiveAccount(payload.account)
            if (payload.accessToken) {
              setAccessToken(payload.accessToken)
            }
          }
          if (event.eventType === EventType.LOGOUT_SUCCESS) {
            setAccessToken(null)
          }
        })

        setIsInitialized(true)
      } catch (error) {
        console.error('MSAL initialization failed:', error)
        setIsInitialized(true) // Still render but in non-auth state
      }
    }

    initializeMsal()
  }, [])

  if (!isInitialized) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    )
  }

  return <MsalProvider instance={msalInstance}>{children}</MsalProvider>
}

// Re-export MSAL components for convenience
export { AuthenticatedTemplate, UnauthenticatedTemplate }
