import { useCallback } from 'react'
import { useMsal } from '@azure/msal-react'
import { loginRequest } from '../auth/msalConfig'
import { setAccessToken } from '../services/api'

/**
 * Hook that provides authenticated API call wrapper.
 * Ensures the access token is acquired/refreshed before each API call.
 */
export function useAuthenticatedApi() {
  const { instance, accounts } = useMsal()

  /**
   * Ensures a valid access token is available before making an API call.
   * Acquires token silently, or falls back to popup if needed.
   */
  const ensureToken = useCallback(async (): Promise<string | null> => {
    const activeAccount = instance.getActiveAccount() || accounts[0]
    if (!activeAccount) {
      console.warn('No active account for token acquisition')
      return null
    }

    try {
      const response = await instance.acquireTokenSilent({
        ...loginRequest,
        account: activeAccount,
      })
      setAccessToken(response.accessToken)
      return response.accessToken
    } catch (error) {
      console.warn('Silent token acquisition failed, trying popup:', error)
      try {
        const response = await instance.acquireTokenPopup(loginRequest)
        setAccessToken(response.accessToken)
        return response.accessToken
      } catch (popupError) {
        console.error('Token acquisition failed:', popupError)
        setAccessToken(null)
        return null
      }
    }
  }, [instance, accounts])

  /**
   * Wraps an API call function to ensure token is valid before execution.
   * @param apiCall - The API function to call
   * @returns The result of the API call
   */
  const withAuth = useCallback(
    async <T>(apiCall: () => Promise<T>): Promise<T> => {
      const token = await ensureToken()
      if (!token) {
        throw new Error('Authentication required. Please sign in.')
      }
      return apiCall()
    },
    [ensureToken]
  )

  return {
    ensureToken,
    withAuth,
  }
}
