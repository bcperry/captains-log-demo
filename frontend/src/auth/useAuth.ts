import { useMsal, useIsAuthenticated } from '@azure/msal-react'
import { loginRequest } from './msalConfig'
import { setAccessToken } from '../services/api'

export interface AuthUser {
  id: string
  name: string
  email: string
}

export interface AuthState {
  isAuthenticated: boolean
  isLoading: boolean
  user: AuthUser | null
  login: (usePopup?: boolean) => Promise<void>
  logout: (usePopup?: boolean) => Promise<void>
  getToken: () => Promise<string | null>
}

export function useAuth(): AuthState {
  const { instance, accounts, inProgress } = useMsal()
  const isAuthenticated = useIsAuthenticated()

  const activeAccount = instance.getActiveAccount() || accounts[0] || null

  const login = async (usePopup = false): Promise<void> => {
    try {
      if (usePopup) {
        const response = await instance.loginPopup(loginRequest)
        instance.setActiveAccount(response.account)
        setAccessToken(response.accessToken)
      } else {
        await instance.loginRedirect(loginRequest)
      }
    } catch (error) {
      console.error('Login failed:', error)
      throw error
    }
  }

  const logout = async (usePopup = false): Promise<void> => {
    try {
      const account = instance.getActiveAccount()
      if (usePopup) {
        await instance.logoutPopup({ account })
      } else {
        await instance.logoutRedirect({ account })
      }
      setAccessToken(null)
    } catch (error) {
      console.error('Logout failed:', error)
      throw error
    }
  }

  const getToken = async (): Promise<string | null> => {
    if (!activeAccount) return null

    try {
      const response = await instance.acquireTokenSilent({
        ...loginRequest,
        account: activeAccount,
      })
      setAccessToken(response.accessToken)
      return response.accessToken
    } catch (error) {
      console.error('Silent token acquisition failed:', error)
      // Fall back to interactive
      try {
        const response = await instance.acquireTokenPopup(loginRequest)
        setAccessToken(response.accessToken)
        return response.accessToken
      } catch (popupError) {
        console.error('Interactive token acquisition failed:', popupError)
        return null
      }
    }
  }

  return {
    isAuthenticated,
    isLoading: inProgress !== 'none',
    user: activeAccount
      ? {
          id: activeAccount.localAccountId,
          name: activeAccount.name || '',
          email: activeAccount.username || '',
        }
      : null,
    login,
    logout,
    getToken,
  }
}
