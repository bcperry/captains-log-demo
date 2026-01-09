import type { ReactNode } from 'react'
import { useAuth, UnauthenticatedTemplate, AuthenticatedTemplate } from '../auth'

interface ProtectedRouteProps {
  children: ReactNode
  fallback?: ReactNode
}

function LoginPrompt() {
  const { login, isLoading } = useAuth()

  const handleLogin = async () => {
    try {
      await login(false) // Use redirect flow
    } catch (error) {
      console.error('Login error:', error)
    }
  }

  return (
    <div className="min-h-[60vh] flex items-center justify-center">
      <div className="bg-white rounded-lg shadow-lg p-8 max-w-md w-full text-center">
        <div className="mb-6">
          <svg
            className="w-16 h-16 mx-auto text-blue-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
            />
          </svg>
        </div>
        <h2 className="text-2xl font-bold text-gray-800 mb-2">Authentication Required</h2>
        <p className="text-gray-600 mb-6">
          Please sign in with your Microsoft account to access this application.
        </p>
        <button
          onClick={handleLogin}
          disabled={isLoading}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {isLoading ? (
            <>
              <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
              Signing in...
            </>
          ) : (
            <>
              <svg className="w-5 h-5" viewBox="0 0 21 21" xmlns="http://www.w3.org/2000/svg">
                <rect x="1" y="1" width="9" height="9" fill="#f25022" />
                <rect x="1" y="11" width="9" height="9" fill="#00a4ef" />
                <rect x="11" y="1" width="9" height="9" fill="#7fba00" />
                <rect x="11" y="11" width="9" height="9" fill="#ffb900" />
              </svg>
              Sign in with Microsoft
            </>
          )}
        </button>
      </div>
    </div>
  )
}

export function ProtectedRoute({ children, fallback }: ProtectedRouteProps): ReactNode {
  return (
    <>
      <AuthenticatedTemplate>{children}</AuthenticatedTemplate>
      <UnauthenticatedTemplate>{fallback || <LoginPrompt />}</UnauthenticatedTemplate>
    </>
  )
}
