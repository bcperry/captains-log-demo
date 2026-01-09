import { useAuth } from '../auth'

interface LoginButtonProps {
  className?: string
  usePopup?: boolean
}

export function LoginButton({ className = '', usePopup = false }: LoginButtonProps) {
  const { isAuthenticated, login, isLoading } = useAuth()

  const handleLogin = async () => {
    try {
      await login(usePopup)
    } catch (error) {
      console.error('Login error:', error)
    }
  }

  if (isAuthenticated) {
    return null
  }

  return (
    <button
      onClick={handleLogin}
      disabled={isLoading}
      className={`bg-white text-blue-600 hover:bg-blue-50 font-semibold py-2 px-4 rounded-lg transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2 ${className}`}
    >
      {isLoading ? (
        <>
          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
          Signing in...
        </>
      ) : (
        <>
          <svg className="w-4 h-4" viewBox="0 0 21 21" xmlns="http://www.w3.org/2000/svg">
            <rect x="1" y="1" width="9" height="9" fill="#f25022" />
            <rect x="1" y="11" width="9" height="9" fill="#00a4ef" />
            <rect x="11" y="1" width="9" height="9" fill="#7fba00" />
            <rect x="11" y="11" width="9" height="9" fill="#ffb900" />
          </svg>
          Sign in
        </>
      )}
    </button>
  )
}
