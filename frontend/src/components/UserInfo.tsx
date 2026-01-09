import { useAuth } from '../auth'

interface UserInfoProps {
  className?: string
}

export function UserInfo({ className = '' }: UserInfoProps) {
  const { isAuthenticated, user, logout, isLoading } = useAuth()

  const handleLogout = async () => {
    try {
      await logout(false) // Use redirect flow
    } catch (error) {
      console.error('Logout error:', error)
    }
  }

  if (isLoading) {
    return (
      <div className={`flex items-center gap-2 ${className}`}>
        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
      </div>
    )
  }

  if (!isAuthenticated || !user) {
    return null
  }

  return (
    <div className={`flex items-center gap-3 ${className}`}>
      <div className="text-right">
        <div className="text-sm font-medium text-white">{user.name}</div>
        <div className="text-xs text-white/70">{user.email}</div>
      </div>
      <button
        onClick={handleLogout}
        className="bg-white/20 hover:bg-white/30 text-white text-sm font-medium py-1.5 px-3 rounded-md transition-colors duration-200"
      >
        Sign out
      </button>
    </div>
  )
}
