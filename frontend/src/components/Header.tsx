import { useAuth, AuthenticatedTemplate, UnauthenticatedTemplate } from '../auth'
import { UserInfo } from './UserInfo'
import { LoginButton } from './LoginButton'

interface HeaderProps {
  onMenuClick: () => void
}

export function Header({ onMenuClick }: HeaderProps) {
  const { isLoading } = useAuth()

  return (
    <header className="bg-gradient-to-r from-[#0078d4] to-[#00bcf2] text-white shadow-lg">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            {/* Mobile menu button */}
            <button
              onClick={onMenuClick}
              className="lg:hidden p-2 hover:bg-white/10 rounded-lg transition-colors"
              aria-label="Toggle sidebar"
            >
              <svg
                className="w-6 h-6"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M4 6h16M4 12h16M4 18h16"
                />
              </svg>
            </button>

            <div>
              <div className="flex items-center gap-3">
                {/* Microphone icon */}
                <svg
                  className="w-8 h-8"
                  fill="currentColor"
                  viewBox="0 0 24 24"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
                  <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
                </svg>
                <h1 className="text-2xl font-bold">Captain&apos;s Log</h1>
              </div>
              <p className="text-white/80 mt-1 hidden sm:block">
                Voice Transcription & Analysis
              </p>
            </div>
          </div>

          {/* Auth controls */}
          <div className="flex items-center">
            {isLoading ? (
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-white"></div>
            ) : (
              <>
                <AuthenticatedTemplate>
                  <UserInfo />
                </AuthenticatedTemplate>
                <UnauthenticatedTemplate>
                  <LoginButton />
                </UnauthenticatedTemplate>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  )
}
