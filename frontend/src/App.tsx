import './App.css'
import { useAuth, AuthenticatedTemplate, UnauthenticatedTemplate } from './auth'
import { UserInfo, LoginButton } from './components'

function App() {
  const { isLoading } = useAuth()

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-gradient-to-r from-[#0078d4] to-[#00bcf2] text-white shadow-lg">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div>
              <div className="flex items-center gap-3">
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
              <p className="text-white/80 mt-1">Voice Transcription & Analysis</p>
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

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        <AuthenticatedTemplate>
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-xl font-semibold text-gray-800 mb-4">
              Welcome to Captain&apos;s Log
            </h2>
            <p className="text-gray-600">
              You are signed in. Transcription and analysis features coming in subsequent user
              stories.
            </p>
          </div>
        </AuthenticatedTemplate>

        <UnauthenticatedTemplate>
          <div className="bg-white rounded-lg shadow-md p-6">
            <div className="text-center py-8">
              <svg
                className="w-16 h-16 mx-auto text-blue-600 mb-4"
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
              <h2 className="text-xl font-semibold text-gray-800 mb-2">Sign in to get started</h2>
              <p className="text-gray-600 mb-6">
                Use your Microsoft account to access voice transcription and AI analysis features.
              </p>
              <LoginButton className="mx-auto" />
            </div>
          </div>
        </UnauthenticatedTemplate>
      </main>

      {/* Footer */}
      <footer className="fixed bottom-0 left-0 right-0 bg-gray-100 border-t border-gray-200 py-2">
        <div className="container mx-auto px-4 text-center text-sm text-gray-500">
          Captain&apos;s Log v1.0.0 | React + TypeScript + Tailwind CSS
        </div>
      </footer>
    </div>
  )
}

export default App
