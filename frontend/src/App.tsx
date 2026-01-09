import './App.css'
import { AuthenticatedTemplate, UnauthenticatedTemplate } from './auth'
import { LoginButton, Layout } from './components'

function App() {
  return (
    <Layout>
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
    </Layout>
  )
}

export default App
