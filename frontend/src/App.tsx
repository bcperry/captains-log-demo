import { useState } from 'react'
import './App.css'
import { AuthenticatedTemplate, UnauthenticatedTemplate } from './auth'
import { LoginButton, Layout, AudioUpload, TranscriptionResults } from './components'
import type { TranscriptionResult } from './types/transcription'

function App() {
  const [transcriptionResult, setTranscriptionResult] = useState<TranscriptionResult | null>(null)

  const handleTranscriptionComplete = (result: TranscriptionResult) => {
    setTranscriptionResult(result)
  }

  const handleClear = () => {
    setTranscriptionResult(null)
  }

  return (
    <Layout>
      <AuthenticatedTemplate>
        <div className="space-y-6">
          <AudioUpload
            language="en-US"
            onTranscriptionComplete={handleTranscriptionComplete}
            onError={(error) => {
              console.error('Transcription error:', error)
            }}
          />
          {transcriptionResult && (
            <TranscriptionResults
              transcription={transcriptionResult}
              onClear={handleClear}
            />
          )}
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
