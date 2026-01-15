import { useState, useCallback, useRef } from 'react'
import { analyzeTranscription as analyzeApi } from '../services/api'
import { useAuthenticatedApi } from './useAuthenticatedApi'
import type { AnalysisResult } from '../types'

export interface AnalysisProgress {
  status: 'idle' | 'analyzing' | 'complete' | 'error'
  message?: string
}

export interface AnalysisState {
  progress: AnalysisProgress
  result: AnalysisResult | null
  error: string | null
}

export interface UseAnalysisOptions {
  onComplete?: (result: AnalysisResult) => void
  onError?: (error: string) => void
}

export interface UseAnalysisReturn {
  state: AnalysisState
  analyze: (text: string, diarizedTranscript?: string) => Promise<void>
  cancel: () => void
  reset: () => void
  isAnalyzing: boolean
  hasResult: boolean
}

const initialProgress: AnalysisProgress = {
  status: 'idle',
}

const initialState: AnalysisState = {
  progress: initialProgress,
  result: null,
  error: null,
}

/**
 * Hook for managing AI analysis of transcription text.
 */
export function useAnalysis(options: UseAnalysisOptions = {}): UseAnalysisReturn {
  const { onComplete, onError } = options
  const [state, setState] = useState<AnalysisState>(initialState)
  const abortControllerRef = useRef<AbortController | null>(null)
  const { withAuth } = useAuthenticatedApi()

  const analyze = useCallback(
    async (text: string, diarizedTranscript?: string) => {
      if (!text.trim()) {
        const errorMsg = 'No text to analyze'
        setState((prev) => ({ ...prev, error: errorMsg }))
        onError?.(errorMsg)
        return
      }

      // Create abort controller for cancellation
      abortControllerRef.current = new AbortController()

      try {
        // Update progress: analyzing
        setState((prev) => ({
          ...prev,
          error: null,
          progress: {
            status: 'analyzing',
            message: 'Analyzing transcription with AI...',
          },
        }))

        // Call API with authentication - pass diarized transcript if available
        const result = await withAuth(() => analyzeApi(text, diarizedTranscript))

        // Check if cancelled
        if (abortControllerRef.current?.signal.aborted) {
          return
        }

        // Update state with result
        setState((prev) => ({
          ...prev,
          result,
          progress: {
            status: 'complete',
            message: 'Analysis complete!',
          },
        }))

        onComplete?.(result)
      } catch (error) {
        // Check if cancelled
        if (abortControllerRef.current?.signal.aborted) {
          return
        }

        const errorMsg = error instanceof Error ? error.message : 'Analysis failed'
        setState((prev) => ({
          ...prev,
          error: errorMsg,
          progress: {
            status: 'error',
            message: errorMsg,
          },
        }))
        onError?.(errorMsg)
      } finally {
        abortControllerRef.current = null
      }
    },
    [onComplete, onError, withAuth]
  )

  const cancel = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    setState((prev) => ({
      ...prev,
      progress: {
        status: 'idle',
        message: 'Analysis cancelled',
      },
    }))
  }, [])

  const reset = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    setState(initialState)
  }, [])

  return {
    state,
    analyze,
    cancel,
    reset,
    isAnalyzing: state.progress.status === 'analyzing',
    hasResult: state.result !== null,
  }
}
