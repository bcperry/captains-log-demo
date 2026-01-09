import { useState, useCallback, useRef } from 'react'
import { transcribeAudio as transcribeAudioApi } from '../services/api'
import { useAuthenticatedApi } from './useAuthenticatedApi'
import type {
  FileInfo,
  TranscriptionProgress,
  TranscriptionResult,
  TranscriptionState,
} from '../types/transcription'

// Re-export constants for use in components
export { SUPPORTED_AUDIO_FORMATS, SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_BYTES } from '../types/transcription'

export interface UseTranscriptionOptions {
  language?: string
  onComplete?: (result: TranscriptionResult) => void
  onError?: (error: string) => void
}

export interface UseTranscriptionReturn {
  state: TranscriptionState
  selectFile: (file: File) => void
  clearFile: () => void
  startTranscription: () => Promise<void>
  cancelTranscription: () => void
  reset: () => void
  isTranscribing: boolean
  hasFile: boolean
  hasResult: boolean
}

const initialProgress: TranscriptionProgress = {
  status: 'idle',
  progress: 0,
}

const initialState: TranscriptionState = {
  file: null,
  progress: initialProgress,
  result: null,
  error: null,
}

/**
 * Hook for managing audio file upload and transcription state.
 */
export function useTranscription(options: UseTranscriptionOptions = {}): UseTranscriptionReturn {
  const { language = 'en-US', onComplete, onError } = options
  const [state, setState] = useState<TranscriptionState>(initialState)
  const abortControllerRef = useRef<AbortController | null>(null)
  const { withAuth } = useAuthenticatedApi()

  const selectFile = useCallback((file: File) => {
    // Create FileInfo from File object
    const fileInfo: FileInfo = {
      file,
      name: file.name,
      size: file.size,
      type: file.type,
    }

    setState((prev) => ({
      ...prev,
      file: fileInfo,
      error: null,
      result: null,
      progress: initialProgress,
    }))
  }, [])

  const clearFile = useCallback(() => {
    setState((prev) => ({
      ...prev,
      file: null,
      error: null,
      result: null,
      progress: initialProgress,
    }))
  }, [])

  const startTranscription = useCallback(async () => {
    if (!state.file) {
      const errorMsg = 'No file selected'
      setState((prev) => ({ ...prev, error: errorMsg }))
      onError?.(errorMsg)
      return
    }

    // Create abort controller for cancellation
    abortControllerRef.current = new AbortController()

    try {
      // Update progress: uploading
      setState((prev) => ({
        ...prev,
        error: null,
        progress: {
          status: 'uploading',
          progress: 10,
          message: 'Uploading audio file...',
        },
      }))

      // Update progress: transcribing
      setState((prev) => ({
        ...prev,
        progress: {
          status: 'transcribing',
          progress: 30,
          message: 'Transcribing audio...',
        },
      }))

      // Call API with authentication
      const response = await withAuth(() => transcribeAudioApi(state.file!.file, language))

      // Check if cancelled
      if (abortControllerRef.current?.signal.aborted) {
        return
      }

      // Transform response to TranscriptionResult
      const result: TranscriptionResult = {
        text: response.text,
        duration: response.duration,
        processingTime: response.processingTime,
        language: response.language,
      }

      // Update state with result
      setState((prev) => ({
        ...prev,
        result,
        progress: {
          status: 'complete',
          progress: 100,
          message: 'Transcription complete!',
        },
      }))

      onComplete?.(result)
    } catch (error) {
      // Check if cancelled
      if (abortControllerRef.current?.signal.aborted) {
        return
      }

      const errorMsg = error instanceof Error ? error.message : 'Transcription failed'
      setState((prev) => ({
        ...prev,
        error: errorMsg,
        progress: {
          status: 'error',
          progress: 0,
          message: errorMsg,
        },
      }))
      onError?.(errorMsg)
    } finally {
      abortControllerRef.current = null
    }
  }, [state.file, language, onComplete, onError, withAuth])

  const cancelTranscription = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    setState((prev) => ({
      ...prev,
      progress: {
        status: 'idle',
        progress: 0,
        message: 'Transcription cancelled',
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
    selectFile,
    clearFile,
    startTranscription,
    cancelTranscription,
    reset,
    isTranscribing:
      state.progress.status === 'uploading' || state.progress.status === 'transcribing',
    hasFile: state.file !== null,
    hasResult: state.result !== null,
  }
}
