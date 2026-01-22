import { useState, useCallback, useRef, useEffect } from 'react'
import {
  transcribeAudio as transcribeAudioApi,
  transcribeWithDiarization as transcribeDiarizeApi,
  submitBatchTranscription,
  getBatchTranscriptionStatus,
  getBatchTranscriptionResult,
} from '../services/api'
import { useAuthenticatedApi } from './useAuthenticatedApi'
import type {
  FileInfo,
  TranscriptionProgress,
  TranscriptionResult,
  TranscriptionState,
} from '../types/transcription'

// Re-export constants for use in components
export { SUPPORTED_AUDIO_FORMATS, SUPPORTED_EXTENSIONS, MAX_FILE_SIZE_BYTES } from '../types/transcription'

// Threshold for using batch transcription (100MB)
const BATCH_TRANSCRIPTION_THRESHOLD_BYTES = 100 * 1024 * 1024

// Polling interval for batch transcription status (5 seconds)
const BATCH_POLLING_INTERVAL_MS = 5000

export interface UseTranscriptionOptions {
  language?: string
  enableDiarization?: boolean
  maxSpeakers?: number
  useBatchTranscription?: boolean // Force batch mode regardless of file size
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
  const { language = 'en-US', enableDiarization = true, maxSpeakers = 5, useBatchTranscription = false, onComplete, onError } = options
  const [state, setState] = useState<TranscriptionState>(initialState)
  const abortControllerRef = useRef<AbortController | null>(null)
  const pollingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const { withAuth } = useAuthenticatedApi()

  // Cleanup polling interval on unmount
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current)
        pollingIntervalRef.current = null
      }
    }
  }, [])

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

  // Helper function to poll for batch transcription status
  const pollBatchStatus = useCallback(async (jobId: string, folderPath?: string): Promise<TranscriptionResult> => {
    return new Promise((resolve, reject) => {
      let pollCount = 0
      const maxPolls = 720 // Max ~1 hour with 5s interval

      pollingIntervalRef.current = setInterval(async () => {
        try {
          pollCount++

          // Check if cancelled
          if (abortControllerRef.current?.signal.aborted) {
            if (pollingIntervalRef.current) {
              clearInterval(pollingIntervalRef.current)
              pollingIntervalRef.current = null
            }
            reject(new Error('Transcription cancelled'))
            return
          }

          const status = await withAuth(() => getBatchTranscriptionStatus(jobId))

          // Update progress based on status
          const progressPercent = Math.min(30 + Math.floor((pollCount / maxPolls) * 60), 90)
          setState((prev) => ({
            ...prev,
            progress: {
              status: 'polling',
              progress: progressPercent,
              message: `Processing transcription... (${status.status})`,
              jobId,
            },
          }))

          if (status.status === 'Succeeded') {
            if (pollingIntervalRef.current) {
              clearInterval(pollingIntervalRef.current)
              pollingIntervalRef.current = null
            }

            // Get the result and save metadata to blob storage for My Recordings
            const batchResult = await withAuth(() => getBatchTranscriptionResult(jobId, folderPath))

            const result: TranscriptionResult = {
              text: batchResult.fullText,
              duration: batchResult.durationMs / 1000,
              processingTime: undefined,
              language: batchResult.language,
              hasDiarization: batchResult.speakerCount > 1,
              speakerCount: batchResult.speakerCount,
              segments: batchResult.segments.map((seg) => ({
                speakerId: seg.speakerId,
                text: seg.text,
                startTimeMs: seg.startTimeMs,
                endTimeMs: seg.endTimeMs,
              })),
              folderPath: folderPath,
            }

            resolve(result)
          } else if (status.status === 'Failed') {
            if (pollingIntervalRef.current) {
              clearInterval(pollingIntervalRef.current)
              pollingIntervalRef.current = null
            }
            reject(new Error(status.errorMessage || 'Batch transcription failed'))
          } else if (pollCount >= maxPolls) {
            if (pollingIntervalRef.current) {
              clearInterval(pollingIntervalRef.current)
              pollingIntervalRef.current = null
            }
            reject(new Error('Transcription timed out after 1 hour'))
          }
        } catch (error) {
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current)
            pollingIntervalRef.current = null
          }
          reject(error)
        }
      }, BATCH_POLLING_INTERVAL_MS)
    })
  }, [withAuth])

  const startTranscription = useCallback(async () => {
    if (!state.file) {
      const errorMsg = 'No file selected'
      setState((prev) => ({ ...prev, error: errorMsg }))
      onError?.(errorMsg)
      return
    }

    // Create abort controller for cancellation
    abortControllerRef.current = new AbortController()

    // Determine if we should use batch transcription
    const shouldUseBatch = useBatchTranscription || state.file.size >= BATCH_TRANSCRIPTION_THRESHOLD_BYTES

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

      let result: TranscriptionResult

      if (shouldUseBatch) {
        // Use batch transcription for large files
        setState((prev) => ({
          ...prev,
          progress: {
            status: 'uploading',
            progress: 15,
            message: 'Uploading large file for batch processing...',
          },
        }))

        // Submit batch job
        const jobResponse = await withAuth(() =>
          submitBatchTranscription(state.file!.file, {
            language,
            enableDiarization,
            maxSpeakers,
          })
        )

        // Check if cancelled
        if (abortControllerRef.current?.signal.aborted) {
          return
        }

        setState((prev) => ({
          ...prev,
          progress: {
            status: 'polling',
            progress: 25,
            message: 'Batch job submitted, waiting for processing...',
            jobId: jobResponse.jobId,
          },
        }))

        // Poll for completion
        result = await pollBatchStatus(jobResponse.jobId, jobResponse.folderPath)
      } else {
        // Use synchronous transcription for smaller files
        setState((prev) => ({
          ...prev,
          progress: {
            status: 'transcribing',
            progress: 30,
            message: enableDiarization ? 'Transcribing with speaker identification...' : 'Transcribing audio...',
          },
        }))

        // Call API with authentication - use diarization endpoint if enabled
        if (enableDiarization) {
          const response = await withAuth(() => transcribeDiarizeApi(state.file!.file, maxSpeakers, language))
          // Transform diarized response to TranscriptionResult
          result = {
            text: response.fullText,
            duration: response.duration,
            processingTime: response.processingTime,
            language: language,
            hasDiarization: true,
            speakerCount: response.speakerCount,
            segments: response.segments.map((seg) => ({
              speakerId: seg.speakerId,
              text: seg.text,
              startTimeMs: seg.startTimeMs,
              endTimeMs: seg.endTimeMs,
            })),
            folderPath: response.folderPath,
            cached: response.cached,
            originalUploadDate: response.originalUploadDate,
            originalFilename: response.originalFilename,
          }
        } else {
          const response = await withAuth(() => transcribeAudioApi(state.file!.file, language))
          // Transform response to TranscriptionResult
          result = {
            text: response.text,
            duration: response.duration,
            processingTime: response.processingTime,
            language: response.language,
            hasDiarization: false,
            folderPath: response.folderPath,
            cached: response.cached,
            originalUploadDate: response.originalUploadDate,
            originalFilename: response.originalFilename,
          }
        }
      }

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
  }, [state.file, language, enableDiarization, maxSpeakers, useBatchTranscription, onComplete, onError, withAuth, pollBatchStatus])

  const cancelTranscription = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
      abortControllerRef.current = null
    }
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current)
      pollingIntervalRef.current = null
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
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current)
      pollingIntervalRef.current = null
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
      state.progress.status === 'uploading' || state.progress.status === 'transcribing' || state.progress.status === 'polling',
    hasFile: state.file !== null,
    hasResult: state.result !== null,
  }
}
