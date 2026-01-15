import { useCallback, useRef, useState, type DragEvent, type ChangeEvent } from 'react'
import type { AudioUploadProps } from '../types/transcription'
import {
  SUPPORTED_EXTENSIONS,
  SUPPORTED_AUDIO_FORMATS,
  MAX_FILE_SIZE_BYTES,
  MAX_FILE_SIZE_MB,
} from '../types/transcription'
import { useTranscription } from '../hooks/useTranscription'

// Default max speakers for diarization
const DEFAULT_MAX_SPEAKERS = 5
const MIN_SPEAKERS = 1
const MAX_SPEAKERS = 10

/**
 * AudioUpload component for uploading and transcribing audio files.
 * Supports drag-and-drop and file picker.
 */
export function AudioUpload({
  onTranscriptionComplete,
  onError,
  language = 'en-US',
  disabled = false,
}: AudioUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false)
  const [audioDuration, setAudioDuration] = useState<number | null>(null)
  const [durationLimit, setDurationLimit] = useState<number | null>(null)
  const [maxSpeakers, setMaxSpeakers] = useState(DEFAULT_MAX_SPEAKERS)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const audioRef = useRef<HTMLAudioElement>(null)

  // Diarization is always enabled (per US-059 requirements)
  const enableDiarization = true

  const {
    state,
    selectFile,
    clearFile,
    startTranscription,
    isTranscribing,
    hasFile,
    hasResult,
  } = useTranscription({
    language,
    enableDiarization,
    maxSpeakers,
    onComplete: onTranscriptionComplete,
    onError,
  })

  // Validate file type and size
  const validateFile = useCallback((file: File): string | null => {
    // Check extension
    const ext = file.name.split('.').pop()?.toLowerCase()
    if (!ext || !SUPPORTED_EXTENSIONS.includes(ext)) {
      return `Unsupported file format. Supported: ${SUPPORTED_EXTENSIONS.join(', ')}`
    }

    // Check size
    if (file.size > MAX_FILE_SIZE_BYTES) {
      return `File too large. Maximum size is ${MAX_FILE_SIZE_MB} MB`
    }

    return null
  }, [])

  // Handle file selection
  const handleFileSelect = useCallback(
    (file: File) => {
      const error = validateFile(file)
      if (error) {
        onError?.(error)
        return
      }

      selectFile(file)
      setAudioDuration(null)
      setDurationLimit(null)

      // Create object URL for audio preview and duration detection
      const audioUrl = URL.createObjectURL(file)
      if (audioRef.current) {
        audioRef.current.src = audioUrl
        audioRef.current.onloadedmetadata = () => {
          if (audioRef.current) {
            const duration = audioRef.current.duration
            setAudioDuration(duration)
            // Default to min of 10 minutes or full duration
            setDurationLimit(Math.min(duration, 10 * 60))
          }
        }
      }
    },
    [validateFile, selectFile, onError]
  )

  // Drag and drop handlers
  const handleDragOver = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      e.stopPropagation()
      if (!disabled && !isTranscribing) {
        setIsDragOver(true)
      }
    },
    [disabled, isTranscribing]
  )

  const handleDragLeave = useCallback((e: DragEvent<HTMLDivElement>) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragOver(false)
  }, [])

  const handleDrop = useCallback(
    (e: DragEvent<HTMLDivElement>) => {
      e.preventDefault()
      e.stopPropagation()
      setIsDragOver(false)

      if (disabled || isTranscribing) return

      const files = e.dataTransfer.files
      if (files.length > 0) {
        handleFileSelect(files[0])
      }
    },
    [disabled, isTranscribing, handleFileSelect]
  )

  // File input change handler
  const handleInputChange = useCallback(
    (e: ChangeEvent<HTMLInputElement>) => {
      const files = e.target.files
      if (files && files.length > 0) {
        handleFileSelect(files[0])
      }
    },
    [handleFileSelect]
  )

  // Click to open file picker
  const handleClick = useCallback(() => {
    if (!disabled && !isTranscribing && fileInputRef.current) {
      fileInputRef.current.click()
    }
  }, [disabled, isTranscribing])

  // Clear file and reset
  const handleClear = useCallback(() => {
    clearFile()
    setAudioDuration(null)
    setDurationLimit(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }, [clearFile])

  // Format file size
  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`
  }

  // Format duration
  const formatDuration = (seconds: number): string => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className="space-y-4">
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        accept={[
          ...SUPPORTED_EXTENSIONS.map((ext) => `.${ext}`),
          ...SUPPORTED_AUDIO_FORMATS,
        ].join(',')}
        onChange={handleInputChange}
        className="hidden"
        disabled={disabled || isTranscribing}
        data-testid="file-input"
      />

      {/* Hidden audio element for duration detection */}
      <audio ref={audioRef} className="hidden" preload="metadata" />

      {/* Upload section */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Upload Audio File</h3>

        {/* Drop zone */}
        {!hasFile && (
          <div
            onClick={handleClick}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
              isDragOver
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
            } ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
            data-testid="drop-zone"
            role="button"
            tabIndex={0}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                handleClick()
              }
            }}
          >
            <svg
              className="w-12 h-12 mx-auto text-gray-400 mb-4"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12"
              />
            </svg>
            <p className="text-gray-600 mb-2">
              <span className="font-medium text-blue-600">Click to upload</span> or drag and drop
            </p>
            <p className="text-sm text-gray-500">
              Supported formats: {SUPPORTED_EXTENSIONS.join(', ').toUpperCase()}
            </p>
            <p className="text-sm text-gray-500">Max file size: {MAX_FILE_SIZE_MB} MB</p>
          </div>
        )}

        {/* File info */}
        {hasFile && state.file && (
          <div className="space-y-4">
            {/* File details */}
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="text-sm font-medium text-gray-700 mb-3">File Information</h4>
              <div className="grid grid-cols-3 gap-4">
                <div className="text-center">
                  <p className="text-xs text-gray-500">File Name</p>
                  <p className="text-sm font-medium text-gray-800 truncate" title={state.file.name}>
                    {state.file.name}
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-xs text-gray-500">File Size</p>
                  <p className="text-sm font-medium text-gray-800">
                    {formatFileSize(state.file.size)}
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-xs text-gray-500">Format</p>
                  <p className="text-sm font-medium text-gray-800">
                    {state.file.name.split('.').pop()?.toUpperCase() || 'Unknown'}
                  </p>
                </div>
              </div>
            </div>

            {/* Audio preview */}
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">Audio Preview</h4>
              <audio
                controls
                src={state.file.file ? URL.createObjectURL(state.file.file) : undefined}
                className="w-full"
                data-testid="audio-preview"
              />
            </div>

            {/* Duration settings */}
            {audioDuration !== null && audioDuration > 60 && (
              <div className="bg-gray-50 rounded-lg p-4">
                <h4 className="text-sm font-medium text-gray-700 mb-3">Duration Settings</h4>
                <div className="flex items-center gap-4">
                  <div className="text-center">
                    <p className="text-xs text-gray-500">Total Duration</p>
                    <p className="text-sm font-medium text-gray-800">
                      {formatDuration(audioDuration)}
                    </p>
                  </div>
                  <div className="flex-1">
                    <label htmlFor="duration-slider" className="text-xs text-gray-500 block mb-1">
                      Duration to transcribe: {durationLimit ? formatDuration(durationLimit) : '--'}
                    </label>
                    <input
                      id="duration-slider"
                      type="range"
                      min={60}
                      max={audioDuration}
                      step={30}
                      value={durationLimit ?? audioDuration}
                      onChange={(e) => setDurationLimit(Number(e.target.value))}
                      className="w-full"
                      disabled={isTranscribing}
                      data-testid="duration-slider"
                    />
                  </div>
                </div>
                {durationLimit !== null && durationLimit < audioDuration && (
                  <p className="text-xs text-blue-600 mt-2">
                    Will transcribe the first {formatDuration(durationLimit)} of{' '}
                    {formatDuration(audioDuration)} total
                  </p>
                )}
              </div>
            )}

            {/* Speaker identification settings - always enabled */}
            <div className="bg-purple-50 rounded-lg p-4" data-testid="diarization-settings">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <h4 className="text-sm font-medium text-gray-700">Speaker Identification</h4>
                  <span className="px-2 py-0.5 text-xs bg-purple-200 text-purple-800 rounded-full">Enabled</span>
                </div>
              </div>
              <p className="text-xs text-gray-500 mb-3">
                Automatically identifies who said what in multi-speaker recordings
              </p>
              <div className="flex items-center gap-4">
                <label className="text-sm text-gray-600">Max speakers:</label>
                <input
                  type="number"
                  min={MIN_SPEAKERS}
                  max={MAX_SPEAKERS}
                  value={maxSpeakers}
                  onChange={(e) => setMaxSpeakers(Math.min(MAX_SPEAKERS, Math.max(MIN_SPEAKERS, Number(e.target.value))))}
                  className="w-16 px-2 py-1 border border-gray-300 rounded text-center"
                  disabled={isTranscribing}
                  data-testid="max-speakers-input"
                />
                <span className="text-xs text-gray-400">(1-{MAX_SPEAKERS})</span>
              </div>
            </div>

            {/* Progress bar */}
            {isTranscribing && (
              <div className="space-y-2" data-testid="progress-container">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">{state.progress.message}</span>
                  <span className="text-gray-800 font-medium">{state.progress.progress}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2.5">
                  <div
                    className="bg-blue-600 h-2.5 rounded-full transition-all duration-300"
                    style={{ width: `${state.progress.progress}%` }}
                    data-testid="progress-bar"
                  />
                </div>
              </div>
            )}

            {/* Error message */}
            {state.error && (
              <div
                className="bg-red-50 border border-red-200 rounded-lg p-3 text-red-700 text-sm flex items-center gap-2"
                data-testid="error-message"
              >
                <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                {state.error}
              </div>
            )}

            {/* Transcription controls */}
            <div className="flex gap-3">
              <button
                onClick={startTranscription}
                disabled={isTranscribing || disabled}
                className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
                  isTranscribing || disabled
                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                    : 'bg-blue-600 text-white hover:bg-blue-700'
                }`}
                data-testid="transcribe-button"
              >
                {isTranscribing ? (
                  <span className="flex items-center justify-center gap-2">
                    <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                    Transcribing...
                  </span>
                ) : (
                  'Start Transcription'
                )}
              </button>
              <button
                onClick={handleClear}
                disabled={isTranscribing}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  isTranscribing
                    ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                    : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                }`}
                data-testid="clear-button"
              >
                Clear
              </button>
            </div>

            {/* Language info */}
            <div className="text-sm text-gray-500">
              <span className="inline-block bg-blue-100 text-blue-700 px-2 py-1 rounded">
                Language: {language}
              </span>
            </div>
          </div>
        )}

        {/* Success result indicator */}
        {hasResult && state.result && (
          <div
            className="mt-4 bg-green-50 border border-green-200 rounded-lg p-3"
            data-testid="success-message"
          >
            <div className="flex items-center text-green-700 gap-2">
              <svg className="w-5 h-5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
              </svg>
              <span className="font-medium">Transcription completed successfully!</span>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
