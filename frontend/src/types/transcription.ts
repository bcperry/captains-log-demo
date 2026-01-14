// Transcription component types

export interface FileInfo {
  file: File
  name: string
  size: number
  type: string
  duration?: number // Duration in seconds
}

export interface TranscriptionProgress {
  status: 'idle' | 'uploading' | 'transcribing' | 'complete' | 'error'
  progress: number // 0-100
  currentChunk?: number
  totalChunks?: number
  message?: string
}

export interface SpeakerSegment {
  speakerId: string
  text: string
  startTimeMs: number
  endTimeMs: number
}

export interface TranscriptionResult {
  text: string
  duration?: number
  processingTime?: number
  language: string
  audioFormat?: string
  fileSizeBytes?: number
  // Diarization fields (optional)
  segments?: SpeakerSegment[]
  speakerCount?: number
  hasDiarization?: boolean
}

export interface TranscriptionState {
  file: FileInfo | null
  progress: TranscriptionProgress
  result: TranscriptionResult | null
  error: string | null
}

export interface AudioUploadProps {
  onTranscriptionComplete?: (result: TranscriptionResult) => void
  onError?: (error: string) => void
  language?: string
  maxDurationMinutes?: number
  disabled?: boolean
}

// Supported audio formats
export const SUPPORTED_AUDIO_FORMATS = [
  'audio/wav',
  'audio/wave',
  'audio/x-wav',
  'audio/mp3',
  'audio/mpeg',
  'audio/mp4',
  'audio/x-m4a',
  'audio/ogg',
  'audio/flac',
  'video/mp4', // Some mp4 files report as video
]

export const SUPPORTED_EXTENSIONS = ['wav', 'mp3', 'm4a', 'ogg', 'flac', 'mp4']

export const MAX_FILE_SIZE_MB = 500
export const MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
