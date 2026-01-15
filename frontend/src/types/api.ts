// User types
export interface UserProfile {
  id: string
  email: string
  name: string
  preferredLanguage: string
  createdAt: string
  updatedAt: string
}

export interface UserPreferences {
  preferredLanguage?: string
  defaultMaxSpeakers?: number
  theme?: 'light' | 'dark' | 'system'
}

// Re-export SpeakerSegment from transcription to avoid duplication
export type { SpeakerSegment } from './transcription'

// Frontend-friendly transcription types (camelCase, seconds)
export interface TranscriptionResponse {
  text: string
  duration: number
  processingTime: number
  language: string
}

export interface DiarizedTranscriptionResponse {
  segments: import('./transcription').SpeakerSegment[]
  fullText: string
  duration: number
  processingTime: number
  speakerCount: number
}

export interface TranscriptionRecord {
  id: string
  userId: string
  text: string
  duration: number
  processingTime: number
  language: string
  fileName: string
  fileSize: number
  createdAt: string
  speakerCount?: number
  speakerIds?: string[]
  languageDetected?: string
}

export interface TranscriptionListResponse {
  transcriptions: TranscriptionRecord[]
  total: number
  page: number
  perPage: number
}

// Health types
export interface DependencyStatus {
  name: string
  healthy: boolean
  configured?: boolean
  message?: string
}

export interface HealthResponse {
  status: string
  timestamp: string
  version: string
}

export interface ReadyResponse {
  status: string
  dependencies: DependencyStatus[]
}

// Transcription content types (from Blob Storage)
export interface TranscriptionContentSegment {
  speaker_id: string
  start_time: number  // seconds
  end_time: number    // seconds
  text: string
}

export interface TranscriptionContent {
  transcript_id: string
  user_id: string
  filename?: string
  upload_date: string
  duration?: number
  speaker_segments: TranscriptionContentSegment[]
  full_text: string
  language: string
  processing_time_ms?: number
}

export interface TranscriptionContentResponse {
  transcript_id: string
  content: TranscriptionContent
  blob_url?: string
}

// Analysis types (from OpenAI)
export interface ActionItem {
  task: string
  assignee?: string
  deadline?: string
  priority: 'high' | 'medium' | 'low'
}

export interface AnalysisResult {
  summary: string
  keyPoints: string[]
  actionItems: ActionItem[]
  participants: string[]
  topics: string[]
  sentiment: 'positive' | 'neutral' | 'negative'
  confidence: number
}
