import { config } from '../config'
import type {
  HealthResponse,
  ReadyResponse,
  UserProfile,
  UserPreferences,
  TranscriptionResponse,
  DiarizedTranscriptionResponse,
  TranscriptionListResponse,
  TranscriptionRecord,
  TranscriptionContentResponse,
  AnalysisResult,
} from '../types'

class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

type HttpMethod = 'GET' | 'POST' | 'PATCH' | 'DELETE'

interface RequestOptions {
  method?: HttpMethod
  body?: FormData | Record<string, unknown>
  headers?: Record<string, string>
}

let accessToken: string | null = null

export const setAccessToken = (token: string | null): void => {
  accessToken = token
}

export const getAccessToken = (): string | null => {
  return accessToken
}

const request = async <T>(endpoint: string, options: RequestOptions = {}): Promise<T> => {
  const { method = 'GET', body, headers = {} } = options

  const requestHeaders: Record<string, string> = {
    ...headers,
  }

  if (accessToken) {
    requestHeaders['Authorization'] = `Bearer ${accessToken}`
  }

  let requestBody: BodyInit | undefined
  if (body instanceof FormData) {
    requestBody = body
  } else if (body) {
    requestHeaders['Content-Type'] = 'application/json'
    requestBody = JSON.stringify(body)
  }

  const response = await fetch(`${config.apiBaseUrl}${endpoint}`, {
    method,
    headers: requestHeaders,
    body: requestBody,
  })

  if (!response.ok) {
    const errorText = await response.text()
    throw new ApiError(response.status, errorText || response.statusText)
  }

  const contentType = response.headers.get('content-type')
  if (contentType?.includes('application/json')) {
    return response.json()
  }
  return response.text() as T
}

// Health endpoints
export const getHealth = (): Promise<HealthResponse> => {
  return request<HealthResponse>('/health')
}

export const getReady = (): Promise<ReadyResponse> => {
  return request<ReadyResponse>('/ready')
}

// Auth endpoints
export const getCurrentUser = (): Promise<UserProfile> => {
  return request<UserProfile>('/auth/me')
}

export const updateUserPreferences = (
  preferences: UserPreferences
): Promise<UserProfile> => {
  return request<UserProfile>('/auth/me/preferences', {
    method: 'PATCH',
    body: preferences as unknown as Record<string, unknown>,
  })
}

// Transcription endpoints

// Raw backend response types (snake_case, milliseconds)
interface TranscriptionResponseRaw {
  text: string
  duration_ms: number | null
  processing_time_ms: number | null
  language: string
  audio_format: string
  file_size_bytes: number
}

interface DiarizedTranscriptionResponseRaw {
  segments: Array<{
    speaker_id: string
    text: string
    start_time_ms: number
    end_time_ms: number
  }>
  full_text: string
  duration_ms: number | null
  processing_time_ms: number | null
  speaker_count: number
  language: string
  audio_format: string
  file_size_bytes: number
}

export const transcribeAudio = async (
  file: File,
  language?: string
): Promise<TranscriptionResponse> => {
  const formData = new FormData()
  formData.append('file', file)
  if (language) {
    formData.append('language', language)
  }
  const raw = await request<TranscriptionResponseRaw>('/transcribe', {
    method: 'POST',
    body: formData,
  })

  // Transform backend response to frontend format (convert ms to seconds)
  return {
    text: raw.text,
    language: raw.language,
    duration: raw.duration_ms ? raw.duration_ms / 1000 : 0,
    processingTime: raw.processing_time_ms ? raw.processing_time_ms / 1000 : 0,
  }
}

export const transcribeWithDiarization = async (
  file: File,
  maxSpeakers?: number,
  language?: string
): Promise<DiarizedTranscriptionResponse> => {
  const formData = new FormData()
  formData.append('file', file)

  // Build query parameters for diarization options
  const params = new URLSearchParams()
  if (maxSpeakers) {
    params.append('max_speakers', maxSpeakers.toString())
  }
  if (language) {
    params.append('language', language)
  }
  const queryString = params.toString()
  const url = `/transcribe/diarize${queryString ? `?${queryString}` : ''}`

  const raw = await request<DiarizedTranscriptionResponseRaw>(url, {
    method: 'POST',
    body: formData,
  })

  // Transform backend response to frontend format (convert ms to seconds)
  return {
    fullText: raw.full_text,
    duration: raw.duration_ms ? raw.duration_ms / 1000 : 0,
    processingTime: raw.processing_time_ms ? raw.processing_time_ms / 1000 : 0,
    speakerCount: raw.speaker_count,
    segments: raw.segments.map((seg) => ({
      speakerId: seg.speaker_id,
      text: seg.text,
      startTimeMs: seg.start_time_ms,
      endTimeMs: seg.end_time_ms,
    })),
  }
}

// Transcription history endpoints
export const getTranscriptions = (
  page = 1,
  perPage = 10
): Promise<TranscriptionListResponse> => {
  return request<TranscriptionListResponse>(
    `/transcriptions?page=${page}&per_page=${perPage}`
  )
}

export const getTranscription = (id: string): Promise<TranscriptionRecord> => {
  // URL-encode the ID since it may contain slashes (e.g., "user_id/filename_timestamp")
  return request<TranscriptionRecord>(`/transcriptions/${encodeURIComponent(id)}`)
}

export const deleteTranscription = (id: string): Promise<void> => {
  // URL-encode the ID since it may contain slashes
  return request<void>(`/transcriptions/${encodeURIComponent(id)}`, { method: 'DELETE' })
}

export const getTranscriptionContent = (id: string): Promise<TranscriptionContentResponse> => {
  // URL-encode the ID since it may contain slashes
  return request<TranscriptionContentResponse>(`/transcriptions/${encodeURIComponent(id)}/content`)
}

// Analysis endpoints
export const analyzeTranscription = (text: string, diarizedTranscript?: string): Promise<AnalysisResult> => {
  const body: Record<string, unknown> = { text }
  if (diarizedTranscript) {
    body.diarized_transcript = diarizedTranscript
  }
  return request<AnalysisResult>('/analyze', {
    method: 'POST',
    body,
  })
}

export { ApiError }
