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
export const transcribeAudio = (
  file: File,
  language?: string
): Promise<TranscriptionResponse> => {
  const formData = new FormData()
  formData.append('file', file)
  if (language) {
    formData.append('language', language)
  }
  return request<TranscriptionResponse>('/transcribe', {
    method: 'POST',
    body: formData,
  })
}

export const transcribeWithDiarization = (
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

  return request<DiarizedTranscriptionResponse>(url, {
    method: 'POST',
    body: formData,
  })
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
  return request<TranscriptionRecord>(`/transcriptions/${id}`)
}

export const deleteTranscription = (id: string): Promise<void> => {
  return request<void>(`/transcriptions/${id}`, { method: 'DELETE' })
}

// Analysis endpoints
export const analyzeTranscription = (text: string): Promise<AnalysisResult> => {
  return request<AnalysisResult>('/analyze', {
    method: 'POST',
    body: { text },
  })
}

export { ApiError }
