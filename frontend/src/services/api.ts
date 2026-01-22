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

type HttpMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE'

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

  // Handle 204 No Content - return undefined without attempting to parse body
  if (response.status === 204) {
    return undefined as T
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
  cached?: boolean
  folder_path?: string
  original_upload_date?: string
  original_filename?: string
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
  folder_path?: string
  cached?: boolean
  original_upload_date?: string
  original_filename?: string
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
    cached: raw.cached,
    folderPath: raw.folder_path,
    originalUploadDate: raw.original_upload_date,
    originalFilename: raw.original_filename,
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
    folderPath: raw.folder_path,
    cached: raw.cached,
    originalUploadDate: raw.original_upload_date,
    originalFilename: raw.original_filename,
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
export const analyzeTranscription = (text: string, diarizedTranscript?: string, folderPath?: string): Promise<AnalysisResult> => {
  const body: Record<string, unknown> = { text }
  if (diarizedTranscript) {
    body.diarized_transcript = diarizedTranscript
  }
  if (folderPath) {
    body.folder_path = folderPath
  }
  return request<AnalysisResult>('/analyze', {
    method: 'POST',
    body,
  })
}

export const saveAnalysis = (transcriptionId: string, analysis: AnalysisResult): Promise<{ message: string; blob_url: string }> => {
  return request<{ message: string; blob_url: string }>(`/transcriptions/${encodeURIComponent(transcriptionId)}/analysis`, {
    method: 'POST',
    body: analysis as unknown as Record<string, unknown>,
  })
}

export const getAnalysis = (transcriptionId: string): Promise<AnalysisResult> => {
  return request<AnalysisResult>(`/transcriptions/${encodeURIComponent(transcriptionId)}/analysis`)
}

// Speaker name update types
export interface SpeakerNameUpdate {
  speaker_names: Record<string, { name: string; confidence: string; ai_identified: boolean }>
}

export interface SpeakerNameResponse {
  message: string
  speaker_names: Record<string, { name: string; confidence: string; ai_identified: boolean }>
}

export const updateSpeakerNames = (
  transcriptionId: string,
  speakerNames: Record<string, { name: string; confidence: string; ai_identified: boolean }>
): Promise<SpeakerNameResponse> => {
  return request<SpeakerNameResponse>(`/transcriptions/${encodeURIComponent(transcriptionId)}/speakers`, {
    method: 'PUT',
    body: { speaker_names: speakerNames } as Record<string, unknown>,
  })
}

// Batch transcription types and functions for large files
import type {
  BatchTranscriptionJobResponse,
  BatchTranscriptionStatusResponse,
  BatchTranscriptionResultResponse,
} from '../types'

// Raw backend response types for batch transcription (snake_case)
interface BatchTranscriptionJobResponseRaw {
  job_id: string
  status: 'NotStarted' | 'Running' | 'Succeeded' | 'Failed'
  display_name: string
  created_at: string
  blob_url?: string
  folder_path?: string
}

interface BatchTranscriptionStatusResponseRaw {
  job_id: string
  status: 'NotStarted' | 'Running' | 'Succeeded' | 'Failed'
  display_name: string
  created_at: string
  completed_at?: string
  error_message?: string
}

interface BatchTranscriptionResultResponseRaw {
  job_id: string
  segments: Array<{
    speaker_id: string
    text: string
    start_time_ms: number
    end_time_ms: number
  }>
  full_text: string
  language: string
  duration_ms: number
  speaker_count: number
}

export interface BatchTranscriptionOptions {
  language?: string
  enableDiarization?: boolean
  maxSpeakers?: number
}

/**
 * Submit a file for batch transcription (async job pattern).
 * Use for large files that might timeout with synchronous transcription.
 */
export const submitBatchTranscription = async (
  file: File,
  options: BatchTranscriptionOptions = {}
): Promise<BatchTranscriptionJobResponse> => {
  const formData = new FormData()
  formData.append('file', file)

  // Build query parameters
  const params = new URLSearchParams()
  if (options.language) {
    params.append('language', options.language)
  }
  if (options.enableDiarization !== undefined) {
    params.append('enable_diarization', options.enableDiarization.toString())
  }
  if (options.maxSpeakers) {
    params.append('max_speakers', options.maxSpeakers.toString())
  }
  const queryString = params.toString()
  const url = `/transcribe/batch${queryString ? `?${queryString}` : ''}`

  const raw = await request<BatchTranscriptionJobResponseRaw>(url, {
    method: 'POST',
    body: formData,
  })

  return {
    jobId: raw.job_id,
    status: raw.status,
    displayName: raw.display_name,
    createdAt: raw.created_at,
    blobUrl: raw.blob_url,
    folderPath: raw.folder_path,
  }
}

/**
 * Poll the status of a batch transcription job.
 */
export const getBatchTranscriptionStatus = async (
  jobId: string
): Promise<BatchTranscriptionStatusResponse> => {
  const raw = await request<BatchTranscriptionStatusResponseRaw>(
    `/transcribe/batch/${encodeURIComponent(jobId)}/status`
  )

  return {
    jobId: raw.job_id,
    status: raw.status,
    displayName: raw.display_name,
    createdAt: raw.created_at,
    completedAt: raw.completed_at,
    errorMessage: raw.error_message,
  }
}

/**
 * Get the result of a completed batch transcription job.
 */
export const getBatchTranscriptionResult = async (
  jobId: string
): Promise<BatchTranscriptionResultResponse> => {
  const raw = await request<BatchTranscriptionResultResponseRaw>(
    `/transcribe/batch/${encodeURIComponent(jobId)}/result`
  )

  return {
    jobId: raw.job_id,
    segments: raw.segments.map((seg) => ({
      speakerId: seg.speaker_id,
      text: seg.text,
      startTimeMs: seg.start_time_ms,
      endTimeMs: seg.end_time_ms,
    })),
    fullText: raw.full_text,
    language: raw.language,
    durationMs: raw.duration_ms,
    speakerCount: raw.speaker_count,
  }
}

export { ApiError }
