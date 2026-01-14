import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import {
  setAccessToken,
  getAccessToken,
  getHealth,
  getReady,
  getCurrentUser,
  updateUserPreferences,
  transcribeAudio,
  transcribeWithDiarization,
  getTranscriptions,
  getTranscription,
  deleteTranscription,
  analyzeTranscription,
  ApiError,
} from '../services/api'

// Mock fetch globally
const mockFetch = vi.fn()
global.fetch = mockFetch

// Mock config
vi.mock('../config', () => ({
  config: {
    apiBaseUrl: 'http://localhost:8001',
  },
}))

describe('API service', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    setAccessToken(null)
  })

  afterEach(() => {
    setAccessToken(null)
  })

  describe('access token management', () => {
    it('setAccessToken stores token', () => {
      setAccessToken('test-token')
      expect(getAccessToken()).toBe('test-token')
    })

    it('setAccessToken can clear token', () => {
      setAccessToken('test-token')
      setAccessToken(null)
      expect(getAccessToken()).toBeNull()
    })

    it('includes Authorization header when token is set', async () => {
      setAccessToken('test-token')
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: () => Promise.resolve({ status: 'ok' }),
      })

      await getHealth()

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8001/health',
        expect.objectContaining({
          headers: expect.objectContaining({
            Authorization: 'Bearer test-token',
          }),
        })
      )
    })

    it('omits Authorization header when no token', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: () => Promise.resolve({ status: 'ok' }),
      })

      await getHealth()

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8001/health',
        expect.objectContaining({
          headers: expect.not.objectContaining({
            Authorization: expect.anything(),
          }),
        })
      )
    })
  })

  describe('error handling', () => {
    it('throws ApiError on non-ok response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        statusText: 'Unauthorized',
        text: () => Promise.resolve('Authentication required'),
      })

      await expect(getHealth()).rejects.toThrow(ApiError)
    })

    it('ApiError message from response text', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        statusText: 'Unauthorized',
        text: () => Promise.resolve('Authentication required'),
      })

      await expect(getHealth()).rejects.toThrow('Authentication required')
    })

    it('ApiError includes status code', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found',
        text: () => Promise.resolve('Resource not found'),
      })

      try {
        await getHealth()
        expect.fail('Should have thrown')
      } catch (error) {
        expect(error).toBeInstanceOf(ApiError)
        expect((error as ApiError).status).toBe(404)
      }
    })
  })

  describe('getHealth', () => {
    it('calls /health endpoint', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: () => Promise.resolve({ status: 'healthy', timestamp: '2024-01-01', version: '1.0.0' }),
      })

      const result = await getHealth()

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8001/health',
        expect.objectContaining({ method: 'GET' })
      )
      expect(result.status).toBe('healthy')
    })
  })

  describe('getReady', () => {
    it('calls /ready endpoint', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: () => Promise.resolve({ status: 'ready', dependencies: [] }),
      })

      const result = await getReady()

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8001/ready',
        expect.anything()
      )
      expect(result.status).toBe('ready')
    })
  })

  describe('getCurrentUser', () => {
    it('calls /auth/me endpoint', async () => {
      setAccessToken('token')
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: () => Promise.resolve({
          id: 'user-1',
          email: 'test@example.com',
          name: 'Test User',
          preferredLanguage: 'en-US',
          createdAt: '2024-01-01',
          updatedAt: '2024-01-01',
        }),
      })

      const result = await getCurrentUser()

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8001/auth/me',
        expect.anything()
      )
      expect(result.email).toBe('test@example.com')
    })
  })

  describe('updateUserPreferences', () => {
    it('calls PATCH /auth/me/preferences', async () => {
      setAccessToken('token')
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: () => Promise.resolve({
          id: 'user-1',
          email: 'test@example.com',
          name: 'Test User',
          preferredLanguage: 'es-ES',
          createdAt: '2024-01-01',
          updatedAt: '2024-01-02',
        }),
      })

      const result = await updateUserPreferences({ preferredLanguage: 'es-ES' })

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8001/auth/me/preferences',
        expect.objectContaining({
          method: 'PATCH',
          headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
        })
      )
      expect(result.preferredLanguage).toBe('es-ES')
    })
  })

  describe('transcribeAudio', () => {
    it('calls POST /transcribe with FormData', async () => {
      setAccessToken('token')
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: () => Promise.resolve({
          text: 'Transcribed text',
          duration_ms: 60000,
          processing_time_ms: 5000,
          language: 'en-US',
          audio_format: 'mp3',
          file_size_bytes: 1024,
        }),
      })

      const file = new File(['audio content'], 'test.mp3', { type: 'audio/mpeg' })
      const result = await transcribeAudio(file, 'en-US')

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8001/transcribe',
        expect.objectContaining({ method: 'POST' })
      )
      // Verify FormData was sent (body should be FormData instance)
      const callArgs = mockFetch.mock.calls[0]
      expect(callArgs[1].body).toBeInstanceOf(FormData)
      expect(result.text).toBe('Transcribed text')
      expect(result.duration).toBe(60)
      expect(result.processingTime).toBe(5)
    })

    it('includes language in FormData when provided', async () => {
      setAccessToken('token')
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: () => Promise.resolve({
          text: 'Texto transcrito',
          duration_ms: 60000,
          processing_time_ms: 5000,
          language: 'es-ES',
          audio_format: 'mp3',
          file_size_bytes: 1024,
        }),
      })

      const file = new File(['audio content'], 'test.mp3', { type: 'audio/mpeg' })
      await transcribeAudio(file, 'es-ES')

      const callArgs = mockFetch.mock.calls[0]
      const formData = callArgs[1].body as FormData
      expect(formData.get('language')).toBe('es-ES')
    })
  })

  describe('transcribeWithDiarization', () => {
    it('calls POST /transcribe/diarize with FormData', async () => {
      setAccessToken('token')
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: () => Promise.resolve({
          segments: [],
          full_text: 'Full text',
          duration_ms: 60000,
          processing_time_ms: 5000,
          speaker_count: 2,
          language: 'en-US',
          audio_format: 'mp3',
          file_size_bytes: 1024,
        }),
      })

      const file = new File(['audio content'], 'test.mp3', { type: 'audio/mpeg' })
      const result = await transcribeWithDiarization(file, 3, 'en-US')

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8001/transcribe/diarize?max_speakers=3&language=en-US',
        expect.objectContaining({ method: 'POST' })
      )
      expect(result.speakerCount).toBe(2)
      expect(result.duration).toBe(60)
      expect(result.processingTime).toBe(5)
    })

    it('includes max_speakers in query params', async () => {
      setAccessToken('token')
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: () => Promise.resolve({
          segments: [],
          full_text: '',
          duration_ms: 0,
          processing_time_ms: 0,
          speaker_count: 0,
          language: 'en-US',
          audio_format: 'mp3',
          file_size_bytes: 0,
        }),
      })

      const file = new File(['audio'], 'test.mp3')
      await transcribeWithDiarization(file, 5, 'en-US')

      const callArgs = mockFetch.mock.calls[0]
      const url = callArgs[0] as string
      expect(url).toContain('max_speakers=5')
      expect(url).toContain('language=en-US')
    })
  })

  describe('getTranscriptions', () => {
    it('calls GET /transcriptions with pagination', async () => {
      setAccessToken('token')
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: () => Promise.resolve({
          transcriptions: [],
          total: 0,
          page: 1,
          perPage: 10,
        }),
      })

      await getTranscriptions(2, 20)

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8001/transcriptions?page=2&per_page=20',
        expect.anything()
      )
    })

    it('uses default pagination values', async () => {
      setAccessToken('token')
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: () => Promise.resolve({
          transcriptions: [],
          total: 0,
          page: 1,
          perPage: 10,
        }),
      })

      await getTranscriptions()

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8001/transcriptions?page=1&per_page=10',
        expect.anything()
      )
    })
  })

  describe('getTranscription', () => {
    it('calls GET /transcriptions/:id', async () => {
      setAccessToken('token')
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: () => Promise.resolve({
          id: 'trans-123',
          text: 'Transcription text',
        }),
      })

      const result = await getTranscription('trans-123')

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8001/transcriptions/trans-123',
        expect.anything()
      )
      expect(result.id).toBe('trans-123')
    })
  })

  describe('deleteTranscription', () => {
    it('calls DELETE /transcriptions/:id', async () => {
      setAccessToken('token')
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: new Headers({ 'content-type': 'text/plain' }),
        text: () => Promise.resolve(''),
      })

      await deleteTranscription('trans-123')

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8001/transcriptions/trans-123',
        expect.objectContaining({ method: 'DELETE' })
      )
    })
  })

  describe('analyzeTranscription', () => {
    it('calls POST /analyze with text', async () => {
      setAccessToken('token')
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: () => Promise.resolve({
          summary: 'Test summary',
          keyPoints: ['Point 1'],
          actionItems: [],
          participants: [],
          topics: [],
          sentiment: 'positive',
          confidence: 0.9,
        }),
      })

      const result = await analyzeTranscription('Text to analyze')

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8001/analyze',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({ 'Content-Type': 'application/json' }),
        })
      )
      expect(result.summary).toBe('Test summary')
    })
  })

  describe('response handling', () => {
    it('parses JSON response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: new Headers({ 'content-type': 'application/json' }),
        json: () => Promise.resolve({ data: 'test' }),
      })

      const result = await getHealth()
      expect(result).toEqual({ data: 'test' })
    })

    it('returns text for non-JSON response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        headers: new Headers({ 'content-type': 'text/plain' }),
        text: () => Promise.resolve('plain text response'),
      })

      const result = await getHealth()
      expect(result).toBe('plain text response')
    })
  })
})

describe('ApiError class', () => {
  it('has name property set to ApiError', () => {
    const error = new ApiError(500, 'Server error')
    expect(error.name).toBe('ApiError')
  })

  it('has status property', () => {
    const error = new ApiError(404, 'Not found')
    expect(error.status).toBe(404)
  })

  it('has message property', () => {
    const error = new ApiError(400, 'Bad request')
    expect(error.message).toBe('Bad request')
  })

  it('is instance of Error', () => {
    const error = new ApiError(500, 'Error')
    expect(error).toBeInstanceOf(Error)
  })
})
