import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { TranscriptionResults } from '../components/TranscriptionResults'
import * as api from '../services/api'
import type { TranscriptionResult } from '../types/transcription'
import type { AnalysisResult } from '../types/api'

// Mock the API module
vi.mock('../services/api', () => ({
  analyzeTranscription: vi.fn(),
}))

// Mock the useAuthenticatedApi hook - execute API calls directly without token
vi.mock('../hooks/useAuthenticatedApi', () => ({
  useAuthenticatedApi: () => ({
    ensureToken: vi.fn().mockResolvedValue('mock-token'),
    withAuth: vi.fn().mockImplementation((apiCall) => apiCall()),
  }),
}))

// Mock URL.createObjectURL and revokeObjectURL
const mockCreateObjectURL = vi.fn(() => 'blob:test-url')
const mockRevokeObjectURL = vi.fn()
global.URL.createObjectURL = mockCreateObjectURL
global.URL.revokeObjectURL = mockRevokeObjectURL

// Mock sample transcription result
const mockTranscription: TranscriptionResult = {
  text: 'This is a test transcription with some content for testing.',
  duration: 120,
  processingTime: 5.5,
  language: 'en-US',
}

// Mock sample analysis result
const mockAnalysisResult: AnalysisResult = {
  summary: 'This is a test summary of the transcription.',
  keyPoints: ['Point 1', 'Point 2', 'Point 3'],
  actionItems: [
    { task: 'Complete the report', assignee: 'John', deadline: '2024-01-15', priority: 'high' },
    { task: 'Review the document', priority: 'medium' },
  ],
  participants: ['Alice', 'Bob', 'Charlie'],
  topics: ['Testing', 'Development', 'Quality'],
  sentiment: 'positive',
  confidence: 0.92,
}

describe('TranscriptionResults', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('initial render', () => {
    it('renders null when no transcription is provided', () => {
      const { container } = render(<TranscriptionResults transcription={null} />)
      expect(container.firstChild).toBeNull()
    })

    it('renders success banner when transcription is provided', () => {
      render(<TranscriptionResults transcription={mockTranscription} />)
      expect(screen.getByTestId('success-banner')).toBeInTheDocument()
      expect(screen.getByText(/Transcription completed successfully/)).toBeInTheDocument()
    })

    it('renders stats panel with correct values', () => {
      render(<TranscriptionResults transcription={mockTranscription} />)
      
      expect(screen.getByTestId('stats-panel')).toBeInTheDocument()
      expect(screen.getByText('5.5s')).toBeInTheDocument() // Processing time
      expect(screen.getByText('120.0s')).toBeInTheDocument() // Duration
    })

    it('renders transcription textarea with text', () => {
      render(<TranscriptionResults transcription={mockTranscription} />)
      
      const textarea = screen.getByTestId('transcription-textarea')
      expect(textarea).toBeInTheDocument()
      expect(textarea).toHaveValue(mockTranscription.text)
    })

    it('renders analyze button', () => {
      render(<TranscriptionResults transcription={mockTranscription} />)
      expect(screen.getByTestId('analyze-button')).toBeInTheDocument()
    })

    it('renders download buttons', () => {
      render(<TranscriptionResults transcription={mockTranscription} />)
      expect(screen.getByTestId('download-txt-button')).toBeInTheDocument()
      expect(screen.getByTestId('download-json-button')).toBeInTheDocument()
    })

    it('renders clear button', () => {
      render(<TranscriptionResults transcription={mockTranscription} />)
      expect(screen.getByTestId('clear-button')).toBeInTheDocument()
    })
  })

  describe('stats calculation', () => {
    it('calculates character count correctly', () => {
      render(<TranscriptionResults transcription={mockTranscription} />)
      // The text has 60 characters
      const charCount = mockTranscription.text.length.toString()
      expect(screen.getByText(charCount)).toBeInTheDocument()
    })

    it('calculates word count correctly', () => {
      render(<TranscriptionResults transcription={mockTranscription} />)
      // The text has 10 words
      expect(screen.getByText('10')).toBeInTheDocument()
    })
  })

  describe('text editing', () => {
    it('allows editing the transcription text', () => {
      render(<TranscriptionResults transcription={mockTranscription} />)
      
      const textarea = screen.getByTestId('transcription-textarea')
      fireEvent.change(textarea, { target: { value: 'New edited text' } })
      
      expect(textarea).toHaveValue('New edited text')
    })

    it('disables textarea when disabled prop is true', () => {
      render(<TranscriptionResults transcription={mockTranscription} disabled />)
      
      const textarea = screen.getByTestId('transcription-textarea')
      expect(textarea).toBeDisabled()
    })
  })

  describe('AI analysis', () => {
    it('triggers analysis when button is clicked', async () => {
      const mockAnalyze = vi.mocked(api.analyzeTranscription)
      mockAnalyze.mockResolvedValue(mockAnalysisResult)

      render(<TranscriptionResults transcription={mockTranscription} />)
      
      fireEvent.click(screen.getByTestId('analyze-button'))
      
      await waitFor(() => {
        expect(mockAnalyze).toHaveBeenCalledWith(mockTranscription.text, undefined)
      })
    })

    it('displays analysis results after successful analysis', async () => {
      const mockAnalyze = vi.mocked(api.analyzeTranscription)
      mockAnalyze.mockResolvedValue(mockAnalysisResult)

      render(<TranscriptionResults transcription={mockTranscription} />)
      
      fireEvent.click(screen.getByTestId('analyze-button'))
      
      await waitFor(() => {
        expect(screen.getByTestId('analysis-display')).toBeInTheDocument()
      })
    })

    it('displays summary in analysis results', async () => {
      const mockAnalyze = vi.mocked(api.analyzeTranscription)
      mockAnalyze.mockResolvedValue(mockAnalysisResult)

      render(<TranscriptionResults transcription={mockTranscription} />)
      
      fireEvent.click(screen.getByTestId('analyze-button'))
      
      await waitFor(() => {
        expect(screen.getByTestId('analysis-summary')).toBeInTheDocument()
        expect(screen.getByText(mockAnalysisResult.summary)).toBeInTheDocument()
      })
    })

    it('displays key points in analysis results', async () => {
      const mockAnalyze = vi.mocked(api.analyzeTranscription)
      mockAnalyze.mockResolvedValue(mockAnalysisResult)

      render(<TranscriptionResults transcription={mockTranscription} />)
      
      fireEvent.click(screen.getByTestId('analyze-button'))
      
      await waitFor(() => {
        expect(screen.getByTestId('analysis-key-points')).toBeInTheDocument()
        expect(screen.getByText('Point 1')).toBeInTheDocument()
        expect(screen.getByText('Point 2')).toBeInTheDocument()
      })
    })

    it('displays action items in analysis results', async () => {
      const mockAnalyze = vi.mocked(api.analyzeTranscription)
      mockAnalyze.mockResolvedValue(mockAnalysisResult)

      render(<TranscriptionResults transcription={mockTranscription} />)
      
      fireEvent.click(screen.getByTestId('analyze-button'))
      
      await waitFor(() => {
        expect(screen.getByTestId('analysis-action-items')).toBeInTheDocument()
        expect(screen.getByTestId('action-item-0')).toBeInTheDocument()
      })
    })

    it('expands action item details when clicked', async () => {
      const mockAnalyze = vi.mocked(api.analyzeTranscription)
      mockAnalyze.mockResolvedValue(mockAnalysisResult)

      render(<TranscriptionResults transcription={mockTranscription} />)
      
      fireEvent.click(screen.getByTestId('analyze-button'))
      
      await waitFor(() => {
        expect(screen.getByTestId('action-item-0')).toBeInTheDocument()
      })

      fireEvent.click(screen.getByTestId('action-item-0'))
      
      await waitFor(() => {
        expect(screen.getByTestId('action-item-0-details')).toBeInTheDocument()
        expect(screen.getByText('John')).toBeInTheDocument()
      })
    })

    it('displays participants in analysis results', async () => {
      const mockAnalyze = vi.mocked(api.analyzeTranscription)
      mockAnalyze.mockResolvedValue(mockAnalysisResult)

      render(<TranscriptionResults transcription={mockTranscription} />)
      
      fireEvent.click(screen.getByTestId('analyze-button'))
      
      await waitFor(() => {
        expect(screen.getByTestId('analysis-participants')).toBeInTheDocument()
        expect(screen.getByText('• Alice')).toBeInTheDocument()
      })
    })

    it('displays sentiment with emoji', async () => {
      const mockAnalyze = vi.mocked(api.analyzeTranscription)
      mockAnalyze.mockResolvedValue(mockAnalysisResult)

      render(<TranscriptionResults transcription={mockTranscription} />)
      
      fireEvent.click(screen.getByTestId('analyze-button'))
      
      await waitFor(() => {
        expect(screen.getByTestId('analysis-sentiment')).toBeInTheDocument()
        expect(screen.getByText(/Positive/)).toBeInTheDocument()
      })
    })

    it('displays confidence score', async () => {
      const mockAnalyze = vi.mocked(api.analyzeTranscription)
      mockAnalyze.mockResolvedValue(mockAnalysisResult)

      render(<TranscriptionResults transcription={mockTranscription} />)
      
      fireEvent.click(screen.getByTestId('analyze-button'))
      
      await waitFor(() => {
        expect(screen.getByText(/92%/)).toBeInTheDocument()
      })
    })

    it('shows download analysis button after analysis', async () => {
      const mockAnalyze = vi.mocked(api.analyzeTranscription)
      mockAnalyze.mockResolvedValue(mockAnalysisResult)

      render(<TranscriptionResults transcription={mockTranscription} />)
      
      fireEvent.click(screen.getByTestId('analyze-button'))
      
      await waitFor(() => {
        expect(screen.getByTestId('download-analysis-button')).toBeInTheDocument()
      })
    })

    it('shows error message on analysis failure', async () => {
      const mockAnalyze = vi.mocked(api.analyzeTranscription)
      mockAnalyze.mockRejectedValue(new Error('Analysis service unavailable'))

      render(<TranscriptionResults transcription={mockTranscription} />)
      
      fireEvent.click(screen.getByTestId('analyze-button'))
      
      await waitFor(() => {
        expect(screen.getByTestId('analysis-error')).toBeInTheDocument()
        expect(screen.getByText(/Analysis service unavailable/)).toBeInTheDocument()
      })
    })

    it('sends diarized transcript when transcription has segments', async () => {
      const mockAnalyze = vi.mocked(api.analyzeTranscription)
      mockAnalyze.mockResolvedValue(mockAnalysisResult)

      const diarizedTranscription: TranscriptionResult = {
        ...mockTranscription,
        hasDiarization: true,
        speakerCount: 2,
        segments: [
          { speakerId: 'Speaker_1', text: 'Hello there', startTimeMs: 0, endTimeMs: 1000 },
          { speakerId: 'Speaker_2', text: 'Hi!', startTimeMs: 1100, endTimeMs: 2000 },
        ],
      }

      render(<TranscriptionResults transcription={diarizedTranscription} />)
      
      fireEvent.click(screen.getByTestId('analyze-button'))
      
      await waitFor(() => {
        expect(mockAnalyze).toHaveBeenCalledWith(
          diarizedTranscription.text,
          'Speaker 1 [00:00:00]: Hello there\nSpeaker 2 [00:00:01]: Hi!'
        )
      })
    })
  })

  describe('clear functionality', () => {
    it('calls onClear when clear button is clicked', () => {
      const onClear = vi.fn()
      render(<TranscriptionResults transcription={mockTranscription} onClear={onClear} />)
      
      fireEvent.click(screen.getByTestId('clear-button'))
      
      expect(onClear).toHaveBeenCalled()
    })
  })

  describe('disabled state', () => {
    it('disables analyze button when disabled', () => {
      render(<TranscriptionResults transcription={mockTranscription} disabled />)
      
      expect(screen.getByTestId('analyze-button')).toBeDisabled()
    })

    it('disables download buttons when disabled', () => {
      render(<TranscriptionResults transcription={mockTranscription} disabled />)
      
      expect(screen.getByTestId('download-txt-button')).toBeDisabled()
      expect(screen.getByTestId('download-json-button')).toBeDisabled()
    })
  })
})

describe('useAnalysis hook', () => {
  it('exports useAnalysis', async () => {
    const hooks = await import('../hooks')
    expect(hooks.useAnalysis).toBeDefined()
    expect(typeof hooks.useAnalysis).toBe('function')
  })
})

describe('components index exports TranscriptionResults', () => {
  it('exports TranscriptionResults component', async () => {
    const components = await import('../components')
    expect(components.TranscriptionResults).toBeDefined()
  })
})

describe('API exports analyzeTranscription', () => {
  it('exports analyzeTranscription function', async () => {
    // Re-import without mock
    vi.resetModules()
    const actualApi = await import('../services/api')
    expect(actualApi.analyzeTranscription).toBeDefined()
  })
})
