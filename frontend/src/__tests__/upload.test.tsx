import { describe, it, expect, vi, beforeEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { AudioUpload } from '../components/AudioUpload'
import * as api from '../services/api'

// Mock the API module
vi.mock('../services/api', () => ({
  transcribeAudio: vi.fn(),
}))

// Mock URL.createObjectURL
const mockCreateObjectURL = vi.fn(() => 'blob:test-url')
const mockRevokeObjectURL = vi.fn()
global.URL.createObjectURL = mockCreateObjectURL
global.URL.revokeObjectURL = mockRevokeObjectURL

describe('AudioUpload', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('initial render', () => {
    it('renders upload dropzone when no file is selected', () => {
      render(<AudioUpload />)
      expect(screen.getByText(/Click to upload/i)).toBeInTheDocument()
      expect(screen.getByText(/drag and drop/i)).toBeInTheDocument()
    })

    it('shows supported formats', () => {
      render(<AudioUpload />)
      expect(screen.getByText(/wav, mp3, m4a, ogg, flac, mp4/i)).toBeInTheDocument()
    })

    it('shows max file size', () => {
      render(<AudioUpload />)
      expect(screen.getByText(/100 MB/i)).toBeInTheDocument()
    })

    it('renders hidden file input', () => {
      render(<AudioUpload />)
      const fileInput = screen.getByTestId('file-input')
      expect(fileInput).toBeInTheDocument()
      expect(fileInput).toHaveClass('hidden')
    })
  })

  describe('file selection', () => {
    it('accepts valid audio file via input', async () => {
      render(<AudioUpload />)

      const file = new File(['audio content'], 'test.mp3', { type: 'audio/mpeg' })
      const fileInput = screen.getByTestId('file-input')

      fireEvent.change(fileInput, { target: { files: [file] } })

      await waitFor(() => {
        expect(screen.getByText('test.mp3')).toBeInTheDocument()
      })
    })

    it('shows file info after selection', async () => {
      render(<AudioUpload />)

      const file = new File(['a'.repeat(1024 * 1024)], 'audio.wav', { type: 'audio/wav' })
      const fileInput = screen.getByTestId('file-input')

      fireEvent.change(fileInput, { target: { files: [file] } })

      await waitFor(() => {
        expect(screen.getByText('audio.wav')).toBeInTheDocument()
        expect(screen.getByText(/1\.00 MB/)).toBeInTheDocument()
        expect(screen.getByText('WAV')).toBeInTheDocument()
      })
    })

    it('rejects unsupported file types', async () => {
      const onError = vi.fn()
      render(<AudioUpload onError={onError} />)

      const file = new File(['text content'], 'document.txt', { type: 'text/plain' })
      const fileInput = screen.getByTestId('file-input')

      fireEvent.change(fileInput, { target: { files: [file] } })

      await waitFor(() => {
        expect(onError).toHaveBeenCalledWith(expect.stringContaining('Unsupported file format'))
      })
    })

    it('rejects files over size limit', async () => {
      const onError = vi.fn()
      render(<AudioUpload onError={onError} />)

      // Create a file that exceeds 100MB
      const largeContent = 'a'.repeat(101 * 1024 * 1024)
      const file = new File([largeContent], 'large.mp3', { type: 'audio/mpeg' })
      Object.defineProperty(file, 'size', { value: 101 * 1024 * 1024 })
      const fileInput = screen.getByTestId('file-input')

      fireEvent.change(fileInput, { target: { files: [file] } })

      await waitFor(() => {
        expect(onError).toHaveBeenCalledWith(expect.stringContaining('File too large'))
      })
    })
  })

  describe('drag and drop', () => {
    it('handles drag over state', () => {
      render(<AudioUpload />)
      const dropZone = screen.getByTestId('drop-zone')

      fireEvent.dragOver(dropZone)
      expect(dropZone).toHaveClass('border-blue-500')
    })

    it('handles drag leave state', () => {
      render(<AudioUpload />)
      const dropZone = screen.getByTestId('drop-zone')

      fireEvent.dragOver(dropZone)
      fireEvent.dragLeave(dropZone)
      expect(dropZone).not.toHaveClass('border-blue-500')
    })

    it('accepts valid file on drop', async () => {
      render(<AudioUpload />)
      const dropZone = screen.getByTestId('drop-zone')

      const file = new File(['audio content'], 'dropped.mp3', { type: 'audio/mpeg' })
      const dataTransfer = { files: [file] }

      fireEvent.drop(dropZone, { dataTransfer })

      await waitFor(() => {
        expect(screen.getByText('dropped.mp3')).toBeInTheDocument()
      })
    })
  })

  describe('transcription controls', () => {
    it('shows transcribe button when file is selected', async () => {
      render(<AudioUpload />)

      const file = new File(['audio content'], 'test.mp3', { type: 'audio/mpeg' })
      const fileInput = screen.getByTestId('file-input')

      fireEvent.change(fileInput, { target: { files: [file] } })

      await waitFor(() => {
        expect(screen.getByTestId('transcribe-button')).toBeInTheDocument()
      })
    })

    it('shows clear button when file is selected', async () => {
      render(<AudioUpload />)

      const file = new File(['audio content'], 'test.mp3', { type: 'audio/mpeg' })
      const fileInput = screen.getByTestId('file-input')

      fireEvent.change(fileInput, { target: { files: [file] } })

      await waitFor(() => {
        expect(screen.getByTestId('clear-button')).toBeInTheDocument()
      })
    })

    it('clears file when clear button is clicked', async () => {
      render(<AudioUpload />)

      const file = new File(['audio content'], 'test.mp3', { type: 'audio/mpeg' })
      const fileInput = screen.getByTestId('file-input')

      fireEvent.change(fileInput, { target: { files: [file] } })

      await waitFor(() => {
        expect(screen.getByText('test.mp3')).toBeInTheDocument()
      })

      fireEvent.click(screen.getByTestId('clear-button'))

      await waitFor(() => {
        expect(screen.getByTestId('drop-zone')).toBeInTheDocument()
      })
    })

    it('shows language info', async () => {
      render(<AudioUpload language="es-ES" />)

      const file = new File(['audio content'], 'test.mp3', { type: 'audio/mpeg' })
      const fileInput = screen.getByTestId('file-input')

      fireEvent.change(fileInput, { target: { files: [file] } })

      await waitFor(() => {
        expect(screen.getByText(/es-ES/)).toBeInTheDocument()
      })
    })
  })

  describe('transcription process', () => {
    it('starts transcription when button is clicked', async () => {
      const mockTranscribe = vi.mocked(api.transcribeAudio)
      mockTranscribe.mockResolvedValue({
        text: 'Transcribed text',
        duration: 60,
        processingTime: 5,
        language: 'en-US',
      })

      render(<AudioUpload />)

      const file = new File(['audio content'], 'test.mp3', { type: 'audio/mpeg' })
      const fileInput = screen.getByTestId('file-input')

      fireEvent.change(fileInput, { target: { files: [file] } })

      await waitFor(() => {
        expect(screen.getByTestId('transcribe-button')).toBeInTheDocument()
      })

      fireEvent.click(screen.getByTestId('transcribe-button'))

      await waitFor(() => {
        expect(mockTranscribe).toHaveBeenCalledWith(file, 'en-US')
      })
    })

    it('shows success message on completion', async () => {
      const mockTranscribe = vi.mocked(api.transcribeAudio)
      mockTranscribe.mockResolvedValue({
        text: 'Transcribed text',
        duration: 60,
        processingTime: 5,
        language: 'en-US',
      })

      const onComplete = vi.fn()
      render(<AudioUpload onTranscriptionComplete={onComplete} />)

      const file = new File(['audio content'], 'test.mp3', { type: 'audio/mpeg' })
      const fileInput = screen.getByTestId('file-input')

      fireEvent.change(fileInput, { target: { files: [file] } })

      await waitFor(() => {
        expect(screen.getByTestId('transcribe-button')).toBeInTheDocument()
      })

      fireEvent.click(screen.getByTestId('transcribe-button'))

      await waitFor(() => {
        expect(screen.getByTestId('success-message')).toBeInTheDocument()
        expect(onComplete).toHaveBeenCalledWith(
          expect.objectContaining({
            text: 'Transcribed text',
          })
        )
      })
    })

    it('shows error message on failure', async () => {
      const mockTranscribe = vi.mocked(api.transcribeAudio)
      mockTranscribe.mockRejectedValue(new Error('Transcription failed'))

      const onError = vi.fn()
      render(<AudioUpload onError={onError} />)

      const file = new File(['audio content'], 'test.mp3', { type: 'audio/mpeg' })
      const fileInput = screen.getByTestId('file-input')

      fireEvent.change(fileInput, { target: { files: [file] } })

      await waitFor(() => {
        expect(screen.getByTestId('transcribe-button')).toBeInTheDocument()
      })

      fireEvent.click(screen.getByTestId('transcribe-button'))

      await waitFor(() => {
        expect(screen.getByTestId('error-message')).toBeInTheDocument()
        expect(onError).toHaveBeenCalledWith('Transcription failed')
      })
    })
  })

  describe('disabled state', () => {
    it('disables drop zone when disabled prop is true', () => {
      render(<AudioUpload disabled />)
      const dropZone = screen.getByTestId('drop-zone')
      expect(dropZone).toHaveClass('opacity-50')
    })

    it('disables file input when disabled', () => {
      render(<AudioUpload disabled />)
      const fileInput = screen.getByTestId('file-input')
      expect(fileInput).toBeDisabled()
    })
  })
})

describe('transcription types', () => {
  it('exports SUPPORTED_EXTENSIONS', async () => {
    const types = await import('../types/transcription')
    expect(types.SUPPORTED_EXTENSIONS).toBeDefined()
    expect(types.SUPPORTED_EXTENSIONS).toContain('wav')
    expect(types.SUPPORTED_EXTENSIONS).toContain('mp3')
    expect(types.SUPPORTED_EXTENSIONS).toContain('m4a')
  })

  it('exports MAX_FILE_SIZE_MB', async () => {
    const types = await import('../types/transcription')
    expect(types.MAX_FILE_SIZE_MB).toBe(100)
  })

  it('exports MAX_FILE_SIZE_BYTES', async () => {
    const types = await import('../types/transcription')
    expect(types.MAX_FILE_SIZE_BYTES).toBe(100 * 1024 * 1024)
  })
})

describe('useTranscription hook', () => {
  it('exports useTranscription', async () => {
    const hooks = await import('../hooks')
    expect(hooks.useTranscription).toBeDefined()
    expect(typeof hooks.useTranscription).toBe('function')
  })
})

describe('components index exports AudioUpload', () => {
  it('exports AudioUpload component', async () => {
    const components = await import('../components')
    expect(components.AudioUpload).toBeDefined()
  })
})
