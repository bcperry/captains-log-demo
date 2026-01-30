import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import { SpeakerNameEditor } from '../components/SpeakerNameEditor'
import * as api from '../services/api'

// Mock the API
vi.mock('../services/api', () => ({
  updateSpeakerNames: vi.fn(),
}))

const mockSpeakerNames = {
  Speaker_1: { name: 'Speaker 1', confidence: 'low' as const, ai_identified: true },
  Speaker_2: { name: 'Bob', confidence: 'high' as const, ai_identified: true },
}

describe('SpeakerNameEditor', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('rendering', () => {
    it('renders nothing when speakerNames is empty', () => {
      const { container } = render(<SpeakerNameEditor speakerNames={{}} />)
      expect(container.firstChild).toBeNull()
    })

    it('renders speaker name inputs for each speaker', () => {
      render(<SpeakerNameEditor speakerNames={mockSpeakerNames} />)
      
      expect(screen.getByTestId('speaker-name-input-Speaker_1')).toBeInTheDocument()
      expect(screen.getByTestId('speaker-name-input-Speaker_2')).toBeInTheDocument()
    })

    it('displays AI-identified names with confidence indicators', () => {
      render(<SpeakerNameEditor speakerNames={mockSpeakerNames} />)
      
      // Check that Bob is displayed
      const bobInput = screen.getByTestId('speaker-name-input-Speaker_2')
      expect(bobInput).toHaveValue('Bob')
      
      // Check confidence indicator is shown
      expect(screen.getByText('high')).toBeInTheDocument()
    })
  })

  describe('editing', () => {
    it('allows editing speaker names', () => {
      render(<SpeakerNameEditor speakerNames={mockSpeakerNames} folderPath="user/test" />)
      
      const input = screen.getByTestId('speaker-name-input-Speaker_1')
      fireEvent.change(input, { target: { value: 'Alice' } })
      
      expect(input).toHaveValue('Alice')
    })

    it('shows save button when changes are made', () => {
      render(<SpeakerNameEditor speakerNames={mockSpeakerNames} folderPath="user/test" />)
      
      const input = screen.getByTestId('speaker-name-input-Speaker_1')
      fireEvent.change(input, { target: { value: 'Alice' } })
      
      const saveButton = screen.getByTestId('save-speaker-names')
      expect(saveButton).not.toBeDisabled()
    })

    it('shows reset button when changes are made', () => {
      render(<SpeakerNameEditor speakerNames={mockSpeakerNames} folderPath="user/test" />)
      
      const input = screen.getByTestId('speaker-name-input-Speaker_1')
      fireEvent.change(input, { target: { value: 'Alice' } })
      
      expect(screen.getByTestId('reset-speaker-names')).toBeInTheDocument()
    })

    it('resets changes when reset button is clicked', () => {
      render(<SpeakerNameEditor speakerNames={mockSpeakerNames} folderPath="user/test" />)
      
      const input = screen.getByTestId('speaker-name-input-Speaker_1')
      fireEvent.change(input, { target: { value: 'Alice' } })
      
      const resetButton = screen.getByTestId('reset-speaker-names')
      fireEvent.click(resetButton)
      
      expect(input).toHaveValue('Speaker 1')
    })
  })

  describe('saving', () => {
    it('calls API to save speaker names on save button click', async () => {
      const mockUpdateSpeakerNames = vi.spyOn(api, 'updateSpeakerNames').mockResolvedValue({
        message: 'Success',
        speaker_names: {
          Speaker_1: { name: 'Alice', confidence: 'high', ai_identified: false },
          Speaker_2: { name: 'Bob', confidence: 'high', ai_identified: true },
        },
      })

      render(<SpeakerNameEditor speakerNames={mockSpeakerNames} folderPath="user/test" />)
      
      const input = screen.getByTestId('speaker-name-input-Speaker_1')
      fireEvent.change(input, { target: { value: 'Alice' } })
      
      const saveButton = screen.getByTestId('save-speaker-names')
      fireEvent.click(saveButton)
      
      await waitFor(() => {
        expect(mockUpdateSpeakerNames).toHaveBeenCalledWith('user/test', expect.objectContaining({
          Speaker_1: expect.objectContaining({ name: 'Alice' }),
        }))
      })
    })

    it('shows success message after successful save', async () => {
      vi.spyOn(api, 'updateSpeakerNames').mockResolvedValue({
        message: 'Success',
        speaker_names: {
          Speaker_1: { name: 'Alice', confidence: 'high', ai_identified: false },
          Speaker_2: { name: 'Bob', confidence: 'high', ai_identified: true },
        },
      })

      render(<SpeakerNameEditor speakerNames={mockSpeakerNames} folderPath="user/test" />)
      
      const input = screen.getByTestId('speaker-name-input-Speaker_1')
      fireEvent.change(input, { target: { value: 'Alice' } })
      
      const saveButton = screen.getByTestId('save-speaker-names')
      fireEvent.click(saveButton)
      
      await waitFor(() => {
        expect(screen.getByText('Speaker names saved successfully!')).toBeInTheDocument()
      })
    })

    it('shows error message on save failure', async () => {
      vi.spyOn(api, 'updateSpeakerNames').mockRejectedValue(new Error('Network error'))

      render(<SpeakerNameEditor speakerNames={mockSpeakerNames} folderPath="user/test" />)
      
      const input = screen.getByTestId('speaker-name-input-Speaker_1')
      fireEvent.change(input, { target: { value: 'Alice' } })
      
      const saveButton = screen.getByTestId('save-speaker-names')
      fireEvent.click(saveButton)
      
      await waitFor(() => {
        expect(screen.getByText('Network error')).toBeInTheDocument()
      })
    })

    it('disables save button when no folder path is provided', () => {
      render(<SpeakerNameEditor speakerNames={mockSpeakerNames} />)
      
      const input = screen.getByTestId('speaker-name-input-Speaker_1')
      fireEvent.change(input, { target: { value: 'Alice' } })
      
      const saveButton = screen.getByTestId('save-speaker-names')
      expect(saveButton).toBeDisabled()
    })
  })

  describe('callbacks', () => {
    it('calls onSpeakerNamesChange after successful save', async () => {
      const mockOnChange = vi.fn()
      vi.spyOn(api, 'updateSpeakerNames').mockResolvedValue({
        message: 'Success',
        speaker_names: {
          Speaker_1: { name: 'Alice', confidence: 'high', ai_identified: false },
          Speaker_2: { name: 'Bob', confidence: 'high', ai_identified: true },
        },
      })

      render(
        <SpeakerNameEditor 
          speakerNames={mockSpeakerNames} 
          folderPath="user/test"
          onSpeakerNamesChange={mockOnChange}
        />
      )
      
      const input = screen.getByTestId('speaker-name-input-Speaker_1')
      fireEvent.change(input, { target: { value: 'Alice' } })
      
      const saveButton = screen.getByTestId('save-speaker-names')
      fireEvent.click(saveButton)
      
      await waitFor(() => {
        expect(mockOnChange).toHaveBeenCalledWith(expect.objectContaining({
          Speaker_1: expect.objectContaining({ name: 'Alice' }),
        }))
      })
    })
  })

  describe('disabled state', () => {
    it('disables inputs when disabled prop is true', () => {
      render(<SpeakerNameEditor speakerNames={mockSpeakerNames} disabled />)
      
      expect(screen.getByTestId('speaker-name-input-Speaker_1')).toBeDisabled()
      expect(screen.getByTestId('speaker-name-input-Speaker_2')).toBeDisabled()
    })
  })
})
