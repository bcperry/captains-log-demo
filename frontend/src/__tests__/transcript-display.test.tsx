import { describe, it, expect, beforeEach, vi } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { TranscriptDisplay } from '../components/TranscriptDisplay'
import {
  groupSegmentsBySpeaker,
  getSpeakerColor,
  resetSpeakerColors,
} from '../utils/transcriptUtils'
import type { SpeakerSegment } from '../types/transcription'

// Mock clipboard API
const mockClipboard = {
  writeText: vi.fn().mockResolvedValue(undefined),
}
Object.assign(navigator, { clipboard: mockClipboard })

describe('TranscriptDisplay', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    resetSpeakerColors()
  })

  describe('groupSegmentsBySpeaker', () => {
    it('returns empty array for empty input', () => {
      expect(groupSegmentsBySpeaker([])).toEqual([])
    })

    it('groups consecutive segments from same speaker', () => {
      const segments: SpeakerSegment[] = [
        { speakerId: 'Speaker_1', text: 'Hello', startTimeMs: 0, endTimeMs: 1000 },
        { speakerId: 'Speaker_1', text: 'World', startTimeMs: 1000, endTimeMs: 2000 },
        { speakerId: 'Speaker_2', text: 'Hi there', startTimeMs: 2000, endTimeMs: 3000 },
      ]

      const groups = groupSegmentsBySpeaker(segments)

      expect(groups).toHaveLength(2)
      expect(groups[0].speakerId).toBe('Speaker_1')
      expect(groups[0].texts).toEqual(['Hello', 'World'])
      expect(groups[0].fullText).toBe('Hello World')
      expect(groups[0].startTimeMs).toBe(0)
      expect(groups[0].endTimeMs).toBe(2000)
      expect(groups[1].speakerId).toBe('Speaker_2')
      expect(groups[1].fullText).toBe('Hi there')
    })

    it('handles alternating speakers correctly', () => {
      const segments: SpeakerSegment[] = [
        { speakerId: 'Speaker_1', text: 'A', startTimeMs: 0, endTimeMs: 1000 },
        { speakerId: 'Speaker_2', text: 'B', startTimeMs: 1000, endTimeMs: 2000 },
        { speakerId: 'Speaker_1', text: 'C', startTimeMs: 2000, endTimeMs: 3000 },
        { speakerId: 'Speaker_2', text: 'D', startTimeMs: 3000, endTimeMs: 4000 },
      ]

      const groups = groupSegmentsBySpeaker(segments)

      expect(groups).toHaveLength(4)
      expect(groups.map(g => g.speakerId)).toEqual([
        'Speaker_1', 'Speaker_2', 'Speaker_1', 'Speaker_2'
      ])
    })

    it('formats speaker names correctly', () => {
      const segments: SpeakerSegment[] = [
        { speakerId: 'Guest_1', text: 'Test', startTimeMs: 0, endTimeMs: 1000 },
      ]

      const groups = groupSegmentsBySpeaker(segments)

      expect(groups[0].speakerName).toBe('Speaker  1')
    })

    it('preserves timing information when grouping', () => {
      const segments: SpeakerSegment[] = [
        { speakerId: 'Speaker_1', text: 'First', startTimeMs: 5000, endTimeMs: 6000 },
        { speakerId: 'Speaker_1', text: 'Second', startTimeMs: 6000, endTimeMs: 8000 },
        { speakerId: 'Speaker_1', text: 'Third', startTimeMs: 8000, endTimeMs: 10000 },
      ]

      const groups = groupSegmentsBySpeaker(segments)

      expect(groups).toHaveLength(1)
      expect(groups[0].startTimeMs).toBe(5000)
      expect(groups[0].endTimeMs).toBe(10000)
    })
  })

  describe('getSpeakerColor', () => {
    beforeEach(() => {
      resetSpeakerColors()
    })

    it('returns consistent color for same speaker', () => {
      const color1 = getSpeakerColor('Speaker_1')
      const color2 = getSpeakerColor('Speaker_1')

      expect(color1).toEqual(color2)
    })

    it('returns different colors for different speakers', () => {
      const color1 = getSpeakerColor('Speaker_1')
      const color2 = getSpeakerColor('Speaker_2')

      expect(color1).not.toEqual(color2)
    })

    it('returns color object with bg, text, and border properties', () => {
      const color = getSpeakerColor('Speaker_1')

      expect(color).toHaveProperty('bg')
      expect(color).toHaveProperty('text')
      expect(color).toHaveProperty('border')
      expect(typeof color.bg).toBe('string')
      expect(typeof color.text).toBe('string')
      expect(typeof color.border).toBe('string')
    })

    it('cycles through color palette for many speakers', () => {
      const colors = []
      for (let i = 1; i <= 10; i++) {
        colors.push(getSpeakerColor(`Speaker_${i}`))
      }

      // First 8 should be unique (palette size)
      const uniqueColors = new Set(colors.slice(0, 8).map(c => c.bg))
      expect(uniqueColors.size).toBe(8)

      // Speaker 9 should cycle back to same color as Speaker 1
      expect(colors[8].bg).toBe(colors[0].bg)
    })
  })

  describe('component rendering', () => {
    it('renders null when no segments provided', () => {
      const { container } = render(<TranscriptDisplay segments={[]} />)
      expect(container.firstChild).toBeNull()
    })

    it('renders transcript display with segments', () => {
      const segments: SpeakerSegment[] = [
        { speakerId: 'Speaker_1', text: 'Hello', startTimeMs: 0, endTimeMs: 1000 },
        { speakerId: 'Speaker_2', text: 'Hi there', startTimeMs: 1000, endTimeMs: 2000 },
      ]

      render(<TranscriptDisplay segments={segments} />)

      expect(screen.getByTestId('transcript-display')).toBeInTheDocument()
      expect(screen.getByText('Conversation Transcript')).toBeInTheDocument()
    })

    it('displays speaker count correctly', () => {
      const segments: SpeakerSegment[] = [
        { speakerId: 'Speaker_1', text: 'Hello', startTimeMs: 0, endTimeMs: 1000 },
        { speakerId: 'Speaker_2', text: 'Hi', startTimeMs: 1000, endTimeMs: 2000 },
        { speakerId: 'Speaker_3', text: 'Hey', startTimeMs: 2000, endTimeMs: 3000 },
      ]

      render(<TranscriptDisplay segments={segments} />)

      expect(screen.getByText('3 speakers identified')).toBeInTheDocument()
    })

    it('displays single speaker count grammatically correct', () => {
      const segments: SpeakerSegment[] = [
        { speakerId: 'Speaker_1', text: 'Hello', startTimeMs: 0, endTimeMs: 1000 },
      ]

      render(<TranscriptDisplay segments={segments} />)

      expect(screen.getByText('1 speaker identified')).toBeInTheDocument()
    })

    it('renders chat bubbles for each grouped segment', () => {
      const segments: SpeakerSegment[] = [
        { speakerId: 'Speaker_1', text: 'First message', startTimeMs: 0, endTimeMs: 1000 },
        { speakerId: 'Speaker_2', text: 'Second message', startTimeMs: 1000, endTimeMs: 2000 },
      ]

      render(<TranscriptDisplay segments={segments} />)

      expect(screen.getByTestId('segment-0')).toBeInTheDocument()
      expect(screen.getByTestId('segment-1')).toBeInTheDocument()
      expect(screen.getByText('First message')).toBeInTheDocument()
      expect(screen.getByText('Second message')).toBeInTheDocument()
    })

    it('displays timestamps in HH:MM:SS format', () => {
      const segments: SpeakerSegment[] = [
        { speakerId: 'Speaker_1', text: 'Test', startTimeMs: 3661000, endTimeMs: 3662000 }, // 1:01:01
      ]

      render(<TranscriptDisplay segments={segments} />)

      expect(screen.getByText('01:01:01')).toBeInTheDocument()
    })

    it('renders speaker legend buttons', () => {
      const segments: SpeakerSegment[] = [
        { speakerId: 'Speaker_1', text: 'Hello', startTimeMs: 0, endTimeMs: 1000 },
        { speakerId: 'Speaker_2', text: 'Hi', startTimeMs: 1000, endTimeMs: 2000 },
      ]

      render(<TranscriptDisplay segments={segments} />)

      // Speaker legend should show speaker names
      expect(screen.getAllByText(/Speaker 1/)).toHaveLength(2) // legend + bubble
      expect(screen.getAllByText(/Speaker 2/)).toHaveLength(2) // legend + bubble
    })

    it('renders copy all button', () => {
      const segments: SpeakerSegment[] = [
        { speakerId: 'Speaker_1', text: 'Hello', startTimeMs: 0, endTimeMs: 1000 },
      ]

      render(<TranscriptDisplay segments={segments} />)

      expect(screen.getByTestId('copy-all-button')).toBeInTheDocument()
      expect(screen.getByText('Copy All')).toBeInTheDocument()
    })
  })

  describe('copy functionality', () => {
    it('copies full transcript when copy all button is clicked', async () => {
      const segments: SpeakerSegment[] = [
        { speakerId: 'Speaker_1', text: 'Hello', startTimeMs: 0, endTimeMs: 1000 },
        { speakerId: 'Speaker_2', text: 'Hi there', startTimeMs: 1000, endTimeMs: 2000 },
      ]

      render(<TranscriptDisplay segments={segments} />)

      const copyAllButton = screen.getByTestId('copy-all-button')
      fireEvent.click(copyAllButton)

      expect(mockClipboard.writeText).toHaveBeenCalledWith(
        'Speaker 1 [00:00:00]: Hello\n\nSpeaker 2 [00:00:01]: Hi there'
      )
    })

    it('copies individual segment when copy segment button is clicked', async () => {
      const onCopySegment = vi.fn()
      const segments: SpeakerSegment[] = [
        { speakerId: 'Speaker_1', text: 'Hello world', startTimeMs: 0, endTimeMs: 1000 },
      ]

      render(<TranscriptDisplay segments={segments} onCopySegment={onCopySegment} />)

      const copySegmentButton = screen.getByTestId('copy-segment-0')
      fireEvent.click(copySegmentButton)

      // Wait for async clipboard operation
      await vi.waitFor(() => {
        expect(mockClipboard.writeText).toHaveBeenCalledWith('Hello world')
        expect(onCopySegment).toHaveBeenCalledWith('Hello world', 'Speaker_1')
      })
    })
  })

  describe('mobile responsiveness', () => {
    it('has responsive max-width classes on bubbles', () => {
      const segments: SpeakerSegment[] = [
        { speakerId: 'Speaker_1', text: 'Test', startTimeMs: 0, endTimeMs: 1000 },
      ]

      render(<TranscriptDisplay segments={segments} />)

      const segment = screen.getByTestId('segment-0')
      const bubble = segment.querySelector('div')
      expect(bubble?.className).toContain('max-w-[85%]')
      expect(bubble?.className).toContain('sm:max-w-[70%]')
    })
  })

  describe('speaker alignment', () => {
    it('alternates speaker alignment left and right', () => {
      const segments: SpeakerSegment[] = [
        { speakerId: 'Speaker_1', text: 'Left aligned', startTimeMs: 0, endTimeMs: 1000 },
        { speakerId: 'Speaker_2', text: 'Right aligned', startTimeMs: 1000, endTimeMs: 2000 },
      ]

      render(<TranscriptDisplay segments={segments} />)

      const segment0 = screen.getByTestId('segment-0')
      const segment1 = screen.getByTestId('segment-1')

      expect(segment0.className).toContain('justify-start')
      expect(segment1.className).toContain('justify-end')
    })
  })

  describe('no emojis', () => {
    it('does not contain any emoji characters in headers or labels', () => {
      const segments: SpeakerSegment[] = [
        { speakerId: 'Speaker_1', text: 'Test content', startTimeMs: 0, endTimeMs: 1000 },
      ]

      render(<TranscriptDisplay segments={segments} />)

      const container = screen.getByTestId('transcript-display')
      const textContent = container.textContent || ''

      // Check for common emojis that might be used
      const emojiPattern = /[\u{1F300}-\u{1F9FF}]|[\u{2600}-\u{26FF}]|[\u{2700}-\u{27BF}]/gu
      expect(textContent.match(emojiPattern)).toBeNull()
    })
  })
})
