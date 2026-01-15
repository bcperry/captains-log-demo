import { useState, useCallback, useRef, useMemo } from 'react'
import type { SpeakerSegment } from '../types/transcription'
import type { SpeakerIdentification } from '../types/api'
import {
  getSpeakerColor,
  groupSegmentsBySpeaker,
  getUniqueSpeakers,
  getSpeakerAlignment,
  formatSpeakerName,
  formatTimeMs,
  type GroupedSegment,
} from '../utils/transcriptUtils'

export interface TranscriptDisplayProps {
  segments: SpeakerSegment[]
  speakerNames?: Record<string, SpeakerIdentification>
  onCopySegment?: (text: string, speakerId: string) => void
}

/**
 * Get display name for a speaker, using AI-identified/user-edited name if available.
 */
function getDisplayName(speakerId: string, speakerNames?: Record<string, SpeakerIdentification>): string {
  if (speakerNames) {
    // Try exact match first
    if (speakerNames[speakerId]) {
      return speakerNames[speakerId].name
    }
    // Try with underscore format (Speaker_1 vs Speaker 1)
    const underscoreId = speakerId.replace(' ', '_')
    if (speakerNames[underscoreId]) {
      return speakerNames[underscoreId].name
    }
    // Try with space format
    const spaceId = speakerId.replace('_', ' ')
    if (speakerNames[spaceId]) {
      return speakerNames[spaceId].name
    }
  }
  // Fall back to formatted speaker name
  return formatSpeakerName(speakerId)
}

/**
 * TranscriptDisplay component renders diarized segments as SMS/iMessage-style chat bubbles.
 * Features:
 * - Alternating left/right speaker positioning
 * - Speaker name at top of first bubble per turn (uses AI-identified names when available)
 * - Timestamps in HH:MM:SS format
 * - Grouped consecutive segments from same speaker
 * - Professional color scheme
 * - Copy to clipboard for segments and full transcript
 * - Mobile responsive
 */
export function TranscriptDisplay({ segments, speakerNames, onCopySegment }: TranscriptDisplayProps) {
  const [copiedIndex, setCopiedIndex] = useState<number | null>(null)
  const [copiedAll, setCopiedAll] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  const groupedSegments = useMemo(() => groupSegmentsBySpeaker(segments), [segments])
  const uniqueSpeakers = useMemo(() => getUniqueSpeakers(segments), [segments])

  // Copy individual segment to clipboard
  const handleCopySegment = useCallback(
    async (group: GroupedSegment, index: number) => {
      try {
        await navigator.clipboard.writeText(group.fullText)
        setCopiedIndex(index)
        onCopySegment?.(group.fullText, group.speakerId)
        setTimeout(() => setCopiedIndex(null), 2000)
      } catch (err) {
        console.error('Failed to copy:', err)
      }
    },
    [onCopySegment]
  )

  // Copy full transcript to clipboard (use identified names)
  const handleCopyAll = useCallback(async () => {
    const fullTranscript = groupedSegments
      .map(g => `${getDisplayName(g.speakerId, speakerNames)} [${formatTimeMs(g.startTimeMs)}]: ${g.fullText}`)
      .join('\n\n')
    try {
      await navigator.clipboard.writeText(fullTranscript)
      setCopiedAll(true)
      setTimeout(() => setCopiedAll(false), 2000)
    } catch (err) {
      console.error('Failed to copy:', err)
    }
  }, [groupedSegments, speakerNames])

  // Scroll to specific segment
  const scrollToSegment = useCallback((index: number) => {
    const element = document.getElementById(`transcript-segment-${index}`)
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }, [])

  if (!segments || segments.length === 0) {
    return null
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200" data-testid="transcript-display">
      {/* Header with speaker legend and copy all button */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 p-4 border-b border-gray-200">
        <div>
          <h3 className="text-base font-semibold text-gray-900">Conversation Transcript</h3>
          <p className="text-sm text-gray-500 mt-0.5">
            {uniqueSpeakers.length} speaker{uniqueSpeakers.length !== 1 ? 's' : ''} identified
          </p>
        </div>
        <div className="flex flex-wrap gap-2 items-center">
          {/* Speaker legend */}
          {uniqueSpeakers.map(speakerId => {
            const color = getSpeakerColor(speakerId)
            const displayName = getDisplayName(speakerId, speakerNames)
            return (
              <button
                key={speakerId}
                onClick={() => {
                  const index = groupedSegments.findIndex(g => g.speakerId === speakerId)
                  if (index >= 0) scrollToSegment(index)
                }}
                className={`px-2 py-1 text-xs font-medium rounded border ${color.bg} ${color.text} ${color.border} hover:opacity-80 transition-opacity`}
                title={`Jump to ${displayName}`}
              >
                {displayName}
              </button>
            )
          })}
          {/* Copy all button */}
          <button
            onClick={handleCopyAll}
            className="px-3 py-1 text-xs font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded border border-gray-200 transition-colors"
            data-testid="copy-all-button"
          >
            {copiedAll ? 'Copied!' : 'Copy All'}
          </button>
        </div>
      </div>

      {/* Chat bubbles container */}
      <div
        ref={containerRef}
        className="p-4 space-y-3 max-h-[500px] overflow-y-auto scroll-smooth"
        data-testid="transcript-segments"
      >
        {groupedSegments.map((group, index) => {
          const color = getSpeakerColor(group.speakerId)
          const alignment = getSpeakerAlignment(group.speakerId, uniqueSpeakers)
          const isRight = alignment === 'right'
          const displayName = getDisplayName(group.speakerId, speakerNames)

          return (
            <div
              key={index}
              id={`transcript-segment-${index}`}
              className={`flex ${isRight ? 'justify-end' : 'justify-start'}`}
              data-testid={`segment-${index}`}
            >
              <div
                className={`max-w-[85%] sm:max-w-[70%] group relative`}
              >
                {/* Speaker name and timestamp */}
                <div
                  className={`flex items-center gap-2 mb-1 ${isRight ? 'justify-end' : 'justify-start'}`}
                >
                  <span className={`text-xs font-medium ${color.text}`} data-testid={`speaker-name-${index}`}>
                    {displayName}
                  </span>
                  <span className="text-xs text-gray-400">
                    {formatTimeMs(group.startTimeMs)}
                  </span>
                </div>

                {/* Chat bubble */}
                <div
                  className={`px-4 py-2.5 rounded-2xl border ${color.bg} ${color.border} ${
                    isRight ? 'rounded-br-md' : 'rounded-bl-md'
                  }`}
                >
                  <p className={`text-sm text-left leading-relaxed ${color.text}`}>
                    {group.fullText}
                  </p>
                </div>

                {/* Copy button (appears on hover) */}
                <button
                  onClick={() => handleCopySegment(group, index)}
                  className={`absolute top-0 ${isRight ? 'left-0 -translate-x-full pr-2' : 'right-0 translate-x-full pl-2'} opacity-0 group-hover:opacity-100 transition-opacity text-gray-400 hover:text-gray-600`}
                  title="Copy segment"
                  data-testid={`copy-segment-${index}`}
                >
                  {copiedIndex === index ? (
                    <svg className="w-4 h-4 text-green-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                  )}
                </button>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
