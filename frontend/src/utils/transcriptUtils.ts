// Utility functions for transcript display
import type { SpeakerSegment } from '../types/transcription'

// Speaker color palette - professional, light-theme friendly
const SPEAKER_COLORS: Record<string, { bg: string; text: string; border: string }> = {}
const COLOR_PALETTE = [
  { bg: 'bg-blue-100', text: 'text-blue-900', border: 'border-blue-200' },
  { bg: 'bg-emerald-100', text: 'text-emerald-900', border: 'border-emerald-200' },
  { bg: 'bg-violet-100', text: 'text-violet-900', border: 'border-violet-200' },
  { bg: 'bg-amber-100', text: 'text-amber-900', border: 'border-amber-200' },
  { bg: 'bg-rose-100', text: 'text-rose-900', border: 'border-rose-200' },
  { bg: 'bg-cyan-100', text: 'text-cyan-900', border: 'border-cyan-200' },
  { bg: 'bg-orange-100', text: 'text-orange-900', border: 'border-orange-200' },
  { bg: 'bg-indigo-100', text: 'text-indigo-900', border: 'border-indigo-200' },
]

// Get consistent color for a speaker (memoized per session)
export function getSpeakerColor(speakerId: string): { bg: string; text: string; border: string } {
  if (!SPEAKER_COLORS[speakerId]) {
    const index = Object.keys(SPEAKER_COLORS).length % COLOR_PALETTE.length
    SPEAKER_COLORS[speakerId] = COLOR_PALETTE[index]
  }
  return SPEAKER_COLORS[speakerId]
}

// Reset speaker colors (useful for testing)
export function resetSpeakerColors(): void {
  Object.keys(SPEAKER_COLORS).forEach(key => delete SPEAKER_COLORS[key])
}

// Format milliseconds to HH:MM:SS format
export function formatTimeMs(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000)
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const seconds = totalSeconds % 60
  return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
}

// Format speaker ID to display name
export function formatSpeakerName(speakerId: string): string {
  return speakerId.replace('_', ' ').replace(/^Guest/, 'Speaker ')
}

// Group consecutive segments from the same speaker
export interface GroupedSegment {
  speakerId: string
  speakerName: string
  startTimeMs: number
  endTimeMs: number
  texts: string[]
  fullText: string
}

export function groupSegmentsBySpeaker(segments: SpeakerSegment[]): GroupedSegment[] {
  if (!segments || segments.length === 0) return []

  const groups: GroupedSegment[] = []
  let currentGroup: GroupedSegment | null = null

  for (const segment of segments) {
    const speakerName = formatSpeakerName(segment.speakerId)

    if (currentGroup && currentGroup.speakerId === segment.speakerId) {
      // Same speaker, add to current group
      currentGroup.texts.push(segment.text)
      currentGroup.endTimeMs = segment.endTimeMs
      currentGroup.fullText = currentGroup.texts.join(' ')
    } else {
      // New speaker, create new group
      if (currentGroup) {
        groups.push(currentGroup)
      }
      currentGroup = {
        speakerId: segment.speakerId,
        speakerName,
        startTimeMs: segment.startTimeMs,
        endTimeMs: segment.endTimeMs,
        texts: [segment.text],
        fullText: segment.text,
      }
    }
  }

  if (currentGroup) {
    groups.push(currentGroup)
  }

  return groups
}

// Get unique speakers from segments
export function getUniqueSpeakers(segments: SpeakerSegment[]): string[] {
  return [...new Set(segments.map(s => s.speakerId))]
}

// Determine if speaker should be on left or right (alternating based on speaker order)
export function getSpeakerAlignment(speakerId: string, uniqueSpeakers: string[]): 'left' | 'right' {
  const index = uniqueSpeakers.indexOf(speakerId)
  return index % 2 === 0 ? 'left' : 'right'
}
