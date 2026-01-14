import { useState, useCallback } from 'react'
import { useAnalysis } from '../hooks/useAnalysis'
import type { TranscriptionResult, SpeakerSegment } from '../types/transcription'
import type { AnalysisResult } from '../types/api'

export interface TranscriptionResultsProps {
  transcription: TranscriptionResult | null
  onClear?: () => void
  disabled?: boolean
}

// Utility to download content as a file
const downloadFile = (content: string, filename: string, mimeType: string) => {
  const blob = new Blob([content], { type: mimeType })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

// Format milliseconds to MM:SS format
const formatTimeMs = (ms: number): string => {
  const totalSeconds = Math.floor(ms / 1000)
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  return `${minutes}:${seconds.toString().padStart(2, '0')}`
}

// Generate consistent color for speaker ID
const getSpeakerColor = (speakerId: string): string => {
  const colors = [
    'bg-blue-100 text-blue-800 border-blue-300',
    'bg-green-100 text-green-800 border-green-300',
    'bg-purple-100 text-purple-800 border-purple-300',
    'bg-orange-100 text-orange-800 border-orange-300',
    'bg-pink-100 text-pink-800 border-pink-300',
    'bg-teal-100 text-teal-800 border-teal-300',
    'bg-yellow-100 text-yellow-800 border-yellow-300',
    'bg-red-100 text-red-800 border-red-300',
    'bg-indigo-100 text-indigo-800 border-indigo-300',
    'bg-gray-100 text-gray-800 border-gray-300',
  ]
  // Extract number from speaker ID or use hash
  const match = speakerId.match(/\d+/)
  const index = match ? parseInt(match[0], 10) - 1 : speakerId.charCodeAt(0)
  return colors[Math.abs(index) % colors.length]
}

// Sentiment emoji mapping
const getSentimentEmoji = (sentiment: string): string => {
  switch (sentiment.toLowerCase()) {
    case 'positive':
      return '😊'
    case 'negative':
      return '😔'
    default:
      return '😐'
  }
}

// Priority color mapping
const getPriorityColor = (priority: string): string => {
  switch (priority.toLowerCase()) {
    case 'high':
      return '🔴'
    case 'medium':
      return '🟡'
    case 'low':
      return '🟢'
    default:
      return '⚪'
  }
}

/**
 * TranscriptionResults component displays transcription text and AI analysis.
 */
export function TranscriptionResults({
  transcription,
  onClear,
  disabled = false,
}: TranscriptionResultsProps) {
  const [editableText, setEditableText] = useState<string>('')
  const [expandedActionItem, setExpandedActionItem] = useState<number | null>(null)

  const { state: analysisState, analyze, isAnalyzing, hasResult, reset: resetAnalysis } = useAnalysis()

  // Sync editable text with transcription
  const handleTextChange = useCallback((text: string) => {
    setEditableText(text)
  }, [])

  // Initialize editable text when transcription changes
  if (transcription?.text && editableText === '') {
    setEditableText(transcription.text)
  }

  // Handle analyze button click
  const handleAnalyze = useCallback(async () => {
    await analyze(editableText)
  }, [analyze, editableText])

  // Handle clear button click
  const handleClear = useCallback(() => {
    setEditableText('')
    resetAnalysis()
    onClear?.()
  }, [resetAnalysis, onClear])

  // Download handlers
  const handleDownloadTxt = useCallback(() => {
    downloadFile(editableText, 'transcription.txt', 'text/plain')
  }, [editableText])

  const handleDownloadJson = useCallback(() => {
    const data = {
      text: editableText,
      metadata: {
        processingTime: transcription?.processingTime,
        duration: transcription?.duration,
        language: transcription?.language,
        characters: editableText.length,
        words: editableText.split(/\s+/).filter(Boolean).length,
      },
    }
    downloadFile(JSON.stringify(data, null, 2), 'transcription.json', 'application/json')
  }, [editableText, transcription])

  const handleDownloadAnalysis = useCallback(() => {
    if (analysisState.result) {
      downloadFile(JSON.stringify(analysisState.result, null, 2), 'analysis.json', 'application/json')
    }
  }, [analysisState.result])

  // Toggle action item expansion
  const toggleActionItem = useCallback((index: number) => {
    setExpandedActionItem((prev) => (prev === index ? null : index))
  }, [])

  if (!transcription) {
    return null
  }

  // Calculate stats
  const stats = {
    processingTime: transcription.processingTime ?? 0,
    duration: transcription.duration ?? 0,
    characters: editableText.length,
    words: editableText.split(/\s+/).filter(Boolean).length,
    speakerCount: transcription.speakerCount ?? 0,
    hasDiarization: transcription.hasDiarization ?? false,
  }

  return (
    <div className="space-y-6" data-testid="transcription-results">
      {/* Success banner */}
      <div className="bg-green-50 border border-green-200 rounded-lg p-4" data-testid="success-banner">
        <div className="flex items-center text-green-700">
          <span className="text-lg mr-2">✅</span>
          <span className="font-medium">
            Transcription completed successfully!
            {stats.hasDiarization && ` (${stats.speakerCount} speaker${stats.speakerCount !== 1 ? 's' : ''} identified)`}
          </span>
        </div>
      </div>

      {/* Quick stats panel */}
      <div className="bg-gray-50 rounded-lg p-4" data-testid="stats-panel">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">📈 Quick Stats</h3>
        <div className={`grid grid-cols-2 ${stats.hasDiarization ? 'md:grid-cols-5' : 'md:grid-cols-4'} gap-4`}>
          <div className="text-center">
            <p className="text-xs text-gray-500">⏱️ Processing Time</p>
            <p className="text-lg font-medium text-gray-800">{stats.processingTime.toFixed(1)}s</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-gray-500">🎵 Duration</p>
            <p className="text-lg font-medium text-gray-800">{stats.duration.toFixed(1)}s</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-gray-500">📝 Characters</p>
            <p className="text-lg font-medium text-gray-800">{stats.characters.toLocaleString()}</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-gray-500">🔤 Words</p>
            <p className="text-lg font-medium text-gray-800">{stats.words.toLocaleString()}</p>
          </div>
          {stats.hasDiarization && (
            <div className="text-center">
              <p className="text-xs text-gray-500">👥 Speakers</p>
              <p className="text-lg font-medium text-gray-800">{stats.speakerCount}</p>
            </div>
          )}
        </div>
      </div>

      {/* Speaker segments - show when diarization is enabled */}
      {stats.hasDiarization && transcription.segments && transcription.segments.length > 0 && (
        <SpeakerSegmentsDisplay segments={transcription.segments} />
      )}

      {/* Transcription text area */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">📝 Transcription Results</h3>
        <textarea
          value={editableText}
          onChange={(e) => handleTextChange(e.target.value)}
          className="w-full h-48 p-3 border border-gray-300 rounded-lg resize-y focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          placeholder="Transcribed text will appear here..."
          disabled={disabled}
          data-testid="transcription-textarea"
        />

        {/* AI Analysis section */}
        <div className="mt-6">
          <h4 className="text-md font-semibold text-gray-700 mb-3">🤖 AI Analysis</h4>
          <div className="flex gap-3 mb-4">
            <button
              onClick={handleAnalyze}
              disabled={isAnalyzing || disabled || !editableText.trim()}
              className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                isAnalyzing || disabled || !editableText.trim()
                  ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  : 'bg-purple-600 text-white hover:bg-purple-700'
              }`}
              data-testid="analyze-button"
            >
              {isAnalyzing ? (
                <>
                  <span className="animate-spin inline-block mr-2">⏳</span>
                  Analyzing...
                </>
              ) : (
                '📊 Analyze & Summarize'
              )}
            </button>
          </div>

          {/* Analysis error */}
          {analysisState.error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4 text-red-700" data-testid="analysis-error">
              ❌ {analysisState.error}
            </div>
          )}

          {/* Analysis results */}
          {hasResult && analysisState.result && (
            <AnalysisDisplay result={analysisState.result} expandedActionItem={expandedActionItem} onToggleActionItem={toggleActionItem} />
          )}
        </div>

        {/* Download and clear buttons */}
        <div className="mt-6 flex flex-wrap gap-3">
          <button
            onClick={handleDownloadTxt}
            disabled={disabled || !editableText.trim()}
            className="px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            data-testid="download-txt-button"
          >
            💾 Download TXT
          </button>
          <button
            onClick={handleDownloadJson}
            disabled={disabled || !editableText.trim()}
            className="px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            data-testid="download-json-button"
          >
            📄 Download JSON
          </button>
          {hasResult && (
            <button
              onClick={handleDownloadAnalysis}
              disabled={disabled}
              className="px-4 py-2 bg-purple-100 text-purple-700 rounded-lg hover:bg-purple-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              data-testid="download-analysis-button"
            >
              🧠 Download Analysis
            </button>
          )}
          <button
            onClick={handleClear}
            disabled={disabled}
            className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            data-testid="clear-button"
          >
            🗑️ Clear Results
          </button>
        </div>
      </div>
    </div>
  )
}

// Sub-component for displaying analysis results
interface AnalysisDisplayProps {
  result: AnalysisResult
  expandedActionItem: number | null
  onToggleActionItem: (index: number) => void
}

function AnalysisDisplay({ result, expandedActionItem, onToggleActionItem }: AnalysisDisplayProps) {
  return (
    <div className="space-y-4" data-testid="analysis-display">
      {/* Summary */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4" data-testid="analysis-summary">
        <h5 className="text-sm font-semibold text-blue-800 mb-2">📋 Summary</h5>
        <p className="text-blue-700">{result.summary}</p>
      </div>

      {/* Key Points */}
      {result.keyPoints.length > 0 && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4" data-testid="analysis-key-points">
          <h5 className="text-sm font-semibold text-green-800 mb-2">🔑 Key Points</h5>
          <ul className="space-y-1">
            {result.keyPoints.map((point, index) => (
              <li key={index} className="text-green-700 flex items-start">
                <span className="mr-2">•</span>
                <span>{point}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Action Items */}
      {result.actionItems.length > 0 && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4" data-testid="analysis-action-items">
          <h5 className="text-sm font-semibold text-yellow-800 mb-2">✅ Action Items</h5>
          <div className="space-y-2">
            {result.actionItems.map((item, index) => (
              <div
                key={index}
                className="border border-yellow-300 rounded-lg overflow-hidden"
              >
                <button
                  onClick={() => onToggleActionItem(index)}
                  className="w-full px-3 py-2 text-left bg-yellow-100 hover:bg-yellow-200 transition-colors flex justify-between items-center"
                  data-testid={`action-item-${index}`}
                >
                  <span className="text-yellow-800 truncate">
                    Action {index + 1}: {item.task.slice(0, 50)}
                    {item.task.length > 50 ? '...' : ''}
                  </span>
                  <span>{expandedActionItem === index ? '▼' : '▶'}</span>
                </button>
                {expandedActionItem === index && (
                  <div className="p-3 bg-white space-y-1 text-sm" data-testid={`action-item-${index}-details`}>
                    <p><strong>Task:</strong> {item.task}</p>
                    {item.assignee && <p><strong>Assignee:</strong> {item.assignee}</p>}
                    {item.deadline && <p><strong>Deadline:</strong> {item.deadline}</p>}
                    <p><strong>Priority:</strong> {getPriorityColor(item.priority)} {item.priority}</p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Additional info grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Participants */}
        {result.participants.length > 0 && (
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-4" data-testid="analysis-participants">
            <h5 className="text-sm font-semibold text-gray-700 mb-2">👥 Participants</h5>
            <ul className="space-y-1">
              {result.participants.map((participant, index) => (
                <li key={index} className="text-gray-600 text-sm">• {participant}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Topics */}
        {result.topics.length > 0 && (
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-4" data-testid="analysis-topics">
            <h5 className="text-sm font-semibold text-gray-700 mb-2">📚 Topics</h5>
            <ul className="space-y-1">
              {result.topics.map((topic, index) => (
                <li key={index} className="text-gray-600 text-sm">• {topic}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Sentiment and Confidence */}
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4" data-testid="analysis-sentiment">
          <h5 className="text-sm font-semibold text-gray-700 mb-2">💭 Sentiment & Confidence</h5>
          <div className="space-y-2">
            <p className="text-gray-600 text-sm">
              Sentiment: {getSentimentEmoji(result.sentiment)} {result.sentiment}
            </p>
            <p className="text-gray-600 text-sm">
              AI Confidence: {Math.round(result.confidence * 100)}%
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}

// Sub-component for displaying speaker segments with diarization
interface SpeakerSegmentsDisplayProps {
  segments: SpeakerSegment[]
}

function SpeakerSegmentsDisplay({ segments }: SpeakerSegmentsDisplayProps) {
  if (!segments || segments.length === 0) {
    return null
  }

  // Get unique speakers
  const uniqueSpeakers = [...new Set(segments.map((s) => s.speakerId))]

  return (
    <div className="bg-white rounded-lg shadow-md p-6" data-testid="speaker-segments">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800">👥 Speaker Segments</h3>
        <div className="flex gap-2">
          {uniqueSpeakers.map((speakerId) => (
            <span
              key={speakerId}
              className={`px-2 py-1 text-xs rounded-full border ${getSpeakerColor(speakerId)}`}
            >
              {speakerId.replace('_', ' ').replace(/^Guest/, 'Speaker ')}
            </span>
          ))}
        </div>
      </div>

      <div className="space-y-3 max-h-96 overflow-y-auto" data-testid="segments-list">
        {segments.map((segment, index) => (
          <div
            key={index}
            className={`p-3 rounded-lg border ${getSpeakerColor(segment.speakerId)}`}
            data-testid={`segment-${index}`}
          >
            <div className="flex items-center justify-between mb-1">
              <span className="font-medium text-sm">
                {segment.speakerId.replace('_', ' ').replace(/^Guest/, 'Speaker ')}
              </span>
              <span className="text-xs opacity-70">
                {formatTimeMs(segment.startTimeMs)} - {formatTimeMs(segment.endTimeMs)}
              </span>
            </div>
            <p className="text-sm">{segment.text}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
