import { useState, useCallback } from 'react'
import { useAnalysis } from '../hooks/useAnalysis'
import type { TranscriptionResult } from '../types/transcription'
import type { AnalysisResult } from '../types/api'
import { TranscriptDisplay } from './TranscriptDisplay'

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

// Format milliseconds to HH:MM:SS format for chat display
const formatTimeMsChat = (ms: number): string => {
  const totalSeconds = Math.floor(ms / 1000)
  const hours = Math.floor(totalSeconds / 3600)
  const minutes = Math.floor((totalSeconds % 3600) / 60)
  const seconds = totalSeconds % 60
  return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`
}

// Sentiment indicator mapping (professional styling)
const getSentimentIndicator = (sentiment: string): { color: string; label: string } => {
  switch (sentiment.toLowerCase()) {
    case 'positive':
      return { color: 'text-green-600', label: 'Positive' }
    case 'negative':
      return { color: 'text-red-600', label: 'Negative' }
    default:
      return { color: 'text-gray-600', label: 'Neutral' }
  }
}

// Priority color mapping (professional styling)
const getPriorityIndicator = (priority: string): { color: string; bgColor: string } => {
  switch (priority.toLowerCase()) {
    case 'high':
      return { color: 'text-red-700', bgColor: 'bg-red-100' }
    case 'medium':
      return { color: 'text-yellow-700', bgColor: 'bg-yellow-100' }
    case 'low':
      return { color: 'text-green-700', bgColor: 'bg-green-100' }
    default:
      return { color: 'text-gray-700', bgColor: 'bg-gray-100' }
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

  const { state: analysisState, analyze, isAnalyzing, hasResult, reset: resetAnalysis } = useAnalysis({
    folderPath: transcription?.folderPath,
  })

  // Sync editable text with transcription
  const handleTextChange = useCallback((text: string) => {
    setEditableText(text)
  }, [])

  // Initialize editable text when transcription changes
  if (transcription?.text && editableText === '') {
    setEditableText(transcription.text)
  }

  // Handle analyze button click - sends diarized transcript for speaker-aware analysis
  const handleAnalyze = useCallback(async () => {
    // Build diarized transcript string if segments are available
    let diarizedTranscript: string | undefined
    if (transcription?.hasDiarization && transcription?.segments && transcription.segments.length > 0) {
      diarizedTranscript = transcription.segments
        .map((seg) => {
          const speakerName = seg.speakerId.replace('_', ' ').replace(/^Guest/, 'Speaker ')
          const timestamp = formatTimeMsChat(seg.startTimeMs)
          return `${speakerName} [${timestamp}]: ${seg.text}`
        })
        .join('\n')
    }
    await analyze(editableText, diarizedTranscript)
  }, [analyze, editableText, transcription])

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
        <div className="flex items-center text-green-700 gap-2">
          <svg className="w-5 h-5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
          </svg>
          <span className="font-medium">
            Transcription completed successfully!
            {stats.hasDiarization && ` (${stats.speakerCount} speaker${stats.speakerCount !== 1 ? 's' : ''} identified)`}
          </span>
        </div>
      </div>

      {/* Quick stats panel */}
      <div className="bg-gray-50 rounded-lg p-4" data-testid="stats-panel">
        <h3 className="text-sm font-semibold text-gray-700 mb-3">Quick Stats</h3>
        <div className={`grid grid-cols-2 ${stats.hasDiarization ? 'md:grid-cols-5' : 'md:grid-cols-4'} gap-4`}>
          <div className="text-center">
            <p className="text-xs text-gray-500">Processing Time</p>
            <p className="text-lg font-medium text-gray-800">{stats.processingTime.toFixed(1)}s</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-gray-500">Duration</p>
            <p className="text-lg font-medium text-gray-800">{stats.duration.toFixed(1)}s</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-gray-500">Characters</p>
            <p className="text-lg font-medium text-gray-800">{stats.characters.toLocaleString()}</p>
          </div>
          <div className="text-center">
            <p className="text-xs text-gray-500">Words</p>
            <p className="text-lg font-medium text-gray-800">{stats.words.toLocaleString()}</p>
          </div>
          {stats.hasDiarization && (
            <div className="text-center">
              <p className="text-xs text-gray-500">Speakers</p>
              <p className="text-lg font-medium text-gray-800">{stats.speakerCount}</p>
            </div>
          )}
        </div>
      </div>

      {/* Speaker segments - show when diarization is enabled */}
      {stats.hasDiarization && transcription.segments && transcription.segments.length > 0 && (
        <TranscriptDisplay segments={transcription.segments} />
      )}

      {/* Transcription text area */}
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-4">Transcription Results</h3>
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
          <h4 className="text-md font-semibold text-gray-700 mb-3">AI Analysis</h4>
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
                <span className="flex items-center justify-center gap-2">
                  <svg className="animate-spin h-4 w-4" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Analyzing...
                </span>
              ) : (
                'Analyze & Summarize'
              )}
            </button>
          </div>

          {/* Analysis error */}
          {analysisState.error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4 text-red-700 flex items-center gap-2" data-testid="analysis-error">
              <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
              </svg>
              {analysisState.error}
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
            Download TXT
          </button>
          <button
            onClick={handleDownloadJson}
            disabled={disabled || !editableText.trim()}
            className="px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            data-testid="download-json-button"
          >
            Download JSON
          </button>
          {hasResult && (
            <button
              onClick={handleDownloadAnalysis}
              disabled={disabled}
              className="px-4 py-2 bg-purple-100 text-purple-700 rounded-lg hover:bg-purple-200 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              data-testid="download-analysis-button"
            >
              Download Analysis
            </button>
          )}
          <button
            onClick={handleClear}
            disabled={disabled}
            className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            data-testid="clear-button"
          >
            Clear Results
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
        <h5 className="text-sm font-semibold text-blue-800 mb-2">Summary</h5>
        <p className="text-blue-700">{result.summary}</p>
      </div>

      {/* Key Points */}
      {result.keyPoints.length > 0 && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4" data-testid="analysis-key-points">
          <h5 className="text-sm font-semibold text-green-800 mb-2">Key Points</h5>
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
          <h5 className="text-sm font-semibold text-yellow-800 mb-2">Action Items</h5>
          <div className="space-y-2">
            {result.actionItems.map((item, index) => {
              const priorityStyle = getPriorityIndicator(item.priority)
              return (
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
                      <p><strong>Priority:</strong> <span className={`px-2 py-0.5 rounded ${priorityStyle.bgColor} ${priorityStyle.color}`}>{item.priority}</span></p>
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Additional info grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Participants */}
        {result.participants.length > 0 && (
          <div className="bg-gray-50 border border-gray-200 rounded-lg p-4" data-testid="analysis-participants">
            <h5 className="text-sm font-semibold text-gray-700 mb-2">Participants</h5>
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
            <h5 className="text-sm font-semibold text-gray-700 mb-2">Topics</h5>
            <ul className="space-y-1">
              {result.topics.map((topic, index) => (
                <li key={index} className="text-gray-600 text-sm">• {topic}</li>
              ))}
            </ul>
          </div>
        )}

        {/* Sentiment and Confidence */}
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4" data-testid="analysis-sentiment">
          <h5 className="text-sm font-semibold text-gray-700 mb-2">Sentiment & Confidence</h5>
          <div className="space-y-2">
            {(() => {
              const sentimentStyle = getSentimentIndicator(result.sentiment)
              return (
                <p className="text-gray-600 text-sm">
                  Sentiment: <span className={sentimentStyle.color}>{sentimentStyle.label}</span>
                </p>
              )
            })()}
            <p className="text-gray-600 text-sm">
              AI Confidence: {Math.round(result.confidence * 100)}%
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}
