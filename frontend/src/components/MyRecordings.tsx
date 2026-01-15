import { useState, useEffect, useCallback } from 'react'
import { getTranscriptions, deleteTranscription, getTranscription } from '../services/api'
import { TranscriptDisplay } from './TranscriptDisplay'
import type { SpeakerSegment } from '../types/transcription'

// Backend response segment type (snake_case)
interface SpeakerSegmentRaw {
  speaker_id: string
  text: string
  start_time_ms: number
  end_time_ms: number
}

// Backend response types (snake_case)
interface TranscriptionRecordRaw {
  id: string
  user_id: string
  text: string
  language: string
  audio_format: string
  file_size_bytes: number
  duration_ms: number | null
  blob_url: string | null
  created_at: string
  has_diarization: boolean
  speaker_count: number | null
  segments: SpeakerSegmentRaw[] | null
}

interface TranscriptionListResponseRaw {
  transcriptions: TranscriptionRecordRaw[]
  total: number
  page: number
  per_page: number
}

// Frontend types (camelCase)
interface TranscriptionItem {
  id: string
  text: string
  language: string
  audioFormat: string
  fileSizeBytes: number
  durationMs: number | null
  createdAt: string
  hasDiarization: boolean
  speakerCount: number | null
  segments: SpeakerSegment[] | null
  status: 'complete' | 'analyzed'
}

interface MyRecordingsProps {
  onViewTranscription?: (transcription: TranscriptionItem) => void
  onBack?: () => void
}

type SortField = 'createdAt' | 'fileSizeBytes' | 'text'
type SortOrder = 'asc' | 'desc'
type StatusFilter = 'all' | 'complete' | 'analyzed'

export function MyRecordings({ onViewTranscription, onBack }: MyRecordingsProps) {
  const [recordings, setRecordings] = useState<TranscriptionItem[]>([])
  const [total, setTotal] = useState(0)
  const [page, setPage] = useState(1)
  const [perPage] = useState(10)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [sortField, setSortField] = useState<SortField>('createdAt')
  const [sortOrder, setSortOrder] = useState<SortOrder>('desc')
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all')
  const [selectedRecording, setSelectedRecording] = useState<TranscriptionItem | null>(null)
  const [deletingId, setDeletingId] = useState<string | null>(null)

  const transformRecord = (raw: TranscriptionRecordRaw): TranscriptionItem => ({
    id: raw.id,
    text: raw.text,
    language: raw.language,
    audioFormat: raw.audio_format,
    fileSizeBytes: raw.file_size_bytes,
    durationMs: raw.duration_ms,
    createdAt: raw.created_at,
    hasDiarization: raw.has_diarization,
    speakerCount: raw.speaker_count,
    segments: raw.segments?.map(seg => ({
      speakerId: seg.speaker_id,
      text: seg.text,
      startTimeMs: seg.start_time_ms,
      endTimeMs: seg.end_time_ms,
    })) ?? null,
    status: raw.has_diarization ? 'analyzed' : 'complete',
  })

  const loadRecordings = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      // Fetch as raw response and transform
      const response = await getTranscriptions(page, perPage) as unknown as TranscriptionListResponseRaw
      const items = response.transcriptions.map(transformRecord)
      setRecordings(items)
      setTotal(response.total)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load recordings')
    } finally {
      setLoading(false)
    }
  }, [page, perPage])

  useEffect(() => {
    loadRecordings()
  }, [loadRecordings])

  const handleDelete = async (id: string) => {
    if (!confirm('Are you sure you want to delete this recording?')) return
    
    setDeletingId(id)
    try {
      await deleteTranscription(id)
      setRecordings((prev) => prev.filter((r) => r.id !== id))
      setTotal((prev) => prev - 1)
      if (selectedRecording?.id === id) {
        setSelectedRecording(null)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to delete recording')
    } finally {
      setDeletingId(null)
    }
  }

  const handleViewDetail = async (recording: TranscriptionItem) => {
    try {
      // Fetch full transcription details
      const fullRecord = await getTranscription(recording.id) as unknown as TranscriptionRecordRaw
      const transformed = transformRecord(fullRecord)
      setSelectedRecording(transformed)
      if (onViewTranscription) {
        onViewTranscription(transformed)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load transcription details')
    }
  }

  // Filter and sort recordings
  const filteredRecordings = recordings
    .filter((r) => {
      // Search filter
      if (searchQuery) {
        const query = searchQuery.toLowerCase()
        return (
          r.text.toLowerCase().includes(query) ||
          r.audioFormat.toLowerCase().includes(query)
        )
      }
      return true
    })
    .filter((r) => {
      // Status filter
      if (statusFilter === 'all') return true
      return r.status === statusFilter
    })
    .sort((a, b) => {
      let comparison = 0
      switch (sortField) {
        case 'createdAt':
          comparison = new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime()
          break
        case 'fileSizeBytes':
          comparison = a.fileSizeBytes - b.fileSizeBytes
          break
        case 'text':
          comparison = a.text.localeCompare(b.text)
          break
      }
      return sortOrder === 'asc' ? comparison : -comparison
    })

  const totalPages = Math.ceil(total / perPage)

  const formatDuration = (ms: number | null) => {
    if (!ms) return 'N/A'
    const seconds = Math.floor(ms / 1000)
    const minutes = Math.floor(seconds / 60)
    const remainingSeconds = seconds % 60
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`
  }

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
  }

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  const getStatusBadge = (status: 'complete' | 'analyzed') => {
    const styles = {
      complete: 'bg-green-100 text-green-800',
      analyzed: 'bg-purple-100 text-purple-800',
    }
    return (
      <span className={`px-2 py-1 text-xs font-medium rounded-full ${styles[status]}`}>
        {status === 'complete' ? '✓ Complete' : '🔍 Analyzed'}
      </span>
    )
  }

  // Detail view
  if (selectedRecording) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-6">
          <button
            onClick={() => setSelectedRecording(null)}
            className="flex items-center gap-2 text-blue-600 hover:text-blue-800 transition-colors"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
            Back to recordings
          </button>
          <button
            onClick={() => handleDelete(selectedRecording.id)}
            disabled={deletingId === selectedRecording.id}
            className="px-4 py-2 text-sm text-red-600 hover:text-red-800 hover:bg-red-50 rounded-lg transition-colors disabled:opacity-50"
          >
            {deletingId === selectedRecording.id ? 'Deleting...' : '🗑️ Delete'}
          </button>
        </div>

        <div className="space-y-4">
          <div className="flex items-center gap-4">
            {getStatusBadge(selectedRecording.status)}
            <span className="text-sm text-gray-500">{selectedRecording.language}</span>
            <span className="text-sm text-gray-500">{selectedRecording.audioFormat.toUpperCase()}</span>
          </div>

          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Duration:</span>
              <p className="font-medium">{formatDuration(selectedRecording.durationMs)}</p>
            </div>
            <div>
              <span className="text-gray-500">File Size:</span>
              <p className="font-medium">{formatFileSize(selectedRecording.fileSizeBytes)}</p>
            </div>
            <div>
              <span className="text-gray-500">Uploaded:</span>
              <p className="font-medium">{formatDate(selectedRecording.createdAt)}</p>
            </div>
            {selectedRecording.speakerCount && (
              <div>
                <span className="text-gray-500">Speakers:</span>
                <p className="font-medium">{selectedRecording.speakerCount}</p>
              </div>
            )}
          </div>

          <div className="mt-6">
            <h3 className="text-lg font-semibold text-gray-800 mb-3">Transcription</h3>
            {/* Show speaker bubbles if diarization is available */}
            {selectedRecording.hasDiarization && selectedRecording.segments && selectedRecording.segments.length > 0 ? (
              <TranscriptDisplay segments={selectedRecording.segments} />
            ) : (
              <div className="bg-gray-50 rounded-lg p-4 max-h-96 overflow-y-auto">
                <p className="text-gray-700 whitespace-pre-wrap">{selectedRecording.text}</p>
              </div>
            )}
          </div>
        </div>
      </div>
    )
  }

  // Empty state
  if (!loading && recordings.length === 0 && !error) {
    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-xl font-semibold text-gray-800">📁 My Recordings</h2>
          {onBack && (
            <button
              onClick={onBack}
              className="text-blue-600 hover:text-blue-800 text-sm transition-colors"
            >
              ← Back to Upload
            </button>
          )}
        </div>
        <div className="text-center py-12">
          <svg
            className="w-16 h-16 mx-auto text-gray-400 mb-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z"
            />
          </svg>
          <h3 className="text-lg font-medium text-gray-700 mb-2">No recordings yet</h3>
          <p className="text-gray-500 mb-6">
            Upload your first audio file to start transcribing!
          </p>
          {onBack && (
            <button
              onClick={onBack}
              className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              Upload Audio
            </button>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow-md p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-semibold text-gray-800">📁 My Recordings</h2>
        {onBack && (
          <button
            onClick={onBack}
            className="text-blue-600 hover:text-blue-800 text-sm transition-colors"
          >
            ← Back to Upload
          </button>
        )}
      </div>

      {/* Error message */}
      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          {error}
          <button onClick={() => setError(null)} className="ml-2 underline">
            Dismiss
          </button>
        </div>
      )}

      {/* Search and filters */}
      <div className="flex flex-col md:flex-row gap-4 mb-6">
        <div className="flex-1">
          <input
            type="text"
            placeholder="Search recordings..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        <div className="flex gap-2">
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as StatusFilter)}
            className="px-3 py-2 border border-gray-300 rounded-lg bg-white focus:ring-2 focus:ring-blue-500"
          >
            <option value="all">All Status</option>
            <option value="complete">Complete</option>
            <option value="analyzed">Analyzed</option>
          </select>
          <select
            value={`${sortField}-${sortOrder}`}
            onChange={(e) => {
              const [field, order] = e.target.value.split('-') as [SortField, SortOrder]
              setSortField(field)
              setSortOrder(order)
            }}
            className="px-3 py-2 border border-gray-300 rounded-lg bg-white focus:ring-2 focus:ring-blue-500"
          >
            <option value="createdAt-desc">Newest First</option>
            <option value="createdAt-asc">Oldest First</option>
            <option value="fileSizeBytes-desc">Largest First</option>
            <option value="fileSizeBytes-asc">Smallest First</option>
            <option value="text-asc">A-Z</option>
            <option value="text-desc">Z-A</option>
          </select>
        </div>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="text-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mx-auto mb-2"></div>
          <p className="text-gray-500">Loading recordings...</p>
        </div>
      )}

      {/* Recordings list */}
      {!loading && filteredRecordings.length > 0 && (
        <div className="space-y-3">
          {filteredRecordings.map((recording) => (
            <div
              key={recording.id}
              className="border border-gray-200 rounded-lg p-4 hover:bg-gray-50 transition-colors"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    {getStatusBadge(recording.status)}
                    <span className="text-xs text-gray-500">
                      {recording.audioFormat.toUpperCase()}
                    </span>
                  </div>
                  <p className="text-gray-700 line-clamp-2 mb-2">
                    {recording.text.slice(0, 150)}
                    {recording.text.length > 150 ? '...' : ''}
                  </p>
                  <div className="flex items-center gap-4 text-xs text-gray-500">
                    <span>📅 {formatDate(recording.createdAt)}</span>
                    <span>⏱️ {formatDuration(recording.durationMs)}</span>
                    <span>📦 {formatFileSize(recording.fileSizeBytes)}</span>
                    {recording.speakerCount && <span>👥 {recording.speakerCount} speakers</span>}
                  </div>
                </div>
                <div className="flex items-center gap-2 ml-4">
                  <button
                    onClick={() => handleViewDetail(recording)}
                    className="px-3 py-1.5 text-sm text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
                  >
                    View
                  </button>
                  <button
                    onClick={() => handleDelete(recording.id)}
                    disabled={deletingId === recording.id}
                    className="px-3 py-1.5 text-sm text-red-600 hover:bg-red-50 rounded-lg transition-colors disabled:opacity-50"
                  >
                    {deletingId === recording.id ? '...' : '🗑️'}
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* No results after filtering */}
      {!loading && recordings.length > 0 && filteredRecordings.length === 0 && (
        <div className="text-center py-8 text-gray-500">
          No recordings match your search criteria.
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-6 pt-4 border-t border-gray-200">
          <p className="text-sm text-gray-500">
            Showing {(page - 1) * perPage + 1} to {Math.min(page * perPage, total)} of {total} recordings
          </p>
          <div className="flex items-center gap-2">
            <button
              onClick={() => setPage((p) => Math.max(1, p - 1))}
              disabled={page === 1}
              className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Previous
            </button>
            <span className="text-sm text-gray-600">
              Page {page} of {totalPages}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
              className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
