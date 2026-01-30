import { useState, useCallback, useEffect } from 'react'
import type { SpeakerIdentification } from '../types/api'
import { updateSpeakerNames } from '../services/api'

export interface SpeakerNameEditorProps {
  speakerNames: Record<string, SpeakerIdentification>
  folderPath?: string
  onSpeakerNamesChange?: (names: Record<string, SpeakerIdentification>) => void
  disabled?: boolean
}

// Confidence indicator styling
const getConfidenceStyle = (confidence: string): { color: string; label: string } => {
  switch (confidence.toLowerCase()) {
    case 'high':
      return { color: 'text-green-600', label: 'High confidence' }
    case 'medium':
      return { color: 'text-yellow-600', label: 'Medium confidence' }
    case 'low':
      return { color: 'text-red-600', label: 'Low confidence' }
    default:
      return { color: 'text-gray-600', label: 'Unknown' }
  }
}

/**
 * SpeakerNameEditor component allows users to view and edit AI-identified speaker names.
 * Features:
 * - Display AI-identified names with confidence indicators
 * - Editable text fields for manual override
 * - Save changes to backend
 * - Reset to original AI-identified names
 */
export function SpeakerNameEditor({
  speakerNames,
  folderPath,
  onSpeakerNamesChange,
  disabled = false,
}: SpeakerNameEditorProps) {
  const [editedNames, setEditedNames] = useState<Record<string, SpeakerIdentification>>({})
  const [originalNames, setOriginalNames] = useState<Record<string, SpeakerIdentification>>({})
  const [isSaving, setIsSaving] = useState(false)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [saveSuccess, setSaveSuccess] = useState(false)

  // Initialize edited names when speakerNames prop changes
  useEffect(() => {
    setEditedNames({ ...speakerNames })
    setOriginalNames({ ...speakerNames })
  }, [speakerNames])

  // Check if there are unsaved changes
  const hasChanges = useCallback(() => {
    return Object.keys(editedNames).some(
      (key) => editedNames[key]?.name !== originalNames[key]?.name
    )
  }, [editedNames, originalNames])

  // Handle name change for a speaker
  const handleNameChange = useCallback((speakerId: string, newName: string) => {
    setEditedNames((prev) => ({
      ...prev,
      [speakerId]: {
        ...prev[speakerId],
        name: newName,
        ai_identified: false, // Mark as user-edited
      },
    }))
    setSaveSuccess(false)
    setSaveError(null)
  }, [])

  // Save changes to backend
  const handleSave = useCallback(async () => {
    if (!folderPath) {
      setSaveError('Cannot save: missing folder path')
      return
    }

    setIsSaving(true)
    setSaveError(null)
    setSaveSuccess(false)

    try {
      const namesToUpdate: Record<string, { name: string; confidence: string; ai_identified: boolean }> = {}
      Object.entries(editedNames).forEach(([speakerId, data]) => {
        namesToUpdate[speakerId] = {
          name: data.name,
          confidence: data.confidence,
          ai_identified: data.ai_identified,
        }
      })

      const response = await updateSpeakerNames(folderPath, namesToUpdate)
      
      // Update original names to reflect saved state
      const updatedNames: Record<string, SpeakerIdentification> = {}
      Object.entries(response.speaker_names).forEach(([id, data]) => {
        updatedNames[id] = {
          name: data.name,
          confidence: data.confidence as 'high' | 'medium' | 'low',
          ai_identified: data.ai_identified,
        }
      })
      setOriginalNames(updatedNames)
      setEditedNames(updatedNames)
      setSaveSuccess(true)
      onSpeakerNamesChange?.(updatedNames)
    } catch (error) {
      setSaveError(error instanceof Error ? error.message : 'Failed to save speaker names')
    } finally {
      setIsSaving(false)
    }
  }, [folderPath, editedNames, onSpeakerNamesChange])

  // Reset to original AI-identified names
  const handleReset = useCallback(() => {
    setEditedNames({ ...originalNames })
    setSaveSuccess(false)
    setSaveError(null)
  }, [originalNames])

  if (!speakerNames || Object.keys(speakerNames).length === 0) {
    return null
  }

  return (
    <div className="bg-indigo-50 border border-indigo-200 rounded-lg p-4" data-testid="speaker-name-editor">
      <div className="flex items-center justify-between mb-3">
        <h5 className="text-sm font-semibold text-indigo-800">Speaker Names</h5>
        <div className="flex gap-2">
          {hasChanges() && (
            <button
              onClick={handleReset}
              disabled={disabled || isSaving}
              className="px-2 py-1 text-xs font-medium text-gray-600 bg-gray-100 hover:bg-gray-200 rounded transition-colors disabled:opacity-50"
              data-testid="reset-speaker-names"
            >
              Reset
            </button>
          )}
          <button
            onClick={handleSave}
            disabled={disabled || isSaving || !hasChanges() || !folderPath}
            className={`px-3 py-1 text-xs font-medium rounded transition-colors ${
              hasChanges() && !isSaving
                ? 'bg-indigo-600 text-white hover:bg-indigo-700'
                : 'bg-gray-200 text-gray-400 cursor-not-allowed'
            }`}
            data-testid="save-speaker-names"
          >
            {isSaving ? 'Saving...' : 'Save Changes'}
          </button>
        </div>
      </div>

      {/* Error message */}
      {saveError && (
        <div className="mb-3 p-2 bg-red-50 border border-red-200 rounded text-red-700 text-xs">
          {saveError}
        </div>
      )}

      {/* Success message */}
      {saveSuccess && (
        <div className="mb-3 p-2 bg-green-50 border border-green-200 rounded text-green-700 text-xs">
          Speaker names saved successfully!
        </div>
      )}

      {/* Speaker name list */}
      <div className="space-y-2">
        {Object.entries(editedNames).map(([speakerId, data]) => {
          const confidenceStyle = getConfidenceStyle(data.confidence)
          const isEdited = !data.ai_identified
          return (
            <div key={speakerId} className="flex items-center gap-3" data-testid={`speaker-row-${speakerId}`}>
              <span className="text-xs text-gray-500 w-20 flex-shrink-0">{speakerId.replace('_', ' ')}:</span>
              <input
                type="text"
                value={data.name}
                onChange={(e) => handleNameChange(speakerId, e.target.value)}
                disabled={disabled}
                className="flex-1 px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                data-testid={`speaker-name-input-${speakerId}`}
              />
              <div className="flex items-center gap-1 w-28 flex-shrink-0">
                {isEdited ? (
                  <span className="text-xs text-blue-600" title="Manually edited">
                    ✏️ Edited
                  </span>
                ) : (
                  <span className={`text-xs ${confidenceStyle.color}`} title={confidenceStyle.label}>
                    {data.confidence === 'high' && '✓'}
                    {data.confidence === 'medium' && '?'}
                    {data.confidence === 'low' && '⚠'}
                    <span className="ml-1">{data.confidence}</span>
                  </span>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
