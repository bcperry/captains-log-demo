"""Transcription models for API requests and responses."""

from datetime import UTC, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TranscriptionRequest(BaseModel):
    """Request parameters for transcription endpoint."""

    language: str = Field(default="en-US", description="Language code for transcription")


class TranscriptionResponse(BaseModel):
    """Response from transcription endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "Hello, this is a sample transcription from the audio file.",
                "language": "en-US",
                "audio_format": "wav",
                "file_size_bytes": 1024000,
                "transcribed_at": "2024-01-15T10:30:00Z",
                "duration_ms": 5000,
                "processing_time_ms": 1234,
            }
        }
    )

    text: str = Field(..., description="Transcribed text from audio")
    language: str = Field(..., description="Language used for transcription")
    audio_format: str = Field(..., description="Detected audio format")
    file_size_bytes: int = Field(..., description="Size of uploaded audio file in bytes")
    transcribed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of transcription",
    )
    duration_ms: Optional[int] = Field(
        default=None, description="Duration of audio in milliseconds if available"
    )
    processing_time_ms: Optional[int] = Field(
        default=None, description="Time taken to process transcription in milliseconds"
    )


class SpeakerSegment(BaseModel):
    """A segment of speech from a specific speaker."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "speaker_id": "Speaker_1",
                "text": "Hello, how are you today?",
                "start_time_ms": 0,
                "end_time_ms": 2500,
            }
        }
    )

    speaker_id: str = Field(..., description="Unique identifier for the speaker")
    text: str = Field(..., description="Transcribed text for this segment")
    start_time_ms: int = Field(..., description="Start time of segment in milliseconds")
    end_time_ms: int = Field(..., description="End time of segment in milliseconds")


class DiarizedTranscriptionResponse(BaseModel):
    """Response from diarized transcription endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "segments": [
                    {"speaker_id": "Speaker_1", "text": "Hello, how are you?", "start_time_ms": 0, "end_time_ms": 2000},
                    {"speaker_id": "Speaker_2", "text": "I'm doing well, thanks!", "start_time_ms": 2100, "end_time_ms": 4000},
                ],
                "full_text": "Hello, how are you? I'm doing well, thanks!",
                "language": "en-US",
                "audio_format": "wav",
                "file_size_bytes": 2048000,
                "speaker_count": 2,
                "max_speakers": 5,
                "transcribed_at": "2024-01-15T10:30:00Z",
                "duration_ms": 4000,
                "processing_time_ms": 2345,
            }
        }
    )

    segments: list[SpeakerSegment] = Field(
        ..., description="List of speaker segments with timestamps"
    )
    full_text: str = Field(..., description="Full transcribed text")
    language: str = Field(..., description="Language used for transcription")
    audio_format: str = Field(..., description="Detected audio format")
    file_size_bytes: int = Field(..., description="Size of uploaded audio file in bytes")
    speaker_count: int = Field(..., description="Number of unique speakers detected")
    max_speakers: int = Field(..., description="Maximum speakers configured")
    transcribed_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of transcription",
    )
    duration_ms: Optional[int] = Field(
        default=None, description="Duration of audio in milliseconds if available"
    )
    processing_time_ms: Optional[int] = Field(
        default=None, description="Time taken to process transcription in milliseconds"
    )
    folder_path: Optional[str] = Field(
        default=None,
        description="Blob storage folder path for saving analysis"
    )


class TranscriptionError(BaseModel):
    """Error response for transcription failures."""

    error: str = Field(..., description="Error type")
    detail: str = Field(..., description="Error details")


# Constants for validation
ALLOWED_CONTENT_TYPES: dict[str, str] = {
    "audio/wav": "wav",
    "audio/wave": "wav",
    "audio/x-wav": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "m4a",
    "audio/x-m4a": "m4a",
    "audio/m4a": "m4a",
    "video/mp4": "mp4",  # MP4 video files (extract audio)
    "audio/ogg": "ogg",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
}

ALLOWED_EXTENSIONS: set[str] = {"wav", "mp3", "m4a", "mp4", "ogg", "flac"}

# Maximum file size: 500 MB (to support large meeting recordings)
MAX_FILE_SIZE_BYTES: int = 500 * 1024 * 1024

# Default maximum speakers for diarization
DEFAULT_MAX_SPEAKERS: int = 5
MIN_SPEAKERS: int = 1
MAX_SPEAKERS: int = 10


class TranscriptionRecord(BaseModel):
    """Transcription record stored in Cosmos DB."""

    id: str = Field(..., description="Unique transcription ID")
    user_id: str = Field(..., description="User ID (partition key)")
    text: str = Field(..., description="Transcribed text")
    language: str = Field(..., description="Language used for transcription")
    audio_format: str = Field(..., description="Audio file format")
    file_size_bytes: int = Field(..., description="Size of original audio file")
    duration_ms: Optional[int] = Field(default=None, description="Audio duration in milliseconds")
    blob_url: Optional[str] = Field(default=None, description="URL of the audio file in Blob Storage")
    blob_storage_url: Optional[str] = Field(
        default=None, description="URL of the transcription JSON in Blob Storage"
    )
    folder_path: Optional[str] = Field(
        default=None,
        description="Hierarchical folder path: {user_id}/{filename}_{timestamp}",
    )
    processing_time_ms: Optional[int] = Field(
        default=None, description="Time taken to process transcription in milliseconds"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when transcription was created",
    )
    # Optional diarization data
    has_diarization: bool = Field(default=False, description="Whether transcription has speaker diarization")
    speaker_count: Optional[int] = Field(default=None, description="Number of speakers if diarized")
    speaker_ids: Optional[list[str]] = Field(
        default=None, description="List of unique speaker IDs/labels from diarization"
    )
    segments: Optional[list[SpeakerSegment]] = Field(
        default=None, description="Speaker segments if diarized"
    )
    # Language detection
    language_detected: Optional[str] = Field(
        default=None, description="Language detected by Azure Speech Services (may differ from requested language)"
    )
    # Audio hash for cache lookup (SHA256 of file content)
    audio_hash: Optional[str] = Field(
        default=None, description="SHA256 hash of audio file content for cache lookup"
    )
    # Cache metadata
    cached: bool = Field(default=False, description="Whether this result was served from cache")
    # Analysis metadata
    has_analysis: bool = Field(default=False, description="Whether AI analysis exists for this transcription")


class TranscriptionListResponse(BaseModel):
    """Response for listing transcriptions."""

    transcriptions: list[TranscriptionRecord] = Field(..., description="List of transcriptions")
    total: int = Field(..., description="Total number of transcriptions")
    page: int = Field(default=1, description="Current page number")
    per_page: int = Field(default=20, description="Results per page")


# Batch Transcription Models


class BatchTranscriptionJobStatus(str, Enum):
    """Status of a batch transcription job."""

    NOT_STARTED = "NotStarted"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"


class BatchTranscriptionRequest(BaseModel):
    """Request for batch transcription."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "language": "en-US",
                "enable_diarization": True,
                "max_speakers": 5,
            }
        }
    )

    language: str = Field(default="en-US", description="Language code for transcription")
    enable_diarization: bool = Field(default=True, description="Enable speaker diarization")
    max_speakers: int = Field(
        default=DEFAULT_MAX_SPEAKERS,
        ge=MIN_SPEAKERS,
        le=MAX_SPEAKERS,
        description="Maximum number of speakers for diarization",
    )


class BatchTranscriptionJobResponse(BaseModel):
    """Response containing batch transcription job info."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "abc123-def456",
                "status": "Running",
                "display_name": "Audio transcription 2024-01-15",
                "created_at": "2024-01-15T10:30:00Z",
                "blob_url": "https://storage.blob.core.windows.net/audio/file.wav",
            }
        }
    )

    job_id: str = Field(..., description="Unique job ID for polling status")
    status: BatchTranscriptionJobStatus = Field(..., description="Current job status")
    display_name: str = Field(..., description="Display name for the job")
    created_at: datetime = Field(..., description="When the job was created")
    blob_url: Optional[str] = Field(default=None, description="URL of the audio file")
    folder_path: Optional[str] = Field(
        default=None,
        description="Hierarchical folder path: {user_id}/{filename}_{timestamp}",
    )
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


class BatchTranscriptionStatusResponse(BaseModel):
    """Response for batch transcription status polling."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "abc123-def456",
                "status": "Succeeded",
                "display_name": "Audio transcription 2024-01-15",
                "created_at": "2024-01-15T10:30:00Z",
                "completed_at": "2024-01-15T10:32:00Z",
                "progress_percent": 100,
            }
        }
    )

    job_id: str = Field(..., description="Unique job ID")
    status: BatchTranscriptionJobStatus = Field(..., description="Current job status")
    display_name: str = Field(..., description="Display name for the job")
    created_at: datetime = Field(..., description="When the job was created")
    completed_at: Optional[datetime] = Field(default=None, description="When the job completed")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")


class BatchTranscriptionResultResponse(BaseModel):
    """Response containing completed batch transcription results."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "job_id": "abc123-def456",
                "segments": [
                    {"speaker_id": "Speaker_1", "text": "Hello, how are you?", "start_time_ms": 0, "end_time_ms": 2000},
                    {"speaker_id": "Speaker_2", "text": "I'm doing well, thanks!", "start_time_ms": 2100, "end_time_ms": 4000},
                ],
                "full_text": "Hello, how are you? I'm doing well, thanks!",
                "language": "en-US",
                "duration_ms": 4000,
                "speaker_count": 2,
            }
        }
    )

    job_id: str = Field(..., description="Unique job ID")
    segments: list[SpeakerSegment] = Field(..., description="Speaker segments with timestamps")
    full_text: str = Field(..., description="Full transcribed text")
    language: str = Field(..., description="Language used for transcription")
    duration_ms: int = Field(..., description="Audio duration in milliseconds")
    speaker_count: int = Field(..., description="Number of speakers detected")


class TranscriptionContentSegment(BaseModel):
    """A speaker segment in the transcription content JSON."""

    speaker_id: str = Field(..., description="Unique speaker identifier")
    start_time: float = Field(..., description="Start time in seconds")
    end_time: float = Field(..., description="End time in seconds")
    text: str = Field(..., description="Transcribed text for this segment")


class TranscriptionContent(BaseModel):
    """Full transcription content stored in Blob Storage as JSON.

    This model represents the complete transcription data saved to blob storage.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transcript_id": "abc123-def456",
                "user_id": "user-uuid",
                "filename": "meeting_recording.wav",
                "upload_date": "2024-01-15T10:30:00Z",
                "duration": 120.5,
                "speaker_segments": [
                    {"speaker_id": "Speaker_1", "start_time": 0.0, "end_time": 5.5, "text": "Hello everyone"},
                    {"speaker_id": "Speaker_2", "start_time": 5.8, "end_time": 10.2, "text": "Hi, thanks for joining"},
                ],
                "full_text": "Hello everyone Hi, thanks for joining",
                "language": "en-US",
                "processing_time_ms": 5234,
            }
        }
    )

    transcript_id: str = Field(..., description="Unique transcription ID")
    user_id: str = Field(..., description="User ID who owns the transcription")
    filename: Optional[str] = Field(default=None, description="Original audio filename")
    upload_date: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the transcription was created",
    )
    duration: Optional[float] = Field(
        default=None, description="Audio duration in seconds"
    )
    speaker_segments: list[TranscriptionContentSegment] = Field(
        default_factory=list, description="Speaker segments with timestamps"
    )
    full_text: str = Field(..., description="Complete transcribed text")
    language: str = Field(..., description="Language used for transcription")
    processing_time_ms: Optional[int] = Field(
        default=None, description="Processing time in milliseconds"
    )
    audio_hash: Optional[str] = Field(
        default=None, description="SHA256 hash of audio file content for cache lookup"
    )


class TranscriptionContentResponse(BaseModel):
    """Response for transcription content retrieval endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "transcript_id": "abc123-def456",
                "content": {
                    "transcript_id": "abc123-def456",
                    "user_id": "user-uuid",
                    "filename": "meeting.wav",
                    "upload_date": "2024-01-15T10:30:00Z",
                    "duration": 120.5,
                    "speaker_segments": [],
                    "full_text": "This is the transcription",
                    "language": "en-US",
                    "processing_time_ms": 5234,
                },
                "blob_url": "https://storage.blob.core.windows.net/transcriptions/user/abc.json",
            }
        }
    )

    transcript_id: str = Field(..., description="Transcription ID")
    content: TranscriptionContent = Field(..., description="Full transcription content")
    blob_url: Optional[str] = Field(default=None, description="Blob storage URL for the content")


class TranscriptionMetadata(BaseModel):
    """Transcription metadata stored as JSON in Blob Storage.

    This replaces Cosmos DB for transcription metadata persistence.
    Stored as metadata.json alongside transcript.json in each transcription folder.
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "meeting_20240115_103000",
                "user_id": "user-uuid-123",
                "filename": "meeting_recording.wav",
                "upload_time": "2024-01-15T10:30:00Z",
                "duration_ms": 120000,
                "speaker_count": 3,
                "language": "en-US",
                "audio_format": "wav",
                "file_size_bytes": 2048000,
                "folder_path": "user-uuid/meeting_recording_20240115_103000",
                "audio_hash": "sha256hash...",
            }
        }
    )

    id: str = Field(..., description="Unique transcription ID (derived from folder path)")
    user_id: str = Field(..., description="User ID who owns the transcription")
    filename: str = Field(..., description="Original audio filename")
    upload_time: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When the transcription was created",
    )
    duration_ms: Optional[int] = Field(default=None, description="Audio duration in milliseconds")
    speaker_count: Optional[int] = Field(default=None, description="Number of speakers detected")
    language: str = Field(default="en-US", description="Language used for transcription")
    audio_format: str = Field(default="wav", description="Audio file format")
    file_size_bytes: int = Field(default=0, description="Size of original audio file")
    folder_path: str = Field(..., description="Blob storage folder path")
    audio_hash: Optional[str] = Field(
        default=None, description="SHA256 hash of audio file for cache lookup"
    )
    text: Optional[str] = Field(default=None, description="Brief preview of transcribed text")
    has_diarization: bool = Field(default=False, description="Whether transcription has speaker diarization")
    has_analysis: bool = Field(default=False, description="Whether AI analysis exists for this transcription")
