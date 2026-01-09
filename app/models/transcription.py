"""Transcription models for API requests and responses."""

from datetime import UTC, datetime
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
}

ALLOWED_EXTENSIONS: set[str] = {"wav", "mp3", "m4a"}

# Maximum file size: 25 MB
MAX_FILE_SIZE_BYTES: int = 25 * 1024 * 1024

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
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when transcription was created",
    )
    # Optional diarization data
    has_diarization: bool = Field(default=False, description="Whether transcription has speaker diarization")
    speaker_count: Optional[int] = Field(default=None, description="Number of speakers if diarized")
    segments: Optional[list[SpeakerSegment]] = Field(
        default=None, description="Speaker segments if diarized"
    )


class TranscriptionListResponse(BaseModel):
    """Response for listing transcriptions."""

    transcriptions: list[TranscriptionRecord] = Field(..., description="List of transcriptions")
    total: int = Field(..., description="Total number of transcriptions")
    page: int = Field(default=1, description="Current page number")
    per_page: int = Field(default=20, description="Results per page")
