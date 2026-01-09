"""Transcription models for API requests and responses."""

from datetime import UTC, datetime
from typing import Optional

from pydantic import BaseModel, Field


class TranscriptionRequest(BaseModel):
    """Request parameters for transcription endpoint."""

    language: str = Field(default="en-US", description="Language code for transcription")


class TranscriptionResponse(BaseModel):
    """Response from transcription endpoint."""

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
