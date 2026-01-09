"""Transcription API endpoints.

This module provides endpoints for audio transcription using Azure Speech Services.
"""

import os
import tempfile
import uuid
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from auth import AuthenticatedUser, get_current_user
from db import get_cosmos_client
from db.cosmos import CosmosClient
from models.transcription import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_EXTENSIONS,
    DEFAULT_MAX_SPEAKERS,
    MAX_FILE_SIZE_BYTES,
    MAX_SPEAKERS,
    MIN_SPEAKERS,
    DiarizedTranscriptionResponse,
    SpeakerSegment,
    TranscriptionRecord,
    TranscriptionResponse,
)
from speech import get_speech_client
from speech.client import (
    SpeechClient,
    SpeechConfigurationError,
    SpeechRecognitionError,
    SpeechServiceError,
    SpeechServiceUnavailableError,
)

router = APIRouter(prefix="/transcribe", tags=["Transcription"])


def get_db() -> CosmosClient:
    """FastAPI dependency for getting database client."""
    return get_cosmos_client()


def get_speech_service() -> SpeechClient:
    """FastAPI dependency for getting the Speech Services client.

    Returns:
        SpeechClient instance

    Raises:
        HTTPException: 503 if Speech Services is not configured or unavailable
    """
    client = get_speech_client()
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Speech Services is not configured",
        )
    return client


def _get_file_extension(filename: Optional[str]) -> Optional[str]:
    """Extract file extension from filename.

    Args:
        filename: The filename to extract extension from

    Returns:
        Lowercase file extension without dot, or None if no extension
    """
    if not filename:
        return None
    ext = os.path.splitext(filename)[1].lower()
    return ext[1:] if ext.startswith(".") else ext if ext else None


def _validate_audio_file(file: UploadFile) -> str:
    """Validate uploaded audio file.

    Args:
        file: Uploaded file to validate

    Returns:
        Detected audio format (wav, mp3, or m4a)

    Raises:
        HTTPException: 400 if file format is invalid or not supported
    """
    # Check content type
    content_type = file.content_type
    audio_format: Optional[str] = None

    if content_type and content_type in ALLOWED_CONTENT_TYPES:
        audio_format = ALLOWED_CONTENT_TYPES[content_type]

    # Fall back to extension if content type not recognized
    if audio_format is None:
        ext = _get_file_extension(file.filename)
        if ext and ext in ALLOWED_EXTENSIONS:
            audio_format = ext
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported audio format. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )

    return audio_format


async def _read_and_validate_file_size(file: UploadFile) -> bytes:
    """Read file content and validate size.

    Args:
        file: Uploaded file to read

    Returns:
        File content as bytes

    Raises:
        HTTPException: 413 if file is too large
    """
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB",
        )
    return content


@router.post(
    "",
    response_model=TranscriptionResponse,
    summary="Transcribe audio file",
    description="Upload an audio file for transcription using Azure Speech Services.",
    responses={
        400: {"description": "Invalid audio format"},
        401: {"description": "Authentication required"},
        413: {"description": "File too large"},
        503: {"description": "Speech Services unavailable"},
    },
)
async def transcribe_audio(
    file: Annotated[UploadFile, File(description="Audio file to transcribe (WAV, MP3, or M4A)")],
    language: Annotated[str, Query(description="Language code for transcription")] = "en-US",
    store: Annotated[bool, Query(description="Store transcription in history")] = True,
    user: AuthenticatedUser = Depends(get_current_user),
    speech_client: SpeechClient = Depends(get_speech_service),
    db: CosmosClient = Depends(get_db),
) -> TranscriptionResponse:
    """Transcribe an uploaded audio file.

    Accepts WAV, MP3, and M4A audio formats.
    Protected by JWT authentication.
    Optionally stores transcription in history.

    Args:
        file: Uploaded audio file
        language: Language code for transcription (default: en-US)
        store: Whether to store transcription in history (default: True)
        user: Authenticated user from Entra ID token
        speech_client: Azure Speech Services client
        db: Database client

    Returns:
        TranscriptionResponse with transcribed text and metadata

    Raises:
        HTTPException: Various status codes for validation/service errors
    """
    # Validate file format
    audio_format = _validate_audio_file(file)

    # Read and validate file size
    content = await _read_and_validate_file_size(file)

    # Save to temporary file for Speech SDK processing
    temp_file_path: Optional[str] = None
    try:
        # Create temp file with appropriate extension
        with tempfile.NamedTemporaryFile(
            suffix=f".{audio_format}",
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Create audio config and transcribe
        audio_config = speech_client.create_audio_config_from_file(temp_file_path)
        transcribed_text = speech_client.recognize_once(audio_config, language)

        response = TranscriptionResponse(
            text=transcribed_text,
            language=language,
            audio_format=audio_format,
            file_size_bytes=len(content),
        )

        # Store transcription in history if requested
        if store:
            record = TranscriptionRecord(
                id=str(uuid.uuid4()),
                user_id=user.oid,
                text=transcribed_text,
                language=language,
                audio_format=audio_format,
                file_size_bytes=len(content),
                has_diarization=False,
            )
            await db.create_transcription(user.oid, record)

        return response

    except SpeechConfigurationError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Speech Services configuration error: {e}",
        ) from e

    except SpeechServiceUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Speech Services unavailable: {e}",
        ) from e

    except SpeechRecognitionError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Speech recognition failed: {e}",
        ) from e

    except SpeechServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech service error: {e}",
        ) from e

    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


@router.post(
    "/diarize",
    response_model=DiarizedTranscriptionResponse,
    summary="Transcribe audio with speaker diarization",
    description="Upload an audio file for transcription with speaker identification.",
    responses={
        400: {"description": "Invalid audio format or max_speakers out of range"},
        401: {"description": "Authentication required"},
        413: {"description": "File too large"},
        503: {"description": "Speech Services unavailable"},
    },
)
async def transcribe_audio_with_diarization(
    file: Annotated[UploadFile, File(description="Audio file to transcribe (WAV, MP3, or M4A)")],
    language: Annotated[str, Query(description="Language code for transcription")] = "en-US",
    max_speakers: Annotated[
        int, Query(description=f"Maximum number of speakers ({MIN_SPEAKERS}-{MAX_SPEAKERS})", ge=MIN_SPEAKERS, le=MAX_SPEAKERS)
    ] = DEFAULT_MAX_SPEAKERS,
    user: AuthenticatedUser = Depends(get_current_user),
    speech_client: SpeechClient = Depends(get_speech_service),
) -> DiarizedTranscriptionResponse:
    """Transcribe an uploaded audio file with speaker diarization.

    Enables speaker identification in the transcription results.
    Returns segments with speaker labels and timestamps.

    Args:
        file: Uploaded audio file
        language: Language code for transcription (default: en-US)
        max_speakers: Maximum number of speakers to identify (1-10)
        user: Authenticated user from Entra ID token
        speech_client: Azure Speech Services client

    Returns:
        DiarizedTranscriptionResponse with speaker segments and metadata

    Raises:
        HTTPException: Various status codes for validation/service errors
    """
    # Validate file format
    audio_format = _validate_audio_file(file)

    # Read and validate file size
    content = await _read_and_validate_file_size(file)

    # Save to temporary file for Speech SDK processing
    temp_file_path: Optional[str] = None
    try:
        # Create temp file with appropriate extension
        with tempfile.NamedTemporaryFile(
            suffix=f".{audio_format}",
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Create audio config and perform diarized transcription
        audio_config = speech_client.create_audio_config_from_file(temp_file_path)
        segments_raw = speech_client.recognize_continuous_with_diarization(
            audio_config, language, max_speakers
        )

        # Convert to SpeakerSegment models
        segments = [
            SpeakerSegment(
                speaker_id=str(seg.get("speaker_id", "Unknown")),
                text=str(seg.get("text", "")),
                start_time_ms=int(str(seg.get("start_time_ms", 0))),
                end_time_ms=int(str(seg.get("end_time_ms", 0))),
            )
            for seg in segments_raw
        ]

        # Compute full text and speaker count
        full_text = " ".join(seg.text for seg in segments if seg.text)
        unique_speakers = set(seg.speaker_id for seg in segments)

        return DiarizedTranscriptionResponse(
            segments=segments,
            full_text=full_text,
            language=language,
            audio_format=audio_format,
            file_size_bytes=len(content),
            speaker_count=len(unique_speakers),
            max_speakers=max_speakers,
        )

    except SpeechConfigurationError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Speech Services configuration error: {e}",
        ) from e

    except SpeechServiceUnavailableError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Speech Services unavailable: {e}",
        ) from e

    except SpeechRecognitionError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Speech recognition failed: {e}",
        ) from e

    except SpeechServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech service error: {e}",
        ) from e

    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
