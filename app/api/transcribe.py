"""Transcription API endpoints.

This module provides endpoints for audio transcription using Azure Speech Services.
"""

import json
import logging
import os
import tempfile
import time
import uuid
from datetime import UTC, datetime
from typing import Annotated, Optional

from fastapi import APIRouter, Depends, File, HTTPException, Path, Query, UploadFile, status

from auth import AuthenticatedUser
from auth.dependencies import get_current_user_azure
from models.transcription import (
    ALLOWED_CONTENT_TYPES,
    ALLOWED_EXTENSIONS,
    DEFAULT_MAX_SPEAKERS,
    MAX_FILE_SIZE_BYTES,
    MAX_SPEAKERS,
    MIN_SPEAKERS,
    BatchTranscriptionJobResponse,
    BatchTranscriptionJobStatus,
    BatchTranscriptionResultResponse,
    BatchTranscriptionStatusResponse,
    DiarizedTranscriptionResponse,
    SpeakerSegment,
    TranscriptionContent,
    TranscriptionContentSegment,
    TranscriptionMetadata,
    TranscriptionRecord,
    TranscriptionResponse,
)
from speech import get_speech_client
from speech.batch import (
    BatchTranscriptionClient,
    BatchTranscriptionError,
    BatchTranscriptionFailedError,
    BatchTranscriptionJobNotFoundError,
    InMemoryBatchTranscriptionClient,
    TranscriptionStatus,
    get_batch_transcription_client,
)
from speech.client import (
    SpeechClient,
    SpeechConfigurationError,
    SpeechRecognitionError,
    SpeechServiceError,
    SpeechServiceUnavailableError,
)
from speech.converter import (
    AudioConversionError,
    NoAudioTrackError,
    convert_to_wav,
    get_audio_duration_ms,
    needs_conversion,
)
from storage import compute_audio_hash, get_cache_metrics, get_storage_client
from storage.blob import BlobNotFoundError, BlobStorageClient, BlobUploadError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/transcribe", tags=["Transcription"])


def get_blob_storage() -> BlobStorageClient:
    """FastAPI dependency for getting the Blob Storage client.

    Returns:
        BlobStorageClient instance
    """
    return get_storage_client()


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
        Detected audio format (wav, mp3, m4a, mp4, ogg, or flac)

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
        400: {"description": "Invalid audio format or MP4 has no audio track"},
        401: {"description": "Authentication required"},
        413: {"description": "File too large"},
        422: {"description": "Audio conversion or speech recognition failed"},
        503: {"description": "Speech Services unavailable"},
    },
)
async def transcribe_audio(
    file: Annotated[UploadFile, File(description="Audio file to transcribe (WAV, MP3, MP4, M4A, OGG, FLAC)")],
    language: Annotated[str, Query(description="Language code for transcription")] = "en-US",
    store: Annotated[bool, Query(description="Store transcription in history")] = True,
    use_cache: Annotated[bool, Query(description="Use cached transcription if available for same audio")] = True,
    user: AuthenticatedUser = Depends(get_current_user_azure),
    speech_client: SpeechClient = Depends(get_speech_service),
    storage: BlobStorageClient = Depends(get_blob_storage),
) -> TranscriptionResponse:
    """Transcribe an uploaded audio file.

    Accepts WAV, MP3, MP4, M4A, OGG, and FLAC audio formats.
    MP3/MP4/M4A/OGG/FLAC files are converted to WAV for optimal Speech SDK processing.
    Protected by JWT authentication.
    Optionally stores transcription in history.
    Saves audio file to Blob Storage when storage is configured.
    
    Caching: Uses SHA256 hash of audio content to detect duplicate uploads.
    If use_cache=True and a transcription exists for the same audio, returns cached result.

    Args:
        file: Uploaded audio file
        language: Language code for transcription (default: en-US)
        store: Whether to store transcription in history (default: True)
        use_cache: Whether to use cached transcription if available (default: True)
        user: Authenticated user from Entra ID token
        speech_client: Azure Speech Services client
        db: Database client
        storage: Blob storage client

    Returns:
        TranscriptionResponse with transcribed text and metadata

    Raises:
        HTTPException: Various status codes for validation/service errors
    """
    # Validate file format
    audio_format = _validate_audio_file(file)

    # Read and validate file size
    content = await _read_and_validate_file_size(file)

    # Compute audio hash for cache lookup
    audio_hash = compute_audio_hash(content)
    logger.debug(f"Computed audio hash: {audio_hash[:16]}...")

    # Get cache metrics for logging
    cache_metrics = get_cache_metrics()

    # Check cache if enabled - scan blob storage for matching audio hash
    if use_cache:
        cached_meta = await storage.get_transcription_by_audio_hash(user.oid, audio_hash)
        if cached_meta:
            cache_metrics.record_hit()
            logger.info(f"Cache HIT for audio hash {audio_hash[:16]}... - returning cached transcription {cached_meta.get('id')}")
            
            # Try to load full transcription from blob storage if available
            cached_folder_path = cached_meta.get("folder_path")
            if cached_folder_path and storage.is_configured():
                try:
                    content_json = await storage.download_transcription_from_user_path(cached_folder_path)
                    content_data = json.loads(content_json)
                    return TranscriptionResponse(
                        text=content_data.get("full_text", cached_meta.get("text", "")),
                        language=cached_meta.get("language", "en-US"),
                        audio_format=cached_meta.get("audio_format", "wav"),
                        file_size_bytes=cached_meta.get("file_size_bytes", 0),
                        duration_ms=cached_meta.get("duration_ms"),
                        processing_time_ms=0,  # No processing needed for cache hit
                    )
                except BlobNotFoundError:
                    logger.warning(f"Cached transcription blob not found, using metadata text")
                except Exception as e:
                    logger.warning(f"Failed to load cached transcription from blob: {e}")
            
            return TranscriptionResponse(
                text=cached_meta.get("text", ""),
                language=cached_meta.get("language", "en-US"),
                audio_format=cached_meta.get("audio_format", "wav"),
                file_size_bytes=cached_meta.get("file_size_bytes", 0),
                duration_ms=cached_meta.get("duration_ms"),
                processing_time_ms=0,  # No processing needed for cache hit
            )
        else:
            cache_metrics.record_miss()
            logger.info(f"Cache MISS for audio hash {audio_hash[:16]}... - proceeding with transcription")

    # Save to temporary file for Speech SDK processing
    temp_file_path: Optional[str] = None
    converted_file_path: Optional[str] = None
    blob_url: Optional[str] = None
    folder_path: Optional[str] = None
    try:
        # Upload to Blob Storage using hierarchical user path
        if storage.is_configured():
            try:
                blob_url, folder_path = await storage.upload_audio_with_user_path(
                    content=content,
                    audio_format=audio_format,
                    user_id=user.oid,
                    original_filename=file.filename,
                    metadata={"audio_hash": audio_hash},
                )
                logger.info(f"Saved audio file to blob storage: {blob_url} (folder: {folder_path})")
            except BlobUploadError as e:
                # Log error but continue with transcription
                logger.warning(f"Failed to save audio to blob storage: {e}")

        # Create temp file with appropriate extension
        with tempfile.NamedTemporaryFile(
            suffix=f".{audio_format}",
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Convert to WAV if needed (MP3, MP4, etc.)
        speech_file_path = temp_file_path
        if needs_conversion(audio_format):
            logger.info(f"Converting {audio_format} to WAV for Speech SDK")
            converted_file_path = convert_to_wav(temp_file_path, audio_format)
            speech_file_path = converted_file_path

        # Get audio duration before transcription
        duration_ms = get_audio_duration_ms(speech_file_path, audio_format)
        logger.debug(f"Audio duration: {duration_ms}ms")

        # Create audio config and transcribe using continuous recognition
        # recognize_continuous handles long audio files (unlike recognize_once which only
        # captures ~15-30 seconds)
        audio_config = speech_client.create_audio_config_from_file(speech_file_path)

        # Capture start time for processing time measurement
        start_time = time.monotonic()
        transcribed_text = speech_client.recognize_continuous(audio_config, language)
        end_time = time.monotonic()

        # Calculate processing time in milliseconds
        processing_time_ms = int((end_time - start_time) * 1000)
        logger.debug(f"Processing time: {processing_time_ms}ms")

        response = TranscriptionResponse(
            text=transcribed_text,
            language=language,
            audio_format=audio_format,
            file_size_bytes=len(content),
            duration_ms=duration_ms,
            processing_time_ms=processing_time_ms,
        )

        # Store transcription in history if requested
        if store:
            transcription_id = str(uuid.uuid4())

            # Create transcription content JSON for blob storage
            transcription_content = TranscriptionContent(
                transcript_id=transcription_id,
                user_id=user.oid,
                filename=file.filename,
                upload_date=datetime.now(UTC),
                duration=duration_ms / 1000.0 if duration_ms else None,
                speaker_segments=[],  # No diarization for basic transcription
                full_text=transcribed_text,
                language=language,
                processing_time_ms=processing_time_ms,
                audio_hash=audio_hash,
            )

            # Save transcription JSON to blob storage using folder path
            blob_storage_url: Optional[str] = None
            if storage.is_configured() and folder_path:
                try:
                    blob_storage_url = await storage.upload_transcription_with_user_path(
                        user_id=user.oid,
                        folder_path=folder_path,
                        content_json=transcription_content.model_dump_json(),
                        metadata={"audio_hash": audio_hash},
                    )
                    logger.info(f"Saved transcription JSON to blob storage: {blob_storage_url}")
                except BlobUploadError as e:
                    # Log but don't fail the transcription
                    logger.warning(f"Failed to save transcription JSON to blob storage: {e}")

            # Save metadata JSON for listing transcriptions (replaces Cosmos DB)
            if storage.is_configured() and folder_path:
                try:
                    metadata = TranscriptionMetadata(
                        id=folder_path,
                        user_id=user.oid,
                        filename=file.filename or "audio",
                        upload_time=datetime.now(UTC),
                        duration_ms=duration_ms,
                        speaker_count=None,
                        language=language,
                        audio_format=audio_format,
                        file_size_bytes=len(content),
                        folder_path=folder_path,
                        audio_hash=audio_hash,
                        text=transcribed_text[:200] if transcribed_text else "",  # Preview text
                        has_diarization=False,
                    )
                    await storage.save_metadata(
                        user_id=user.oid,
                        folder_path=folder_path,
                        metadata_json=metadata.model_dump_json(),
                    )
                    logger.info(f"Saved transcription metadata to blob storage: {folder_path}/metadata.json")
                except BlobUploadError as e:
                    logger.warning(f"Failed to save metadata JSON: {e}")

        return response

    except NoAudioTrackError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"MP4 file has no audio track: {e}",
        ) from e

    except AudioConversionError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Audio conversion failed: {e}",
        ) from e

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
        # Clean up temporary files
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if converted_file_path and os.path.exists(converted_file_path):
            os.unlink(converted_file_path)


@router.post(
    "/diarize",
    response_model=DiarizedTranscriptionResponse,
    summary="Transcribe audio with speaker diarization",
    description="Upload an audio file for transcription with speaker identification.",
    responses={
        400: {"description": "Invalid audio format, max_speakers out of range, or MP4 has no audio track"},
        401: {"description": "Authentication required"},
        413: {"description": "File too large"},
        422: {"description": "Audio conversion or speech recognition failed"},
        503: {"description": "Speech Services unavailable"},
    },
)
async def transcribe_audio_with_diarization(
    file: Annotated[UploadFile, File(description="Audio file to transcribe (WAV, MP3, MP4, M4A, OGG, FLAC)")],
    language: Annotated[str, Query(description="Language code for transcription")] = "en-US",
    max_speakers: Annotated[
        int, Query(description=f"Maximum number of speakers ({MIN_SPEAKERS}-{MAX_SPEAKERS})", ge=MIN_SPEAKERS, le=MAX_SPEAKERS)
    ] = DEFAULT_MAX_SPEAKERS,
    store: Annotated[bool, Query(description="Store transcription in history")] = True,
    use_cache: Annotated[bool, Query(description="Use cached transcription if available for same audio")] = True,
    user: AuthenticatedUser = Depends(get_current_user_azure),
    speech_client: SpeechClient = Depends(get_speech_service),
    storage: BlobStorageClient = Depends(get_blob_storage),
) -> DiarizedTranscriptionResponse:
    """Transcribe an uploaded audio file with speaker diarization.

    Enables speaker identification in the transcription results.
    MP3/MP4/M4A/OGG/FLAC files are converted to WAV for optimal Speech SDK processing.
    Returns segments with speaker labels and timestamps.
    Optionally stores transcription with diarization data in history.
    Saves audio file to Blob Storage when storage is configured.
    
    Caching: Uses SHA256 hash of audio content to detect duplicate uploads.
    If use_cache=True and a diarized transcription exists for the same audio, returns cached result.

    Args:
        file: Uploaded audio file
        language: Language code for transcription (default: en-US)
        max_speakers: Maximum number of speakers to identify (1-10)
        store: Whether to store transcription in history (default: True)
        use_cache: Whether to use cached transcription if available (default: True)
        user: Authenticated user from Entra ID token
        speech_client: Azure Speech Services client
        db: Database client
        storage: Blob storage client

    Returns:
        DiarizedTranscriptionResponse with speaker segments and metadata

    Raises:
        HTTPException: Various status codes for validation/service errors
    """
    # Validate file format
    audio_format = _validate_audio_file(file)

    # Read and validate file size
    content = await _read_and_validate_file_size(file)

    # Compute audio hash for cache lookup
    audio_hash = compute_audio_hash(content)
    logger.debug(f"Computed audio hash: {audio_hash[:16]}...")

    # Get cache metrics for logging
    cache_metrics = get_cache_metrics()

    # Check cache if enabled - only use cache if it has diarization data
    if use_cache:
        cached_meta = await storage.get_transcription_by_audio_hash(user.oid, audio_hash)
        if cached_meta and cached_meta.get("has_diarization"):
            cache_metrics.record_hit()
            logger.info(f"Cache HIT for diarized audio hash {audio_hash[:16]}... - returning cached transcription")
            
            # Try to load full transcription from blob storage
            cached_folder_path = cached_meta.get("folder_path")
            if cached_folder_path and storage.is_configured():
                try:
                    content_json = await storage.download_transcription_from_user_path(cached_folder_path)
                    content_data = json.loads(content_json)
                    segments = [SpeakerSegment(**seg) for seg in content_data.get("speaker_segments", [])]
                    return DiarizedTranscriptionResponse(
                        segments=segments,
                        full_text=content_data.get("full_text", ""),
                        language=cached_meta.get("language", "en-US"),
                        audio_format=cached_meta.get("audio_format", "wav"),
                        file_size_bytes=cached_meta.get("file_size_bytes", 0),
                        speaker_count=cached_meta.get("speaker_count") or 0,
                        max_speakers=max_speakers,
                        duration_ms=cached_meta.get("duration_ms"),
                        processing_time_ms=0,
                        folder_path=cached_folder_path,
                    )
                except Exception as e:
                    logger.warning(f"Failed to load cached diarized transcription: {e}")
            
            cache_metrics.record_miss()
        else:
            cache_metrics.record_miss()
            logger.info(f"Cache MISS for diarized audio hash {audio_hash[:16]}... - proceeding with transcription")

    # Save to temporary file for Speech SDK processing
    temp_file_path: Optional[str] = None
    converted_file_path: Optional[str] = None
    blob_url: Optional[str] = None
    folder_path: Optional[str] = None
    try:
        # Upload to Blob Storage using hierarchical user path
        if storage.is_configured():
            try:
                blob_url, folder_path = await storage.upload_audio_with_user_path(
                    content=content,
                    audio_format=audio_format,
                    user_id=user.oid,
                    original_filename=file.filename,
                    metadata={"audio_hash": audio_hash},
                )
                logger.info(f"Saved audio file to blob storage: {blob_url} (folder: {folder_path})")
            except BlobUploadError as e:
                # Log error but continue with transcription
                logger.warning(f"Failed to save audio to blob storage: {e}")

        # Create temp file with appropriate extension
        with tempfile.NamedTemporaryFile(
            suffix=f".{audio_format}",
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Convert to WAV if needed (MP3, MP4, etc.)
        speech_file_path = temp_file_path
        if needs_conversion(audio_format):
            logger.info(f"Converting {audio_format} to WAV for diarized transcription")
            converted_file_path = convert_to_wav(temp_file_path, audio_format)
            speech_file_path = converted_file_path

        # Get audio duration before transcription
        duration_ms = get_audio_duration_ms(speech_file_path, audio_format)
        logger.debug(f"Audio duration: {duration_ms}ms")

        # Create audio config and perform diarized transcription
        audio_config = speech_client.create_audio_config_from_file(speech_file_path)

        # Capture start time for processing time measurement
        start_time = time.monotonic()
        segments_raw = speech_client.recognize_continuous_with_diarization(
            audio_config, language, max_speakers
        )
        end_time = time.monotonic()

        # Calculate processing time in milliseconds
        processing_time_ms = int((end_time - start_time) * 1000)
        logger.debug(f"Processing time: {processing_time_ms}ms")

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

        response = DiarizedTranscriptionResponse(
            segments=segments,
            full_text=full_text,
            language=language,
            audio_format=audio_format,
            file_size_bytes=len(content),
            speaker_count=len(unique_speakers),
            max_speakers=max_speakers,
            duration_ms=duration_ms,
            processing_time_ms=processing_time_ms,
            folder_path=folder_path,
        )

        # Store transcription with diarization data in history if requested
        if store:
            transcription_id = str(uuid.uuid4())

            # Convert speaker segments to content format (start/end in seconds)
            content_segments = [
                TranscriptionContentSegment(
                    speaker_id=seg.speaker_id,
                    start_time=seg.start_time_ms / 1000.0,
                    end_time=seg.end_time_ms / 1000.0,
                    text=seg.text,
                )
                for seg in segments
            ]

            # Create transcription content JSON for blob storage
            transcription_content = TranscriptionContent(
                transcript_id=transcription_id,
                user_id=user.oid,
                filename=file.filename,
                upload_date=datetime.now(UTC),
                duration=duration_ms / 1000.0 if duration_ms else None,
                speaker_segments=content_segments,
                full_text=full_text,
                language=language,
                processing_time_ms=processing_time_ms,
                audio_hash=audio_hash,
            )

            # Save transcription JSON to blob storage using folder path
            blob_storage_url: Optional[str] = None
            if storage.is_configured() and folder_path:
                try:
                    blob_storage_url = await storage.upload_transcription_with_user_path(
                        user_id=user.oid,
                        folder_path=folder_path,
                        content_json=transcription_content.model_dump_json(),
                        metadata={"audio_hash": audio_hash},
                    )
                    logger.info(f"Saved transcription JSON to blob storage: {blob_storage_url}")
                except BlobUploadError as e:
                    # Log but don't fail the transcription
                    logger.warning(f"Failed to save transcription JSON to blob storage: {e}")

            # Save metadata JSON for listing transcriptions (replaces Cosmos DB)
            if storage.is_configured() and folder_path:
                try:
                    metadata = TranscriptionMetadata(
                        id=folder_path,
                        user_id=user.oid,
                        filename=file.filename or "audio",
                        upload_time=datetime.now(UTC),
                        duration_ms=duration_ms,
                        speaker_count=len(unique_speakers),
                        language=language,
                        audio_format=audio_format,
                        file_size_bytes=len(content),
                        folder_path=folder_path,
                        audio_hash=audio_hash,
                        text=full_text[:200] if full_text else "",  # Preview text
                        has_diarization=True,
                    )
                    await storage.save_metadata(
                        user_id=user.oid,
                        folder_path=folder_path,
                        metadata_json=metadata.model_dump_json(),
                    )
                    logger.info(f"Saved transcription metadata to blob storage: {folder_path}/metadata.json")
                except BlobUploadError as e:
                    logger.warning(f"Failed to save metadata JSON: {e}")

        return response

    except NoAudioTrackError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"MP4 file has no audio track: {e}",
        ) from e

    except AudioConversionError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Audio conversion failed: {e}",
        ) from e

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
        # Clean up temporary files
        if temp_file_path and os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        if converted_file_path and os.path.exists(converted_file_path):
            os.unlink(converted_file_path)


# Batch Transcription Endpoints


def get_batch_service() -> BatchTranscriptionClient:
    """FastAPI dependency for getting the Batch Transcription client.

    Returns:
        BatchTranscriptionClient instance (real or in-memory)

    Raises:
        HTTPException: 503 if Speech Services is not configured
    """
    client = get_batch_transcription_client()
    if client is None:
        # Return in-memory client for development/testing
        return InMemoryBatchTranscriptionClient()
    return client


def _map_status(batch_status: TranscriptionStatus) -> BatchTranscriptionJobStatus:
    """Map internal TranscriptionStatus to API BatchTranscriptionJobStatus."""
    mapping = {
        TranscriptionStatus.NOT_STARTED: BatchTranscriptionJobStatus.NOT_STARTED,
        TranscriptionStatus.RUNNING: BatchTranscriptionJobStatus.RUNNING,
        TranscriptionStatus.SUCCEEDED: BatchTranscriptionJobStatus.SUCCEEDED,
        TranscriptionStatus.FAILED: BatchTranscriptionJobStatus.FAILED,
    }
    return mapping.get(batch_status, BatchTranscriptionJobStatus.NOT_STARTED)


@router.post(
    "/batch",
    response_model=BatchTranscriptionJobResponse,
    summary="Submit batch transcription job",
    description="Upload an audio file for batch transcription. The file is stored in Blob Storage and processed asynchronously.",
    responses={
        400: {"description": "Invalid audio format"},
        401: {"description": "Authentication required"},
        413: {"description": "File too large"},
        503: {"description": "Speech Services unavailable"},
    },
)
async def create_batch_transcription(
    file: Annotated[UploadFile, File(description="Audio file to transcribe (WAV, MP3, or M4A)")],
    language: Annotated[str, Query(description="Language code for transcription")] = "en-US",
    enable_diarization: Annotated[bool, Query(description="Enable speaker diarization")] = True,
    max_speakers: Annotated[
        int, Query(description=f"Maximum speakers ({MIN_SPEAKERS}-{MAX_SPEAKERS})", ge=MIN_SPEAKERS, le=MAX_SPEAKERS)
    ] = DEFAULT_MAX_SPEAKERS,
    user: AuthenticatedUser = Depends(get_current_user_azure),
    batch_client: BatchTranscriptionClient = Depends(get_batch_service),
    storage: BlobStorageClient = Depends(get_blob_storage),
) -> BatchTranscriptionJobResponse:
    """Submit an audio file for batch transcription.

    The file is uploaded to Blob Storage and a batch transcription job is started.
    Use the returned job_id to poll for status and retrieve results.

    Args:
        file: Uploaded audio file
        language: Language code for transcription
        enable_diarization: Whether to enable speaker identification
        max_speakers: Maximum number of speakers for diarization
        user: Authenticated user
        batch_client: Batch transcription client
        storage: Blob storage client

    Returns:
        BatchTranscriptionJobResponse with job_id for polling
    """
    # Validate file format
    audio_format = _validate_audio_file(file)

    # Read and validate file size
    content = await _read_and_validate_file_size(file)

    # Upload to Blob Storage using hierarchical user path
    try:
        blob_url, folder_path = await storage.upload_audio_with_user_path(
            content=content,
            audio_format=audio_format,
            user_id=user.oid,
            original_filename=file.filename,
        )
        logger.info(f"Uploaded audio file to blob storage: {blob_url} (folder: {folder_path})")
    except BlobUploadError as e:
        logger.error(f"Failed to upload audio to blob storage: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to upload audio file: {e}",
        ) from e

    # Generate SAS URL for batch transcription
    try:
        blob_name = storage.extract_blob_name_from_url(blob_url)
        sas_url = storage.get_blob_sas_url(blob_name, expiry_hours=24)
    except Exception as e:
        logger.error(f"Failed to generate SAS URL: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate SAS URL: {e}",
        ) from e

    # Create batch transcription job
    display_name = f"Transcription {datetime.now(UTC).strftime('%Y-%m-%d %H:%M:%S')} - {user.oid[:8]}"

    try:
        job_id = await batch_client.create_transcription_job(
            content_urls=[sas_url],
            display_name=display_name,
            locale=language,
            enable_diarization=enable_diarization,
            enable_word_level_timestamps=True,
            max_speaker_count=max_speakers,
        )

        # Get initial job status
        job = await batch_client.get_transcription_status(job_id)

        return BatchTranscriptionJobResponse(
            job_id=job_id,
            status=_map_status(job.status),
            display_name=display_name,
            created_at=job.created_date_time,
            blob_url=blob_url,
            folder_path=folder_path,
        )

    except BatchTranscriptionError as e:
        logger.error(f"Failed to create batch transcription job: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to create transcription job: {e}",
        ) from e


@router.get(
    "/batch/{job_id}/status",
    response_model=BatchTranscriptionStatusResponse,
    summary="Get batch transcription status",
    description="Poll the status of a batch transcription job.",
    responses={
        401: {"description": "Authentication required"},
        404: {"description": "Job not found"},
        503: {"description": "Speech Services unavailable"},
    },
)
async def get_batch_transcription_status(
    job_id: Annotated[str, Path(description="Batch transcription job ID")],
    user: AuthenticatedUser = Depends(get_current_user_azure),
    batch_client: BatchTranscriptionClient = Depends(get_batch_service),
) -> BatchTranscriptionStatusResponse:
    """Get the status of a batch transcription job.

    Args:
        job_id: ID of the batch transcription job
        user: Authenticated user
        batch_client: Batch transcription client

    Returns:
        BatchTranscriptionStatusResponse with current status
    """
    try:
        job = await batch_client.get_transcription_status(job_id)

        return BatchTranscriptionStatusResponse(
            job_id=job_id,
            status=_map_status(job.status),
            display_name=job.display_name,
            created_at=job.created_date_time,
            completed_at=job.last_action_date_time if job.status == TranscriptionStatus.SUCCEEDED else None,
            error_message=job.error_message,
        )

    except BatchTranscriptionJobNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transcription job not found: {job_id}",
        )
    except BatchTranscriptionError as e:
        logger.error(f"Failed to get transcription status: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to get transcription status: {e}",
        ) from e


@router.get(
    "/batch/{job_id}/result",
    response_model=BatchTranscriptionResultResponse,
    summary="Get batch transcription result",
    description="Retrieve the result of a completed batch transcription job.",
    responses={
        401: {"description": "Authentication required"},
        404: {"description": "Job not found"},
        409: {"description": "Job not complete or failed"},
        503: {"description": "Speech Services unavailable"},
    },
)
async def get_batch_transcription_result(
    job_id: Annotated[str, Path(description="Batch transcription job ID")],
    user: AuthenticatedUser = Depends(get_current_user_azure),
    batch_client: BatchTranscriptionClient = Depends(get_batch_service),
) -> BatchTranscriptionResultResponse:
    """Get the result of a completed batch transcription job.

    Args:
        job_id: ID of the batch transcription job
        user: Authenticated user
        batch_client: Batch transcription client

    Returns:
        BatchTranscriptionResultResponse with transcribed segments
    """
    try:
        result = await batch_client.get_transcription_result(job_id)

        # Convert internal segments to API model
        segments = [
            SpeakerSegment(
                speaker_id=seg.speaker_id,
                text=seg.text,
                start_time_ms=seg.start_time_ms,
                end_time_ms=seg.end_time_ms,
            )
            for seg in result.segments
        ]

        return BatchTranscriptionResultResponse(
            job_id=job_id,
            segments=segments,
            full_text=result.full_text,
            language=result.language,
            duration_ms=result.duration_ms,
            speaker_count=result.speaker_count,
        )

    except BatchTranscriptionJobNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transcription job not found: {job_id}",
        )
    except BatchTranscriptionFailedError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Transcription job failed: {e}",
        )
    except BatchTranscriptionError as e:
        if "not complete" in str(e).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=str(e),
            )
        logger.error(f"Failed to get transcription result: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to get transcription result: {e}",
        ) from e
