"""Transcription history API endpoints.

This module provides endpoints for managing stored transcriptions.
Uses Blob Storage JSON metadata instead of Cosmos DB.
"""

import json
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from auth import AuthenticatedUser
from auth.dependencies import get_current_user_azure
from models.transcription import (
    TranscriptionContent,
    TranscriptionContentResponse,
    TranscriptionListResponse,
    TranscriptionMetadata,
    TranscriptionRecord,
)
from storage import BlobNotFoundError, get_storage_client
from storage.blob import BlobStorageClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/transcriptions", tags=["Transcription History"])


def get_blob_storage() -> BlobStorageClient:
    """FastAPI dependency for getting the Blob Storage client."""
    return get_storage_client()


@router.get(
    "",
    response_model=TranscriptionListResponse,
    summary="List user's transcriptions",
    description="Get a paginated list of the authenticated user's transcriptions from blob storage.",
)
async def list_transcriptions(
    page: Annotated[int, Query(description="Page number", ge=1)] = 1,
    per_page: Annotated[int, Query(description="Results per page", ge=1, le=100)] = 20,
    user: AuthenticatedUser = Depends(get_current_user_azure),
    storage: BlobStorageClient = Depends(get_blob_storage),
) -> TranscriptionListResponse:
    """List transcriptions for the authenticated user from blob storage.

    Scans the user's folder in blob storage for metadata.json files.

    Args:
        page: Page number (1-indexed)
        per_page: Number of results per page
        user: Authenticated user from Entra ID token
        storage: Blob storage client

    Returns:
        TranscriptionListResponse with paginated transcriptions
    """
    try:
        metadata_list, total = await storage.list_user_transcriptions(
            user.oid, page, per_page
        )

        # Convert metadata dicts to TranscriptionRecord
        transcriptions = []
        for meta in metadata_list:
            try:
                from datetime import UTC, datetime
                
                upload_time = meta.get("upload_time")
                if isinstance(upload_time, str):
                    try:
                        created_at = datetime.fromisoformat(upload_time.replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        created_at = datetime.now(UTC)
                elif isinstance(upload_time, datetime):
                    created_at = upload_time
                else:
                    created_at = datetime.now(UTC)

                record = TranscriptionRecord(
                    id=meta.get("id", meta.get("folder_path", "unknown")),
                    user_id=meta.get("user_id", user.oid),
                    text=meta.get("text", ""),
                    language=meta.get("language", "en-US"),
                    audio_format=meta.get("audio_format", "wav"),
                    file_size_bytes=meta.get("file_size_bytes", 0),
                    duration_ms=meta.get("duration_ms"),
                    folder_path=meta.get("folder_path"),
                    created_at=created_at,
                    has_diarization=meta.get("has_diarization", False),
                    speaker_count=meta.get("speaker_count"),
                    audio_hash=meta.get("audio_hash"),
                )
                transcriptions.append(record)
            except Exception as e:
                logger.warning(f"Failed to parse transcription metadata: {e}")
                continue

        return TranscriptionListResponse(
            transcriptions=transcriptions,
            total=total,
            page=page,
            per_page=per_page,
        )
    except Exception as e:
        logger.error(f"Failed to list transcriptions: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to list transcriptions: {e}",
        )


@router.get(
    "/{transcription_id}",
    response_model=TranscriptionRecord,
    summary="Get a specific transcription",
    description="Get details of a specific transcription by ID (folder path).",
    responses={
        404: {"description": "Transcription not found"},
    },
)
async def get_transcription(
    transcription_id: Annotated[str, Path(description="Transcription ID (folder path)")],
    user: AuthenticatedUser = Depends(get_current_user_azure),
    storage: BlobStorageClient = Depends(get_blob_storage),
) -> TranscriptionRecord:
    """Get a specific transcription by ID from blob storage.

    The transcription_id is the folder path (e.g., "user_id/filename_timestamp").

    Args:
        transcription_id: Transcription ID (folder path)
        user: Authenticated user from Entra ID token
        storage: Blob storage client

    Returns:
        TranscriptionRecord if found

    Raises:
        HTTPException: 404 if transcription not found
    """
    # Construct folder path - the ID should already include user_id prefix
    folder_path = transcription_id
    if not folder_path.startswith(f"{user.oid}/"):
        folder_path = f"{user.oid}/{transcription_id}"

    try:
        metadata_json = await storage.get_metadata(folder_path)
        meta = json.loads(metadata_json)

        return TranscriptionRecord(
            id=meta.get("id", folder_path),
            user_id=meta.get("user_id", user.oid),
            text=meta.get("text", ""),
            language=meta.get("language", "en-US"),
            audio_format=meta.get("audio_format", "wav"),
            file_size_bytes=meta.get("file_size_bytes", 0),
            duration_ms=meta.get("duration_ms"),
            folder_path=folder_path,
            created_at=meta.get("upload_time"),
            has_diarization=meta.get("has_diarization", False),
            speaker_count=meta.get("speaker_count"),
            audio_hash=meta.get("audio_hash"),
        )
    except BlobNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transcription with ID '{transcription_id}' not found",
        )
    except Exception as e:
        logger.error(f"Failed to get transcription {transcription_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get transcription: {e}",
        )


@router.delete(
    "/{transcription_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a transcription",
    description="Delete a specific transcription by ID (folder path).",
    responses={
        404: {"description": "Transcription not found"},
        501: {"description": "Delete not implemented for blob storage"},
    },
)
async def delete_transcription(
    transcription_id: Annotated[str, Path(description="Transcription ID (folder path)")],
    user: AuthenticatedUser = Depends(get_current_user_azure),
    storage: BlobStorageClient = Depends(get_blob_storage),
) -> None:
    """Delete a specific transcription by ID.

    Note: This would delete the folder and all contents from blob storage.

    Args:
        transcription_id: Transcription ID (folder path)
        user: Authenticated user from Entra ID token
        storage: Blob storage client

    Raises:
        HTTPException: 404 if transcription not found, 501 if not implemented
    """
    # For now, return 501 Not Implemented - blob deletion requires more work
    # to safely delete entire folders
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Transcription deletion not yet implemented for blob storage",
    )


@router.get(
    "/by-hash/{audio_hash}",
    response_model=TranscriptionRecord,
    summary="Get transcription by audio hash",
    description="Find a transcription by SHA256 hash of the audio file content. Used for cache lookup.",
    responses={
        404: {"description": "No transcription found with this audio hash"},
    },
)
async def get_transcription_by_hash(
    audio_hash: Annotated[str, Path(description="SHA256 hash of audio file content")],
    user: AuthenticatedUser = Depends(get_current_user_azure),
    storage: BlobStorageClient = Depends(get_blob_storage),
) -> TranscriptionRecord:
    """Get a transcription by its audio file hash.

    This endpoint enables client-side cache lookups by computing the audio
    hash before uploading, allowing detection of duplicate uploads.

    Args:
        audio_hash: SHA256 hash of the audio file content
        user: Authenticated user from Entra ID token
        storage: Blob storage client

    Returns:
        TranscriptionRecord if found

    Raises:
        HTTPException: 404 if no transcription with this hash found
    """
    logger.debug(f"Looking up transcription by audio hash: {audio_hash[:16]}...")

    meta = await storage.get_transcription_by_audio_hash(user.oid, audio_hash)

    if meta is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No transcription found with audio hash '{audio_hash[:16]}...'",
        )

    from datetime import UTC, datetime
    
    upload_time = meta.get("upload_time")
    if isinstance(upload_time, str):
        try:
            created_at = datetime.fromisoformat(upload_time.replace('Z', '+00:00'))
        except (ValueError, AttributeError):
            created_at = datetime.now(UTC)
    elif isinstance(upload_time, datetime):
        created_at = upload_time
    else:
        created_at = datetime.now(UTC)

    record = TranscriptionRecord(
        id=meta.get("id", meta.get("folder_path", "unknown")),
        user_id=meta.get("user_id", user.oid),
        text=meta.get("text", ""),
        language=meta.get("language", "en-US"),
        audio_format=meta.get("audio_format", "wav"),
        file_size_bytes=meta.get("file_size_bytes", 0),
        duration_ms=meta.get("duration_ms"),
        folder_path=meta.get("folder_path"),
        created_at=created_at,
        has_diarization=meta.get("has_diarization", False),
        speaker_count=meta.get("speaker_count"),
        audio_hash=meta.get("audio_hash"),
    )

    logger.info(f"Found transcription {record.id} for audio hash {audio_hash[:16]}...")
    return record


@router.get(
    "/{transcription_id}/content",
    response_model=TranscriptionContentResponse,
    summary="Get transcription content from blob storage",
    description="Retrieve the full transcription content JSON from blob storage.",
    responses={
        404: {"description": "Transcription or content not found"},
        503: {"description": "Blob storage unavailable"},
    },
)
async def get_transcription_content(
    transcription_id: Annotated[str, Path(description="Transcription ID (folder path)")],
    user: AuthenticatedUser = Depends(get_current_user_azure),
    storage: BlobStorageClient = Depends(get_blob_storage),
) -> TranscriptionContentResponse:
    """Get the full transcription content from blob storage.

    Args:
        transcription_id: Transcription ID (folder path)
        user: Authenticated user from Entra ID token
        storage: Blob storage client

    Returns:
        TranscriptionContentResponse with full content from blob

    Raises:
        HTTPException: 404 if transcription or content not found
    """
    # Construct folder path - the ID should already include user_id prefix
    folder_path = transcription_id
    if not folder_path.startswith(f"{user.oid}/"):
        folder_path = f"{user.oid}/{transcription_id}"

    try:
        # Download transcript content from blob storage
        content_json = await storage.download_transcription_from_user_path(folder_path)

        # Parse the JSON into TranscriptionContent model
        content_data = json.loads(content_json)
        content = TranscriptionContent(**content_data)

        return TranscriptionContentResponse(
            transcript_id=transcription_id,
            content=content,
            blob_url=None,  # SAS URL generation removed for simplicity
        )

    except BlobNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transcription content not found in blob storage for ID '{transcription_id}'",
        )
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse transcription content JSON: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to parse transcription content",
        )
    except Exception as e:
        logger.error(f"Failed to retrieve transcription content: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to retrieve transcription content: {e}",
        )
