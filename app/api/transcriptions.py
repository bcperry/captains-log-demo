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
                    has_analysis=meta.get("has_analysis", False),
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
        has_analysis=meta.get("has_analysis", False),
    )

    logger.info(f"Found transcription {record.id} for audio hash {audio_hash[:16]}...")
    return record

@router.get(
    "/{transcription_id:path}/content",
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


@router.get(
    "/{transcription_id:path}/analysis",
    summary="Get analysis for a transcription",
    description="Retrieve the AI analysis results for a specific transcription.",
    responses={
        404: {"description": "Analysis not found"},
        503: {"description": "Blob storage unavailable"},
    },
)
async def get_transcription_analysis(
    transcription_id: Annotated[str, Path(description="Transcription ID (folder path)")],
    user: AuthenticatedUser = Depends(get_current_user_azure),
    storage: BlobStorageClient = Depends(get_blob_storage),
) -> dict:
    """Get the AI analysis results for a transcription.

    Args:
        transcription_id: Transcription ID (folder path)
        user: Authenticated user from Entra ID token
        storage: Blob storage client

    Returns:
        AnalysisResult as dict

    Raises:
        HTTPException: 404 if analysis not found
    """
    folder_path = transcription_id
    if not folder_path.startswith(f"{user.oid}/"):
        folder_path = f"{user.oid}/{transcription_id}"

    try:
        analysis_json = await storage.get_analysis_json(folder_path)
        result: dict = json.loads(analysis_json)
        return result

    except BlobNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis not found for transcription '{transcription_id}'",
        )
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse analysis JSON: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to parse analysis content",
        )
    except Exception as e:
        logger.error(f"Failed to retrieve analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to retrieve analysis: {e}",
        )


@router.post(
    "/{transcription_id:path}/analysis",
    status_code=status.HTTP_201_CREATED,
    summary="Save analysis for a transcription",
    description="Save AI analysis results for a specific transcription.",
    responses={
        404: {"description": "Transcription not found"},
        503: {"description": "Blob storage unavailable"},
    },
)
async def save_transcription_analysis(
    transcription_id: Annotated[str, Path(description="Transcription ID (folder path)")],
    analysis: dict,
    user: AuthenticatedUser = Depends(get_current_user_azure),
    storage: BlobStorageClient = Depends(get_blob_storage),
) -> dict:
    """Save AI analysis results for a transcription.

    Args:
        transcription_id: Transcription ID (folder path)
        analysis: AnalysisResult data
        user: Authenticated user from Entra ID token
        storage: Blob storage client

    Returns:
        Success message with blob URL

    Raises:
        HTTPException: 404 if transcription not found, 503 on storage error
    """
    folder_path = transcription_id
    if not folder_path.startswith(f"{user.oid}/"):
        folder_path = f"{user.oid}/{transcription_id}"

    try:
        # First verify the transcription exists
        await storage.get_metadata(folder_path)

        # Save the analysis
        analysis_json = json.dumps(analysis)
        blob_url = await storage.save_analysis_json(user.oid, folder_path, analysis_json)

        # Update metadata to indicate analysis exists
        try:
            metadata_json = await storage.get_metadata(folder_path)
            metadata = json.loads(metadata_json)
            metadata["has_analysis"] = True
            await storage.save_metadata(user.oid, folder_path, json.dumps(metadata))
        except Exception as e:
            logger.warning(f"Failed to update metadata has_analysis flag: {e}")

        logger.info(f"Saved analysis for transcription {transcription_id}")
        return {"message": "Analysis saved successfully", "blob_url": blob_url}

    except BlobNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transcription with ID '{transcription_id}' not found",
        )
    except Exception as e:
        logger.error(f"Failed to save analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to save analysis: {e}",
        )


from models.analysis import SpeakerConfidence, SpeakerIdentification
from pydantic import BaseModel, Field


class SpeakerNameUpdate(BaseModel):
    """Request body for updating speaker names."""

    speaker_names: dict[str, dict] = Field(
        ...,
        description="Mapping of speaker IDs to SpeakerIdentification objects",
        examples=[{
            "Speaker_1": {"name": "Bob", "confidence": "high", "ai_identified": False},
            "Speaker_2": {"name": "Christine", "confidence": "medium", "ai_identified": False},
        }],
    )


class SpeakerNameResponse(BaseModel):
    """Response for speaker name update."""

    message: str
    speaker_names: dict[str, dict]


@router.put(
    "/{transcription_id:path}/speakers",
    response_model=SpeakerNameResponse,
    summary="Update speaker names for a transcription",
    description="Update or override AI-identified speaker names with user-edited names. Persists to analysis.json.",
    responses={
        404: {"description": "Transcription or analysis not found"},
        503: {"description": "Blob storage unavailable"},
    },
)
async def update_speaker_names(
    transcription_id: Annotated[str, Path(description="Transcription ID (folder path)")],
    update: SpeakerNameUpdate,
    user: AuthenticatedUser = Depends(get_current_user_azure),
    storage: BlobStorageClient = Depends(get_blob_storage),
) -> SpeakerNameResponse:
    """Update speaker names for a transcription.

    Allows users to override AI-identified speaker names with their own edits.
    Updates are persisted to analysis.json in blob storage.

    Args:
        transcription_id: Transcription ID (folder path)
        update: SpeakerNameUpdate with speaker_names mapping
        user: Authenticated user from Entra ID token
        storage: Blob storage client

    Returns:
        SpeakerNameResponse with updated speaker names

    Raises:
        HTTPException: 404 if transcription/analysis not found, 503 on storage error
    """
    folder_path = transcription_id
    if not folder_path.startswith(f"{user.oid}/"):
        folder_path = f"{user.oid}/{transcription_id}"

    try:
        # Get existing analysis
        try:
            analysis_json = await storage.get_analysis_json(folder_path)
            analysis = json.loads(analysis_json)
        except BlobNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Analysis not found for transcription '{transcription_id}'. Run analysis first.",
            )

        # Update speaker_names in the analysis
        # Merge with existing speaker_names, user edits take precedence
        existing_speaker_names = analysis.get("speakerNames", analysis.get("speaker_names", {}))
        
        for speaker_id, speaker_data in update.speaker_names.items():
            # Validate and convert speaker_data
            if isinstance(speaker_data, dict):
                # Mark as user-edited (not AI identified)
                speaker_data["ai_identified"] = False
                existing_speaker_names[speaker_id] = speaker_data

        # Update analysis with new speaker names (use camelCase for consistency)
        analysis["speakerNames"] = existing_speaker_names
        # Also update snake_case for backward compatibility
        analysis["speaker_names"] = existing_speaker_names

        # Save updated analysis back to blob storage
        updated_json = json.dumps(analysis)
        await storage.save_analysis_json(user.oid, folder_path, updated_json)

        logger.info(f"Updated speaker names for transcription {transcription_id}")
        return SpeakerNameResponse(
            message="Speaker names updated successfully",
            speaker_names=existing_speaker_names,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update speaker names: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to update speaker names: {e}",
        )


@router.get(
    "/{transcription_id:path}",
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

        # Construct blob_storage_url from folder_path so frontend knows content exists
        blob_storage_url = None
        if folder_path and storage.is_configured():
            blob_storage_url = f"{folder_path}/transcript.json"

        return TranscriptionRecord(
            id=meta.get("id", folder_path),
            user_id=meta.get("user_id", user.oid),
            text=meta.get("text", ""),
            language=meta.get("language", "en-US"),
            audio_format=meta.get("audio_format", "wav"),
            file_size_bytes=meta.get("file_size_bytes", 0),
            duration_ms=meta.get("duration_ms"),
            folder_path=folder_path,
            blob_storage_url=blob_storage_url,
            created_at=meta.get("upload_time"),
            has_diarization=meta.get("has_diarization", False),
            speaker_count=meta.get("speaker_count"),
            audio_hash=meta.get("audio_hash"),
            has_analysis=meta.get("has_analysis", False),
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
    "/{transcription_id:path}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a transcription",
    description="Delete a specific transcription by ID (folder path). Removes all files in the transcription folder.",
    responses={
        204: {"description": "Transcription deleted successfully"},
        403: {"description": "Not authorized to delete this transcription"},
        404: {"description": "Transcription not found"},
        500: {"description": "Partial deletion failure - some files could not be deleted"},
    },
)
async def delete_transcription(
    transcription_id: Annotated[str, Path(description="Transcription ID (folder path)")],
    user: AuthenticatedUser = Depends(get_current_user_azure),
    storage: BlobStorageClient = Depends(get_blob_storage),
) -> None:
    """Delete a specific transcription by ID.

    Deletes the entire transcription folder including:
    - audio file
    - transcript.json
    - metadata.json
    - analysis.json (if exists)

    Args:
        transcription_id: Transcription ID (folder path)
        user: Authenticated user from Entra ID token
        storage: Blob storage client

    Raises:
        HTTPException: 403 if not authorized, 404 if not found, 500 on partial failure
    """
    # Construct folder path - verify user owns the transcription
    folder_path = transcription_id
    
    # Check if the path already includes user's OID
    if folder_path.startswith(f"{user.oid}/"):
        # User owns this transcription
        pass
    elif "/" in folder_path:
        # Path includes a different user's ID - not authorized
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to delete this transcription",
        )
    else:
        # Just a folder name, prepend user's OID
        folder_path = f"{user.oid}/{transcription_id}"

    try:
        # Verify transcription exists before attempting delete
        try:
            await storage.get_metadata(folder_path)
        except BlobNotFoundError:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Transcription with ID '{transcription_id}' not found",
            )

        # Delete all blobs in the folder
        deleted_blobs, failed_blobs = await storage.delete_transcription_folder(folder_path)

        if failed_blobs:
            logger.error(
                f"Partial deletion failure for {folder_path}. "
                f"Deleted: {deleted_blobs}, Failed: {failed_blobs}"
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Partial deletion failure. {len(failed_blobs)} files could not be deleted.",
            )

        logger.info(f"Successfully deleted transcription {folder_path} ({len(deleted_blobs)} files)")
        # Return 204 No Content on success (no response body)
        return None

    except BlobNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transcription with ID '{transcription_id}' not found",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete transcription {transcription_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete transcription: {e}",
        )
