"""Transcription history API endpoints.

This module provides endpoints for managing stored transcriptions.
"""

import json
import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from auth import AuthenticatedUser
from auth.dependencies import get_current_user_azure
from db import get_cosmos_client
from db.cosmos import CosmosClient
from models.transcription import (
    TranscriptionContent,
    TranscriptionContentResponse,
    TranscriptionListResponse,
    TranscriptionRecord,
)
from storage import BlobNotFoundError, get_storage_client
from storage.blob import BlobStorageClient

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/transcriptions", tags=["Transcription History"])


def get_db() -> CosmosClient:
    """FastAPI dependency for getting database client."""
    return get_cosmos_client()


def get_blob_storage() -> BlobStorageClient:
    """FastAPI dependency for getting the Blob Storage client."""
    return get_storage_client()


@router.get(
    "",
    response_model=TranscriptionListResponse,
    summary="List user's transcriptions",
    description="Get a paginated list of the authenticated user's transcriptions.",
)
async def list_transcriptions(
    page: Annotated[int, Query(description="Page number", ge=1)] = 1,
    per_page: Annotated[int, Query(description="Results per page", ge=1, le=100)] = 20,
    user: AuthenticatedUser = Depends(get_current_user_azure),
    db: CosmosClient = Depends(get_db),
) -> TranscriptionListResponse:
    """List transcriptions for the authenticated user.

    Args:
        page: Page number (1-indexed)
        per_page: Number of results per page
        user: Authenticated user from Entra ID token
        db: Cosmos DB client

    Returns:
        TranscriptionListResponse with paginated transcriptions
    """
    transcriptions, total = await db.list_transcriptions(user.oid, page, per_page)

    return TranscriptionListResponse(
        transcriptions=transcriptions,
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get(
    "/{transcription_id}",
    response_model=TranscriptionRecord,
    summary="Get a specific transcription",
    description="Get details of a specific transcription by ID.",
    responses={
        404: {"description": "Transcription not found"},
    },
)
async def get_transcription(
    transcription_id: Annotated[str, Path(description="Transcription ID")],
    user: AuthenticatedUser = Depends(get_current_user_azure),
    db: CosmosClient = Depends(get_db),
) -> TranscriptionRecord:
    """Get a specific transcription by ID.

    Args:
        transcription_id: Transcription ID
        user: Authenticated user from Entra ID token
        db: Cosmos DB client

    Returns:
        TranscriptionRecord if found

    Raises:
        HTTPException: 404 if transcription not found
    """
    transcription = await db.get_transcription(user.oid, transcription_id)

    if transcription is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transcription with ID '{transcription_id}' not found",
        )

    return transcription


@router.delete(
    "/{transcription_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a transcription",
    description="Delete a specific transcription by ID.",
    responses={
        404: {"description": "Transcription not found"},
    },
)
async def delete_transcription(
    transcription_id: Annotated[str, Path(description="Transcription ID")],
    user: AuthenticatedUser = Depends(get_current_user_azure),
    db: CosmosClient = Depends(get_db),
) -> None:
    """Delete a specific transcription by ID.

    Args:
        transcription_id: Transcription ID
        user: Authenticated user from Entra ID token
        db: Cosmos DB client

    Raises:
        HTTPException: 404 if transcription not found
    """
    deleted = await db.delete_transcription(user.oid, transcription_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transcription with ID '{transcription_id}' not found",
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
    db: CosmosClient = Depends(get_db),
) -> TranscriptionRecord:
    """Get a transcription by its audio file hash.

    This endpoint enables client-side cache lookups by computing the audio
    hash before uploading, allowing detection of duplicate uploads.

    Args:
        audio_hash: SHA256 hash of the audio file content
        user: Authenticated user from Entra ID token
        db: Cosmos DB client

    Returns:
        TranscriptionRecord if found

    Raises:
        HTTPException: 404 if no transcription with this hash found
    """
    logger.debug(f"Looking up transcription by audio hash: {audio_hash[:16]}...")

    transcription = await db.get_transcription_by_audio_hash(user.oid, audio_hash)

    if transcription is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No transcription found with audio hash '{audio_hash[:16]}...'",
        )

    logger.info(f"Found transcription {transcription.id} for audio hash {audio_hash[:16]}...")
    return transcription


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
    transcription_id: Annotated[str, Path(description="Transcription ID")],
    user: AuthenticatedUser = Depends(get_current_user_azure),
    db: CosmosClient = Depends(get_db),
    storage: BlobStorageClient = Depends(get_blob_storage),
) -> TranscriptionContentResponse:
    """Get the full transcription content from blob storage.

    Args:
        transcription_id: Transcription ID
        user: Authenticated user from Entra ID token
        db: Cosmos DB client
        storage: Blob storage client

    Returns:
        TranscriptionContentResponse with full content from blob

    Raises:
        HTTPException: 404 if transcription or content not found
    """
    # First verify the transcription exists and belongs to this user
    transcription = await db.get_transcription(user.oid, transcription_id)

    if transcription is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transcription with ID '{transcription_id}' not found",
        )

    # Check if blob storage URL exists
    if not transcription.blob_storage_url:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transcription content not available in blob storage for ID '{transcription_id}'",
        )

    try:
        # Download content from blob storage
        content_json = await storage.download_transcription_json(
            user_id=user.oid,
            transcription_id=transcription_id,
        )

        # Parse the JSON into TranscriptionContent model
        content_data = json.loads(content_json)
        content = TranscriptionContent(**content_data)

        # Generate SAS URL for direct access
        sas_url = storage.get_transcription_sas_url(
            user_id=user.oid,
            transcription_id=transcription_id,
            expiry_hours=1,
        )

        return TranscriptionContentResponse(
            transcript_id=transcription_id,
            content=content,
            blob_url=sas_url,
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
