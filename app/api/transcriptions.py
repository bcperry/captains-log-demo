"""Transcription history API endpoints.

This module provides endpoints for managing stored transcriptions.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status

from auth import AuthenticatedUser
from auth.dependencies import get_current_user_azure
from db import get_cosmos_client
from db.cosmos import CosmosClient
from models.transcription import TranscriptionListResponse, TranscriptionRecord

router = APIRouter(prefix="/transcriptions", tags=["Transcription History"])


def get_db() -> CosmosClient:
    """FastAPI dependency for getting database client."""
    return get_cosmos_client()


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
