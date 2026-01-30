"""Analyze API endpoint for AI-powered transcription analysis.

This module provides the /analyze endpoint for analyzing transcription text
using Azure OpenAI to extract summaries, key points, action items, and sentiment.
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from auth import AuthenticatedUser
from auth.dependencies import get_current_user_azure
from models.analysis import AnalysisResult, AnalyzeRequest
from ai import (
    OpenAIClient,
    OpenAIClientError,
    OpenAINotConfiguredError,
    get_openai_client,
)
from storage.blob import BlobStorageClient, get_storage_client

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Analysis"])


def get_openai_service() -> OpenAIClient:
    """FastAPI dependency for getting the OpenAI client.

    Returns:
        OpenAIClient instance
    """
    return get_openai_client()


def get_blob_storage() -> BlobStorageClient:
    """FastAPI dependency for getting the Blob Storage client."""
    return get_storage_client()


@router.post(
    "/analyze",
    response_model=AnalysisResult,
    summary="Analyze transcription text",
    description="Analyze a transcription using AI to extract summary, key points, action items, and sentiment.",
    responses={
        400: {"description": "Invalid request"},
        401: {"description": "Authentication required"},
        503: {"description": "Azure OpenAI unavailable"},
    },
)
async def analyze_transcription(
    request: AnalyzeRequest,
    user: AuthenticatedUser = Depends(get_current_user_azure),
    openai_client: OpenAIClient = Depends(get_openai_service),
    storage: BlobStorageClient = Depends(get_blob_storage),
) -> AnalysisResult:
    """Analyze transcription text using Azure OpenAI.

    Extracts:
    - Executive summary
    - Key points
    - Action items with assignees and priorities
    - Participants mentioned
    - Main topics
    - Overall sentiment
    - Confidence score

    If folder_path is provided, saves analysis.json to blob storage and
    updates metadata.json with has_analysis: true.

    Args:
        request: AnalyzeRequest with text to analyze and optional folder_path
        user: Authenticated user from Entra ID token
        openai_client: Azure OpenAI client
        storage: Blob storage client

    Returns:
        AnalysisResult with extracted insights

    Raises:
        HTTPException: 503 if Azure OpenAI is unavailable
    """
    logger.info(f"Analyzing transcription for user {user.oid} ({len(request.text)} chars)")

    try:
        result = openai_client.analyze_transcription(request.text, request.diarized_transcript)
        logger.info(f"Analysis complete: {len(result.keyPoints)} key points, {len(result.actionItems)} action items")

        # Save analysis to blob storage if folder_path is provided
        if request.folder_path:
            folder_path = request.folder_path
            # Ensure folder_path includes user prefix
            if not folder_path.startswith(f"{user.oid}/"):
                folder_path = f"{user.oid}/{folder_path}"

            try:
                # Save analysis.json
                analysis_json = result.model_dump_json()
                blob_url = await storage.save_analysis_json(user.oid, folder_path, analysis_json)
                logger.info(f"Saved analysis to blob storage: {blob_url}")

                # Update metadata.json to set has_analysis: true
                try:
                    metadata_json = await storage.get_metadata(folder_path)
                    metadata = json.loads(metadata_json)
                    metadata["has_analysis"] = True
                    await storage.save_metadata(user.oid, folder_path, json.dumps(metadata))
                    logger.info(f"Updated metadata has_analysis flag for {folder_path}")
                except Exception as metadata_err:
                    logger.warning(f"Failed to update metadata has_analysis flag: {metadata_err}")

            except Exception as save_err:
                # Log warning but don't fail the request
                logger.warning(f"Failed to save analysis to blob storage: {save_err}")

        return result

    except OpenAINotConfiguredError as e:
        logger.error(f"Azure OpenAI not configured: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Azure OpenAI is not configured. Analysis service unavailable.",
        ) from e

    except OpenAIClientError as e:
        logger.error(f"Azure OpenAI analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Analysis failed: {e}",
        ) from e

    except Exception as e:
        logger.exception(f"Unexpected error during analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during analysis.",
        ) from e
