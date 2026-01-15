"""Analyze API endpoint for AI-powered transcription analysis.

This module provides the /analyze endpoint for analyzing transcription text
using Azure OpenAI to extract summaries, key points, action items, and sentiment.
"""

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

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Analysis"])


def get_openai_service() -> OpenAIClient:
    """FastAPI dependency for getting the OpenAI client.

    Returns:
        OpenAIClient instance
    """
    return get_openai_client()


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

    Args:
        request: AnalyzeRequest with text to analyze
        user: Authenticated user from Entra ID token
        openai_client: Azure OpenAI client

    Returns:
        AnalysisResult with extracted insights

    Raises:
        HTTPException: 503 if Azure OpenAI is unavailable
    """
    logger.info(f"Analyzing transcription for user {user.oid} ({len(request.text)} chars)")

    try:
        result = openai_client.analyze_transcription(request.text, request.diarized_transcript)
        logger.info(f"Analysis complete: {len(result.keyPoints)} key points, {len(result.actionItems)} action items")
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
