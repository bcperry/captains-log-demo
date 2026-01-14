"""OpenAI client for AI-powered transcription analysis.

This module provides a client for Azure OpenAI to analyze transcriptions
and extract summaries, key points, action items, and sentiment.
"""

import json
import logging
from typing import List, Optional

from openai import AzureOpenAI
from pydantic import ValidationError

from config.settings import get_settings
from models.analysis import ActionItem, AnalysisResult, Priority, Sentiment

logger = logging.getLogger(__name__)


ANALYSIS_SYSTEM_PROMPT = """You are an AI assistant that analyzes meeting transcriptions.
Given a transcription, extract and return a JSON object with these fields:

- summary: A concise 2-3 sentence executive summary
- key_points: Array of 3-7 key points discussed (strings)
- action_items: Array of action items, each with:
  - task: Description of what needs to be done
  - assignee: Person responsible (or null if not specified)
  - deadline: Due date if mentioned (or null)
  - priority: "high", "medium", or "low"
- participants: Array of participant names mentioned
- topics: Array of main topics discussed
- sentiment: Overall sentiment - "positive", "neutral", or "negative"
- confidence: Your confidence in this analysis from 0.0 to 1.0

Return ONLY valid JSON, no markdown formatting or explanation."""


class OpenAIClientError(Exception):
    """Base exception for OpenAI client errors."""

    pass


class OpenAINotConfiguredError(OpenAIClientError):
    """Raised when Azure OpenAI is not configured."""

    pass


class OpenAIClient:
    """Client for Azure OpenAI transcription analysis."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        deployment: Optional[str] = None,
        api_version: str = "2024-02-15-preview",
    ):
        """Initialize the OpenAI client.

        Args:
            endpoint: Azure OpenAI endpoint URL
            api_key: Azure OpenAI API key
            deployment: Azure OpenAI deployment name
            api_version: API version to use
        """
        settings = get_settings()

        self.endpoint = endpoint or settings.azure_openai_endpoint
        self.api_key = api_key or settings.azure_openai_key
        self.deployment = deployment or settings.azure_openai_deployment
        self.api_version = api_version or settings.azure_openai_api_version

        self._client: Optional[AzureOpenAI] = None

    def is_configured(self) -> bool:
        """Check if Azure OpenAI is properly configured."""
        return bool(self.endpoint and self.api_key and self.deployment)

    def _get_client(self) -> AzureOpenAI:
        """Get or create the Azure OpenAI client.

        Returns:
            AzureOpenAI client instance

        Raises:
            OpenAINotConfiguredError: If Azure OpenAI is not configured
        """
        if not self.is_configured():
            raise OpenAINotConfiguredError(
                "Azure OpenAI is not configured. Please set AZURE_OPENAI_ENDPOINT, "
                "AZURE_OPENAI_KEY, and AZURE_OPENAI_DEPLOYMENT environment variables."
            )

        if self._client is None:
            self._client = AzureOpenAI(
                azure_endpoint=self.endpoint,  # type: ignore[arg-type]
                api_key=self.api_key,
                api_version=self.api_version,
            )

        return self._client

    def _parse_analysis_response(self, content: str) -> AnalysisResult:
        """Parse the OpenAI response into an AnalysisResult.

        Args:
            content: Raw JSON string from OpenAI

        Returns:
            Parsed AnalysisResult

        Raises:
            OpenAIClientError: If parsing fails
        """
        try:
            # Clean up potential markdown formatting
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            data = json.loads(content)

            # Handle action items with proper enum conversion
            action_items: List[ActionItem] = []
            for item in data.get("action_items", []):
                priority_str = item.get("priority", "medium").lower()
                try:
                    priority = Priority(priority_str)
                except ValueError:
                    priority = Priority.MEDIUM

                action_items.append(
                    ActionItem(
                        task=item.get("task", ""),
                        assignee=item.get("assignee"),
                        deadline=item.get("deadline"),
                        priority=priority,
                    )
                )

            # Parse sentiment
            sentiment_str = data.get("sentiment", "neutral").lower()
            try:
                sentiment = Sentiment(sentiment_str)
            except ValueError:
                sentiment = Sentiment.NEUTRAL

            return AnalysisResult(
                summary=data.get("summary", "No summary available"),
                keyPoints=data.get("key_points", []),
                actionItems=action_items,
                participants=data.get("participants", []),
                topics=data.get("topics", []),
                sentiment=sentiment,
                confidence=float(data.get("confidence", 0.8)),
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response as JSON: {e}")
            raise OpenAIClientError(f"Failed to parse analysis response: {e}") from e
        except ValidationError as e:
            logger.error(f"Failed to validate analysis response: {e}")
            raise OpenAIClientError(f"Invalid analysis response format: {e}") from e

    def analyze_transcription(self, text: str) -> AnalysisResult:
        """Analyze a transcription using Azure OpenAI.

        Args:
            text: The transcription text to analyze

        Returns:
            AnalysisResult with summary, key points, action items, etc.

        Raises:
            OpenAINotConfiguredError: If Azure OpenAI is not configured
            OpenAIClientError: If analysis fails
        """
        client = self._get_client()

        try:
            response = client.chat.completions.create(
                model=self.deployment,  # type: ignore[arg-type]
                messages=[
                    {"role": "system", "content": ANALYSIS_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Analyze this transcription:\n\n{text}"},
                ],
                temperature=0.3,
                max_tokens=2000,
            )

            content = response.choices[0].message.content
            if not content:
                raise OpenAIClientError("Empty response from Azure OpenAI")

            return self._parse_analysis_response(content)

        except OpenAIClientError:
            raise
        except Exception as e:
            logger.error(f"Azure OpenAI analysis failed: {e}")
            raise OpenAIClientError(f"Analysis failed: {e}") from e


# Module-level client instance
_openai_client: Optional[OpenAIClient] = None


def get_openai_client() -> OpenAIClient:
    """Get or create the global OpenAI client instance.

    Returns:
        OpenAIClient instance
    """
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAIClient()
    return _openai_client
