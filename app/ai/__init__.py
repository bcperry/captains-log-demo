"""OpenAI client for AI-powered transcription analysis.

This module provides a client for Azure OpenAI to analyze transcriptions
and extract summaries, key points, action items, and sentiment.
"""

import json
import logging
from typing import Dict, List, Optional

from openai import AzureOpenAI
from pydantic import ValidationError

from config.settings import get_settings
from models.analysis import (
    ActionItem,
    AnalysisResult,
    Priority,
    Sentiment,
    SpeakerConfidence,
    SpeakerIdentification,
)

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


DIARIZED_ANALYSIS_SYSTEM_PROMPT = """You are an AI assistant that analyzes meeting transcriptions with speaker identification.
The transcript includes speaker labels (e.g., "Speaker 1 [00:01:23]: Hello").

Given a diarized transcription, extract and return a JSON object with these fields:

- summary: A concise 2-3 sentence executive summary that references key speakers and their contributions
- key_points: Array of 3-7 key points discussed, attributing to speakers when relevant (e.g., "Speaker 1 proposed...")
- action_items: Array of action items, each with:
  - task: Description of what needs to be done
  - assignee: Person responsible - use speaker labels if names aren't mentioned (e.g., "Speaker 2")
  - deadline: Due date if mentioned (or null)
  - priority: "high", "medium", or "low"
- participants: Array of speaker labels (e.g., ["Speaker 1", "Speaker 2"]) and any named participants
- topics: Array of main topics discussed
- sentiment: Overall sentiment - "positive", "neutral", or "negative"
- confidence: Your confidence in this analysis from 0.0 to 1.0
- speaker_names: Object mapping speaker labels to identified names. Identify speakers by name if mentioned in the conversation.
  For each speaker, provide:
  - name: The actual name if mentioned (e.g., "Bob", "Dr. Smith"), or leave as the speaker label if not mentioned
  - confidence: "high" if explicitly introduced/named, "medium" if inferred from context, "low" if guessing
  - ai_identified: true (since this is AI identification)
  Example: {"Speaker_1": {"name": "Bob", "confidence": "high", "ai_identified": true}, "Speaker_2": {"name": "Speaker 2", "confidence": "low", "ai_identified": true}}
  IMPORTANT: Only identify names if explicitly mentioned in the conversation. If unsure, leave as the speaker label.

Focus on speaker perspectives and contributions in your analysis.
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

            # Parse speaker names
            speaker_names: Dict[str, SpeakerIdentification] = {}
            raw_speaker_names = data.get("speaker_names", {})
            for speaker_id, speaker_data in raw_speaker_names.items():
                if isinstance(speaker_data, dict):
                    confidence_str = speaker_data.get("confidence", "medium").lower()
                    try:
                        confidence = SpeakerConfidence(confidence_str)
                    except ValueError:
                        confidence = SpeakerConfidence.MEDIUM

                    speaker_names[speaker_id] = SpeakerIdentification(
                        name=speaker_data.get("name", speaker_id),
                        confidence=confidence,
                        ai_identified=speaker_data.get("ai_identified", True),
                    )
                elif isinstance(speaker_data, str):
                    # Handle simple string mapping (backward compatibility)
                    speaker_names[speaker_id] = SpeakerIdentification(
                        name=speaker_data,
                        confidence=SpeakerConfidence.MEDIUM,
                        ai_identified=True,
                    )

            return AnalysisResult(
                summary=data.get("summary", "No summary available"),
                keyPoints=data.get("key_points", []),
                actionItems=action_items,
                participants=data.get("participants", []),
                topics=data.get("topics", []),
                sentiment=sentiment,
                confidence=float(data.get("confidence", 0.8)),
                speakerNames=speaker_names,
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response as JSON: {e}")
            raise OpenAIClientError(f"Failed to parse analysis response: {e}") from e
        except ValidationError as e:
            logger.error(f"Failed to validate analysis response: {e}")
            raise OpenAIClientError(f"Invalid analysis response format: {e}") from e

    def analyze_transcription(self, text: str, diarized_transcript: Optional[str] = None) -> AnalysisResult:
        """Analyze a transcription using Azure OpenAI.

        Args:
            text: The transcription text to analyze
            diarized_transcript: Optional diarized transcript with speaker labels for better analysis

        Returns:
            AnalysisResult with summary, key points, action items, etc.

        Raises:
            OpenAINotConfiguredError: If Azure OpenAI is not configured
            OpenAIClientError: If analysis fails
        """
        client = self._get_client()

        # Use diarized transcript and speaker-aware prompt if available
        if diarized_transcript:
            system_prompt = DIARIZED_ANALYSIS_SYSTEM_PROMPT
            analysis_text = diarized_transcript
        else:
            system_prompt = ANALYSIS_SYSTEM_PROMPT
            analysis_text = text

        try:
            response = client.chat.completions.create(
                model=self.deployment,  # type: ignore[arg-type]
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Analyze this transcription:\n\n{analysis_text}"},
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
