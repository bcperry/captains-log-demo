"""Analysis models for AI-powered transcription analysis.

This module defines Pydantic models for the analyze endpoint request/response.
"""

from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class SpeakerConfidence(str, Enum):
    """Confidence level for AI-identified speaker names."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SpeakerIdentification(BaseModel):
    """Speaker identification mapping from AI analysis."""

    name: str = Field(..., description="Identified or user-edited speaker name")
    confidence: SpeakerConfidence = Field(
        SpeakerConfidence.MEDIUM,
        description="AI confidence in the identification (high/medium/low)",
    )
    ai_identified: bool = Field(
        True,
        description="True if AI identified this name, False if user manually set it",
    )


class Priority(str, Enum):
    """Priority level for action items."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Sentiment(str, Enum):
    """Sentiment classification for transcription."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class ActionItem(BaseModel):
    """An action item extracted from the transcription."""

    task: str = Field(..., description="Description of the action to take")
    assignee: Optional[str] = Field(None, description="Person responsible for the task")
    deadline: Optional[str] = Field(None, description="Due date for the task")
    priority: Priority = Field(Priority.MEDIUM, description="Priority level of the task")


class AnalyzeRequest(BaseModel):
    """Request body for the analyze endpoint."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="The transcription text to analyze",
    )
    diarized_transcript: Optional[str] = Field(
        default=None,
        max_length=100000,
        description="Optional diarized transcript with speaker labels (e.g., 'Speaker 1 [00:00:01]: Hello')",
    )
    folder_path: Optional[str] = Field(
        default=None,
        description="Optional folder path for saving analysis to blob storage (e.g., 'user-guid/recording_20240115_103000')",
    )


class AnalysisResult(BaseModel):
    """Result of AI analysis on a transcription."""

    summary: str = Field(..., description="Executive summary of the transcription")
    keyPoints: List[str] = Field(
        default_factory=list,
        description="Key points extracted from the transcription",
        validation_alias="key_points",
    )
    actionItems: List[ActionItem] = Field(
        default_factory=list,
        description="Action items with assignees and priorities",
        validation_alias="action_items",
    )
    participants: List[str] = Field(
        default_factory=list,
        description="Participants mentioned in the transcription",
    )
    topics: List[str] = Field(
        default_factory=list,
        description="Main topics discussed",
    )
    sentiment: Sentiment = Field(
        Sentiment.NEUTRAL,
        description="Overall sentiment of the transcription",
    )
    confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the analysis (0-1)",
    )
    speakerNames: Dict[str, SpeakerIdentification] = Field(
        default_factory=dict,
        description="Mapping of speaker IDs (e.g., 'Speaker_1') to identified names with confidence",
        validation_alias="speaker_names",
    )

    model_config = {"populate_by_name": True}
