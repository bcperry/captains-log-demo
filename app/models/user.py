"""User profile models for Cosmos DB storage and API responses."""

from datetime import UTC, datetime
from typing import Optional

from pydantic import BaseModel, Field


class UserPreferences(BaseModel):
    """User preferences and settings."""

    theme: str = Field(default="system", description="UI theme preference")
    language: str = Field(default="en-US", description="Preferred language for transcription")
    notifications_enabled: bool = Field(default=True, description="Enable notifications")
    auto_transcribe: bool = Field(default=False, description="Auto-start transcription on upload")


class UserProfile(BaseModel):
    """User profile stored in Cosmos DB."""

    id: str = Field(..., description="User ID (Azure Entra ID object ID)")
    email: Optional[str] = Field(default=None, description="User email address")
    name: Optional[str] = Field(default=None, description="User display name")
    preferred_username: Optional[str] = Field(default=None, description="Preferred username")
    tenant_id: Optional[str] = Field(default=None, description="Azure tenant ID")
    preferences: UserPreferences = Field(default_factory=UserPreferences)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    last_login_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Cosmos DB partition key
    partition_key: str = Field(default="user", description="Partition key for Cosmos DB")


class UserProfileCreate(BaseModel):
    """Data required to create a new user profile."""

    oid: str = Field(..., description="Azure Entra ID object ID")
    email: Optional[str] = None
    name: Optional[str] = None
    preferred_username: Optional[str] = None
    tenant_id: Optional[str] = None


class UserProfileResponse(BaseModel):
    """User profile response for API."""

    id: str = Field(..., description="User ID")
    email: Optional[str] = None
    name: Optional[str] = None
    preferred_username: Optional[str] = None
    preferences: UserPreferences
    created_at: datetime
    last_login_at: datetime
