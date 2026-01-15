"""Azure Cosmos DB client for data storage.

This module provides async operations for Cosmos DB using the Azure SDK.
"""

import uuid
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, Optional

import httpx

from config.settings import Settings, get_settings
from models.transcription import TranscriptionRecord
from models.user import UserProfile, UserProfileCreate


class CosmosDBError(Exception):
    """Base exception for Cosmos DB operations."""

    pass


class DocumentNotFoundError(CosmosDBError):
    """Raised when a document is not found."""

    pass


class CosmosClient:
    """Async Cosmos DB client using REST API.

    Uses httpx for async HTTP operations with Azure Cosmos DB REST API.
    """

    USERS_CONTAINER = "users"
    TRANSCRIPTIONS_CONTAINER = "transcriptions"

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize the Cosmos DB client.

        Args:
            settings: Application settings. If None, uses default settings.
        """
        self._settings = settings or get_settings()
        self._http_client: Optional[httpx.AsyncClient] = None

    @property
    def endpoint(self) -> str:
        """Get Cosmos DB endpoint URL."""
        return self._settings.azure_cosmos_endpoint or ""

    @property
    def key(self) -> str:
        """Get Cosmos DB key."""
        return self._settings.azure_cosmos_key or ""

    @property
    def database(self) -> str:
        """Get Cosmos DB database name."""
        return self._settings.azure_cosmos_database

    def is_configured(self) -> bool:
        """Check if Cosmos DB is properly configured."""
        return self._settings.is_cosmos_configured()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile by ID.

        Args:
            user_id: User ID (Azure Entra ID object ID)

        Returns:
            UserProfile if found, None otherwise
        """
        if not self.is_configured():
            return None

        # For simplicity, this implementation uses a dictionary as storage
        # In production, this would call the Cosmos DB REST API
        # The actual implementation will be added when Cosmos DB is set up
        return None

    async def create_user_profile(self, user_data: UserProfileCreate) -> UserProfile:
        """Create a new user profile.

        Args:
            user_data: User data from authentication

        Returns:
            Created UserProfile
        """
        now = datetime.now(UTC)
        profile = UserProfile(
            id=user_data.oid,
            email=user_data.email,
            name=user_data.name,
            preferred_username=user_data.preferred_username,
            tenant_id=user_data.tenant_id,
            created_at=now,
            updated_at=now,
            last_login_at=now,
        )

        # In production, this would persist to Cosmos DB
        # For now, return the created profile
        return profile

    async def update_last_login(self, user_id: str) -> Optional[UserProfile]:
        """Update user's last login timestamp.

        Args:
            user_id: User ID

        Returns:
            Updated UserProfile if found
        """
        profile = await self.get_user_profile(user_id)
        if profile:
            profile.last_login_at = datetime.now(UTC)
            profile.updated_at = datetime.now(UTC)
        return profile

    async def get_or_create_user_profile(
        self, user_data: UserProfileCreate
    ) -> tuple[UserProfile, bool]:
        """Get existing user profile or create new one.

        Args:
            user_data: User data from authentication

        Returns:
            Tuple of (UserProfile, created) where created is True if new profile
        """
        existing = await self.get_user_profile(user_data.oid)
        if existing:
            # Update last login time
            existing.last_login_at = datetime.now(UTC)
            existing.updated_at = datetime.now(UTC)
            return existing, False

        # Create new profile
        profile = await self.create_user_profile(user_data)
        return profile, True

    # Transcription storage methods

    async def create_transcription(
        self, user_id: str, transcription: TranscriptionRecord
    ) -> TranscriptionRecord:
        """Store a new transcription.

        Args:
            user_id: User ID (partition key)
            transcription: Transcription record to store

        Returns:
            Created TranscriptionRecord with generated ID
        """
        if not transcription.id:
            transcription.id = str(uuid.uuid4())
        transcription.user_id = user_id
        # In production, this would persist to Cosmos DB
        return transcription

    async def get_transcription(
        self, user_id: str, transcription_id: str
    ) -> Optional[TranscriptionRecord]:
        """Get a specific transcription by ID.

        Args:
            user_id: User ID (partition key)
            transcription_id: Transcription ID

        Returns:
            TranscriptionRecord if found, None otherwise
        """
        if not self.is_configured():
            return None
        # In production, this would query Cosmos DB
        return None

    async def list_transcriptions(
        self, user_id: str, page: int = 1, per_page: int = 20
    ) -> tuple[list[TranscriptionRecord], int]:
        """List transcriptions for a user.

        Args:
            user_id: User ID (partition key)
            page: Page number (1-indexed)
            per_page: Results per page

        Returns:
            Tuple of (transcriptions, total_count)
        """
        if not self.is_configured():
            return [], 0
        # In production, this would query Cosmos DB with pagination
        return [], 0

    async def delete_transcription(self, user_id: str, transcription_id: str) -> bool:
        """Delete a transcription.

        Args:
            user_id: User ID (partition key)
            transcription_id: Transcription ID

        Returns:
            True if deleted, False if not found
        """
        if not self.is_configured():
            return False
        # In production, this would delete from Cosmos DB
        return False

    async def get_transcription_by_audio_hash(
        self, user_id: str, audio_hash: str
    ) -> Optional[TranscriptionRecord]:
        """Get a transcription by audio file hash (for cache lookup).

        Args:
            user_id: User ID (partition key)
            audio_hash: SHA256 hash of the audio file content

        Returns:
            TranscriptionRecord if found, None otherwise
        """
        if not self.is_configured():
            return None
        # In production, this would query Cosmos DB with:
        # SELECT * FROM c WHERE c.user_id = @user_id AND c.audio_hash = @audio_hash
        return None


# In-memory storage for development/testing (when Cosmos DB is not configured)
_in_memory_profiles: dict[str, dict[str, Any]] = {}
_in_memory_transcriptions: dict[str, dict[str, dict[str, Any]]] = {}  # user_id -> id -> record


class InMemoryCosmosClient(CosmosClient):
    """In-memory implementation of CosmosClient for development and testing."""

    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Get user profile from in-memory storage."""
        if user_id in _in_memory_profiles:
            return UserProfile(**_in_memory_profiles[user_id])
        return None

    async def create_user_profile(self, user_data: UserProfileCreate) -> UserProfile:
        """Create user profile in in-memory storage."""
        now = datetime.now(UTC)
        profile = UserProfile(
            id=user_data.oid,
            email=user_data.email,
            name=user_data.name,
            preferred_username=user_data.preferred_username,
            tenant_id=user_data.tenant_id,
            created_at=now,
            updated_at=now,
            last_login_at=now,
        )
        _in_memory_profiles[user_data.oid] = profile.model_dump()
        return profile

    async def update_last_login(self, user_id: str) -> Optional[UserProfile]:
        """Update last login in in-memory storage."""
        if user_id in _in_memory_profiles:
            _in_memory_profiles[user_id]["last_login_at"] = datetime.now(UTC)
            _in_memory_profiles[user_id]["updated_at"] = datetime.now(UTC)
            return UserProfile(**_in_memory_profiles[user_id])
        return None

    # Transcription methods for in-memory storage

    async def create_transcription(
        self, user_id: str, transcription: TranscriptionRecord
    ) -> TranscriptionRecord:
        """Store a new transcription in memory."""
        if not transcription.id:
            transcription.id = str(uuid.uuid4())
        transcription.user_id = user_id

        if user_id not in _in_memory_transcriptions:
            _in_memory_transcriptions[user_id] = {}

        _in_memory_transcriptions[user_id][transcription.id] = transcription.model_dump(
            mode="json"
        )
        return transcription

    async def get_transcription(
        self, user_id: str, transcription_id: str
    ) -> Optional[TranscriptionRecord]:
        """Get a specific transcription from in-memory storage."""
        if user_id in _in_memory_transcriptions:
            if transcription_id in _in_memory_transcriptions[user_id]:
                return TranscriptionRecord(
                    **_in_memory_transcriptions[user_id][transcription_id]
                )
        return None

    async def list_transcriptions(
        self, user_id: str, page: int = 1, per_page: int = 20
    ) -> tuple[list[TranscriptionRecord], int]:
        """List transcriptions from in-memory storage with pagination."""
        if user_id not in _in_memory_transcriptions:
            return [], 0

        all_records = list(_in_memory_transcriptions[user_id].values())
        # Sort by created_at descending
        all_records.sort(key=lambda x: x.get("created_at", ""), reverse=True)

        total = len(all_records)
        start = (page - 1) * per_page
        end = start + per_page
        paginated = all_records[start:end]

        transcriptions = [TranscriptionRecord(**record) for record in paginated]
        return transcriptions, total

    async def delete_transcription(self, user_id: str, transcription_id: str) -> bool:
        """Delete a transcription from in-memory storage."""
        if user_id in _in_memory_transcriptions:
            if transcription_id in _in_memory_transcriptions[user_id]:
                del _in_memory_transcriptions[user_id][transcription_id]
                return True
        return False

    async def get_transcription_by_audio_hash(
        self, user_id: str, audio_hash: str
    ) -> Optional[TranscriptionRecord]:
        """Get a transcription by audio hash from in-memory storage."""
        if user_id not in _in_memory_transcriptions:
            return None
        
        for record in _in_memory_transcriptions[user_id].values():
            if record.get("audio_hash") == audio_hash:
                return TranscriptionRecord(**record)
        return None


def clear_in_memory_storage() -> None:
    """Clear in-memory storage. Useful for tests."""
    _in_memory_profiles.clear()
    _in_memory_transcriptions.clear()


@lru_cache
def get_cosmos_client() -> CosmosClient:
    """Get cached Cosmos DB client instance.

    Returns InMemoryCosmosClient if Cosmos DB is not configured.
    """
    settings = get_settings()
    if settings.is_cosmos_configured():
        return CosmosClient(settings)
    return InMemoryCosmosClient(settings)
