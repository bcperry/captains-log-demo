"""Tests for the Cosmos DB client module."""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from config.settings import Settings
from db.cosmos import (
    CosmosClient,
    CosmosDBError,
    DocumentNotFoundError,
    InMemoryCosmosClient,
    clear_in_memory_storage,
    get_cosmos_client,
)
from models.user import UserProfile, UserProfileCreate


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings for testing."""
    return Settings(
        azure_cosmos_endpoint="https://test.documents.azure.com:443/",
        azure_cosmos_key="test-key",
        azure_cosmos_database="test-db",
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def mock_settings_unconfigured() -> Settings:
    """Create mock settings without Cosmos configuration."""
    return Settings(
        azure_cosmos_endpoint=None,
        azure_cosmos_key=None,
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def user_data() -> UserProfileCreate:
    """Create test user data."""
    return UserProfileCreate(
        oid="test-user-123",
        email="test@example.com",
        name="Test User",
        preferred_username="testuser@example.com",
        tenant_id="test-tenant-id",
    )


@pytest.fixture(autouse=True)
def cleanup_storage() -> None:
    """Clear in-memory storage before each test."""
    clear_in_memory_storage()


class TestCosmosDBExceptions:
    """Tests for Cosmos DB exception classes."""

    def test_cosmos_db_error(self) -> None:
        """Test CosmosDBError base exception."""
        error = CosmosDBError("Test error")
        assert str(error) == "Test error"

    def test_document_not_found_error(self) -> None:
        """Test DocumentNotFoundError exception."""
        error = DocumentNotFoundError("Document not found")
        assert str(error) == "Document not found"
        assert isinstance(error, CosmosDBError)


class TestCosmosClient:
    """Tests for CosmosClient class."""

    def test_initialization(self, mock_settings: Settings) -> None:
        """Test client initialization with settings."""
        client = CosmosClient(mock_settings)
        assert client.endpoint == "https://test.documents.azure.com:443/"
        assert client.key == "test-key"
        assert client.database == "test-db"

    def test_initialization_without_settings(self) -> None:
        """Test client initialization uses default settings."""
        with patch("db.cosmos.get_settings") as mock_get_settings:
            mock_get_settings.return_value = MagicMock()
            mock_get_settings.return_value.azure_cosmos_endpoint = "https://default.com/"
            mock_get_settings.return_value.azure_cosmos_key = "default-key"
            mock_get_settings.return_value.azure_cosmos_database = "default-db"
            client = CosmosClient()
            mock_get_settings.assert_called_once()

    def test_endpoint_property_empty(self, mock_settings_unconfigured: Settings) -> None:
        """Test endpoint returns empty string when not configured."""
        client = CosmosClient(mock_settings_unconfigured)
        assert client.endpoint == ""

    def test_key_property_empty(self, mock_settings_unconfigured: Settings) -> None:
        """Test key returns empty string when not configured."""
        client = CosmosClient(mock_settings_unconfigured)
        assert client.key == ""

    def test_is_configured_true(self, mock_settings: Settings) -> None:
        """Test is_configured returns True when configured."""
        client = CosmosClient(mock_settings)
        assert client.is_configured() is True

    def test_is_configured_false(self, mock_settings_unconfigured: Settings) -> None:
        """Test is_configured returns False when not configured."""
        client = CosmosClient(mock_settings_unconfigured)
        assert client.is_configured() is False

    @pytest.mark.asyncio
    async def test_get_user_profile_unconfigured(
        self, mock_settings_unconfigured: Settings
    ) -> None:
        """Test get_user_profile returns None when not configured."""
        client = CosmosClient(mock_settings_unconfigured)
        result = await client.get_user_profile("user-123")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_user_profile_configured_not_found(
        self, mock_settings: Settings
    ) -> None:
        """Test get_user_profile returns None when user not found."""
        client = CosmosClient(mock_settings)
        result = await client.get_user_profile("user-123")
        # Currently returns None as the actual Cosmos DB implementation is pending
        assert result is None

    @pytest.mark.asyncio
    async def test_create_user_profile(
        self, mock_settings: Settings, user_data: UserProfileCreate
    ) -> None:
        """Test create_user_profile creates a profile."""
        client = CosmosClient(mock_settings)
        profile = await client.create_user_profile(user_data)

        assert profile.id == user_data.oid
        assert profile.email == user_data.email
        assert profile.name == user_data.name
        assert profile.preferred_username == user_data.preferred_username
        assert profile.tenant_id == user_data.tenant_id
        assert profile.created_at is not None
        assert profile.updated_at is not None
        assert profile.last_login_at is not None

    @pytest.mark.asyncio
    async def test_update_last_login_not_found(self, mock_settings: Settings) -> None:
        """Test update_last_login returns None when user not found."""
        client = CosmosClient(mock_settings)
        result = await client.update_last_login("nonexistent-user")
        # Since get_user_profile returns None, update returns None
        assert result is None

    @pytest.mark.asyncio
    async def test_get_or_create_creates_new_profile(
        self, mock_settings: Settings, user_data: UserProfileCreate
    ) -> None:
        """Test get_or_create_user_profile creates new profile."""
        client = CosmosClient(mock_settings)
        profile, created = await client.get_or_create_user_profile(user_data)

        assert created is True
        assert profile.id == user_data.oid
        assert profile.email == user_data.email

    @pytest.mark.asyncio
    async def test_close_client(self, mock_settings: Settings) -> None:
        """Test closing the HTTP client."""
        client = CosmosClient(mock_settings)
        # Get client first
        http_client = await client._get_client()
        assert http_client is not None

        # Close it
        await client.close()
        assert client._http_client is None

    @pytest.mark.asyncio
    async def test_get_client_creates_new(self, mock_settings: Settings) -> None:
        """Test _get_client creates new client when none exists."""
        client = CosmosClient(mock_settings)
        assert client._http_client is None

        http_client = await client._get_client()
        assert http_client is not None
        assert client._http_client is not None

        # Clean up
        await client.close()

    @pytest.mark.asyncio
    async def test_get_client_reuses_existing(self, mock_settings: Settings) -> None:
        """Test _get_client reuses existing client."""
        client = CosmosClient(mock_settings)

        http_client1 = await client._get_client()
        http_client2 = await client._get_client()

        assert http_client1 is http_client2

        # Clean up
        await client.close()


class TestInMemoryCosmosClient:
    """Tests for InMemoryCosmosClient class."""

    @pytest.mark.asyncio
    async def test_create_user_profile(self, user_data: UserProfileCreate) -> None:
        """Test creating a user profile in memory."""
        client = InMemoryCosmosClient()
        profile = await client.create_user_profile(user_data)

        assert profile.id == user_data.oid
        assert profile.email == user_data.email
        assert profile.name == user_data.name

    @pytest.mark.asyncio
    async def test_get_user_profile_exists(self, user_data: UserProfileCreate) -> None:
        """Test retrieving an existing user profile."""
        client = InMemoryCosmosClient()
        await client.create_user_profile(user_data)

        profile = await client.get_user_profile(user_data.oid)
        assert profile is not None
        assert profile.id == user_data.oid
        assert profile.email == user_data.email

    @pytest.mark.asyncio
    async def test_get_user_profile_not_exists(self) -> None:
        """Test retrieving a non-existent user profile."""
        client = InMemoryCosmosClient()
        profile = await client.get_user_profile("nonexistent-user")
        assert profile is None

    @pytest.mark.asyncio
    async def test_update_last_login_exists(
        self, user_data: UserProfileCreate
    ) -> None:
        """Test updating last login for existing user."""
        client = InMemoryCosmosClient()
        await client.create_user_profile(user_data)

        original_profile = await client.get_user_profile(user_data.oid)
        assert original_profile is not None
        original_login = original_profile.last_login_at

        # Update last login
        updated_profile = await client.update_last_login(user_data.oid)
        assert updated_profile is not None
        assert updated_profile.last_login_at >= original_login

    @pytest.mark.asyncio
    async def test_update_last_login_not_exists(self) -> None:
        """Test updating last login for non-existent user."""
        client = InMemoryCosmosClient()
        result = await client.update_last_login("nonexistent-user")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_or_create_creates_new(
        self, user_data: UserProfileCreate
    ) -> None:
        """Test get_or_create creates new profile."""
        client = InMemoryCosmosClient()
        profile, created = await client.get_or_create_user_profile(user_data)

        assert created is True
        assert profile.id == user_data.oid

    @pytest.mark.asyncio
    async def test_get_or_create_returns_existing(
        self, user_data: UserProfileCreate
    ) -> None:
        """Test get_or_create returns existing profile."""
        client = InMemoryCosmosClient()

        # Create first
        await client.create_user_profile(user_data)

        # Get or create should return existing
        profile, created = await client.get_or_create_user_profile(user_data)

        assert created is False
        assert profile.id == user_data.oid

    @pytest.mark.asyncio
    async def test_get_or_create_updates_last_login(
        self, user_data: UserProfileCreate
    ) -> None:
        """Test get_or_create updates last login for existing profile."""
        client = InMemoryCosmosClient()

        # Create first
        original_profile = await client.create_user_profile(user_data)
        original_login = original_profile.last_login_at

        # Get or create should update last login
        profile, created = await client.get_or_create_user_profile(user_data)

        assert created is False
        assert profile.last_login_at >= original_login


class TestClearInMemoryStorage:
    """Tests for clear_in_memory_storage function."""

    @pytest.mark.asyncio
    async def test_clears_storage(self, user_data: UserProfileCreate) -> None:
        """Test that clear_in_memory_storage clears all data."""
        client = InMemoryCosmosClient()
        await client.create_user_profile(user_data)

        # Verify profile exists
        assert await client.get_user_profile(user_data.oid) is not None

        # Clear storage
        clear_in_memory_storage()

        # Verify profile is gone
        assert await client.get_user_profile(user_data.oid) is None


class TestGetCosmosClient:
    """Tests for get_cosmos_client function."""

    def test_returns_in_memory_when_not_configured(self) -> None:
        """Test returns InMemoryCosmosClient when Cosmos is not configured."""
        with patch("db.cosmos.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.is_cosmos_configured.return_value = False
            mock_get_settings.return_value = mock_settings

            # Clear cache to force new call
            get_cosmos_client.cache_clear()

            client = get_cosmos_client()
            assert isinstance(client, InMemoryCosmosClient)

    def test_returns_cosmos_client_when_configured(self) -> None:
        """Test returns CosmosClient when Cosmos is configured."""
        with patch("db.cosmos.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.is_cosmos_configured.return_value = True
            mock_settings.azure_cosmos_endpoint = "https://test.documents.azure.com:443/"
            mock_settings.azure_cosmos_key = "test-key"
            mock_settings.azure_cosmos_database = "test-db"
            mock_get_settings.return_value = mock_settings

            # Clear cache to force new call
            get_cosmos_client.cache_clear()

            client = get_cosmos_client()
            assert isinstance(client, CosmosClient)

    def test_client_is_cached(self) -> None:
        """Test that get_cosmos_client returns cached instance."""
        with patch("db.cosmos.get_settings") as mock_get_settings:
            mock_settings = MagicMock()
            mock_settings.is_cosmos_configured.return_value = False
            mock_get_settings.return_value = mock_settings

            # Clear cache first
            get_cosmos_client.cache_clear()

            client1 = get_cosmos_client()
            client2 = get_cosmos_client()

            assert client1 is client2
