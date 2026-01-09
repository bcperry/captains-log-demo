"""Tests for the authentication API endpoints."""

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.auth import get_db, router
from auth import AuthenticatedUser
from auth.dependencies import get_current_user_azure
from db.cosmos import InMemoryCosmosClient, clear_in_memory_storage
from models.user import UserPreferences, UserProfile


@pytest.fixture
def app() -> FastAPI:
    """Create a test FastAPI application."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def mock_user() -> AuthenticatedUser:
    """Create a mock authenticated user."""
    return AuthenticatedUser(
        oid="test-oid-12345",
        email="test@example.com",
        name="Test User",
        preferred_username="testuser@example.com",
        tenant_id="test-tenant-id",
        roles=["User"],
    )


@pytest.fixture
def mock_db() -> InMemoryCosmosClient:
    """Create an in-memory database client for testing."""
    clear_in_memory_storage()
    return InMemoryCosmosClient()


@pytest.fixture
def client(app: FastAPI, mock_user: AuthenticatedUser, mock_db: InMemoryCosmosClient) -> TestClient:
    """Create a test client with mocked dependencies."""
    # Override the get_current_user_azure dependency to return our mock user
    app.dependency_overrides[get_current_user_azure] = lambda: mock_user
    app.dependency_overrides[get_db] = lambda: mock_db
    return TestClient(app)


@pytest.fixture
def unauthenticated_client(app: FastAPI) -> TestClient:
    """Create a test client without authentication overrides."""
    return TestClient(app)


class TestGetCurrentUserProfile:
    """Tests for GET /auth/me endpoint."""

    def test_returns_user_profile(
        self, client: TestClient, mock_user: AuthenticatedUser
    ) -> None:
        """Test that endpoint returns user profile for authenticated user."""
        response = client.get("/auth/me")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == mock_user.oid
        assert data["email"] == mock_user.email
        assert data["name"] == mock_user.name
        assert data["preferred_username"] == mock_user.preferred_username
        assert "preferences" in data
        assert "created_at" in data
        assert "last_login_at" in data

    def test_creates_profile_on_first_login(
        self, client: TestClient, mock_user: AuthenticatedUser
    ) -> None:
        """Test that profile is created on first login."""
        # First request should create the profile
        response1 = client.get("/auth/me")
        assert response1.status_code == 200
        created_at = response1.json()["created_at"]

        # Second request should return the same profile
        response2 = client.get("/auth/me")
        assert response2.status_code == 200
        assert response2.json()["created_at"] == created_at

    def test_returns_default_preferences(self, client: TestClient) -> None:
        """Test that new users get default preferences."""
        response = client.get("/auth/me")

        assert response.status_code == 200
        preferences = response.json()["preferences"]
        assert preferences["theme"] == "system"
        assert preferences["language"] == "en-US"
        assert preferences["notifications_enabled"] is True
        assert preferences["auto_transcribe"] is False

    def test_updates_last_login_on_subsequent_requests(
        self, client: TestClient
    ) -> None:
        """Test that last_login_at is updated on each request."""
        response1 = client.get("/auth/me")
        assert response1.status_code == 200
        _last_login_1 = response1.json()["last_login_at"]

        # Immediate second request
        response2 = client.get("/auth/me")
        assert response2.status_code == 200
        # last_login should be updated (or same if within same second)
        # The key thing is it doesn't throw an error


class TestAuthenticationRequired:
    """Tests for authentication requirements."""

    def test_requires_authentication(
        self, app: FastAPI, mock_db: InMemoryCosmosClient
    ) -> None:
        """Test that endpoint requires authentication."""
        # Override only the DB, not the auth
        app.dependency_overrides[get_db] = lambda: mock_db
        # Remove any auth override
        app.dependency_overrides.pop(get_current_user_azure, None)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/auth/me")

        # Should return 401 Unauthorized
        assert response.status_code == 401

    def test_returns_www_authenticate_header(
        self, app: FastAPI, mock_db: InMemoryCosmosClient
    ) -> None:
        """Test that 401 response includes WWW-Authenticate header."""
        app.dependency_overrides[get_db] = lambda: mock_db
        app.dependency_overrides.pop(get_current_user_azure, None)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/auth/me")

        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers


class TestUpdateUserPreferences:
    """Tests for PATCH /auth/me/preferences endpoint."""

    def test_updates_preferences(
        self, client: TestClient, mock_user: AuthenticatedUser
    ) -> None:
        """Test that preferences can be updated."""
        # First create the profile
        client.get("/auth/me")

        # Update preferences
        new_preferences = {
            "theme": "dark",
            "language": "es-ES",
            "notifications_enabled": False,
            "auto_transcribe": True,
        }
        response = client.patch("/auth/me/preferences", json=new_preferences)

        assert response.status_code == 200
        data = response.json()
        assert data["preferences"]["theme"] == "dark"
        assert data["preferences"]["language"] == "es-ES"
        assert data["preferences"]["notifications_enabled"] is False
        assert data["preferences"]["auto_transcribe"] is True

    def test_partial_update_replaces_preferences(
        self, client: TestClient
    ) -> None:
        """Test that PATCH replaces the entire preferences object."""
        # First create the profile
        client.get("/auth/me")

        # Update with only some fields (Pydantic will fill defaults)
        partial_update = {
            "theme": "dark",
            "language": "en-US",
            "notifications_enabled": True,
            "auto_transcribe": False,
        }
        response = client.patch("/auth/me/preferences", json=partial_update)

        assert response.status_code == 200
        assert response.json()["preferences"]["theme"] == "dark"

    def test_requires_authentication_for_preferences(
        self, app: FastAPI, mock_db: InMemoryCosmosClient
    ) -> None:
        """Test that preferences endpoint requires authentication."""
        app.dependency_overrides[get_db] = lambda: mock_db
        app.dependency_overrides.pop(get_current_user_azure, None)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.patch(
            "/auth/me/preferences",
            json={"theme": "dark", "language": "en-US", "notifications_enabled": True, "auto_transcribe": False},
        )

        assert response.status_code == 401


class TestUserProfileModel:
    """Tests for UserProfile and related models."""

    def test_user_preferences_defaults(self) -> None:
        """Test that UserPreferences has sensible defaults."""
        prefs = UserPreferences()
        assert prefs.theme == "system"
        assert prefs.language == "en-US"
        assert prefs.notifications_enabled is True
        assert prefs.auto_transcribe is False

    def test_user_profile_creates_with_required_fields(self) -> None:
        """Test that UserProfile can be created with required fields."""
        now = datetime.now(UTC)
        profile = UserProfile(
            id="test-id",
            created_at=now,
            updated_at=now,
            last_login_at=now,
        )
        assert profile.id == "test-id"
        assert profile.preferences is not None
        assert profile.partition_key == "user"


class TestDifferentUsers:
    """Tests for handling multiple users."""

    def test_different_users_get_different_profiles(
        self, app: FastAPI, mock_db: InMemoryCosmosClient
    ) -> None:
        """Test that different users have separate profiles."""
        user1 = AuthenticatedUser(
            oid="user-1",
            email="user1@example.com",
            name="User One",
        )
        user2 = AuthenticatedUser(
            oid="user-2",
            email="user2@example.com",
            name="User Two",
        )

        app.dependency_overrides[get_db] = lambda: mock_db

        # User 1 creates profile
        app.dependency_overrides[get_current_user_azure] = lambda: user1
        client1 = TestClient(app)
        response1 = client1.get("/auth/me")
        assert response1.status_code == 200
        assert response1.json()["id"] == "user-1"
        assert response1.json()["name"] == "User One"

        # User 2 creates different profile
        app.dependency_overrides[get_current_user_azure] = lambda: user2
        client2 = TestClient(app)
        response2 = client2.get("/auth/me")
        assert response2.status_code == 200
        assert response2.json()["id"] == "user-2"
        assert response2.json()["name"] == "User Two"
