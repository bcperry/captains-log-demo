"""Tests for the authentication API endpoints."""

from datetime import UTC, datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.auth import router
from auth import AuthenticatedUser
from auth.dependencies import get_current_user_azure
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
def client(app: FastAPI, mock_user: AuthenticatedUser) -> TestClient:
    """Create a test client with mocked dependencies."""
    # Override the get_current_user_azure dependency to return our mock user
    app.dependency_overrides[get_current_user_azure] = lambda: mock_user
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

    def test_returns_default_preferences(self, client: TestClient) -> None:
        """Test that users get default preferences."""
        response = client.get("/auth/me")

        assert response.status_code == 200
        preferences = response.json()["preferences"]
        assert preferences["theme"] == "system"
        assert preferences["language"] == "en-US"
        assert preferences["notifications_enabled"] is True
        assert preferences["auto_transcribe"] is False


class TestAuthenticationRequired:
    """Tests for authentication requirements."""

    def test_requires_authentication(
        self, app: FastAPI
    ) -> None:
        """Test that endpoint requires authentication."""
        # Remove any auth override
        app.dependency_overrides.pop(get_current_user_azure, None)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/auth/me")

        # Should return 401 Unauthorized
        assert response.status_code == 401


class TestUpdateUserPreferences:
    """Tests for PATCH /auth/me/preferences endpoint."""

    def test_updates_preferences(
        self, client: TestClient, mock_user: AuthenticatedUser
    ) -> None:
        """Test that preferences can be updated."""
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

    def test_requires_authentication_for_preferences(
        self, app: FastAPI
    ) -> None:
        """Test that preferences endpoint requires authentication."""
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
