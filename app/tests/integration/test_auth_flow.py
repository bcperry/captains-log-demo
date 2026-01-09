"""Integration tests for authentication flow.

Tests the complete authentication workflow including:
- User profile creation on first login
- Profile retrieval on subsequent requests
- Preference updates
- User isolation
"""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth import AuthenticatedUser
from auth.dependencies import get_current_user_azure
from db.cosmos import InMemoryCosmosClient


class TestAuthenticationFlow:
    """Integration tests for the full authentication flow."""

    def test_first_login_creates_profile(
        self, authenticated_client: TestClient, test_user: AuthenticatedUser
    ) -> None:
        """Test that first login creates a user profile."""
        response = authenticated_client.get("/auth/me")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == test_user.oid
        assert data["email"] == test_user.email
        assert data["name"] == test_user.name

    def test_subsequent_login_returns_existing_profile(
        self, authenticated_client: TestClient, test_user: AuthenticatedUser
    ) -> None:
        """Test that subsequent logins return the existing profile."""
        # First login
        response1 = authenticated_client.get("/auth/me")
        assert response1.status_code == 200
        created_at = response1.json()["created_at"]

        # Second login
        response2 = authenticated_client.get("/auth/me")
        assert response2.status_code == 200

        # Created timestamp should be the same
        assert response2.json()["created_at"] == created_at

    def test_profile_contains_default_preferences(
        self, authenticated_client: TestClient
    ) -> None:
        """Test that new profiles have default preferences."""
        response = authenticated_client.get("/auth/me")

        assert response.status_code == 200
        preferences = response.json()["preferences"]
        assert preferences["theme"] == "system"
        assert preferences["language"] == "en-US"
        assert preferences["notifications_enabled"] is True
        assert preferences["auto_transcribe"] is False

    def test_update_preferences_persists(
        self, authenticated_client: TestClient
    ) -> None:
        """Test that preference updates are persisted."""
        # Create profile
        authenticated_client.get("/auth/me")

        # Update preferences
        new_prefs = {
            "theme": "dark",
            "language": "es-ES",
            "notifications_enabled": False,
            "auto_transcribe": True,
        }
        update_response = authenticated_client.patch(
            "/auth/me/preferences", json=new_prefs
        )
        assert update_response.status_code == 200

        # Verify preferences are returned correctly
        profile = update_response.json()
        assert profile["preferences"]["theme"] == "dark"
        assert profile["preferences"]["language"] == "es-ES"


class TestUserIsolation:
    """Integration tests for user data isolation."""

    def test_different_users_have_separate_profiles(
        self,
        integration_app: FastAPI,
        test_user: AuthenticatedUser,
        another_test_user: AuthenticatedUser,
        mock_db: InMemoryCosmosClient,
    ) -> None:
        """Test that different users have completely separate profiles."""
        from api.auth import get_db

        integration_app.dependency_overrides[get_db] = lambda: mock_db

        # User 1 creates profile
        integration_app.dependency_overrides[get_current_user_azure] = lambda: test_user
        client1 = TestClient(integration_app)
        response1 = client1.get("/auth/me")
        assert response1.status_code == 200
        assert response1.json()["id"] == test_user.oid
        assert response1.json()["email"] == test_user.email

        # User 2 creates profile
        integration_app.dependency_overrides[get_current_user_azure] = lambda: another_test_user
        client2 = TestClient(integration_app)
        response2 = client2.get("/auth/me")
        assert response2.status_code == 200
        assert response2.json()["id"] == another_test_user.oid
        assert response2.json()["email"] == another_test_user.email

        # Verify profiles are different
        assert response1.json()["id"] != response2.json()["id"]

    @pytest.mark.asyncio
    async def test_user_cannot_access_other_user_data(
        self,
        integration_app: FastAPI,
        test_user: AuthenticatedUser,
        another_test_user: AuthenticatedUser,
        mock_db: InMemoryCosmosClient,
    ) -> None:
        """Test that users cannot access each other's transcriptions."""
        from api.transcriptions import get_db
        from models.transcription import TranscriptionRecord

        integration_app.dependency_overrides[get_db] = lambda: mock_db

        # Create transcription for user 1
        record = TranscriptionRecord(
            id="user1-transcription",
            user_id=test_user.oid,
            text="User 1's private transcription",
            language="en-US",
            audio_format="wav",
            file_size_bytes=1024,
        )
        await mock_db.create_transcription(test_user.oid, record)

        # User 2 tries to access it
        integration_app.dependency_overrides[get_current_user_azure] = lambda: another_test_user
        client2 = TestClient(integration_app)
        response = client2.get("/transcriptions/user1-transcription")

        # Should not find it
        assert response.status_code == 404


class TestUnauthenticatedAccess:
    """Integration tests for unauthenticated access attempts."""

    def test_protected_endpoints_require_auth(
        self, integration_app: FastAPI
    ) -> None:
        """Test that protected endpoints return 401 without authentication."""
        client = TestClient(integration_app, raise_server_exceptions=False)

        # Auth endpoints
        response = client.get("/auth/me")
        assert response.status_code == 401

        # Transcription endpoints
        response = client.get("/transcriptions")
        assert response.status_code == 401

    def test_health_endpoints_do_not_require_auth(
        self, integration_app: FastAPI
    ) -> None:
        """Test that health endpoints are accessible without authentication."""
        client = TestClient(integration_app)

        response = client.get("/health")
        assert response.status_code == 200

        response = client.get("/ready")
        assert response.status_code == 200
