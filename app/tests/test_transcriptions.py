"""Tests for the transcription history API endpoints."""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.transcriptions import get_blob_storage, router
from auth import AuthenticatedUser
from auth.dependencies import get_current_user_azure
from models.transcription import TranscriptionMetadata
from storage.blob import InMemoryBlobClient, clear_in_memory_blobs

# Mock user OID that matches the mock_user fixture
MOCK_USER_OID = "test-user-123"


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
        oid="test-user-123",
        email="test@example.com",
        name="Test User",
        preferred_username="testuser@example.com",
        tenant_id="test-tenant-id",
        roles=["User"],
    )


@pytest.fixture
def mock_blob_storage() -> InMemoryBlobClient:
    """Create an in-memory blob storage client for testing."""
    clear_in_memory_blobs()
    return InMemoryBlobClient()


@pytest.fixture
def client(
    app: FastAPI, mock_user: AuthenticatedUser, mock_blob_storage: InMemoryBlobClient
) -> TestClient:
    """Create a test client with mocked dependencies."""
    app.dependency_overrides[get_current_user_azure] = lambda: mock_user
    app.dependency_overrides[get_blob_storage] = lambda: mock_blob_storage
    return TestClient(app)


async def create_test_transcription(
    storage: InMemoryBlobClient,
    user_id: str,
    folder_path: str = "test-user-123/test_20240115_120000",
    text: str = "Test transcription text",
) -> TranscriptionMetadata:
    """Helper to create a test transcription in blob storage."""
    from datetime import UTC, datetime
    
    metadata = TranscriptionMetadata(
        id=folder_path,
        user_id=user_id,
        filename="test.wav",
        upload_time=datetime.now(UTC),
        duration_ms=5000,
        speaker_count=1,
        language="en-US",
        audio_format="wav",
        file_size_bytes=1024,
        folder_path=folder_path,
        text=text,
        has_diarization=False,
    )
    await storage.save_metadata(user_id, folder_path, metadata.model_dump_json())
    return metadata


class TestListTranscriptions:
    """Tests for GET /transcriptions endpoint."""

    @pytest.mark.asyncio
    async def test_returns_empty_list_initially(
        self, client: TestClient, mock_blob_storage: InMemoryBlobClient
    ) -> None:
        """Test that endpoint returns empty list when no transcriptions exist."""
        response = client.get("/transcriptions")

        assert response.status_code == 200
        data = response.json()
        assert data["transcriptions"] == []
        assert data["total"] == 0
        assert data["page"] == 1
        assert data["per_page"] == 20

    @pytest.mark.asyncio
    async def test_returns_user_transcriptions(
        self,
        client: TestClient,
        mock_blob_storage: InMemoryBlobClient,
        mock_user: AuthenticatedUser,
    ) -> None:
        """Test that endpoint returns user's transcriptions."""
        await create_test_transcription(mock_blob_storage, mock_user.oid, f"{mock_user.oid}/t1_20240115_120001", "First")
        await create_test_transcription(mock_blob_storage, mock_user.oid, f"{mock_user.oid}/t2_20240115_120002", "Second")

        response = client.get("/transcriptions")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["transcriptions"]) == 2
        assert data["total"] == 2
        assert len(data["transcriptions"]) == 2


class TestDeleteTranscription:
    """Tests for DELETE /transcriptions/{transcription_id} endpoint."""

    @pytest.mark.asyncio
    async def test_delete_transcription_success(
        self,
        client: TestClient,
        mock_blob_storage: InMemoryBlobClient,
        mock_user: AuthenticatedUser,
    ) -> None:
        """Test successful deletion of a transcription."""
        folder_path = f"{mock_user.oid}/test_20240115_120000"
        await create_test_transcription(mock_blob_storage, mock_user.oid, folder_path)

        response = client.delete(f"/transcriptions/{folder_path}")

        assert response.status_code == 204

        # Verify transcription is deleted
        get_response = client.get(f"/transcriptions/{folder_path}")
        assert get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_transcription_not_found(
        self,
        client: TestClient,
        mock_user: AuthenticatedUser,
    ) -> None:
        """Test deletion of non-existent transcription returns 404."""
        # Use user's own folder to avoid 403
        response = client.delete(f"/transcriptions/{mock_user.oid}/nonexistent_folder")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_other_users_transcription_forbidden(
        self,
        app: FastAPI,
        mock_blob_storage: InMemoryBlobClient,
    ) -> None:
        """Test that deleting another user's transcription returns 403."""
        # Create transcription for different user
        other_user_id = "other-user-456"
        folder_path = f"{other_user_id}/test_20240115_120000"
        await create_test_transcription(mock_blob_storage, other_user_id, folder_path)

        # Try to delete as different user
        current_user = AuthenticatedUser(
            oid="test-user-123",
            email="test@example.com",
            name="Test User",
            preferred_username="testuser@example.com",
            tenant_id="test-tenant-id",
            roles=["User"],
        )
        app.dependency_overrides[get_current_user_azure] = lambda: current_user
        app.dependency_overrides[get_blob_storage] = lambda: mock_blob_storage
        test_client = TestClient(app)

        response = test_client.delete(f"/transcriptions/{folder_path}")

        assert response.status_code == 403
        assert "Not authorized" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_delete_transcription_removes_all_files(
        self,
        client: TestClient,
        mock_blob_storage: InMemoryBlobClient,
        mock_user: AuthenticatedUser,
    ) -> None:
        """Test that deletion removes all files in transcription folder."""
        folder_path = f"{mock_user.oid}/test_20240115_120000"

        # Create transcription with metadata, transcript, and analysis
        await create_test_transcription(mock_blob_storage, mock_user.oid, folder_path)
        transcript_content = json.dumps({"text": "Test transcript", "words": []})
        await mock_blob_storage.upload_transcription_with_user_path(
            mock_user.oid, folder_path, transcript_content
        )
        analysis_content = json.dumps({"summary": "Test summary"})
        await mock_blob_storage.save_analysis_json(
            mock_user.oid, folder_path, analysis_content
        )

        # Delete the transcription
        response = client.delete(f"/transcriptions/{folder_path}")

        assert response.status_code == 204

        # Verify all related endpoints return 404
        assert client.get(f"/transcriptions/{folder_path}").status_code == 404
        assert client.get(f"/transcriptions/{folder_path}/content").status_code == 404
        assert client.get(f"/transcriptions/{folder_path}/analysis").status_code == 404


