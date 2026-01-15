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

