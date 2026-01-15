"""Tests for the transcription history API endpoints."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.transcriptions import get_blob_storage, get_db, router
from auth import AuthenticatedUser
from auth.dependencies import get_current_user_azure
from db.cosmos import InMemoryCosmosClient, clear_in_memory_storage
from models.transcription import SpeakerSegment, TranscriptionRecord
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
def mock_db() -> InMemoryCosmosClient:
    """Create an in-memory database client for testing."""
    clear_in_memory_storage()
    return InMemoryCosmosClient()


@pytest.fixture
def mock_blob_storage() -> InMemoryBlobClient:
    """Create an in-memory blob storage client for testing."""
    clear_in_memory_blobs()
    return InMemoryBlobClient()


@pytest.fixture
def client(
    app: FastAPI, mock_user: AuthenticatedUser, mock_db: InMemoryCosmosClient, mock_blob_storage: InMemoryBlobClient
) -> TestClient:
    """Create a test client with mocked dependencies."""
    app.dependency_overrides[get_current_user_azure] = lambda: mock_user
    app.dependency_overrides[get_db] = lambda: mock_db
    app.dependency_overrides[get_blob_storage] = lambda: mock_blob_storage
    return TestClient(app)


@pytest.fixture
def unauthenticated_client(app: FastAPI, mock_db: InMemoryCosmosClient) -> TestClient:
    """Create a test client without authentication."""
    app.dependency_overrides[get_db] = lambda: mock_db
    return TestClient(app, raise_server_exceptions=False)


async def create_test_transcription(
    db: InMemoryCosmosClient,
    user_id: str,
    transcription_id: str = "test-transcription-1",
    text: str = "Test transcription text",
) -> TranscriptionRecord:
    """Helper to create a test transcription."""
    record = TranscriptionRecord(
        id=transcription_id,
        user_id=user_id,
        text=text,
        language="en-US",
        audio_format="wav",
        file_size_bytes=1024,
    )
    return await db.create_transcription(user_id, record)


class TestListTranscriptions:
    """Tests for GET /transcriptions endpoint."""

    @pytest.mark.asyncio
    async def test_returns_empty_list_initially(
        self, client: TestClient, mock_db: InMemoryCosmosClient
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
        mock_db: InMemoryCosmosClient,
        mock_user: AuthenticatedUser,
    ) -> None:
        """Test that endpoint returns user's transcriptions."""
        await create_test_transcription(mock_db, mock_user.oid, "t1", "First")
        await create_test_transcription(mock_db, mock_user.oid, "t2", "Second")

        response = client.get("/transcriptions")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["transcriptions"]) == 2

    @pytest.mark.asyncio
    async def test_pagination(
        self,
        client: TestClient,
        mock_db: InMemoryCosmosClient,
        mock_user: AuthenticatedUser,
    ) -> None:
        """Test pagination works correctly."""
        # Create 5 transcriptions
        for i in range(5):
            await create_test_transcription(
                mock_db, mock_user.oid, f"t{i}", f"Text {i}"
            )

        # Get first page with 2 per page
        response = client.get("/transcriptions?page=1&per_page=2")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert len(data["transcriptions"]) == 2
        assert data["page"] == 1
        assert data["per_page"] == 2

    @pytest.mark.asyncio
    async def test_does_not_return_other_users_transcriptions(
        self,
        client: TestClient,
        mock_db: InMemoryCosmosClient,
        mock_user: AuthenticatedUser,
    ) -> None:
        """Test that only the current user's transcriptions are returned."""
        await create_test_transcription(mock_db, mock_user.oid, "t1", "My transcript")
        await create_test_transcription(mock_db, "other-user", "t2", "Other's transcript")

        response = client.get("/transcriptions")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["transcriptions"][0]["id"] == "t1"

    def test_requires_authentication(self, unauthenticated_client: TestClient) -> None:
        """Test that endpoint requires authentication."""
        response = unauthenticated_client.get("/transcriptions")
        assert response.status_code == 401


class TestGetTranscription:
    """Tests for GET /transcriptions/{id} endpoint."""

    @pytest.mark.asyncio
    async def test_returns_transcription_by_id(
        self,
        client: TestClient,
        mock_db: InMemoryCosmosClient,
        mock_user: AuthenticatedUser,
    ) -> None:
        """Test that endpoint returns specific transcription."""
        await create_test_transcription(mock_db, mock_user.oid, "t1", "Test text")

        response = client.get("/transcriptions/t1")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "t1"
        assert data["text"] == "Test text"

    @pytest.mark.asyncio
    async def test_returns_404_for_nonexistent(
        self, client: TestClient, mock_db: InMemoryCosmosClient
    ) -> None:
        """Test that endpoint returns 404 for non-existent transcription."""
        response = client.get("/transcriptions/nonexistent")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_returns_404_for_other_users_transcription(
        self,
        client: TestClient,
        mock_db: InMemoryCosmosClient,
    ) -> None:
        """Test that endpoint returns 404 for another user's transcription."""
        await create_test_transcription(mock_db, "other-user", "t1", "Other's text")

        response = client.get("/transcriptions/t1")

        assert response.status_code == 404

    def test_requires_authentication(self, unauthenticated_client: TestClient) -> None:
        """Test that endpoint requires authentication."""
        response = unauthenticated_client.get("/transcriptions/t1")
        assert response.status_code == 401


class TestDeleteTranscription:
    """Tests for DELETE /transcriptions/{id} endpoint."""

    @pytest.mark.asyncio
    async def test_deletes_transcription(
        self,
        client: TestClient,
        mock_db: InMemoryCosmosClient,
        mock_user: AuthenticatedUser,
    ) -> None:
        """Test that endpoint deletes transcription."""
        await create_test_transcription(mock_db, mock_user.oid, "t1", "To delete")

        response = client.delete("/transcriptions/t1")

        assert response.status_code == 204

        # Verify it's deleted
        get_response = client.get("/transcriptions/t1")
        assert get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_404_for_nonexistent(
        self, client: TestClient, mock_db: InMemoryCosmosClient
    ) -> None:
        """Test that endpoint returns 404 for non-existent transcription."""
        response = client.delete("/transcriptions/nonexistent")

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_cannot_delete_other_users_transcription(
        self,
        client: TestClient,
        mock_db: InMemoryCosmosClient,
    ) -> None:
        """Test that endpoint returns 404 when trying to delete another user's transcription."""
        await create_test_transcription(mock_db, "other-user", "t1", "Other's text")

        response = client.delete("/transcriptions/t1")

        assert response.status_code == 404

    def test_requires_authentication(self, unauthenticated_client: TestClient) -> None:
        """Test that endpoint requires authentication."""
        response = unauthenticated_client.delete("/transcriptions/t1")
        assert response.status_code == 401


class TestTranscriptionRecordModel:
    """Tests for TranscriptionRecord model."""

    def test_creates_basic_record(self) -> None:
        """Test creating a basic transcription record."""
        record = TranscriptionRecord(
            id="test-id",
            user_id="user-123",
            text="Hello world",
            language="en-US",
            audio_format="wav",
            file_size_bytes=1024,
        )
        assert record.id == "test-id"
        assert record.user_id == "user-123"
        assert record.text == "Hello world"
        assert record.has_diarization is False
        assert record.segments is None

    def test_creates_diarized_record(self) -> None:
        """Test creating a diarized transcription record."""
        segments = [
            SpeakerSegment(
                speaker_id="S1", text="Hello", start_time_ms=0, end_time_ms=500
            ),
            SpeakerSegment(
                speaker_id="S2", text="Hi", start_time_ms=600, end_time_ms=900
            ),
        ]
        record = TranscriptionRecord(
            id="test-id",
            user_id="user-123",
            text="Hello Hi",
            language="en-US",
            audio_format="wav",
            file_size_bytes=1024,
            has_diarization=True,
            speaker_count=2,
            segments=segments,
        )
        assert record.has_diarization is True
        assert record.speaker_count == 2
        assert record.segments is not None and len(record.segments) == 2


class TestGetTranscriptionByHash:
    """Tests for GET /transcriptions/by-hash/{audio_hash} endpoint."""

    @pytest.mark.asyncio
    async def test_returns_transcription_by_hash(
        self,
        client: TestClient,
        mock_db: InMemoryCosmosClient,
        mock_user: AuthenticatedUser,
    ) -> None:
        """Test that endpoint returns transcription matching audio hash."""
        # Create transcription with audio_hash
        record = TranscriptionRecord(
            id="t1",
            user_id=mock_user.oid,
            text="Test text",
            language="en-US",
            audio_format="wav",
            file_size_bytes=1024,
            audio_hash="abc123def456789",
        )
        await mock_db.create_transcription(mock_user.oid, record)

        response = client.get("/transcriptions/by-hash/abc123def456789")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "t1"
        assert data["audio_hash"] == "abc123def456789"

    @pytest.mark.asyncio
    async def test_returns_404_for_nonexistent_hash(
        self, client: TestClient, mock_db: InMemoryCosmosClient
    ) -> None:
        """Test that endpoint returns 404 for non-existent hash."""
        response = client.get("/transcriptions/by-hash/nonexistenthash123")

        assert response.status_code == 404
        assert "no transcription found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_returns_404_for_other_users_hash(
        self,
        client: TestClient,
        mock_db: InMemoryCosmosClient,
    ) -> None:
        """Test that endpoint returns 404 for another user's transcription hash."""
        # Create transcription with audio_hash for different user
        record = TranscriptionRecord(
            id="t1",
            user_id="other-user",
            text="Other's text",
            language="en-US",
            audio_format="wav",
            file_size_bytes=1024,
            audio_hash="otherhash123456",
        )
        await mock_db.create_transcription("other-user", record)

        response = client.get("/transcriptions/by-hash/otherhash123456")

        assert response.status_code == 404

    def test_requires_authentication(self, unauthenticated_client: TestClient) -> None:
        """Test that endpoint requires authentication."""
        response = unauthenticated_client.get("/transcriptions/by-hash/somehash")
        assert response.status_code == 401


class TestTranscriptionContent:
    """Tests for GET /transcriptions/{id}/content endpoint."""

    def test_returns_content_for_transcription(
        self, client: TestClient
    ) -> None:
        """Test retrieval of transcription content from blob storage."""
        import json

        from db.cosmos import _in_memory_transcriptions
        from storage.blob import _in_memory_transcription_blobs

        # Create a transcription record with blob_storage_url
        transcription_id = "test-content-id"
        user_id = MOCK_USER_OID
        content_json = {
            "transcript_id": transcription_id,
            "user_id": user_id,
            "filename": "test.wav",
            "upload_date": "2024-01-15T10:30:00Z",
            "duration": 60.0,
            "speaker_segments": [
                {"speaker_id": "Speaker_1", "start_time": 0.0, "end_time": 5.0, "text": "Hello world"}
            ],
            "full_text": "Hello world",
            "language": "en-US",
            "processing_time_ms": 1234,
        }

        # Store in in-memory transcription blobs
        blob_name = f"{user_id}/{transcription_id}.json"
        _in_memory_transcription_blobs[blob_name] = json.dumps(content_json)

        # Store transcription record in Cosmos
        if user_id not in _in_memory_transcriptions:
            _in_memory_transcriptions[user_id] = {}
        _in_memory_transcriptions[user_id][transcription_id] = {
            "id": transcription_id,
            "user_id": user_id,
            "text": "Hello world",
            "language": "en-US",
            "audio_format": "wav",
            "file_size_bytes": 1000,
            "blob_storage_url": f"https://inmemory.blob.local/transcriptions/{blob_name}",
            "has_diarization": True,
            "created_at": "2024-01-15T10:30:00Z",
        }

        response = client.get(f"/transcriptions/{transcription_id}/content")
        assert response.status_code == 200
        data = response.json()
        assert data["transcript_id"] == transcription_id
        assert data["content"]["full_text"] == "Hello world"
        assert data["content"]["language"] == "en-US"
        assert len(data["content"]["speaker_segments"]) == 1
        assert "blob_url" in data

    def test_returns_404_for_nonexistent_transcription(
        self, client: TestClient
    ) -> None:
        """Test that 404 is returned for non-existent transcription."""
        response = client.get("/transcriptions/nonexistent-id/content")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_returns_404_when_no_blob_storage_url(
        self, client: TestClient
    ) -> None:
        """Test that 404 is returned when transcription has no blob_storage_url."""
        from db.cosmos import _in_memory_transcriptions

        transcription_id = "no-blob-url-id"
        user_id = MOCK_USER_OID

        if user_id not in _in_memory_transcriptions:
            _in_memory_transcriptions[user_id] = {}
        _in_memory_transcriptions[user_id][transcription_id] = {
            "id": transcription_id,
            "user_id": user_id,
            "text": "Test",
            "language": "en-US",
            "audio_format": "wav",
            "file_size_bytes": 1000,
            "blob_storage_url": None,  # No blob URL
            "has_diarization": False,
            "created_at": "2024-01-15T10:30:00Z",
        }

        response = client.get(f"/transcriptions/{transcription_id}/content")
        assert response.status_code == 404
        assert "not available" in response.json()["detail"].lower()

    def test_requires_authentication(self, unauthenticated_client: TestClient) -> None:
        """Test that authentication is required."""
        response = unauthenticated_client.get("/transcriptions/test-id/content")
        assert response.status_code == 401
