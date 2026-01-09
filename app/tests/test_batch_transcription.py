"""Tests for the batch transcription API and client."""

import io
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.transcribe import get_batch_service, get_blob_storage, router
from auth import AuthenticatedUser
from auth.dependencies import get_current_user_azure
from models.transcription import BatchTranscriptionJobStatus
from speech.batch import (
    BatchTranscriptionClient,
    BatchTranscriptionConfig,
    BatchTranscriptionError,
    BatchTranscriptionFailedError,
    BatchTranscriptionJob,
    BatchTranscriptionJobNotFoundError,
    BatchTranscriptionResult,
    InMemoryBatchTranscriptionClient,
    TranscriptionSegment,
    TranscriptionStatus,
)
from storage.blob import InMemoryBlobClient


@pytest.fixture
def app() -> FastAPI:
    """Create a test FastAPI application."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


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
def mock_batch_client() -> MagicMock:
    """Create a mock batch transcription client."""
    client = MagicMock(spec=BatchTranscriptionClient)
    # Make async methods return AsyncMock
    client.create_transcription_job = AsyncMock()
    client.get_transcription_status = AsyncMock()
    client.get_transcription_result = AsyncMock()
    client.delete_transcription = AsyncMock()
    return client


@pytest.fixture
def mock_storage() -> InMemoryBlobClient:
    """Create a mock in-memory storage client."""
    return InMemoryBlobClient()


@pytest.fixture
def client(
    app: FastAPI,
    mock_user: AuthenticatedUser,
    mock_batch_client: MagicMock,
    mock_storage: InMemoryBlobClient,
) -> TestClient:
    """Create a test client with mocked dependencies."""
    app.dependency_overrides[get_current_user_azure] = lambda: mock_user
    app.dependency_overrides[get_batch_service] = lambda: mock_batch_client
    app.dependency_overrides[get_blob_storage] = lambda: mock_storage
    return TestClient(app)


class TestBatchTranscriptionConfig:
    """Tests for BatchTranscriptionConfig."""

    def test_config_government_endpoint(self) -> None:
        """Test Government cloud endpoint generation."""
        from config.settings import AzureCloud

        config = BatchTranscriptionConfig(
            subscription_key="test-key",
            region="usgovvirginia",
            cloud=AzureCloud.GOVERNMENT,
        )
        assert "usgovvirginia.api.cognitive.azure.us" in config.base_url
        assert "speechtotext/v3.1" in config.base_url

    def test_config_commercial_endpoint(self) -> None:
        """Test Commercial cloud endpoint generation."""
        from config.settings import AzureCloud

        config = BatchTranscriptionConfig(
            subscription_key="test-key",
            region="eastus",
            cloud=AzureCloud.COMMERCIAL,
        )
        assert "eastus.api.cognitive.microsoft.com" in config.base_url


class TestBatchTranscriptionClient:
    """Tests for BatchTranscriptionClient."""

    def test_parse_duration_to_ms(self) -> None:
        """Test ISO 8601 duration parsing."""
        from config.settings import AzureCloud

        config = BatchTranscriptionConfig(
            subscription_key="test-key",
            region="test-region",
            cloud=AzureCloud.LOCAL,
        )
        client = BatchTranscriptionClient(config)

        # Test various duration formats
        assert client._parse_duration_to_ms("PT30S") == 30000
        assert client._parse_duration_to_ms("PT1M30S") == 90000
        assert client._parse_duration_to_ms("PT1H") == 3600000
        assert client._parse_duration_to_ms("PT1H2M3.456S") == 3723456
        assert client._parse_duration_to_ms("PT0.5S") == 500
        assert client._parse_duration_to_ms("") == 0
        assert client._parse_duration_to_ms("invalid") == 0

    def test_parse_job_status(self) -> None:
        """Test job status parsing from API response."""
        from config.settings import AzureCloud

        config = BatchTranscriptionConfig(
            subscription_key="test-key",
            region="test-region",
            cloud=AzureCloud.LOCAL,
        )
        client = BatchTranscriptionClient(config)

        # Test successful job
        data = {
            "status": "Succeeded",
            "displayName": "Test Job",
            "createdDateTime": "2024-01-15T10:30:00Z",
            "lastActionDateTime": "2024-01-15T10:32:00Z",
        }
        job = client._parse_job_status(data, "job-123")
        assert job.id == "job-123"
        assert job.status == TranscriptionStatus.SUCCEEDED
        assert job.display_name == "Test Job"
        assert job.error_message is None

        # Test failed job
        data = {
            "status": "Failed",
            "displayName": "Failed Job",
            "createdDateTime": "2024-01-15T10:30:00Z",
            "properties": {"error": {"message": "Audio file not found"}},
        }
        job = client._parse_job_status(data, "job-456")
        assert job.status == TranscriptionStatus.FAILED
        assert job.error_message == "Audio file not found"


class TestInMemoryBatchTranscriptionClient:
    """Tests for InMemoryBatchTranscriptionClient."""

    @pytest.mark.asyncio
    async def test_create_and_get_status(self) -> None:
        """Test creating a job and getting its status."""
        client = InMemoryBatchTranscriptionClient()

        job_id = await client.create_transcription_job(
            content_urls=["https://storage.blob/audio.wav"],
            display_name="Test Transcription",
            locale="en-US",
        )

        assert job_id.startswith("mock-job-")

        status = await client.get_transcription_status(job_id)
        assert status.status == TranscriptionStatus.SUCCEEDED
        assert status.display_name == "Test Transcription"

    @pytest.mark.asyncio
    async def test_get_result(self) -> None:
        """Test getting transcription result."""
        client = InMemoryBatchTranscriptionClient()

        job_id = await client.create_transcription_job(
            content_urls=["https://storage.blob/audio.wav"],
            display_name="Test Transcription",
        )

        result = await client.get_transcription_result(job_id)
        assert result.job_id == job_id
        assert len(result.segments) > 0
        assert result.full_text != ""

    @pytest.mark.asyncio
    async def test_job_not_found(self) -> None:
        """Test error when job is not found."""
        client = InMemoryBatchTranscriptionClient()

        with pytest.raises(BatchTranscriptionJobNotFoundError):
            await client.get_transcription_status("nonexistent-job")

    @pytest.mark.asyncio
    async def test_delete_job(self) -> None:
        """Test deleting a job."""
        client = InMemoryBatchTranscriptionClient()

        job_id = await client.create_transcription_job(
            content_urls=["https://storage.blob/audio.wav"],
            display_name="Test Transcription",
        )

        await client.delete_transcription(job_id)

        with pytest.raises(BatchTranscriptionJobNotFoundError):
            await client.get_transcription_status(job_id)


class TestBatchTranscriptionAPI:
    """Tests for batch transcription API endpoints."""

    def test_create_batch_transcription(
        self,
        client: TestClient,
        mock_batch_client: MagicMock,
    ) -> None:
        """Test creating a batch transcription job."""
        # Setup mock responses
        mock_batch_client.create_transcription_job.return_value = "job-123"
        mock_batch_client.get_transcription_status.return_value = BatchTranscriptionJob(
            id="job-123",
            status=TranscriptionStatus.RUNNING,
            display_name="Test Transcription",
            created_date_time=datetime.now(),
        )

        # Create a test audio file
        audio_content = b"RIFF" + b"\x00" * 100  # Minimal WAV header
        files = {"file": ("test.wav", io.BytesIO(audio_content), "audio/wav")}

        response = client.post("/transcribe/batch", files=files)

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "job-123"
        assert data["status"] == "Running"

    def test_create_batch_transcription_invalid_format(
        self,
        client: TestClient,
    ) -> None:
        """Test error when audio format is invalid."""
        files = {"file": ("test.txt", io.BytesIO(b"not audio"), "text/plain")}

        response = client.post("/transcribe/batch", files=files)

        assert response.status_code == 400
        assert "Unsupported audio format" in response.json()["detail"]

    def test_get_batch_status(
        self,
        client: TestClient,
        mock_batch_client: MagicMock,
    ) -> None:
        """Test getting batch transcription status."""
        mock_batch_client.get_transcription_status.return_value = BatchTranscriptionJob(
            id="job-123",
            status=TranscriptionStatus.SUCCEEDED,
            display_name="Test Transcription",
            created_date_time=datetime.now(),
            last_action_date_time=datetime.now(),
        )

        response = client.get("/transcribe/batch/job-123/status")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "job-123"
        assert data["status"] == "Succeeded"

    def test_get_batch_status_not_found(
        self,
        client: TestClient,
        mock_batch_client: MagicMock,
    ) -> None:
        """Test 404 when job is not found."""
        mock_batch_client.get_transcription_status.side_effect = BatchTranscriptionJobNotFoundError(
            "Job not found"
        )

        response = client.get("/transcribe/batch/nonexistent/status")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_get_batch_result(
        self,
        client: TestClient,
        mock_batch_client: MagicMock,
    ) -> None:
        """Test getting batch transcription result."""
        mock_batch_client.get_transcription_result.return_value = BatchTranscriptionResult(
            job_id="job-123",
            segments=[
                TranscriptionSegment(
                    speaker_id="Speaker_1",
                    text="Hello, world!",
                    start_time_ms=0,
                    end_time_ms=2000,
                    confidence=0.95,
                )
            ],
            full_text="Hello, world!",
            duration_ms=2000,
            speaker_count=1,
            language="en-US",
        )

        response = client.get("/transcribe/batch/job-123/result")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "job-123"
        assert data["full_text"] == "Hello, world!"
        assert len(data["segments"]) == 1
        assert data["segments"][0]["speaker_id"] == "Speaker_1"

    def test_get_batch_result_job_failed(
        self,
        client: TestClient,
        mock_batch_client: MagicMock,
    ) -> None:
        """Test error when job failed."""
        mock_batch_client.get_transcription_result.side_effect = BatchTranscriptionFailedError(
            "Audio file corrupt"
        )

        response = client.get("/transcribe/batch/job-123/result")

        assert response.status_code == 409
        assert "failed" in response.json()["detail"].lower()

    def test_get_batch_result_job_not_complete(
        self,
        client: TestClient,
        mock_batch_client: MagicMock,
    ) -> None:
        """Test error when job is not complete."""
        mock_batch_client.get_transcription_result.side_effect = BatchTranscriptionError(
            "Transcription job not complete. Current status: Running"
        )

        response = client.get("/transcribe/batch/job-123/result")

        assert response.status_code == 409
        assert "not complete" in response.json()["detail"].lower()


class TestBatchModels:
    """Tests for batch transcription models."""

    def test_batch_job_status_enum(self) -> None:
        """Test BatchTranscriptionJobStatus enum values."""
        assert BatchTranscriptionJobStatus.NOT_STARTED == "NotStarted"
        assert BatchTranscriptionJobStatus.RUNNING == "Running"
        assert BatchTranscriptionJobStatus.SUCCEEDED == "Succeeded"
        assert BatchTranscriptionJobStatus.FAILED == "Failed"

    def test_batch_job_response_model(self) -> None:
        """Test BatchTranscriptionJobResponse model."""
        from models.transcription import BatchTranscriptionJobResponse

        response = BatchTranscriptionJobResponse(
            job_id="job-123",
            status=BatchTranscriptionJobStatus.RUNNING,
            display_name="Test Job",
            created_at=datetime.now(),
            blob_url="https://storage.blob/audio.wav",
        )

        assert response.job_id == "job-123"
        assert response.status == BatchTranscriptionJobStatus.RUNNING

    def test_batch_result_response_model(self) -> None:
        """Test BatchTranscriptionResultResponse model."""
        from models.transcription import (
            BatchTranscriptionResultResponse,
            SpeakerSegment,
        )

        response = BatchTranscriptionResultResponse(
            job_id="job-123",
            segments=[
                SpeakerSegment(
                    speaker_id="Speaker_1",
                    text="Hello",
                    start_time_ms=0,
                    end_time_ms=1000,
                )
            ],
            full_text="Hello",
            language="en-US",
            duration_ms=1000,
            speaker_count=1,
        )

        assert response.job_id == "job-123"
        assert len(response.segments) == 1
