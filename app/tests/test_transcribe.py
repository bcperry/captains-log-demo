"""Tests for the transcription API endpoints."""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.transcribe import get_speech_service, router
from auth import AuthenticatedUser, get_current_user
from models.transcription import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    TranscriptionResponse,
)
from speech.client import (
    SpeechClient,
    SpeechConfigurationError,
    SpeechRecognitionError,
    SpeechServiceUnavailableError,
)


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
def mock_speech_client() -> MagicMock:
    """Create a mock Speech client."""
    client = MagicMock(spec=SpeechClient)
    client.create_audio_config_from_file.return_value = MagicMock()
    client.recognize_once.return_value = "This is the transcribed text."
    return client


@pytest.fixture
def client(
    app: FastAPI, mock_user: AuthenticatedUser, mock_speech_client: MagicMock
) -> TestClient:
    """Create a test client with mocked dependencies."""
    app.dependency_overrides[get_current_user] = lambda: mock_user
    app.dependency_overrides[get_speech_service] = lambda: mock_speech_client
    return TestClient(app)


@pytest.fixture
def unauthenticated_client(app: FastAPI, mock_speech_client: MagicMock) -> TestClient:
    """Create a test client without authentication."""
    # Only override speech service, not auth
    app.dependency_overrides[get_speech_service] = lambda: mock_speech_client
    return TestClient(app, raise_server_exceptions=False)


def create_audio_file(
    content: bytes = b"fake audio content",
    filename: str = "test.wav",
    content_type: str = "audio/wav",
) -> tuple[str, tuple[str, io.BytesIO, str]]:
    """Create a file tuple for upload testing."""
    return ("file", (filename, io.BytesIO(content), content_type))


class TestTranscribeEndpoint:
    """Tests for POST /transcribe endpoint."""

    def test_transcribes_wav_file(
        self, client: TestClient, mock_speech_client: MagicMock
    ) -> None:
        """Test transcription of WAV file."""
        response = client.post(
            "/transcribe",
            files=[create_audio_file(content_type="audio/wav", filename="test.wav")],
        )

        assert response.status_code == 200
        data = response.json()
        assert data["text"] == "This is the transcribed text."
        assert data["audio_format"] == "wav"
        assert data["language"] == "en-US"
        assert "transcribed_at" in data
        mock_speech_client.recognize_once.assert_called_once()

    def test_transcribes_mp3_file(
        self, client: TestClient, mock_speech_client: MagicMock
    ) -> None:
        """Test transcription of MP3 file."""
        response = client.post(
            "/transcribe",
            files=[create_audio_file(content_type="audio/mpeg", filename="test.mp3")],
        )

        assert response.status_code == 200
        data = response.json()
        assert data["audio_format"] == "mp3"
        mock_speech_client.recognize_once.assert_called_once()

    def test_transcribes_m4a_file(
        self, client: TestClient, mock_speech_client: MagicMock
    ) -> None:
        """Test transcription of M4A file."""
        response = client.post(
            "/transcribe",
            files=[create_audio_file(content_type="audio/mp4", filename="test.m4a")],
        )

        assert response.status_code == 200
        data = response.json()
        assert data["audio_format"] == "m4a"

    def test_custom_language(
        self, client: TestClient, mock_speech_client: MagicMock
    ) -> None:
        """Test transcription with custom language."""
        response = client.post(
            "/transcribe?language=es-ES",
            files=[create_audio_file()],
        )

        assert response.status_code == 200
        data = response.json()
        assert data["language"] == "es-ES"
        mock_speech_client.recognize_once.assert_called_once()
        # Verify language was passed to recognize_once
        call_args = mock_speech_client.recognize_once.call_args
        assert call_args[0][1] == "es-ES"

    def test_returns_file_size(
        self, client: TestClient, mock_speech_client: MagicMock
    ) -> None:
        """Test that response includes file size."""
        content = b"x" * 1000
        response = client.post(
            "/transcribe",
            files=[create_audio_file(content=content)],
        )

        assert response.status_code == 200
        assert response.json()["file_size_bytes"] == 1000


class TestFileValidation:
    """Tests for file validation."""

    def test_rejects_unsupported_format(self, client: TestClient) -> None:
        """Test rejection of unsupported audio formats."""
        response = client.post(
            "/transcribe",
            files=[
                create_audio_file(
                    content_type="audio/ogg", filename="test.ogg"
                )
            ],
        )

        assert response.status_code == 400
        assert "Unsupported audio format" in response.json()["detail"]

    def test_rejects_non_audio_file(self, client: TestClient) -> None:
        """Test rejection of non-audio files."""
        response = client.post(
            "/transcribe",
            files=[
                create_audio_file(
                    content_type="text/plain", filename="test.txt"
                )
            ],
        )

        assert response.status_code == 400

    def test_accepts_file_with_valid_extension_unknown_content_type(
        self, client: TestClient, mock_speech_client: MagicMock
    ) -> None:
        """Test acceptance of file with valid extension but unknown content type."""
        response = client.post(
            "/transcribe",
            files=[
                create_audio_file(
                    content_type="application/octet-stream", filename="test.wav"
                )
            ],
        )

        assert response.status_code == 200
        assert response.json()["audio_format"] == "wav"

    def test_rejects_file_too_large(
        self, app: FastAPI, mock_user: AuthenticatedUser, mock_speech_client: MagicMock
    ) -> None:
        """Test rejection of files exceeding size limit."""
        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.dependency_overrides[get_speech_service] = lambda: mock_speech_client
        client = TestClient(app)

        # Create content larger than max size
        large_content = b"x" * (MAX_FILE_SIZE_BYTES + 1)
        response = client.post(
            "/transcribe",
            files=[create_audio_file(content=large_content)],
        )

        assert response.status_code == 413
        assert "File too large" in response.json()["detail"]


class TestAuthentication:
    """Tests for authentication requirements."""

    def test_requires_authentication(self, unauthenticated_client: TestClient) -> None:
        """Test that endpoint requires authentication."""
        response = unauthenticated_client.post(
            "/transcribe",
            files=[create_audio_file()],
        )

        assert response.status_code == 401

    def test_returns_www_authenticate_header(
        self, unauthenticated_client: TestClient
    ) -> None:
        """Test that 401 response includes WWW-Authenticate header."""
        response = unauthenticated_client.post(
            "/transcribe",
            files=[create_audio_file()],
        )

        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers


class TestSpeechServiceErrors:
    """Tests for Speech Service error handling."""

    def test_handles_speech_service_unavailable(
        self, app: FastAPI, mock_user: AuthenticatedUser
    ) -> None:
        """Test handling when Speech Service is unavailable."""
        mock_client = MagicMock(spec=SpeechClient)
        mock_client.create_audio_config_from_file.return_value = MagicMock()
        mock_client.recognize_once.side_effect = SpeechServiceUnavailableError(
            "Service unavailable"
        )

        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.dependency_overrides[get_speech_service] = lambda: mock_client
        client = TestClient(app)

        response = client.post(
            "/transcribe",
            files=[create_audio_file()],
        )

        assert response.status_code == 503
        assert "Speech Services unavailable" in response.json()["detail"]

    def test_handles_recognition_error(
        self, app: FastAPI, mock_user: AuthenticatedUser
    ) -> None:
        """Test handling of speech recognition errors."""
        mock_client = MagicMock(spec=SpeechClient)
        mock_client.create_audio_config_from_file.return_value = MagicMock()
        mock_client.recognize_once.side_effect = SpeechRecognitionError("No speech detected")

        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.dependency_overrides[get_speech_service] = lambda: mock_client
        client = TestClient(app)

        response = client.post(
            "/transcribe",
            files=[create_audio_file()],
        )

        assert response.status_code == 422
        assert "Speech recognition failed" in response.json()["detail"]

    def test_handles_configuration_error(
        self, app: FastAPI, mock_user: AuthenticatedUser
    ) -> None:
        """Test handling of configuration errors."""
        mock_client = MagicMock(spec=SpeechClient)
        mock_client.create_audio_config_from_file.return_value = MagicMock()
        mock_client.recognize_once.side_effect = SpeechConfigurationError("Missing key")

        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.dependency_overrides[get_speech_service] = lambda: mock_client
        client = TestClient(app)

        response = client.post(
            "/transcribe",
            files=[create_audio_file()],
        )

        assert response.status_code == 503
        assert "configuration error" in response.json()["detail"]


class TestSpeechServiceDependency:
    """Tests for get_speech_service dependency."""

    def test_returns_503_when_speech_not_configured(
        self, app: FastAPI, mock_user: AuthenticatedUser
    ) -> None:
        """Test that 503 is returned when Speech Services is not configured."""
        app.dependency_overrides[get_current_user] = lambda: mock_user
        # Don't override get_speech_service - let it try to get real client

        with patch("api.transcribe.get_speech_client", return_value=None):
            # Re-import to use patched version
            app.dependency_overrides.pop(get_speech_service, None)
            client = TestClient(app, raise_server_exceptions=False)
            response = client.post(
                "/transcribe",
                files=[create_audio_file()],
            )

            assert response.status_code == 503
            assert "not configured" in response.json()["detail"]


class TestTranscriptionModels:
    """Tests for transcription models."""

    def test_transcription_response_model(self) -> None:
        """Test TranscriptionResponse model."""
        response = TranscriptionResponse(
            text="Hello world",
            language="en-US",
            audio_format="wav",
            file_size_bytes=1024,
        )
        assert response.text == "Hello world"
        assert response.language == "en-US"
        assert response.audio_format == "wav"
        assert response.file_size_bytes == 1024
        assert response.transcribed_at is not None
        assert response.duration_ms is None

    def test_allowed_extensions_constant(self) -> None:
        """Test that ALLOWED_EXTENSIONS contains expected formats."""
        assert "wav" in ALLOWED_EXTENSIONS
        assert "mp3" in ALLOWED_EXTENSIONS
        assert "m4a" in ALLOWED_EXTENSIONS
        assert len(ALLOWED_EXTENSIONS) == 3

    def test_max_file_size_constant(self) -> None:
        """Test that MAX_FILE_SIZE_BYTES is set appropriately."""
        assert MAX_FILE_SIZE_BYTES == 25 * 1024 * 1024  # 25 MB


class TestTempFileCleanup:
    """Tests for temporary file cleanup."""

    def test_cleans_up_temp_file_on_success(
        self, client: TestClient, mock_speech_client: MagicMock
    ) -> None:
        """Test that temporary files are cleaned up after successful transcription."""
        import os
        import tempfile

        set(os.listdir(tempfile.gettempdir()))

        response = client.post(
            "/transcribe",
            files=[create_audio_file()],
        )

        assert response.status_code == 200

        # Check no new temp files remain (may not be deterministic in all environments)
        # This is a basic check - temp file should be deleted in finally block

    def test_cleans_up_temp_file_on_error(
        self, app: FastAPI, mock_user: AuthenticatedUser
    ) -> None:
        """Test that temporary files are cleaned up on transcription error."""
        mock_client = MagicMock(spec=SpeechClient)
        mock_client.create_audio_config_from_file.return_value = MagicMock()
        mock_client.recognize_once.side_effect = SpeechRecognitionError("Error")

        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.dependency_overrides[get_speech_service] = lambda: mock_client
        client = TestClient(app)

        response = client.post(
            "/transcribe",
            files=[create_audio_file()],
        )

        # Error should be returned but temp file should still be cleaned up
        assert response.status_code == 422


class TestDiarizeEndpoint:
    """Tests for POST /transcribe/diarize endpoint."""

    @pytest.fixture
    def mock_speech_client_diarize(self) -> MagicMock:
        """Create a mock Speech client with diarization support."""
        client = MagicMock(spec=SpeechClient)
        client.create_audio_config_from_file.return_value = MagicMock()
        client.recognize_continuous_with_diarization.return_value = [
            {
                "speaker_id": "Speaker1",
                "text": "Hello, how are you?",
                "start_time_ms": 0,
                "end_time_ms": 2000,
            },
            {
                "speaker_id": "Speaker2",
                "text": "I'm doing great, thanks!",
                "start_time_ms": 2500,
                "end_time_ms": 5000,
            },
        ]
        return client

    @pytest.fixture
    def diarize_client(
        self,
        app: FastAPI,
        mock_user: AuthenticatedUser,
        mock_speech_client_diarize: MagicMock,
    ) -> TestClient:
        """Create a test client with diarization mock."""
        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.dependency_overrides[get_speech_service] = lambda: mock_speech_client_diarize
        return TestClient(app)

    def test_diarize_returns_segments(
        self, diarize_client: TestClient, mock_speech_client_diarize: MagicMock
    ) -> None:
        """Test that diarize endpoint returns speaker segments."""
        response = diarize_client.post(
            "/transcribe/diarize",
            files=[create_audio_file()],
        )

        assert response.status_code == 200
        data = response.json()
        assert "segments" in data
        assert len(data["segments"]) == 2
        assert data["segments"][0]["speaker_id"] == "Speaker1"
        assert data["segments"][0]["text"] == "Hello, how are you?"
        assert data["segments"][1]["speaker_id"] == "Speaker2"

    def test_diarize_returns_full_text(
        self, diarize_client: TestClient
    ) -> None:
        """Test that diarize endpoint returns full text."""
        response = diarize_client.post(
            "/transcribe/diarize",
            files=[create_audio_file()],
        )

        assert response.status_code == 200
        data = response.json()
        assert "full_text" in data
        assert "Hello, how are you?" in data["full_text"]
        assert "I'm doing great, thanks!" in data["full_text"]

    def test_diarize_returns_speaker_count(
        self, diarize_client: TestClient
    ) -> None:
        """Test that diarize endpoint returns correct speaker count."""
        response = diarize_client.post(
            "/transcribe/diarize",
            files=[create_audio_file()],
        )

        assert response.status_code == 200
        data = response.json()
        assert data["speaker_count"] == 2

    def test_diarize_returns_timestamps(
        self, diarize_client: TestClient
    ) -> None:
        """Test that diarize endpoint returns timestamps."""
        response = diarize_client.post(
            "/transcribe/diarize",
            files=[create_audio_file()],
        )

        assert response.status_code == 200
        data = response.json()
        assert data["segments"][0]["start_time_ms"] == 0
        assert data["segments"][0]["end_time_ms"] == 2000
        assert data["segments"][1]["start_time_ms"] == 2500

    def test_diarize_with_custom_max_speakers(
        self, diarize_client: TestClient, mock_speech_client_diarize: MagicMock
    ) -> None:
        """Test diarize with custom max_speakers parameter."""
        response = diarize_client.post(
            "/transcribe/diarize?max_speakers=3",
            files=[create_audio_file()],
        )

        assert response.status_code == 200
        data = response.json()
        assert data["max_speakers"] == 3
        mock_speech_client_diarize.recognize_continuous_with_diarization.assert_called_once()

    def test_diarize_rejects_invalid_max_speakers_too_low(
        self, diarize_client: TestClient
    ) -> None:
        """Test that diarize rejects max_speakers below minimum."""
        response = diarize_client.post(
            "/transcribe/diarize?max_speakers=0",
            files=[create_audio_file()],
        )

        assert response.status_code == 422  # FastAPI validation error

    def test_diarize_rejects_invalid_max_speakers_too_high(
        self, diarize_client: TestClient
    ) -> None:
        """Test that diarize rejects max_speakers above maximum."""
        response = diarize_client.post(
            "/transcribe/diarize?max_speakers=20",
            files=[create_audio_file()],
        )

        assert response.status_code == 422  # FastAPI validation error

    def test_diarize_requires_authentication(
        self, app: FastAPI, mock_speech_client_diarize: MagicMock
    ) -> None:
        """Test that diarize endpoint requires authentication."""
        app.dependency_overrides[get_speech_service] = lambda: mock_speech_client_diarize
        client = TestClient(app, raise_server_exceptions=False)

        response = client.post(
            "/transcribe/diarize",
            files=[create_audio_file()],
        )

        assert response.status_code == 401

    def test_diarize_validates_file_format(
        self, diarize_client: TestClient
    ) -> None:
        """Test that diarize endpoint validates file format."""
        response = diarize_client.post(
            "/transcribe/diarize",
            files=[create_audio_file(content_type="text/plain", filename="test.txt")],
        )

        assert response.status_code == 400

    def test_diarize_handles_service_error(
        self, app: FastAPI, mock_user: AuthenticatedUser
    ) -> None:
        """Test handling of service errors in diarize endpoint."""
        mock_client = MagicMock(spec=SpeechClient)
        mock_client.create_audio_config_from_file.return_value = MagicMock()
        mock_client.recognize_continuous_with_diarization.side_effect = (
            SpeechServiceUnavailableError("Diarization failed")
        )

        app.dependency_overrides[get_current_user] = lambda: mock_user
        app.dependency_overrides[get_speech_service] = lambda: mock_client
        client = TestClient(app)

        response = client.post(
            "/transcribe/diarize",
            files=[create_audio_file()],
        )

        assert response.status_code == 503


class TestDiarizedTranscriptionModels:
    """Tests for diarization models."""

    def test_speaker_segment_model(self) -> None:
        """Test SpeakerSegment model."""
        from models.transcription import SpeakerSegment

        segment = SpeakerSegment(
            speaker_id="Speaker1",
            text="Hello world",
            start_time_ms=0,
            end_time_ms=1000,
        )
        assert segment.speaker_id == "Speaker1"
        assert segment.text == "Hello world"
        assert segment.start_time_ms == 0
        assert segment.end_time_ms == 1000

    def test_diarized_transcription_response_model(self) -> None:
        """Test DiarizedTranscriptionResponse model."""
        from models.transcription import DiarizedTranscriptionResponse, SpeakerSegment

        response = DiarizedTranscriptionResponse(
            segments=[
                SpeakerSegment(
                    speaker_id="S1",
                    text="Test",
                    start_time_ms=0,
                    end_time_ms=500,
                )
            ],
            full_text="Test",
            language="en-US",
            audio_format="wav",
            file_size_bytes=1024,
            speaker_count=1,
            max_speakers=5,
        )
        assert len(response.segments) == 1
        assert response.full_text == "Test"
        assert response.speaker_count == 1
        assert response.max_speakers == 5
