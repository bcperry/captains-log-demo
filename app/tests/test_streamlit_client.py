"""Tests for the Streamlit API client."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from streamlit_client import APIClient, APIError


class TestAPIClientInit:
    """Tests for APIClient initialization."""

    def test_creates_client_with_base_url(self) -> None:
        """Client stores base URL correctly."""
        client = APIClient("http://localhost:8000")
        assert client.base_url == "http://localhost:8000"

    def test_strips_trailing_slash(self) -> None:
        """Client removes trailing slash from base URL."""
        client = APIClient("http://localhost:8000/")
        assert client.base_url == "http://localhost:8000"

    def test_creates_client_with_token(self) -> None:
        """Client stores access token when provided."""
        client = APIClient("http://localhost:8000", access_token="test-token")
        assert client.access_token == "test-token"

    def test_creates_client_without_token(self) -> None:
        """Client works without access token."""
        client = APIClient("http://localhost:8000")
        assert client.access_token is None


class TestAPIClientHeaders:
    """Tests for APIClient header handling."""

    def test_headers_without_token(self) -> None:
        """Headers include Accept without auth when no token."""
        client = APIClient("http://localhost:8000")
        headers = client._get_headers()
        assert headers["Accept"] == "application/json"
        assert "Authorization" not in headers

    def test_headers_with_token(self) -> None:
        """Headers include Authorization when token set."""
        client = APIClient("http://localhost:8000", access_token="test-token")
        headers = client._get_headers()
        assert headers["Authorization"] == "Bearer test-token"

    def test_set_access_token(self) -> None:
        """set_access_token updates the token."""
        client = APIClient("http://localhost:8000")
        client.set_access_token("new-token")
        assert client.access_token == "new-token"
        headers = client._get_headers()
        assert headers["Authorization"] == "Bearer new-token"

    def test_clear_access_token(self) -> None:
        """clear_access_token removes the token."""
        client = APIClient("http://localhost:8000", access_token="test-token")
        client.clear_access_token()
        assert client.access_token is None


class TestAPIClientErrors:
    """Tests for APIClient error handling."""

    def test_raises_api_error_on_4xx(self) -> None:
        """Client raises APIError on 4xx status codes."""
        client = APIClient("http://localhost:8000")

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"detail": "Unauthorized"}

        with pytest.raises(APIError) as exc_info:
            client._handle_response(mock_response)

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Unauthorized"

    def test_raises_api_error_on_5xx(self) -> None:
        """Client raises APIError on 5xx status codes."""
        client = APIClient("http://localhost:8000")

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {"detail": "Internal error"}

        with pytest.raises(APIError) as exc_info:
            client._handle_response(mock_response)

        assert exc_info.value.status_code == 500
        assert exc_info.value.detail == "Internal error"

    def test_handles_non_json_error_response(self) -> None:
        """Client handles error responses without JSON."""
        client = APIClient("http://localhost:8000")

        mock_response = MagicMock()
        mock_response.status_code = 502
        mock_response.json.side_effect = ValueError("No JSON")
        mock_response.text = "Bad Gateway"

        with pytest.raises(APIError) as exc_info:
            client._handle_response(mock_response)

        assert exc_info.value.status_code == 502
        assert exc_info.value.detail == "Bad Gateway"


class TestAPIClientHealth:
    """Tests for health endpoint methods."""

    def test_get_health(self) -> None:
        """get_health calls correct endpoint."""
        client = APIClient("http://localhost:8000")

        with patch.object(client._client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "healthy"}
            mock_get.return_value = mock_response

            result = client.get_health()

            mock_get.assert_called_once()
            assert "health" in mock_get.call_args[0][0]
            assert result["status"] == "healthy"

    def test_get_ready(self) -> None:
        """get_ready calls correct endpoint."""
        client = APIClient("http://localhost:8000")

        with patch.object(client._client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "ready"}
            mock_get.return_value = mock_response

            result = client.get_ready()

            mock_get.assert_called_once()
            assert "ready" in mock_get.call_args[0][0]
            assert result["status"] == "ready"


class TestAPIClientAuth:
    """Tests for auth endpoint methods."""

    def test_get_user_profile(self) -> None:
        """get_user_profile calls correct endpoint."""
        client = APIClient("http://localhost:8000", access_token="test-token")

        with patch.object(client._client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"name": "Test User"}
            mock_get.return_value = mock_response

            result = client.get_user_profile()

            mock_get.assert_called_once()
            assert "/auth/me" in mock_get.call_args[0][0]
            assert result["name"] == "Test User"

    def test_update_user_preferences(self) -> None:
        """update_user_preferences calls correct endpoint with payload."""
        client = APIClient("http://localhost:8000", access_token="test-token")

        with patch.object(client._client, "patch") as mock_patch:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"preferences": {"theme": "dark"}}
            mock_patch.return_value = mock_response

            result = client.update_user_preferences({"theme": "dark"})

            mock_patch.assert_called_once()
            assert "/auth/me/preferences" in mock_patch.call_args[0][0]
            assert result["preferences"]["theme"] == "dark"


class TestAPIClientTranscription:
    """Tests for transcription endpoint methods."""

    def test_transcribe_audio(self) -> None:
        """transcribe_audio sends file and params correctly."""
        client = APIClient("http://localhost:8000", access_token="test-token")

        with patch.object(client._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"text": "Hello world"}
            mock_post.return_value = mock_response

            result = client.transcribe_audio(
                file_content=b"audio data",
                filename="test.wav",
                language="en-US",
                store=True,
            )

            mock_post.assert_called_once()
            assert "/transcribe" in mock_post.call_args[0][0]
            assert result["text"] == "Hello world"

    def test_transcribe_audio_with_diarization(self) -> None:
        """transcribe_audio_with_diarization sends correct params."""
        client = APIClient("http://localhost:8000", access_token="test-token")

        with patch.object(client._client, "post") as mock_post:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "segments": [],
                "full_text": "Hello",
                "speaker_count": 2,
            }
            mock_post.return_value = mock_response

            result = client.transcribe_audio_with_diarization(
                file_content=b"audio data",
                filename="test.wav",
                language="en-US",
                max_speakers=5,
            )

            mock_post.assert_called_once()
            assert "/transcribe/diarize" in mock_post.call_args[0][0]
            assert result["speaker_count"] == 2


class TestAPIClientTranscriptionHistory:
    """Tests for transcription history endpoint methods."""

    def test_list_transcriptions(self) -> None:
        """list_transcriptions calls correct endpoint with pagination."""
        client = APIClient("http://localhost:8000", access_token="test-token")

        with patch.object(client._client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "transcriptions": [],
                "total": 0,
            }
            mock_get.return_value = mock_response

            result = client.list_transcriptions(page=2, per_page=10)

            mock_get.assert_called_once()
            assert "/transcriptions" in mock_get.call_args[0][0]
            assert mock_get.call_args[1]["params"]["page"] == 2
            assert mock_get.call_args[1]["params"]["per_page"] == 10

    def test_get_transcription(self) -> None:
        """get_transcription calls correct endpoint with ID."""
        client = APIClient("http://localhost:8000", access_token="test-token")

        with patch.object(client._client, "get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"id": "test-id", "text": "Hello"}
            mock_get.return_value = mock_response

            result = client.get_transcription("test-id")

            mock_get.assert_called_once()
            assert "/transcriptions/test-id" in mock_get.call_args[0][0]
            assert result["id"] == "test-id"

    def test_delete_transcription(self) -> None:
        """delete_transcription calls correct endpoint."""
        client = APIClient("http://localhost:8000", access_token="test-token")

        with patch.object(client._client, "delete") as mock_delete:
            mock_response = MagicMock()
            mock_response.status_code = 204
            mock_delete.return_value = mock_response

            client.delete_transcription("test-id")

            mock_delete.assert_called_once()
            assert "/transcriptions/test-id" in mock_delete.call_args[0][0]

    def test_delete_transcription_raises_on_404(self) -> None:
        """delete_transcription raises APIError on 404."""
        client = APIClient("http://localhost:8000", access_token="test-token")

        with patch.object(client._client, "delete") as mock_delete:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_response.json.return_value = {"detail": "Not found"}
            mock_delete.return_value = mock_response

            with pytest.raises(APIError) as exc_info:
                client.delete_transcription("nonexistent")

            assert exc_info.value.status_code == 404


class TestAPIError:
    """Tests for APIError exception."""

    def test_api_error_attributes(self) -> None:
        """APIError stores status code and detail."""
        error = APIError(401, "Unauthorized")
        assert error.status_code == 401
        assert error.detail == "Unauthorized"

    def test_api_error_str(self) -> None:
        """APIError has readable string representation."""
        error = APIError(500, "Server error")
        assert "500" in str(error)
        assert "Server error" in str(error)
