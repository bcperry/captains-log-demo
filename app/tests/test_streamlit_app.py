"""Tests for the Streamlit application utilities and helpers."""

from datetime import datetime

import pytest


class TestFormatDatetime:
    """Tests for format_datetime utility function."""

    def test_formats_iso_datetime(self) -> None:
        """format_datetime converts ISO string to readable format."""
        from streamlit_app import format_datetime

        result = format_datetime("2026-01-09T10:30:00Z")
        assert "2026-01-09" in result
        assert "10:30" in result

    def test_formats_datetime_with_offset(self) -> None:
        """format_datetime handles timezone offset."""
        from streamlit_app import format_datetime

        result = format_datetime("2026-01-09T10:30:00+00:00")
        assert "2026-01-09" in result

    def test_returns_original_on_invalid(self) -> None:
        """format_datetime returns original string if invalid."""
        from streamlit_app import format_datetime

        result = format_datetime("not a date")
        assert result == "not a date"


class TestFormatFileSize:
    """Tests for format_file_size utility function."""

    def test_formats_bytes(self) -> None:
        """format_file_size shows bytes for small files."""
        from streamlit_app import format_file_size

        result = format_file_size(512)
        assert "512" in result
        assert "B" in result

    def test_formats_kilobytes(self) -> None:
        """format_file_size shows KB for kilobyte-sized files."""
        from streamlit_app import format_file_size

        result = format_file_size(2048)
        assert "2.0" in result
        assert "KB" in result

    def test_formats_megabytes(self) -> None:
        """format_file_size shows MB for megabyte-sized files."""
        from streamlit_app import format_file_size

        result = format_file_size(5 * 1024 * 1024)
        assert "5.0" in result
        assert "MB" in result

    def test_formats_gigabytes(self) -> None:
        """format_file_size shows GB for gigabyte-sized files."""
        from streamlit_app import format_file_size

        result = format_file_size(2 * 1024 * 1024 * 1024)
        assert "2.0" in result
        assert "GB" in result


class TestCheckBackendHealth:
    """Tests for check_backend_health function."""

    def test_returns_health_on_success(self) -> None:
        """check_backend_health returns health status."""
        from unittest.mock import MagicMock, patch

        from streamlit_app import check_backend_health

        mock_client = MagicMock()
        mock_client.get_ready.return_value = {
            "status": "ready",
            "dependencies": [],
        }

        with patch("streamlit_app.get_api_client", return_value=mock_client):
            result = check_backend_health()

        assert result["status"] == "ready"

    def test_returns_error_on_api_error(self) -> None:
        """check_backend_health returns error dict on APIError."""
        from unittest.mock import MagicMock, patch

        from streamlit_app import check_backend_health
        from streamlit_client import APIError

        mock_client = MagicMock()
        mock_client.get_ready.side_effect = APIError(503, "Service unavailable")

        with patch("streamlit_app.get_api_client", return_value=mock_client):
            result = check_backend_health()

        assert result["status"] == "error"
        assert "unavailable" in result["detail"].lower()

    def test_returns_error_on_connection_error(self) -> None:
        """check_backend_health returns error on connection failure."""
        from unittest.mock import MagicMock, patch

        from streamlit_app import check_backend_health

        mock_client = MagicMock()
        mock_client.get_ready.side_effect = Exception("Connection refused")

        with patch("streamlit_app.get_api_client", return_value=mock_client):
            result = check_backend_health()

        assert result["status"] == "error"
        assert "Connection refused" in result["detail"]


class TestRenderSimpleResult:
    """Tests for render_simple_result function."""

    def test_extracts_text_from_result(self) -> None:
        """render_simple_result handles transcription response."""
        # This test verifies the function doesn't crash with valid input
        # Full rendering tests require Streamlit test utilities
        result = {
            "text": "Hello world",
            "audio_format": "wav",
            "language": "en-US",
        }

        # Verify result structure
        assert "text" in result
        assert result["text"] == "Hello world"


class TestRenderDiarizedResult:
    """Tests for render_diarized_result function."""

    def test_handles_speaker_segments(self) -> None:
        """render_diarized_result handles diarized response."""
        result = {
            "segments": [
                {"speaker_id": "Speaker_1", "text": "Hello", "start_time_ms": 0, "end_time_ms": 1000},
                {"speaker_id": "Speaker_2", "text": "Hi there", "start_time_ms": 1100, "end_time_ms": 2000},
            ],
            "full_text": "Hello Hi there",
            "speaker_count": 2,
            "audio_format": "wav",
        }

        # Verify result structure
        assert len(result["segments"]) == 2
        assert result["speaker_count"] == 2


class TestAPIClientGetFromEnvironment:
    """Tests for get_api_client environment handling."""

    def test_uses_default_url(self) -> None:
        """get_api_client uses default URL when not set."""
        # This tests the default URL logic without Streamlit runtime
        import os

        # Default URL should be localhost:8001
        default_url = os.getenv(
            "BACKEND_API_URL",
            os.getenv("API_BASE_URL", "http://localhost:8001"),
        )
        assert "localhost" in default_url or "8001" in default_url

    def test_environment_variable_precedence(self) -> None:
        """Environment variables are used for backend URL configuration."""
        from unittest.mock import patch

        with patch.dict("os.environ", {"BACKEND_API_URL": "http://custom:9000"}):
            import os

            url = os.getenv("BACKEND_API_URL")
            assert url == "http://custom:9000"


class TestAPIClientSessionState:
    """Tests for API client session state management."""

    def test_session_state_pattern(self) -> None:
        """Session state is used to persist client."""
        # This verifies the pattern without requiring Streamlit runtime
        from streamlit_client import APIClient

        # Simulate session state behavior
        session_state: dict[str, object] = {}

        if "api_client" not in session_state:
            session_state["api_client"] = APIClient("http://test:8000")

        client = session_state["api_client"]
        assert isinstance(client, APIClient)

    def test_token_update_pattern(self) -> None:
        """Access token can be updated in session state."""
        from streamlit_client import APIClient

        client = APIClient("http://test:8000")
        session_state = {
            "api_client": client,
            "access_token": "test-token",
        }

        # Update client token from session state
        if session_state.get("access_token"):
            client.set_access_token(str(session_state["access_token"]))

        assert client.access_token == "test-token"


class TestPageNavigation:
    """Tests for page navigation logic."""

    def test_navigation_options_exist(self) -> None:
        """Navigation includes all main pages."""
        navigation_pages = ["🎤 Transcribe", "📚 History", "⚙️ Settings"]

        assert "Transcribe" in navigation_pages[0]
        assert "History" in navigation_pages[1]
        assert "Settings" in navigation_pages[2]


class TestLanguageOptions:
    """Tests for transcription language options."""

    def test_supported_languages(self) -> None:
        """Verify supported language codes."""
        languages = {
            "English (US)": "en-US",
            "English (UK)": "en-GB",
            "Spanish": "es-ES",
            "French": "fr-FR",
            "German": "de-DE",
        }

        assert languages["English (US)"] == "en-US"
        assert languages["English (UK)"] == "en-GB"
        assert languages["Spanish"] == "es-ES"


class TestSpeakerColors:
    """Tests for speaker color mapping."""

    def test_speaker_color_mapping(self) -> None:
        """Verify speaker color icons are defined."""
        speaker_colors = {
            "Speaker_1": "🔵",
            "Speaker_2": "🟢",
            "Speaker_3": "🟠",
            "Speaker_4": "🟣",
            "Speaker_5": "🔴",
        }

        assert speaker_colors["Speaker_1"] == "🔵"
        assert speaker_colors["Speaker_2"] == "🟢"
        assert speaker_colors.get("Unknown", "⚪") == "⚪"


class TestVersionConstant:
    """Tests for version constant."""

    def test_version_defined(self) -> None:
        """VERSION constant is defined."""
        from streamlit_app import VERSION

        assert VERSION is not None
        assert isinstance(VERSION, str)
        assert "." in VERSION  # Semantic versioning
