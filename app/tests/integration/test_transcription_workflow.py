"""Integration tests for transcription workflow.

Tests the complete transcription workflow including:
- Audio file upload and transcription
- Diarization with speaker identification
- Transcription storage and retrieval
- Transcription history management
"""

import io

from fastapi.testclient import TestClient


def create_test_audio(
    content: bytes = b"fake audio content",
    filename: str = "test.wav",
    content_type: str = "audio/wav",
) -> tuple[str, tuple[str, io.BytesIO, str]]:
    """Create a test audio file tuple for upload."""
    return ("file", (filename, io.BytesIO(content), content_type))


class TestTranscriptionWorkflow:
    """Integration tests for the transcription workflow."""

    def test_transcribe_and_store(
        self,
        authenticated_client_with_speech: TestClient,
    ) -> None:
        """Test that transcription is stored in history."""
        # Transcribe audio
        response = authenticated_client_with_speech.post(
            "/transcribe",
            files=[create_test_audio()],
        )

        assert response.status_code == 200
        assert response.json()["text"] == "Integration test transcribed text."

        # Check it's in history
        list_response = authenticated_client_with_speech.get("/transcriptions")
        assert list_response.status_code == 200
        assert list_response.json()["total"] == 1

    def test_transcribe_without_storage(
        self,
        authenticated_client_with_speech: TestClient,
    ) -> None:
        """Test transcription without storing in history."""
        # Transcribe without storing
        response = authenticated_client_with_speech.post(
            "/transcribe?store=false",
            files=[create_test_audio()],
        )

        assert response.status_code == 200
        assert response.json()["text"] == "Integration test transcribed text."

        # Check history is empty
        list_response = authenticated_client_with_speech.get("/transcriptions")
        assert list_response.status_code == 200
        assert list_response.json()["total"] == 0

    def test_transcribe_with_diarization(
        self,
        authenticated_client_with_speech: TestClient,
    ) -> None:
        """Test transcription with speaker diarization."""
        response = authenticated_client_with_speech.post(
            "/transcribe/diarize",
            files=[create_test_audio()],
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["segments"]) == 2
        assert data["speaker_count"] == 2
        assert "Hello from integration test" in data["full_text"]

    def test_transcribe_with_custom_language(
        self,
        authenticated_client_with_speech: TestClient,
    ) -> None:
        """Test transcription with custom language."""
        response = authenticated_client_with_speech.post(
            "/transcribe?language=es-ES",
            files=[create_test_audio()],
        )

        assert response.status_code == 200
        assert response.json()["language"] == "es-ES"


class TestTranscriptionHistory:
    """Integration tests for transcription history management."""

    def test_list_transcriptions_with_pagination(
        self,
        authenticated_client_with_speech: TestClient,
    ) -> None:
        """Test listing transcriptions with pagination."""
        # Create multiple transcriptions
        for _ in range(5):
            authenticated_client_with_speech.post(
                "/transcribe",
                files=[create_test_audio()],
            )

        # Get first page
        response = authenticated_client_with_speech.get(
            "/transcriptions?page=1&per_page=2"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 5
        assert len(data["transcriptions"]) == 2
        assert data["page"] == 1
        assert data["per_page"] == 2

    def test_retrieve_specific_transcription(
        self,
        authenticated_client_with_speech: TestClient,
    ) -> None:
        """Test retrieving a specific transcription by ID."""
        # Create a transcription
        authenticated_client_with_speech.post(
            "/transcribe",
            files=[create_test_audio()],
        )

        # Get the list to find the ID
        list_response = authenticated_client_with_speech.get("/transcriptions")
        transcription_id = list_response.json()["transcriptions"][0]["id"]

        # Retrieve specific transcription
        response = authenticated_client_with_speech.get(
            f"/transcriptions/{transcription_id}"
        )

        assert response.status_code == 200
        assert response.json()["id"] == transcription_id
        assert response.json()["text"] == "Integration test transcribed text."

    def test_delete_transcription(
        self,
        authenticated_client_with_speech: TestClient,
    ) -> None:
        """Test deleting a transcription."""
        # Create a transcription
        authenticated_client_with_speech.post(
            "/transcribe",
            files=[create_test_audio()],
        )

        # Get the ID
        list_response = authenticated_client_with_speech.get("/transcriptions")
        transcription_id = list_response.json()["transcriptions"][0]["id"]

        # Delete it
        delete_response = authenticated_client_with_speech.delete(
            f"/transcriptions/{transcription_id}"
        )
        assert delete_response.status_code == 204

        # Verify it's gone
        get_response = authenticated_client_with_speech.get(
            f"/transcriptions/{transcription_id}"
        )
        assert get_response.status_code == 404


class TestFileValidation:
    """Integration tests for file upload validation."""

    def test_rejects_invalid_audio_format(
        self,
        authenticated_client_with_speech: TestClient,
    ) -> None:
        """Test that invalid audio formats are rejected."""
        response = authenticated_client_with_speech.post(
            "/transcribe",
            files=[create_test_audio(filename="test.txt", content_type="text/plain")],
        )

        assert response.status_code == 400
        assert "Unsupported audio format" in response.json()["detail"]

    def test_accepts_various_audio_formats(
        self,
        authenticated_client_with_speech: TestClient,
    ) -> None:
        """Test that various audio formats are accepted."""
        formats = [
            ("test.wav", "audio/wav"),
            ("test.mp3", "audio/mpeg"),
            ("test.m4a", "audio/mp4"),
        ]

        for filename, content_type in formats:
            response = authenticated_client_with_speech.post(
                "/transcribe?store=false",
                files=[create_test_audio(filename=filename, content_type=content_type)],
            )
            assert response.status_code == 200, f"Failed for {filename}"


class TestEndToEndWorkflow:
    """End-to-end integration tests for complete user workflows."""

    def test_new_user_complete_workflow(
        self,
        authenticated_client_with_speech: TestClient,
    ) -> None:
        """Test complete workflow for a new user."""
        # 1. Get profile (creates on first access)
        profile_response = authenticated_client_with_speech.get("/auth/me")
        assert profile_response.status_code == 200
        user_id = profile_response.json()["id"]

        # 2. Update preferences
        prefs = {
            "theme": "dark",
            "language": "en-GB",
            "notifications_enabled": True,
            "auto_transcribe": False,
        }
        prefs_response = authenticated_client_with_speech.patch(
            "/auth/me/preferences", json=prefs
        )
        assert prefs_response.status_code == 200
        assert prefs_response.json()["preferences"]["theme"] == "dark"

        # 3. Transcribe audio
        transcribe_response = authenticated_client_with_speech.post(
            "/transcribe",
            files=[create_test_audio()],
        )
        assert transcribe_response.status_code == 200

        # 4. View transcription history
        history_response = authenticated_client_with_speech.get("/transcriptions")
        assert history_response.status_code == 200
        assert history_response.json()["total"] == 1

        # 5. Get specific transcription
        transcription_id = history_response.json()["transcriptions"][0]["id"]
        detail_response = authenticated_client_with_speech.get(
            f"/transcriptions/{transcription_id}"
        )
        assert detail_response.status_code == 200
        assert detail_response.json()["user_id"] == user_id

        # 6. Delete transcription
        delete_response = authenticated_client_with_speech.delete(
            f"/transcriptions/{transcription_id}"
        )
        assert delete_response.status_code == 204

        # 7. Verify empty history
        final_history = authenticated_client_with_speech.get("/transcriptions")
        assert final_history.json()["total"] == 0
