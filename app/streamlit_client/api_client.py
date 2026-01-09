"""API client for communicating with the FastAPI backend.

This module provides a typed client for making HTTP requests to the
Captain's Log FastAPI backend API.
"""

from typing import Any, Optional

import httpx


class APIError(Exception):
    """Exception raised when API requests fail."""

    def __init__(self, status_code: int, detail: str):
        """Initialize API error.

        Args:
            status_code: HTTP status code
            detail: Error message detail
        """
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API Error {status_code}: {detail}")


class APIClient:
    """HTTP client for FastAPI backend communication.

    Provides methods for all API endpoints with proper typing and error handling.
    """

    def __init__(self, base_url: str, access_token: Optional[str] = None):
        """Initialize the API client.

        Args:
            base_url: Base URL of the FastAPI backend (e.g., http://localhost:8000)
            access_token: Optional Azure Entra ID access token for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.access_token = access_token
        self._client = httpx.Client(timeout=60.0)

    def _get_headers(self) -> dict[str, str]:
        """Get request headers including authorization if token is set."""
        headers: dict[str, str] = {"Accept": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    def _handle_response(self, response: httpx.Response) -> dict[str, Any]:
        """Handle API response and raise errors if needed.

        Args:
            response: HTTP response from httpx

        Returns:
            Parsed JSON response

        Raises:
            APIError: If response status code indicates an error
        """
        if response.status_code >= 400:
            try:
                error_detail = response.json().get("detail", response.text)
            except Exception:
                error_detail = response.text or "Unknown error"
            raise APIError(response.status_code, error_detail)
        result: dict[str, Any] = response.json()
        return result

    def set_access_token(self, token: str) -> None:
        """Update the access token for subsequent requests.

        Args:
            token: Azure Entra ID access token
        """
        self.access_token = token

    def clear_access_token(self) -> None:
        """Clear the access token (logout)."""
        self.access_token = None

    # Health endpoints
    def get_health(self) -> dict[str, Any]:
        """Get application health status.

        Returns:
            Health status response
        """
        response = self._client.get(
            f"{self.base_url}/health",
            headers=self._get_headers(),
        )
        return self._handle_response(response)

    def get_ready(self) -> dict[str, Any]:
        """Get application readiness status.

        Returns:
            Readiness status with dependency health
        """
        response = self._client.get(
            f"{self.base_url}/ready",
            headers=self._get_headers(),
        )
        return self._handle_response(response)

    # Auth endpoints
    def get_user_profile(self) -> dict[str, Any]:
        """Get the current user's profile.

        Returns:
            User profile data

        Raises:
            APIError: 401 if not authenticated
        """
        response = self._client.get(
            f"{self.base_url}/auth/me",
            headers=self._get_headers(),
        )
        return self._handle_response(response)

    def update_user_preferences(
        self, preferences: dict[str, Any]
    ) -> dict[str, Any]:
        """Update the current user's preferences.

        Args:
            preferences: New preference values

        Returns:
            Updated user profile

        Raises:
            APIError: 401 if not authenticated
        """
        response = self._client.patch(
            f"{self.base_url}/auth/me/preferences",
            headers=self._get_headers(),
            json=preferences,
        )
        return self._handle_response(response)

    # Transcription endpoints
    def transcribe_audio(
        self,
        file_content: bytes,
        filename: str,
        language: str = "en-US",
        store: bool = True,
    ) -> dict[str, Any]:
        """Transcribe an audio file.

        Args:
            file_content: Audio file content as bytes
            filename: Original filename
            language: Language code for transcription
            store: Whether to store transcription in history

        Returns:
            Transcription response with text and metadata

        Raises:
            APIError: Various errors including 413 (file too large)
        """
        files = {"file": (filename, file_content)}
        params: dict[str, str | int | bool] = {"language": language, "store": store}

        response = self._client.post(
            f"{self.base_url}/transcribe",
            headers={"Authorization": f"Bearer {self.access_token}"}
            if self.access_token
            else {},
            files=files,
            params=params,
        )
        return self._handle_response(response)

    def transcribe_audio_with_diarization(
        self,
        file_content: bytes,
        filename: str,
        language: str = "en-US",
        max_speakers: int = 5,
    ) -> dict[str, Any]:
        """Transcribe an audio file with speaker diarization.

        Args:
            file_content: Audio file content as bytes
            filename: Original filename
            language: Language code for transcription
            max_speakers: Maximum number of speakers to identify

        Returns:
            Diarized transcription response with speaker segments

        Raises:
            APIError: Various errors including 413 (file too large)
        """
        files = {"file": (filename, file_content)}
        params: dict[str, str | int] = {"language": language, "max_speakers": max_speakers}

        response = self._client.post(
            f"{self.base_url}/transcribe/diarize",
            headers={"Authorization": f"Bearer {self.access_token}"}
            if self.access_token
            else {},
            files=files,
            params=params,
        )
        return self._handle_response(response)

    # Transcription history endpoints
    def list_transcriptions(
        self, page: int = 1, per_page: int = 20
    ) -> dict[str, Any]:
        """List the current user's transcriptions.

        Args:
            page: Page number (1-indexed)
            per_page: Results per page

        Returns:
            Paginated list of transcriptions

        Raises:
            APIError: 401 if not authenticated
        """
        response = self._client.get(
            f"{self.base_url}/transcriptions",
            headers=self._get_headers(),
            params={"page": page, "per_page": per_page},
        )
        return self._handle_response(response)

    def get_transcription(self, transcription_id: str) -> dict[str, Any]:
        """Get a specific transcription by ID.

        Args:
            transcription_id: Transcription ID

        Returns:
            Transcription record

        Raises:
            APIError: 401 if not authenticated, 404 if not found
        """
        response = self._client.get(
            f"{self.base_url}/transcriptions/{transcription_id}",
            headers=self._get_headers(),
        )
        return self._handle_response(response)

    def delete_transcription(self, transcription_id: str) -> None:
        """Delete a specific transcription.

        Args:
            transcription_id: Transcription ID

        Raises:
            APIError: 401 if not authenticated, 404 if not found
        """
        response = self._client.delete(
            f"{self.base_url}/transcriptions/{transcription_id}",
            headers=self._get_headers(),
        )
        if response.status_code >= 400:
            try:
                error_detail = response.json().get("detail", response.text)
            except Exception:
                error_detail = response.text or "Unknown error"
            raise APIError(response.status_code, error_detail)

    def close(self) -> None:
        """Close the HTTP client connection."""
        self._client.close()
