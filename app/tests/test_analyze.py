"""Tests for the analyze API endpoint."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai import OpenAIClient, OpenAIClientError, OpenAINotConfiguredError
from api.analyze import get_openai_service, router
from auth import AuthenticatedUser
from auth.dependencies import get_current_user_azure
from models.analysis import ActionItem, AnalysisResult, Priority, Sentiment


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
def mock_analysis_result() -> AnalysisResult:
    """Create a mock analysis result."""
    return AnalysisResult(
        summary="This is a test summary of the transcription.",
        keyPoints=["Point 1", "Point 2", "Point 3"],
        actionItems=[
            ActionItem(
                task="Complete the project",
                assignee="John",
                deadline="2024-01-15",
                priority=Priority.HIGH,
            ),
            ActionItem(
                task="Review documents",
                assignee=None,
                deadline=None,
                priority=Priority.MEDIUM,
            ),
        ],
        participants=["Alice", "Bob", "Charlie"],
        topics=["Project planning", "Budget review"],
        sentiment=Sentiment.POSITIVE,
        confidence=0.85,
    )


@pytest.fixture
def mock_openai_client(mock_analysis_result: AnalysisResult) -> MagicMock:
    """Create a mock OpenAI client."""
    client = MagicMock(spec=OpenAIClient)
    client.analyze_transcription.return_value = mock_analysis_result
    client.is_configured.return_value = True
    return client


@pytest.fixture
def client(
    app: FastAPI, mock_user: AuthenticatedUser, mock_openai_client: MagicMock
) -> TestClient:
    """Create a test client with mocked dependencies."""
    app.dependency_overrides[get_current_user_azure] = lambda: mock_user
    app.dependency_overrides[get_openai_service] = lambda: mock_openai_client
    return TestClient(app)


class TestAnalyzeEndpoint:
    """Tests for POST /analyze endpoint."""

    def test_analyze_returns_analysis_result(
        self, client: TestClient, mock_openai_client: MagicMock
    ) -> None:
        """Test that analyze endpoint returns analysis result."""
        response = client.post("/analyze", json={"text": "Test transcription text"})

        assert response.status_code == 200
        data = response.json()
        assert data["summary"] == "This is a test summary of the transcription."
        assert len(data["keyPoints"]) == 3
        assert len(data["actionItems"]) == 2
        assert data["sentiment"] == "positive"
        assert data["confidence"] == 0.85

        mock_openai_client.analyze_transcription.assert_called_once_with(
            "Test transcription text", None
        )

    def test_analyze_returns_key_points(
        self, client: TestClient, mock_analysis_result: AnalysisResult
    ) -> None:
        """Test that key points are returned correctly."""
        response = client.post("/analyze", json={"text": "Test text"})

        assert response.status_code == 200
        data = response.json()
        assert data["keyPoints"] == ["Point 1", "Point 2", "Point 3"]

    def test_analyze_returns_action_items(
        self, client: TestClient, mock_analysis_result: AnalysisResult
    ) -> None:
        """Test that action items are returned correctly."""
        response = client.post("/analyze", json={"text": "Test text"})

        assert response.status_code == 200
        data = response.json()
        assert len(data["actionItems"]) == 2
        assert data["actionItems"][0]["task"] == "Complete the project"
        assert data["actionItems"][0]["assignee"] == "John"
        assert data["actionItems"][0]["priority"] == "high"

    def test_analyze_returns_participants(
        self, client: TestClient, mock_analysis_result: AnalysisResult
    ) -> None:
        """Test that participants are returned correctly."""
        response = client.post("/analyze", json={"text": "Test text"})

        assert response.status_code == 200
        data = response.json()
        assert data["participants"] == ["Alice", "Bob", "Charlie"]

    def test_analyze_returns_topics(
        self, client: TestClient, mock_analysis_result: AnalysisResult
    ) -> None:
        """Test that topics are returned correctly."""
        response = client.post("/analyze", json={"text": "Test text"})

        assert response.status_code == 200
        data = response.json()
        assert data["topics"] == ["Project planning", "Budget review"]

    def test_analyze_with_diarized_transcript(
        self,
        client: TestClient,
        mock_openai_client: MagicMock,
        mock_analysis_result: AnalysisResult,
    ) -> None:
        """Test that diarized transcript is passed to OpenAI client."""
        diarized = "Speaker 1 [00:00:01]: Hello\nSpeaker 2 [00:00:05]: Hi there"
        response = client.post(
            "/analyze",
            json={"text": "Hello Hi there", "diarized_transcript": diarized},
        )

        assert response.status_code == 200
        mock_openai_client.analyze_transcription.assert_called_once_with(
            "Hello Hi there", diarized
        )

    def test_analyze_requires_text(self, client: TestClient) -> None:
        """Test that text field is required."""
        response = client.post("/analyze", json={})

        assert response.status_code == 422

    def test_analyze_rejects_empty_text(self, client: TestClient) -> None:
        """Test that empty text is rejected."""
        response = client.post("/analyze", json={"text": ""})

        assert response.status_code == 422


class TestAnalyzeAuthentication:
    """Tests for analyze endpoint authentication."""

    def test_requires_authentication(self, app: FastAPI) -> None:
        """Test that endpoint requires authentication."""
        # Don't override auth dependency
        app.dependency_overrides.pop(get_current_user_azure, None)

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/analyze", json={"text": "Test text"})

        assert response.status_code == 401


class TestAnalyzeErrorHandling:
    """Tests for analyze endpoint error handling."""

    def test_returns_503_when_openai_not_configured(
        self, app: FastAPI, mock_user: AuthenticatedUser
    ) -> None:
        """Test that 503 is returned when OpenAI is not configured."""
        mock_client = MagicMock(spec=OpenAIClient)
        mock_client.analyze_transcription.side_effect = OpenAINotConfiguredError(
            "Azure OpenAI is not configured"
        )

        app.dependency_overrides[get_current_user_azure] = lambda: mock_user
        app.dependency_overrides[get_openai_service] = lambda: mock_client

        client = TestClient(app)
        response = client.post("/analyze", json={"text": "Test text"})

        assert response.status_code == 503
        assert "not configured" in response.json()["detail"]

    def test_returns_503_on_openai_client_error(
        self, app: FastAPI, mock_user: AuthenticatedUser
    ) -> None:
        """Test that 503 is returned on OpenAI client errors."""
        mock_client = MagicMock(spec=OpenAIClient)
        mock_client.analyze_transcription.side_effect = OpenAIClientError(
            "API call failed"
        )

        app.dependency_overrides[get_current_user_azure] = lambda: mock_user
        app.dependency_overrides[get_openai_service] = lambda: mock_client

        client = TestClient(app)
        response = client.post("/analyze", json={"text": "Test text"})

        assert response.status_code == 503
        assert "Analysis failed" in response.json()["detail"]

    def test_returns_500_on_unexpected_error(
        self, app: FastAPI, mock_user: AuthenticatedUser
    ) -> None:
        """Test that 500 is returned on unexpected errors."""
        mock_client = MagicMock(spec=OpenAIClient)
        mock_client.analyze_transcription.side_effect = RuntimeError("Unexpected error")

        app.dependency_overrides[get_current_user_azure] = lambda: mock_user
        app.dependency_overrides[get_openai_service] = lambda: mock_client

        client = TestClient(app)
        response = client.post("/analyze", json={"text": "Test text"})

        assert response.status_code == 500


class TestOpenAIClientUnit:
    """Unit tests for the OpenAI client."""

    def test_is_configured_returns_false_when_missing_config(self) -> None:
        """Test that is_configured returns False when config is missing."""
        with patch("ai.get_settings") as mock_settings:
            mock_settings.return_value.azure_openai_endpoint = None
            mock_settings.return_value.azure_openai_key = None
            mock_settings.return_value.azure_openai_deployment = None
            mock_settings.return_value.azure_openai_api_version = "2024-02-15-preview"
            client = OpenAIClient(endpoint=None, api_key=None, deployment=None)
            assert client.is_configured() is False

    def test_is_configured_returns_true_when_all_set(self) -> None:
        """Test that is_configured returns True when all config is set."""
        client = OpenAIClient(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            deployment="test-deployment",
        )
        assert client.is_configured() is True

    def test_parse_analysis_response_handles_markdown(self) -> None:
        """Test that markdown formatting is stripped from response."""
        client = OpenAIClient(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            deployment="test-deployment",
        )

        json_response = '''```json
{
    "summary": "Test summary",
    "key_points": ["Point 1"],
    "action_items": [],
    "participants": [],
    "topics": [],
    "sentiment": "neutral",
    "confidence": 0.9
}
```'''

        result = client._parse_analysis_response(json_response)
        assert result.summary == "Test summary"
        assert result.confidence == 0.9

    def test_parse_analysis_response_raises_on_invalid_json(self) -> None:
        """Test that invalid JSON raises OpenAIClientError."""
        client = OpenAIClient(
            endpoint="https://test.openai.azure.com",
            api_key="test-key",
            deployment="test-deployment",
        )

        with pytest.raises(OpenAIClientError) as exc_info:
            client._parse_analysis_response("not valid json")

        assert "Failed to parse" in str(exc_info.value)
