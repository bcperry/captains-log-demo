"""Tests for the health check API endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.health import (
    DependencyStatus,
    HealthResponse,
    ReadinessResponse,
    check_openai_health,
    check_speech_health,
    router,
)
from config.version import API_VERSION


@pytest.fixture
def app() -> FastAPI:
    """Create a test FastAPI application."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


class TestHealthEndpoint:
    """Tests for GET /health endpoint."""

    def test_returns_healthy(self, client: TestClient) -> None:
        """Test that health endpoint returns healthy status."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["version"] == API_VERSION

    def test_always_returns_200(self, client: TestClient) -> None:
        """Test that health endpoint always returns 200 for liveness."""
        # Multiple calls should all succeed
        for _ in range(3):
            response = client.get("/health")
            assert response.status_code == 200


class TestReadinessEndpoint:
    """Tests for GET /ready endpoint."""

    def test_returns_not_ready_when_speech_not_configured(
        self, client: TestClient
    ) -> None:
        """Test that readiness returns not_ready when speech is not configured."""
        with patch("api.health.get_settings") as mock_settings, patch(
            "api.health.check_openai_health", new_callable=AsyncMock
        ) as mock_openai:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.is_speech_configured.return_value = False

            # Mock the async openai health check
            mock_openai.return_value = DependencyStatus(
                name="azure_openai", healthy=False, configured=False, message="Not configured"
            )

            response = client.get("/ready")

            assert response.status_code == 200
            data = response.json()
            # Speech not configured returns healthy=False, so overall status is not_ready
            assert data["status"] == "not_ready"
            assert "dependencies" in data
            assert len(data["dependencies"]) == 2  # speech and openai

    def test_returns_not_ready_when_dependency_unhealthy(
        self, client: TestClient
    ) -> None:
        """Test that readiness returns not_ready when a dependency fails."""
        with patch(
            "api.health.check_speech_health"
        ) as mock_speech, patch(
            "api.health.check_openai_health", new_callable=AsyncMock
        ) as mock_openai:
            mock_speech.return_value = DependencyStatus(
                name="speech_services", healthy=False, message="Connection failed"
            )
            # Mock the async openai health check
            mock_openai.return_value = DependencyStatus(
                name="azure_openai", healthy=True, configured=True, message="OK"
            )

            response = client.get("/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "not_ready"


class TestCheckSpeechHealth:
    """Tests for check_speech_health function."""

    def test_returns_unhealthy_when_not_configured(self) -> None:
        """Test returns unhealthy with configured=False when Speech not configured."""
        with patch("api.health.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.is_speech_configured.return_value = False

            result = check_speech_health()

            assert result.healthy is False
            assert result.configured is False
            assert result.name == "speech_services"
            assert result.message is not None
            assert "not configured" in result.message.lower()

    def test_returns_healthy_when_available(self) -> None:
        """Test returns healthy when Speech Services is available."""
        with patch("api.health.get_settings") as mock_settings, patch(
            "api.health.get_speech_client"
        ) as mock_client:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.is_speech_configured.return_value = True
            mock_client.return_value = MagicMock()
            mock_client.return_value.is_available.return_value = True

            result = check_speech_health()

            assert result.healthy is True
            assert result.configured is True
            assert result.name == "speech_services"

    def test_returns_unhealthy_when_unavailable(self) -> None:
        """Test returns unhealthy when Speech Services is unavailable."""
        with patch("api.health.get_settings") as mock_settings, patch(
            "api.health.get_speech_client"
        ) as mock_client:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.is_speech_configured.return_value = True
            mock_client.return_value = None

            result = check_speech_health()

            assert result.healthy is False
            assert result.message is not None
            assert "unavailable" in result.message.lower()

    def test_returns_unhealthy_on_exception(self) -> None:
        """Test returns unhealthy when an exception occurs."""
        with patch("api.health.get_settings") as mock_settings:
            mock_settings.side_effect = Exception("Service error")

            result = check_speech_health()

            assert result.healthy is False
            assert result.message is not None
            assert "Service error" in result.message


class TestHealthModels:
    """Tests for health check models."""

    def test_dependency_status_model(self) -> None:
        """Test DependencyStatus model."""
        status = DependencyStatus(name="test", healthy=True, message="OK")
        assert status.name == "test"
        assert status.healthy is True
        assert status.message == "OK"

    def test_dependency_status_without_message(self) -> None:
        """Test DependencyStatus model without message."""
        status = DependencyStatus(name="test", healthy=True)
        assert status.message is None

    def test_health_response_model(self) -> None:
        """Test HealthResponse model."""
        response = HealthResponse(status="healthy")
        assert response.status == "healthy"
        assert response.version == API_VERSION
        assert response.timestamp is not None

    def test_readiness_response_model(self) -> None:
        """Test ReadinessResponse model."""
        deps = [DependencyStatus(name="test", healthy=True)]
        response = ReadinessResponse(status="ready", dependencies=deps)
        assert response.status == "ready"
        assert len(response.dependencies) == 1
        assert response.timestamp is not None


class TestCheckOpenAIHealth:
    """Tests for check_openai_health function."""

    @pytest.mark.asyncio
    async def test_returns_unhealthy_when_not_configured(self) -> None:
        """Test returns unhealthy with configured=False when OpenAI not configured."""
        with patch("api.health.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.is_openai_configured.return_value = False
            mock_settings.return_value.azure_openai_endpoint = None
            mock_settings.return_value.azure_openai_key = None
            mock_settings.return_value.azure_openai_deployment = None

            result = await check_openai_health()

            assert result.healthy is False
            assert result.configured is False
            assert result.name == "azure_openai"
            assert result.message is not None
            assert "not configured" in result.message.lower()

    @pytest.mark.asyncio
    async def test_returns_healthy_when_api_succeeds(self) -> None:
        """Test returns healthy when API call succeeds."""
        with patch("api.health.get_settings") as mock_settings, patch(
            "api.health.httpx.AsyncClient"
        ) as mock_client:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.is_openai_configured.return_value = True
            mock_settings.return_value.azure_openai_endpoint = "https://test.openai.azure.us"
            mock_settings.return_value.azure_openai_key = "test-key"
            mock_settings.return_value.azure_openai_deployment = "gpt-4"
            mock_settings.return_value.azure_openai_api_version = "2024-02-15-preview"

            # Mock successful response with models data containing the deployment
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"data": [{"id": "gpt-4"}, {"id": "gpt-35-turbo"}]}
            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            result = await check_openai_health()

            assert result.healthy is True
            assert result.configured is True
            assert result.name == "azure_openai"
            assert result.message == "Connection successful"

    @pytest.mark.asyncio
    async def test_returns_unhealthy_on_401(self) -> None:
        """Test returns unhealthy with invalid key message on 401."""
        with patch("api.health.get_settings") as mock_settings, patch(
            "api.health.httpx.AsyncClient"
        ) as mock_client:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.is_openai_configured.return_value = True
            mock_settings.return_value.azure_openai_endpoint = "https://test.openai.azure.us"
            mock_settings.return_value.azure_openai_key = "invalid-key"
            mock_settings.return_value.azure_openai_deployment = "gpt-4"
            mock_settings.return_value.azure_openai_api_version = "2024-02-15-preview"

            mock_response = MagicMock()
            mock_response.status_code = 401
            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            result = await check_openai_health()

            assert result.healthy is False
            assert result.configured is True
            assert "Invalid API key" in result.message

    @pytest.mark.asyncio
    async def test_returns_unhealthy_on_404(self) -> None:
        """Test returns unhealthy with deployment not found on 404."""
        with patch("api.health.get_settings") as mock_settings, patch(
            "api.health.httpx.AsyncClient"
        ) as mock_client:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.is_openai_configured.return_value = True
            mock_settings.return_value.azure_openai_endpoint = "https://test.openai.azure.us"
            mock_settings.return_value.azure_openai_key = "test-key"
            mock_settings.return_value.azure_openai_deployment = "nonexistent"
            mock_settings.return_value.azure_openai_api_version = "2024-02-15-preview"

            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_client_instance = AsyncMock()
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            result = await check_openai_health()

            assert result.healthy is False
            assert result.configured is True
            assert "not found" in result.message.lower()

    @pytest.mark.asyncio
    async def test_returns_unhealthy_on_timeout(self) -> None:
        """Test returns unhealthy on connection timeout."""
        with patch("api.health.get_settings") as mock_settings, patch(
            "api.health.httpx.AsyncClient"
        ) as mock_client:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.is_openai_configured.return_value = True
            mock_settings.return_value.azure_openai_endpoint = "https://test.openai.azure.us"
            mock_settings.return_value.azure_openai_key = "test-key"
            mock_settings.return_value.azure_openai_deployment = "gpt-4"
            mock_settings.return_value.azure_openai_api_version = "2024-02-15-preview"

            mock_client_instance = AsyncMock()
            mock_client_instance.get.side_effect = httpx.TimeoutException("Timeout")
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            result = await check_openai_health()

            assert result.healthy is False
            assert result.configured is True
            assert "timed out" in result.message.lower()

    @pytest.mark.asyncio
    async def test_returns_unhealthy_on_connect_error(self) -> None:
        """Test returns unhealthy on connection error."""
        with patch("api.health.get_settings") as mock_settings, patch(
            "api.health.httpx.AsyncClient"
        ) as mock_client:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.is_openai_configured.return_value = True
            mock_settings.return_value.azure_openai_endpoint = "https://test.openai.azure.us"
            mock_settings.return_value.azure_openai_key = "test-key"
            mock_settings.return_value.azure_openai_deployment = "gpt-4"
            mock_settings.return_value.azure_openai_api_version = "2024-02-15-preview"

            mock_client_instance = AsyncMock()
            mock_client_instance.get.side_effect = httpx.ConnectError("Connection refused")
            mock_client.return_value.__aenter__.return_value = mock_client_instance

            result = await check_openai_health()

            assert result.healthy is False
            assert result.configured is True
            assert "unreachable" in result.message.lower()
