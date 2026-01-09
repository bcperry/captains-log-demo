"""Tests for the health check API endpoints."""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.health import (
    DependencyStatus,
    HealthResponse,
    ReadinessResponse,
    check_cosmos_health,
    check_speech_health,
    router,
)


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
        assert data["version"] == "0.6.0"

    def test_always_returns_200(self, client: TestClient) -> None:
        """Test that health endpoint always returns 200 for liveness."""
        # Multiple calls should all succeed
        for _ in range(3):
            response = client.get("/health")
            assert response.status_code == 200


class TestReadinessEndpoint:
    """Tests for GET /ready endpoint."""

    def test_returns_ready_with_dependencies(self, client: TestClient) -> None:
        """Test that readiness endpoint returns dependency statuses."""
        with patch("api.health.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.is_cosmos_configured.return_value = False
            mock_settings.return_value.is_speech_configured.return_value = False

            response = client.get("/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ready"
            assert "dependencies" in data
            assert len(data["dependencies"]) == 2

    def test_returns_not_ready_when_dependency_unhealthy(
        self, client: TestClient
    ) -> None:
        """Test that readiness returns not_ready when a dependency fails."""
        with patch("api.health.check_cosmos_health") as mock_cosmos, patch(
            "api.health.check_speech_health"
        ) as mock_speech:
            mock_cosmos.return_value = DependencyStatus(
                name="cosmos_db", healthy=True, message="OK"
            )
            mock_speech.return_value = DependencyStatus(
                name="speech_services", healthy=False, message="Connection failed"
            )

            response = client.get("/ready")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "not_ready"


class TestCheckCosmosHealth:
    """Tests for check_cosmos_health function."""

    def test_returns_healthy_when_not_configured(self) -> None:
        """Test returns healthy with message when Cosmos not configured."""
        with patch("api.health.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.is_cosmos_configured.return_value = False

            result = check_cosmos_health()

            assert result.healthy is True
            assert result.name == "cosmos_db"
            assert result.message is not None
            assert "not configured" in result.message.lower()

    def test_returns_healthy_when_configured(self) -> None:
        """Test returns healthy when Cosmos is configured."""
        with patch("api.health.get_settings") as mock_settings, patch(
            "api.health.get_cosmos_client"
        ) as mock_client:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.is_cosmos_configured.return_value = True
            mock_client.return_value = MagicMock()
            mock_client.return_value.is_configured.return_value = True

            result = check_cosmos_health()

            assert result.healthy is True
            assert result.name == "cosmos_db"

    def test_returns_unhealthy_on_exception(self) -> None:
        """Test returns unhealthy when an exception occurs."""
        with patch("api.health.get_settings") as mock_settings:
            mock_settings.side_effect = Exception("Connection error")

            result = check_cosmos_health()

            assert result.healthy is False
            assert result.message is not None
            assert "Connection error" in result.message


class TestCheckSpeechHealth:
    """Tests for check_speech_health function."""

    def test_returns_healthy_when_not_configured(self) -> None:
        """Test returns healthy with message when Speech not configured."""
        with patch("api.health.get_settings") as mock_settings:
            mock_settings.return_value = MagicMock()
            mock_settings.return_value.is_speech_configured.return_value = False

            result = check_speech_health()

            assert result.healthy is True
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
        assert response.version == "0.6.0"
        assert response.timestamp is not None

    def test_readiness_response_model(self) -> None:
        """Test ReadinessResponse model."""
        deps = [DependencyStatus(name="test", healthy=True)]
        response = ReadinessResponse(status="ready", dependencies=deps)
        assert response.status == "ready"
        assert len(response.dependencies) == 1
        assert response.timestamp is not None
