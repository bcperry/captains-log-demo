"""Shared fixtures for integration tests.

This module provides common fixtures for integration testing,
including storage isolation and configurable service mocking.
"""

import os
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import auth_router, health_router, transcribe_router, transcriptions_router
from auth import AuthenticatedUser
from auth.dependencies import get_current_user_azure
from storage.blob import clear_in_memory_blobs


def is_using_real_services() -> bool:
    """Check if integration tests should use real Azure services.

    Set INTEGRATION_USE_REAL_SERVICES=true to run against real services.
    """
    return os.environ.get("INTEGRATION_USE_REAL_SERVICES", "false").lower() == "true"


@pytest.fixture(scope="function", autouse=True)
def isolate_storage() -> Generator[None, None, None]:
    """Ensure storage is isolated for each test.

    Clears in-memory blob storage before and after each test.
    """
    clear_in_memory_blobs()
    yield
    clear_in_memory_blobs()


@pytest.fixture
def integration_app() -> FastAPI:
    """Create a FastAPI application with all routers for integration testing."""
    app = FastAPI(title="Captain's Log API - Integration Tests")
    app.include_router(auth_router)
    app.include_router(health_router)
    app.include_router(transcribe_router)
    app.include_router(transcriptions_router)
    return app


@pytest.fixture
def test_user() -> AuthenticatedUser:
    """Create a test user for authenticated endpoints."""
    return AuthenticatedUser(
        oid="integration-test-user-001",
        email="integration@test.local",
        name="Integration Test User",
        preferred_username="integration@test.local",
        tenant_id="integration-test-tenant",
        roles=["User"],
    )


@pytest.fixture
def another_test_user() -> AuthenticatedUser:
    """Create another test user for multi-user scenarios."""
    return AuthenticatedUser(
        oid="integration-test-user-002",
        email="another@test.local",
        name="Another Test User",
        preferred_username="another@test.local",
        tenant_id="integration-test-tenant",
        roles=["User"],
    )


@pytest.fixture
def authenticated_client(
    integration_app: FastAPI,
    test_user: AuthenticatedUser,
) -> TestClient:
    """Create an authenticated test client for integration testing."""
    integration_app.dependency_overrides[get_current_user_azure] = lambda: test_user
    return TestClient(integration_app)


@pytest.fixture
def mock_speech_client() -> MagicMock:
    """Create a mock Speech client for transcription testing."""
    from speech.client import SpeechClient

    client = MagicMock(spec=SpeechClient)
    client.create_audio_config_from_file.return_value = MagicMock()
    client.recognize_once.return_value = "Integration test transcribed text."
    client.recognize_continuous.return_value = "Integration test transcribed text."
    client.recognize_continuous_with_diarization.return_value = [
        {
            "speaker_id": "Speaker1",
            "text": "Hello from integration test.",
            "start_time_ms": 0,
            "end_time_ms": 2000,
        },
        {
            "speaker_id": "Speaker2",
            "text": "Reply from speaker two.",
            "start_time_ms": 2500,
            "end_time_ms": 4500,
        },
    ]
    return client


@pytest.fixture(autouse=True)
def mock_converter() -> Generator[MagicMock, None, None]:
    """Mock the audio converter to avoid ffmpeg dependency in tests."""
    with patch("api.transcribe.needs_conversion", return_value=False), \
         patch("api.transcribe.convert_to_wav") as mock_convert:
        mock_convert.return_value = "/tmp/mock_converted.wav"
        yield mock_convert


@pytest.fixture
def authenticated_client_with_speech(
    integration_app: FastAPI,
    test_user: AuthenticatedUser,
    mock_speech_client: MagicMock,
) -> TestClient:
    """Create an authenticated test client with mocked speech services."""
    from api.transcribe import get_speech_service

    integration_app.dependency_overrides[get_current_user_azure] = lambda: test_user
    integration_app.dependency_overrides[get_speech_service] = lambda: mock_speech_client

    return TestClient(integration_app)
