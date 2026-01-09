"""Shared fixtures for integration tests.

This module provides common fixtures for integration testing,
including database isolation and configurable service mocking.
"""

import os
from typing import Generator
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from api import auth_router, health_router, transcribe_router, transcriptions_router
from auth import AuthenticatedUser
from auth.dependencies import get_current_user_azure
from db.cosmos import InMemoryCosmosClient, clear_in_memory_storage


def is_using_real_services() -> bool:
    """Check if integration tests should use real Azure services.

    Set INTEGRATION_USE_REAL_SERVICES=true to run against real services.
    """
    return os.environ.get("INTEGRATION_USE_REAL_SERVICES", "false").lower() == "true"


@pytest.fixture(scope="function", autouse=True)
def isolate_database() -> Generator[None, None, None]:
    """Ensure database is isolated for each test.

    Clears in-memory storage before and after each test.
    """
    clear_in_memory_storage()
    yield
    clear_in_memory_storage()


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
def mock_db() -> InMemoryCosmosClient:
    """Get the in-memory database client."""
    return InMemoryCosmosClient()


@pytest.fixture
def authenticated_client(
    integration_app: FastAPI,
    test_user: AuthenticatedUser,
    mock_db: InMemoryCosmosClient,
) -> TestClient:
    """Create an authenticated test client for integration testing."""
    from api.auth import get_db as auth_get_db
    from api.transcriptions import get_db as transcriptions_get_db

    integration_app.dependency_overrides[get_current_user_azure] = lambda: test_user
    integration_app.dependency_overrides[auth_get_db] = lambda: mock_db
    integration_app.dependency_overrides[transcriptions_get_db] = lambda: mock_db

    return TestClient(integration_app)


@pytest.fixture
def mock_speech_client() -> MagicMock:
    """Create a mock Speech client for transcription testing."""
    from speech.client import SpeechClient

    client = MagicMock(spec=SpeechClient)
    client.create_audio_config_from_file.return_value = MagicMock()
    client.recognize_once.return_value = "Integration test transcribed text."
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


@pytest.fixture
def authenticated_client_with_speech(
    integration_app: FastAPI,
    test_user: AuthenticatedUser,
    mock_db: InMemoryCosmosClient,
    mock_speech_client: MagicMock,
) -> TestClient:
    """Create an authenticated test client with mocked speech services."""
    from api.auth import get_db as auth_get_db
    from api.transcribe import get_db as transcribe_get_db
    from api.transcribe import get_speech_service
    from api.transcriptions import get_db as transcriptions_get_db

    integration_app.dependency_overrides[get_current_user_azure] = lambda: test_user
    integration_app.dependency_overrides[auth_get_db] = lambda: mock_db
    integration_app.dependency_overrides[transcribe_get_db] = lambda: mock_db
    integration_app.dependency_overrides[transcriptions_get_db] = lambda: mock_db
    integration_app.dependency_overrides[get_speech_service] = lambda: mock_speech_client

    return TestClient(integration_app)
