"""Tests for FastAPI authentication dependencies."""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from auth.dependencies import (
    get_current_user,
    get_optional_user,
    get_token_validator,
)
from auth.entra import (
    AuthenticatedUser,
    EntraTokenValidator,
    TokenExpiredError,
    TokenValidationError,
)
from config.settings import AzureCloud, Settings


@pytest.fixture
def mock_settings() -> Settings:
    """Create mock settings for testing."""
    return Settings(
        azure_cloud=AzureCloud.GOVERNMENT,
        azure_tenant_id="test-tenant-id",
        azure_client_id="test-client-id",
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def mock_validator(mock_settings: Settings) -> EntraTokenValidator:
    """Create mock validator for testing."""
    return EntraTokenValidator(settings=mock_settings)


@pytest.fixture
def mock_user() -> AuthenticatedUser:
    """Create mock authenticated user."""
    return AuthenticatedUser(
        oid="test-user-oid",
        email="test@example.com",
        name="Test User",
        preferred_username="testuser@example.com",
        tenant_id="test-tenant-id",
        roles=["User"],
    )


@pytest.fixture
def app(mock_validator: EntraTokenValidator, mock_user: AuthenticatedUser) -> FastAPI:
    """Create test FastAPI application."""
    app = FastAPI()

    @app.get("/protected")
    async def protected_route(user: AuthenticatedUser = pytest.importorskip("fastapi").Depends(get_current_user)) -> dict[str, Any]:
        return {"user_id": user.oid, "email": user.email}

    @app.get("/optional")
    async def optional_route(user: AuthenticatedUser | None = pytest.importorskip("fastapi").Depends(get_optional_user)) -> dict[str, Any]:
        if user:
            return {"authenticated": True, "user_id": user.oid}
        return {"authenticated": False}

    return app


class TestGetTokenValidator:
    """Tests for get_token_validator dependency."""

    def test_returns_validator_instance(self) -> None:
        """Test that get_token_validator returns EntraTokenValidator."""
        from auth.entra import get_entra_validator

        # Clear cache to ensure fresh instance
        get_entra_validator.cache_clear()

        with patch.object(EntraTokenValidator, "__init__", return_value=None):
            validator = get_token_validator()
            assert isinstance(validator, EntraTokenValidator)


class TestGetCurrentUser:
    """Tests for get_current_user dependency."""

    @pytest.fixture
    def test_app(self) -> FastAPI:
        """Create minimal test app."""
        from fastapi import Depends

        app = FastAPI()

        @app.get("/protected")
        async def protected_route(user: AuthenticatedUser = Depends(get_current_user)) -> dict[str, Any]:
            return {"user_id": user.oid, "email": user.email}

        return app

    def test_missing_authorization_header(self, test_app: FastAPI) -> None:
        """Test 401 returned when Authorization header is missing."""
        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/protected")

        assert response.status_code == 401
        assert "WWW-Authenticate" in response.headers
        assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_empty_authorization_header(self, test_app: FastAPI) -> None:
        """Test 401 returned when Authorization header is empty."""
        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/protected", headers={"Authorization": ""})

        assert response.status_code == 401

    def test_invalid_authorization_format(self, test_app: FastAPI) -> None:
        """Test 401 returned for invalid Authorization header format."""
        client = TestClient(test_app, raise_server_exceptions=False)
        response = client.get("/protected", headers={"Authorization": "Basic abc123"})

        assert response.status_code == 401
        assert "Invalid Authorization header format" in response.json()["detail"]

    def test_expired_token(self, test_app: FastAPI) -> None:
        """Test 401 returned for expired token with specific error description."""
        with patch("auth.dependencies.get_entra_validator") as mock_get_validator:
            mock_validator = MagicMock()
            mock_validator.extract_token_from_header.return_value = "expired.token"
            mock_validator.validate_token.side_effect = TokenExpiredError("Token has expired")
            mock_get_validator.return_value = mock_validator

            client = TestClient(test_app, raise_server_exceptions=False)
            response = client.get("/protected", headers={"Authorization": "Bearer expired.token"})

            assert response.status_code == 401
            assert "Token has expired" in response.json()["detail"]
            assert 'error="invalid_token"' in response.headers["WWW-Authenticate"]

    def test_invalid_token(self, test_app: FastAPI) -> None:
        """Test 401 returned for invalid token."""
        with patch("auth.dependencies.get_entra_validator") as mock_get_validator:
            mock_validator = MagicMock()
            mock_validator.extract_token_from_header.return_value = "invalid.token"
            mock_validator.validate_token.side_effect = TokenValidationError("Invalid signature")
            mock_get_validator.return_value = mock_validator

            client = TestClient(test_app, raise_server_exceptions=False)
            response = client.get("/protected", headers={"Authorization": "Bearer invalid.token"})

            assert response.status_code == 401
            assert "Invalid signature" in response.json()["detail"]

    def test_successful_authentication(self, test_app: FastAPI, mock_user: AuthenticatedUser) -> None:
        """Test successful authentication returns user data."""
        with patch("auth.dependencies.get_entra_validator") as mock_get_validator:
            mock_validator = MagicMock()
            mock_validator.extract_token_from_header.return_value = "valid.token"
            mock_validator.validate_token.return_value = mock_user
            mock_get_validator.return_value = mock_validator

            client = TestClient(test_app)
            response = client.get("/protected", headers={"Authorization": "Bearer valid.token"})

            assert response.status_code == 200
            assert response.json()["user_id"] == "test-user-oid"
            assert response.json()["email"] == "test@example.com"

    def test_user_stored_in_request_state(self, mock_user: AuthenticatedUser) -> None:
        """Test that authenticated user is stored in request.state."""
        from fastapi import Depends, Request

        app = FastAPI()

        @app.get("/check-state")
        async def check_state(request: Request, user: AuthenticatedUser = Depends(get_current_user)) -> dict[str, Any]:
            # Verify user is in request state
            return {"state_user_oid": request.state.user.oid}

        with patch("auth.dependencies.get_entra_validator") as mock_get_validator:
            mock_validator = MagicMock()
            mock_validator.extract_token_from_header.return_value = "valid.token"
            mock_validator.validate_token.return_value = mock_user
            mock_get_validator.return_value = mock_validator

            client = TestClient(app)
            response = client.get("/check-state", headers={"Authorization": "Bearer valid.token"})

            assert response.status_code == 200
            assert response.json()["state_user_oid"] == "test-user-oid"


class TestGetOptionalUser:
    """Tests for get_optional_user dependency."""

    @pytest.fixture
    def optional_app(self) -> FastAPI:
        """Create test app with optional auth endpoint."""
        from fastapi import Depends

        app = FastAPI()

        @app.get("/optional")
        async def optional_route(user: AuthenticatedUser | None = Depends(get_optional_user)) -> dict[str, Any]:
            if user:
                return {"authenticated": True, "user_id": user.oid}
            return {"authenticated": False}

        return app

    def test_no_authorization_header_returns_none(self, optional_app: FastAPI) -> None:
        """Test that missing Authorization header returns None (not 401)."""
        client = TestClient(optional_app)
        response = client.get("/optional")

        assert response.status_code == 200
        assert response.json()["authenticated"] is False

    def test_empty_authorization_returns_none(self, optional_app: FastAPI) -> None:
        """Test that empty Authorization header returns None."""
        client = TestClient(optional_app)
        response = client.get("/optional", headers={"Authorization": ""})

        assert response.status_code == 200
        assert response.json()["authenticated"] is False

    def test_invalid_token_returns_none(self, optional_app: FastAPI) -> None:
        """Test that invalid token returns None instead of 401."""
        with patch("auth.dependencies.get_entra_validator") as mock_get_validator:
            mock_validator = MagicMock()
            mock_validator.extract_token_from_header.return_value = "invalid.token"
            mock_validator.validate_token.side_effect = TokenValidationError("Invalid token")
            mock_get_validator.return_value = mock_validator

            client = TestClient(optional_app)
            response = client.get("/optional", headers={"Authorization": "Bearer invalid.token"})

            assert response.status_code == 200
            assert response.json()["authenticated"] is False

    def test_expired_token_returns_none(self, optional_app: FastAPI) -> None:
        """Test that expired token returns None instead of 401."""
        with patch("auth.dependencies.get_entra_validator") as mock_get_validator:
            mock_validator = MagicMock()
            mock_validator.extract_token_from_header.return_value = "expired.token"
            mock_validator.validate_token.side_effect = TokenExpiredError("Token expired")
            mock_get_validator.return_value = mock_validator

            client = TestClient(optional_app)
            response = client.get("/optional", headers={"Authorization": "Bearer expired.token"})

            assert response.status_code == 200
            assert response.json()["authenticated"] is False

    def test_valid_token_returns_user(self, optional_app: FastAPI, mock_user: AuthenticatedUser) -> None:
        """Test that valid token returns authenticated user."""
        with patch("auth.dependencies.get_entra_validator") as mock_get_validator:
            mock_validator = MagicMock()
            mock_validator.extract_token_from_header.return_value = "valid.token"
            mock_validator.validate_token.return_value = mock_user
            mock_get_validator.return_value = mock_validator

            client = TestClient(optional_app)
            response = client.get("/optional", headers={"Authorization": "Bearer valid.token"})

            assert response.status_code == 200
            assert response.json()["authenticated"] is True
            assert response.json()["user_id"] == "test-user-oid"


class TestAzureGovernmentSupport:
    """Tests for Azure Government endpoint support."""

    def test_government_endpoints_used(self) -> None:
        """Test that Azure Government endpoints are used for Government cloud."""
        settings = Settings(
            azure_cloud=AzureCloud.GOVERNMENT,
            azure_tenant_id="gov-tenant-id",
            azure_client_id="gov-client-id",
            _env_file=None,  # type: ignore[call-arg]
        )
        validator = EntraTokenValidator(settings=settings)

        assert "microsoftonline.us" in validator.issuer
        assert "microsoftonline.us" in validator.jwks_uri

    def test_commercial_endpoints_used(self) -> None:
        """Test that Azure Commercial endpoints are used for Commercial cloud."""
        settings = Settings(
            azure_cloud=AzureCloud.COMMERCIAL,
            azure_tenant_id="com-tenant-id",
            azure_client_id="com-client-id",
            _env_file=None,  # type: ignore[call-arg]
        )
        validator = EntraTokenValidator(settings=settings)

        assert "microsoftonline.com" in validator.issuer
        assert "microsoftonline.com" in validator.jwks_uri


class TestJWKSCaching:
    """Tests for JWKS key caching behavior."""

    def test_jwks_client_cached(self, mock_settings: Settings) -> None:
        """Test that JWKS client is cached between requests."""
        validator = EntraTokenValidator(settings=mock_settings)

        with patch("auth.entra.PyJWKClient") as mock_jwk_client_class:
            mock_client = MagicMock()
            mock_jwk_client_class.return_value = mock_client

            # First call creates client
            client1 = validator._get_jwks_client()
            # Second call should reuse cached client
            client2 = validator._get_jwks_client()

            assert client1 is client2
            assert mock_jwk_client_class.call_count == 1

    def test_jwks_client_uses_lifespan(self, mock_settings: Settings) -> None:
        """Test that JWKS client is created with lifespan parameter."""
        validator = EntraTokenValidator(settings=mock_settings)

        with patch("auth.entra.PyJWKClient") as mock_jwk_client_class:
            mock_client = MagicMock()
            mock_jwk_client_class.return_value = mock_client

            validator._get_jwks_client()

            mock_jwk_client_class.assert_called_once()
            call_kwargs = mock_jwk_client_class.call_args.kwargs
            assert call_kwargs.get("cache_keys") is True
            assert call_kwargs.get("lifespan") == validator.JWKS_CACHE_LIFETIME


class TestHTTPExceptionDetails:
    """Tests for HTTP exception response details."""

    def test_missing_token_response_format(self) -> None:
        """Test that missing token error has correct format."""
        from fastapi import Depends

        app = FastAPI()

        @app.get("/test")
        async def test_route(user: AuthenticatedUser = Depends(get_current_user)) -> dict[str, str]:
            return {"user": user.oid}

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/test")

        assert response.status_code == 401
        assert "detail" in response.json()
        assert response.headers["WWW-Authenticate"] == "Bearer"

    def test_expired_token_includes_error_description(self) -> None:
        """Test that expired token error includes error_description in WWW-Authenticate."""
        from fastapi import Depends

        app = FastAPI()

        @app.get("/test")
        async def test_route(user: AuthenticatedUser = Depends(get_current_user)) -> dict[str, str]:
            return {"user": user.oid}

        with patch("auth.dependencies.get_entra_validator") as mock_get_validator:
            mock_validator = MagicMock()
            mock_validator.extract_token_from_header.return_value = "expired.token"
            mock_validator.validate_token.side_effect = TokenExpiredError("Token expired")
            mock_get_validator.return_value = mock_validator

            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/test", headers={"Authorization": "Bearer expired.token"})

            assert response.status_code == 401
            www_auth = response.headers["WWW-Authenticate"]
            assert "error_description" in www_auth
            assert "Token has expired" in www_auth
