"""Tests for Azure Entra ID authentication module."""

import time
from unittest.mock import MagicMock, patch

import jwt
import pytest

from auth.entra import (
    AuthenticatedUser,
    EntraAuthError,
    EntraTokenValidator,
    MissingTokenError,
    TokenExpiredError,
    TokenValidationError,
    get_entra_validator,
)
from config.settings import AzureCloud, Settings


@pytest.fixture
def government_settings() -> Settings:
    """Create Settings for Azure Government environment."""
    return Settings(
        azure_cloud=AzureCloud.GOVERNMENT,
        azure_tenant_id="test-tenant-id",
        azure_client_id="test-client-id",
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def commercial_settings() -> Settings:
    """Create Settings for Azure Commercial environment."""
    return Settings(
        azure_cloud=AzureCloud.COMMERCIAL,
        azure_tenant_id="test-tenant-id",
        azure_client_id="test-client-id",
        _env_file=None,  # type: ignore[call-arg]
    )


@pytest.fixture
def validator(government_settings: Settings) -> EntraTokenValidator:
    """Create EntraTokenValidator for testing."""
    return EntraTokenValidator(settings=government_settings)


@pytest.fixture
def commercial_validator(commercial_settings: Settings) -> EntraTokenValidator:
    """Create EntraTokenValidator for commercial cloud testing."""
    return EntraTokenValidator(settings=commercial_settings)


class TestAuthenticatedUser:
    """Tests for AuthenticatedUser dataclass."""

    def test_from_claims_full(self) -> None:
        """Test creating AuthenticatedUser with all claims."""
        claims = {
            "oid": "user-oid-123",
            "email": "user@example.com",
            "name": "Test User",
            "preferred_username": "testuser@example.com",
            "tid": "tenant-123",
            "roles": ["User", "Admin"],
        }
        user = AuthenticatedUser.from_claims(claims)

        assert user.oid == "user-oid-123"
        assert user.email == "user@example.com"
        assert user.name == "Test User"
        assert user.preferred_username == "testuser@example.com"
        assert user.tenant_id == "tenant-123"
        assert user.roles == ["User", "Admin"]

    def test_from_claims_minimal(self) -> None:
        """Test creating AuthenticatedUser with minimal claims."""
        claims = {"oid": "user-oid-456"}
        user = AuthenticatedUser.from_claims(claims)

        assert user.oid == "user-oid-456"
        assert user.email is None
        assert user.name is None
        assert user.preferred_username is None
        assert user.tenant_id is None
        assert user.roles is None

    def test_from_claims_uses_sub_fallback(self) -> None:
        """Test that 'sub' claim is used as fallback for 'oid'."""
        claims = {"sub": "sub-claim-789"}
        user = AuthenticatedUser.from_claims(claims)

        assert user.oid == "sub-claim-789"

    def test_from_claims_email_from_preferred_username(self) -> None:
        """Test that email falls back to preferred_username."""
        claims = {"oid": "user-oid", "preferred_username": "user@example.org"}
        user = AuthenticatedUser.from_claims(claims)

        assert user.email == "user@example.org"


class TestEntraTokenValidator:
    """Tests for EntraTokenValidator class."""

    def test_government_issuer(self, validator: EntraTokenValidator) -> None:
        """Test issuer URL for Azure Government."""
        assert validator.issuer == "https://login.microsoftonline.us/test-tenant-id/v2.0"

    def test_commercial_issuer(self, commercial_validator: EntraTokenValidator) -> None:
        """Test issuer URL for Azure Commercial."""
        assert commercial_validator.issuer == "https://login.microsoftonline.com/test-tenant-id/v2.0"

    def test_audience(self, validator: EntraTokenValidator) -> None:
        """Test audience returns client ID."""
        assert validator.audience == "test-client-id"

    def test_government_jwks_uri(self, validator: EntraTokenValidator) -> None:
        """Test JWKS URI for Azure Government."""
        assert (
            validator.jwks_uri
            == "https://login.microsoftonline.us/test-tenant-id/discovery/v2.0/keys"
        )

    def test_commercial_jwks_uri(self, commercial_validator: EntraTokenValidator) -> None:
        """Test JWKS URI for Azure Commercial."""
        assert (
            commercial_validator.jwks_uri
            == "https://login.microsoftonline.com/test-tenant-id/discovery/v2.0/keys"
        )

    def test_extract_token_from_header_valid(self, validator: EntraTokenValidator) -> None:
        """Test extracting Bearer token from valid Authorization header."""
        token = validator.extract_token_from_header("Bearer abc123token")
        assert token == "abc123token"

    def test_extract_token_from_header_case_insensitive(
        self, validator: EntraTokenValidator
    ) -> None:
        """Test Bearer scheme is case-insensitive."""
        token = validator.extract_token_from_header("bearer abc123token")
        assert token == "abc123token"

        token = validator.extract_token_from_header("BEARER abc123token")
        assert token == "abc123token"

    def test_extract_token_from_header_missing(self, validator: EntraTokenValidator) -> None:
        """Test error when Authorization header is missing."""
        with pytest.raises(MissingTokenError, match="Authorization header is required"):
            validator.extract_token_from_header(None)

    def test_extract_token_from_header_empty(self, validator: EntraTokenValidator) -> None:
        """Test error when Authorization header is empty."""
        with pytest.raises(MissingTokenError, match="Authorization header is required"):
            validator.extract_token_from_header("")

    def test_extract_token_from_header_invalid_format(
        self, validator: EntraTokenValidator
    ) -> None:
        """Test error when Authorization header format is invalid."""
        with pytest.raises(MissingTokenError, match="Invalid Authorization header format"):
            validator.extract_token_from_header("Basic abc123")

    def test_extract_token_from_header_no_token(self, validator: EntraTokenValidator) -> None:
        """Test error when Bearer has no token."""
        with pytest.raises(MissingTokenError, match="Invalid Authorization header format"):
            validator.extract_token_from_header("Bearer")

    def test_validate_token_missing(self, validator: EntraTokenValidator) -> None:
        """Test error when token is missing."""
        with pytest.raises(MissingTokenError, match="Token is required"):
            validator.validate_token("")

    def test_validate_token_none(self, validator: EntraTokenValidator) -> None:
        """Test error when token is None."""
        with pytest.raises(MissingTokenError, match="Token is required"):
            validator.validate_token(None)  # type: ignore[arg-type]

    @patch("auth.entra.PyJWKClient")
    def test_validate_token_expired(
        self, mock_jwk_client_class: MagicMock, validator: EntraTokenValidator
    ) -> None:
        """Test error when token is expired."""
        # Create a mock signing key
        mock_signing_key = MagicMock()
        mock_signing_key.key = "mock-key"

        mock_client = MagicMock()
        mock_client.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwk_client_class.return_value = mock_client

        # Patch jwt.decode to raise ExpiredSignatureError
        with patch("auth.entra.jwt.decode") as mock_decode:
            mock_decode.side_effect = jwt.ExpiredSignatureError("Token expired")

            with pytest.raises(TokenExpiredError, match="Token has expired"):
                validator.validate_token("expired.token.here")

    @patch("auth.entra.PyJWKClient")
    def test_validate_token_invalid_audience(
        self, mock_jwk_client_class: MagicMock, validator: EntraTokenValidator
    ) -> None:
        """Test error when token has invalid audience."""
        mock_signing_key = MagicMock()
        mock_signing_key.key = "mock-key"

        mock_client = MagicMock()
        mock_client.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwk_client_class.return_value = mock_client

        with patch("auth.entra.jwt.decode") as mock_decode:
            mock_decode.side_effect = jwt.InvalidAudienceError("Invalid audience")

            with pytest.raises(TokenValidationError, match="Invalid audience"):
                validator.validate_token("invalid.aud.token")

    @patch("auth.entra.PyJWKClient")
    def test_validate_token_invalid_issuer(
        self, mock_jwk_client_class: MagicMock, validator: EntraTokenValidator
    ) -> None:
        """Test error when token has invalid issuer."""
        mock_signing_key = MagicMock()
        mock_signing_key.key = "mock-key"

        mock_client = MagicMock()
        mock_client.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwk_client_class.return_value = mock_client

        with patch("auth.entra.jwt.decode") as mock_decode:
            mock_decode.side_effect = jwt.InvalidIssuerError("Invalid issuer")

            with pytest.raises(TokenValidationError, match="Invalid issuer"):
                validator.validate_token("invalid.iss.token")

    @patch("auth.entra.PyJWKClient")
    def test_validate_token_decode_error(
        self, mock_jwk_client_class: MagicMock, validator: EntraTokenValidator
    ) -> None:
        """Test error when token cannot be decoded."""
        mock_signing_key = MagicMock()
        mock_signing_key.key = "mock-key"

        mock_client = MagicMock()
        mock_client.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwk_client_class.return_value = mock_client

        with patch("auth.entra.jwt.decode") as mock_decode:
            mock_decode.side_effect = jwt.DecodeError("Malformed token")

            with pytest.raises(TokenValidationError, match="Failed to decode token"):
                validator.validate_token("malformed.token")

    @patch("auth.entra.PyJWKClient")
    def test_validate_token_success(
        self, mock_jwk_client_class: MagicMock, validator: EntraTokenValidator
    ) -> None:
        """Test successful token validation."""
        mock_signing_key = MagicMock()
        mock_signing_key.key = "mock-key"

        mock_client = MagicMock()
        mock_client.get_signing_key_from_jwt.return_value = mock_signing_key
        mock_jwk_client_class.return_value = mock_client

        claims = {
            "oid": "user-object-id",
            "email": "user@example.com",
            "name": "Test User",
            "preferred_username": "testuser@example.com",
            "tid": "tenant-id",
            "roles": ["User"],
        }

        with patch("auth.entra.jwt.decode") as mock_decode:
            mock_decode.return_value = claims

            user = validator.validate_token("valid.token.here")

            assert user.oid == "user-object-id"
            assert user.email == "user@example.com"
            assert user.name == "Test User"
            assert user.tenant_id == "tenant-id"
            assert user.roles == ["User"]

    def test_jwks_client_caching(self, validator: EntraTokenValidator) -> None:
        """Test that JWKS client is cached and reused."""
        with patch("auth.entra.PyJWKClient") as mock_jwk_client_class:
            mock_client = MagicMock()
            mock_jwk_client_class.return_value = mock_client

            # First call creates client
            client1 = validator._get_jwks_client()
            # Second call should reuse
            client2 = validator._get_jwks_client()

            assert client1 is client2
            assert mock_jwk_client_class.call_count == 1

    def test_jwks_client_refresh_after_expiry(self, validator: EntraTokenValidator) -> None:
        """Test that JWKS client is refreshed after cache lifetime."""
        with patch("auth.entra.PyJWKClient") as mock_jwk_client_class:
            mock_client = MagicMock()
            mock_jwk_client_class.return_value = mock_client

            # First call
            validator._get_jwks_client()

            # Simulate time passage beyond cache lifetime
            validator._jwks_last_refresh = time.time() - (validator.JWKS_CACHE_LIFETIME + 1)

            # This should trigger a refresh
            validator._get_jwks_client()

            assert mock_jwk_client_class.call_count == 2


class TestExceptions:
    """Tests for custom exception hierarchy."""

    def test_entra_auth_error_is_base(self) -> None:
        """Test EntraAuthError is base exception."""
        assert issubclass(TokenValidationError, EntraAuthError)
        assert issubclass(TokenExpiredError, EntraAuthError)
        assert issubclass(MissingTokenError, EntraAuthError)

    def test_exceptions_can_be_raised_with_message(self) -> None:
        """Test exceptions carry message."""
        error = TokenValidationError("Custom message")
        assert str(error) == "Custom message"


class TestGetEntraValidator:
    """Tests for get_entra_validator cached function."""

    def test_returns_validator_instance(self) -> None:
        """Test get_entra_validator returns EntraTokenValidator."""
        # Clear the cache to ensure fresh instance
        get_entra_validator.cache_clear()

        with patch.object(EntraTokenValidator, "__init__", return_value=None):
            validator = get_entra_validator()
            assert isinstance(validator, EntraTokenValidator)

    def test_returns_same_instance(self) -> None:
        """Test get_entra_validator returns cached instance."""
        get_entra_validator.cache_clear()

        with patch.object(EntraTokenValidator, "__init__", return_value=None):
            validator1 = get_entra_validator()
            validator2 = get_entra_validator()
            assert validator1 is validator2
