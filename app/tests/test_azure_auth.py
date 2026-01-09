"""Tests for Azure Entra ID authentication using fastapi-azure-auth.

Tests the azure_auth module which provides OAuth2 authentication
with support for Azure Commercial and Government clouds.
"""

from unittest.mock import patch

import pytest

from auth.azure_auth import (
    create_azure_scheme,
    get_authorization_url,
    get_azure_authority_host,
    get_azure_scheme,
    get_openid_config_url,
    get_token_url,
)
from config.settings import AzureCloud, Settings


class TestGetAzureAuthorityHost:
    """Tests for get_azure_authority_host function."""

    def test_government_cloud_returns_us_endpoint(self) -> None:
        """Government cloud should use login.microsoftonline.us."""
        result = get_azure_authority_host(AzureCloud.GOVERNMENT)
        assert result == "https://login.microsoftonline.us"

    def test_commercial_cloud_returns_com_endpoint(self) -> None:
        """Commercial cloud should use login.microsoftonline.com."""
        result = get_azure_authority_host(AzureCloud.COMMERCIAL)
        assert result == "https://login.microsoftonline.com"

    def test_local_cloud_returns_com_endpoint(self) -> None:
        """Local environment should default to commercial endpoint."""
        result = get_azure_authority_host(AzureCloud.LOCAL)
        assert result == "https://login.microsoftonline.com"


class TestGetOpenidConfigUrl:
    """Tests for get_openid_config_url function."""

    def test_government_cloud_openid_url(self) -> None:
        """Government cloud should use .us domain for OpenID config."""
        tenant_id = "test-tenant-123"
        result = get_openid_config_url(tenant_id, AzureCloud.GOVERNMENT)
        expected = f"https://login.microsoftonline.us/{tenant_id}/v2.0/.well-known/openid-configuration"
        assert result == expected

    def test_commercial_cloud_openid_url(self) -> None:
        """Commercial cloud should use .com domain for OpenID config."""
        tenant_id = "test-tenant-456"
        result = get_openid_config_url(tenant_id, AzureCloud.COMMERCIAL)
        expected = f"https://login.microsoftonline.com/{tenant_id}/v2.0/.well-known/openid-configuration"
        assert result == expected

    def test_local_cloud_openid_url(self) -> None:
        """Local environment should default to commercial OpenID config."""
        tenant_id = "local-tenant"
        result = get_openid_config_url(tenant_id, AzureCloud.LOCAL)
        expected = f"https://login.microsoftonline.com/{tenant_id}/v2.0/.well-known/openid-configuration"
        assert result == expected


class TestGetAuthorizationUrl:
    """Tests for get_authorization_url function."""

    def test_government_authorization_url(self) -> None:
        """Government cloud should use .us domain for authorization."""
        tenant_id = "gov-tenant"
        result = get_authorization_url(tenant_id, AzureCloud.GOVERNMENT)
        expected = f"https://login.microsoftonline.us/{tenant_id}/oauth2/v2.0/authorize"
        assert result == expected

    def test_commercial_authorization_url(self) -> None:
        """Commercial cloud should use .com domain for authorization."""
        tenant_id = "com-tenant"
        result = get_authorization_url(tenant_id, AzureCloud.COMMERCIAL)
        expected = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/authorize"
        assert result == expected


class TestGetTokenUrl:
    """Tests for get_token_url function."""

    def test_government_token_url(self) -> None:
        """Government cloud should use .us domain for token endpoint."""
        tenant_id = "gov-tenant"
        result = get_token_url(tenant_id, AzureCloud.GOVERNMENT)
        expected = f"https://login.microsoftonline.us/{tenant_id}/oauth2/v2.0/token"
        assert result == expected

    def test_commercial_token_url(self) -> None:
        """Commercial cloud should use .com domain for token endpoint."""
        tenant_id = "com-tenant"
        result = get_token_url(tenant_id, AzureCloud.COMMERCIAL)
        expected = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
        assert result == expected


class TestCreateAzureScheme:
    """Tests for create_azure_scheme function."""

    def test_raises_value_error_without_client_id(self) -> None:
        """Should raise ValueError when azure_client_id is missing."""
        settings = Settings(
            azure_tenant_id="test-tenant",
            azure_client_id=None,
            _env_file=None,  # type: ignore[call-arg]
        )
        with pytest.raises(ValueError, match="AZURE_CLIENT_ID is required"):
            create_azure_scheme(settings)

    def test_raises_value_error_without_tenant_id(self) -> None:
        """Should raise ValueError when azure_tenant_id is missing."""
        settings = Settings(
            azure_tenant_id=None,
            azure_client_id="test-client",
            _env_file=None,  # type: ignore[call-arg]
        )
        with pytest.raises(ValueError, match="AZURE_TENANT_ID is required"):
            create_azure_scheme(settings)

    def test_creates_scheme_with_valid_config(self) -> None:
        """Should create scheme when all required config is provided."""
        settings = Settings(
            azure_tenant_id="test-tenant-id",
            azure_client_id="test-client-id",
            azure_cloud=AzureCloud.COMMERCIAL,
            _env_file=None,  # type: ignore[call-arg]
        )
        scheme = create_azure_scheme(settings)

        assert scheme is not None
        assert scheme.app_client_id == "test-client-id"
        # Verify authorization URL is set correctly
        assert "test-tenant-id" in scheme.authorization_url
        assert "login.microsoftonline.com" in scheme.authorization_url

    def test_creates_scheme_for_government_cloud(self) -> None:
        """Should create scheme with Government cloud endpoints."""
        settings = Settings(
            azure_tenant_id="gov-tenant-id",
            azure_client_id="gov-client-id",
            azure_cloud=AzureCloud.GOVERNMENT,
            _env_file=None,  # type: ignore[call-arg]
        )
        scheme = create_azure_scheme(settings)

        assert scheme is not None
        # Verify Government URLs
        assert "login.microsoftonline.us" in scheme.authorization_url
        assert "login.microsoftonline.us" in scheme.token_url
        # Verify OpenID config URL is overridden for Government
        assert "login.microsoftonline.us" in scheme.openid_config.config_url

    def test_scheme_has_correct_scopes(self) -> None:
        """Should configure scheme with correct API scope."""
        settings = Settings(
            azure_tenant_id="test-tenant",
            azure_client_id="test-client-id",
            azure_cloud=AzureCloud.COMMERCIAL,
            _env_file=None,  # type: ignore[call-arg]
        )
        scheme = create_azure_scheme(settings)

        # Verify scopes include user_impersonation
        expected_scope = "api://test-client-id/user_impersonation"
        assert expected_scope in scheme.oauth.model.flows.authorizationCode.scopes


class TestGetAzureScheme:
    """Tests for get_azure_scheme caching function."""

    def test_returns_cached_scheme(self) -> None:
        """Should return the same cached scheme on multiple calls."""
        # Clear cache first
        get_azure_scheme.cache_clear()

        with patch("auth.azure_auth.get_settings") as mock_settings:
            mock_settings.return_value = Settings(
                azure_tenant_id="cache-test-tenant",
                azure_client_id="cache-test-client",
                azure_cloud=AzureCloud.COMMERCIAL,
                _env_file=None,  # type: ignore[call-arg]
            )

            scheme1 = get_azure_scheme()
            scheme2 = get_azure_scheme()

            assert scheme1 is scheme2
            # Settings should only be called once due to caching
            assert mock_settings.call_count == 1

        # Clean up
        get_azure_scheme.cache_clear()


class TestSettingsIntegration:
    """Integration tests for Settings with azure_auth."""

    def test_effective_openapi_client_id_uses_azure_client_id(self) -> None:
        """Should fall back to azure_client_id when openapi_client_id not set."""
        settings = Settings(
            azure_client_id="backend-client",
            openapi_client_id=None,
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.effective_openapi_client_id == "backend-client"

    def test_effective_openapi_client_id_uses_explicit_value(self) -> None:
        """Should use openapi_client_id when explicitly set."""
        settings = Settings(
            azure_client_id="backend-client",
            openapi_client_id="swagger-client",
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.effective_openapi_client_id == "swagger-client"

    def test_api_scope_format(self) -> None:
        """Should format API scope correctly."""
        settings = Settings(
            azure_client_id="my-app-id",
            _env_file=None,  # type: ignore[call-arg]
        )
        assert settings.api_scope == "api://my-app-id/user_impersonation"
