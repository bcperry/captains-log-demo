"""Tests for application settings configuration."""

import os
from typing import Any
from unittest.mock import patch

import pytest

from config.settings import AzureCloud, Settings, get_settings


def create_settings(**kwargs: Any) -> Settings:
    """Create Settings instance without loading .env file."""
    return Settings(_env_file=None, **kwargs)  # type: ignore[call-arg]


class TestAzureCloud:
    """Tests for AzureCloud enum."""

    def test_azure_cloud_values(self) -> None:
        """Test AzureCloud enum has expected values."""
        assert AzureCloud.COMMERCIAL.value == "commercial"
        assert AzureCloud.GOVERNMENT.value == "government"
        assert AzureCloud.LOCAL.value == "local"


class TestSettings:
    """Tests for Settings configuration."""

    def test_default_settings(self) -> None:
        """Test default settings are applied."""
        with patch.dict(os.environ, {}, clear=True):
            settings = create_settings()

            assert settings.azure_cloud == AzureCloud.GOVERNMENT
            assert settings.azure_speech_region == "usgovvirginia"
            assert settings.debug is False
            assert settings.log_level == "INFO"
            assert settings.app_name == "Captain's Log"

    def test_settings_from_env(self) -> None:
        """Test settings are loaded from environment variables."""
        env = {
            "AZURE_CLOUD": "commercial",
            "AZURE_SPEECH_KEY": "test-speech-key",
            "AZURE_SPEECH_REGION": "eastus",
            "DEBUG": "true",
            "LOG_LEVEL": "DEBUG",
        }
        with patch.dict(os.environ, env, clear=True):
            settings = create_settings()

            assert settings.azure_cloud == AzureCloud.COMMERCIAL
            assert settings.azure_speech_key == "test-speech-key"
            assert settings.azure_speech_region == "eastus"
            assert settings.debug is True
            assert settings.log_level == "DEBUG"

    def test_azure_cloud_validation(self) -> None:
        """Test azure_cloud field validation."""
        with patch.dict(os.environ, {"AZURE_CLOUD": "government"}, clear=True):
            settings = create_settings()
            assert settings.azure_cloud == AzureCloud.GOVERNMENT

        with patch.dict(os.environ, {"AZURE_CLOUD": "COMMERCIAL"}, clear=True):
            settings = create_settings()
            assert settings.azure_cloud == AzureCloud.COMMERCIAL

    def test_invalid_log_level_raises_error(self) -> None:
        """Test invalid log level raises validation error."""
        with patch.dict(os.environ, {"LOG_LEVEL": "INVALID"}, clear=True):
            with pytest.raises(ValueError, match="Invalid log level"):
                create_settings()

    def test_speech_endpoint_url_government(self) -> None:
        """Test speech endpoint URL for government cloud."""
        with patch.dict(
            os.environ,
            {"AZURE_CLOUD": "government", "AZURE_SPEECH_REGION": "usgovvirginia"},
            clear=True,
        ):
            settings = create_settings()
            assert settings.speech_endpoint_url == "wss://usgovvirginia.stt.speech.azure.us"

    def test_speech_endpoint_url_commercial(self) -> None:
        """Test speech endpoint URL for commercial cloud."""
        with patch.dict(
            os.environ,
            {"AZURE_CLOUD": "commercial", "AZURE_SPEECH_REGION": "eastus"},
            clear=True,
        ):
            settings = create_settings()
            assert settings.speech_endpoint_url == "wss://eastus.stt.speech.microsoft.com"

    def test_speech_endpoint_url_custom(self) -> None:
        """Test custom speech endpoint URL takes precedence."""
        with patch.dict(
            os.environ,
            {
                "AZURE_CLOUD": "government",
                "AZURE_SPEECH_ENDPOINT": "wss://custom.endpoint.example.com",
            },
            clear=True,
        ):
            settings = create_settings()
            assert settings.speech_endpoint_url == "wss://custom.endpoint.example.com"

    def test_entra_authority_government(self) -> None:
        """Test Entra authority URL for government cloud."""
        with patch.dict(
            os.environ,
            {"AZURE_CLOUD": "government", "AZURE_TENANT_ID": "test-tenant-id"},
            clear=True,
        ):
            settings = create_settings()
            assert settings.entra_authority == "https://login.microsoftonline.us/test-tenant-id"

    def test_entra_authority_commercial(self) -> None:
        """Test Entra authority URL for commercial cloud."""
        with patch.dict(
            os.environ,
            {"AZURE_CLOUD": "commercial", "AZURE_TENANT_ID": "test-tenant-id"},
            clear=True,
        ):
            settings = create_settings()
            assert settings.entra_authority == "https://login.microsoftonline.com/test-tenant-id"

    def test_entra_jwks_uri_government(self) -> None:
        """Test Entra JWKS URI for government cloud."""
        with patch.dict(
            os.environ,
            {"AZURE_CLOUD": "government", "AZURE_TENANT_ID": "test-tenant-id"},
            clear=True,
        ):
            settings = create_settings()
            expected = "https://login.microsoftonline.us/test-tenant-id/discovery/v2.0/keys"
            assert settings.entra_jwks_uri == expected

    def test_entra_jwks_uri_commercial(self) -> None:
        """Test Entra JWKS URI for commercial cloud."""
        with patch.dict(
            os.environ,
            {"AZURE_CLOUD": "commercial", "AZURE_TENANT_ID": "test-tenant-id"},
            clear=True,
        ):
            settings = create_settings()
            expected = "https://login.microsoftonline.com/test-tenant-id/discovery/v2.0/keys"
            assert settings.entra_jwks_uri == expected

    def test_is_speech_configured_true(self) -> None:
        """Test is_speech_configured returns True when configured."""
        with patch.dict(
            os.environ,
            {"AZURE_SPEECH_KEY": "test-key", "AZURE_SPEECH_REGION": "eastus"},
            clear=True,
        ):
            settings = create_settings()
            assert settings.is_speech_configured() is True

    def test_is_speech_configured_false(self) -> None:
        """Test is_speech_configured returns False when not configured."""
        with patch.dict(os.environ, {}, clear=True):
            settings = create_settings()
            assert settings.is_speech_configured() is False

    def test_is_openai_configured_true(self) -> None:
        """Test is_openai_configured returns True when configured."""
        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
                "AZURE_OPENAI_KEY": "test-key",
                "AZURE_OPENAI_DEPLOYMENT": "gpt-4",
            },
            clear=True,
        ):
            settings = create_settings()
            assert settings.is_openai_configured() is True

    def test_is_openai_configured_false(self) -> None:
        """Test is_openai_configured returns False when not configured."""
        with patch.dict(os.environ, {}, clear=True):
            settings = create_settings()
            assert settings.is_openai_configured() is False

    def test_is_openai_configured_with_model_name_alias(self) -> None:
        """Test azure_openai_deployment accepts AZURE_OPENAI_MODEL_NAME alias."""
        with patch.dict(
            os.environ,
            {
                "AZURE_OPENAI_ENDPOINT": "https://test.openai.azure.com",
                "AZURE_OPENAI_KEY": "test-key",
                "AZURE_OPENAI_MODEL_NAME": "gpt-4o",
            },
            clear=True,
        ):
            settings = create_settings()
            assert settings.azure_openai_deployment == "gpt-4o"
            assert settings.is_openai_configured() is True

    def test_is_entra_configured_true(self) -> None:
        """Test is_entra_configured returns True when configured."""
        with patch.dict(
            os.environ,
            {"AZURE_TENANT_ID": "test-tenant", "AZURE_CLIENT_ID": "test-client"},
            clear=True,
        ):
            settings = create_settings()
            assert settings.is_entra_configured() is True

    def test_is_entra_configured_false(self) -> None:
        """Test is_entra_configured returns False when not configured."""
        with patch.dict(os.environ, {}, clear=True):
            settings = create_settings()
            assert settings.is_entra_configured() is False


class TestGetSettings:
    """Tests for get_settings function."""

    def test_get_settings_returns_settings(self) -> None:
        """Test get_settings returns a Settings instance."""
        # Clear the cache first
        get_settings.cache_clear()
        settings = get_settings()
        assert isinstance(settings, Settings)

    def test_get_settings_is_cached(self) -> None:
        """Test get_settings returns cached instance."""
        get_settings.cache_clear()
        settings1 = get_settings()
        settings2 = get_settings()
        assert settings1 is settings2
