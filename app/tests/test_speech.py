"""Tests for Azure Speech Services integration.

This module provides comprehensive tests for the speech client,
mocking the Azure Speech SDK for isolated unit testing.
"""

from unittest.mock import MagicMock, Mock, patch

import pytest

from config.settings import AzureCloud, Settings
from speech.client import (
    SpeechClient,
    SpeechConfig,
    SpeechConfigurationError,
    SpeechRecognitionError,
    SpeechServiceError,
    SpeechServiceUnavailableError,
    clear_speech_client_cache,
    get_speech_client,
)


class TestSpeechConfig:
    """Tests for SpeechConfig dataclass."""

    def test_speech_config_government_endpoint(self) -> None:
        """Test Government cloud endpoint generation."""
        config = SpeechConfig(
            subscription_key="test-key",
            region="usgovvirginia",
            cloud=AzureCloud.GOVERNMENT,
        )
        assert config.speech_endpoint == "https://usgovvirginia.api.cognitive.azure.us/"
        assert config.stt_endpoint == "wss://usgovvirginia.stt.speech.azure.us"

    def test_speech_config_commercial_endpoint(self) -> None:
        """Test Commercial cloud endpoint generation."""
        config = SpeechConfig(
            subscription_key="test-key",
            region="eastus",
            cloud=AzureCloud.COMMERCIAL,
        )
        assert config.speech_endpoint == "https://eastus.api.cognitive.microsoft.com/"
        assert config.stt_endpoint == "wss://eastus.stt.speech.microsoft.com"

    def test_speech_config_local_uses_commercial(self) -> None:
        """Test Local environment uses Commercial endpoints."""
        config = SpeechConfig(
            subscription_key="test-key",
            region="westus",
            cloud=AzureCloud.LOCAL,
        )
        assert config.speech_endpoint == "https://westus.api.cognitive.microsoft.com/"
        assert config.stt_endpoint == "wss://westus.stt.speech.microsoft.com"

    def test_speech_config_custom_endpoint(self) -> None:
        """Test custom endpoint overrides auto-generated endpoint."""
        config = SpeechConfig(
            subscription_key="test-key",
            region="usgovvirginia",
            endpoint="https://custom.endpoint.com/",
            cloud=AzureCloud.GOVERNMENT,
        )
        assert config.speech_endpoint == "https://custom.endpoint.com/"

    def test_speech_config_from_settings(self) -> None:
        """Test creating SpeechConfig from Settings."""
        settings = Settings(
            azure_speech_key="test-key",
            azure_speech_region="usgovvirginia",
            azure_cloud=AzureCloud.GOVERNMENT,
            _env_file=None,  # type: ignore[call-arg]
        )
        config = SpeechConfig.from_settings(settings)

        assert config.subscription_key == "test-key"
        assert config.region == "usgovvirginia"
        assert config.cloud == AzureCloud.GOVERNMENT

    def test_speech_config_from_settings_missing_key(self) -> None:
        """Test SpeechConfig raises error when key is missing."""
        settings = Settings(
            azure_speech_key=None,
            azure_speech_region="usgovvirginia",
            _env_file=None,  # type: ignore[call-arg]
        )
        with pytest.raises(SpeechConfigurationError) as exc_info:
            SpeechConfig.from_settings(settings)
        assert "AZURE_SPEECH_KEY" in str(exc_info.value)

    def test_speech_config_from_settings_missing_region(self) -> None:
        """Test SpeechConfig raises error when region is missing."""
        settings = Settings(
            azure_speech_key="test-key",
            azure_speech_region="",
            _env_file=None,  # type: ignore[call-arg]
        )
        with pytest.raises(SpeechConfigurationError) as exc_info:
            SpeechConfig.from_settings(settings)
        assert "AZURE_SPEECH_REGION" in str(exc_info.value)

    def test_speech_config_with_custom_endpoint_from_settings(self) -> None:
        """Test SpeechConfig with custom endpoint from settings."""
        settings = Settings(
            azure_speech_key="test-key",
            azure_speech_region="usgovvirginia",
            azure_speech_endpoint="https://custom.endpoint.com/",
            azure_cloud=AzureCloud.GOVERNMENT,
            _env_file=None,  # type: ignore[call-arg]
        )
        config = SpeechConfig.from_settings(settings)

        assert config.endpoint == "https://custom.endpoint.com/"
        assert config.speech_endpoint == "https://custom.endpoint.com/"


class TestSpeechClient:
    """Tests for SpeechClient class."""

    def test_speech_client_initialization(self) -> None:
        """Test SpeechClient initializes correctly."""
        config = SpeechConfig(
            subscription_key="test-key",
            region="usgovvirginia",
            cloud=AzureCloud.GOVERNMENT,
        )
        client = SpeechClient(config)

        assert client.config == config
        assert client.region == "usgovvirginia"
        assert client.cloud == AzureCloud.GOVERNMENT

    @patch("speech.client.speechsdk")
    def test_get_speech_config_government(self, mock_sdk: MagicMock) -> None:
        """Test Speech SDK configuration for Government cloud."""
        mock_speech_config = MagicMock()
        mock_sdk.SpeechConfig.return_value = mock_speech_config
        mock_sdk.OutputFormat.Detailed = "Detailed"

        config = SpeechConfig(
            subscription_key="test-key",
            region="usgovvirginia",
            cloud=AzureCloud.GOVERNMENT,
        )
        client = SpeechClient(config)
        result = client._get_speech_config()

        assert result == mock_speech_config
        mock_sdk.SpeechConfig.assert_called_once_with(
            subscription="test-key",
            endpoint="https://usgovvirginia.api.cognitive.azure.us/",
        )
        # Note: We do NOT set region when using endpoint - they conflict (SPXERR_INVALID_ARG)
        mock_speech_config.set_property.assert_not_called()

    @patch("speech.client.speechsdk")
    def test_get_speech_config_commercial(self, mock_sdk: MagicMock) -> None:
        """Test Speech SDK configuration for Commercial cloud."""
        mock_speech_config = MagicMock()
        mock_sdk.SpeechConfig.return_value = mock_speech_config
        mock_sdk.OutputFormat.Detailed = "Detailed"

        config = SpeechConfig(
            subscription_key="test-key",
            region="eastus",
            cloud=AzureCloud.COMMERCIAL,
        )
        client = SpeechClient(config)
        result = client._get_speech_config()

        assert result == mock_speech_config
        mock_sdk.SpeechConfig.assert_called_once_with(
            subscription="test-key",
            region="eastus",
        )

    @patch("speech.client.speechsdk")
    def test_get_speech_config_custom_endpoint(self, mock_sdk: MagicMock) -> None:
        """Test Speech SDK configuration with custom endpoint (no region setting)."""
        mock_speech_config = MagicMock()
        mock_sdk.SpeechConfig.return_value = mock_speech_config
        mock_sdk.OutputFormat.Detailed = "Detailed"

        # Custom endpoint simulates Azure Cognitive Services multi-service endpoint
        config = SpeechConfig(
            subscription_key="test-key",
            region="usgovarizona",
            endpoint="https://my-cog-service.cognitiveservices.azure.us/",
            cloud=AzureCloud.GOVERNMENT,
        )
        client = SpeechClient(config)
        result = client._get_speech_config()

        assert result == mock_speech_config
        # When custom endpoint is provided, use endpoint directly (not speech_endpoint)
        mock_sdk.SpeechConfig.assert_called_once_with(
            subscription="test-key",
            endpoint="https://my-cog-service.cognitiveservices.azure.us/",
        )
        # Must NOT set region when using endpoint - causes SPXERR_INVALID_ARG
        mock_speech_config.set_property.assert_not_called()

    @patch("speech.client.speechsdk")
    def test_get_speech_config_caching(self, mock_sdk: MagicMock) -> None:
        """Test Speech SDK configuration is cached."""
        mock_speech_config = MagicMock()
        mock_sdk.SpeechConfig.return_value = mock_speech_config
        mock_sdk.OutputFormat.Detailed = "Detailed"

        config = SpeechConfig(
            subscription_key="test-key",
            region="eastus",
            cloud=AzureCloud.COMMERCIAL,
        )
        client = SpeechClient(config)

        # Call twice
        client._get_speech_config()
        client._get_speech_config()

        # Should only create once
        mock_sdk.SpeechConfig.assert_called_once()

    @patch("speech.client.speechsdk")
    def test_get_speech_config_error(self, mock_sdk: MagicMock) -> None:
        """Test Speech SDK configuration error handling."""
        mock_sdk.SpeechConfig.side_effect = Exception("Configuration error")

        config = SpeechConfig(
            subscription_key="test-key",
            region="eastus",
            cloud=AzureCloud.COMMERCIAL,
        )
        client = SpeechClient(config)

        with pytest.raises(SpeechServiceUnavailableError) as exc_info:
            client._get_speech_config()
        assert "Failed to configure Speech Services" in str(exc_info.value)

    @patch("speech.client.speechsdk")
    def test_create_audio_config_from_file(self, mock_sdk: MagicMock) -> None:
        """Test creating AudioConfig from file."""
        mock_audio_config = MagicMock()
        mock_sdk.AudioConfig.return_value = mock_audio_config

        config = SpeechConfig(
            subscription_key="test-key",
            region="eastus",
            cloud=AzureCloud.COMMERCIAL,
        )
        client = SpeechClient(config)
        result = client.create_audio_config_from_file("/path/to/audio.wav")

        assert result == mock_audio_config
        mock_sdk.AudioConfig.assert_called_once_with(filename="/path/to/audio.wav")

    @patch("speech.client.speechsdk")
    def test_create_audio_config_from_file_error(self, mock_sdk: MagicMock) -> None:
        """Test error handling when creating AudioConfig from file."""
        mock_sdk.AudioConfig.side_effect = Exception("File not found")

        config = SpeechConfig(
            subscription_key="test-key",
            region="eastus",
            cloud=AzureCloud.COMMERCIAL,
        )
        client = SpeechClient(config)

        with pytest.raises(SpeechServiceError) as exc_info:
            client.create_audio_config_from_file("/nonexistent/audio.wav")
        assert "Failed to configure audio" in str(exc_info.value)

    @patch("speech.client.speechsdk")
    def test_create_audio_config_from_stream(self, mock_sdk: MagicMock) -> None:
        """Test creating AudioConfig from stream."""
        mock_audio_config = MagicMock()
        mock_sdk.AudioConfig.return_value = mock_audio_config
        mock_stream = MagicMock()

        config = SpeechConfig(
            subscription_key="test-key",
            region="eastus",
            cloud=AzureCloud.COMMERCIAL,
        )
        client = SpeechClient(config)
        result = client.create_audio_config_from_stream(mock_stream)

        assert result == mock_audio_config
        mock_sdk.AudioConfig.assert_called_once_with(stream=mock_stream)

    @patch("speech.client.speechsdk")
    def test_create_recognizer(self, mock_sdk: MagicMock) -> None:
        """Test creating a speech recognizer."""
        mock_speech_config = MagicMock()
        mock_recognizer = MagicMock()
        mock_audio_config = MagicMock()

        mock_sdk.SpeechConfig.return_value = mock_speech_config
        mock_sdk.SpeechRecognizer.return_value = mock_recognizer
        mock_sdk.OutputFormat.Detailed = "Detailed"

        config = SpeechConfig(
            subscription_key="test-key",
            region="eastus",
            cloud=AzureCloud.COMMERCIAL,
        )
        client = SpeechClient(config)
        result = client.create_recognizer(mock_audio_config, language="es-ES")

        assert result == mock_recognizer
        assert mock_speech_config.speech_recognition_language == "es-ES"
        mock_sdk.SpeechRecognizer.assert_called_once_with(
            speech_config=mock_speech_config,
            audio_config=mock_audio_config,
        )

    @patch("speech.client.speechsdk")
    def test_create_recognizer_default_language(self, mock_sdk: MagicMock) -> None:
        """Test recognizer uses default English language."""
        mock_speech_config = MagicMock()
        mock_sdk.SpeechConfig.return_value = mock_speech_config
        mock_sdk.SpeechRecognizer.return_value = MagicMock()
        mock_sdk.OutputFormat.Detailed = "Detailed"

        config = SpeechConfig(
            subscription_key="test-key",
            region="eastus",
            cloud=AzureCloud.COMMERCIAL,
        )
        client = SpeechClient(config)
        client.create_recognizer(MagicMock())

        assert mock_speech_config.speech_recognition_language == "en-US"

    @patch("speech.client.speechsdk")
    def test_recognize_once_success(self, mock_sdk: MagicMock) -> None:
        """Test successful speech recognition."""
        mock_result = Mock()
        mock_result.reason = mock_sdk.ResultReason.RecognizedSpeech
        mock_result.text = "Hello world"

        mock_recognizer = Mock()
        mock_recognizer.recognize_once.return_value = mock_result

        mock_sdk.SpeechConfig.return_value = MagicMock()
        mock_sdk.SpeechRecognizer.return_value = mock_recognizer
        mock_sdk.OutputFormat.Detailed = "Detailed"

        config = SpeechConfig(
            subscription_key="test-key",
            region="eastus",
            cloud=AzureCloud.COMMERCIAL,
        )
        client = SpeechClient(config)
        result = client.recognize_once(MagicMock())

        assert result == "Hello world"

    @patch("speech.client.speechsdk")
    def test_recognize_once_no_match(self, mock_sdk: MagicMock) -> None:
        """Test recognition with no speech detected."""
        mock_result = Mock()
        mock_result.reason = mock_sdk.ResultReason.NoMatch

        mock_no_match_detail = Mock()
        mock_no_match_detail.reason = "NoSpeechRecognized"
        mock_sdk.NoMatchDetails.return_value = mock_no_match_detail

        mock_recognizer = Mock()
        mock_recognizer.recognize_once.return_value = mock_result

        mock_sdk.SpeechConfig.return_value = MagicMock()
        mock_sdk.SpeechRecognizer.return_value = mock_recognizer
        mock_sdk.OutputFormat.Detailed = "Detailed"

        config = SpeechConfig(
            subscription_key="test-key",
            region="eastus",
            cloud=AzureCloud.COMMERCIAL,
        )
        client = SpeechClient(config)

        with pytest.raises(SpeechRecognitionError) as exc_info:
            client.recognize_once(MagicMock())
        assert "No speech recognized" in str(exc_info.value)

    @patch("speech.client.speechsdk")
    def test_recognize_once_canceled_with_error(self, mock_sdk: MagicMock) -> None:
        """Test recognition canceled due to service error."""
        mock_result = Mock()
        mock_result.reason = mock_sdk.ResultReason.Canceled

        mock_cancellation = Mock()
        mock_cancellation.reason = mock_sdk.CancellationReason.Error
        mock_cancellation.error_details = "Service unavailable"
        mock_sdk.CancellationDetails.return_value = mock_cancellation

        mock_recognizer = Mock()
        mock_recognizer.recognize_once.return_value = mock_result

        mock_sdk.SpeechConfig.return_value = MagicMock()
        mock_sdk.SpeechRecognizer.return_value = mock_recognizer
        mock_sdk.OutputFormat.Detailed = "Detailed"

        config = SpeechConfig(
            subscription_key="test-key",
            region="eastus",
            cloud=AzureCloud.COMMERCIAL,
        )
        client = SpeechClient(config)

        with pytest.raises(SpeechServiceUnavailableError) as exc_info:
            client.recognize_once(MagicMock())
        assert "Service unavailable" in str(exc_info.value)

    @patch("speech.client.speechsdk")
    def test_recognize_once_canceled_other_reason(self, mock_sdk: MagicMock) -> None:
        """Test recognition canceled for non-error reason."""
        mock_result = Mock()
        mock_result.reason = mock_sdk.ResultReason.Canceled

        mock_cancellation = Mock()
        mock_cancellation.reason = mock_sdk.CancellationReason.EndOfStream
        mock_sdk.CancellationDetails.return_value = mock_cancellation

        mock_recognizer = Mock()
        mock_recognizer.recognize_once.return_value = mock_result

        mock_sdk.SpeechConfig.return_value = MagicMock()
        mock_sdk.SpeechRecognizer.return_value = mock_recognizer
        mock_sdk.OutputFormat.Detailed = "Detailed"

        config = SpeechConfig(
            subscription_key="test-key",
            region="eastus",
            cloud=AzureCloud.COMMERCIAL,
        )
        client = SpeechClient(config)

        with pytest.raises(SpeechRecognitionError) as exc_info:
            client.recognize_once(MagicMock())
        assert "Recognition canceled" in str(exc_info.value)

    @patch("speech.client.speechsdk")
    def test_recognize_once_unexpected_result(self, mock_sdk: MagicMock) -> None:
        """Test recognition with unexpected result reason."""
        mock_result = Mock()
        mock_result.reason = "UnknownReason"

        mock_recognizer = Mock()
        mock_recognizer.recognize_once.return_value = mock_result

        mock_sdk.SpeechConfig.return_value = MagicMock()
        mock_sdk.SpeechRecognizer.return_value = mock_recognizer
        mock_sdk.OutputFormat.Detailed = "Detailed"
        mock_sdk.ResultReason.RecognizedSpeech = "RecognizedSpeech"
        mock_sdk.ResultReason.NoMatch = "NoMatch"
        mock_sdk.ResultReason.Canceled = "Canceled"

        config = SpeechConfig(
            subscription_key="test-key",
            region="eastus",
            cloud=AzureCloud.COMMERCIAL,
        )
        client = SpeechClient(config)

        with pytest.raises(SpeechRecognitionError) as exc_info:
            client.recognize_once(MagicMock())
        assert "Unexpected result" in str(exc_info.value)

    @patch("speech.client.speechsdk")
    def test_is_available_true(self, mock_sdk: MagicMock) -> None:
        """Test is_available returns True when configured."""
        mock_sdk.SpeechConfig.return_value = MagicMock()
        mock_sdk.OutputFormat.Detailed = "Detailed"

        config = SpeechConfig(
            subscription_key="test-key",
            region="eastus",
            cloud=AzureCloud.COMMERCIAL,
        )
        client = SpeechClient(config)

        assert client.is_available() is True

    @patch("speech.client.speechsdk")
    def test_is_available_false(self, mock_sdk: MagicMock) -> None:
        """Test is_available returns False when configuration fails."""
        mock_sdk.SpeechConfig.side_effect = Exception("Config error")

        config = SpeechConfig(
            subscription_key="test-key",
            region="eastus",
            cloud=AzureCloud.COMMERCIAL,
        )
        client = SpeechClient(config)

        assert client.is_available() is False


class TestGetSpeechClient:
    """Tests for get_speech_client function."""

    def test_get_speech_client_not_configured(self) -> None:
        """Test get_speech_client returns None when not configured."""
        settings = Settings(
            azure_speech_key=None,
            azure_speech_region="usgovvirginia",
            _env_file=None,  # type: ignore[call-arg]
        )
        # Clear the cache to test with new settings
        clear_speech_client_cache()

        result = get_speech_client(settings)

        assert result is None

    def test_get_speech_client_configured(self) -> None:
        """Test get_speech_client returns client when configured."""
        settings = Settings(
            azure_speech_key="test-key",
            azure_speech_region="usgovvirginia",
            azure_cloud=AzureCloud.GOVERNMENT,
            _env_file=None,  # type: ignore[call-arg]
        )
        # Clear the cache to test with new settings
        clear_speech_client_cache()

        result = get_speech_client(settings)

        assert result is not None
        assert isinstance(result, SpeechClient)
        assert result.region == "usgovvirginia"
        assert result.cloud == AzureCloud.GOVERNMENT

    def test_get_speech_client_with_settings_not_cached(self) -> None:
        """Test get_speech_client with settings creates new instance each time."""
        settings = Settings(
            azure_speech_key="test-key",
            azure_speech_region="usgovvirginia",
            azure_cloud=AzureCloud.GOVERNMENT,
            _env_file=None,  # type: ignore[call-arg]
        )
        # Clear the cache
        clear_speech_client_cache()

        client1 = get_speech_client(settings)
        client2 = get_speech_client(settings)

        # With explicit settings, should create new instances
        assert client1 is not client2
        assert client1 is not None
        assert client2 is not None
