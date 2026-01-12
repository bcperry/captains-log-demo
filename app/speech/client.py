"""Azure Speech Services client integration.

This module provides a client for Azure Speech Services SDK with support for:
- Azure Commercial and Government cloud endpoints
- Configuration via environment variables
- Proper error handling for service unavailability
"""

from dataclasses import dataclass
from typing import Optional

import azure.cognitiveservices.speech as speechsdk

from config.settings import AzureCloud, Settings, get_settings


class SpeechServiceError(Exception):
    """Base exception for Speech Service errors."""

    pass


class SpeechServiceUnavailableError(SpeechServiceError):
    """Exception raised when the Speech Service is unavailable."""

    pass


class SpeechRecognitionError(SpeechServiceError):
    """Exception raised when speech recognition fails."""

    pass


class SpeechConfigurationError(SpeechServiceError):
    """Exception raised when Speech Service is not properly configured."""

    pass


@dataclass
class SpeechConfig:
    """Configuration for Azure Speech Services.

    Attributes:
        subscription_key: Azure Speech Services subscription key
        region: Azure Speech Services region
        endpoint: Optional custom endpoint URL
        cloud: Azure cloud environment (commercial, government, local)
    """

    subscription_key: str
    region: str
    endpoint: Optional[str] = None
    cloud: AzureCloud = AzureCloud.GOVERNMENT

    @property
    def speech_endpoint(self) -> str:
        """Get the Speech Services endpoint based on cloud environment.

        Returns:
            The appropriate endpoint URL for the configured cloud environment.
        """
        if self.endpoint:
            return self.endpoint

        if self.cloud == AzureCloud.GOVERNMENT:
            return f"https://{self.region}.api.cognitive.azure.us/"
        else:
            return f"https://{self.region}.api.cognitive.microsoft.com/"

    @property
    def stt_endpoint(self) -> str:
        """Get the Speech-to-Text WebSocket endpoint.

        Returns:
            The WebSocket endpoint URL for real-time speech recognition.
        """
        if self.cloud == AzureCloud.GOVERNMENT:
            return f"wss://{self.region}.stt.speech.azure.us"
        else:
            return f"wss://{self.region}.stt.speech.microsoft.com"

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "SpeechConfig":
        """Create SpeechConfig from application settings.

        Args:
            settings: Application settings. If None, uses default settings.

        Returns:
            SpeechConfig instance configured from settings.

        Raises:
            SpeechConfigurationError: If required settings are missing.
        """
        settings = settings or get_settings()

        if not settings.azure_speech_key:
            raise SpeechConfigurationError(
                "Azure Speech key is required. Set AZURE_SPEECH_KEY environment variable."
            )
        if not settings.azure_speech_region:
            raise SpeechConfigurationError(
                "Azure Speech region is required. Set AZURE_SPEECH_REGION environment variable."
            )

        return cls(
            subscription_key=settings.azure_speech_key,
            region=settings.azure_speech_region,
            endpoint=settings.azure_speech_endpoint,
            cloud=settings.azure_cloud,
        )


class SpeechClient:
    """Client for Azure Speech Services.

    Provides methods for speech recognition using the Azure Speech SDK,
    with support for Azure Government endpoints and proper error handling.
    """

    def __init__(self, config: SpeechConfig) -> None:
        """Initialize the Speech client.

        Args:
            config: SpeechConfig with Azure Speech Services configuration.
        """
        self._config = config
        self._speech_config: Optional[speechsdk.SpeechConfig] = None

    @property
    def config(self) -> SpeechConfig:
        """Get the SpeechConfig instance."""
        return self._config

    def _get_speech_config(self) -> speechsdk.SpeechConfig:
        """Get or create the Azure Speech SDK configuration.

        Returns:
            speechsdk.SpeechConfig configured for the appropriate cloud.

        Raises:
            SpeechServiceUnavailableError: If configuration fails.
        """
        if self._speech_config is not None:
            return self._speech_config

        try:
            # Check if a custom endpoint is configured (including Government cloud)
            # When using endpoint, do NOT set region - they conflict and cause SPXERR_INVALID_ARG
            if self._config.endpoint:
                # Custom endpoint provided - use endpoint-only configuration
                self._speech_config = speechsdk.SpeechConfig(
                    subscription=self._config.subscription_key,
                    endpoint=self._config.endpoint,
                )
            elif self._config.cloud == AzureCloud.GOVERNMENT:
                # Government cloud without custom endpoint - use standard speech endpoint
                self._speech_config = speechsdk.SpeechConfig(
                    subscription=self._config.subscription_key,
                    endpoint=self._config.speech_endpoint,
                )
            else:
                # Commercial cloud uses standard region-based configuration
                self._speech_config = speechsdk.SpeechConfig(
                    subscription=self._config.subscription_key,
                    region=self._config.region,
                )

            # Enable detailed recognition results
            self._speech_config.output_format = speechsdk.OutputFormat.Detailed

            return self._speech_config

        except Exception as e:
            raise SpeechServiceUnavailableError(
                f"Failed to configure Speech Services: {e}"
            ) from e

    def create_audio_config_from_file(
        self, audio_file_path: str
    ) -> speechsdk.AudioConfig:
        """Create an AudioConfig from an audio file.

        Args:
            audio_file_path: Path to the audio file.

        Returns:
            speechsdk.AudioConfig for the audio file.

        Raises:
            SpeechServiceError: If audio configuration fails.
        """
        try:
            return speechsdk.AudioConfig(filename=audio_file_path)
        except Exception as e:
            raise SpeechServiceError(f"Failed to configure audio: {e}") from e

    def create_audio_config_from_stream(
        self, audio_stream: speechsdk.audio.AudioInputStream
    ) -> speechsdk.AudioConfig:
        """Create an AudioConfig from an audio stream.

        Args:
            audio_stream: Audio input stream.

        Returns:
            speechsdk.AudioConfig for the audio stream.

        Raises:
            SpeechServiceError: If audio configuration fails.
        """
        try:
            return speechsdk.AudioConfig(stream=audio_stream)
        except Exception as e:
            raise SpeechServiceError(f"Failed to configure audio stream: {e}") from e

    def create_recognizer(
        self,
        audio_config: speechsdk.AudioConfig,
        language: str = "en-US",
    ) -> speechsdk.SpeechRecognizer:
        """Create a speech recognizer.

        Args:
            audio_config: Audio configuration for the recognizer.
            language: Language code for recognition (default: en-US).

        Returns:
            speechsdk.SpeechRecognizer configured for the audio source.

        Raises:
            SpeechServiceUnavailableError: If recognizer creation fails.
        """
        try:
            speech_config = self._get_speech_config()
            speech_config.speech_recognition_language = language

            return speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config,
            )
        except SpeechServiceUnavailableError:
            raise
        except Exception as e:
            raise SpeechServiceUnavailableError(
                f"Failed to create speech recognizer: {e}"
            ) from e

    def recognize_once(
        self,
        audio_config: speechsdk.AudioConfig,
        language: str = "en-US",
    ) -> str:
        """Perform single-shot speech recognition.

        Args:
            audio_config: Audio configuration for the recognizer.
            language: Language code for recognition (default: en-US).

        Returns:
            Recognized text from the audio.

        Raises:
            SpeechRecognitionError: If recognition fails or returns no results.
            SpeechServiceUnavailableError: If the service is unavailable.
        """
        recognizer = self.create_recognizer(audio_config, language)

        try:
            result = recognizer.recognize_once()

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                return str(result.text)
            elif result.reason == speechsdk.ResultReason.NoMatch:
                no_match_detail = speechsdk.NoMatchDetails(result)
                raise SpeechRecognitionError(
                    f"No speech recognized. Reason: {no_match_detail.reason}"
                )
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = speechsdk.CancellationDetails(result)
                if cancellation.reason == speechsdk.CancellationReason.Error:
                    raise SpeechServiceUnavailableError(
                        f"Speech service error: {cancellation.error_details}"
                    )
                raise SpeechRecognitionError(
                    f"Recognition canceled: {cancellation.reason}"
                )
            else:
                raise SpeechRecognitionError(f"Unexpected result: {result.reason}")

        except (SpeechRecognitionError, SpeechServiceUnavailableError):
            raise
        except Exception as e:
            raise SpeechServiceError(f"Speech recognition failed: {e}") from e

    def is_available(self) -> bool:
        """Check if the Speech Service is available and properly configured.

        Returns:
            True if the service is available, False otherwise.
        """
        try:
            # Attempt to create the speech config - this validates configuration
            self._get_speech_config()
            return True
        except SpeechServiceUnavailableError:
            return False

    def recognize_continuous_with_diarization(
        self,
        audio_config: speechsdk.AudioConfig,
        language: str = "en-US",
        max_speakers: int = 5,
    ) -> list[dict[str, object]]:
        """Perform continuous speech recognition with speaker diarization.

        Uses ConversationTranscriber for multi-speaker transcription.

        Args:
            audio_config: Audio configuration for the recognizer.
            language: Language code for recognition (default: en-US).
            max_speakers: Maximum number of speakers to identify (1-10).

        Returns:
            List of diarized segments with speaker info and timestamps.

        Raises:
            SpeechRecognitionError: If recognition fails.
            SpeechServiceUnavailableError: If the service is unavailable.
        """
        import threading

        speech_config = self._get_speech_config()
        speech_config.speech_recognition_language = language

        # Set speaker diarization properties
        speech_config.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_LanguageIdMode,
            "Continuous"
        )

        try:
            # Create conversation transcriber
            conversation_transcriber = speechsdk.transcription.ConversationTranscriber(
                speech_config=speech_config,
                audio_config=audio_config,
            )

            segments: list[dict[str, object]] = []
            errors: list[str] = []
            done_event = threading.Event()

            def handle_transcribed(evt: speechsdk.SpeechRecognitionEventArgs) -> None:
                """Handle transcribed speech events."""
                if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
                    segment = {
                        "speaker_id": getattr(evt.result, "speaker_id", "Unknown"),
                        "text": evt.result.text,
                        "start_time_ms": int(
                            evt.result.offset / 10000
                        ),  # Convert ticks to ms
                        "end_time_ms": int(
                            (evt.result.offset + evt.result.duration) / 10000
                        ),
                    }
                    segments.append(segment)

            def handle_canceled(evt: speechsdk.SpeechRecognitionCanceledEventArgs) -> None:
                """Handle cancellation events."""
                if evt.cancellation_details.reason == speechsdk.CancellationReason.Error:
                    errors.append(evt.cancellation_details.error_details)
                done_event.set()

            def handle_session_stopped(
                evt: speechsdk.SessionEventArgs,
            ) -> None:
                """Handle session stopped events."""
                done_event.set()

            # Connect event handlers
            conversation_transcriber.transcribed.connect(handle_transcribed)
            conversation_transcriber.canceled.connect(handle_canceled)
            conversation_transcriber.session_stopped.connect(handle_session_stopped)

            # Start transcription
            conversation_transcriber.start_transcribing_async().get()

            # Wait for completion (timeout after 5 minutes)
            done_event.wait(timeout=300)

            # Stop transcription
            conversation_transcriber.stop_transcribing_async().get()

            if errors:
                raise SpeechServiceUnavailableError(
                    f"Diarization failed: {errors[0]}"
                )

            return segments

        except (SpeechRecognitionError, SpeechServiceUnavailableError):
            raise
        except Exception as e:
            raise SpeechServiceError(
                f"Diarization failed: {e}"
            ) from e

    @property
    def region(self) -> str:
        """Get the configured region."""
        return self._config.region

    @property
    def cloud(self) -> AzureCloud:
        """Get the configured cloud environment."""
        return self._config.cloud


_speech_client_instance: Optional[SpeechClient] = None


def get_speech_client(settings: Optional[Settings] = None) -> Optional[SpeechClient]:
    """Get a SpeechClient instance.

    When called without settings, returns a cached singleton instance.
    When called with settings, creates a new instance (for testing).

    Args:
        settings: Optional settings override. If None, uses get_settings().

    Returns:
        SpeechClient instance if Speech Services is configured, None otherwise.
    """
    global _speech_client_instance

    # If settings provided, create a new instance (for testing)
    if settings is not None:
        if not settings.is_speech_configured():
            return None
        try:
            config = SpeechConfig.from_settings(settings)
            return SpeechClient(config)
        except SpeechConfigurationError:
            return None

    # Use cached singleton for default settings
    if _speech_client_instance is not None:
        return _speech_client_instance

    default_settings = get_settings()
    if not default_settings.is_speech_configured():
        return None

    try:
        config = SpeechConfig.from_settings(default_settings)
        _speech_client_instance = SpeechClient(config)
        return _speech_client_instance
    except SpeechConfigurationError:
        return None


def clear_speech_client_cache() -> None:
    """Clear the cached SpeechClient instance.

    Useful for testing to reset the singleton.
    """
    global _speech_client_instance
    _speech_client_instance = None
