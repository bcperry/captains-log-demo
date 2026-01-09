"""Azure Speech Services integration module.

This module provides integration with Azure Speech Services SDK,
supporting both Azure Commercial and Azure Government cloud environments.
"""

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

__all__ = [
    "SpeechClient",
    "SpeechConfig",
    "SpeechConfigurationError",
    "SpeechServiceError",
    "SpeechServiceUnavailableError",
    "SpeechRecognitionError",
    "clear_speech_client_cache",
    "get_speech_client",
]
