"""Azure Speech Services integration module.

This module provides integration with Azure Speech Services SDK,
supporting both Azure Commercial and Azure Government cloud environments.
"""

from speech.batch import (
    BatchTranscriptionClient,
    BatchTranscriptionConfig,
    BatchTranscriptionError,
    BatchTranscriptionFailedError,
    BatchTranscriptionJob,
    BatchTranscriptionJobNotFoundError,
    BatchTranscriptionResult,
    InMemoryBatchTranscriptionClient,
    TranscriptionSegment,
    TranscriptionStatus,
    clear_batch_client_cache,
    get_batch_transcription_client,
)
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
from speech.converter import (
    AudioConversionError,
    NoAudioTrackError,
    convert_to_wav,
    get_audio_duration_ms,
    needs_conversion,
)

__all__ = [
    # Real-time speech client
    "SpeechClient",
    "SpeechConfig",
    "SpeechConfigurationError",
    "SpeechServiceError",
    "SpeechServiceUnavailableError",
    "SpeechRecognitionError",
    "clear_speech_client_cache",
    "get_speech_client",
    # Batch transcription client
    "BatchTranscriptionClient",
    "BatchTranscriptionConfig",
    "BatchTranscriptionError",
    "BatchTranscriptionFailedError",
    "BatchTranscriptionJob",
    "BatchTranscriptionJobNotFoundError",
    "BatchTranscriptionResult",
    "InMemoryBatchTranscriptionClient",
    "TranscriptionSegment",
    "TranscriptionStatus",
    "clear_batch_client_cache",
    "get_batch_transcription_client",
    # Audio conversion
    "AudioConversionError",
    "NoAudioTrackError",
    "convert_to_wav",
    "get_audio_duration_ms",
    "needs_conversion",
]
