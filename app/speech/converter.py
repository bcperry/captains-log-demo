"""Audio format conversion utilities for Azure Speech Services.

This module provides utilities for converting audio files to WAV format
optimized for Azure Speech SDK (16kHz, 16-bit, mono PCM).

Uses ffmpeg directly via subprocess for Python 3.14+ compatibility.
"""

import logging
import os
import subprocess
import tempfile
from typing import Optional

logger = logging.getLogger(__name__)

# Azure Speech SDK optimal settings
SPEECH_SDK_SAMPLE_RATE = 16000  # 16 kHz
SPEECH_SDK_SAMPLE_WIDTH = 2  # 16-bit (2 bytes)
SPEECH_SDK_CHANNELS = 1  # Mono


class AudioConversionError(Exception):
    """Exception raised when audio conversion fails."""

    pass


class NoAudioTrackError(AudioConversionError):
    """Exception raised when MP4 file has no audio track."""

    pass


def _check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available on the system."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def convert_to_wav(
    input_path: str,
    audio_format: str,
    output_path: Optional[str] = None,
) -> str:
    """Convert audio file to WAV format optimized for Azure Speech SDK.

    Converts audio to 16kHz, 16-bit, mono PCM WAV format using ffmpeg.

    Args:
        input_path: Path to the input audio file
        audio_format: Format of the input file (mp3, mp4, m4a, wav, etc.)
        output_path: Optional output path. If None, creates a temp file.

    Returns:
        Path to the converted WAV file

    Raises:
        AudioConversionError: If conversion fails
        NoAudioTrackError: If MP4 file has no audio track
    """
    if not _check_ffmpeg_available():
        raise AudioConversionError("ffmpeg is not available on this system")

    # Determine output path
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    try:
        # Build ffmpeg command for conversion
        # -y: overwrite output
        # -i: input file
        # -ar: sample rate
        # -ac: audio channels
        # -acodec: audio codec
        # -sample_fmt: sample format (s16 = signed 16-bit)
        cmd = [
            "ffmpeg",
            "-y",
            "-i", input_path,
            "-ar", str(SPEECH_SDK_SAMPLE_RATE),
            "-ac", str(SPEECH_SDK_CHANNELS),
            "-acodec", "pcm_s16le",
            output_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            error_output = result.stderr.lower()
            # Check for no audio stream error
            if "no audio" in error_output or "does not contain any stream" in error_output:
                raise NoAudioTrackError(f"File has no audio track: {input_path}")
            raise AudioConversionError(f"ffmpeg conversion failed: {result.stderr}")

        # Verify output file exists and has content
        if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            raise AudioConversionError(f"Conversion produced empty output: {output_path}")

        logger.info(
            f"Converted {audio_format} to WAV: {input_path} -> {output_path} "
            f"(sample_rate: {SPEECH_SDK_SAMPLE_RATE}Hz)"
        )

        return output_path

    except (NoAudioTrackError, AudioConversionError):
        # Clean up output file on error
        if output_path and os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise
    except Exception as e:
        # Clean up output file on error
        if output_path and os.path.exists(output_path):
            try:
                os.unlink(output_path)
            except OSError:
                pass
        raise AudioConversionError(f"Failed to convert {audio_format} to WAV: {e}") from e


def needs_conversion(audio_format: str) -> bool:
    """Check if the audio format needs conversion for Azure Speech SDK.

    Azure Speech SDK natively supports WAV, but for best results with
    non-WAV formats, we convert to ensure optimal audio settings.

    Args:
        audio_format: The audio format extension (mp3, mp4, wav, etc.)

    Returns:
        True if conversion is recommended
    """
    # Convert everything except WAV to ensure optimal settings
    # Even WAV files might not be in the optimal 16kHz/16-bit/mono format
    return audio_format.lower() != "wav"


def get_audio_duration_ms(file_path: str, audio_format: str) -> Optional[int]:
    """Get the duration of an audio file in milliseconds.

    Args:
        file_path: Path to the audio file
        audio_format: Format of the audio file (unused, for API compatibility)

    Returns:
        Duration in milliseconds, or None if unable to determine
    """
    try:
        # Use ffprobe to get duration
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode == 0 and result.stdout.strip():
            duration_seconds = float(result.stdout.strip())
            return int(duration_seconds * 1000)
        return None
    except Exception as e:
        logger.warning(f"Failed to get audio duration: {e}")
        return None
