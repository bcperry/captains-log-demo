"""Tests for audio format conversion utilities."""

import os
import subprocess
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from speech.converter import (
    AudioConversionError,
    NoAudioTrackError,
    convert_to_wav,
    get_audio_duration_ms,
    needs_conversion,
)


class TestNeedsConversion:
    """Tests for needs_conversion function."""

    def test_wav_does_not_need_conversion(self) -> None:
        """WAV files don't need conversion."""
        assert needs_conversion("wav") is False
        assert needs_conversion("WAV") is False

    def test_mp3_needs_conversion(self) -> None:
        """MP3 files need conversion."""
        assert needs_conversion("mp3") is True

    def test_mp4_needs_conversion(self) -> None:
        """MP4 files need conversion."""
        assert needs_conversion("mp4") is True

    def test_m4a_needs_conversion(self) -> None:
        """M4A files need conversion."""
        assert needs_conversion("m4a") is True

    def test_ogg_needs_conversion(self) -> None:
        """OGG files need conversion."""
        assert needs_conversion("ogg") is True

    def test_flac_needs_conversion(self) -> None:
        """FLAC files need conversion."""
        assert needs_conversion("flac") is True


class TestConvertToWav:
    """Tests for convert_to_wav function."""

    @patch("speech.converter.subprocess.run")
    @patch("speech.converter._check_ffmpeg_available", return_value=True)
    @patch("os.path.exists", return_value=True)
    @patch("os.path.getsize", return_value=1024)
    def test_convert_mp3_to_wav(
        self,
        mock_getsize: MagicMock,
        mock_exists: MagicMock,
        mock_ffmpeg_check: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """Should convert MP3 to WAV format using ffmpeg."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            input_path = f.name

        try:
            output_path = convert_to_wav(input_path, "mp3")
            assert output_path.endswith(".wav")

            # Verify ffmpeg was called with correct arguments
            call_args = mock_run.call_args[0][0]
            assert call_args[0] == "ffmpeg"
            assert "-ar" in call_args
            assert "16000" in call_args
            assert "-ac" in call_args
            assert "1" in call_args
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if output_path and os.path.exists(output_path):
                os.unlink(output_path)

    @patch("speech.converter.subprocess.run")
    @patch("speech.converter._check_ffmpeg_available", return_value=True)
    @patch("os.path.exists", return_value=True)
    @patch("os.path.getsize", return_value=1024)
    def test_convert_mp4_to_wav(
        self,
        mock_getsize: MagicMock,
        mock_exists: MagicMock,
        mock_ffmpeg_check: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """Should extract audio from MP4 and convert to WAV."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            input_path = f.name

        try:
            output_path = convert_to_wav(input_path, "mp4")
            assert output_path.endswith(".wav")
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)
            if output_path and os.path.exists(output_path):
                os.unlink(output_path)

    @patch("speech.converter.subprocess.run")
    @patch("speech.converter._check_ffmpeg_available", return_value=True)
    def test_no_audio_track_error(
        self,
        mock_ffmpeg_check: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """Should raise NoAudioTrackError when MP4 has no audio."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="Output file #0 does not contain any stream",
        )

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            input_path = f.name

        try:
            with pytest.raises(NoAudioTrackError):
                convert_to_wav(input_path, "mp4")
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)

    @patch("speech.converter._check_ffmpeg_available", return_value=False)
    def test_ffmpeg_not_available_error(self, mock_ffmpeg_check: MagicMock) -> None:
        """Should raise AudioConversionError when ffmpeg not available."""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            input_path = f.name

        try:
            with pytest.raises(AudioConversionError, match="ffmpeg is not available"):
                convert_to_wav(input_path, "mp3")
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)

    @patch("speech.converter.subprocess.run")
    @patch("speech.converter._check_ffmpeg_available", return_value=True)
    def test_conversion_error_handling(
        self,
        mock_ffmpeg_check: MagicMock,
        mock_run: MagicMock,
    ) -> None:
        """Should raise AudioConversionError on conversion failure."""
        mock_run.return_value = MagicMock(
            returncode=1,
            stderr="Error processing audio",
        )

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            input_path = f.name

        try:
            with pytest.raises(AudioConversionError):
                convert_to_wav(input_path, "mp3")
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)


class TestGetAudioDurationMs:
    """Tests for get_audio_duration_ms function."""

    @patch("speech.converter.subprocess.run")
    def test_get_audio_duration(self, mock_run: MagicMock) -> None:
        """Should return duration in milliseconds."""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="5.5\n",
        )

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            input_path = f.name

        try:
            duration = get_audio_duration_ms(input_path, "mp3")
            assert duration == 5500
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)

    @patch("speech.converter.subprocess.run")
    def test_get_duration_returns_none_on_error(self, mock_run: MagicMock) -> None:
        """Should return None when unable to get duration."""
        mock_run.return_value = MagicMock(returncode=1, stdout="")

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            input_path = f.name

        try:
            duration = get_audio_duration_ms(input_path, "mp3")
            assert duration is None
        finally:
            if os.path.exists(input_path):
                os.unlink(input_path)

