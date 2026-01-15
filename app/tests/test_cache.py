"""Tests for audio caching functionality."""

import pytest

from storage.cache import (
    CacheMetrics,
    compute_audio_hash,
    compute_audio_hash_from_file,
    get_cache_metrics,
)


class TestComputeAudioHash:
    """Tests for compute_audio_hash function."""

    def test_compute_hash_returns_sha256_hex(self) -> None:
        """Test that hash is a valid SHA256 hex string."""
        content = b"test audio content"
        hash_value = compute_audio_hash(content)
        
        assert len(hash_value) == 64  # SHA256 produces 64 hex characters
        assert all(c in "0123456789abcdef" for c in hash_value)

    def test_same_content_produces_same_hash(self) -> None:
        """Test that identical content produces identical hash."""
        content = b"identical content"
        hash1 = compute_audio_hash(content)
        hash2 = compute_audio_hash(content)
        
        assert hash1 == hash2

    def test_different_content_produces_different_hash(self) -> None:
        """Test that different content produces different hash."""
        hash1 = compute_audio_hash(b"content one")
        hash2 = compute_audio_hash(b"content two")
        
        assert hash1 != hash2

    def test_empty_content_produces_valid_hash(self) -> None:
        """Test that empty content produces valid hash."""
        hash_value = compute_audio_hash(b"")
        
        # SHA256 of empty string
        assert hash_value == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"

    def test_large_content_produces_hash(self) -> None:
        """Test that large content produces valid hash efficiently."""
        # 1MB of data
        content = b"x" * (1024 * 1024)
        hash_value = compute_audio_hash(content)
        
        assert len(hash_value) == 64


class TestComputeAudioHashFromFile:
    """Tests for compute_audio_hash_from_file function."""

    def test_compute_hash_from_file(self, tmp_path) -> None:
        """Test computing hash from a file."""
        content = b"file content for testing"
        test_file = tmp_path / "test_audio.wav"
        test_file.write_bytes(content)
        
        file_hash = compute_audio_hash_from_file(str(test_file))
        memory_hash = compute_audio_hash(content)
        
        assert file_hash == memory_hash

    def test_large_file_chunked_reading(self, tmp_path) -> None:
        """Test that large files are read in chunks correctly."""
        # Create file larger than default chunk size
        content = b"y" * (16 * 1024)  # 16KB
        test_file = tmp_path / "large_audio.wav"
        test_file.write_bytes(content)
        
        file_hash = compute_audio_hash_from_file(str(test_file), chunk_size=1024)
        memory_hash = compute_audio_hash(content)
        
        assert file_hash == memory_hash


class TestCacheMetrics:
    """Tests for CacheMetrics class."""

    def test_initial_state_is_zero(self) -> None:
        """Test that new metrics start at zero."""
        metrics = CacheMetrics()
        
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.total == 0
        assert metrics.hit_rate == 0.0

    def test_record_hit_increments_hits(self) -> None:
        """Test that record_hit increments hit counter."""
        metrics = CacheMetrics()
        metrics.record_hit()
        
        assert metrics.hits == 1
        assert metrics.misses == 0
        assert metrics.total == 1

    def test_record_miss_increments_misses(self) -> None:
        """Test that record_miss increments miss counter."""
        metrics = CacheMetrics()
        metrics.record_miss()
        
        assert metrics.hits == 0
        assert metrics.misses == 1
        assert metrics.total == 1

    def test_hit_rate_calculation(self) -> None:
        """Test hit rate percentage calculation."""
        metrics = CacheMetrics()
        metrics.record_hit()
        metrics.record_hit()
        metrics.record_miss()
        metrics.record_miss()
        
        assert metrics.hit_rate == 50.0

    def test_hit_rate_all_hits(self) -> None:
        """Test hit rate when all are hits."""
        metrics = CacheMetrics()
        metrics.record_hit()
        metrics.record_hit()
        metrics.record_hit()
        
        assert metrics.hit_rate == 100.0

    def test_hit_rate_all_misses(self) -> None:
        """Test hit rate when all are misses."""
        metrics = CacheMetrics()
        metrics.record_miss()
        metrics.record_miss()
        
        assert metrics.hit_rate == 0.0

    def test_reset_clears_all_counters(self) -> None:
        """Test that reset clears all counters."""
        metrics = CacheMetrics()
        metrics.record_hit()
        metrics.record_miss()
        metrics.reset()
        
        assert metrics.hits == 0
        assert metrics.misses == 0
        assert metrics.total == 0


class TestGetCacheMetrics:
    """Tests for get_cache_metrics singleton."""

    def test_returns_same_instance(self) -> None:
        """Test that get_cache_metrics returns singleton."""
        metrics1 = get_cache_metrics()
        metrics2 = get_cache_metrics()
        
        assert metrics1 is metrics2

    def test_state_persists_across_calls(self) -> None:
        """Test that state persists in singleton."""
        metrics = get_cache_metrics()
        initial_hits = metrics.hits
        metrics.record_hit()
        
        metrics2 = get_cache_metrics()
        assert metrics2.hits == initial_hits + 1
