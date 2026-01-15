"""Audio caching utilities for transcription deduplication.

This module provides functionality for computing audio file hashes
and managing transcription cache lookups.
"""

import hashlib
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def compute_audio_hash(content: bytes) -> str:
    """Compute SHA256 hash of audio file content.
    
    Args:
        content: Raw audio file bytes
        
    Returns:
        Hexadecimal SHA256 hash string
    """
    return hashlib.sha256(content).hexdigest()


def compute_audio_hash_from_file(file_path: str, chunk_size: int = 8192) -> str:
    """Compute SHA256 hash of audio file from disk.
    
    Reads file in chunks for memory efficiency with large files.
    
    Args:
        file_path: Path to the audio file
        chunk_size: Size of chunks to read at a time
        
    Returns:
        Hexadecimal SHA256 hash string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


class CacheMetrics:
    """Simple in-memory cache metrics tracker.
    
    Tracks cache hit/miss statistics for monitoring.
    """
    
    def __init__(self) -> None:
        self._hits = 0
        self._misses = 0
    
    @property
    def hits(self) -> int:
        return self._hits
    
    @property
    def misses(self) -> int:
        return self._misses
    
    @property
    def total(self) -> int:
        return self._hits + self._misses
    
    @property
    def hit_rate(self) -> float:
        """Return hit rate as percentage (0-100)."""
        if self.total == 0:
            return 0.0
        return (self._hits / self.total) * 100
    
    def record_hit(self) -> None:
        """Record a cache hit."""
        self._hits += 1
        logger.info(f"Cache HIT - Total: {self._hits} hits, {self._misses} misses, {self.hit_rate:.1f}% hit rate")
    
    def record_miss(self) -> None:
        """Record a cache miss."""
        self._misses += 1
        logger.info(f"Cache MISS - Total: {self._hits} hits, {self._misses} misses, {self.hit_rate:.1f}% hit rate")
    
    def reset(self) -> None:
        """Reset all metrics."""
        self._hits = 0
        self._misses = 0


# Global cache metrics instance
_cache_metrics: Optional[CacheMetrics] = None


def get_cache_metrics() -> CacheMetrics:
    """Get the global cache metrics instance."""
    global _cache_metrics
    if _cache_metrics is None:
        _cache_metrics = CacheMetrics()
    return _cache_metrics
