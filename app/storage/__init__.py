"""Storage client module for Azure Blob Storage."""

from storage.blob import (
    BlobNotFoundError,
    BlobStorageClient,
    BlobStorageError,
    BlobUploadError,
    get_storage_client,
    sanitize_filename,
)
from storage.cache import (
    CacheMetrics,
    compute_audio_hash,
    compute_audio_hash_from_file,
    get_cache_metrics,
)

__all__ = [
    "BlobNotFoundError",
    "BlobStorageClient",
    "BlobStorageError",
    "BlobUploadError",
    "get_storage_client",
    "sanitize_filename",
    "CacheMetrics",
    "compute_audio_hash",
    "compute_audio_hash_from_file",
    "get_cache_metrics",
]
