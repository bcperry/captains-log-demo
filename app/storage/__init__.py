"""Storage client module for Azure Blob Storage."""

from storage.blob import (
    BlobNotFoundError,
    BlobStorageClient,
    BlobStorageError,
    BlobUploadError,
    get_storage_client,
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
    "CacheMetrics",
    "compute_audio_hash",
    "compute_audio_hash_from_file",
    "get_cache_metrics",
]
