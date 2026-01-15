"""Azure Blob Storage client for audio file storage.

This module provides functionality to upload audio files to Azure Blob Storage
and generate SAS URLs for time-limited access.
"""

import logging
import uuid
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Optional

from azure.core.exceptions import AzureError, ResourceExistsError, ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import (
    BlobSasPermissions,
    BlobServiceClient,
    ContentSettings,
    generate_blob_sas,
)

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class BlobStorageError(Exception):
    """Base exception for Blob Storage operations."""

    pass


class BlobUploadError(BlobStorageError):
    """Raised when a blob upload fails."""

    pass


class BlobNotFoundError(BlobStorageError):
    """Raised when a blob is not found."""

    pass


# Content type mappings for audio formats
AUDIO_CONTENT_TYPES: dict[str, str] = {
    "wav": "audio/wav",
    "mp3": "audio/mpeg",
    "m4a": "audio/mp4",
}


class BlobStorageClient:
    """Client for Azure Blob Storage operations.

    Supports both connection string and managed identity authentication.
    """

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize the Blob Storage client.

        Args:
            settings: Application settings. If None, uses default settings.
        """
        self._settings = settings or get_settings()
        self._service_client: Optional[BlobServiceClient] = None

    @property
    def container_name(self) -> str:
        """Get the blob container name."""
        return self._settings.azure_storage_container

    @property
    def account_name(self) -> Optional[str]:
        """Get the storage account name."""
        return self._settings.azure_storage_account

    @property
    def endpoint(self) -> Optional[str]:
        """Get the storage endpoint URL."""
        return self._settings.storage_endpoint_url

    def is_configured(self) -> bool:
        """Check if blob storage is properly configured."""
        return self._settings.is_storage_configured()

    def _get_service_client(self) -> BlobServiceClient:
        """Get or create the blob service client.

        Returns:
            BlobServiceClient instance

        Raises:
            BlobStorageError: If storage is not configured
        """
        if self._service_client is not None:
            return self._service_client

        if not self.is_configured():
            raise BlobStorageError("Azure Blob Storage is not configured")

        try:
            # Prefer connection string if available
            if self._settings.azure_storage_connection_string:
                self._service_client = BlobServiceClient.from_connection_string(
                    self._settings.azure_storage_connection_string
                )
            else:
                # Use managed identity with account URL
                credential = DefaultAzureCredential()
                endpoint = self.endpoint
                if not endpoint:
                    raise BlobStorageError(
                        "Storage endpoint could not be determined. "
                        "Provide AZURE_STORAGE_ENDPOINT or AZURE_STORAGE_ACCOUNT."
                    )
                self._service_client = BlobServiceClient(
                    account_url=endpoint,
                    credential=credential,
                )

            return self._service_client

        except AzureError as e:
            raise BlobStorageError(f"Failed to create blob service client: {e}") from e

    def _generate_blob_name(
        self,
        original_filename: Optional[str],
        audio_format: str,
        user_id: str,
    ) -> str:
        """Generate a unique blob name.

        Args:
            original_filename: Original file name if available
            audio_format: Audio format extension (wav, mp3, m4a)
            user_id: User ID for ownership tracking

        Returns:
            Unique blob name with format: {user_id}/{uuid}.{extension}
        """
        unique_id = uuid.uuid4().hex
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        extension = audio_format.lower()

        # Include user_id as a prefix for logical partitioning
        return f"{user_id}/{timestamp}_{unique_id}.{extension}"

    async def upload_audio_file(
        self,
        content: bytes,
        audio_format: str,
        user_id: str,
        original_filename: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> str:
        """Upload an audio file to blob storage.

        Args:
            content: File content as bytes
            audio_format: Audio format (wav, mp3, m4a)
            user_id: User ID for ownership tracking
            original_filename: Original file name if available
            metadata: Additional metadata to store with the blob

        Returns:
            Full blob URL after successful upload

        Raises:
            BlobUploadError: If upload fails
        """
        if not self.is_configured():
            raise BlobUploadError("Azure Blob Storage is not configured")

        try:
            service_client = self._get_service_client()
            container_client = service_client.get_container_client(self.container_name)

            # Ensure container exists (auto-create if it doesn't)
            try:
                container_client.create_container()
                logger.info(f"Created '{self.container_name}' container in blob storage")
            except ResourceExistsError:
                # Container already exists, which is fine
                pass
            except AzureError as create_err:
                # Log but continue - container might exist but we lack create permission
                if "ContainerAlreadyExists" not in str(create_err):
                    logger.warning(f"Could not create '{self.container_name}' container: {create_err}")

            # Generate unique blob name
            blob_name = self._generate_blob_name(
                original_filename, audio_format, user_id
            )

            # Get content type for audio format
            content_type = AUDIO_CONTENT_TYPES.get(audio_format.lower(), "application/octet-stream")

            # Build metadata
            blob_metadata = {
                "user_id": user_id,
                "audio_format": audio_format,
                "upload_timestamp": datetime.now(UTC).isoformat(),
            }
            if original_filename:
                blob_metadata["original_filename"] = original_filename
            if metadata:
                blob_metadata.update(metadata)

            # Upload the blob
            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(
                content,
                overwrite=True,
                content_settings=ContentSettings(content_type=content_type),
                metadata=blob_metadata,
            )

            logger.info(f"Uploaded audio file to blob: {blob_name}")

            return blob_client.url

        except AzureError as e:
            error_msg = str(e)
            if "ContainerNotFound" in error_msg:
                logger.error(
                    f"Storage container '{self.container_name}' not found. "
                    f"Please ensure the container exists in your storage account. "
                    f"Check AZURE_STORAGE_CONTAINER environment variable."
                )
                raise BlobUploadError(
                    f"Storage container '{self.container_name}' not found in Azure Blob Storage. "
                    "Please deploy infrastructure or create the container manually."
                ) from e
            logger.error(f"Failed to upload audio file: {e}")
            raise BlobUploadError(f"Failed to upload audio file: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error uploading audio file: {e}")
            raise BlobUploadError(f"Unexpected error uploading audio file: {e}") from e

    def get_blob_sas_url(
        self,
        blob_name: str,
        expiry_hours: int = 1,
        permissions: Optional[BlobSasPermissions] = None,
    ) -> str:
        """Generate a SAS URL for time-limited access to a blob.

        Args:
            blob_name: Name of the blob
            expiry_hours: Number of hours until the SAS token expires
            permissions: Blob permissions (default: read only)

        Returns:
            Full URL with SAS token for accessing the blob

        Raises:
            BlobStorageError: If SAS generation fails or storage not configured
        """
        if not self.is_configured():
            raise BlobStorageError("Azure Blob Storage is not configured")

        if permissions is None:
            permissions = BlobSasPermissions(read=True)

        try:
            # For connection string auth, we can extract the account key
            if self._settings.azure_storage_connection_string:
                # Parse connection string to get account name and key
                conn_str = self._settings.azure_storage_connection_string
                parts = dict(item.split("=", 1) for item in conn_str.split(";") if "=" in item)
                account_name = parts.get("AccountName", "")
                account_key = parts.get("AccountKey", "")

                sas_token = generate_blob_sas(
                    account_name=account_name,
                    container_name=self.container_name,
                    blob_name=blob_name,
                    account_key=account_key,
                    permission=permissions,
                    expiry=datetime.now(UTC) + timedelta(hours=expiry_hours),
                )

                # Build full URL with SAS token
                endpoint = self.endpoint or f"https://{account_name}.blob.core.windows.net"
                return f"{endpoint}/{self.container_name}/{blob_name}?{sas_token}"

            else:
                # For managed identity, we need to use user delegation key
                # This requires additional setup and is more complex
                # For now, return the blob URL without SAS (requires auth)
                service_client = self._get_service_client()
                container_client = service_client.get_container_client(self.container_name)
                blob_client = container_client.get_blob_client(blob_name)

                # Get user delegation key (requires "Microsoft.Storage/storageAccounts/blobServices/generateUserDelegationKey/action" permission)
                start_time = datetime.now(UTC) - timedelta(minutes=5)
                expiry_time = datetime.now(UTC) + timedelta(hours=expiry_hours)

                user_delegation_key = service_client.get_user_delegation_key(
                    key_start_time=start_time,
                    key_expiry_time=expiry_time,
                )

                sas_token = generate_blob_sas(
                    account_name=self.account_name or "",
                    container_name=self.container_name,
                    blob_name=blob_name,
                    user_delegation_key=user_delegation_key,
                    permission=permissions,
                    expiry=expiry_time,
                    start=start_time,
                )

                return f"{blob_client.url}?{sas_token}"

        except AzureError as e:
            logger.error(f"Failed to generate SAS URL: {e}")
            raise BlobStorageError(f"Failed to generate SAS URL: {e}") from e

    def extract_blob_name_from_url(self, blob_url: str) -> str:
        """Extract the blob name from a full blob URL.

        Args:
            blob_url: Full blob URL

        Returns:
            Blob name (path within container)
        """
        # URL format: https://account.blob.core.windows.net/container/blob/path
        # We need to extract everything after the container name
        container_pattern = f"/{self.container_name}/"
        if container_pattern in blob_url:
            # Split on container name and take the path portion
            parts = blob_url.split(container_pattern, 1)
            if len(parts) > 1:
                # Remove any query string (SAS token)
                blob_path = parts[1].split("?")[0]
                return blob_path

        # Fallback: assume the last path segments are the blob name
        from urllib.parse import urlparse

        parsed = urlparse(blob_url)
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) > 1:
            # Skip container name (first part) and return the rest
            return "/".join(path_parts[1:])

        return blob_url

    def _generate_transcription_blob_name(
        self,
        user_id: str,
        transcription_id: str,
    ) -> str:
        """Generate blob name for transcription JSON.

        Format: {user_id}/{transcription_id}.json

        Args:
            user_id: User ID for ownership
            transcription_id: Unique transcription ID

        Returns:
            Blob name path
        """
        return f"{user_id}/{transcription_id}.json"

    async def upload_transcription_json(
        self,
        user_id: str,
        transcription_id: str,
        content_json: str,
        metadata: Optional[dict[str, str]] = None,
    ) -> str:
        """Upload transcription JSON to blob storage.

        Uploads to 'transcriptions' container with format {user_id}/{transcription_id}.json

        Args:
            user_id: User ID for ownership
            transcription_id: Unique transcription ID
            content_json: JSON string content
            metadata: Additional metadata

        Returns:
            Full blob URL

        Raises:
            BlobUploadError: If upload fails
        """
        if not self.is_configured():
            raise BlobUploadError("Azure Blob Storage is not configured")

        try:
            service_client = self._get_service_client()
            # Use 'transcriptions' container for transcription content
            container_client = service_client.get_container_client("transcriptions")

            # Ensure container exists (auto-create if it doesn't)
            try:
                container_client.create_container()
                logger.info("Created 'transcriptions' container in blob storage")
            except ResourceExistsError:
                # Container already exists, which is fine
                pass
            except AzureError as create_err:
                # Log but continue - container might exist but we lack create permission
                if "ContainerAlreadyExists" not in str(create_err):
                    logger.warning(f"Could not create 'transcriptions' container: {create_err}")

            blob_name = self._generate_transcription_blob_name(user_id, transcription_id)

            blob_metadata = {
                "user_id": user_id,
                "transcription_id": transcription_id,
                "upload_timestamp": datetime.now(UTC).isoformat(),
            }
            if metadata:
                blob_metadata.update(metadata)

            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(
                content_json.encode("utf-8"),
                overwrite=True,
                content_settings=ContentSettings(content_type="application/json"),
                metadata=blob_metadata,
            )

            logger.info(f"Uploaded transcription JSON to blob: {blob_name}")
            return blob_client.url

        except AzureError as e:
            error_msg = str(e)
            if "ContainerNotFound" in error_msg:
                logger.error(
                    f"Transcriptions container not found. "
                    f"Please ensure the 'transcriptions' container exists in your storage account. "
                    f"Original error: {e}"
                )
                raise BlobUploadError(
                    "Transcriptions container not found in Azure Blob Storage. "
                    "Please deploy infrastructure or create the container manually."
                ) from e
            logger.error(f"Failed to upload transcription JSON: {e}")
            raise BlobUploadError(f"Failed to upload transcription JSON: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error uploading transcription JSON: {e}")
            raise BlobUploadError(f"Unexpected error uploading transcription JSON: {e}") from e

    async def download_transcription_json(
        self,
        user_id: str,
        transcription_id: str,
    ) -> str:
        """Download transcription JSON from blob storage.

        Args:
            user_id: User ID for ownership
            transcription_id: Unique transcription ID

        Returns:
            JSON string content

        Raises:
            BlobNotFoundError: If blob doesn't exist
            BlobStorageError: If download fails
        """
        if not self.is_configured():
            raise BlobStorageError("Azure Blob Storage is not configured")

        try:
            service_client = self._get_service_client()
            container_client = service_client.get_container_client("transcriptions")

            blob_name = self._generate_transcription_blob_name(user_id, transcription_id)
            blob_client = container_client.get_blob_client(blob_name)

            # Download blob content
            download_stream = blob_client.download_blob()
            content = download_stream.readall()

            return content.decode("utf-8")

        except AzureError as e:
            error_msg = str(e)
            if "BlobNotFound" in error_msg or "NotFound" in error_msg:
                raise BlobNotFoundError(f"Transcription not found: {transcription_id}") from e
            if "ContainerNotFound" in error_msg:
                logger.error(
                    f"Transcriptions container not found. "
                    f"Please ensure the 'transcriptions' container exists in your storage account."
                )
                raise BlobNotFoundError(
                    f"Transcription not found: {transcription_id} (container does not exist)"
                ) from e
            logger.error(f"Failed to download transcription JSON: {e}")
            raise BlobStorageError(f"Failed to download transcription JSON: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error downloading transcription JSON: {e}")
            raise BlobStorageError(f"Unexpected error downloading transcription JSON: {e}") from e

    def get_transcription_sas_url(
        self,
        user_id: str,
        transcription_id: str,
        expiry_hours: int = 1,
    ) -> str:
        """Generate a SAS URL for transcription JSON.

        Args:
            user_id: User ID for ownership
            transcription_id: Transcription ID
            expiry_hours: Hours until SAS expires

        Returns:
            SAS URL for the transcription JSON
        """
        blob_name = self._generate_transcription_blob_name(user_id, transcription_id)

        # Temporarily override container for transcriptions
        original_container = self._settings.azure_storage_container
        self._settings.azure_storage_container = "transcriptions"

        try:
            sas_url = self.get_blob_sas_url(
                blob_name=blob_name,
                expiry_hours=expiry_hours,
                permissions=BlobSasPermissions(read=True),
            )
            return sas_url
        finally:
            self._settings.azure_storage_container = original_container


# In-memory transcription storage
_in_memory_transcription_blobs: dict[str, str] = {}  # blob_name -> content


# In-memory storage for development/testing (when Blob Storage is not configured)
_in_memory_blobs: dict[str, tuple[bytes, dict[str, str]]] = {}  # blob_name -> (content, metadata)


class InMemoryBlobClient(BlobStorageClient):
    """In-memory implementation of BlobStorageClient for development and testing."""

    def is_configured(self) -> bool:
        """Always returns True for in-memory client."""
        return True

    def _get_service_client(self) -> BlobServiceClient:
        """Not used in in-memory implementation."""
        raise NotImplementedError("In-memory client does not use BlobServiceClient")

    async def upload_audio_file(
        self,
        content: bytes,
        audio_format: str,
        user_id: str,
        original_filename: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ) -> str:
        """Upload file to in-memory storage."""
        blob_name = self._generate_blob_name(original_filename, audio_format, user_id)

        blob_metadata = {
            "user_id": user_id,
            "audio_format": audio_format,
            "upload_timestamp": datetime.now(UTC).isoformat(),
        }
        if original_filename:
            blob_metadata["original_filename"] = original_filename
        if metadata:
            blob_metadata.update(metadata)

        _in_memory_blobs[blob_name] = (content, blob_metadata)

        # Return a fake URL for testing
        return f"https://inmemory.blob.local/{self.container_name}/{blob_name}"

    def get_blob_sas_url(
        self,
        blob_name: str,
        expiry_hours: int = 1,
        permissions: Optional[BlobSasPermissions] = None,
    ) -> str:
        """Generate a fake SAS URL for in-memory blob."""
        return f"https://inmemory.blob.local/{self.container_name}/{blob_name}?sas=mock_token"

    async def upload_transcription_json(
        self,
        user_id: str,
        transcription_id: str,
        content_json: str,
        metadata: Optional[dict[str, str]] = None,
    ) -> str:
        """Upload transcription JSON to in-memory storage."""
        blob_name = self._generate_transcription_blob_name(user_id, transcription_id)
        _in_memory_transcription_blobs[blob_name] = content_json
        return f"https://inmemory.blob.local/transcriptions/{blob_name}"

    async def download_transcription_json(
        self,
        user_id: str,
        transcription_id: str,
    ) -> str:
        """Download transcription JSON from in-memory storage."""
        blob_name = self._generate_transcription_blob_name(user_id, transcription_id)
        if blob_name not in _in_memory_transcription_blobs:
            raise BlobNotFoundError(f"Transcription not found: {transcription_id}")
        return _in_memory_transcription_blobs[blob_name]

    def get_transcription_sas_url(
        self,
        user_id: str,
        transcription_id: str,
        expiry_hours: int = 1,
    ) -> str:
        """Generate a fake SAS URL for in-memory transcription."""
        blob_name = self._generate_transcription_blob_name(user_id, transcription_id)
        return f"https://inmemory.blob.local/transcriptions/{blob_name}?sas=mock_token"


def clear_in_memory_blobs() -> None:
    """Clear in-memory blob storage. Useful for tests."""
    _in_memory_blobs.clear()
    _in_memory_transcription_blobs.clear()


@lru_cache
def get_storage_client() -> BlobStorageClient:
    """Get cached Blob Storage client instance.

    Returns InMemoryBlobClient if Blob Storage is not configured.
    """
    settings = get_settings()
    if settings.is_storage_configured():
        return BlobStorageClient(settings)
    return InMemoryBlobClient(settings)
