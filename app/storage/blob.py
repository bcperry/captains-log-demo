"""Azure Blob Storage client for audio file storage.

This module provides functionality to upload audio files to Azure Blob Storage
and generate SAS URLs for time-limited access.

Hierarchical storage structure:
- Audio files: {container}/{user_id}/{sanitized_filename}_YYYYMMDD_HHMMSS/audio/{filename}.{ext}
- Transcripts: {container}/{user_id}/{sanitized_filename}_YYYYMMDD_HHMMSS/transcript.json
"""

import json
import logging
import os
import re
import uuid
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Optional
from urllib.parse import quote, unquote

from azure.core.exceptions import AzureError, ResourceExistsError, ResourceNotFoundError
from azure.identity import DefaultAzureCredential
from azure.storage.blob import (
    BlobSasPermissions,
    BlobServiceClient,
    ContainerSasPermissions,
    ContentSettings,
    generate_blob_sas,
    generate_container_sas,
)

from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """Sanitize a filename for use in blob storage paths.

    Removes or replaces characters that are invalid in blob paths:
    - Replaces spaces with underscores
    - Removes special characters except alphanumeric, dash, underscore, dot
    - Limits length to 100 characters
    - Strips leading/trailing whitespace and dots

    Args:
        filename: Original filename to sanitize

    Returns:
        Sanitized filename safe for blob storage paths
    """
    if not filename:
        return "unnamed"

    # Remove extension first, we'll add it back if needed
    name_without_ext = os.path.splitext(filename)[0]

    # Replace spaces with underscores
    sanitized = name_without_ext.replace(" ", "_")

    # Keep only alphanumeric, dash, underscore, dot
    sanitized = re.sub(r"[^a-zA-Z0-9\-_.]", "", sanitized)

    # Remove consecutive underscores/dashes
    sanitized = re.sub(r"[_\-]{2,}", "_", sanitized)

    # Strip leading/trailing dots and underscores
    sanitized = sanitized.strip("._-")

    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:100]

    # Ensure we have a valid name
    if not sanitized:
        sanitized = "unnamed"

    return sanitized


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

                # Build full URL with SAS token (URL-encode blob_name to handle spaces/special chars)
                endpoint = self.endpoint or f"https://{account_name}.blob.core.windows.net"
                encoded_blob_name = quote(blob_name, safe="/")
                return f"{endpoint}/{self.container_name}/{encoded_blob_name}?{sas_token}"

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

                # URL-encode blob_name in URL to handle spaces/special chars
                encoded_url = quote(blob_client.url, safe="/:@")
                return f"{encoded_url}?{sas_token}"

        except AzureError as e:
            logger.error(f"Failed to generate SAS URL: {e}")
            raise BlobStorageError(f"Failed to generate SAS URL: {e}") from e

    def extract_blob_name_from_url(self, blob_url: str) -> str:
        """Extract the blob name from a full blob URL.

        Args:
            blob_url: Full blob URL

        Returns:
            Blob name (path within container), URL-decoded to handle spaces/special chars
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
                # URL-decode to handle spaces and special characters
                return unquote(blob_path)

        # Fallback: assume the last path segments are the blob name
        from urllib.parse import urlparse

        parsed = urlparse(blob_url)
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) > 1:
            # Skip container name (first part) and return the rest, URL-decoded
            return unquote("/".join(path_parts[1:]))

        return unquote(blob_url)

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

    def _generate_user_path_folder(
        self,
        user_id: str,
        original_filename: Optional[str],
        timestamp: Optional[datetime] = None,
    ) -> str:
        """Generate hierarchical folder path for user-scoped storage.

        Format: {user_id}/{sanitized_filename}_YYYYMMDD_HHMMSS

        Args:
            user_id: User ID (GUID from Azure AD oid claim)
            original_filename: Original filename (will be sanitized)
            timestamp: Optional timestamp (defaults to now)

        Returns:
            Folder path string
        """
        ts = timestamp or datetime.now(UTC)
        timestamp_str = ts.strftime("%Y%m%d_%H%M%S")

        sanitized_name = sanitize_filename(original_filename or "unnamed")

        return f"{user_id}/{sanitized_name}_{timestamp_str}"

    async def upload_audio_with_user_path(
        self,
        content: bytes,
        audio_format: str,
        user_id: str,
        original_filename: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> tuple[str, str]:
        """Upload audio file to user-scoped hierarchical blob path.

        Stores audio in: {container}/{user_id}/{sanitized_filename}_YYYYMMDD_HHMMSS/audio/{original_filename}.{ext}

        Args:
            content: File content as bytes
            audio_format: Audio format (wav, mp3, m4a, etc.)
            user_id: User ID (GUID from Azure AD oid claim)
            original_filename: Original file name
            metadata: Additional metadata to store with the blob
            timestamp: Optional timestamp for folder name (defaults to now)

        Returns:
            Tuple of (blob_url, folder_path) - full URL and folder path for related files

        Raises:
            BlobUploadError: If upload fails
        """
        if not self.is_configured():
            raise BlobUploadError("Azure Blob Storage is not configured")

        ts = timestamp or datetime.now(UTC)

        try:
            service_client = self._get_service_client()
            container_client = service_client.get_container_client(self.container_name)

            # Ensure container exists
            try:
                container_client.create_container()
                logger.info(f"Created '{self.container_name}' container in blob storage")
            except ResourceExistsError:
                pass
            except AzureError as create_err:
                if "ContainerAlreadyExists" not in str(create_err):
                    logger.warning(f"Could not create '{self.container_name}' container: {create_err}")

            # Generate hierarchical folder path
            folder_path = self._generate_user_path_folder(user_id, original_filename, ts)

            # Preserve original filename with extension
            sanitized_name = sanitize_filename(original_filename or "audio")
            extension = audio_format.lower()
            filename_with_ext = f"{sanitized_name}.{extension}"

            # Full blob path: folder/audio/{filename}
            blob_name = f"{folder_path}/audio/{filename_with_ext}"

            # Get content type
            content_type = AUDIO_CONTENT_TYPES.get(extension, "application/octet-stream")

            # Build metadata
            blob_metadata = {
                "user_id": user_id,
                "audio_format": audio_format,
                "original_filename": original_filename or "",
                "upload_timestamp": ts.isoformat(),
                "folder_path": folder_path,
            }
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

            logger.info(f"Uploaded audio file to hierarchical path: {blob_name}")

            return blob_client.url, folder_path

        except AzureError as e:
            error_msg = str(e)
            if "ContainerNotFound" in error_msg:
                raise BlobUploadError(
                    f"Storage container '{self.container_name}' not found in Azure Blob Storage."
                ) from e
            raise BlobUploadError(f"Failed to upload audio file: {e}") from e
        except Exception as e:
            raise BlobUploadError(f"Unexpected error uploading audio file: {e}") from e

    async def upload_transcription_with_user_path(
        self,
        user_id: str,
        folder_path: str,
        content_json: str,
        metadata: Optional[dict[str, str]] = None,
    ) -> str:
        """Upload transcription JSON to the same folder as the audio file.

        Stores in: {container}/{folder_path}/transcript.json

        Args:
            user_id: User ID for ownership
            folder_path: Folder path returned from upload_audio_with_user_path
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
            container_client = service_client.get_container_client(self.container_name)

            # Ensure container exists
            try:
                container_client.create_container()
            except ResourceExistsError:
                pass
            except AzureError:
                pass

            # Transcript goes in same folder as audio
            blob_name = f"{folder_path}/transcript.json"

            blob_metadata = {
                "user_id": user_id,
                "content_type": "transcription",
                "upload_timestamp": datetime.now(UTC).isoformat(),
                "folder_path": folder_path,
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

            logger.info(f"Uploaded transcription JSON to hierarchical path: {blob_name}")
            return blob_client.url

        except AzureError as e:
            raise BlobUploadError(f"Failed to upload transcription JSON: {e}") from e
        except Exception as e:
            raise BlobUploadError(f"Unexpected error uploading transcription JSON: {e}") from e

    async def download_audio_from_user_path(
        self,
        folder_path: str,
        filename: str,
    ) -> bytes:
        """Download audio file from user-scoped hierarchical path.

        Args:
            folder_path: Folder path (e.g., "{user_id}/{filename}_{timestamp}")
            filename: Filename with extension

        Returns:
            Audio content as bytes

        Raises:
            BlobNotFoundError: If file doesn't exist
            BlobStorageError: If download fails
        """
        if not self.is_configured():
            raise BlobStorageError("Azure Blob Storage is not configured")

        try:
            service_client = self._get_service_client()
            container_client = service_client.get_container_client(self.container_name)

            blob_name = f"{folder_path}/audio/{filename}"
            blob_client = container_client.get_blob_client(blob_name)

            download_stream = blob_client.download_blob()
            return download_stream.readall()

        except AzureError as e:
            error_msg = str(e)
            if "BlobNotFound" in error_msg or "NotFound" in error_msg:
                raise BlobNotFoundError(f"Audio file not found: {filename}") from e
            raise BlobStorageError(f"Failed to download audio file: {e}") from e

    async def download_transcription_from_user_path(
        self,
        folder_path: str,
    ) -> str:
        """Download transcription JSON from user-scoped hierarchical path.

        Args:
            folder_path: Folder path (e.g., "{user_id}/{filename}_{timestamp}")

        Returns:
            JSON string content

        Raises:
            BlobNotFoundError: If file doesn't exist
            BlobStorageError: If download fails
        """
        if not self.is_configured():
            raise BlobStorageError("Azure Blob Storage is not configured")

        try:
            service_client = self._get_service_client()
            container_client = service_client.get_container_client(self.container_name)

            blob_name = f"{folder_path}/transcript.json"
            blob_client = container_client.get_blob_client(blob_name)

            download_stream = blob_client.download_blob()
            return download_stream.readall().decode("utf-8")

        except AzureError as e:
            error_msg = str(e)
            if "BlobNotFound" in error_msg or "NotFound" in error_msg:
                raise BlobNotFoundError(f"Transcription not found at: {folder_path}") from e
            raise BlobStorageError(f"Failed to download transcription: {e}") from e

    async def save_metadata(
        self,
        user_id: str,
        folder_path: str,
        metadata_json: str,
    ) -> str:
        """Save transcription metadata JSON to blob storage.

        Stores as: {container}/{folder_path}/metadata.json

        Args:
            user_id: User ID for ownership
            folder_path: Folder path from upload_audio_with_user_path
            metadata_json: JSON string with metadata

        Returns:
            Full blob URL

        Raises:
            BlobUploadError: If upload fails
        """
        if not self.is_configured():
            raise BlobUploadError("Azure Blob Storage is not configured")

        try:
            service_client = self._get_service_client()
            container_client = service_client.get_container_client(self.container_name)

            # Ensure container exists
            try:
                container_client.create_container()
            except ResourceExistsError:
                pass
            except AzureError:
                pass

            blob_name = f"{folder_path}/metadata.json"

            blob_metadata = {
                "user_id": user_id,
                "content_type": "metadata",
                "upload_timestamp": datetime.now(UTC).isoformat(),
                "folder_path": folder_path,
            }

            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(
                metadata_json.encode("utf-8"),
                overwrite=True,
                content_settings=ContentSettings(content_type="application/json"),
                metadata=blob_metadata,
            )

            logger.info(f"Saved metadata JSON to: {blob_name}")
            return blob_client.url

        except AzureError as e:
            raise BlobUploadError(f"Failed to save metadata JSON: {e}") from e
        except Exception as e:
            raise BlobUploadError(f"Unexpected error saving metadata JSON: {e}") from e

    async def get_metadata(
        self,
        folder_path: str,
    ) -> str:
        """Download metadata JSON from blob storage.

        Args:
            folder_path: Folder path (e.g., "{user_id}/{filename}_{timestamp}")

        Returns:
            JSON string content

        Raises:
            BlobNotFoundError: If metadata doesn't exist
            BlobStorageError: If download fails
        """
        if not self.is_configured():
            raise BlobStorageError("Azure Blob Storage is not configured")

        try:
            service_client = self._get_service_client()
            container_client = service_client.get_container_client(self.container_name)

            blob_name = f"{folder_path}/metadata.json"
            blob_client = container_client.get_blob_client(blob_name)

            download_stream = blob_client.download_blob()
            return download_stream.readall().decode("utf-8")

        except AzureError as e:
            error_msg = str(e)
            if "BlobNotFound" in error_msg or "NotFound" in error_msg:
                raise BlobNotFoundError(f"Metadata not found at: {folder_path}") from e
            raise BlobStorageError(f"Failed to download metadata: {e}") from e

    async def save_analysis_json(
        self,
        user_id: str,
        folder_path: str,
        analysis_json: str,
    ) -> str:
        """Save AI analysis JSON to blob storage.

        Stores as: {container}/{folder_path}/analysis.json

        Args:
            user_id: User ID for ownership
            folder_path: Folder path from upload_audio_with_user_path
            analysis_json: JSON string with analysis results

        Returns:
            Full blob URL

        Raises:
            BlobUploadError: If upload fails
        """
        if not self.is_configured():
            raise BlobUploadError("Azure Blob Storage is not configured")

        try:
            service_client = self._get_service_client()
            container_client = service_client.get_container_client(self.container_name)

            # Ensure container exists
            try:
                container_client.create_container()
            except ResourceExistsError:
                pass
            except AzureError:
                pass

            blob_name = f"{folder_path}/analysis.json"

            blob_metadata = {
                "user_id": user_id,
                "content_type": "analysis",
                "upload_timestamp": datetime.now(UTC).isoformat(),
                "folder_path": folder_path,
            }

            blob_client = container_client.get_blob_client(blob_name)
            blob_client.upload_blob(
                analysis_json.encode("utf-8"),
                overwrite=True,
                content_settings=ContentSettings(content_type="application/json"),
                metadata=blob_metadata,
            )

            logger.info(f"Saved analysis JSON to: {blob_name}")
            return blob_client.url

        except AzureError as e:
            raise BlobUploadError(f"Failed to save analysis JSON: {e}") from e
        except Exception as e:
            raise BlobUploadError(f"Unexpected error saving analysis JSON: {e}") from e

    async def get_analysis_json(
        self,
        folder_path: str,
    ) -> str:
        """Download analysis JSON from blob storage.

        Args:
            folder_path: Folder path (e.g., "{user_id}/{filename}_{timestamp}")

        Returns:
            JSON string content

        Raises:
            BlobNotFoundError: If analysis doesn't exist
            BlobStorageError: If download fails
        """
        if not self.is_configured():
            raise BlobStorageError("Azure Blob Storage is not configured")

        try:
            service_client = self._get_service_client()
            container_client = service_client.get_container_client(self.container_name)

            blob_name = f"{folder_path}/analysis.json"
            blob_client = container_client.get_blob_client(blob_name)

            download_stream = blob_client.download_blob()
            return download_stream.readall().decode("utf-8")

        except AzureError as e:
            error_msg = str(e)
            if "BlobNotFound" in error_msg or "NotFound" in error_msg:
                raise BlobNotFoundError(f"Analysis not found at: {folder_path}") from e
            raise BlobStorageError(f"Failed to download analysis: {e}") from e

    async def analysis_exists(
        self,
        folder_path: str,
    ) -> bool:
        """Check if analysis.json exists for a transcription.

        Args:
            folder_path: Folder path (e.g., "{user_id}/{filename}_{timestamp}")

        Returns:
            True if analysis exists, False otherwise
        """
        if not self.is_configured():
            return False

        try:
            service_client = self._get_service_client()
            container_client = service_client.get_container_client(self.container_name)

            blob_name = f"{folder_path}/analysis.json"
            blob_client = container_client.get_blob_client(blob_name)

            return blob_client.exists()

        except AzureError:
            return False

    async def list_user_transcriptions(
        self,
        user_id: str,
        page: int = 1,
        per_page: int = 20,
    ) -> tuple[list[dict], int]:
        """List all transcriptions for a user by scanning blob storage.

        Scans for metadata.json files under the user's folder prefix.

        Args:
            user_id: User ID to list transcriptions for
            page: Page number (1-indexed)
            per_page: Results per page

        Returns:
            Tuple of (list of metadata dicts, total count)

        Raises:
            BlobStorageError: If listing fails
        """
        if not self.is_configured():
            raise BlobStorageError("Azure Blob Storage is not configured")

        try:

            service_client = self._get_service_client()
            container_client = service_client.get_container_client(self.container_name)

            # List all blobs under user's folder and find metadata.json files
            metadata_blobs: list[str] = []
            blobs = container_client.list_blobs(name_starts_with=f"{user_id}/")

            for blob in blobs:
                if blob.name.endswith("/metadata.json"):
                    metadata_blobs.append(blob.name)

            # Sort by name (which includes timestamp) in reverse order (newest first)
            metadata_blobs.sort(reverse=True)

            total = len(metadata_blobs)

            # Apply pagination
            start = (page - 1) * per_page
            end = start + per_page
            paginated_blobs = metadata_blobs[start:end]

            # Fetch metadata content for each blob
            metadata_list: list[dict] = []
            for blob_name in paginated_blobs:
                try:
                    blob_client = container_client.get_blob_client(blob_name)
                    download_stream = blob_client.download_blob()
                    content = download_stream.readall().decode("utf-8")
                    metadata = json.loads(content)
                    metadata_list.append(metadata)
                except Exception as e:
                    logger.warning(f"Failed to read metadata from {blob_name}: {e}")
                    continue

            return metadata_list, total

        except AzureError as e:
            raise BlobStorageError(f"Failed to list user transcriptions: {e}") from e

    async def get_transcription_by_audio_hash(
        self,
        user_id: str,
        audio_hash: str,
    ) -> dict | None:
        """Find a transcription by audio hash for cache lookup.

        Scans metadata.json files for matching audio_hash.

        Args:
            user_id: User ID to search for
            audio_hash: SHA256 hash of audio file content

        Returns:
            Metadata dict if found, None otherwise
        """
        if not self.is_configured():
            return None

        try:

            service_client = self._get_service_client()
            container_client = service_client.get_container_client(self.container_name)

            # List all metadata.json files for user
            blobs = container_client.list_blobs(name_starts_with=f"{user_id}/")

            for blob in blobs:
                if blob.name.endswith("/metadata.json"):
                    try:
                        blob_client = container_client.get_blob_client(blob.name)
                        download_stream = blob_client.download_blob()
                        content = download_stream.readall().decode("utf-8")
                        metadata: dict = json.loads(content)

                        if metadata.get("audio_hash") == audio_hash:
                            return metadata
                    except Exception as e:
                        logger.warning(f"Failed to read metadata from {blob.name}: {e}")
                        continue

            return None

        except AzureError as e:
            logger.error(f"Failed to search by audio hash: {e}")
            return None

    def get_user_folder_sas_url(
        self,
        user_id: str,
        expiry_hours: int = 1,
    ) -> str:
        """Generate a SAS URL scoped to a user's folder in the container.

        This allows read access to all blobs under the user's GUID prefix.

        Args:
            user_id: User ID (GUID from Azure AD)
            expiry_hours: Hours until SAS expires

        Returns:
            SAS URL with container-level permissions scoped by prefix

        Raises:
            BlobStorageError: If SAS generation fails
        """
        if not self.is_configured():
            raise BlobStorageError("Azure Blob Storage is not configured")

        try:
            if self._settings.azure_storage_connection_string:
                conn_str = self._settings.azure_storage_connection_string
                parts = dict(item.split("=", 1) for item in conn_str.split(";") if "=" in item)
                account_name = parts.get("AccountName", "")
                account_key = parts.get("AccountKey", "")

                sas_token = generate_container_sas(
                    account_name=account_name,
                    container_name=self.container_name,
                    account_key=account_key,
                    permission=ContainerSasPermissions(read=True, list=True),
                    expiry=datetime.now(UTC) + timedelta(hours=expiry_hours),
                )

                endpoint = self.endpoint or f"https://{account_name}.blob.core.windows.net"
                # Return container URL with SAS - client should filter by user_id prefix
                return f"{endpoint}/{self.container_name}?{sas_token}&prefix={user_id}/"
            else:
                # For managed identity, use user delegation key
                service_client = self._get_service_client()

                start_time = datetime.now(UTC) - timedelta(minutes=5)
                expiry_time = datetime.now(UTC) + timedelta(hours=expiry_hours)

                user_delegation_key = service_client.get_user_delegation_key(
                    key_start_time=start_time,
                    key_expiry_time=expiry_time,
                )

                sas_token = generate_container_sas(
                    account_name=self.account_name or "",
                    container_name=self.container_name,
                    user_delegation_key=user_delegation_key,
                    permission=ContainerSasPermissions(read=True, list=True),
                    expiry=expiry_time,
                    start=start_time,
                )

                endpoint = self.endpoint or f"https://{self.account_name}.blob.core.windows.net"
                return f"{endpoint}/{self.container_name}?{sas_token}&prefix={user_id}/"

        except AzureError as e:
            raise BlobStorageError(f"Failed to generate user folder SAS URL: {e}") from e

    async def archive_old_recordings(
        self,
        user_id: str,
        retention_days: int = 90,
        archive_container: str = "archive",
    ) -> list[str]:
        """Archive recordings older than retention period.

        Moves recordings older than retention_days to an archive container.

        Args:
            user_id: User ID to archive recordings for
            retention_days: Number of days to retain recordings before archiving
            archive_container: Name of the archive container

        Returns:
            List of archived blob paths

        Raises:
            BlobStorageError: If archiving fails
        """
        if not self.is_configured():
            raise BlobStorageError("Azure Blob Storage is not configured")

        archived_paths: list[str] = []
        cutoff_date = datetime.now(UTC) - timedelta(days=retention_days)

        try:
            service_client = self._get_service_client()
            source_container = service_client.get_container_client(self.container_name)

            # Ensure archive container exists
            archive_container_client = service_client.get_container_client(archive_container)
            try:
                archive_container_client.create_container()
                logger.info(f"Created archive container: {archive_container}")
            except ResourceExistsError:
                pass

            # List blobs under user's folder
            blobs = source_container.list_blobs(name_starts_with=f"{user_id}/")

            for blob in blobs:
                # Check blob creation time
                if blob.last_modified and blob.last_modified.replace(tzinfo=UTC) < cutoff_date:
                    # Copy to archive
                    source_blob = source_container.get_blob_client(blob.name)
                    archive_blob = archive_container_client.get_blob_client(blob.name)

                    # Copy blob to archive
                    archive_blob.start_copy_from_url(source_blob.url)

                    # Delete from source
                    source_blob.delete_blob()

                    archived_paths.append(blob.name)
                    logger.info(f"Archived blob: {blob.name}")

            return archived_paths

        except AzureError as e:
            raise BlobStorageError(f"Failed to archive recordings: {e}") from e

    async def delete_old_recordings(
        self,
        user_id: str,
        retention_days: int = 365,
    ) -> list[str]:
        """Delete recordings older than retention period.

        Permanently deletes recordings older than retention_days.

        Args:
            user_id: User ID to delete recordings for
            retention_days: Number of days to retain before deletion

        Returns:
            List of deleted blob paths

        Raises:
            BlobStorageError: If deletion fails
        """
        if not self.is_configured():
            raise BlobStorageError("Azure Blob Storage is not configured")

        deleted_paths: list[str] = []
        cutoff_date = datetime.now(UTC) - timedelta(days=retention_days)

        try:
            service_client = self._get_service_client()
            container_client = service_client.get_container_client(self.container_name)

            # List blobs under user's folder
            blobs = container_client.list_blobs(name_starts_with=f"{user_id}/")

            for blob in blobs:
                if blob.last_modified and blob.last_modified.replace(tzinfo=UTC) < cutoff_date:
                    blob_client = container_client.get_blob_client(blob.name)
                    blob_client.delete_blob()
                    deleted_paths.append(blob.name)
                    logger.info(f"Deleted old blob: {blob.name}")

            return deleted_paths

        except AzureError as e:
            raise BlobStorageError(f"Failed to delete old recordings: {e}") from e

    async def delete_transcription_folder(
        self,
        folder_path: str,
    ) -> tuple[list[str], list[str]]:
        """Delete all blobs in a transcription folder.

        Deletes the entire transcription folder including:
        - audio/{filename}
        - transcript.json
        - metadata.json
        - analysis.json (if exists)

        Args:
            folder_path: Full folder path (e.g., "user_id/filename_timestamp")

        Returns:
            Tuple of (deleted_blobs, failed_blobs) - lists of blob names

        Raises:
            BlobNotFoundError: If no blobs exist in the folder
            BlobStorageError: If deletion fails entirely
        """
        if not self.is_configured():
            raise BlobStorageError("Azure Blob Storage is not configured")

        deleted_blobs: list[str] = []
        failed_blobs: list[str] = []

        try:
            service_client = self._get_service_client()
            container_client = service_client.get_container_client(self.container_name)

            # Ensure folder_path ends with / for proper prefix matching
            prefix = folder_path if folder_path.endswith("/") else f"{folder_path}/"

            # List all blobs in the folder
            blobs = list(container_client.list_blobs(name_starts_with=prefix))

            if not blobs:
                raise BlobNotFoundError(f"No blobs found in folder: {folder_path}")

            # Delete each blob
            for blob in blobs:
                try:
                    blob_client = container_client.get_blob_client(blob.name)
                    blob_client.delete_blob()
                    deleted_blobs.append(blob.name)
                    logger.info(f"Deleted blob: {blob.name}")
                except AzureError as e:
                    logger.warning(f"Failed to delete blob {blob.name}: {e}")
                    failed_blobs.append(blob.name)

            if deleted_blobs:
                logger.info(
                    f"Deleted {len(deleted_blobs)} blobs from folder {folder_path}"
                )

            return deleted_blobs, failed_blobs

        except ResourceNotFoundError:
            raise BlobNotFoundError(f"Folder not found: {folder_path}")
        except BlobNotFoundError:
            raise
        except AzureError as e:
            raise BlobStorageError(f"Failed to delete transcription folder: {e}") from e


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

    async def upload_audio_with_user_path(
        self,
        content: bytes,
        audio_format: str,
        user_id: str,
        original_filename: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> tuple[str, str]:
        """Upload audio file to in-memory storage with user path structure."""
        ts = timestamp or datetime.now(UTC)
        folder_path = self._generate_user_path_folder(user_id, original_filename, ts)

        sanitized_name = sanitize_filename(original_filename or "audio")
        extension = audio_format.lower()
        filename_with_ext = f"{sanitized_name}.{extension}"
        blob_name = f"{folder_path}/audio/{filename_with_ext}"

        blob_metadata = {
            "user_id": user_id,
            "audio_format": audio_format,
            "original_filename": original_filename or "",
            "upload_timestamp": ts.isoformat(),
            "folder_path": folder_path,
        }
        if metadata:
            blob_metadata.update(metadata)

        _in_memory_blobs[blob_name] = (content, blob_metadata)

        url = f"https://inmemory.blob.local/{self.container_name}/{blob_name}"
        return url, folder_path

    async def upload_transcription_with_user_path(
        self,
        user_id: str,
        folder_path: str,
        content_json: str,
        metadata: Optional[dict[str, str]] = None,
    ) -> str:
        """Upload transcription JSON to in-memory storage with user path structure."""
        blob_name = f"{folder_path}/transcript.json"
        _in_memory_transcription_blobs[blob_name] = content_json
        return f"https://inmemory.blob.local/{self.container_name}/{blob_name}"

    async def download_audio_from_user_path(
        self,
        folder_path: str,
        filename: str,
    ) -> bytes:
        """Download audio file from in-memory storage."""
        blob_name = f"{folder_path}/audio/{filename}"
        if blob_name not in _in_memory_blobs:
            raise BlobNotFoundError(f"Audio file not found: {filename}")
        return _in_memory_blobs[blob_name][0]

    async def download_transcription_from_user_path(
        self,
        folder_path: str,
    ) -> str:
        """Download transcription JSON from in-memory storage."""
        blob_name = f"{folder_path}/transcript.json"
        if blob_name not in _in_memory_transcription_blobs:
            raise BlobNotFoundError(f"Transcription not found at: {folder_path}")
        return _in_memory_transcription_blobs[blob_name]

    async def save_metadata(
        self,
        user_id: str,
        folder_path: str,
        metadata_json: str,
    ) -> str:
        """Save metadata JSON to in-memory storage."""
        blob_name = f"{folder_path}/metadata.json"
        _in_memory_transcription_blobs[blob_name] = metadata_json
        return f"https://inmemory.blob.local/{self.container_name}/{blob_name}"

    async def get_metadata(
        self,
        folder_path: str,
    ) -> str:
        """Download metadata JSON from in-memory storage."""
        blob_name = f"{folder_path}/metadata.json"
        if blob_name not in _in_memory_transcription_blobs:
            raise BlobNotFoundError(f"Metadata not found at: {folder_path}")
        return _in_memory_transcription_blobs[blob_name]

    async def save_analysis_json(
        self,
        user_id: str,
        folder_path: str,
        analysis_json: str,
    ) -> str:
        """Save analysis JSON to in-memory storage."""
        blob_name = f"{folder_path}/analysis.json"
        _in_memory_transcription_blobs[blob_name] = analysis_json
        return f"https://inmemory.blob.local/{self.container_name}/{blob_name}"

    async def get_analysis_json(
        self,
        folder_path: str,
    ) -> str:
        """Download analysis JSON from in-memory storage."""
        blob_name = f"{folder_path}/analysis.json"
        if blob_name not in _in_memory_transcription_blobs:
            raise BlobNotFoundError(f"Analysis not found at: {folder_path}")
        return _in_memory_transcription_blobs[blob_name]

    async def analysis_exists(
        self,
        folder_path: str,
    ) -> bool:
        """Check if analysis.json exists in in-memory storage."""
        blob_name = f"{folder_path}/analysis.json"
        return blob_name in _in_memory_transcription_blobs

    async def list_user_transcriptions(
        self,
        user_id: str,
        page: int = 1,
        per_page: int = 20,
    ) -> tuple[list[dict], int]:
        """List transcriptions from in-memory storage."""

        # Find all metadata.json files for this user
        metadata_blobs: list[str] = []
        for blob_name in _in_memory_transcription_blobs:
            if blob_name.startswith(f"{user_id}/") and blob_name.endswith("/metadata.json"):
                metadata_blobs.append(blob_name)

        # Sort by name (which includes timestamp) in reverse order (newest first)
        metadata_blobs.sort(reverse=True)

        total = len(metadata_blobs)

        # Apply pagination
        start = (page - 1) * per_page
        end = start + per_page
        paginated_blobs = metadata_blobs[start:end]

        # Parse metadata content
        metadata_list: list[dict] = []
        for blob_name in paginated_blobs:
            try:
                content = _in_memory_transcription_blobs[blob_name]
                metadata = json.loads(content)
                metadata_list.append(metadata)
            except Exception:
                continue

        return metadata_list, total

    async def get_transcription_by_audio_hash(
        self,
        user_id: str,
        audio_hash: str,
    ) -> dict | None:
        """Find transcription by audio hash in in-memory storage."""

        for blob_name, content in _in_memory_transcription_blobs.items():
            if blob_name.startswith(f"{user_id}/") and blob_name.endswith("/metadata.json"):
                try:
                    metadata: dict = json.loads(content)
                    if metadata.get("audio_hash") == audio_hash:
                        return metadata
                except Exception:
                    continue

        return None

    def get_user_folder_sas_url(
        self,
        user_id: str,
        expiry_hours: int = 1,
    ) -> str:
        """Generate a fake SAS URL for in-memory user folder."""
        return f"https://inmemory.blob.local/{self.container_name}?sas=mock_token&prefix={user_id}/"

    async def archive_old_recordings(
        self,
        user_id: str,
        retention_days: int = 90,
        archive_container: str = "archive",
    ) -> list[str]:
        """Archive old recordings (in-memory: moves to archive dict)."""
        # In-memory implementation just tracks archived paths
        archived: list[str] = []
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)

        keys_to_archive = []
        for blob_name, (content, metadata) in _in_memory_blobs.items():
            if blob_name.startswith(f"{user_id}/"):
                upload_ts_str = metadata.get("upload_timestamp", "")
                if upload_ts_str:
                    try:
                        upload_ts = datetime.fromisoformat(upload_ts_str)
                        if upload_ts < cutoff:
                            keys_to_archive.append(blob_name)
                    except ValueError:
                        pass

        for key in keys_to_archive:
            # Move to "archived" storage (prefix with archive_container)
            archived_key = f"{archive_container}/{key}"
            _in_memory_blobs[archived_key] = _in_memory_blobs.pop(key)
            archived.append(key)

        return archived

    async def delete_old_recordings(
        self,
        user_id: str,
        retention_days: int = 365,
    ) -> list[str]:
        """Delete old recordings from in-memory storage."""
        deleted: list[str] = []
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)

        keys_to_delete = []
        for blob_name, (content, metadata) in _in_memory_blobs.items():
            if blob_name.startswith(f"{user_id}/"):
                upload_ts_str = metadata.get("upload_timestamp", "")
                if upload_ts_str:
                    try:
                        upload_ts = datetime.fromisoformat(upload_ts_str)
                        if upload_ts < cutoff:
                            keys_to_delete.append(blob_name)
                    except ValueError:
                        pass

        for key in keys_to_delete:
            del _in_memory_blobs[key]
            deleted.append(key)

        return deleted

    async def delete_transcription_folder(
        self,
        folder_path: str,
    ) -> tuple[list[str], list[str]]:
        """Delete all blobs in a transcription folder from in-memory storage."""
        deleted_blobs: list[str] = []
        failed_blobs: list[str] = []

        # Ensure folder_path ends with / for proper prefix matching
        prefix = folder_path if folder_path.endswith("/") else f"{folder_path}/"

        # Find keys to delete
        keys_to_delete = []
        for blob_name in list(_in_memory_blobs.keys()):
            if blob_name.startswith(prefix):
                keys_to_delete.append(blob_name)

        # Also check transcription blobs
        for blob_name in list(_in_memory_transcription_blobs.keys()):
            if blob_name.startswith(prefix):
                keys_to_delete.append(blob_name)

        if not keys_to_delete:
            raise BlobNotFoundError(f"No blobs found in folder: {folder_path}")

        # Delete the blobs
        for key in keys_to_delete:
            if key in _in_memory_blobs:
                del _in_memory_blobs[key]
                deleted_blobs.append(key)
            if key in _in_memory_transcription_blobs:
                del _in_memory_transcription_blobs[key]
                deleted_blobs.append(key)

        return deleted_blobs, failed_blobs


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
