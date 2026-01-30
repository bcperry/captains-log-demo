"""Tests for Azure Blob Storage client module."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, UTC

from storage.blob import (
    BlobStorageClient,
    BlobStorageError,
    BlobUploadError,
    InMemoryBlobClient,
    clear_in_memory_blobs,
    get_storage_client,
    AUDIO_CONTENT_TYPES,
)
from config.settings import Settings, AzureCloud


class TestBlobStorageClient:
    """Tests for BlobStorageClient."""

    def test_is_configured_returns_false_when_not_configured(self) -> None:
        """Test is_configured returns False when storage is not configured."""
        settings = Settings(
            azure_storage_account=None,
            azure_storage_connection_string=None,
            _env_file=None,  # type: ignore[call-arg]
        )
        client = BlobStorageClient(settings)
        assert client.is_configured() is False

    def test_is_configured_returns_true_with_account_name(self) -> None:
        """Test is_configured returns True when account name is set."""
        settings = Settings(
            azure_storage_account="mystorageaccount",
            _env_file=None,  # type: ignore[call-arg]
        )
        client = BlobStorageClient(settings)
        assert client.is_configured() is True

    def test_is_configured_returns_true_with_connection_string(self) -> None:
        """Test is_configured returns True when connection string is set."""
        settings = Settings(
            azure_storage_connection_string="DefaultEndpointsProtocol=https;AccountName=test",
            _env_file=None,  # type: ignore[call-arg]
        )
        client = BlobStorageClient(settings)
        assert client.is_configured() is True

    def test_container_name_from_settings(self) -> None:
        """Test container name comes from settings."""
        settings = Settings(
            azure_storage_container="my-custom-container",
            _env_file=None,  # type: ignore[call-arg]
        )
        client = BlobStorageClient(settings)
        assert client.container_name == "my-custom-container"

    def test_account_name_from_settings(self) -> None:
        """Test account name comes from settings."""
        settings = Settings(
            azure_storage_account="mystorageaccount",
            _env_file=None,  # type: ignore[call-arg]
        )
        client = BlobStorageClient(settings)
        assert client.account_name == "mystorageaccount"

    def test_endpoint_for_government_cloud(self) -> None:
        """Test endpoint URL is generated correctly for government cloud."""
        settings = Settings(
            azure_cloud=AzureCloud.GOVERNMENT,
            azure_storage_account="mystorageaccount",
            _env_file=None,  # type: ignore[call-arg]
        )
        client = BlobStorageClient(settings)
        assert client.endpoint == "https://mystorageaccount.blob.core.usgovcloudapi.net"

    def test_endpoint_for_commercial_cloud(self) -> None:
        """Test endpoint URL is generated correctly for commercial cloud."""
        settings = Settings(
            azure_cloud=AzureCloud.COMMERCIAL,
            azure_storage_account="mystorageaccount",
            _env_file=None,  # type: ignore[call-arg]
        )
        client = BlobStorageClient(settings)
        assert client.endpoint == "https://mystorageaccount.blob.core.windows.net"

    def test_endpoint_uses_explicit_setting(self) -> None:
        """Test explicit endpoint setting takes precedence."""
        settings = Settings(
            azure_storage_account="mystorageaccount",
            azure_storage_endpoint="https://custom.endpoint.example.com",
            _env_file=None,  # type: ignore[call-arg]
        )
        client = BlobStorageClient(settings)
        assert client.endpoint == "https://custom.endpoint.example.com"

    def test_generate_blob_name_includes_user_id(self) -> None:
        """Test generated blob name includes user ID prefix."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        client = BlobStorageClient(settings)
        blob_name = client._generate_blob_name("test.wav", "wav", "user123")
        assert blob_name.startswith("user123/")
        assert blob_name.endswith(".wav")

    def test_generate_blob_name_uses_correct_extension(self) -> None:
        """Test generated blob name has correct file extension."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        client = BlobStorageClient(settings)
        
        for ext in ["wav", "mp3", "m4a"]:
            blob_name = client._generate_blob_name(f"test.{ext}", ext, "user123")
            assert blob_name.endswith(f".{ext}")

    def test_extract_blob_name_from_url(self) -> None:
        """Test extracting blob name from full URL."""
        settings = Settings(
            azure_storage_container="audio-uploads",
            _env_file=None,  # type: ignore[call-arg]
        )
        client = BlobStorageClient(settings)
        
        url = "https://storage.blob.core.windows.net/audio-uploads/user123/file.wav"
        blob_name = client.extract_blob_name_from_url(url)
        assert blob_name == "user123/file.wav"

    def test_extract_blob_name_from_url_with_sas(self) -> None:
        """Test extracting blob name from URL with SAS token."""
        settings = Settings(
            azure_storage_container="audio-uploads",
            _env_file=None,  # type: ignore[call-arg]
        )
        client = BlobStorageClient(settings)
        
        url = "https://storage.blob.core.windows.net/audio-uploads/user123/file.wav?sv=2021-06-08&sas=token"
        blob_name = client.extract_blob_name_from_url(url)
        assert blob_name == "user123/file.wav"

    def test_extract_blob_name_from_url_with_spaces(self) -> None:
        """Test extracting blob name from URL-encoded URL with spaces."""
        settings = Settings(
            azure_storage_container="audio-uploads",
            _env_file=None,  # type: ignore[call-arg]
        )
        client = BlobStorageClient(settings)

        # URL-encoded spaces (%20)
        url = "https://storage.blob.core.windows.net/audio-uploads/user123/Captains%20Log%20Meeting_Jan%2015%202026/audio/file.wav"
        blob_name = client.extract_blob_name_from_url(url)
        assert blob_name == "user123/Captains Log Meeting_Jan 15 2026/audio/file.wav"

    def test_extract_blob_name_from_url_with_special_chars(self) -> None:
        """Test extracting blob name from URL with special characters."""
        settings = Settings(
            azure_storage_container="audio-uploads",
            _env_file=None,  # type: ignore[call-arg]
        )
        client = BlobStorageClient(settings)

        # URL-encoded special characters
        url = "https://storage.blob.core.windows.net/audio-uploads/user123/file%23name%40test/audio/audio.wav?sas=token"
        blob_name = client.extract_blob_name_from_url(url)
        assert blob_name == "user123/file#name@test/audio/audio.wav"


class TestInMemoryBlobClient:
    """Tests for InMemoryBlobClient."""

    def setup_method(self) -> None:
        """Clear in-memory storage before each test."""
        clear_in_memory_blobs()

    def teardown_method(self) -> None:
        """Clear in-memory storage after each test."""
        clear_in_memory_blobs()

    def test_is_configured_always_returns_true(self) -> None:
        """Test in-memory client is always configured."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        client = InMemoryBlobClient(settings)
        assert client.is_configured() is True

    @pytest.mark.asyncio
    async def test_upload_audio_file_returns_url(self) -> None:
        """Test uploading file returns a URL."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        client = InMemoryBlobClient(settings)
        
        content = b"test audio content"
        url = await client.upload_audio_file(
            content=content,
            audio_format="wav",
            user_id="user123",
            original_filename="test.wav",
        )
        
        assert url.startswith("https://inmemory.blob.local/")
        assert "audio-uploads" in url
        assert ".wav" in url

    @pytest.mark.asyncio
    async def test_upload_audio_file_with_metadata(self) -> None:
        """Test uploading file with custom metadata."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        client = InMemoryBlobClient(settings)
        
        content = b"test audio content"
        url = await client.upload_audio_file(
            content=content,
            audio_format="mp3",
            user_id="user456",
            original_filename="recording.mp3",
            metadata={"custom_key": "custom_value"},
        )
        
        assert url is not None
        assert ".mp3" in url

    def test_get_blob_sas_url_returns_mock_url(self) -> None:
        """Test SAS URL generation returns mock URL."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        client = InMemoryBlobClient(settings)
        
        sas_url = client.get_blob_sas_url("user123/file.wav")
        
        assert "?sas=mock_token" in sas_url
        assert "audio-uploads" in sas_url


class TestGetStorageClient:
    """Tests for get_storage_client factory function."""

    def test_returns_in_memory_client_when_not_configured(self) -> None:
        """Test factory returns InMemoryBlobClient when storage is not configured."""
        # Clear any cached client
        get_storage_client.cache_clear()
        
        with patch('storage.blob.get_settings') as mock_settings:
            mock_settings.return_value = Settings(
                azure_storage_account=None,
                azure_storage_connection_string=None,
                _env_file=None,  # type: ignore[call-arg]
            )
            client = get_storage_client()
            assert isinstance(client, InMemoryBlobClient)
        
        # Clear cache after test
        get_storage_client.cache_clear()

    def test_returns_real_client_when_configured(self) -> None:
        """Test factory returns BlobStorageClient when storage is configured."""
        # Clear any cached client
        get_storage_client.cache_clear()
        
        with patch('storage.blob.get_settings') as mock_settings:
            mock_settings.return_value = Settings(
                azure_storage_account="mystorageaccount",
                _env_file=None,  # type: ignore[call-arg]
            )
            client = get_storage_client()
            # Should be BlobStorageClient, not InMemoryBlobClient
            assert isinstance(client, BlobStorageClient)
            # But not specifically InMemoryBlobClient
            assert type(client).__name__ == "BlobStorageClient"
        
        # Clear cache after test
        get_storage_client.cache_clear()


class TestAudioContentTypes:
    """Tests for audio content type mappings."""

    def test_wav_content_type(self) -> None:
        """Test WAV content type mapping."""
        assert AUDIO_CONTENT_TYPES["wav"] == "audio/wav"

    def test_mp3_content_type(self) -> None:
        """Test MP3 content type mapping."""
        assert AUDIO_CONTENT_TYPES["mp3"] == "audio/mpeg"

    def test_m4a_content_type(self) -> None:
        """Test M4A content type mapping."""
        assert AUDIO_CONTENT_TYPES["m4a"] == "audio/mp4"


class TestBlobStorageClientWithMockedAzure:
    """Tests for BlobStorageClient with mocked Azure SDK."""

    @pytest.mark.asyncio
    async def test_upload_audio_file_with_connection_string(self) -> None:
        """Test upload with connection string authentication."""
        settings = Settings(
            azure_storage_connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=testkey==;EndpointSuffix=core.windows.net",
            azure_storage_container="test-container",
            _env_file=None,  # type: ignore[call-arg]
        )
        client = BlobStorageClient(settings)
        
        # Mock the blob service client
        with patch.object(client, '_get_service_client') as mock_get_client:
            mock_blob_client = MagicMock()
            mock_blob_client.url = "https://test.blob.core.windows.net/test-container/user123/file.wav"
            mock_blob_client.upload_blob = MagicMock()
            
            mock_container_client = MagicMock()
            mock_container_client.get_blob_client.return_value = mock_blob_client
            
            mock_service_client = MagicMock()
            mock_service_client.get_container_client.return_value = mock_container_client
            mock_get_client.return_value = mock_service_client
            
            content = b"test audio content"
            url = await client.upload_audio_file(
                content=content,
                audio_format="wav",
                user_id="user123",
            )
            
            assert url == "https://test.blob.core.windows.net/test-container/user123/file.wav"
            mock_blob_client.upload_blob.assert_called_once()

    @pytest.mark.asyncio
    async def test_upload_audio_file_raises_on_error(self) -> None:
        """Test upload raises BlobUploadError on Azure error."""
        from azure.core.exceptions import AzureError
        
        settings = Settings(
            azure_storage_account="mystorageaccount",
            _env_file=None,  # type: ignore[call-arg]
        )
        client = BlobStorageClient(settings)
        
        with patch.object(client, '_get_service_client') as mock_get_client:
            mock_get_client.side_effect = AzureError("Connection failed")
            
            with pytest.raises(BlobUploadError) as exc_info:
                await client.upload_audio_file(
                    content=b"test",
                    audio_format="wav",
                    user_id="user123",
                )
            
            assert "Failed to upload audio file" in str(exc_info.value)

    def test_get_service_client_raises_when_not_configured(self) -> None:
        """Test _get_service_client raises error when not configured."""
        settings = Settings(
            azure_storage_account=None,
            azure_storage_connection_string=None,
            _env_file=None,  # type: ignore[call-arg]
        )
        client = BlobStorageClient(settings)
        
        with pytest.raises(BlobStorageError) as exc_info:
            client._get_service_client()
        
        assert "not configured" in str(exc_info.value)


class TestBlobStorageClientSASGeneration:
    """Tests for SAS URL generation."""

    def test_sas_url_with_connection_string(self) -> None:
        """Test SAS URL generation with connection string."""
        settings = Settings(
            azure_storage_connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=dGVzdGtleQ==;EndpointSuffix=core.windows.net",
            azure_storage_container="test-container",
            _env_file=None,  # type: ignore[call-arg]
        )
        client = BlobStorageClient(settings)
        
        # This should work with connection string
        sas_url = client.get_blob_sas_url("user123/file.wav", expiry_hours=1)
        
        assert "test-container" in sas_url
        assert "user123/file.wav" in sas_url
        # SAS token should be present (starts with various query params)
        assert "?" in sas_url

    def test_sas_url_raises_when_not_configured(self) -> None:
        """Test SAS URL generation raises when not configured."""
        settings = Settings(
            azure_storage_account=None,
            azure_storage_connection_string=None,
            _env_file=None,  # type: ignore[call-arg]
        )
        client = BlobStorageClient(settings)
        
        with pytest.raises(BlobStorageError) as exc_info:
            client.get_blob_sas_url("user123/file.wav")
        
        assert "not configured" in str(exc_info.value)


class TestSanitizeFilename:
    """Tests for the sanitize_filename function."""

    def test_sanitize_removes_special_characters(self) -> None:
        """Test sanitize_filename removes special characters."""
        from storage.blob import sanitize_filename

        assert sanitize_filename("test@file#name!.wav") == "testfilename"
        assert sanitize_filename("file with spaces.mp3") == "file_with_spaces"

    def test_sanitize_handles_empty_filename(self) -> None:
        """Test sanitize_filename handles empty or None."""
        from storage.blob import sanitize_filename

        assert sanitize_filename("") == "unnamed"
        assert sanitize_filename("   ") == "unnamed"

    def test_sanitize_limits_length(self) -> None:
        """Test sanitize_filename limits to 100 characters."""
        from storage.blob import sanitize_filename

        long_name = "a" * 150
        result = sanitize_filename(long_name)
        assert len(result) <= 100

    def test_sanitize_preserves_valid_characters(self) -> None:
        """Test sanitize_filename preserves valid characters."""
        from storage.blob import sanitize_filename

        # Extension is stripped, only the base name is sanitized
        assert sanitize_filename("valid-file_name.test") == "valid-file_name"
        assert sanitize_filename("Meeting2024-01-15") == "Meeting2024-01-15"

    def test_sanitize_strips_extension(self) -> None:
        """Test sanitize_filename removes file extension."""
        from storage.blob import sanitize_filename

        # The function removes the extension during sanitization
        assert "wav" not in sanitize_filename("test.wav") or sanitize_filename("test.wav") == "test"

    def test_sanitize_handles_captains_log_filename(self) -> None:
        """Test sanitize_filename handles the bug report filename with spaces."""
        from storage.blob import sanitize_filename

        # This is the exact filename from the bug report
        result = sanitize_filename("Captains Log Meeting_Jan 15 2026.mp4")
        # Spaces should be replaced with underscores
        assert result == "Captains_Log_Meeting_Jan_15_2026"


class TestUserPathMethods:
    """Tests for hierarchical user path storage methods."""

    def setup_method(self) -> None:
        """Clear in-memory storage before each test."""
        clear_in_memory_blobs()

    def teardown_method(self) -> None:
        """Clear in-memory storage after each test."""
        clear_in_memory_blobs()

    def test_generate_user_path_folder(self) -> None:
        """Test folder path generation for user-scoped storage."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        client = BlobStorageClient(settings)

        ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        folder = client._generate_user_path_folder("user123", "my-audio.wav", ts)

        assert folder.startswith("user123/")
        assert "my-audio_20240115_103000" in folder

    def test_generate_user_path_folder_sanitizes_filename(self) -> None:
        """Test folder path sanitizes special characters in filename."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        client = BlobStorageClient(settings)

        ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        folder = client._generate_user_path_folder("user123", "file with spaces!@#.wav", ts)

        assert folder.startswith("user123/")
        assert "file_with_spaces" in folder
        assert " " not in folder
        assert "@" not in folder

    @pytest.mark.asyncio
    async def test_in_memory_upload_audio_with_user_path(self) -> None:
        """Test in-memory client uploads with user path structure."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        client = InMemoryBlobClient(settings)

        content = b"test audio content"
        ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)

        url, folder_path = await client.upload_audio_with_user_path(
            content=content,
            audio_format="wav",
            user_id="user123",
            original_filename="meeting.wav",
            timestamp=ts,
        )

        assert "user123" in url
        assert "meeting_20240115_103000" in folder_path
        assert "/audio/" in url
        assert url.endswith(".wav")

    @pytest.mark.asyncio
    async def test_in_memory_upload_transcription_with_user_path(self) -> None:
        """Test in-memory client uploads transcription to same folder."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        client = InMemoryBlobClient(settings)

        # First upload audio to get folder path
        ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        _, folder_path = await client.upload_audio_with_user_path(
            content=b"audio",
            audio_format="wav",
            user_id="user123",
            original_filename="meeting.wav",
            timestamp=ts,
        )

        # Upload transcription to same folder
        transcript_url = await client.upload_transcription_with_user_path(
            user_id="user123",
            folder_path=folder_path,
            content_json='{"text": "Hello world"}',
        )

        assert folder_path in transcript_url
        assert transcript_url.endswith("/transcript.json")

    @pytest.mark.asyncio
    async def test_in_memory_download_from_user_path(self) -> None:
        """Test downloading files from user-scoped paths."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        client = InMemoryBlobClient(settings)

        # Upload files
        ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)
        url, folder_path = await client.upload_audio_with_user_path(
            content=b"audio content here",
            audio_format="wav",
            user_id="user456",
            original_filename="recording.wav",
            timestamp=ts,
        )

        transcript_json = '{"text": "Test transcript"}'
        await client.upload_transcription_with_user_path(
            user_id="user456",
            folder_path=folder_path,
            content_json=transcript_json,
        )

        # Download audio
        audio_content = await client.download_audio_from_user_path(
            folder_path=folder_path,
            filename="recording.wav",
        )
        assert audio_content == b"audio content here"

        # Download transcription
        downloaded_json = await client.download_transcription_from_user_path(
            folder_path=folder_path,
        )
        assert downloaded_json == transcript_json

    @pytest.mark.asyncio
    async def test_in_memory_download_not_found(self) -> None:
        """Test download raises BlobNotFoundError for missing files."""
        from storage.blob import BlobNotFoundError

        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        client = InMemoryBlobClient(settings)

        with pytest.raises(BlobNotFoundError):
            await client.download_audio_from_user_path(
                folder_path="nonexistent/path",
                filename="missing.wav",
            )

        with pytest.raises(BlobNotFoundError):
            await client.download_transcription_from_user_path(
                folder_path="nonexistent/path",
            )

    def test_get_user_folder_sas_url(self) -> None:
        """Test user folder SAS URL generation (in-memory)."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        client = InMemoryBlobClient(settings)

        sas_url = client.get_user_folder_sas_url("user789", expiry_hours=2)

        assert "user789" in sas_url
        assert "sas=mock_token" in sas_url

    @pytest.mark.asyncio
    async def test_user_isolation_in_folder_structure(self) -> None:
        """Test that different users have separate folder hierarchies."""
        settings = Settings(_env_file=None)  # type: ignore[call-arg]
        client = InMemoryBlobClient(settings)

        ts = datetime(2024, 1, 15, 10, 30, 0, tzinfo=UTC)

        # User 1 uploads
        url1, folder1 = await client.upload_audio_with_user_path(
            content=b"user1 audio",
            audio_format="wav",
            user_id="user-guid-1",
            original_filename="shared_name.wav",
            timestamp=ts,
        )

        # User 2 uploads same filename
        url2, folder2 = await client.upload_audio_with_user_path(
            content=b"user2 audio",
            audio_format="wav",
            user_id="user-guid-2",
            original_filename="shared_name.wav",
            timestamp=ts,
        )

        # Verify different paths
        assert "user-guid-1" in folder1
        assert "user-guid-2" in folder2
        assert folder1 != folder2

        # Verify different URLs
        assert url1 != url2
        assert "user-guid-1" in url1
        assert "user-guid-2" in url2
