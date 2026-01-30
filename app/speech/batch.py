"""Azure Speech Services Batch Transcription client.

This module provides a client for Azure Speech Services Batch Transcription API with:
- Support for Azure Commercial and Government cloud endpoints
- Submission of batch transcription jobs with audio files from Blob Storage
- Status polling for long-running transcription jobs
- Result retrieval with speaker diarization support
- Retry logic with exponential backoff for transient network errors
- DNS validation and clear error messaging

Reference: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/batch-transcription
"""

import asyncio
import logging
import socket
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Optional
from urllib.parse import urlparse

import httpx

from config.settings import AzureCloud, Settings, get_settings

logger = logging.getLogger(__name__)


class BatchTranscriptionError(Exception):
    """Base exception for batch transcription errors."""

    pass


class BatchTranscriptionDNSError(BatchTranscriptionError):
    """Raised when DNS resolution fails for the batch transcription endpoint."""

    pass


class BatchTranscriptionNetworkError(BatchTranscriptionError):
    """Raised when network connectivity fails (transient error)."""

    pass


class BatchTranscriptionConfigError(BatchTranscriptionError):
    """Raised when configuration is invalid or missing."""

    pass


class BatchTranscriptionJobNotFoundError(BatchTranscriptionError):
    """Raised when a transcription job is not found."""

    pass


class BatchTranscriptionFailedError(BatchTranscriptionError):
    """Raised when a transcription job fails."""

    pass


class TranscriptionStatus(str, Enum):
    """Status of a batch transcription job."""

    NOT_STARTED = "NotStarted"
    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"


@dataclass
class BatchTranscriptionJob:
    """Information about a batch transcription job."""

    id: str
    status: TranscriptionStatus
    display_name: str
    created_date_time: datetime
    last_action_date_time: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class TranscriptionSegment:
    """A segment of transcribed speech with speaker info."""

    speaker_id: str
    text: str
    start_time_ms: int
    end_time_ms: int
    confidence: float = 0.0


@dataclass
class BatchTranscriptionResult:
    """Result of a completed batch transcription."""

    job_id: str
    segments: list[TranscriptionSegment]
    full_text: str
    duration_ms: int
    speaker_count: int
    language: str


class BatchTranscriptionConfig:
    """Configuration for batch transcription API."""

    # Retry configuration
    MAX_RETRIES = 3
    INITIAL_RETRY_DELAY = 1.0  # seconds
    MAX_RETRY_DELAY = 10.0  # seconds
    RETRY_BACKOFF_MULTIPLIER = 2.0

    def __init__(
        self,
        subscription_key: str,
        region: str,
        cloud: AzureCloud = AzureCloud.GOVERNMENT,
    ) -> None:
        """Initialize batch transcription configuration.

        Args:
            subscription_key: Azure Speech Services subscription key
            region: Azure Speech Services region
            cloud: Azure cloud environment

        Raises:
            BatchTranscriptionConfigError: If region is empty or invalid
        """
        if not region or not region.strip():
            raise BatchTranscriptionConfigError(
                "Azure Speech region is empty. Set AZURE_SPEECH_REGION environment variable "
                "(e.g., 'usgovvirginia', 'eastus2')."
            )

        self.subscription_key = subscription_key
        self.region = region.strip().lower()
        self.cloud = cloud

        # Log the configuration for debugging
        logger.debug(
            f"BatchTranscriptionConfig initialized: region={self.region}, "
            f"batch_region={self.batch_region}, "
            f"cloud={self.cloud.value}, base_url={self.base_url}"
        )

    # Azure Government batch transcription API is only available in usgovvirginia
    # https://learn.microsoft.com/en-us/azure/ai-services/speech-service/batch-transcription
    GOVERNMENT_BATCH_REGION = "usgovvirginia"

    @property
    def batch_region(self) -> str:
        """Get the region for batch transcription API.

        For Azure Government, batch transcription is only available in usgovvirginia,
        regardless of where the Speech resource is deployed.
        """
        if self.cloud == AzureCloud.GOVERNMENT:
            return self.GOVERNMENT_BATCH_REGION
        return self.region

    @property
    def base_url(self) -> str:
        """Get the batch transcription API base URL."""
        if self.cloud == AzureCloud.GOVERNMENT:
            # Azure Government uses .microsoft.us domain for cognitive services API
            return f"https://{self.batch_region}.api.cognitive.microsoft.us/speechtotext/v3.1"
        else:
            return f"https://{self.batch_region}.api.cognitive.microsoft.com/speechtotext/v3.1"

    @property
    def hostname(self) -> str:
        """Extract the hostname from the base URL for DNS validation."""
        parsed = urlparse(self.base_url)
        return parsed.netloc

    def validate_dns(self) -> bool:
        """Validate that the hostname can be resolved via DNS.

        Returns:
            True if DNS resolution succeeds

        Raises:
            BatchTranscriptionDNSError: If DNS resolution fails
        """
        hostname = self.hostname
        logger.debug(f"Validating DNS for hostname: {hostname}")

        try:
            socket.gethostbyname(hostname)
            logger.debug(f"DNS resolution succeeded for {hostname}")
            return True
        except socket.gaierror as e:
            error_msg = (
                f"DNS resolution failed for {hostname}. "
                f"Error: {e}. "
                f"Possible causes: "
                f"1) Batch API region='{self.batch_region}' is not available "
                f"(Azure Gov only supports 'usgovvirginia' for batch transcription), "
                f"2) AZURE_CLOUD='{self.cloud.value}' may not match your Speech Services deployment, "
                f"3) Network/DNS issues in your environment. "
                f"Full URL: {self.base_url}"
            )
            logger.error(error_msg)
            raise BatchTranscriptionDNSError(error_msg) from e

    @classmethod
    def from_settings(cls, settings: Optional[Settings] = None) -> "BatchTranscriptionConfig":
        """Create configuration from application settings.

        Args:
            settings: Application settings. If None, uses default settings.

        Returns:
            BatchTranscriptionConfig instance

        Raises:
            BatchTranscriptionConfigError: If required settings are missing
        """
        settings = settings or get_settings()

        if not settings.azure_speech_key:
            raise BatchTranscriptionConfigError(
                "Azure Speech key is required. Set AZURE_SPEECH_KEY environment variable."
            )
        if not settings.azure_speech_region:
            raise BatchTranscriptionConfigError(
                "Azure Speech region is required. Set AZURE_SPEECH_REGION environment variable."
            )

        return cls(
            subscription_key=settings.azure_speech_key,
            region=settings.azure_speech_region,
            cloud=settings.azure_cloud,
        )


class BatchTranscriptionClient:
    """Client for Azure Speech Services Batch Transcription API.

    Provides methods to submit, poll, and retrieve batch transcription jobs.
    Includes retry logic with exponential backoff for transient network errors.
    """

    def __init__(self, config: BatchTranscriptionConfig) -> None:
        """Initialize the batch transcription client.

        Args:
            config: Batch transcription configuration
        """
        self._config = config
        self._http_client: Optional[httpx.AsyncClient] = None
        self._dns_validated = False

    @property
    def config(self) -> BatchTranscriptionConfig:
        """Get the configuration."""
        return self._config

    def _get_headers(self) -> dict[str, str]:
        """Get HTTP headers for API requests."""
        return {
            "Ocp-Apim-Subscription-Key": self._config.subscription_key,
            "Content-Type": "application/json",
        }

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=30.0)
        return self._http_client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    def _validate_dns_once(self) -> None:
        """Validate DNS resolution once per client lifetime.

        Raises:
            BatchTranscriptionDNSError: If DNS resolution fails
        """
        if not self._dns_validated:
            self._config.validate_dns()
            self._dns_validated = True

    async def _execute_with_retry(
        self,
        operation_name: str,
        request_func: "Callable[[], Coroutine[Any, Any, httpx.Response]]",
    ) -> httpx.Response:
        """Execute an HTTP request with retry logic for transient errors.

        Args:
            operation_name: Name of the operation for logging
            request_func: Callable that returns a coroutine performing the HTTP request

        Returns:
            HTTP response

        Raises:
            BatchTranscriptionDNSError: If DNS resolution fails
            BatchTranscriptionNetworkError: If network error persists after retries
        """
        last_exception: Optional[Exception] = None
        delay = BatchTranscriptionConfig.INITIAL_RETRY_DELAY

        for attempt in range(BatchTranscriptionConfig.MAX_RETRIES):
            try:
                # Execute the request - callable returns a coroutine
                return await request_func()

            except httpx.ConnectError as e:
                last_exception = e
                error_str = str(e)

                # Check if this is a DNS error
                if "Name or service not known" in error_str or "getaddrinfo" in error_str:
                    # DNS error - validate and provide clear message
                    try:
                        self._config.validate_dns()
                    except BatchTranscriptionDNSError:
                        raise  # Re-raise with detailed message

                    # DNS passes but still failing - network issue
                    logger.warning(
                        f"{operation_name} failed with DNS-like error but DNS resolves. "
                        f"Attempt {attempt + 1}/{BatchTranscriptionConfig.MAX_RETRIES}. "
                        f"URL: {self._config.base_url}. Error: {e}"
                    )
                else:
                    # Other connection error - retry
                    logger.warning(
                        f"{operation_name} connection error. "
                        f"Attempt {attempt + 1}/{BatchTranscriptionConfig.MAX_RETRIES}. "
                        f"Retrying in {delay:.1f}s. Error: {e}"
                    )

                if attempt < BatchTranscriptionConfig.MAX_RETRIES - 1:
                    await asyncio.sleep(delay)
                    delay = min(
                        delay * BatchTranscriptionConfig.RETRY_BACKOFF_MULTIPLIER,
                        BatchTranscriptionConfig.MAX_RETRY_DELAY,
                    )

            except httpx.TimeoutException as e:
                last_exception = e
                logger.warning(
                    f"{operation_name} timeout. "
                    f"Attempt {attempt + 1}/{BatchTranscriptionConfig.MAX_RETRIES}. "
                    f"Retrying in {delay:.1f}s. Error: {e}"
                )

                if attempt < BatchTranscriptionConfig.MAX_RETRIES - 1:
                    await asyncio.sleep(delay)
                    delay = min(
                        delay * BatchTranscriptionConfig.RETRY_BACKOFF_MULTIPLIER,
                        BatchTranscriptionConfig.MAX_RETRY_DELAY,
                    )

        # All retries exhausted
        error_msg = (
            f"{operation_name} failed after {BatchTranscriptionConfig.MAX_RETRIES} attempts. "
            f"URL: {self._config.base_url}. Last error: {last_exception}"
        )
        logger.error(error_msg)
        raise BatchTranscriptionNetworkError(error_msg) from last_exception

    async def create_transcription_job(
        self,
        content_urls: list[str],
        display_name: str,
        locale: str = "en-US",
        enable_diarization: bool = True,
        enable_word_level_timestamps: bool = True,
        max_speaker_count: int = 5,
    ) -> str:
        """Create a batch transcription job.

        Args:
            content_urls: List of SAS URLs to audio files in Blob Storage
            display_name: Display name for the transcription job
            locale: Language code for transcription
            enable_diarization: Enable speaker diarization
            enable_word_level_timestamps: Enable word-level timestamps
            max_speaker_count: Maximum number of speakers for diarization

        Returns:
            Job ID for the created transcription

        Raises:
            BatchTranscriptionDNSError: If DNS resolution fails
            BatchTranscriptionNetworkError: If network error persists after retries
            BatchTranscriptionError: If job creation fails
        """
        # Validate DNS on first request
        self._validate_dns_once()

        client = await self._get_client()
        url = f"{self._config.base_url}/transcriptions"

        logger.info(f"Creating batch transcription job. URL: {url}")

        # Build transcription request body
        properties: dict[str, object] = {
            "wordLevelTimestampsEnabled": enable_word_level_timestamps,
            "diarizationEnabled": enable_diarization,
        }

        if enable_diarization:
            properties["diarization"] = {
                "speakers": {
                    "minCount": 1,
                    "maxCount": max_speaker_count,
                }
            }

        body: dict[str, object] = {
            "contentUrls": content_urls,
            "displayName": display_name,
            "locale": locale,
            "properties": properties,
        }

        try:
            response = await self._execute_with_retry(
                "Create transcription job",
                lambda: client.post(url, headers=self._get_headers(), json=body),
            )

            if response.status_code == 201:
                data = response.json()
                # Extract job ID from the self link or id field
                self_url = str(data.get("self", ""))
                job_id = self_url.split("/")[-1] if self_url else str(data.get("id", ""))
                logger.info(f"Created batch transcription job: {job_id}")
                return job_id
            else:
                error_detail = response.text
                logger.error(f"Failed to create transcription job: {response.status_code} - {error_detail}")
                raise BatchTranscriptionError(
                    f"Failed to create transcription job: {response.status_code} - {error_detail}"
                )

        except (BatchTranscriptionDNSError, BatchTranscriptionNetworkError):
            raise  # Re-raise our custom exceptions
        except httpx.RequestError as e:
            error_str = str(e)
            # Check for DNS-like errors that weren't caught by retry logic
            if "Name or service not known" in error_str or "getaddrinfo" in error_str:
                logger.error(f"DNS error creating transcription job. URL: {url}. Error: {e}")
                raise BatchTranscriptionDNSError(
                    f"DNS resolution failed for batch transcription endpoint. "
                    f"URL: {url}. Error: {e}. "
                    f"Check AZURE_SPEECH_REGION and AZURE_CLOUD settings."
                ) from e
            logger.error(f"Request error creating transcription job: {e}")
            raise BatchTranscriptionError(f"Request error: {e}") from e

    async def get_transcription_status(self, job_id: str) -> BatchTranscriptionJob:
        """Get the status of a batch transcription job.

        Args:
            job_id: ID of the transcription job

        Returns:
            BatchTranscriptionJob with current status

        Raises:
            BatchTranscriptionJobNotFoundError: If job is not found
            BatchTranscriptionDNSError: If DNS resolution fails
            BatchTranscriptionNetworkError: If network error persists after retries
            BatchTranscriptionError: If status check fails
        """
        # Validate DNS on first request
        self._validate_dns_once()

        client = await self._get_client()
        url = f"{self._config.base_url}/transcriptions/{job_id}"

        try:
            response = await self._execute_with_retry(
                "Get transcription status",
                lambda: client.get(url, headers=self._get_headers()),
            )

            if response.status_code == 200:
                data = response.json()
                return self._parse_job_status(data, job_id)
            elif response.status_code == 404:
                raise BatchTranscriptionJobNotFoundError(f"Transcription job not found: {job_id}")
            else:
                error_detail = response.text
                raise BatchTranscriptionError(
                    f"Failed to get transcription status: {response.status_code} - {error_detail}"
                )

        except (BatchTranscriptionDNSError, BatchTranscriptionNetworkError, BatchTranscriptionJobNotFoundError):
            raise  # Re-raise our custom exceptions
        except httpx.RequestError as e:
            logger.error(f"Request error getting transcription status: {e}")
            raise BatchTranscriptionError(f"Request error: {e}") from e

    def _parse_job_status(self, data: dict, job_id: str) -> BatchTranscriptionJob:
        """Parse job status from API response."""
        status_str = data.get("status", "NotStarted")
        try:
            status = TranscriptionStatus(status_str)
        except ValueError:
            status = TranscriptionStatus.NOT_STARTED

        # Parse dates
        created_str = data.get("createdDateTime", "")
        created_date = datetime.fromisoformat(created_str.replace("Z", "+00:00")) if created_str else datetime.now()

        last_action_str = data.get("lastActionDateTime")
        last_action_date = None
        if last_action_str:
            last_action_date = datetime.fromisoformat(last_action_str.replace("Z", "+00:00"))

        # Get error message if failed
        error_message = None
        if status == TranscriptionStatus.FAILED:
            error_info = data.get("properties", {}).get("error", {})
            error_message = error_info.get("message", "Unknown error")

        return BatchTranscriptionJob(
            id=job_id,
            status=status,
            display_name=data.get("displayName", ""),
            created_date_time=created_date,
            last_action_date_time=last_action_date,
            error_message=error_message,
        )

    async def get_transcription_result(self, job_id: str) -> BatchTranscriptionResult:
        """Get the result of a completed batch transcription.

        Args:
            job_id: ID of the transcription job

        Returns:
            BatchTranscriptionResult with transcribed segments

        Raises:
            BatchTranscriptionJobNotFoundError: If job is not found
            BatchTranscriptionFailedError: If job failed or not complete
            BatchTranscriptionDNSError: If DNS resolution fails
            BatchTranscriptionNetworkError: If network error persists after retries
            BatchTranscriptionError: If result retrieval fails
        """
        # First check job status
        job = await self.get_transcription_status(job_id)

        if job.status == TranscriptionStatus.FAILED:
            raise BatchTranscriptionFailedError(
                f"Transcription job failed: {job.error_message or 'Unknown error'}"
            )

        if job.status != TranscriptionStatus.SUCCEEDED:
            raise BatchTranscriptionError(
                f"Transcription job not complete. Current status: {job.status.value}"
            )

        # Get transcription files (results)
        client = await self._get_client()
        files_url = f"{self._config.base_url}/transcriptions/{job_id}/files"

        try:
            response = await self._execute_with_retry(
                "Get transcription files",
                lambda: client.get(files_url, headers=self._get_headers()),
            )

            if response.status_code != 200:
                raise BatchTranscriptionError(
                    f"Failed to get transcription files: {response.status_code}"
                )

            files_data = response.json()
            values = files_data.get("values", [])

            # Find the transcription result file
            result_url = None
            for file_info in values:
                if file_info.get("kind") == "Transcription":
                    result_url = file_info.get("links", {}).get("contentUrl")
                    break

            if not result_url:
                raise BatchTranscriptionError("No transcription result file found")

            # Download and parse the result file
            result_response = await self._execute_with_retry(
                "Download transcription result",
                lambda: client.get(result_url),
            )
            if result_response.status_code != 200:
                raise BatchTranscriptionError("Failed to download transcription result")

            result_data = result_response.json()
            return self._parse_transcription_result(result_data, job_id)

        except (BatchTranscriptionDNSError, BatchTranscriptionNetworkError):
            raise  # Re-raise our custom exceptions
        except httpx.RequestError as e:
            logger.error(f"Request error getting transcription result: {e}")
            raise BatchTranscriptionError(f"Request error: {e}") from e

    def _parse_transcription_result(
        self, data: dict, job_id: str
    ) -> BatchTranscriptionResult:
        """Parse the batch transcription result JSON.

        The result format follows the Azure Speech batch transcription output schema.
        """
        segments: list[TranscriptionSegment] = []
        full_texts: list[str] = []
        speakers: set[str] = set()
        total_duration_ms = 0
        language = "en-US"

        # Parse combined recognized phrases with speaker info
        combined_phrases = data.get("combinedRecognizedPhrases", [])
        for phrase in combined_phrases:
            lexical = phrase.get("lexical", "")
            if lexical:
                full_texts.append(lexical)

        # Parse recognized phrases for segments with diarization
        recognized_phrases = data.get("recognizedPhrases", [])
        for phrase in recognized_phrases:
            speaker = phrase.get("speaker", 1)
            speaker_id = f"Speaker_{speaker}"
            speakers.add(speaker_id)

            # Get timing info (in ticks, 100ns units)
            offset_str = phrase.get("offset", "PT0S")
            duration_str = phrase.get("duration", "PT0S")

            offset_ms = self._parse_duration_to_ms(offset_str)
            duration_ms = self._parse_duration_to_ms(duration_str)

            end_time_ms = offset_ms + duration_ms
            if end_time_ms > total_duration_ms:
                total_duration_ms = end_time_ms

            # Get best result from n-best list
            n_best = phrase.get("nBest", [])
            if n_best:
                best = n_best[0]
                text = best.get("display", best.get("lexical", ""))
                confidence = best.get("confidence", 0.0)

                if text:
                    segments.append(
                        TranscriptionSegment(
                            speaker_id=speaker_id,
                            text=text,
                            start_time_ms=offset_ms,
                            end_time_ms=end_time_ms,
                            confidence=confidence,
                        )
                    )

        # Get language from source info
        source = data.get("source", "")
        if isinstance(source, dict):
            language = source.get("locale", "en-US")

        # Build full text from segments if not available from combined
        if not full_texts and segments:
            full_texts = [seg.text for seg in segments]

        return BatchTranscriptionResult(
            job_id=job_id,
            segments=segments,
            full_text=" ".join(full_texts),
            duration_ms=total_duration_ms,
            speaker_count=len(speakers) if speakers else 1,
            language=language,
        )

    def _parse_duration_to_ms(self, duration_str: str) -> int:
        """Parse ISO 8601 duration string to milliseconds.

        Format: PT#H#M#.###S (e.g., PT1H2M3.456S, PT30.5S)
        """
        if not duration_str or not duration_str.startswith("PT"):
            return 0

        import re

        # Remove PT prefix
        duration_str = duration_str[2:]

        total_seconds = 0.0

        # Parse hours
        hours_match = re.search(r"(\d+)H", duration_str)
        if hours_match:
            total_seconds += int(hours_match.group(1)) * 3600

        # Parse minutes
        minutes_match = re.search(r"(\d+)M", duration_str)
        if minutes_match:
            total_seconds += int(minutes_match.group(1)) * 60

        # Parse seconds (including decimal)
        seconds_match = re.search(r"([\d.]+)S", duration_str)
        if seconds_match:
            total_seconds += float(seconds_match.group(1))

        return int(total_seconds * 1000)

    async def delete_transcription(self, job_id: str) -> None:
        """Delete a batch transcription job.

        Args:
            job_id: ID of the transcription job to delete

        Raises:
            BatchTranscriptionJobNotFoundError: If job is not found
            BatchTranscriptionDNSError: If DNS resolution fails
            BatchTranscriptionNetworkError: If network error persists after retries
            BatchTranscriptionError: If deletion fails
        """
        # Validate DNS on first request
        self._validate_dns_once()

        client = await self._get_client()
        url = f"{self._config.base_url}/transcriptions/{job_id}"

        try:
            response = await self._execute_with_retry(
                "Delete transcription job",
                lambda: client.delete(url, headers=self._get_headers()),
            )

            if response.status_code == 204:
                logger.info(f"Deleted batch transcription job: {job_id}")
            elif response.status_code == 404:
                raise BatchTranscriptionJobNotFoundError(f"Transcription job not found: {job_id}")
            else:
                raise BatchTranscriptionError(
                    f"Failed to delete transcription job: {response.status_code}"
                )

        except (BatchTranscriptionDNSError, BatchTranscriptionNetworkError, BatchTranscriptionJobNotFoundError):
            raise  # Re-raise our custom exceptions
        except httpx.RequestError as e:
            logger.error(f"Request error deleting transcription job: {e}")
            raise BatchTranscriptionError(f"Request error: {e}") from e


# Singleton instance management
_batch_client_instance: Optional[BatchTranscriptionClient] = None


def get_batch_transcription_client(
    settings: Optional[Settings] = None,
) -> Optional[BatchTranscriptionClient]:
    """Get a BatchTranscriptionClient instance.

    Args:
        settings: Optional settings override for testing

    Returns:
        BatchTranscriptionClient if configured, None otherwise
    """
    global _batch_client_instance

    # If settings provided, create a new instance (for testing)
    if settings is not None:
        if not settings.is_speech_configured():
            return None
        try:
            config = BatchTranscriptionConfig.from_settings(settings)
            return BatchTranscriptionClient(config)
        except BatchTranscriptionError:
            return None

    # Use cached singleton for default settings
    if _batch_client_instance is not None:
        return _batch_client_instance

    default_settings = get_settings()
    if not default_settings.is_speech_configured():
        return None

    try:
        config = BatchTranscriptionConfig.from_settings(default_settings)
        _batch_client_instance = BatchTranscriptionClient(config)
        return _batch_client_instance
    except BatchTranscriptionError:
        return None


def clear_batch_client_cache() -> None:
    """Clear the cached BatchTranscriptionClient instance."""
    global _batch_client_instance
    _batch_client_instance = None


# In-memory batch client for development/testing
class InMemoryBatchTranscriptionClient(BatchTranscriptionClient):
    """In-memory implementation for development and testing."""

    def __init__(self) -> None:
        """Initialize in-memory client with dummy config."""
        config = BatchTranscriptionConfig(
            subscription_key="test-key",
            region="test-region",
            cloud=AzureCloud.LOCAL,
        )
        super().__init__(config)
        self._jobs: dict[str, BatchTranscriptionJob] = {}
        self._results: dict[str, BatchTranscriptionResult] = {}
        self._job_counter = 0

    async def create_transcription_job(
        self,
        content_urls: list[str],
        display_name: str,
        locale: str = "en-US",
        enable_diarization: bool = True,
        enable_word_level_timestamps: bool = True,
        max_speaker_count: int = 5,
    ) -> str:
        """Create a mock transcription job."""
        self._job_counter += 1
        job_id = f"mock-job-{self._job_counter}"

        job = BatchTranscriptionJob(
            id=job_id,
            status=TranscriptionStatus.RUNNING,
            display_name=display_name,
            created_date_time=datetime.now(),
        )
        self._jobs[job_id] = job

        # Auto-complete the job with mock result
        segments = [
            TranscriptionSegment(
                speaker_id="Speaker_1",
                text="This is a mock transcription result.",
                start_time_ms=0,
                end_time_ms=3000,
                confidence=0.95,
            )
        ]
        self._results[job_id] = BatchTranscriptionResult(
            job_id=job_id,
            segments=segments,
            full_text="This is a mock transcription result.",
            duration_ms=3000,
            speaker_count=1,
            language=locale,
        )

        # Mark as succeeded
        self._jobs[job_id] = BatchTranscriptionJob(
            id=job_id,
            status=TranscriptionStatus.SUCCEEDED,
            display_name=display_name,
            created_date_time=job.created_date_time,
            last_action_date_time=datetime.now(),
        )

        return job_id

    async def get_transcription_status(self, job_id: str) -> BatchTranscriptionJob:
        """Get mock job status."""
        if job_id not in self._jobs:
            raise BatchTranscriptionJobNotFoundError(f"Job not found: {job_id}")
        return self._jobs[job_id]

    async def get_transcription_result(self, job_id: str) -> BatchTranscriptionResult:
        """Get mock transcription result."""
        if job_id not in self._jobs:
            raise BatchTranscriptionJobNotFoundError(f"Job not found: {job_id}")

        job = self._jobs[job_id]
        if job.status == TranscriptionStatus.FAILED:
            raise BatchTranscriptionFailedError("Mock job failed")
        if job.status != TranscriptionStatus.SUCCEEDED:
            raise BatchTranscriptionError(f"Job not complete: {job.status.value}")

        return self._results[job_id]

    async def delete_transcription(self, job_id: str) -> None:
        """Delete mock transcription job."""
        if job_id not in self._jobs:
            raise BatchTranscriptionJobNotFoundError(f"Job not found: {job_id}")
        del self._jobs[job_id]
        if job_id in self._results:
            del self._results[job_id]

    async def close(self) -> None:
        """No-op for in-memory client."""
        pass
