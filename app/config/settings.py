"""Application settings using Pydantic Settings for configuration validation.

This module provides centralized configuration management with support for:
- Environment variables
- .env files
- Multiple Azure cloud environments (Commercial, Government)
"""

from enum import Enum
from functools import lru_cache
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AzureCloud(str, Enum):
    """Azure cloud environment types."""

    COMMERCIAL = "commercial"
    GOVERNMENT = "government"
    LOCAL = "local"


class Settings(BaseSettings):
    """Application configuration settings.

    Settings are loaded from environment variables and .env files.
    Environment variables take precedence over .env file values.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Azure Cloud Environment
    azure_cloud: AzureCloud = Field(
        default=AzureCloud.GOVERNMENT,
        description="Azure cloud environment (commercial, government, local)",
    )

    # Azure Speech Services Configuration
    azure_speech_key: Optional[str] = Field(
        default=None,
        description="Azure Speech Services subscription key",
    )
    azure_speech_region: str = Field(
        default="usgovvirginia",
        description="Azure Speech Services region",
    )
    azure_speech_endpoint: Optional[str] = Field(
        default=None,
        description="Azure Speech Services endpoint URL (optional, auto-generated if not provided)",
    )

    # Azure OpenAI Configuration
    azure_openai_endpoint: Optional[str] = Field(
        default=None,
        description="Azure OpenAI endpoint URL",
    )
    azure_openai_key: Optional[str] = Field(
        default=None,
        description="Azure OpenAI API key",
    )
    azure_openai_deployment: Optional[str] = Field(
        default=None,
        description="Azure OpenAI deployment name",
    )
    azure_openai_api_version: str = Field(
        default="2024-02-15-preview",
        description="Azure OpenAI API version",
    )

    # Azure Cosmos DB Configuration
    azure_cosmos_endpoint: Optional[str] = Field(
        default=None,
        description="Azure Cosmos DB endpoint URL",
    )
    azure_cosmos_key: Optional[str] = Field(
        default=None,
        description="Azure Cosmos DB key",
    )
    azure_cosmos_database: str = Field(
        default="captainslog",
        description="Azure Cosmos DB database name",
    )

    # Azure Entra ID Configuration
    azure_tenant_id: Optional[str] = Field(
        default=None,
        description="Azure Entra ID tenant ID",
    )
    azure_client_id: Optional[str] = Field(
        default=None,
        description="Azure Entra ID application (client) ID",
    )
    azure_client_secret: Optional[str] = Field(
        default=None,
        description="Azure Entra ID client secret (for service-to-service auth)",
    )

    # Application Settings
    app_name: str = Field(
        default="Captain's Log",
        description="Application display name",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )

    @field_validator("azure_cloud", mode="before")
    @classmethod
    def validate_azure_cloud(cls, v: str) -> AzureCloud:
        """Convert string to AzureCloud enum."""
        if isinstance(v, AzureCloud):
            return v
        return AzureCloud(v.lower())

    @field_validator("log_level", mode="before")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper_v = v.upper()
        if upper_v not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return upper_v

    @property
    def speech_endpoint_url(self) -> str:
        """Get the Azure Speech Services endpoint URL based on cloud environment."""
        if self.azure_speech_endpoint:
            return self.azure_speech_endpoint

        if self.azure_cloud == AzureCloud.GOVERNMENT:
            return f"wss://{self.azure_speech_region}.stt.speech.azure.us"
        elif self.azure_cloud == AzureCloud.COMMERCIAL:
            return f"wss://{self.azure_speech_region}.stt.speech.microsoft.com"
        else:
            # Local development - use commercial endpoint by default
            return f"wss://{self.azure_speech_region}.stt.speech.microsoft.com"

    @property
    def entra_authority(self) -> str:
        """Get the Azure Entra ID authority URL based on cloud environment."""
        if self.azure_cloud == AzureCloud.GOVERNMENT:
            return f"https://login.microsoftonline.us/{self.azure_tenant_id}"
        else:
            return f"https://login.microsoftonline.com/{self.azure_tenant_id}"

    @property
    def entra_jwks_uri(self) -> str:
        """Get the Azure Entra ID JWKS URI based on cloud environment."""
        if self.azure_cloud == AzureCloud.GOVERNMENT:
            return f"https://login.microsoftonline.us/{self.azure_tenant_id}/discovery/v2.0/keys"
        else:
            return f"https://login.microsoftonline.com/{self.azure_tenant_id}/discovery/v2.0/keys"

    @property
    def cosmos_endpoint_url(self) -> Optional[str]:
        """Get the Azure Cosmos DB endpoint URL."""
        if self.azure_cosmos_endpoint:
            return self.azure_cosmos_endpoint

        # Cannot auto-generate Cosmos endpoint - must be provided
        return None

    def is_speech_configured(self) -> bool:
        """Check if Azure Speech Services is properly configured."""
        return bool(self.azure_speech_key and self.azure_speech_region)

    def is_openai_configured(self) -> bool:
        """Check if Azure OpenAI is properly configured."""
        return bool(
            self.azure_openai_endpoint
            and self.azure_openai_key
            and self.azure_openai_deployment
        )

    def is_cosmos_configured(self) -> bool:
        """Check if Azure Cosmos DB is properly configured."""
        return bool(self.azure_cosmos_endpoint and self.azure_cosmos_key)

    def is_entra_configured(self) -> bool:
        """Check if Azure Entra ID is properly configured."""
        return bool(self.azure_tenant_id and self.azure_client_id)


@lru_cache
def get_settings() -> Settings:
    """Get cached application settings instance.

    Returns:
        Settings: Application settings loaded from environment.
    """
    return Settings()
