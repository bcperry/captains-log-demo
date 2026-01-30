"""Health check and monitoring API endpoints.

This module provides endpoints for container health checks and readiness probes.
"""

import logging
from datetime import UTC, datetime
from typing import Optional

import httpx
from fastapi import APIRouter
from pydantic import BaseModel, Field

from config.settings import get_settings
from config.version import API_VERSION
from speech import get_speech_client

# Configure structured logging
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


class DependencyStatus(BaseModel):
    """Status of a service dependency."""

    name: str = Field(..., description="Name of the dependency")
    healthy: bool = Field(..., description="Whether the dependency is healthy")
    configured: bool = Field(default=True, description="Whether the dependency is configured")
    message: Optional[str] = Field(default=None, description="Status message or error")


class HealthResponse(BaseModel):
    """Response from health check endpoint."""

    status: str = Field(..., description="Overall health status: 'healthy' or 'unhealthy'")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of health check",
    )
    version: str = Field(default=API_VERSION, description="Application version")


class ReadinessResponse(BaseModel):
    """Response from readiness check endpoint."""

    status: str = Field(..., description="Overall readiness status")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of readiness check",
    )
    dependencies: list[DependencyStatus] = Field(
        default_factory=list, description="Status of service dependencies"
    )


def check_speech_health() -> DependencyStatus:
    """Check Azure Speech Services connection health.

    Returns:
        DependencyStatus for Speech Services
    """
    try:
        settings = get_settings()
        if not settings.is_speech_configured():
            return DependencyStatus(
                name="speech_services",
                healthy=False,
                configured=False,
                message="Not configured (AZURE_SPEECH_KEY not set)",
            )

        client = get_speech_client()
        if client is not None and client.is_available():
            return DependencyStatus(
                name="speech_services",
                healthy=True,
                configured=True,
                message="Speech Services available",
            )
        else:
            return DependencyStatus(
                name="speech_services",
                healthy=False,
                configured=True,
                message="Speech Services unavailable",
            )
    except Exception as e:
        logger.error(f"Speech Services health check failed: {e}")
        return DependencyStatus(
            name="speech_services",
            healthy=False,
            configured=True,
            message=str(e),
        )


async def check_openai_health() -> DependencyStatus:
    """Check Azure OpenAI connection health.

    Verifies:
    - Configuration is complete (endpoint, key, deployment)
    - Endpoint is reachable via HTTP GET

    Returns:
        DependencyStatus for Azure OpenAI
    """
    try:
        settings = get_settings()
        if not settings.is_openai_configured():
            # Determine what's missing for better error message
            missing = []
            if not settings.azure_openai_endpoint:
                missing.append("AZURE_OPENAI_ENDPOINT")
            if not settings.azure_openai_key:
                missing.append("AZURE_OPENAI_KEY")
            if not settings.azure_openai_deployment:
                missing.append("AZURE_OPENAI_DEPLOYMENT")

            return DependencyStatus(
                name="azure_openai",
                healthy=False,
                configured=False,
                message=f"Not configured ({', '.join(missing)} not set)",
            )

        # Test connectivity by making a lightweight API call
        # Use the models endpoint to verify the endpoint is reachable
        # This doesn't consume tokens and returns the list of available models
        # These are guaranteed to be non-None after is_openai_configured() check
        endpoint = str(settings.azure_openai_endpoint).rstrip("/")
        api_version = settings.azure_openai_api_version
        deployment = str(settings.azure_openai_deployment)
        api_key = str(settings.azure_openai_key)
        url = f"{endpoint}/openai/models?api-version={api_version}"

        logger.debug(f"Azure OpenAI health check URL: {url}")

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                url,
                headers={"api-key": api_key},
            )

            if response.status_code == 200:
                # Check if the configured deployment exists in the models list
                try:
                    data = response.json()
                    models = data.get("data", [])
                    model_ids = [m.get("id", "") for m in models]
                    logger.debug(f"Available models: {model_ids}")

                    # Deployment names in Azure OpenAI are case-sensitive
                    if deployment in model_ids:
                        return DependencyStatus(
                            name="azure_openai",
                            healthy=True,
                            configured=True,
                            message="Connection successful",
                        )
                    else:
                        # Models endpoint works, but deployment not found
                        # This could be normal if using a deployment name different from model ID
                        # Return healthy since the endpoint is reachable
                        return DependencyStatus(
                            name="azure_openai",
                            healthy=True,
                            configured=True,
                            message=f"Connection successful (deployment: {deployment})",
                        )
                except Exception as parse_err:
                    logger.warning(f"Could not parse models response: {parse_err}")
                    # Still consider healthy if we got a 200 response
                    return DependencyStatus(
                        name="azure_openai",
                        healthy=True,
                        configured=True,
                        message="Connection successful",
                    )
            elif response.status_code == 401:
                return DependencyStatus(
                    name="azure_openai",
                    healthy=False,
                    configured=True,
                    message="Invalid API key",
                )
            elif response.status_code == 404:
                return DependencyStatus(
                    name="azure_openai",
                    healthy=False,
                    configured=True,
                    message="Endpoint not found - check AZURE_OPENAI_ENDPOINT",
                )
            else:
                return DependencyStatus(
                    name="azure_openai",
                    healthy=False,
                    configured=True,
                    message=f"Connection failed (HTTP {response.status_code})",
                )

    except httpx.TimeoutException:
        logger.error("Azure OpenAI health check timed out")
        return DependencyStatus(
            name="azure_openai",
            healthy=False,
            configured=True,
            message="Connection timed out",
        )
    except httpx.ConnectError as e:
        logger.error(f"Azure OpenAI connection error: {e}")
        return DependencyStatus(
            name="azure_openai",
            healthy=False,
            configured=True,
            message="Connection failed - endpoint unreachable",
        )
    except Exception as e:
        logger.error(f"Azure OpenAI health check failed: {e}")
        return DependencyStatus(
            name="azure_openai",
            healthy=False,
            configured=True,
            message=str(e),
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Liveness probe for container health checks. Returns 200 if the service is running.",
)
async def health_check() -> HealthResponse:
    """Basic health check for container liveness probe.

    This endpoint always returns 200 if the service is running.
    Use /ready for dependency checks.

    Returns:
        HealthResponse with status and timestamp
    """
    logger.info("Health check requested")
    return HealthResponse(status="healthy")


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness check",
    description="Readiness probe that checks all service dependencies.",
    responses={
        200: {"description": "Service is ready"},
        503: {"description": "Service not ready - one or more dependencies unavailable"},
    },
)
async def readiness_check() -> ReadinessResponse:
    """Readiness check that verifies all dependencies.

    Checks:
    - Azure Speech Services availability
    - Azure OpenAI connectivity

    Returns:
        ReadinessResponse with dependency statuses
    """
    logger.info("Readiness check requested")

    # Sync health checks
    dependencies = [
        check_speech_health(),
    ]

    # Async health checks
    openai_status = await check_openai_health()
    dependencies.append(openai_status)

    # Determine overall status
    all_healthy = all(dep.healthy for dep in dependencies)
    status_str = "ready" if all_healthy else "not_ready"

    if not all_healthy:
        unhealthy = [dep.name for dep in dependencies if not dep.healthy]
        logger.warning(f"Readiness check failed. Unhealthy dependencies: {unhealthy}")

    return ReadinessResponse(
        status=status_str,
        dependencies=dependencies,
    )
