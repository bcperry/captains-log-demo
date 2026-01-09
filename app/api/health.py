"""Health check and monitoring API endpoints.

This module provides endpoints for container health checks and readiness probes.
"""

import logging
from datetime import UTC, datetime
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from config.settings import get_settings
from db import get_cosmos_client
from speech import get_speech_client

# Configure structured logging
logger = logging.getLogger(__name__)

router = APIRouter(tags=["Health"])


class DependencyStatus(BaseModel):
    """Status of a service dependency."""

    name: str = Field(..., description="Name of the dependency")
    healthy: bool = Field(..., description="Whether the dependency is healthy")
    message: Optional[str] = Field(default=None, description="Status message or error")


class HealthResponse(BaseModel):
    """Response from health check endpoint."""

    status: str = Field(..., description="Overall health status: 'healthy' or 'unhealthy'")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp of health check",
    )
    version: str = Field(default="0.6.0", description="Application version")


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


def check_cosmos_health() -> DependencyStatus:
    """Check Cosmos DB connection health.

    Returns:
        DependencyStatus for Cosmos DB
    """
    try:
        settings = get_settings()
        if not settings.is_cosmos_configured():
            return DependencyStatus(
                name="cosmos_db",
                healthy=True,
                message="Using in-memory storage (Cosmos DB not configured)",
            )

        client = get_cosmos_client()
        if client.is_configured():
            return DependencyStatus(
                name="cosmos_db",
                healthy=True,
                message="Cosmos DB configured",
            )
        else:
            return DependencyStatus(
                name="cosmos_db",
                healthy=True,
                message="Using in-memory fallback",
            )
    except Exception as e:
        logger.error(f"Cosmos DB health check failed: {e}")
        return DependencyStatus(
            name="cosmos_db",
            healthy=False,
            message=str(e),
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
                healthy=True,
                message="Speech Services not configured (optional)",
            )

        client = get_speech_client()
        if client is not None and client.is_available():
            return DependencyStatus(
                name="speech_services",
                healthy=True,
                message="Speech Services available",
            )
        else:
            return DependencyStatus(
                name="speech_services",
                healthy=False,
                message="Speech Services unavailable",
            )
    except Exception as e:
        logger.error(f"Speech Services health check failed: {e}")
        return DependencyStatus(
            name="speech_services",
            healthy=False,
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
    - Cosmos DB connectivity
    - Azure Speech Services availability

    Returns:
        ReadinessResponse with dependency statuses
    """
    logger.info("Readiness check requested")

    dependencies = [
        check_cosmos_health(),
        check_speech_health(),
    ]

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
