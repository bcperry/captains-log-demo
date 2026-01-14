"""Captain's Log - FastAPI Application.

This is the main FastAPI application for Captain's Log, an audio transcription API
that uses Azure Speech Services for transcription and Azure Cosmos DB for storage.

Features:
- Audio transcription with Azure Speech Services
- Speaker diarization for multi-speaker audio
- Transcription history storage in Cosmos DB
- Azure Entra ID authentication with OAuth2 Swagger UI integration
- Health check and monitoring endpoints
- React frontend served as static files
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, List, Any

from fastapi import FastAPI, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import auth_router, health_router, transcribe_router, transcriptions_router, analyze_router
from auth.azure_auth import get_azure_scheme
from config.settings import get_settings

# Configure logging - set level from LOG_LEVEL env var (default INFO)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# API metadata for OpenAPI documentation
API_TITLE = "Captain's Log API"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
# Captain's Log API

A FastAPI-based audio transcription service using Azure Speech Services.

## Features

- **Audio Transcription**: Upload audio files (WAV, MP3, M4A) and get text transcriptions
- **Speaker Diarization**: Identify different speakers in multi-speaker audio
- **Transcription History**: Store and retrieve past transcriptions
- **Azure Entra ID Auth**: Secure endpoints with Azure Active Directory authentication

## Authentication

Click the **Authorize** button in Swagger UI to log in with your Microsoft account.
All endpoints except `/health` and `/ready` require Azure Entra ID authentication.

For programmatic access, include a valid Bearer token in the `Authorization` header:

```
Authorization: Bearer <your-access-token>
```

## Rate Limits

- Maximum file size: 25 MB
- Supported formats: WAV, MP3, M4A

## Endpoints Overview

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe - always returns 200 |
| `/ready` | GET | Readiness probe - checks dependencies |
| `/auth/me` | GET | Get current user profile |
| `/transcribe` | POST | Transcribe audio file |
| `/transcribe/diarize` | POST | Transcribe with speaker diarization |
| `/transcriptions` | GET | List user's transcriptions |
| `/transcriptions/{id}` | GET | Get specific transcription |
| `/transcriptions/{id}` | DELETE | Delete transcription |
"""

# OpenAPI tags with descriptions
OPENAPI_TAGS = [
    {
        "name": "Health",
        "description": "Health check and monitoring endpoints for container orchestration.",
    },
    {
        "name": "Authentication",
        "description": "User authentication and profile management using Azure Entra ID.",
    },
    {
        "name": "Transcription",
        "description": "Audio transcription endpoints using Azure Speech Services.",
    },
    {
        "name": "Transcription History",
        "description": "Endpoints for managing stored transcription records.",
    },
]


# Get settings for OAuth2 configuration
settings = get_settings()

# Get the Azure authentication scheme (may raise ValueError if not configured)
# We lazily initialize this only if Entra is configured
azure_scheme = None
if settings.is_entra_configured():
    try:
        azure_scheme = get_azure_scheme()
    except ValueError:
        # Auth not configured, Swagger UI won't have OAuth
        pass


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager.

    Loads OpenID configuration on startup for faster first authentication.
    """
    if azure_scheme is not None:
        # Log the OpenID config URL to verify it's correct for Azure Government
        logger.info(f"OpenID config URL: {azure_scheme.openid_config.config_url}")
        await azure_scheme.openid_config.load_config()
        # Log the loaded issuer to verify it's from the correct cloud
        logger.info(f"Loaded issuer: {azure_scheme.openid_config.issuer}")
    yield


# Build Swagger UI OAuth2 configuration
swagger_ui_init_oauth = None
if settings.is_entra_configured() and settings.effective_openapi_client_id:
    swagger_ui_init_oauth = {
        "usePkceWithAuthorizationCodeGrant": True,
        "clientId": settings.effective_openapi_client_id,
        "scopes": settings.api_scope,
    }


def custom_openapi() -> dict:
    """Generate custom OpenAPI schema with enhanced documentation.

    Adds security schemes, contact info, and license to the OpenAPI spec.
    OAuth2 security scheme is added by fastapi-azure-auth when azure_scheme is used.
    """
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    openapi_schema = get_openapi(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        routes=app.routes,
        tags=OPENAPI_TAGS,
    )

    # Add contact and license info
    openapi_schema["info"]["contact"] = {
        "name": "Captain's Log Support",
        "url": "https://github.com/your-org/captains-log-demo",
    }
    openapi_schema["info"]["license"] = {
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Create FastAPI application with OAuth2 support
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    openapi_tags=OPENAPI_TAGS,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_oauth2_redirect_url="/oauth2-redirect",
    swagger_ui_init_oauth=swagger_ui_init_oauth,
    lifespan=lifespan,
)

# CORS configuration for development mode
# In production, the React app is served from the same origin
# Default includes common Vite dev ports (3000 and 5173)
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["Authorization", "Content-Type", "Accept", "Origin", "X-Requested-With"],
)

# Override OpenAPI schema with custom version
app.openapi = custom_openapi  # type: ignore[method-assign]

# Include API routers
# Health router has no authentication
app.include_router(health_router)

# Build security dependencies for authenticated routes
# When azure_scheme is configured, Security(azure_scheme) adds OAuth2 to OpenAPI
auth_dependencies: List[Any] = []
if azure_scheme is not None:
    auth_dependencies = [Security(azure_scheme)]

# Authenticated routers with OAuth2 security
app.include_router(auth_router, dependencies=auth_dependencies)
app.include_router(transcribe_router, dependencies=auth_dependencies)
app.include_router(transcriptions_router, dependencies=auth_dependencies)
app.include_router(analyze_router, dependencies=auth_dependencies)

# Static files and SPA routing
# Check if frontend dist directory exists (production build)
STATIC_DIR = Path(__file__).parent / "static"

if STATIC_DIR.exists():
    # Serve static assets (JS, CSS, images, etc.)
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    # Catch-all route for SPA - must be defined after all API routes
    @app.get("/{full_path:path}")
    async def serve_spa(request: Request, full_path: str) -> FileResponse:
        """Serve React SPA for all non-API routes.

        This enables client-side routing by returning index.html for all paths
        that don't match API endpoints or static assets.
        """
        # Check if requesting a specific file
        file_path = STATIC_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)

        # Return index.html for SPA routing
        return FileResponse(STATIC_DIR / "index.html")
