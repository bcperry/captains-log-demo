"""Captain's Log - FastAPI Application.

This is the main FastAPI application for Captain's Log, an audio transcription API
that uses Azure Speech Services for transcription and Azure Cosmos DB for storage.

Features:
- Audio transcription with Azure Speech Services
- Speaker diarization for multi-speaker audio
- Transcription history storage in Cosmos DB
- Azure Entra ID authentication
- Health check and monitoring endpoints
- React frontend served as static files
"""

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api import auth_router, health_router, transcribe_router, transcriptions_router

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

All endpoints except `/health` and `/ready` require Azure Entra ID authentication.
Include a valid Bearer token in the `Authorization` header:

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


def custom_openapi() -> dict:
    """Generate custom OpenAPI schema with enhanced documentation."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=API_TITLE,
        version=API_VERSION,
        description=API_DESCRIPTION,
        routes=app.routes,
        tags=OPENAPI_TAGS,
    )

    # Add security scheme for Azure Entra ID
    openapi_schema["components"]["securitySchemes"] = {
        "AzureEntraID": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Azure Entra ID Bearer token",
        }
    }

    # Apply security globally to all endpoints
    openapi_schema["security"] = [{"AzureEntraID": []}]

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


# Create FastAPI application
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    openapi_tags=OPENAPI_TAGS,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS configuration for development mode
# In production, the React app is served from the same origin
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Override OpenAPI schema with custom version
app.openapi = custom_openapi  # type: ignore[method-assign]

# Include API routers
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(transcribe_router)
app.include_router(transcriptions_router)

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
