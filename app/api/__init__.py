"""API routers for the Captain's Log FastAPI application."""

from api.auth import router as auth_router
from api.health import router as health_router
from api.transcribe import router as transcribe_router
from api.transcriptions import router as transcriptions_router

__all__ = [
    "auth_router",
    "health_router",
    "transcribe_router",
    "transcriptions_router",
]
