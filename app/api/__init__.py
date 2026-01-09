"""API routers for the Captain's Log FastAPI application."""

from api.auth import router as auth_router

__all__ = [
    "auth_router",
]
