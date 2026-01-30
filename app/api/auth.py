"""Authentication API endpoints.

This module provides endpoints for user authentication and profile management
using Azure Entra ID with fastapi-azure-auth for OAuth2 Swagger UI integration.
"""

from datetime import UTC, datetime

from fastapi import APIRouter, Depends

from auth import AuthenticatedUser
from auth.dependencies import get_current_user_azure
from models.user import UserPreferences, UserProfileResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.get(
    "/me",
    response_model=UserProfileResponse,
    summary="Get current user profile",
    description="Returns the authenticated user's profile from token claims.",
)
async def get_current_user_profile(
    user: AuthenticatedUser = Depends(get_current_user_azure),
) -> UserProfileResponse:
    """Get the current authenticated user's profile.

    Extracts user info from Azure Entra ID token claims.

    Args:
        user: Authenticated user from Entra ID token

    Returns:
        UserProfileResponse with user info
    """
    return UserProfileResponse(
        id=user.oid,
        email=user.email,
        name=user.name,
        preferred_username=user.preferred_username,
        preferences=UserPreferences(),
        created_at=datetime.now(UTC),  # Not persisted - always current time
        last_login_at=datetime.now(UTC),
    )


@router.patch(
    "/me/preferences",
    response_model=UserProfileResponse,
    summary="Update user preferences",
    description="Update the authenticated user's preferences. Note: preferences are not persisted.",
)
async def update_user_preferences(
    preferences: UserPreferences,
    user: AuthenticatedUser = Depends(get_current_user_azure),
) -> UserProfileResponse:
    """Update the current user's preferences.

    Note: Without Cosmos DB, preferences are not persisted.

    Args:
        preferences: New preference values
        user: Authenticated user from Entra ID token

    Returns:
        UserProfileResponse with updated preferences
    """
    return UserProfileResponse(
        id=user.oid,
        email=user.email,
        name=user.name,
        preferred_username=user.preferred_username,
        preferences=preferences,
        created_at=datetime.now(UTC),
        last_login_at=datetime.now(UTC),
    )
