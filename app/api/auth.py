"""Authentication API endpoints.

This module provides endpoints for user authentication and profile management
using Azure Entra ID with fastapi-azure-auth for OAuth2 Swagger UI integration.
"""

from fastapi import APIRouter, Depends

from auth import AuthenticatedUser
from auth.dependencies import get_current_user_azure
from db import get_cosmos_client
from db.cosmos import CosmosClient
from models.user import UserPreferences, UserProfileCreate, UserProfileResponse

router = APIRouter(prefix="/auth", tags=["Authentication"])


def get_db() -> CosmosClient:
    """FastAPI dependency for getting database client."""
    return get_cosmos_client()


@router.get(
    "/me",
    response_model=UserProfileResponse,
    summary="Get current user profile",
    description="Returns the authenticated user's profile. Creates a new profile on first login.",
)
async def get_current_user_profile(
    user: AuthenticatedUser = Depends(get_current_user_azure),
    db: CosmosClient = Depends(get_db),
) -> UserProfileResponse:
    """Get the current authenticated user's profile.

    Extracts user info from Azure Entra ID token claims.
    Stores user profile in Cosmos DB on first login.
    Returns user preferences and settings.

    Args:
        user: Authenticated user from Entra ID token
        db: Cosmos DB client

    Returns:
        UserProfileResponse with user info and preferences
    """
    # Prepare user data from token claims
    user_data = UserProfileCreate(
        oid=user.oid,
        email=user.email,
        name=user.name,
        preferred_username=user.preferred_username,
        tenant_id=user.tenant_id,
    )

    # Get or create user profile
    profile, _created = await db.get_or_create_user_profile(user_data)

    # Return profile response
    return UserProfileResponse(
        id=profile.id,
        email=profile.email,
        name=profile.name,
        preferred_username=profile.preferred_username,
        preferences=profile.preferences,
        created_at=profile.created_at,
        last_login_at=profile.last_login_at,
    )


@router.patch(
    "/me/preferences",
    response_model=UserProfileResponse,
    summary="Update user preferences",
    description="Update the authenticated user's preferences.",
)
async def update_user_preferences(
    preferences: UserPreferences,
    user: AuthenticatedUser = Depends(get_current_user_azure),
    db: CosmosClient = Depends(get_db),
) -> UserProfileResponse:
    """Update the current user's preferences.

    Args:
        preferences: New preference values
        user: Authenticated user from Entra ID token
        db: Cosmos DB client

    Returns:
        Updated UserProfileResponse
    """
    # Get or create user profile first
    user_data = UserProfileCreate(
        oid=user.oid,
        email=user.email,
        name=user.name,
        preferred_username=user.preferred_username,
        tenant_id=user.tenant_id,
    )
    profile, _ = await db.get_or_create_user_profile(user_data)

    # Update preferences
    profile.preferences = preferences

    return UserProfileResponse(
        id=profile.id,
        email=profile.email,
        name=profile.name,
        preferred_username=profile.preferred_username,
        preferences=profile.preferences,
        created_at=profile.created_at,
        last_login_at=profile.last_login_at,
    )
