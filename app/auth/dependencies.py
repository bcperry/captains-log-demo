"""FastAPI dependencies for Azure Entra ID authentication.

This module provides FastAPI dependencies for protecting endpoints with
Azure Entra ID token validation, supporting both Commercial and Government clouds.

Two authentication approaches are available:
1. Legacy: get_current_user - uses custom EntraTokenValidator (PyJWT-based)
2. Modern: get_current_user_azure - uses fastapi-azure-auth with OAuth2/Swagger UI

The modern approach (get_current_user_azure) is preferred as it provides:
- Full OAuth2 Authorization Code flow with PKCE in Swagger UI
- Automatic JWKS caching and OpenID configuration loading
- Better integration with fastapi-azure-auth library features
"""

import logging
from typing import Any, Optional

from fastapi import Depends, Header, HTTPException, Request, Security, status

from auth.entra import (
    AuthenticatedUser,
    EntraTokenValidator,
    MissingTokenError,
    TokenExpiredError,
    TokenValidationError,
    get_entra_validator,
)
from config.settings import get_settings

logger = logging.getLogger(__name__)


def get_token_validator() -> EntraTokenValidator:
    """FastAPI dependency to get the cached EntraTokenValidator.

    Returns:
        EntraTokenValidator: Cached token validator instance
    """
    return get_entra_validator()


async def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    validator: EntraTokenValidator = Depends(get_token_validator),
) -> AuthenticatedUser:
    """FastAPI dependency for authenticating requests with Azure Entra ID tokens.

    Validates Bearer token from Authorization header against Azure Entra ID JWKS.
    Supports Azure Government endpoints.
    Caches JWKS keys for performance (via EntraTokenValidator).

    Args:
        request: FastAPI request object for storing user in state
        authorization: Authorization header value (Bearer <token>)
        validator: EntraTokenValidator instance (injected)

    Returns:
        AuthenticatedUser: Authenticated user with claims extracted from token

    Raises:
        HTTPException: 401 Unauthorized for missing, invalid, or expired tokens
    """
    try:
        # Extract Bearer token from Authorization header
        token = validator.extract_token_from_header(authorization)

        # Validate token and get authenticated user
        user = validator.validate_token(token)

        # Store user in request state for access in route handlers
        request.state.user = user

        return user

    except MissingTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

    except TokenExpiredError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": 'Bearer error="invalid_token", error_description="Token has expired"'},
        ) from e

    except TokenValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": 'Bearer error="invalid_token"'},
        ) from e


async def get_optional_user(
    request: Request,
    authorization: Optional[str] = Header(default=None, alias="Authorization"),
    validator: EntraTokenValidator = Depends(get_token_validator),
) -> Optional[AuthenticatedUser]:
    """FastAPI dependency for optionally authenticating requests.

    Unlike get_current_user, this dependency does not raise an exception
    if authentication fails - it returns None instead. Useful for endpoints
    that behave differently for authenticated vs anonymous users.

    Args:
        request: FastAPI request object for storing user in state
        authorization: Authorization header value (Bearer <token>)
        validator: EntraTokenValidator instance (injected)

    Returns:
        AuthenticatedUser if valid token present, None otherwise
    """
    if not authorization:
        return None

    try:
        token = validator.extract_token_from_header(authorization)
        user = validator.validate_token(token)
        request.state.user = user
        return user
    except (MissingTokenError, TokenExpiredError, TokenValidationError):
        return None


# Type alias for dependency injection
CurrentUser = AuthenticatedUser


def get_azure_scheme_dependency():  # type: ignore[no-untyped-def]
    """Get the Azure scheme for use as a dependency.

    This is a factory function that lazily imports and returns the azure_scheme.
    This avoids circular imports between auth modules.

    Returns:
        SingleTenantAzureAuthorizationCodeBearer or None if not configured
    """
    settings = get_settings()
    if not settings.is_entra_configured():
        return None

    from auth.azure_auth import get_azure_scheme

    return get_azure_scheme()


def _create_azure_user_dependency() -> Any:
    """Create a dependency function that uses Security() with azure_scheme.

    This factory returns a dependency that properly integrates with FastAPI's
    Security() wrapper, which provides the security_scopes argument that
    SingleTenantAzureAuthorizationCodeBearer requires.

    Returns:
        A dependency function or None if Azure auth is not configured.
    """
    azure_scheme = get_azure_scheme_dependency()
    if azure_scheme is None:
        return None

    # Return the scheme itself - it will be wrapped with Security() when used
    return azure_scheme


async def get_current_user_azure(
    request: Request,
) -> AuthenticatedUser:
    """FastAPI dependency for authenticating requests with fastapi-azure-auth.

    This is the preferred authentication method as it provides full OAuth2
    Authorization Code flow with PKCE support in Swagger UI.

    IMPORTANT: This dependency should be used AFTER Security(azure_scheme) has
    been applied at the router level. The Security() dependency handles token
    validation and makes the user available via request.state.user.

    When azure_scheme is not configured, this falls back to returning a 401.

    Args:
        request: FastAPI request object containing user from Security() dependency

    Returns:
        AuthenticatedUser: Authenticated user with claims extracted from token

    Raises:
        HTTPException: 401 Unauthorized if authentication is not configured or no user
    """
    # Check if user was already set by Security(azure_scheme) at router level
    # and is already converted to our AuthenticatedUser type
    if hasattr(request.state, "user") and request.state.user is not None:
        if isinstance(request.state.user, AuthenticatedUser):
            return request.state.user
        # If it's a fastapi-azure-auth User, we need to convert it below

    # Try to get user from fastapi-azure-auth User if set
    # This happens when Security(azure_scheme) is used
    azure_scheme = get_azure_scheme_dependency()
    if azure_scheme is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is not configured",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if Security(azure_scheme) already validated the token and set the user
    # The fastapi-azure-auth library sets the user in request.state.user
    azure_user = getattr(request.state, "user", None)
    
    if azure_user is None:
        # Extract token from Authorization header and validate manually
        # This is needed when the dependency is called directly without Security() wrapper
        from fastapi.security import SecurityScopes

        try:
            # Call the scheme with an empty SecurityScopes - token validation happens here
            azure_user = await azure_scheme(request, SecurityScopes(scopes=[]))
        except HTTPException:
            # Re-raise HTTP exceptions (401, 403, etc)
            raise
        except Exception as e:
            # Any other error during token validation
            logger.error(f"Token validation failed: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Token validation failed: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            ) from e

    # Log all claims for debugging - helps identify what's in the token
    logger.info(f"Azure user claims available: {list(azure_user.claims.keys())}")
    logger.debug(f"Azure user full claims: {azure_user.claims}")
    logger.info(f"Azure user oid: {azure_user.oid}")
    logger.info(f"Azure user sub: {azure_user.sub}")
    logger.info(f"Azure user name: {azure_user.name}")
    logger.info(f"Azure user email: {azure_user.email}")
    logger.info(f"Azure user preferred_username: {azure_user.preferred_username}")
    logger.info(f"Azure user tid: {azure_user.tid}")

    # Extract user identifier - prefer oid, fall back to sub
    # oid is the stable object ID in Azure AD, sub is always present in access tokens
    # In some Azure Gov configurations, oid may be missing but sub is always there
    user_id = azure_user.oid or azure_user.sub or azure_user.claims.get("oid") or azure_user.claims.get("sub") or ""
    
    if not user_id:
        # Log all claims to help debug
        logger.error(f"No oid or sub claim found in token. Available claims: {list(azure_user.claims.keys())}")
        logger.error(f"Full claims for debugging: {azure_user.claims}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: missing user identifier. Available claims: {list(azure_user.claims.keys())}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract email - try multiple claim sources
    # Azure tokens may have email in different places depending on token version
    email = (
        azure_user.email
        or azure_user.preferred_username
        or azure_user.claims.get("email")
        or azure_user.claims.get("preferred_username")
        or azure_user.claims.get("upn")  # User Principal Name
        or ""
    )

    # Extract name - try name claim first, then construct from given/family name
    name = azure_user.name
    if not name:
        given_name = azure_user.claims.get("given_name", "")
        family_name = azure_user.claims.get("family_name", "")
        if given_name or family_name:
            name = f"{given_name} {family_name}".strip()
        else:
            # Fall back to email/username as display name
            name = email or user_id

    # Extract preferred_username
    preferred_username = (
        azure_user.preferred_username
        or azure_user.claims.get("preferred_username")
        or azure_user.claims.get("upn")
        or email
        or ""
    )

    # Convert fastapi-azure-auth User to our AuthenticatedUser for compatibility
    user = AuthenticatedUser(
        oid=user_id,
        email=email if email else None,
        name=name if name else None,
        preferred_username=preferred_username if preferred_username else None,
        tenant_id=azure_user.tid or azure_user.claims.get("tid") or None,
    )

    logger.info(f"Authenticated user: oid={user.oid}, email={user.email}, name={user.name}")

    # Store user in request state for access in route handlers
    request.state.user = user

    return user
