"""FastAPI dependencies for Azure Entra ID authentication.

This module provides FastAPI dependencies for protecting endpoints with
Azure Entra ID token validation, supporting both Commercial and Government clouds.
"""

from typing import Optional

from fastapi import Depends, Header, HTTPException, Request, status

from auth.entra import (
    AuthenticatedUser,
    EntraTokenValidator,
    MissingTokenError,
    TokenExpiredError,
    TokenValidationError,
    get_entra_validator,
)


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
