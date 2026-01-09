"""Azure Entra ID authentication module.

This module provides JWT token validation for Azure Entra ID (Azure AD),
supporting both Azure Commercial and Azure Government cloud environments.
"""

from auth.dependencies import (
    CurrentUser,
    get_current_user,
    get_optional_user,
    get_token_validator,
)
from auth.entra import (
    AuthenticatedUser,
    EntraAuthError,
    EntraTokenValidator,
    MissingTokenError,
    TokenExpiredError,
    TokenValidationError,
    get_entra_validator,
)

__all__ = [
    # Dependencies for FastAPI
    "CurrentUser",
    "get_current_user",
    "get_optional_user",
    "get_token_validator",
    # Token validation
    "AuthenticatedUser",
    "EntraTokenValidator",
    "get_entra_validator",
    # Exceptions
    "EntraAuthError",
    "MissingTokenError",
    "TokenExpiredError",
    "TokenValidationError",
]
