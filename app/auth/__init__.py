"""Azure Entra ID authentication module.

This module provides JWT token validation for Azure Entra ID (Azure AD),
supporting both Azure Commercial and Azure Government cloud environments.

Two authentication approaches are available:
1. Custom EntraTokenValidator (auth.entra) - direct JWT validation with PyJWT
2. fastapi-azure-auth integration (auth.azure_auth) - OAuth2 with Swagger UI support
"""

from auth.azure_auth import (
    AzureUser,
    create_azure_scheme,
    get_authorization_url,
    get_azure_authority_host,
    get_azure_scheme,
    get_openid_config_url,
    get_token_url,
)
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
    # fastapi-azure-auth integration
    "AzureUser",
    "create_azure_scheme",
    "get_azure_scheme",
    "get_azure_authority_host",
    "get_authorization_url",
    "get_token_url",
    "get_openid_config_url",
    # Dependencies for FastAPI
    "CurrentUser",
    "get_current_user",
    "get_optional_user",
    "get_token_validator",
    # Token validation (legacy)
    "AuthenticatedUser",
    "EntraTokenValidator",
    "get_entra_validator",
    # Exceptions
    "EntraAuthError",
    "MissingTokenError",
    "TokenExpiredError",
    "TokenValidationError",
]
