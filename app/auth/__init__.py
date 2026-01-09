"""Azure Entra ID authentication module.

This module provides JWT token validation for Azure Entra ID (Azure AD),
supporting both Azure Commercial and Azure Government cloud environments.
"""

from auth.entra import (
    AuthenticatedUser,
    EntraTokenValidator,
    get_entra_validator,
)

__all__ = [
    "AuthenticatedUser",
    "EntraTokenValidator",
    "get_entra_validator",
]
