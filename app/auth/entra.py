"""Azure Entra ID token validation.

This module provides JWT token validation for Azure Entra ID (Azure AD) access tokens,
supporting both Azure Commercial and Azure Government cloud environments.
"""

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Optional

import jwt
from jwt import PyJWKClient

from config.settings import AzureCloud, Settings, get_settings


class EntraAuthError(Exception):
    """Base exception for Entra authentication errors."""

    pass


class TokenValidationError(EntraAuthError):
    """Exception raised when token validation fails."""

    pass


class TokenExpiredError(EntraAuthError):
    """Exception raised when token has expired."""

    pass


class MissingTokenError(EntraAuthError):
    """Exception raised when token is missing."""

    pass


@dataclass
class AuthenticatedUser:
    """Represents an authenticated user from Azure Entra ID token claims."""

    oid: str  # Object ID (unique user identifier) - uses sub as fallback
    email: Optional[str] = None
    name: Optional[str] = None
    preferred_username: Optional[str] = None
    tenant_id: Optional[str] = None
    roles: list[str] | None = None

    @classmethod
    def from_claims(cls, claims: dict[str, Any]) -> "AuthenticatedUser":
        """Create AuthenticatedUser from JWT claims.

        Handles various Azure token formats (v1, v2, ID tokens, access tokens)
        by checking multiple possible claim locations.

        Args:
            claims: Decoded JWT token claims

        Returns:
            AuthenticatedUser instance with extracted user information

        Note:
            - oid: Object ID is preferred, but sub (subject) is always present
            - email: May be in email, preferred_username, or upn claims
            - name: May need to be constructed from given_name + family_name
        """
        # User identifier: prefer oid, fall back to sub (always present in Azure tokens)
        user_id = claims.get("oid") or claims.get("sub") or ""

        # Email: try multiple sources - Azure tokens vary in which claims are present
        email = (
            claims.get("email")
            or claims.get("preferred_username")
            or claims.get("upn")  # User Principal Name - common in Azure Gov
            or claims.get("unique_name")  # v1 tokens
            or None
        )

        # Name: may be in name claim or need to be constructed
        name = claims.get("name")
        if not name:
            given_name = claims.get("given_name", "")
            family_name = claims.get("family_name", "")
            if given_name or family_name:
                name = f"{given_name} {family_name}".strip()

        # Preferred username: try multiple sources
        preferred_username = (
            claims.get("preferred_username")
            or claims.get("upn")
            or email
        )

        return cls(
            oid=user_id,
            email=email,
            name=name,
            preferred_username=preferred_username,
            tenant_id=claims.get("tid"),
            roles=claims.get("roles"),
        )


class EntraTokenValidator:
    """Validates Azure Entra ID JWT access tokens.

    Supports Azure Commercial and Government cloud environments.
    Caches JWKS keys for performance.
    """

    # JWKS cache lifetime in seconds (default 1 hour)
    JWKS_CACHE_LIFETIME = 3600

    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize the token validator.

        Args:
            settings: Application settings. If None, will use default settings.
        """
        self._settings = settings or get_settings()
        self._jwks_client: Optional[PyJWKClient] = None
        self._jwks_last_refresh: float = 0

    @property
    def settings(self) -> Settings:
        """Get the settings instance."""
        return self._settings

    @property
    def issuer(self) -> str:
        """Get the expected token issuer based on cloud environment."""
        tenant_id = self._settings.azure_tenant_id
        if self._settings.azure_cloud == AzureCloud.GOVERNMENT:
            return f"https://login.microsoftonline.us/{tenant_id}/v2.0"
        return f"https://login.microsoftonline.com/{tenant_id}/v2.0"

    @property
    def audience(self) -> str:
        """Get the expected token audience (client ID)."""
        return self._settings.azure_client_id or ""

    @property
    def jwks_uri(self) -> str:
        """Get the JWKS URI based on cloud environment."""
        return self._settings.entra_jwks_uri

    def _get_jwks_client(self) -> PyJWKClient:
        """Get or create the JWKS client with caching.

        Returns:
            PyJWKClient instance for fetching signing keys
        """
        current_time = time.time()

        # Check if we need to refresh the JWKS client
        if (
            self._jwks_client is None
            or current_time - self._jwks_last_refresh > self.JWKS_CACHE_LIFETIME
        ):
            self._jwks_client = PyJWKClient(
                self.jwks_uri,
                cache_keys=True,
                lifespan=self.JWKS_CACHE_LIFETIME,
            )
            self._jwks_last_refresh = current_time

        return self._jwks_client

    def validate_token(self, token: str) -> AuthenticatedUser:
        """Validate an Azure Entra ID access token.

        Args:
            token: JWT access token from Authorization header

        Returns:
            AuthenticatedUser with extracted claims

        Raises:
            MissingTokenError: If token is empty or None
            TokenExpiredError: If token has expired
            TokenValidationError: If token validation fails
        """
        if not token:
            raise MissingTokenError("Token is required")

        try:
            # Get the signing key from JWKS
            jwks_client = self._get_jwks_client()
            signing_key = jwks_client.get_signing_key_from_jwt(token)

            # Decode and validate the token
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self.audience,
                issuer=self.issuer,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_nbf": True,
                    "verify_iat": True,
                    "verify_aud": True,
                    "verify_iss": True,
                    "require": ["exp", "iat", "aud", "iss"],
                },
            )

            return AuthenticatedUser.from_claims(claims)

        except jwt.ExpiredSignatureError as e:
            raise TokenExpiredError("Token has expired") from e
        except jwt.InvalidAudienceError as e:
            raise TokenValidationError(f"Invalid audience: {e}") from e
        except jwt.InvalidIssuerError as e:
            raise TokenValidationError(f"Invalid issuer: {e}") from e
        except jwt.DecodeError as e:
            raise TokenValidationError(f"Failed to decode token: {e}") from e
        except jwt.InvalidTokenError as e:
            raise TokenValidationError(f"Invalid token: {e}") from e
        except Exception as e:
            raise TokenValidationError(f"Token validation failed: {e}") from e

    def extract_token_from_header(self, authorization: Optional[str]) -> str:
        """Extract Bearer token from Authorization header.

        Args:
            authorization: Authorization header value

        Returns:
            Extracted token string

        Raises:
            MissingTokenError: If Authorization header is missing or invalid
        """
        if not authorization:
            raise MissingTokenError("Authorization header is required")

        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise MissingTokenError("Invalid Authorization header format. Expected: Bearer <token>")

        return parts[1]


@lru_cache
def get_entra_validator() -> EntraTokenValidator:
    """Get cached EntraTokenValidator instance.

    Returns:
        EntraTokenValidator singleton instance
    """
    return EntraTokenValidator()
