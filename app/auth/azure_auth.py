"""Azure Entra ID authentication using fastapi-azure-auth.

This module provides OAuth2 authentication for FastAPI using the fastapi-azure-auth
library. It supports both Azure Commercial and Azure Government cloud environments.

The library handles:
- OAuth2 Authorization Code Flow with PKCE
- Token validation against Azure Entra ID JWKS
- OpenID configuration loading and caching
- Swagger UI OAuth2 integration
"""

from functools import lru_cache
from typing import Optional

from fastapi_azure_auth import SingleTenantAzureAuthorizationCodeBearer
from fastapi_azure_auth.user import User

from config.settings import AzureCloud, Settings, get_settings


def get_azure_authority_host(cloud: AzureCloud) -> str:
    """Get the Azure authority host based on cloud environment.

    Args:
        cloud: The Azure cloud environment

    Returns:
        The authority host URL (e.g., login.microsoftonline.us for Government)
    """
    if cloud == AzureCloud.GOVERNMENT:
        return "https://login.microsoftonline.us"
    return "https://login.microsoftonline.com"


def get_openid_config_url(tenant_id: str, cloud: AzureCloud) -> str:
    """Get the OpenID configuration URL for the tenant.

    Args:
        tenant_id: The Azure tenant ID
        cloud: The Azure cloud environment

    Returns:
        The OpenID configuration URL
    """
    authority_host = get_azure_authority_host(cloud)
    return f"{authority_host}/{tenant_id}/v2.0/.well-known/openid-configuration"


def get_authorization_url(tenant_id: str, cloud: AzureCloud) -> str:
    """Get the OAuth2 authorization URL.

    Args:
        tenant_id: The Azure tenant ID
        cloud: The Azure cloud environment

    Returns:
        The authorization URL for OAuth2 flow
    """
    authority_host = get_azure_authority_host(cloud)
    return f"{authority_host}/{tenant_id}/oauth2/v2.0/authorize"


def get_token_url(tenant_id: str, cloud: AzureCloud) -> str:
    """Get the OAuth2 token URL.

    Args:
        tenant_id: The Azure tenant ID
        cloud: The Azure cloud environment

    Returns:
        The token URL for OAuth2 flow
    """
    authority_host = get_azure_authority_host(cloud)
    return f"{authority_host}/{tenant_id}/oauth2/v2.0/token"


def create_azure_scheme(settings: Optional[Settings] = None) -> SingleTenantAzureAuthorizationCodeBearer:
    """Create the Azure authentication scheme.

    Creates a SingleTenantAzureAuthorizationCodeBearer configured for the
    appropriate Azure cloud environment (Commercial or Government).

    Args:
        settings: Application settings. Uses default settings if not provided.

    Returns:
        Configured authentication scheme for Azure Entra ID

    Raises:
        ValueError: If required Azure configuration is missing
    """
    if settings is None:
        settings = get_settings()

    if not settings.azure_client_id:
        raise ValueError("AZURE_CLIENT_ID is required for authentication")
    if not settings.azure_tenant_id:
        raise ValueError("AZURE_TENANT_ID is required for authentication")

    # Build scope configuration
    scope_name = f"api://{settings.azure_client_id}/user_impersonation"
    scopes = {scope_name: "user_impersonation"}

    # Get cloud-specific URLs
    cloud = settings.azure_cloud
    tenant_id = settings.azure_tenant_id

    # Create the scheme with Government cloud support
    # The key insight: we must override openapi_authorization_url and openapi_token_url
    # for Swagger UI to use the correct endpoints, AND use openid_config_url for
    # token validation via the parent class
    scheme = SingleTenantAzureAuthorizationCodeBearer(
        app_client_id=settings.azure_client_id,
        tenant_id=tenant_id,
        scopes=scopes,
        openapi_authorization_url=get_authorization_url(tenant_id, cloud),
        openapi_token_url=get_token_url(tenant_id, cloud),
        openapi_description="Azure Entra ID OAuth2 - leave Client Secret blank",
        allow_guest_users=False,
    )

    # Override the OpenID config URL for Government cloud
    # This ensures token validation uses the correct JWKS endpoint
    if cloud == AzureCloud.GOVERNMENT:
        scheme.openid_config.config_url = get_openid_config_url(tenant_id, cloud)

    return scheme


@lru_cache
def get_azure_scheme() -> SingleTenantAzureAuthorizationCodeBearer:
    """Get the cached Azure authentication scheme.

    Returns:
        SingleTenantAzureAuthorizationCodeBearer: Cached authentication scheme
    """
    return create_azure_scheme()


# Type alias for the authenticated user from fastapi-azure-auth
AzureUser = User
