# Azure Entra ID App Registration Setup Guide

This guide documents the process for configuring Azure Entra ID (formerly Azure Active Directory) authentication for the Captain's Log application. The setup supports both Azure Commercial and Azure Government clouds.

## Prerequisites

- Azure subscription with Owner or Application Administrator role
- Azure CLI installed and configured
- Access to Azure Portal or Azure Government Portal

## Cloud Environment Endpoints

| Component | Commercial Cloud | Government Cloud |
|-----------|-----------------|------------------|
| Portal | portal.azure.com | portal.azure.us |
| Login Endpoint | login.microsoftonline.com | login.microsoftonline.us |
| Graph API | graph.microsoft.com | graph.microsoft.us |
| JWKS URI | `https://login.microsoftonline.com/{tenant}/discovery/v2.0/keys` | `https://login.microsoftonline.us/{tenant}/discovery/v2.0/keys` |

## Step 1: Create App Registration

### Azure Portal Method

1. Navigate to [Azure Portal](https://portal.azure.com) (or [Azure Government Portal](https://portal.azure.us))
2. Go to **Microsoft Entra ID** > **App registrations**
3. Click **New registration**
4. Configure the following:
   - **Name**: `captains-log-api` (or your preferred name)
   - **Supported account types**: Choose based on your requirements:
     - *Single tenant*: Accounts in this organizational directory only
     - *Multitenant*: Accounts in any organizational directory
   - **Redirect URI**: 
     - Platform: **Web**
     - URI: `https://<your-app-url>/.auth/login/aad/callback`
5. Click **Register**
6. Note the **Application (client) ID** and **Directory (tenant) ID**

### Azure CLI Method

```bash
# Commercial Cloud
az cloud set --name AzureCloud
az login

# OR Government Cloud
az cloud set --name AzureUSGovernment
az login

# Create app registration
az ad app create \
  --display-name "captains-log-api" \
  --sign-in-audience "AzureADMyOrg" \
  --web-redirect-uris "https://<your-app-url>/.auth/login/aad/callback"

# Note the appId from output
```

## Step 2: Configure API Permissions

The application requires the following Microsoft Graph API permissions:

### Required Permissions

| Permission | Type | Description |
|------------|------|-------------|
| User.Read | Delegated | Sign in and read user profile |
| openid | Delegated | Sign users in |
| profile | Delegated | View users' basic profile |
| email | Delegated | View users' email address |

### Portal Method

1. Go to **App registrations** > Your app > **API permissions**
2. Click **Add a permission**
3. Select **Microsoft Graph** > **Delegated permissions**
4. Add: `User.Read`, `openid`, `profile`, `email`
5. Click **Grant admin consent for [Your Organization]** (if you have admin rights)

### CLI Method

```bash
# Get the app ID
APP_ID="<your-application-id>"

# Add Microsoft Graph permissions
az ad app permission add \
  --id $APP_ID \
  --api 00000003-0000-0000-c000-000000000000 \
  --api-permissions e1fe6dd8-ba31-4d61-89e7-88639da4683d=Scope # User.Read

# Grant admin consent
az ad app permission admin-consent --id $APP_ID
```

## Step 3: Expose an API (Optional)

If you need to protect backend API endpoints:

1. Go to **App registrations** > Your app > **Expose an API**
2. Click **Set** next to Application ID URI
3. Accept the default `api://<client-id>` or customize
4. Add scopes as needed:

### Add API Scope

1. Click **Add a scope**
2. Configure:
   - **Scope name**: `access_as_user`
   - **Who can consent**: Admins and users
   - **Admin consent display name**: Access Captain's Log API
   - **Admin consent description**: Allows the app to access the Captain's Log API on behalf of the signed-in user.
   - **User consent display name**: Access Captain's Log API
   - **User consent description**: Allow the application to access Captain's Log on your behalf.
3. Click **Add scope**

## Step 4: Configure Token Claims

Configure optional claims for user identification:

1. Go to **App registrations** > Your app > **Token configuration**
2. Click **Add optional claim**
3. Select **ID** token type
4. Add the following claims:
   - `email`
   - `given_name`
   - `family_name`
   - `upn` (User Principal Name)
5. Repeat for **Access** token type if needed

## Step 5: Configure Redirect URIs

Add all required redirect URIs for your environments:

1. Go to **App registrations** > Your app > **Authentication**
2. Under **Web** platform, add redirect URIs:

```
# Production
https://<app-name>.azurewebsites.net/.auth/login/aad/callback
https://<app-name>.azurewebsites.us/.auth/login/aad/callback  # Gov

# Local Development
http://localhost:8501/callback  # Streamlit default
http://localhost:8000/auth/callback  # FastAPI default
```

3. Configure **Implicit grant and hybrid flows** (if using frontend SPA):
   - Check **ID tokens** for implicit flow
   
4. Set **Supported account types** as needed

## Step 6: Create Client Secret (Optional)

If your backend needs to call Microsoft Graph or validate tokens server-side:

1. Go to **App registrations** > Your app > **Certificates & secrets**
2. Click **New client secret**
3. Add a description and expiration period
4. Click **Add**
5. **Copy the secret value immediately** (it won't be shown again)

## Step 7: Configure Application Settings

Set the following environment variables in your application:

```bash
# Required
AZURE_TENANT_ID=<your-tenant-id>
AZURE_CLIENT_ID=<your-application-id>

# Optional - only if using client credentials flow
AZURE_CLIENT_SECRET=<your-client-secret>

# Cloud environment
AZURE_CLOUD=government  # or 'commercial'
```

## Step 8: Deploy Bicep Infrastructure

The `entra-auth.bicep` module configures App Service authentication:

```bash
# Deploy with Azure CLI
az deployment group create \
  --resource-group <your-rg> \
  --template-file infra/entra-auth.bicep \
  --parameters \
    appServiceName=<your-app-service> \
    tenantId=<your-tenant-id> \
    clientId=<your-client-id> \
    azureCloud=government
```

Or include in your main Bicep deployment:

```bicep
module entraAuth 'entra-auth.bicep' = {
  name: 'entra-auth'
  params: {
    appServiceName: appService.name
    tenantId: tenantId
    clientId: clientId
    azureCloud: 'government'
  }
}
```

## Validation

### Test Token Validation

```bash
# Get a token (interactive login)
az account get-access-token \
  --resource api://<client-id> \
  --query accessToken -o tsv

# Test protected endpoint
curl -H "Authorization: Bearer <token>" \
  https://<your-app-url>/auth/me
```

### Verify JWKS Endpoint

```bash
# Commercial
curl https://login.microsoftonline.com/<tenant-id>/discovery/v2.0/keys

# Government
curl https://login.microsoftonline.us/<tenant-id>/discovery/v2.0/keys
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| AADSTS50011: Reply URL mismatch | Verify redirect URIs match exactly in app registration |
| AADSTS700016: Application not found | Check tenant ID and client ID are correct |
| AADSTS65001: User consent required | Grant admin consent or configure user consent |
| Token validation fails | Verify audience and issuer match app registration |

### Government Cloud Specific

- Ensure you're logged into the correct cloud: `az cloud set --name AzureUSGovernment`
- Use `.us` TLD for all endpoints
- Microsoft Graph endpoint is `graph.microsoft.us` not `graph.microsoft.com`

## Security Best Practices

1. **Rotate client secrets** regularly (recommended: 6 months)
2. **Use managed identities** when possible instead of client secrets
3. **Enable Conditional Access** policies for additional security
4. **Monitor sign-in logs** in Entra ID for suspicious activity
5. **Use certificate credentials** instead of secrets for production
6. **Implement token caching** to reduce authentication overhead

## Related Documentation

- [Microsoft Entra ID Documentation](https://learn.microsoft.com/en-us/entra/identity/)
- [Azure Government Entra ID](https://learn.microsoft.com/en-us/azure/azure-government/documentation-government-services-securityandidentity)
- [App Registration Best Practices](https://learn.microsoft.com/en-us/entra/identity-platform/security-best-practices-for-app-registration)
