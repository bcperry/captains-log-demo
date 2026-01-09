# CI/CD Pipeline Configuration

This document describes how to configure the GitHub Actions CI/CD pipeline for deploying Captain's Log to Azure Commercial and Azure Government clouds.

## Overview

The pipeline includes:
- **Build and Test**: Runs on all pushes and PRs, includes mypy type checking and pytest
- **Docker Build**: Builds the container image after tests pass
- **Deploy Commercial**: Automatic deployment to Azure Commercial on main branch pushes
- **Deploy Government**: Manual deployment to Azure Government via workflow dispatch

## Required GitHub Secrets

### Azure Commercial Environment

Create a GitHub Environment named `commercial` and configure these secrets:

| Secret Name | Description |
|-------------|-------------|
| `AZURE_CLIENT_ID` | Azure AD application (service principal) client ID |
| `AZURE_TENANT_ID` | Azure AD tenant ID |
| `AZURE_SUBSCRIPTION_ID` | Target Azure subscription ID |

### Azure Government Environment

Create a GitHub Environment named `government` and configure these secrets:

| Secret Name | Description |
|-------------|-------------|
| `AZURE_GOV_CLIENT_ID` | Azure Government AD application client ID |
| `AZURE_GOV_TENANT_ID` | Azure Government AD tenant ID |
| `AZURE_GOV_SUBSCRIPTION_ID` | Azure Government subscription ID |

## Setting Up Federated Credentials

The pipeline uses OpenID Connect (OIDC) for secure, secretless authentication with Azure. This requires setting up federated credentials in Azure AD.

### 1. Create an App Registration

```bash
# Commercial cloud
az ad app create --display-name "captains-log-github-actions"

# Government cloud
az cloud set --name AzureUSGovernment
az ad app create --display-name "captains-log-github-actions-gov"
```

### 2. Create a Service Principal

```bash
az ad sp create --id <APP_ID>
```

### 3. Assign Role to Subscription

```bash
az role assignment create \
  --role "Contributor" \
  --assignee <APP_ID> \
  --scope /subscriptions/<SUBSCRIPTION_ID>
```

### 4. Add Federated Credential

In the Azure Portal:
1. Navigate to Azure AD → App registrations → Your app
2. Go to Certificates & secrets → Federated credentials → Add credential
3. Select "GitHub Actions deploying Azure resources"
4. Configure:
   - Organization: `<your-github-org>`
   - Repository: `captains-log-demo`
   - Entity type: `Branch` (for commercial: `main`, for government: use `Environment`)
   - Environment: `commercial` or `government`

## Manual Deployment to Azure Government

To deploy to Azure Government:

1. Go to Actions tab in GitHub
2. Select "CI/CD Pipeline" workflow
3. Click "Run workflow"
4. Select `environment: government`
5. Click "Run workflow"

## Environment-Specific Configuration

The pipeline automatically sets these environment variables based on the target cloud:

| Variable | Commercial | Government |
|----------|------------|------------|
| `AZURE_CLOUD` | `commercial` | `government` |
| `AZURE_LOCATION` | `eastus` | `usgovvirginia` |

These are passed to the Bicep templates via `azd` to provision the correct endpoints.

## Smoke Tests

After deployment, the pipeline runs a health check against the deployed endpoint:

```bash
curl -sf "${ENDPOINT}/health"
```

This validates that:
- The container is running
- The FastAPI application started successfully
- The health endpoint is accessible

## Troubleshooting

### Authentication Failures

If you see "AADSTS..." errors:
1. Verify the federated credential is configured correctly
2. Check that the entity type matches (branch vs environment)
3. Ensure the service principal has Contributor access to the subscription

### Deployment Failures

1. Check the `azd` output for specific error messages
2. Verify all required resources are available in the target region
3. For Government cloud, ensure the service supports Azure Government

### Test Failures

1. Run tests locally: `cd app && uv run pytest -v`
2. Check for environment-specific issues (missing dependencies)
3. Verify mypy passes: `uv run mypy . --ignore-missing-imports`
