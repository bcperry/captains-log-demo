# Azure Government Deployment Guide

This guide provides specific instructions for deploying the Audio Transcription app to Azure Government.

## 🏛️ Azure Government Overview

Azure Government is a dedicated cloud environment for US government agencies and their partners, providing enhanced security and compliance features.

### Key Differences for Azure Government

- **Endpoints**: Uses `.azure.us` instead of `.azure.com`
- **Regions**: Limited to specific US government regions
- **Compliance**: FedRAMP High, DoD Impact Level 4, CJIS, etc.
- **Authentication**: Same Azure AD/Entra ID but with gov-specific endpoints

## 🔧 Prerequisites

- Azure Government subscription
- Access to Azure Government portal: https://portal.azure.us
- Azure CLI configured for Azure Government
- Azure Developer CLI (azd) configured for Azure Government

## ⚙️ Azure CLI Configuration

Configure Azure CLI for Azure Government:

```bash
# Set Azure CLI to use Azure Government cloud
az cloud set --name AzureUSGovernment

# Login to Azure Government
az login

# Verify you're connected to Azure Government
az account show
```

## 🚀 Deployment Steps

### 1. Configure Azure Developer CLI for Azure Government

```bash
# Set AZD to use Azure Government
azd config set cloud.azureGov true

# Authenticate with Azure Government
azd auth login
```

### 2. Deploy the Application

```bash
# From the project root directory
azd up
```

The deployment will automatically:
- Use Azure Government endpoints (`.azure.us`)
- Deploy to `usgovvirginia` region
- Configure Speech service for Azure Government
- Set up proper authentication for Azure Government

### 3. Verify Deployment

After deployment:

1. **Check the app URL** - should end with `.azurewebsites.us`
2. **Test transcription** - upload an audio file and verify it works
3. **Check logs** - ensure no connection errors to `.azure.com` endpoints

## 🔍 Azure Government Specific Configuration

### Infrastructure Configuration

The app is specifically configured for Azure Government:

```bicep
// Speech service deployed to Azure Government region
resource azureSpeechService 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  location: 'usgovvirginia'
  // ... other configuration
}

// Uses Azure Government endpoints
AZURE_SPEECH_ENDPOINT: 'https://${azureSpeechService.name}.cognitiveservices.azure.us/'
```

### Application Configuration

The app automatically detects and configures for Azure Government:

```python
# Detects Azure Government environment
def _detect_azure_government(self) -> bool:
    speech_endpoint = os.getenv("AZURE_SPEECH_ENDPOINT", "")
    return "azure.us" in speech_endpoint.lower()

# Converts HTTPS endpoint to WebSocket for Azure Government
def _get_azure_gov_websocket_endpoint(self, speech_endpoint: str) -> str:
    if speech_endpoint.startswith('https://'):
        base_url = speech_endpoint.replace('https://', 'wss://').rstrip('/')
        return f"{base_url}/speech/recognition/conversation/cognitiveservices/v1"
    return speech_endpoint

# Configures correct Azure Government endpoints
if self.is_azure_gov and speech_endpoint:
    websocket_endpoint = self._get_azure_gov_websocket_endpoint(speech_endpoint)
    self.speech_config = speechsdk.SpeechConfig(
        subscription=speech_key,
        endpoint=websocket_endpoint
    )
```

## 🌎 Available Azure Government Regions

| Region | Location | Code |
|--------|----------|------|
| US Gov Virginia | Virginia | `usgovvirginia` |
| US Gov Iowa | Iowa | `usgoviowa` |
| US Gov Arizona | Arizona | `usgovarizona` |
| US Gov Texas | Texas | `usgovtexas` |

**Note**: The app is configured to deploy to `usgovvirginia` by default. To change regions, modify the `location` parameter in `infra/resources.bicep`.

## 🔐 Security and Compliance

### Authentication

- **Managed Identity**: Used for secure service-to-service authentication
- **Azure Government AD**: Integrated with Azure Government Active Directory
- **No Secrets in Code**: All credentials managed through Azure Key Vault and managed identity

### Compliance Features

- **Data Residency**: All data stays within Azure Government datacenters
- **Encryption**: Data encrypted in transit and at rest
- **Audit Logging**: All operations logged for compliance tracking
- **Network Security**: Private endpoints and network security groups configured

## 🛠️ Local Development with Azure Government

For local development against Azure Government services:

### 1. Configure Environment Variables

```bash
# .env file for Azure Government
AZURE_SPEECH_KEY=your-azure-gov-speech-key
AZURE_SPEECH_REGION=usgovvirginia
AZURE_SPEECH_ENDPOINT=https://your-speech-service.cognitiveservices.azure.us/
```

### 2. Configure Azure CLI for Development

```bash
# Ensure Azure CLI uses Azure Government
az cloud set --name AzureUSGovernment
az login

# Get Speech service details
az cognitiveservices account show --name your-speech-service --resource-group your-rg
```

### 3. Test Local Connection

```bash
# Run the test script to verify Azure Government connectivity
python test_setup.py
```

## 🔄 Troubleshooting Azure Government Issues

### Common Issues

| Issue | Symptoms | Solution |
|-------|----------|----------|
| **Wrong Cloud Endpoint** | Connection errors to `.com` endpoints | Verify Azure CLI cloud setting: `az cloud show` |
| **Authentication Failures** | 401/403 errors | Ensure using Azure Government authentication endpoints |
| **Region Mismatch** | Service not found errors | Verify all services deployed to same Azure Gov region |
| **DNS Resolution** | Cannot resolve `.azure.us` | Check network allows access to Azure Government |
| **WebSocket Endpoint Error** | "You must specify a WS or WSS scheme" | App automatically converts HTTPS to WSS for Azure Gov |

### Debug Commands

```bash
# Check Azure CLI configuration
az cloud show
az account show

# Check AZD configuration  
azd config show

# Test Speech service connectivity
az cognitiveservices account list --resource-group your-rg
```

### Logs Analysis

Check application logs for Azure Government specific issues:

```bash
# Azure App Service logs
az webapp log tail --resource-group your-rg --name your-app

# Look for these indicators of successful Azure Government configuration:
# - "Speech service configured for Azure Government"
# - "Azure Government Cloud" in UI
# - Connections to ".azure.us" endpoints
```

## 📊 Monitoring in Azure Government

### Application Insights

Deploy Application Insights in Azure Government for monitoring:

```bicep
resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
  name: 'appinsights-${resourceToken}'
  location: 'usgovvirginia'
  kind: 'web'
  properties: {
    Application_Type: 'web'
  }
}
```

### Cost Management

- **Azure Government Pricing**: May differ from commercial Azure
- **Reserved Instances**: Available for Azure Government
- **Cost Monitoring**: Use Azure Government cost management tools

## 🆘 Azure Government Support

- **Azure Government Documentation**: [Azure Government Docs](https://docs.microsoft.com/azure/azure-government/)
- **Azure Government Support**: [Support Center](https://azure.microsoft.com/support/government/)
- **Compliance Information**: [Azure Government Compliance](https://docs.microsoft.com/azure/azure-government/compliance/)

## ✅ Deployment Checklist

- [ ] Azure CLI configured for Azure Government (`az cloud show`)
- [ ] AZD configured for Azure Government (`azd config show`)
- [ ] Authenticated to Azure Government (`az account show`)
- [ ] Deployed to Azure Government region (`usgovvirginia`)
- [ ] App accessible via `.azurewebsites.us` URL
- [ ] Speech service using `.azure.us` endpoints
- [ ] No connection attempts to `.azure.com` endpoints
- [ ] Transcription working with test audio files
- [ ] UI shows "Azure Government Cloud" status
