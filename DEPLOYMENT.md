# Deployment Guide for Audio Transcription App

This guide covers both local development and Azure cloud deployment of the Streamlit Audio Transcription app.

## 🏠 Local Development

### Prerequisites
- Python 3.8 or higher
- Azure Speech service resource (with subscription key)
- Git (for cloning)

### Quick Setup

1. **Configure Environment**
   ```bash
   cd app
   copy .env.example .env
   ```
   
2. **Edit .env file** with your Azure Speech service details:
   ```
   AZURE_SPEECH_KEY=your-32-character-speech-service-key
   AZURE_SPEECH_REGION=eastus
   ```

3. **Install and Run**
   ```bash
   # Windows - Simple startup
   start.bat
   
   # Or manual installation
   pip install -r requirements.txt
   python run_local.py
   ```

4. **Test Setup**
   ```bash
   python test_setup.py
   ```

5. **Access the App**
   - Open: http://localhost:8501
   - Upload an audio file and test transcription

### Getting Azure Speech Service Key

1. Go to [Azure Portal](https://portal.azure.com)
2. Create or navigate to your Speech service resource
3. Go to "Keys and Endpoint" section
4. Copy "Key 1" and the "Region"

## ☁️ Azure Deployment

### Prerequisites
- Azure subscription
- [Azure Developer CLI (azd)](https://learn.microsoft.com/azure/developer/azure-developer-cli/install-azd) installed
- Azure CLI authenticated: `az login`

### Deployment Steps

1. **Initialize Azure Developer CLI**
   ```bash
   # From project root directory
   azd auth login
   azd init
   ```

2. **Deploy to Azure**
   ```bash
   azd up
   ```
   
   This will:
   - Create Azure resource group
   - Deploy Azure Speech service
   - Deploy Azure App Service
   - Configure managed identity authentication
   - Set up environment variables

3. **Access Your App**
   - The deployment will output the app URL
   - Example: `https://app-captains-log-xyz123.azurewebsites.net`

### What Gets Created

The deployment creates these Azure resources:

| Resource | Purpose | Configuration |
|----------|---------|---------------|
| **Resource Group** | Container for all resources | Named with environment name |
| **Speech Service** | Azure AI Speech-to-Text | S0 tier, multi-language |
| **App Service Plan** | Hosting infrastructure | Linux, B1 tier, auto-scaling |
| **App Service** | Web application hosting | Python 3.11, Streamlit configured |
| **Managed Identity** | Secure authentication | No keys stored in code |

### Environment-Specific Deployments

Deploy to different environments:

```bash
# Development environment
azd env new dev
azd up

# Production environment  
azd env new prod
azd up
```

## 🔧 Configuration Management

### Local Development Variables

Create a `.env` file in the `app` directory:

```bash
# Required for local development
AZURE_SPEECH_KEY=your-speech-service-key
AZURE_SPEECH_REGION=eastus

# Optional for testing managed identity locally
AZURE_SPEECH_ENDPOINT=https://your-speech-service.cognitiveservices.azure.com/
AZURE_CLIENT_ID=your-managed-identity-client-id
```

### Azure Production Variables

Automatically configured by the deployment:

- `AZURE_SPEECH_KEY`: Injected from Speech service
- `AZURE_SPEECH_REGION`: Set to deployment region
- `AZURE_SPEECH_ENDPOINT`: Generated from Speech service
- `AZURE_CLIENT_ID`: Managed identity client ID

## 🔐 Security Configuration

### Local Development Security

- Use service principal or subscription key authentication
- Store keys in `.env` file (never commit to git)
- Use Azure CLI for authentication: `az login`

### Production Security

- **Managed Identity**: No keys stored in application code
- **RBAC**: Least privilege access to Speech service
- **HTTPS Only**: All traffic encrypted
- **Network Security**: App Service firewall rules

### Role Assignments

The deployment automatically assigns these roles:

| Principal | Resource | Role | Purpose |
|-----------|----------|------|---------|
| User Account | Speech Service | Cognitive Services Speech User | Development access |
| Managed Identity | Speech Service | Cognitive Services Speech User | App runtime access |

## 🚀 Scaling and Performance

### App Service Scaling

```bash
# Scale up to higher tier
az appservice plan update --resource-group myResourceGroup --name myAppServicePlan --sku P1V2

# Scale out to multiple instances  
az appservice plan update --resource-group myResourceGroup --name myAppServicePlan --number-of-workers 3
```

### Speech Service Scaling

- **S0 Tier**: 20 concurrent requests, unlimited transactions
- **Auto-scaling**: Handles burst traffic automatically
- **Global Availability**: Deploy in multiple regions

## 🔍 Monitoring and Troubleshooting

### Application Insights (Optional)

Add Application Insights for detailed monitoring:

1. Create Application Insights resource
2. Add to `resources.bicep`:
   ```bicep
   resource applicationInsights 'Microsoft.Insights/components@2020-02-02' = {
     name: 'appinsights-${resourceToken}'
     location: location
     kind: 'web'
     properties: {
       Application_Type: 'web'
     }
   }
   ```

### Common Issues and Solutions

| Issue | Symptoms | Solution |
|-------|----------|----------|
| Authentication failed | 401 errors, can't transcribe | Check Speech service key and region |
| No speech detected | Empty transcription results | Verify audio quality and language settings |
| App won't start | 500 errors, deployment failed | Check App Service logs in Azure portal |
| Slow transcription | Long processing times | Check Speech service region proximity |

### Viewing Logs

```bash
# Azure App Service logs
az webapp log tail --resource-group myResourceGroup --name myWebApp

# Or in Azure portal: App Service > Monitoring > Log stream
```

## 💰 Cost Optimization

### Speech Service Costs

- **S0 Tier**: $1 per 1,000 transactions
- **Optimize**: Use appropriate audio quality (16kHz mono)
- **Monitor**: Set up billing alerts

### App Service Costs

- **B1 Tier**: ~$13/month (development)
- **P1V2 Tier**: ~$73/month (production)
- **Auto-shutdown**: For development environments

### Cost Monitoring

```bash
# Set up cost alert
az consumption budget create --budget-name "AudioTranscriptionBudget" --amount 50 --time-grain Monthly
```

## 🔄 Updates and Maintenance

### Application Updates

```bash
# Deploy updates
azd deploy

# Or redeploy everything
azd up
```

### Infrastructure Updates

1. Modify `infra/resources.bicep`
2. Run `azd up` to apply changes
3. Test the updated deployment

### Backup and Recovery

- **Configuration**: Store in version control
- **Secrets**: Use Azure Key Vault for production
- **Data**: App is stateless, no backup needed

## 📊 Usage Analytics

### Built-in Analytics

The app provides:
- Processing time metrics
- Confidence scores
- Language detection accuracy
- File format statistics

### Custom Analytics

Add Azure Application Insights for:
- User behavior tracking
- Performance monitoring
- Error rate analysis
- Usage patterns

## 🆘 Support Resources

- **Azure Speech Documentation**: [Microsoft Docs](https://docs.microsoft.com/azure/cognitive-services/speech-service/)
- **Azure Developer CLI**: [AZD Documentation](https://learn.microsoft.com/azure/developer/azure-developer-cli/)
- **Streamlit Documentation**: [Streamlit Docs](https://docs.streamlit.io/)
- **Azure Support**: [Azure Support Center](https://azure.microsoft.com/support/)

## 🎯 Next Steps

After successful deployment:

1. **Test the application** with various audio files
2. **Configure custom domain** (optional)
3. **Set up monitoring alerts** for production
4. **Scale resources** based on usage patterns
5. **Implement additional features** as needed
