param location string

@minLength(3)
@maxLength(22)
param resourceToken string

param tags object

param principalId string

@description('Container image reference for the web application')
param containerImage string = ''

@description('Azure Entra ID tenant ID for authentication')
param entraIdTenantId string = ''

@description('Azure Entra ID client ID (application ID) from app registration')
param entraIdClientId string = ''

@description('Azure cloud environment for Entra ID endpoints')
@allowed(['commercial', 'government'])
param azureCloud string = 'government'

@description('Enable Azure Entra ID authentication on App Service')
param enableEntraAuth bool = false

var abbrs = loadJsonContent('./abbreviations.json')

// Cloud-specific configurations using environment() function for cloud compatibility
var isGovernment = azureCloud == 'government'
var cognitiveServicesEndpointSuffix = isGovernment ? 'azure.us' : 'azure.com'
// Use environment().suffixes.storage for all clouds to avoid hardcoded URLs
var blobEndpointSuffix = environment().suffixes.storage
var azureRegion = isGovernment ? 'usgovvirginia' : location

// Compliance tags for Azure Government
var complianceTags = {
  environment: azureCloud
  dataClassification: 'sensitive'
  managedBy: 'azd'
  'compliance-framework': isGovernment ? 'FedRAMP' : 'standard'
}
var allTags = union(tags, complianceTags)

// Azure OpenAI Service - Essential for GPT and embeddings
resource azureOpenAI 'Microsoft.CognitiveServices/accounts@2024-04-01-preview' = {
  name: '${abbrs.cognitiveServicesAccounts}${resourceToken}'
  location: azureRegion
  kind: 'OpenAI'
  tags: allTags
  sku: {
    name: 'S0'
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    customSubDomainName: '${abbrs.cognitiveServicesAccounts}${resourceToken}'
  }
}

// Azure Speech Service - Essential for speech-to-text functionality
resource azureSpeechService 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: '${abbrs.cognitiveServicesAccounts}speech${resourceToken}'
  location: azureRegion
  kind: 'SpeechServices'
  tags: allTags
  sku: {
    name: 'S0'
    capacity: 10
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    customSubDomainName: '${abbrs.cognitiveServicesAccounts}speech${resourceToken}'
    publicNetworkAccess: 'Enabled'
  }
}

// Azure Storage Account for audio file uploads
resource storageAccount 'Microsoft.Storage/storageAccounts@2023-01-01' = {
  name: '${abbrs.storageStorageAccounts}${resourceToken}'
  location: azureRegion
  tags: allTags
  kind: 'StorageV2'
  sku: {
    name: 'Standard_LRS'  // Use Standard_GRS for production geo-redundancy
  }
  properties: {
    accessTier: 'Hot'
    allowBlobPublicAccess: false
    allowSharedKeyAccess: true  // Required for connection string auth, disable when using managed identity only
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
    networkAcls: {
      defaultAction: 'Allow'
      bypass: 'AzureServices'
    }
  }
}

// Blob service configuration
resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-01-01' = {
  parent: storageAccount
  name: 'default'
  properties: {
    deleteRetentionPolicy: {
      enabled: true
      days: 7
    }
    isVersioningEnabled: true  // Enable versioning for audit trail
  }
}

// Container for audio file uploads
resource audioUploadsContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobService
  name: 'audio-uploads'
  properties: {
    publicAccess: 'None'
    metadata: {
      purpose: 'audio-file-uploads-for-batch-transcription'
    }
  }
}

// Container for transcription JSON files
resource transcriptionsJsonContainer 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-01-01' = {
  parent: blobService
  name: 'transcriptions'
  properties: {
    publicAccess: 'None'
    metadata: {
      purpose: 'transcription-json-storage'
    }
  }
}

// OpenAI model deployments
var deployments = [
  {
    name: 'gpt-4o'
    skuName: 'Standard'
    modelVersion: '2024-11-20'
    capacity: 10
  }
  {
    name: 'text-embedding-ada-002'
    skuName: 'Standard'
    modelVersion: '2'
    capacity: 100
  }
]

@batchSize(1)
resource azureOpenAIModel 'Microsoft.CognitiveServices/accounts/deployments@2024-04-01-preview' = [for deployment in deployments: {  
  name: deployment.name
  parent: azureOpenAI
  sku: {
    name: deployment.skuName
    capacity: deployment.capacity
  }
  properties: {
    model: {
      format: 'OpenAI'
      name: deployment.name
      version: deployment.modelVersion
    }
  }  
}]

// User-assigned managed identity for app service authentication
resource userAssignedIdentity 'Microsoft.ManagedIdentity/userAssignedIdentities@2018-11-30' = {
  name: '${abbrs.managedIdentityUserAssignedIdentities}${resourceToken}'
  location: resourceGroup().location
  tags: allTags
}

// Azure Container Registry for hosting container images
resource containerRegistry 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: '${abbrs.containerRegistryRegistries}${resourceToken}'
  location: location
  tags: allTags
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: true
    publicNetworkAccess: 'Enabled'
  }
}

// Use our container registry if no image is provided
var defaultImage = '${containerRegistry.properties.loginServer}/web:latest'
var actualContainerImage = !empty(containerImage) ? containerImage : defaultImage

// Log Analytics Workspace for Container Apps
resource logAnalyticsWorkspace 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${abbrs.operationalInsightsWorkspaces}${resourceToken}'
  location: location
  tags: allTags
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
    publicNetworkAccessForIngestion: 'Enabled'
    publicNetworkAccessForQuery: 'Enabled'
  }
}

// Azure Container Apps Environment
resource containerAppsEnvironment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${abbrs.appManagedEnvironments}${resourceToken}'
  location: location
  tags: allTags
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalyticsWorkspace.properties.customerId
        sharedKey: logAnalyticsWorkspace.listKeys().primarySharedKey
      }
    }
    workloadProfiles: [
      {
        name: 'Consumption'
        workloadProfileType: 'Consumption'
      }
    ]
    zoneRedundant: false
  }
}

// Azure Container App for hosting the FastAPI application
resource containerApp 'Microsoft.App/containerApps@2024-03-01' = {
  name: '${abbrs.appContainerApps}${resourceToken}'
  location: location
  tags: union(allTags, { 'azd-service-name': 'api' })
  // Ensure AcrPull role assignment completes before Container App tries to pull images
  dependsOn: [
    acrPullRoleAssignment
  ]
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${userAssignedIdentity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: containerAppsEnvironment.id
    workloadProfileName: 'Consumption'
    configuration: {
      ingress: {
        external: true
        targetPort: 8001
        transport: 'auto'
        allowInsecure: false
        corsPolicy: {
          allowedOrigins: ['*']
          allowedMethods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS']
          allowedHeaders: ['*']
          exposeHeaders: ['*']
          maxAge: 600
        }
      }
      registries: [
        {
          server: containerRegistry.properties.loginServer
          identity: userAssignedIdentity.id
        }
      ]
      secrets: [
        {
          name: 'azure-speech-key'
          value: azureSpeechService.listKeys().key1
        }
        {
          name: 'azure-openai-key'
          value: azureOpenAI.listKeys().key1
        }
      ]
    }
    template: {
      containers: [
        {
          name: 'api'
          image: actualContainerImage
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          env: [
            {
              name: 'AZURE_CLOUD'
              value: azureCloud
            }
            {
              name: 'AZURE_SPEECH_KEY'
              secretRef: 'azure-speech-key'
            }
            {
              name: 'AZURE_SPEECH_REGION'
              value: azureRegion
            }
            {
              name: 'AZURE_SPEECH_ENDPOINT'
              value: 'https://${azureSpeechService.name}.cognitiveservices.${cognitiveServicesEndpointSuffix}/'
            }
            {
              name: 'AZURE_OPENAI_KEY'
              secretRef: 'azure-openai-key'
            }
            {
              name: 'AZURE_OPENAI_ENDPOINT'
              value: 'https://${azureOpenAI.name}.openai.${cognitiveServicesEndpointSuffix}/'
            }
            {
              name: 'AZURE_OPENAI_MODEL_NAME'
              value: deployments[0].name
            }
            {
              name: 'AZURE_CLIENT_ID'
              value: userAssignedIdentity.properties.clientId
            }
            {
              name: 'AZURE_STORAGE_ACCOUNT'
              value: storageAccount.name
            }
            {
              name: 'AZURE_STORAGE_CONTAINER'
              value: audioUploadsContainer.name
            }
            {
              name: 'AZURE_STORAGE_ENDPOINT'
              value: 'https://${storageAccount.name}.blob.${blobEndpointSuffix}'
            }
            {
              name: 'AZURE_TENANT_ID'
              value: entraIdTenantId
            }
            {
              name: 'AZURE_ENTRA_CLIENT_ID'
              value: entraIdClientId
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 3
      }
    }
  }
}

// App Service Plan for hosting the application
resource appServicePlan 'Microsoft.Web/serverfarms@2022-09-01' = {
  name: '${abbrs.webServerFarms}${resourceToken}'
  location: location
  tags: allTags
  properties: {
    reserved: true
  }
  sku: {
    name: 'B1'
  }
  kind: 'linux'
}

// App Service for hosting the application as a container
resource appService 'Microsoft.Web/sites@2022-09-01' = {
  name: '${abbrs.webSitesAppService}${resourceToken}'
  location: location
  tags: union(allTags, { 'azd-service-name': 'web' })
  // Ensure AcrPull role assignment completes before App Service tries to pull images
  dependsOn: [
    acrPullRoleAssignment
  ]
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${userAssignedIdentity.id}': {}
    }
  }
  properties: {
    serverFarmId: appServicePlan.id
    httpsOnly: true
    siteConfig: {
      alwaysOn: true
      linuxFxVersion: 'DOCKER|${actualContainerImage}'
      cors: {
        allowedOrigins: ['*']
        supportCredentials: false
      }
      acrUseManagedIdentityCreds: true
      acrUserManagedIdentityID: userAssignedIdentity.properties.clientId
      appSettings: [
        {
          name: 'WEBSITES_ENABLE_APP_SERVICE_STORAGE'
          value: 'false'
        }
        {
          name: 'DOCKER_REGISTRY_SERVER_URL'
          value: 'https://${containerRegistry.properties.loginServer}'
        }
        {
          name: 'DOCKER_ENABLE_CI'
          value: 'true'
        }
      ]
    }
  }
  
  resource appSettings 'config' = {
    name: 'appsettings'
    properties: {
      SCM_DO_BUILD_DURING_DEPLOYMENT: 'true'
      AZURE_CLOUD: azureCloud
      AZURE_OPENAI_ENDPOINT: 'https://${azureOpenAI.name}.openai.${cognitiveServicesEndpointSuffix}/'
      AZURE_SPEECH_ENDPOINT: 'https://${azureSpeechService.name}.cognitiveservices.${cognitiveServicesEndpointSuffix}/'
      AZURE_SPEECH_REGION: azureRegion
      AZURE_SPEECH_KEY: azureSpeechService.listKeys().key1
      AZURE_CLIENT_ID: userAssignedIdentity.properties.clientId
      AZURE_OPENAI_KEY: azureOpenAI.listKeys().key1
      AZURE_OPENAI_MODEL_NAME: deployments[0].name
      AZURE_OPENAI_MODEL_VERSION: deployments[0].modelVersion
      AZURE_OPENAI_API_VERSION: '2024-02-15-preview'
      AZURE_OPENAI_EMBEDDING_MODEL_NAME: deployments[1].name
      AZURE_OPENAI_EMBEDDING_MODEL_VERSION: deployments[1].modelVersion
      AZURE_STORAGE_ACCOUNT: storageAccount.name
      AZURE_STORAGE_CONTAINER: audioUploadsContainer.name
      AZURE_STORAGE_ENDPOINT: 'https://${storageAccount.name}.blob.${blobEndpointSuffix}'
      AZURE_TENANT_ID: entraIdTenantId
      AZURE_ENTRA_CLIENT_ID: entraIdClientId
    }
  }
}

// Role assignments for accessing Azure OpenAI from user account
resource cognitiveServicesOpenAIUserForUser 'Microsoft.Authorization/roleAssignments@2020-04-01-preview' = {
  scope: azureOpenAI
  name: guid(azureOpenAI.id, principalId, resourceId('Microsoft.Authorization/roleDefinitions', '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'))
  properties: {
    roleDefinitionId: resourceId('Microsoft.Authorization/roleDefinitions', '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd')
    principalId: principalId
    principalType: 'User'
  }
}

// Role assignments for accessing Azure OpenAI from app service
resource cognitiveServicesOpenAIUserForAppService 'Microsoft.Authorization/roleAssignments@2020-04-01-preview' = {
  scope: azureOpenAI
  name: guid(azureOpenAI.id, userAssignedIdentity.id, resourceId('Microsoft.Authorization/roleDefinitions', '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd'))
  properties: {
    roleDefinitionId: resourceId('Microsoft.Authorization/roleDefinitions', '5e0bd9bd-7b93-4f28-af87-19fc36ad61bd')
    principalId: userAssignedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Role assignments for accessing Azure Speech Service from user account
resource cognitiveServicesSpeechUserForUser 'Microsoft.Authorization/roleAssignments@2020-04-01-preview' = {
  scope: azureSpeechService
  name: guid(azureSpeechService.id, principalId, resourceId('Microsoft.Authorization/roleDefinitions', 'f2dc8367-1007-4938-bd23-fe263f013447'))
  properties: {
    roleDefinitionId: resourceId('Microsoft.Authorization/roleDefinitions', 'f2dc8367-1007-4938-bd23-fe263f013447')
    principalId: principalId
    principalType: 'User'
  }
}

// Role assignments for accessing Azure Speech Service from app service
resource cognitiveServicesSpeechUserForAppService 'Microsoft.Authorization/roleAssignments@2020-04-01-preview' = {
  scope: azureSpeechService
  name: guid(azureSpeechService.id, userAssignedIdentity.id, resourceId('Microsoft.Authorization/roleDefinitions', 'f2dc8367-1007-4938-bd23-fe263f013447'))
  properties: {
    roleDefinitionId: resourceId('Microsoft.Authorization/roleDefinitions', 'f2dc8367-1007-4938-bd23-fe263f013447')
    principalId: userAssignedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Role assignment for ACR Pull access from app service managed identity
resource acrPullRoleAssignment 'Microsoft.Authorization/roleAssignments@2020-04-01-preview' = {
  scope: containerRegistry
  name: guid(containerRegistry.id, userAssignedIdentity.id, resourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d'))
  properties: {
    roleDefinitionId: resourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d') // AcrPull role
    principalId: userAssignedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Role assignment for Storage Blob Data Contributor access from user account
resource storageBlobDataContributorForUser 'Microsoft.Authorization/roleAssignments@2020-04-01-preview' = {
  scope: storageAccount
  name: guid(storageAccount.id, principalId, resourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'))
  properties: {
    roleDefinitionId: resourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe') // Storage Blob Data Contributor
    principalId: principalId
    principalType: 'User'
  }
}

// Role assignment for Storage Blob Data Contributor access from app service managed identity
resource storageBlobDataContributorForAppService 'Microsoft.Authorization/roleAssignments@2020-04-01-preview' = {
  scope: storageAccount
  name: guid(storageAccount.id, userAssignedIdentity.id, resourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe'))
  properties: {
    roleDefinitionId: resourceId('Microsoft.Authorization/roleDefinitions', 'ba92f5b4-2d11-453d-a403-e96b0029c9fe') // Storage Blob Data Contributor
    principalId: userAssignedIdentity.properties.principalId
    principalType: 'ServicePrincipal'
  }
}

// Optional: Configure Azure Entra ID authentication on App Service
module entraAuth 'entra-auth.bicep' = if (enableEntraAuth && !empty(entraIdTenantId) && !empty(entraIdClientId)) {
  name: 'entra-auth'
  params: {
    appServiceName: appService.name
    tenantId: entraIdTenantId
    clientId: entraIdClientId
    azureCloud: azureCloud
  }
}

// Outputs for environment variables (non-sensitive only - keys should not be in outputs)
output AZURE_OPENAI_ENDPOINT string = 'https://${azureOpenAI.name}.openai.${cognitiveServicesEndpointSuffix}/'
output AZURE_OPENAI_MODEL_NAME string = deployments[0].name
output AZURE_OPENAI_MODEL_VERSION string = deployments[0].modelVersion
output AZURE_OPENAI_API_VERSION string = '2024-02-15-preview'
output AZURE_OPENAI_EMBEDDING_MODEL_NAME string = deployments[1].name
output AZURE_OPENAI_EMBEDDING_MODEL_VERSION string = deployments[1].modelVersion
output AZURE_SPEECH_ENDPOINT string = 'https://${azureSpeechService.name}.cognitiveservices.${cognitiveServicesEndpointSuffix}/'

// Container Apps outputs
output CONTAINER_APP_URL string = 'https://${containerApp.properties.configuration.ingress.fqdn}'
output CONTAINER_APP_NAME string = containerApp.name
output CONTAINER_ENVIRONMENT_NAME string = containerAppsEnvironment.name

// Container Registry outputs
output AZURE_CONTAINER_REGISTRY_NAME string = containerRegistry.name
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = containerRegistry.properties.loginServer

// Storage Account outputs
output AZURE_STORAGE_ACCOUNT string = storageAccount.name
output AZURE_STORAGE_CONTAINER string = audioUploadsContainer.name
output AZURE_STORAGE_ENDPOINT string = 'https://${storageAccount.name}.blob.${blobEndpointSuffix}'
