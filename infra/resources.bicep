param location string

@minLength(3)
@maxLength(22)
param resourceToken string

param tags object

param principalId string

var abbrs = loadJsonContent('./abbreviations.json')

// Azure OpenAI Service - Essential for GPT and embeddings
resource azureOpenAI 'Microsoft.CognitiveServices/accounts@2024-04-01-preview' = {
  name: '${abbrs.cognitiveServicesAccounts}${resourceToken}'
  location: 'usgovvirginia'
  kind: 'OpenAI'
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
  location: 'usgovvirginia'  // Ensure consistent Azure Gov region
  kind: 'SpeechServices'
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

// OpenAI model deployments
var deployments = [
  {
    name: 'gpt-4o'
    skuName: 'Standard'
    modelVersion: '2024-05-13'
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
}

// App Service Plan for hosting the application
resource appServicePlan 'Microsoft.Web/serverfarms@2022-09-01' = {
  name: '${abbrs.webServerFarms}${resourceToken}'
  location: location
  properties: {
    reserved: true
  }
  sku: {
    name: 'B1'
  }
  kind: 'linux'
}

// App Service for hosting the Streamlit application
resource appService 'Microsoft.Web/sites@2022-03-01' = {
  name: '${abbrs.webSitesAppService}${resourceToken}'
  location: location
  tags: union(tags, { 'azd-service-name': 'web' })
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
      linuxFxVersion: 'PYTHON|3.11'
      appCommandLine: 'python -m streamlit run app.py --server.port 8000 --server.address 0.0.0.0 --server.runOnSave false'
      cors: {
        allowedOrigins: ['*']
        supportCredentials: false
      }
    }
  }
  
  resource appSettings 'config' = {
    name: 'appsettings'
    properties: {
      SCM_DO_BUILD_DURING_DEPLOYMENT: 'true'
      AZURE_OPENAI_ENDPOINT: 'https://${azureOpenAI.name}.openai.azure.us/'
      AZURE_SPEECH_ENDPOINT: 'https://${azureSpeechService.name}.cognitiveservices.azure.us/'
      AZURE_SPEECH_REGION: 'usgovvirginia'  // Hardcode for Azure Gov consistency
      AZURE_SPEECH_KEY: azureSpeechService.listKeys().key1
      AZURE_CLIENT_ID: userAssignedIdentity.properties.clientId
      AZURE_OPENAI_KEY: azureOpenAI.listKeys().key1
      AZURE_OPENAI_MODEL_NAME: deployments[0].name
      AZURE_OPENAI_MODEL_VERSION: deployments[0].modelVersion
      AZURE_OPENAI_API_VERSION: '2024-02-15-preview'
      AZURE_OPENAI_EMBEDDING_MODEL_NAME: deployments[1].name
      AZURE_OPENAI_EMBEDDING_MODEL_VERSION: deployments[1].modelVersion
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

// Outputs for environment variables
output AZURE_OPENAI_ENDPOINT string = 'https://${azureOpenAI.name}.openai.azure.us/'
output AZURE_OPENAI_KEY string = azureOpenAI.listKeys().key1
output AZURE_OPENAI_MODEL_NAME string = deployments[0].name
output AZURE_OPENAI_MODEL_VERSION string = deployments[0].modelVersion
output AZURE_OPENAI_API_VERSION string = '2024-02-15-preview'
output AZURE_OPENAI_EMBEDDING_MODEL_NAME string = deployments[1].name
output AZURE_OPENAI_EMBEDDING_MODEL_VERSION string = deployments[1].modelVersion
output AZURE_SPEECH_ENDPOINT string = 'https://${azureSpeechService.name}.cognitiveservices.azure.us/'
output AZURE_SPEECH_KEY string = azureSpeechService.listKeys().key1
