targetScope = 'subscription'

@description('Name of the environment used to generate a short unique hash for resources.')
@minLength(1)
@maxLength(64)
param environmentName string

@description('Primary location for all resources')
param location string

@description('Resource group name')
param resourceGroupName string

param principalId string

@description('Azure Entra ID tenant ID for authentication')
param entraIdTenantId string = ''

@description('Azure Entra ID client ID (application ID) from app registration')
param entraIdClientId string = ''

@description('Azure cloud environment for Entra ID endpoints')
@allowed(['commercial', 'government'])
param azureCloud string = 'government'

@description('Enable Azure Entra ID authentication on App Service')
param enableEntraAuth bool = false

var tags = { 'azd-env-name': environmentName }

var resourceToken = toLower(uniqueString(subscription().id, environmentName))

resource resourceGroup 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: resourceGroupName
  location: location
  tags: tags
}

@description('Container image for the web service')
param webContainerImage string = ''

module resources 'resources.bicep' = {
  name: 'resources'
  scope: resourceGroup
  params: {
    location: location
    resourceToken: resourceToken
    tags: tags
    principalId: principalId
    containerImage: !empty(webContainerImage) ? webContainerImage : ''
    entraIdTenantId: entraIdTenantId
    entraIdClientId: entraIdClientId
    azureCloud: azureCloud
    enableEntraAuth: enableEntraAuth
  }
}


output AZURE_SPEECH_ENDPOINT string = resources.outputs.AZURE_SPEECH_ENDPOINT
output AZURE_SPEECH_KEY string = resources.outputs.AZURE_SPEECH_KEY
output AZURE_SPEECH_REGION string = location

output AZURE_OPENAI_ENDPOINT string = resources.outputs.AZURE_OPENAI_ENDPOINT
output AZURE_OPENAI_KEY string = resources.outputs.AZURE_OPENAI_KEY
output AZURE_OPENAI_MODEL_NAME string = resources.outputs.AZURE_OPENAI_MODEL_NAME
output AZURE_OPENAI_MODEL_VERSION string = resources.outputs.AZURE_OPENAI_MODEL_VERSION
output AZURE_OPENAI_API_VERSION string = resources.outputs.AZURE_OPENAI_API_VERSION

output AZURE_COSMOS_ENDPOINT string = resources.outputs.AZURE_COSMOS_ENDPOINT
output AZURE_COSMOS_DATABASE string = resources.outputs.AZURE_COSMOS_DATABASE

output RESOURCE_GROUP_ID string = resourceGroup.id

output CONTAINER_APP_URL string = resources.outputs.CONTAINER_APP_URL
output CONTAINER_APP_NAME string = resources.outputs.CONTAINER_APP_NAME
output CONTAINER_ENVIRONMENT_NAME string = resources.outputs.CONTAINER_ENVIRONMENT_NAME

output AZURE_CONTAINER_REGISTRY_NAME string = resources.outputs.AZURE_CONTAINER_REGISTRY_NAME
output AZURE_CONTAINER_REGISTRY_ENDPOINT string = resources.outputs.AZURE_CONTAINER_REGISTRY_ENDPOINT

output AZURE_STORAGE_ACCOUNT string = resources.outputs.AZURE_STORAGE_ACCOUNT
output AZURE_STORAGE_CONTAINER string = resources.outputs.AZURE_STORAGE_CONTAINER
output AZURE_STORAGE_ENDPOINT string = resources.outputs.AZURE_STORAGE_ENDPOINT
