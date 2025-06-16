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

var tags = { 'azd-env-name': environmentName }

var resourceToken = toLower(uniqueString(subscription().id, environmentName))

resource resourceGroup 'Microsoft.Resources/resourceGroups@2021-04-01' = {
  name: resourceGroupName
  location: location
  tags: tags
}

module resources 'resources.bicep' = {
  name: 'resources'
  scope: resourceGroup
  params: {
    location: location
    resourceToken: resourceToken
    tags: tags
    principalId: principalId
  }
}

output AZURE_OPENAI_ENDPOINT string = resources.outputs.AZURE_OPENAI_ENDPOINT
output AZURE_SPEECH_ENDPOINT string = resources.outputs.AZURE_SPEECH_ENDPOINT
output AZURE_SPEECH_REGION string = resources.outputs.AZURE_SPEECH_REGION
output AZURE_SPEECH_KEY string = resources.outputs.AZURE_SPEECH_KEY
output RESOURCE_GROUP_ID string = resourceGroup.id
