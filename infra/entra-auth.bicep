@description('Name of the App Service to configure authentication for')
param appServiceName string

@description('Azure Entra ID tenant ID')
param tenantId string

@description('Azure Entra ID client ID (application ID) from app registration')
param clientId string

@description('Azure cloud environment (commercial or government)')
@allowed(['commercial', 'government'])
param azureCloud string = 'government'

@description('Redirect URI path for authentication callback')
param redirectPath string = '/.auth/login/aad/callback'

@description('Allowed token audiences (typically the API identifier URI)')
param allowedAudiences array = []

@description('Whether to require authentication for all requests')
param requireAuthentication bool = true

@description('Action when request is not authenticated')
@allowed(['RedirectToLoginPage', 'AllowAnonymous', 'Return401', 'Return403'])
param unauthenticatedAction string = 'Return401'

// Determine the correct Entra ID endpoints using environment() function for cloud compatibility
// environment().authentication.loginEndpoint returns the correct login URL for any Azure cloud
// (commercial, government, China, etc.) without hardcoding URLs
var loginEndpoint = environment().authentication.loginEndpoint
var issuer = '${loginEndpoint}${tenantId}/v2.0'
var openIdIssuer = issuer

// Build allowed audiences list - always include client ID
var defaultAudiences = [
  'api://${clientId}'
  clientId
]
var finalAudiences = empty(allowedAudiences) ? defaultAudiences : union(defaultAudiences, allowedAudiences)

// Reference existing App Service
resource appService 'Microsoft.Web/sites@2022-09-01' existing = {
  name: appServiceName
}

// Configure Azure Entra ID authentication on App Service
resource authSettings 'Microsoft.Web/sites/config@2022-09-01' = {
  parent: appService
  name: 'authsettingsV2'
  properties: {
    globalValidation: {
      requireAuthentication: requireAuthentication
      unauthenticatedClientAction: unauthenticatedAction
      redirectToProvider: 'azureactivedirectory'
    }
    identityProviders: {
      azureActiveDirectory: {
        enabled: true
        registration: {
          openIdIssuer: openIdIssuer
          clientId: clientId
        }
        validation: {
          jwtClaimChecks: {}
          allowedAudiences: finalAudiences
          defaultAuthorizationPolicy: {
            allowedPrincipals: {}
          }
        }
        login: {
          disableWWWAuthenticate: false
        }
      }
    }
    login: {
      tokenStore: {
        enabled: true
      }
      preserveUrlFragmentsForLogins: false
      allowedExternalRedirectUrls: []
    }
    httpSettings: {
      requireHttps: true
      routes: {
        apiPrefix: '/.auth'
      }
      forwardProxy: {
        convention: 'NoProxy'
      }
    }
    platform: {
      enabled: true
      runtimeVersion: '~1'
    }
  }
}

// Outputs for reference
output authConfigured bool = true
output issuerUrl string = issuer
output configuredAudiences array = finalAudiences
output cloudEnvironment string = azureCloud
