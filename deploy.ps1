#!/usr/bin/env pwsh

# PowerShell script to build, tag, and push Docker image to Azure Container Registry
# For Azure US Government Cloud

param(
    [string]$ImageTag = "latest",
    [string]$Version = $null
)

# Set error action preference
$ErrorActionPreference = "Stop"

Write-Host "🚀 Starting deployment process..." -ForegroundColor Green

# Set Azure cloud environment to US Government
Write-Host "🌐 Setting Azure cloud to US Government..." -ForegroundColor Yellow
az cloud set --name AzureUSGovernment

# Verify we're in the correct directory
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot
Write-Host "📁 Working directory: $ProjectRoot" -ForegroundColor Cyan

# Load environment variables from .azure/captainslog/.env
$EnvFile = Join-Path $ProjectRoot ".azure\captainslog\.env"
if (-not (Test-Path $EnvFile)) {
    throw "Environment file not found: $EnvFile"
}

Write-Host "📋 Loading environment variables from .env file..." -ForegroundColor Yellow
$EnvVars = @{}
Get-Content $EnvFile | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]*?)\s*=\s*"?([^"]*)"?\s*$') {
        $EnvVars[$Matches[1]] = $Matches[2]
    }
}

# Extract configuration from environment variables
$ImageName = "captains-log-web"
$AcrName = $EnvVars["AZURE_CONTAINER_REGISTRY_NAME"]
$AcrEndpoint = $EnvVars["AZURE_CONTAINER_REGISTRY_ENDPOINT"]
$Repository = "web"
$ResourceGroupId = $EnvVars["RESOURCE_GROUP_ID"]
$ResourceGroupName = if ($ResourceGroupId -match "/resourceGroups/([^/]+)") { $Matches[1] } else { $null }

# Frontend build args for Vite (baked into JS bundle at build time)
$ViteAzureClientId = $EnvVars["AZURE_ENTRA_CLIENT_ID"]
$ViteAzureTenantId = $EnvVars["AZURE_ENTRA_TENANT_ID"]
$ViteAzureCloud = if ($EnvVars["AZURE_CLOUD"]) { $EnvVars["AZURE_CLOUD"] } else { "government" }

# Validate required variables
if (-not $AcrName) { throw "AZURE_CONTAINER_REGISTRY_NAME not found in environment file" }
if (-not $AcrEndpoint) { throw "AZURE_CONTAINER_REGISTRY_ENDPOINT not found in environment file" }
if (-not $ResourceGroupName) { throw "Could not extract resource group name from RESOURCE_GROUP_ID" }
if (-not $ViteAzureClientId) { throw "AZURE_ENTRA_CLIENT_ID not found in environment file (required for frontend auth)" }
if (-not $ViteAzureTenantId) { throw "AZURE_ENTRA_TENANT_ID not found in environment file (required for frontend auth)" }

Write-Host "🔧 Configuration:" -ForegroundColor Cyan
Write-Host "  - Image Name: $ImageName" -ForegroundColor White
Write-Host "  - ACR Name: $AcrName" -ForegroundColor White
Write-Host "  - ACR Endpoint: $AcrEndpoint" -ForegroundColor White
Write-Host "  - Repository: $Repository" -ForegroundColor White
Write-Host "  - Resource Group: $ResourceGroupName" -ForegroundColor White
Write-Host "  - Tag: $ImageTag" -ForegroundColor White
Write-Host "  - VITE_AZURE_CLIENT_ID: $ViteAzureClientId" -ForegroundColor White
Write-Host "  - VITE_AZURE_TENANT_ID: $ViteAzureTenantId" -ForegroundColor White
Write-Host "  - VITE_AZURE_CLOUD: $ViteAzureCloud" -ForegroundColor White

try {
    # Step 1: Build the Docker image with frontend auth config baked in
    Write-Host "🔨 Building Docker image with frontend auth configuration..." -ForegroundColor Yellow
    docker build `
        --build-arg VITE_AZURE_CLIENT_ID=$ViteAzureClientId `
        --build-arg VITE_AZURE_TENANT_ID=$ViteAzureTenantId `
        --build-arg VITE_AZURE_CLOUD=$ViteAzureCloud `
        -t $ImageName .
    if ($LASTEXITCODE -ne 0) {
        throw "Docker build failed"
    }
    Write-Host "✅ Docker image built successfully" -ForegroundColor Green

    # Step 2: Tag the image for ACR
    Write-Host "🏷️  Tagging image for ACR..." -ForegroundColor Yellow
    $FullImageName = "$AcrEndpoint/${Repository}:$ImageTag"
    docker tag $ImageName $FullImageName
    if ($LASTEXITCODE -ne 0) {
        throw "Docker tag failed"
    }
    Write-Host "✅ Image tagged as: $FullImageName" -ForegroundColor Green

    # Step 3: Tag with version if provided
    if ($Version) {
        Write-Host "🏷️  Tagging image with version $Version..." -ForegroundColor Yellow
        $VersionedImageName = "$AcrEndpoint/${Repository}:$Version"
        docker tag $ImageName $VersionedImageName
        if ($LASTEXITCODE -ne 0) {
            throw "Docker version tag failed"
        }
        Write-Host "✅ Image tagged as: $VersionedImageName" -ForegroundColor Green
    }

    # Step 4: Login to ACR
    Write-Host "🔐 Logging into Azure Container Registry..." -ForegroundColor Yellow
    az acr login --name $AcrName --resource-group $ResourceGroupName
    if ($LASTEXITCODE -ne 0) {
        throw "ACR login failed"
    }
    Write-Host "✅ Successfully logged into ACR" -ForegroundColor Green

    # Step 5: Push the latest image
    Write-Host "📤 Pushing image to ACR..." -ForegroundColor Yellow
    docker push $FullImageName
    if ($LASTEXITCODE -ne 0) {
        throw "Docker push failed"
    }
    Write-Host "✅ Image pushed successfully: $FullImageName" -ForegroundColor Green

    # Step 6: Push versioned image if provided
    if ($Version) {
        Write-Host "📤 Pushing versioned image to ACR..." -ForegroundColor Yellow
        docker push $VersionedImageName
        if ($LASTEXITCODE -ne 0) {
            throw "Docker versioned push failed"
        }
        Write-Host "✅ Versioned image pushed successfully: $VersionedImageName" -ForegroundColor Green
    }

    # Step 7: Clean up local images (optional)
    Write-Host "🧹 Cleaning up local images..." -ForegroundColor Yellow
    docker rmi $ImageName -f 2>$null
    docker rmi $FullImageName -f 2>$null
    if ($Version) {
        docker rmi $VersionedImageName -f 2>$null
    }

    Write-Host "🎉 Deployment completed successfully!" -ForegroundColor Green
    Write-Host "📋 Summary:" -ForegroundColor Cyan
    Write-Host "  - Image pushed to: $FullImageName" -ForegroundColor White
    if ($Version) {
        Write-Host "  - Versioned image: $VersionedImageName" -ForegroundColor White
    }
    Write-Host "  - ACR: $AcrEndpoint" -ForegroundColor White

} catch {
    Write-Host "❌ Deployment failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}