# Migration Prompt: vsa-demo-perry - Remove Client Secret for Compliance

## Context
I need to migrate my Azure AD app registration from **client secret (password-based) authentication** to comply with Microsoft's Safe Secrets Standard. The deadline is **February 3, 2026**.

**There are 3 options depending on how the app authenticates:**
1. **Public Client with PKCE** (EASIEST - if frontend-only SPA doing user sign-in)
2. **Federated Identity Credentials** (if backend needs to call APIs using managed identity)
3. **Certificate-based Auth** (if backend needs confidential client auth)

## App Registration Details
- **App Name:** vsa-demo-perry
- **App ID (Client ID):** <YOUR_CLIENT_ID>
- **Tenant:** Fairfax (US Gov) - `<YOUR_TENANT_ID>`
- **Current Secret Name:** `bot` (expires Nov 6, 2027 - but must be removed for compliance)
- **Secret Key ID:** <YOUR_SECRET_KEY_ID>

## App Hosting
- **Platform:** Azure Container Apps (Gov)
- **URL:** `https://ca-6nobqi2gz72ee.victoriouswater-0600b8cb.usgovarizona.azurecontainerapps.us`

## Redirect URIs Configured
- `https://ca-6nobqi2gz72ee.victoriouswater-0600b8cb.usgovarizona.azurecontainerapps.us` (SPA)
- `http://localhost:8000/oauth2-redirect` (SPA - local dev)
- `http://localhost:5173` (SPA - local dev)
- `http://localhost:3000` (SPA - local dev)
- `http://localhost:8001/auth/callback` (Web)

## API Permissions & Scopes
- **Exposes API:** `user_impersonation` scope ("Captain's Log API")
- **Requests:** Delegated (user) permissions from Microsoft Graph
- **Token Version:** v2.0

---

# OPTION 1: Public Client with PKCE (EASIEST)
**Use this if:** Your app is a SPA (React, Vue, etc.) that only does user sign-in and doesn't need backend client credentials.

Since you have SPA redirect URIs and only request delegated permissions, you likely don't need a client secret at all! PKCE (Proof Key for Code Exchange) handles auth securely without secrets.

### Step 1: Enable Public Client
```bash
az ad app update --id <YOUR_CLIENT_ID> --is-fallback-public-client true
```

### Step 2: Update Your Frontend Code
Make sure your MSAL.js config uses PKCE (it's the default in MSAL.js 2.x+):

```javascript
// MSAL.js configuration - no client secret needed!
const msalConfig = {
    auth: {
        clientId: "<YOUR_CLIENT_ID>",
        authority: "https://login.microsoftonline.us/<YOUR_TENANT_ID>",
        redirectUri: "https://ca-6nobqi2gz72ee.victoriouswater-0600b8cb.usgovarizona.azurecontainerapps.us"
    }
};

const pca = new msal.PublicClientApplication(msalConfig);

// Login with PKCE (automatic)
await pca.loginPopup({ scopes: ["User.Read"] });
```

### Step 3: Remove the Client Secret
```bash
az ad app credential delete \
  --id <YOUR_CLIENT_ID> \
  --key-id <YOUR_SECRET_KEY_ID>
```

### Step 4: Remove Secret from Code/Config
- Delete `AZURE_CLIENT_SECRET` from environment variables
- Remove any `client_secret` references in code

**That's it!** No certificates, no secrets, no rotation needed.

---

# OPTION 2: Federated Identity Credentials
**Use this if:** Your Container App backend needs to authenticate to Azure services (Key Vault, Storage, etc.) or call APIs using the app's identity (not user's identity).

### Step 1: Enable Managed Identity on Container App
```bash
# Get your Container App resource ID first
az containerapp identity assign \
  --name <your-container-app-name> \
  --resource-group <your-resource-group> \
  --system-assigned
```

### Step 2: Get the Managed Identity Principal ID
```bash
az containerapp show \
  --name <your-container-app-name> \
  --resource-group <your-resource-group> \
  --query "identity.principalId" -o tsv
```

### Step 3: Add Federated Identity Credential to App Registration
```bash
# Create a federated credential that trusts the managed identity
az ad app federated-credential create \
  --id <YOUR_CLIENT_ID> \
  --parameters '{
    "name": "container-app-federation",
    "issuer": "https://login.microsoftonline.us/<YOUR_TENANT_ID>/v2.0",
    "subject": "<managed-identity-principal-id>",
    "audiences": ["api://AzureADTokenExchange"]
  }'
```

### Step 4: Update Backend Code
```python
from azure.identity import ManagedIdentityCredential, ClientAssertionCredential

# For calling Azure services directly
credential = ManagedIdentityCredential()

# OR for getting tokens as the app registration
credential = ClientAssertionCredential(
    tenant_id="<YOUR_TENANT_ID>",
    client_id="<YOUR_CLIENT_ID>",
    func=lambda: get_managed_identity_token()  # Gets MI token to exchange
)
```

### Step 5: Remove the Old Secret
```bash
az ad app credential delete \
  --id <YOUR_CLIENT_ID> \
  --key-id <YOUR_SECRET_KEY_ID>
```

---

# OPTION 3: Certificate-based Authentication
**Use this if:** You need confidential client auth but can't use managed identity (e.g., local dev, non-Azure hosting, or specific requirements).

## Migration Steps Required

### 1. Generate a Certificate
Generate a self-signed certificate (or use one from a CA):
```bash
openssl req -x509 -nodes -newkey rsa:2048 \
  -keyout vsa-demo-perry.key \
  -out vsa-demo-perry.crt \
  -days 365 \
  -subj "/CN=vsa-demo-perry"
```

### 2. Upload Certificate to Azure AD App Registration
```bash
az ad app credential reset \
  --id <YOUR_CLIENT_ID> \
  --cert @vsa-demo-perry.crt \
  --append
```

### 3. Update Application Code
Replace the client secret authentication with certificate-based auth. 

**Before (using client secret):**
```python
# Example - adjust for your framework
client_secret = os.environ.get("AZURE_CLIENT_SECRET")
```

**After (using certificate):**
```python
# Example for MSAL Python
from msal import ConfidentialClientApplication

# Load the certificate and private key
with open("vsa-demo-perry.key", "r") as f:
    private_key = f.read()
with open("vsa-demo-perry.crt", "r") as f:
    certificate = f.read()

app = ConfidentialClientApplication(
    client_id="<YOUR_CLIENT_ID>",
    authority="https://login.microsoftonline.us/<YOUR_TENANT_ID>",
    client_credential={
        "private_key": private_key,
        "thumbprint": "<certificate_thumbprint>"  # Get this from the uploaded cert
    }
)
```

### 4. Store Certificate Securely
For Azure Container Apps, store the certificate in:
- **Azure Key Vault** (recommended) - reference it as a secret in Container Apps
- **Container Apps secrets** - less ideal but works

### 5. Update Environment Variables
Replace:
- `AZURE_CLIENT_SECRET` → Remove this
- Add: `AZURE_CLIENT_CERTIFICATE_PATH` or load from Key Vault

### 6. Remove the Old Client Secret
After confirming certificate auth works:
```bash
az ad app credential delete \
  --id <YOUR_CLIENT_ID> \
  --key-id <YOUR_SECRET_KEY_ID>
```

## Alternative: Managed Identity (If Applicable)
If this app only needs to authenticate to Azure resources (not user sign-in), consider using **Managed Identity** instead:
1. Enable system-assigned managed identity on the Container App
2. Grant the managed identity the necessary permissions
3. Use `DefaultAzureCredential` which automatically uses the managed identity
4. No certificates or secrets needed

---

# How to Choose the Right Option

| Scenario | Best Option |
|----------|-------------|
| SPA frontend doing user sign-in only | **Option 1: Public Client + PKCE** |
| Backend calling Azure services (Key Vault, Storage, etc.) | **Option 2: Federated Credentials** |
| Backend calling external APIs with app identity | **Option 2 or 3** |
| Local development / Non-Azure hosting | **Option 3: Certificate** |
| Hybrid (SPA frontend + backend API calls) | **Option 1 for frontend + Option 2 for backend** |

**Recommendation:** Based on your SPA redirect URIs and delegated permissions, start with **Option 1 (Public Client + PKCE)**. If your backend also needs credentials, add **Option 2 (Federated Credentials)** for that.

## Files to Update
Search the codebase for:
- `AZURE_CLIENT_SECRET`
- `client_secret`
- The actual secret value (if hardcoded - bad practice)
- MSAL or Azure Identity SDK configuration

## Testing
1. Test locally with the certificate files
2. Deploy to Container Apps with cert in Key Vault
3. Verify OAuth login flow still works
4. Remove the old secret only after confirming everything works

## Deadline
**February 3, 2026** - Microsoft will disable apps using password-based auth after this date.
