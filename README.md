# Captain's Log: AI-Assisted Transcription and Analysis

A full-stack application for transcribing long-form audio and turning it into summaries, action items, and structured reports. The project evolved from a prototype built to process complete two-hour test events for downstream analysis and event notes.

## Portfolio summary

- **Problem:** Long technical events produce hours of audio that are difficult to search, review, and convert into usable notes.
- **Solution:** React and FastAPI application using Azure Speech Services, Azure OpenAI, and Cosmos DB.
- **Enterprise design:** Microsoft Entra authentication, Azure Government support, container deployment, and automated tests.
- **User experience:** Time-range selection, processing statistics, confidence metrics, and TXT/JSON/report exports.

## Features

- **High-quality transcription** with Azure Speech Services and multiple-language support
- **AI-assisted analysis** for summaries and action-item extraction
- **Flexible duration control** for full files or selected time ranges
- **Enterprise security** with Microsoft Entra ID and Azure Government support
- **Processing statistics** including processing time, word count, and confidence metrics
- **Multiple export formats** including TXT, JSON, and analysis reports
- **Responsive web interface** built with React, TypeScript, Vite, and Tailwind CSS
- **Container deployment** for Azure Container Apps

## Architecture

- **Frontend**: React 19 + TypeScript + Vite + Tailwind CSS
- **Backend**: FastAPI + Python 3.11+
- **Authentication**: Azure Entra ID (MSAL)
- **Services**: Azure Speech Services, Azure OpenAI, Azure Cosmos DB

## Quick start

### Prerequisites

- Python 3.11 or higher
- Node.js 22 or higher
- Azure subscription with Speech Services and OpenAI resources
- FFmpeg (for audio processing)

### Local Development

1. **Clone the repository**

2. **Install uv (Python package manager)**
   ```bash
   # macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Windows (PowerShell)
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

3. **Install backend dependencies**
   ```bash
   cd app
   uv sync
   ```

4. **Set up environment variables**
   Create a single `.env` file in the **project root** directory (not in app/ or frontend/):
   ```bash
   cp .env.example .env
   ```
   
   Edit the `.env` file with your Azure credentials:
   ```env
   # Shared configuration
   AZURE_CLOUD=commercial  # or 'government' for Azure Government
   AZURE_TENANT_ID=your_tenant_id
   AZURE_CLIENT_ID=your_client_id
   
   # Frontend uses VITE_ prefixed versions (must match above)
   VITE_AZURE_TENANT_ID=your_tenant_id
   VITE_AZURE_CLIENT_ID=your_client_id
   VITE_AZURE_CLOUD=commercial
   
   # Backend-only settings
   AZURE_SPEECH_KEY=your_speech_service_key
   AZURE_SPEECH_REGION=your_speech_region
   AZURE_OPENAI_ENDPOINT=your_openai_endpoint
   AZURE_OPENAI_KEY=your_openai_key
   ```

5. **Run the backend**
   ```bash
   cd app
   uv run uvicorn main:app --reload --port 8001
   ```

6. **Install frontend dependencies**
   ```bash
   cd frontend
   npm install
   ```

7. **Run the frontend**
   ```bash
   npm run dev
   ```

8. **One-liner for local development** (builds frontend and starts backend with hot reload)
   ```bash
   cd frontend && npm run build:deploy && cd ../app && uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

9. **Open your browser** to `http://localhost:8000`

### Using Docker

1. **Build the Docker image**
   ```bash
   docker build -t captains-log .
   ```

2. **Run the container**
   ```bash
   docker run -p 8001:8001 --env-file .env captains-log
   ```

## ☁️ Azure Deployment

### Prerequisites

- Azure CLI installed and configured
- Azure Developer CLI (azd) installed

### One-Click Deployment

1. **Initialize the project**
   ```bash
   azd init
   ```

2. **Deploy to Azure**
   ```bash
   azd up
   ```

This will:
- Create necessary Azure resources (Container Apps, Speech Services, OpenAI)
- Build and deploy the application
- Configure environment variables
- Set up Azure Entra ID authentication

## 📋 Supported Audio Formats

- **WAV** - Recommended for best quality
- **MP3** - Most common format
- **M4A** - Apple audio format
- **OGG** - Open source format
- **FLAC** - Lossless compression
- **MP4** - Video files with audio

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `AZURE_SPEECH_KEY` | Azure Speech Services API key | Yes |
| `AZURE_SPEECH_REGION` | Azure region (e.g., eastus) | Yes |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint | Yes |
| `AZURE_OPENAI_KEY` | Azure OpenAI API key | Yes |
| `AZURE_OPENAI_MODEL_NAME` | Model name (e.g., gpt-4) | Yes |
| `AZURE_CLOUD` | Cloud type: 'commercial' or 'government' | Yes |
| `AZURE_TENANT_ID` | Azure Entra ID tenant ID | Yes |
| `AZURE_CLIENT_ID` | Azure Entra ID client ID | Yes |

### Azure Government Support

The application automatically configures for Azure Government clouds:
- Uses `*.speech.azure.us` endpoints
- Uses `login.microsoftonline.us` for authentication
- Supports government-specific compliance requirements

## 🧪 Testing

### Backend Tests
```bash
cd app
uv run pytest tests/ -v
```

### Frontend Tests
```bash
cd frontend
npm run test
```

### Test Coverage
```bash
# Backend
uv run pytest tests/ --cov

# Frontend
npm run test -- --coverage
```

## 🔐 Security Features

- **Azure Entra ID** - Enterprise SSO authentication
- **JWT Validation** - Secure token verification
- **CORS Configuration** - Controlled cross-origin access
- **Managed Identity** - Secure Azure resource access
- **HTTPS Only** - Secure communication in production

## 🐛 Troubleshooting

### Common Issues

1. **"No speech detected"**
   - Ensure audio contains clear speech
   - Check audio format compatibility

2. **Authentication errors**
   - Verify Azure Entra ID configuration
   - Check client ID and tenant ID
   - Ensure redirect URI is configured correctly

3. **"Invalid token claims" or "Invalid issuer" error**
   - **Root cause**: App registration is issuing v1 tokens instead of v2 tokens
   - **Fix**: Go to Azure Portal → App Registrations → Your App → Manifest
   - Change `"accessTokenAcceptedVersion": null` to `"accessTokenAcceptedVersion": 2`
   - Save the manifest
   - Sign out and sign back in to get a new token
   - v2 tokens use issuer format `https://login.microsoftonline.us/{tenant}/v2.0` which is required

4. **API connection errors**
   - Check if backend is running on port 8000
   - Verify CORS configuration

5. **Long processing times**
   - Large files take more time to process
   - Consider using duration limits for testing

## 🤝 Contributing

We welcome contributions!

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Microsoft Azure** - For excellent Speech and OpenAI services
- **React** - For the frontend framework
- **FastAPI** - For the high-performance API framework
- **Vite** - For the blazing fast build tooling

---

<div align="center">
  <p>
    <a href="https://github.com/bcperry/captains-log-demo">⭐ Star this repo</a> |
    <a href="https://github.com/bcperry/captains-log-demo/issues">🐛 Report Bug</a> |
    <a href="https://github.com/bcperry/captains-log-demo/issues">💡 Request Feature</a>
  </p>
</div>
