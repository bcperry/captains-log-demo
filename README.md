# 🎤 Captain's Log - AI-Powered Audio Transcription & Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Azure](https://img.shields.io/badge/Azure-Speech%20%26%20OpenAI-blue)](https://azure.microsoft.com/)
[![React](https://img.shields.io/badge/React-19+-blue)](https://react.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128+-green)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://www.python.org/)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

A modern, AI-powered audio transcription and analysis application built with React, FastAPI, Azure Speech Services, and Azure OpenAI. Perfect for transcribing meetings, interviews, lectures, and other audio content with intelligent summarization and action item extraction.

## ✨ Features

- 🎯 **High-Quality Audio Transcription** - Powered by Azure Speech Services with support for multiple languages
- 🤖 **AI-Powered Analysis** - Intelligent summarization and action item extraction using Azure OpenAI
- ⏱️ **Flexible Duration Control** - Transcribe full audio or select specific time ranges
- 🔒 **Enterprise Security** - Azure Entra ID authentication with Azure Government support
- 📊 **Real-time Statistics** - Processing time, word count, and confidence metrics
- 💾 **Multiple Export Formats** - Download as TXT, JSON, or comprehensive analysis reports
- 🎨 **Modern Web Interface** - Clean, responsive React UI with Tailwind CSS
- 🚀 **Easy Deployment** - Ready for Azure Container Apps with Docker support

## 🏗️ Architecture

- **Frontend**: React 19 + TypeScript + Vite + Tailwind CSS
- **Backend**: FastAPI + Python 3.11+
- **Authentication**: Azure Entra ID (MSAL)
- **Services**: Azure Speech Services, Azure OpenAI, Azure Cosmos DB

## 🚀 Quick Start

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

8. **Open your browser** to `http://localhost:3000`

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
