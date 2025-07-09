# 🎤 Captain's Log - AI-Powered Audio Transcription & Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Azure](https://img.shields.io/badge/Azure-Speech%20%26%20OpenAI-blue)](https://azure.microsoft.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://www.python.org/)

A modern, AI-powered audio transcription and analysis application built with Streamlit, Azure Speech Services, and Azure OpenAI. Perfect for transcribing meetings, interviews, lectures, and other audio content with intelligent summarization and action item extraction.

## ✨ Features

- 🎯 **High-Quality Audio Transcription** - Powered by Azure Speech Services with support for multiple languages
- 🤖 **AI-Powered Analysis** - Intelligent summarization and action item extraction using Azure OpenAI
- ⏱️ **Flexible Duration Control** - Transcribe full audio or select specific time ranges
- 🔒 **Enterprise Security** - Built for Azure Government and Commercial clouds with managed identity support
- 📊 **Real-time Statistics** - Processing time, word count, and confidence metrics
- 💾 **Multiple Export Formats** - Download as TXT, JSON, or comprehensive analysis reports
- 🎨 **Modern Web Interface** - Clean, responsive UI built with Streamlit
- 🚀 **Easy Deployment** - Ready for Azure App Service with Docker support

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Azure subscription with Speech Services and OpenAI resources
- FFmpeg (for audio processing)

### Local Development

1. **Clone the repository**

2. **Install dependencies**
   ```bash
   pip install -r app/requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file in the `.azure/captainslog/` directory:
   ```env
   AZURE_SPEECH_KEY=your_speech_service_key
   AZURE_SPEECH_REGION=your_speech_region
   AZURE_SPEECH_ENDPOINT=your_speech_endpoint
   AZURE_OPENAI_ENDPOINT=your_openai_endpoint
   AZURE_OPENAI_KEY=your_openai_key
   AZURE_OPENAI_MODEL_NAME=gpt-4
   AZURE_OPENAI_API_VERSION=2024-02-01
   ```

4. **Run the application**
   ```bash
   cd app
   streamlit run app.py
   ```

5. **Open your browser** to `http://localhost:8501`

### Using Docker

1. **Build the Docker image**
   ```bash
   docker build -t captains-log .
   ```

2. **Run the container**
   ```bash
   docker run -p 8501:8501 --env-file .env captains-log
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
- Create necessary Azure resources (App Service, Speech Services, OpenAI)
- Deploy the application
- Configure environment variables
- Set up managed identity authentication

### Manual Azure Setup

1. **Create Azure Resources**
   - Speech Services resource
   - OpenAI resource
   - App Service or Container App
   - Key Vault (recommended for secrets)

2. **Configure Environment Variables**
   Set the required environment variables in your Azure App Service configuration.

3. **Deploy Application**
   Use the included Bicep templates or deploy directly via Azure CLI.

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
| `AZURE_SPEECH_ENDPOINT` | Speech service endpoint | Yes |
| `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint | Yes |
| `AZURE_OPENAI_KEY` | Azure OpenAI API key | Yes |
| `AZURE_OPENAI_MODEL_NAME` | Model name (e.g., gpt-4) | Yes |
| `AZURE_OPENAI_API_VERSION` | API version | Yes |

### Azure Government Support

The application automatically detects and configures for Azure Government clouds:
- Uses `*.speech.azure.us` endpoints
- Supports government-specific authentication
- Maintains compliance requirements


## 🔐 Security Features

- **Managed Identity** - Secure authentication without storing credentials
- **Environment Variables** - Sensitive data stored securely
- **Azure Key Vault** - Integration ready for enterprise secrets management
- **HTTPS Only** - Secure communication in production
- **No Data Persistence** - Audio files are processed in memory only

## 🐛 Troubleshooting

### Common Issues

1. **"No speech detected"**
   - Ensure audio contains clear speech
   - Check audio format compatibility

2. **Authentication errors**
   - Verify Azure Speech service key and region
   - Check OpenAI endpoint and API key
   - Ensure managed identity is properly configured

3. **Format issues**
   - Try converting audio to WAV format
   - Check if FFmpeg is properly installed
   - Ensure file size is within limits

4. **Long processing times**
   - Large files take more time to process
   - Consider using duration limits for testing
   - Check Azure service quotas

### Debug Mode

Enable debug logging by setting:
```env
STREAMLIT_LOGGER_LEVEL=DEBUG
```

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

- **Microsoft Azure** - For providing excellent Speech and OpenAI services
- **Streamlit** - For the amazing web framework
- **PyDub** - For audio processing capabilities
- **Open Source Community** - For the various libraries and tools used

---

<div align="center">
  <p>
    <a href="https://github.com/yourusername/captains-log-v2">⭐ Star this repo</a> |
    <a href="https://github.com/yourusername/captains-log-v2/issues">🐛 Report Bug</a> |
    <a href="https://github.com/yourusername/captains-log-v2/issues">💡 Request Feature</a>
  </p>
</div>
