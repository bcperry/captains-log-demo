# Azure Speech-to-Text Audio Transcription App

A modern Streamlit web application that transcribes audio files using Azure Cognitive Services Speech-to-Text API with secure authentication and multi-language support.

## 🌟 Features

- 🎤 **Multi-format Support**: Upload audio files in WAV, MP3, M4A, OGG, FLAC formats
- 🌍 **Multi-language**: Support for 10+ languages including English, Spanish, French, German, Chinese, Japanese
- 🔄 **Smart Conversion**: Automatic audio format optimization for best speech recognition
- 🔐 **Secure Authentication**: Azure Managed Identity for production, subscription key fallback
- 📊 **Rich Analytics**: Processing time, confidence scores, word/character counts
- 💾 **Export Options**: Download results as TXT or JSON with metadata
- 🎨 **Modern UI**: Responsive design with real-time progress indicators
- 📱 **Mobile-friendly**: Works on desktop and mobile devices

## 🚀 Quick Start

### Local Development

1. **Clone and Setup**
   ```bash
   cd app
   copy .env.example .env
   # Edit .env with your Azure Speech service details
   ```

2. **Install and Run**
   ```bash
   # Windows
   start.bat
   
   # Or manually
   pip install -r requirements.txt
   python run_local.py
   ```

3. **Access the App**
   - Open http://localhost:8501 in your browser

### Azure Deployment

Deploy to Azure using Azure Developer CLI:

```bash
# From project root
azd up
```

## 🔧 Configuration

### Required Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `AZURE_SPEECH_KEY` | Azure Speech service subscription key | `your-32-char-key-here` |
| `AZURE_SPEECH_REGION` | Azure region where service is deployed | `eastus` |

### Optional Variables

| Variable | Description | Usage |
|----------|-------------|-------|
| `AZURE_SPEECH_ENDPOINT` | Speech service endpoint URL | For managed identity auth |
| `AZURE_CLIENT_ID` | Managed identity client ID | Azure deployment |

## 🎯 Supported Languages

- **English**: US (`en-US`), UK (`en-GB`)
- **European**: Spanish (`es-ES`), French (`fr-FR`), German (`de-DE`), Italian (`it-IT`)
- **Portuguese**: Brazil (`pt-BR`)
- **Asian**: Chinese Mandarin (`zh-CN`), Japanese (`ja-JP`), Korean (`ko-KR`)

## 📁 Supported Audio Formats

| Format | Extension | Notes |
|--------|-----------|-------|
| WAV | `.wav` | Recommended - direct support |
| MP3 | `.mp3` | Auto-converted to WAV |
| M4A | `.m4a` | Auto-converted to WAV |
| OGG | `.ogg` | Auto-converted to WAV |
| FLAC | `.flac` | Auto-converted to WAV |

**Optimization**: All files are converted to 16kHz mono WAV for optimal speech recognition.

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit UI  │────│   Audio Processor │────│ Azure Speech AI │
│                 │    │                  │    │                 │
│ • File Upload   │    │ • Format Convert │    │ • Speech-to-Text│
│ • Language Sel  │    │ • Audio Optimize │    │ • Multi-language│
│ • Results View  │    │ • Temp File Mgmt │    │ • Confidence    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────────┐
                    │  Azure App Service  │
                    │                     │
                    │ • Managed Identity  │
                    │ • Auto-scaling      │
                    │ • HTTPS/Security    │
                    └─────────────────────┘
```

## 🔒 Security Features

- ✅ **Managed Identity**: No hardcoded credentials in production
- ✅ **Secure Token Handling**: Automatic token refresh
- ✅ **Temporary File Cleanup**: No data persistence on server
- ✅ **Input Validation**: File type and size restrictions
- ✅ **Error Handling**: Comprehensive logging and user feedback
- ✅ **HTTPS Only**: Encrypted data transmission

## ⚡ Performance Optimizations

- **Audio Processing**: Optimized conversion to 16kHz mono WAV
- **Memory Management**: Efficient temporary file handling
- **Connection Pooling**: Reused Azure service connections
- **Progress Indicators**: Real-time user feedback
- **Error Recovery**: Graceful handling of service failures

## 🛠️ Development

### Project Structure
```
app/
├── app.py      # Main application
├── requirements.txt      # Python dependencies
├── run_local.py         # Local development runner
├── start.bat           # Windows startup script
├── .streamlit/         # Streamlit configuration
│   └── config.toml
├── .env.example        # Environment template
└── README.md          # This file
```

### Adding New Languages

1. Add language to the `languages` dictionary in `app.py`
2. Use the correct Azure Speech service language code
3. Test with sample audio in that language

### Extending Audio Formats

1. Add format to `_convert_audio_format()` method
2. Update file uploader `type` parameter
3. Add conversion logic using `pydub`

## 🐛 Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| "No speech detected" | Audio quality, background noise | Use clear audio, reduce noise |
| "Authentication failed" | Wrong key/region | Check `AZURE_SPEECH_KEY` and `AZURE_SPEECH_REGION` |
| "Format not supported" | Unusual audio codec | Convert to WAV first |
| "Service quota exceeded" | API limits reached | Check Azure usage metrics |

### Debug Mode

Enable detailed logging:
```python
import logging
logging.getLogger().setLevel(logging.DEBUG)
```

### Testing Audio Quality

- **Recommended**: 16kHz sample rate, mono channel
- **Acceptable**: 8kHz to 48kHz, mono or stereo
- **Speech clarity**: Clear pronunciation, minimal background noise

## 📊 Usage Analytics

The app provides detailed transcription analytics:

- **Processing Time**: How long transcription took
- **Confidence Score**: Azure's confidence in the result (0-100%)
- **Character/Word Count**: Text statistics
- **Language Detection**: Verification of selected language

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Azure Speech Documentation**: [Microsoft Docs](https://docs.microsoft.com/azure/cognitive-services/speech-service/)
- **Streamlit Documentation**: [Streamlit Docs](https://docs.streamlit.io/)
- **Issues**: Open an issue on GitHub for bugs or feature requests
