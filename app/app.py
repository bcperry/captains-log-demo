import streamlit as st
import requests
from azure.identity import DefaultAzureCredential
import tempfile
import os
from typing import Optional, Tuple, Dict, Any
import logging
from pydub import AudioSegment
from io import BytesIO
import time
import json
import urllib.parse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioTranscriber:
    """Azure Speech-to-Text transcriber using REST APIs with Azure Government support."""
    
    def __init__(self):
        """Initialize the transcriber with Azure Speech service configuration."""
        self.region = None
        self.subscription_key = None
        self.token = None
        self.is_azure_gov = self._detect_azure_government()
        self.token_endpoint = None
        self.stt_endpoint = None
        self._setup_speech_config()
    
    def _detect_azure_government(self) -> bool:
        """Detect if running in Azure Government environment."""
        speech_endpoint = os.getenv("AZURE_SPEECH_ENDPOINT", "")
        speech_region = os.getenv("AZURE_SPEECH_REGION", "")
        return ("azure.us" in speech_endpoint.lower() or 
                speech_region.lower() in ["usgovarizona", "usgovvirginia"])
    
    def _setup_speech_config(self) -> None:
        """Setup Azure Speech configuration for Azure Government."""
        try:
            self.region = os.getenv("AZURE_SPEECH_REGION", "usgovvirginia")
            self.subscription_key = os.getenv("AZURE_SPEECH_KEY")
            speech_endpoint = os.getenv("AZURE_SPEECH_ENDPOINT")
            
            logger.info(f"Configuring Speech service for {'Azure Government' if self.is_azure_gov else 'Azure Commercial'}")
            
            # Map region identifiers according to Azure Government documentation
            region_mapping = {
                "usgovarizona": "usgovarizona",
                "usgovvirginia": "usgovvirginia"
            }
            
            if self.is_azure_gov:
                # Use Azure Government endpoints as specified in the documentation
                region_id = region_mapping.get(self.region.lower(), "usgovvirginia")
                
                # Token endpoint for Azure Government
                self.token_endpoint = f"https://{region_id}.api.cognitive.microsoft.us/sts/v1.0/issueToken"
                
                # Speech-to-text endpoint for Azure Government (short audio)
                self.stt_endpoint = f"https://{region_id}.stt.speech.azure.us/speech/recognition/conversation/cognitiveservices/v1"
                
            else:
                # Azure Commercial endpoints
                self.token_endpoint = f"https://{self.region}.api.cognitive.microsoft.com/sts/v1.0/issueToken"
                self.stt_endpoint = f"https://{self.region}.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1"
            
            # Try to get access token if no subscription key
            if not self.subscription_key:
                self._get_access_token()
            
            logger.info(f"Speech service configured - Region: {self.region}, Azure Gov: {self.is_azure_gov}")
            logger.info(f"Token endpoint: {self.token_endpoint}")
            logger.info(f"STT endpoint: {self.stt_endpoint}")
            
        except Exception as e:
            logger.error(f"Failed to setup speech configuration: {str(e)}")
            st.error(f"Failed to configure Azure Speech service: {str(e)}")

    def _get_access_token(self) -> bool:
        """Get access token using managed identity or issue token endpoint."""
        try:
            # Try managed identity first
            credential = DefaultAzureCredential()
            token_scope = "https://cognitiveservices.azure.us/.default" if self.is_azure_gov else "https://cognitiveservices.azure.com/.default"
            
            try:
                token_response = credential.get_token(token_scope)
                self.token = token_response.token
                logger.info("Successfully obtained access token via managed identity")
                return True
            except Exception as mi_error:
                logger.warning(f"Managed identity failed: {str(mi_error)}")
                
                # Fall back to subscription key token endpoint if available
                if self.subscription_key and self.token_endpoint:
                    return self._get_token_from_endpoint()
                
                return False
                
        except Exception as e:
            logger.error(f"Failed to get access token: {str(e)}")
            return False
    
    def _get_token_from_endpoint(self) -> bool:
        """Get access token from the token endpoint using subscription key."""
        try:
            headers = {
                'Ocp-Apim-Subscription-Key': self.subscription_key,
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            response = requests.post(self.token_endpoint, headers=headers, timeout=10)
            
            if response.status_code == 200:
                self.token = response.text
                logger.info("Successfully obtained access token from endpoint")
                return True
            else:
                logger.error(f"Token endpoint returned status: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to get token from endpoint: {str(e)}")
            return False

    def test_speech_service_connection(self) -> Tuple[bool, str]:
        """Test the Speech service connection."""
        if not self.stt_endpoint:
            return False, "Speech service not configured"
        
        try:
            headers = self._get_headers()
            if not headers:
                return False, "No valid authentication method available"
            
            # Test with a simple OPTIONS request to check if the endpoint is reachable
            # Remove Content-Type for OPTIONS request
            test_headers = {k: v for k, v in headers.items() if k != 'Content-Type'}
            
            response = requests.options(self.stt_endpoint, headers=test_headers, timeout=10)
            
            # Options request should return 200 or 204 for a valid endpoint
            if response.status_code in [200, 204]:
                return True, "Speech service connection successful"
            elif response.status_code == 401:
                return False, "Authentication failed - check your subscription key or token"
            elif response.status_code == 403:
                return False, "Access forbidden - check your permissions"
            else:
                return False, f"Service responded with status: {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return False, "Connection failed - check your network and endpoint URL"
        except requests.exceptions.Timeout:
            return False, "Connection timed out"
        except Exception as e:
            return False, f"Connection test failed: {str(e)}"

    def _get_headers(self) -> Optional[Dict[str, str]]:
        """Get appropriate headers for API requests."""
        if self.subscription_key:
            return {
                'Ocp-Apim-Subscription-Key': self.subscription_key,
                'Content-Type': 'audio/wav; codecs=audio/pcm; samplerate=16000',
                'Accept': 'application/json'
            }
        elif self.token:
            return {
                'Authorization': f'Bearer {self.token}',
                'Content-Type': 'audio/wav; codecs=audio/pcm; samplerate=16000',
                'Accept': 'application/json'
            }
        else:
            return None

    def _get_auth_headers(self) -> Optional[Dict[str, str]]:
        """Get authentication headers only (without content-type)."""
        if self.subscription_key:
            return {
                'Ocp-Apim-Subscription-Key': self.subscription_key
            }
        elif self.token:
            return {
                'Authorization': f'Bearer {self.token}'
            }
        else:
            return None

    def _convert_audio_format(self, audio_file: BytesIO, filename: str) -> Tuple[str, bool]:
        """Convert audio file to WAV format if necessary."""
        try:
            # Determine file format from extension
            file_extension = os.path.splitext(filename)[1].lower()
            
            if file_extension in ['.wav']:
                # Already in correct format
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                    temp_file.write(audio_file.read())
                    return temp_file.name, True
            
            # Convert to WAV using pydub
            audio_file.seek(0)
            
            # Load audio file
            if file_extension == '.mp3':
                audio = AudioSegment.from_mp3(audio_file)
            elif file_extension == '.m4a':
                audio = AudioSegment.from_file(audio_file, format='m4a')
            elif file_extension == '.ogg':
                audio = AudioSegment.from_ogg(audio_file)
            elif file_extension == '.flac':
                audio = AudioSegment.from_file(audio_file, format='flac')
            else:
                # Try to load as generic audio file
                audio = AudioSegment.from_file(audio_file)
            
            # Convert to WAV with optimal settings for speech recognition
            # 16kHz mono is optimal for Azure Speech service
            audio = audio.set_frame_rate(16000).set_channels(1)
            
            # Export to temporary WAV file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                audio.export(temp_file.name, format='wav')
                return temp_file.name, True
                
        except Exception as e:
            logger.error(f"Failed to convert audio file: {str(e)}")
            return "", False
    
    def transcribe_audio(self, audio_file: BytesIO, filename: str, language: str = "en-US") -> Tuple[Optional[str], bool, Dict[str, Any]]:
        """
        Transcribe audio file using Azure Speech-to-Text REST API.
        
        Args:
            audio_file: Audio file as BytesIO object
            filename: Original filename for format detection
            language: Language code for speech recognition
            
        Returns:
            Tuple of (transcription_text, success_flag, metadata)
        """
        if not self.stt_endpoint:
            return "Azure Speech service not configured properly", False, {}
        
        temp_file_path = None
        start_time = time.time()
        
        try:
            # Convert audio to proper format
            temp_file_path, conversion_success = self._convert_audio_format(audio_file, filename)
            if not conversion_success:
                return "Failed to process audio file format", False, {}
            
            # Get authentication headers
            auth_headers = self._get_auth_headers()
            if not auth_headers:
                return "No valid authentication method available", False, {}
            
            # Set up complete headers for the transcription request
            headers = auth_headers.copy()
            headers.update({
                'Content-Type': 'audio/wav; codecs=audio/pcm; samplerate=16000',
                'Accept': 'application/json'
            })
            
            # Set up query parameters for the Speech-to-Text API
            params = {
                'language': language,
                'format': 'detailed'
            }
            
            # Log the request details for debugging
            logger.info(f"Transcription URL: {self.stt_endpoint}")
            logger.info(f"Language: {language}")
            logger.info(f"Using Azure Government: {self.is_azure_gov}")
            
            # Read the audio file
            with open(temp_file_path, 'rb') as audio_data:
                audio_content = audio_data.read()
            
            # Perform transcription request with retry logic
            with st.spinner("🎤 Transcribing audio... This may take a moment."):
                response = self._make_request_with_retry(
                    self.stt_endpoint,
                    headers=headers,
                    params=params,
                    data=audio_content,
                    timeout=120
                )
            
            processing_time = time.time() - start_time
            
            if response and response.status_code == 200:
                result_json = response.json()
                
                # Extract text from the synchronous response
                transcription = ""
                confidence_score = None
                
                # Try different response formats that Azure Speech API might return
                if 'DisplayText' in result_json:
                    # Simple response format
                    transcription = result_json['DisplayText']
                    confidence_score = result_json.get('Confidence', None)
                elif 'NBest' in result_json and result_json['NBest']:
                    # N-Best response format
                    best_result = result_json['NBest'][0]
                    transcription = best_result.get('Display', '')
                    confidence_score = best_result.get('Confidence', None)
                elif 'RecognitionStatus' in result_json:
                    # Check if recognition was successful
                    if result_json['RecognitionStatus'] == 'Success':
                        transcription = result_json.get('DisplayText', '')
                        confidence_score = result_json.get('Confidence', None)
                    else:
                        error_msg = f"Recognition failed: {result_json['RecognitionStatus']}"
                        logger.error(error_msg)
                        return error_msg, False, {'processing_time': round(processing_time, 2)}
                
                if transcription:
                    final_processing_time = time.time() - start_time
                    
                    metadata = {
                        'processing_time': round(final_processing_time, 2),
                        'language': language,
                        'confidence_score': confidence_score,
                        'file_format': os.path.splitext(filename)[1].lower(),
                        'characters': len(transcription),
                        'words': len(transcription.split()) if transcription else 0,
                        'azure_government': self.is_azure_gov,
                        'region': self.region
                    }
                    
                    logger.info("Audio transcription successful")
                    return transcription, True, metadata
                else:
                    return "No speech could be recognized in the audio file. Please check if the audio contains clear speech.", False, {'processing_time': round(processing_time, 2)}
                    
            elif response:
                error_msg = f"API request failed with status {response.status_code}"
                try:
                    error_details = response.json()
                    if 'error' in error_details:
                        error_msg += f": {error_details['error'].get('message', 'Unknown error')}"
                    elif 'message' in error_details:
                        error_msg += f": {error_details['message']}"
                except:
                    error_msg += f": {response.text[:200]}"  # Limit error text length
                
                # Provide helpful error suggestions
                if response.status_code == 401:
                    error_msg += "\n\n💡 Tip: Check your Azure Speech service key or managed identity configuration."
                elif response.status_code == 403:
                    error_msg += "\n\n💡 Tip: Access denied - verify your Azure Speech service permissions."
                elif response.status_code == 429:
                    error_msg += "\n\n💡 Tip: Rate limit exceeded - try again in a moment."
                elif response.status_code == 400:
                    error_msg += "\n\n💡 Tip: Bad request - check audio format and language settings."
                
                logger.error(error_msg)
                return error_msg, False, {'processing_time': processing_time}
            else:
                return "Failed to get response from Speech service", False, {'processing_time': processing_time}
                
        except requests.exceptions.RequestException as e:
            processing_time = time.time() - start_time
            error_msg = f"Network error during transcription: {str(e)}"
            logger.error(error_msg)
            return error_msg, False, {'processing_time': processing_time}
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Transcription failed: {str(e)}"
            logger.error(error_msg)
            return error_msg, False, {'processing_time': processing_time}
        
        finally:
            # Clean up temporary file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Failed to clean up temporary file: {str(e)}")

    def _make_request_with_retry(self, url: str, headers: Dict[str, str], params: Dict[str, str], 
                                 data: bytes, timeout: int, max_retries: int = 3) -> Optional[requests.Response]:
        """Make HTTP request with exponential backoff retry logic."""
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, params=params, data=data, timeout=timeout)
                
                # If successful or client error (4xx), don't retry
                if response.status_code < 500:
                    return response
                
                # Server error (5xx), retry with backoff
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2  # Exponential backoff: 2, 4, 8 seconds
                    logger.warning(f"Request failed with status {response.status_code}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    return response
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2
                    logger.warning(f"Request timed out, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error("Request timed out after all retries")
                    return None
                    
            except requests.exceptions.ConnectionError as e:
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 2
                    logger.warning(f"Connection error: {str(e)}, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Connection failed after all retries: {str(e)}")
                    return None
        
        return None

    def _poll_transcription_job(self, job_url: str, headers: dict, max_wait_time: int = 300) -> Optional[dict]:
        """Poll the transcription job until completion."""
        start_time = time.time()
        poll_interval = 5  # seconds
        
        while time.time() - start_time < max_wait_time:
            try:
                response = requests.get(job_url, headers=headers, timeout=30)
                
                if response.status_code == 200:
                    job_status = response.json()
                    status = job_status.get('status')
                    
                    if status == 'Succeeded':
                        # Get the transcription results
                        files_url = job_status.get('links', {}).get('files')
                        if files_url:
                            return self._get_transcription_results(files_url, headers)
                    elif status == 'Failed':
                        logger.error(f"Transcription job failed: {job_status.get('properties', {}).get('error', 'Unknown error')}")
                        return None
                    elif status in ['Running', 'NotStarted']:
                        # Still processing, wait and try again
                        time.sleep(poll_interval)
                        continue
                else:
                    logger.error(f"Failed to poll job status: {response.status_code}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error polling transcription job: {str(e)}")
                return None
        
        logger.error("Transcription job timed out")
        return None

    def _get_transcription_results(self, files_url: str, headers: dict) -> Optional[dict]:
        """Get the actual transcription text from the completed job."""
        try:
            response = requests.get(files_url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                files_data = response.json()
                
                # Find the transcription file
                for file_info in files_data.get('values', []):
                    if file_info.get('kind') == 'Transcription':
                        content_url = file_info.get('links', {}).get('contentUrl')
                        if content_url:
                            # Download the actual transcription
                            transcription_response = requests.get(content_url, headers=headers, timeout=30)
                            if transcription_response.status_code == 200:
                                transcription_data = transcription_response.json()
                                
                                # Extract the combined text
                                combined_phrases = transcription_data.get('combinedRecognizedPhrases', [])
                                if combined_phrases:
                                    text = combined_phrases[0].get('display', '')
                                    confidence = combined_phrases[0].get('confidence', None)
                                    return {'text': text, 'confidence': confidence}
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting transcription results: {str(e)}")
            return None

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Audio Transcription App",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #0078d4 0%, #00bcf2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #0078d4;
    }
    .success-card {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🎤 Audio Transcription with Azure Speech-to-Text</h1>
        <p>Upload an audio file to get an AI-powered transcription using Azure Cognitive Services</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize transcriber
    if 'transcriber' not in st.session_state:
        st.session_state.transcriber = AudioTranscriber()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Language selection
        languages = {
            "English (US)": "en-US",
            "English (UK)": "en-GB", 
            "Spanish": "es-ES",
            "French": "fr-FR",
            "German": "de-DE",
            "Italian": "it-IT",
            "Portuguese": "pt-BR",
            "Chinese (Mandarin)": "zh-CN",
            "Japanese": "ja-JP",
            "Korean": "ko-KR"
        }
        
        selected_language = st.selectbox(
            "Speech Language",
            options=list(languages.keys()),
            index=0,
            help="Select the primary language spoken in your audio file"
        )
        
        st.markdown("---")
        
        # Service status
        st.subheader("🔗 Service Status")
        if st.session_state.transcriber.stt_endpoint:
            st.success("✅ Azure Speech Service Connected")
            # Show Azure Government status
            if st.session_state.transcriber.is_azure_gov:
                st.info("🏛️ Azure Government Cloud")
            else:
                st.info("☁️ Azure Commercial Cloud")
        else:
            st.error("❌ Azure Speech Service Not Configured")
            
        # Environment info
        speech_region = st.session_state.transcriber.region or "Not Set"
        st.info(f"Region: {speech_region}")
        
        if st.session_state.transcriber.stt_endpoint:
            # Show masked endpoint for security
            endpoint = st.session_state.transcriber.stt_endpoint
            if len(endpoint) > 30:
                masked_endpoint = endpoint[:25] + "..." + endpoint[-15:]
            else:
                masked_endpoint = endpoint
            st.info(f"Endpoint: {masked_endpoint}")
        
        # Test connection button
        if st.button("🔍 Test Connection", use_container_width=True):
            success, message = st.session_state.transcriber.test_speech_service_connection()
            if success:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
        
        st.markdown("---")
        st.markdown("""
        **Supported Formats:**
        - WAV (recommended)
        - MP3, M4A, OGG, FLAC
        - Max file size: 100MB
        - Optimal: 16kHz mono
        """)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📁 Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'ogg', 'flac'],
            help="Upload an audio file for transcription"
        )
        
        if uploaded_file is not None:
            # File information
            st.markdown("#### 📊 File Information")
            col_name, col_size, col_type = st.columns(3)
            with col_name:
                st.metric("File Name", uploaded_file.name)
            with col_size:
                st.metric("File Size", f"{uploaded_file.size / 1024 / 1024:.2f} MB")
            with col_type:
                st.metric("File Type", uploaded_file.type or "Unknown")
            
            # Audio player
            st.markdown("#### 🔊 Audio Preview")
            st.audio(uploaded_file, format=uploaded_file.type)
            
            # Transcription controls
            st.markdown("#### 🚀 Transcription")
            
            col_btn, col_lang = st.columns([1, 2])
            with col_btn:
                transcribe_btn = st.button(
                    "🎯 Start Transcription", 
                    type="primary",
                    use_container_width=True
                )
            with col_lang:
                st.info(f"Language: {selected_language}")
            
            if transcribe_btn:
                # Create BytesIO object from uploaded file
                audio_bytes = BytesIO(uploaded_file.read())
                language_code = languages[selected_language]
                
                # Perform transcription
                transcription, success, metadata = st.session_state.transcriber.transcribe_audio(
                    audio_bytes, uploaded_file.name, language_code
                )
                
                # Store results in session state
                st.session_state.last_transcription = transcription
                st.session_state.last_success = success
                st.session_state.last_metadata = metadata
                st.session_state.last_filename = uploaded_file.name
    
    with col2:
        st.markdown("### 📈 Quick Stats")
        if hasattr(st.session_state, 'last_metadata') and st.session_state.last_metadata:
            metadata = st.session_state.last_metadata
            
            if 'processing_time' in metadata:
                st.metric("⏱️ Processing Time", f"{metadata['processing_time']}s")
            if 'characters' in metadata:
                st.metric("📝 Characters", metadata['characters'])
            if 'words' in metadata:
                st.metric("🔤 Words", metadata['words'])
            if 'confidence_score' in metadata and metadata['confidence_score']:
                confidence_pct = round(metadata['confidence_score'] * 100, 1)
                st.metric("🎯 Confidence", f"{confidence_pct}%")
        else:
            st.info("Upload and transcribe an audio file to see statistics")
    
    # Results section
    if hasattr(st.session_state, 'last_transcription'):
        st.markdown("---")
        st.markdown("### 📝 Transcription Results")
        
        if st.session_state.last_success:
            st.markdown('<div class="success-card">', unsafe_allow_html=True)
            st.success("✅ Transcription completed successfully!")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display transcription
            st.markdown("#### Transcribed Text:")
            transcription_text = st.text_area(
                "Result:",
                value=st.session_state.last_transcription,
                height=200,
                help="Copy this text to use elsewhere",
                label_visibility="collapsed"
            )
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    label="💾 Download as TXT",
                    data=st.session_state.last_transcription,
                    file_name=f"{os.path.splitext(st.session_state.last_filename)[0]}_transcription.txt",
                    mime="text/plain",
                    use_container_width=True
                )
            with col2:
                # Create JSON export with metadata
                export_data = {
                    "transcription": st.session_state.last_transcription,
                    "metadata": st.session_state.last_metadata,
                    "filename": st.session_state.last_filename,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                st.download_button(
                    label="📊 Download as JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"{os.path.splitext(st.session_state.last_filename)[0]}_transcription.json",
                    mime="application/json",
                    use_container_width=True
                )
            with col3:
                if st.button("🔄 Clear Results", use_container_width=True):
                    for key in ['last_transcription', 'last_success', 'last_metadata', 'last_filename']:
                        if hasattr(st.session_state, key):
                            delattr(st.session_state, key)
                    st.rerun()
        else:
            st.error("❌ Transcription failed!")
            st.error(st.session_state.last_transcription)  # Show error message
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <h4>🔧 Troubleshooting Tips</h4>
        <ul style='text-align: left; max-width: 600px; margin: 0 auto;'>
            <li><strong>No speech detected:</strong> Ensure audio contains clear speech</li>
            <li><strong>Authentication errors:</strong> Check Azure Speech service key and region</li>
            <li><strong>Format issues:</strong> Try converting to WAV format first</li>
            <li><strong>Long processing:</strong> Larger files take more time to process</li>
        </ul>
        <br>
        <small>Built with Streamlit and Azure Cognitive Services | Secure authentication with managed identity</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
