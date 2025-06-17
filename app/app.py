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
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioTranscriber:
    def __init__(self):
        # Load environment variables
        load_dotenv('.azure/captainslog/.env')
        
        self.speech_key = os.getenv('AZURE_SPEECH_KEY')
        self.speech_region = os.getenv('AZURE_SPEECH_REGION')
        self.speech_endpoint = os.getenv('AZURE_SPEECH_ENDPOINT')
        
        # Print out the environment variables for debugging
        print(f"AZURE_SPEECH_KEY: {self.speech_key}")
        print(f"AZURE_SPEECH_REGION: {self.speech_region}")
        print(f"AZURE_SPEECH_ENDPOINT: {self.speech_endpoint}")
        
        
        # Set up Azure Government endpoint
        self.stt_endpoint = f"wss://{self.speech_region}.stt.speech.azure.us"
        self.is_azure_gov = True

        print(f"Using Azure Government endpoint: {self.stt_endpoint}")
        
        # Configure speech SDK for Azure Government
        self.speech_config = speechsdk.SpeechConfig(
            endpoint=self.stt_endpoint, 
            subscription=self.speech_key
        )
        # self.speech_config.speech_recognition_language = "en-US"
    
    def trim_audio(self, input_file, duration_minutes):
        """Trim audio file to specified duration in minutes"""
        audio = AudioSegment.from_file(input_file)
        duration_ms = duration_minutes * 60 * 1000  # Convert minutes to milliseconds
        
        # If requested duration is longer than the audio, return the full audio
        if duration_ms >= len(audio):
            return audio
        
        # Trim audio to the specified duration
        trimmed_audio = audio[:duration_ms]
        return trimmed_audio
    
    def split_audio(self, input_file, duration_minutes=None):
        if duration_minutes:
            # Trim the audio first if duration is specified
            audio = self.trim_audio(input_file, duration_minutes)
        else:
            audio = AudioSegment.from_file(input_file)
        
        chunk_duration_ms = 30 * 1000  # 30 seconds in milliseconds
        num_chunks = (len(audio) // chunk_duration_ms) + 1
        if len(audio) == chunk_duration_ms:
            num_chunks = 1
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_duration_ms
            end = (i + 1) * chunk_duration_ms
            chunk = audio[start:end]
            chunks.append(chunk)
        return chunks
    
    def transcribe(self, filename: str, duration_minutes=None):
        # Split audio file into chunks (with optional duration limit)
        audio_chunks = self.split_audio(filename, duration_minutes)
        print("Number of audio chunks: {}".format(len(audio_chunks)))
        print(f"Transcribing {len(audio_chunks)} audio chunks from {filename}")
        
        if duration_minutes:
            print(f"Transcribing first {duration_minutes} minutes of audio")
        else:
            print(f"Audio duration: {len(AudioSegment.from_file(filename)) / 1000} seconds")

        full_transcription = ""
        # Transcribe each chunk
        for i, chunk in enumerate(audio_chunks, 1):
            logger.info(f"Transcribing chunk {i}/{len(audio_chunks)}")

            # Create a temporary file to store the audio chunk
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                chunk.export(temp_audio_file.name, format="wav")
                temp_audio_file.flush()  # Ensure the file is written to disk
                audio_config = speechsdk.audio.AudioConfig(filename=temp_audio_file.name)
                speech_recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=audio_config)
                
                transcription = speech_recognizer.recognize_once()
                if transcription.reason == speechsdk.ResultReason.RecognizedSpeech:
                    text = transcription.text
                    st.toast(f"Chunk {i} completed.")
                elif transcription.reason == speechsdk.ResultReason.NoMatch:
                    st.toast(f"No speech could be recognized in chunk {i}: {transcription.no_match_details}")
                elif transcription.reason == speechsdk.ResultReason.Canceled:
                    cancellation_details = transcription.cancellation_details
                    st.toast(f"Speech Recognition canceled for chunk {i}: {cancellation_details.reason}")
                    if cancellation_details.reason == speechsdk.CancellationReason.Error:
                        st.toast(f"Error in chunk {i}: {cancellation_details.error_details}")

            # return transcription
                if isinstance(transcription, dict):
                    text = transcription['text']
                else:
                    text = transcription.text
                print(text)
                full_transcription = full_transcription + text
                
            # Close and Delete the temporary audio file
            temp_audio_file.close()
            # os.unlink(temp_audio_file.name)
        return full_transcription
    
    @property
    def region(self):
        return self.speech_region
    
    def test_speech_service_connection(self):
        try:
            # Simple test by creating a recognizer
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
            speech_recognizer = speechsdk.SpeechRecognizer(
                speech_config=self.speech_config, 
                audio_config=audio_config
            )
            return True, "Connection successful"
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def transcribe_audio(self, audio_bytes, filename, language_code, duration_minutes=None):
        try:
            # Update language
            self.speech_config.speech_recognition_language = language_code
            
            # Save audio bytes to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                audio_bytes.seek(0)
                temp_file.write(audio_bytes.getvalue())
                temp_file_path = temp_file.name
            
                start_time = time.time()
                logger.info(f"Starting transcription for {filename} ({temp_file_path}) in {language_code}")
                if duration_minutes:
                    logger.info(f"Transcribing first {duration_minutes} minutes of audio")
                transcription = self.transcribe(temp_file_path, duration_minutes)
                processing_time = round(time.time() - start_time, 2)
            
            # Clean up temp file
            # os.unlink(temp_file_path)
              # Create metadata
            metadata = {
                'processing_time': processing_time,
                'characters': len(transcription),
                'words': len(transcription.split()) if transcription else 0,
            }
            
            # Add duration information if it was limited
            if duration_minutes:
                metadata['transcribed_duration_minutes'] = duration_minutes
            
            temp_file.close()  # Close the temp file
            return transcription, True, metadata
            
        except Exception as e:
            return f"Transcription failed: {str(e)}", False, {}
    
    def get_audio_duration_minutes(self, input_file):
        """Get the duration of audio file in minutes"""
        audio = AudioSegment.from_file(input_file)
        duration_seconds = len(audio) / 1000
        return duration_seconds / 60


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
            type=['wav', 'mp3', 'm4a', 'ogg', 'flac', 'mp4'],
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
            
            # Audio duration controls
            st.markdown("#### ⏱️ Duration Settings")
            
            # Get audio duration for the slider
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_file_path = temp_file.name
                
                try:
                    audio_duration_minutes = st.session_state.transcriber.get_audio_duration_minutes(temp_file_path)
                    
                    col_duration_info, col_duration_control = st.columns([1, 2])
                    
                    with col_duration_info:
                        st.metric("Total Duration", f"{audio_duration_minutes:.1f} min")
                    
                    with col_duration_control:
                        # Duration slider
                        if audio_duration_minutes > 1:
                            duration_to_transcribe = st.slider(
                                "Duration to transcribe (minutes)",
                                min_value=1.0,
                                max_value=float(audio_duration_minutes),
                                value=min(10.0, float(audio_duration_minutes)),  # Default to 10 minutes or less
                                step=0.5,
                                help="Select how many minutes from the beginning of the audio to transcribe"
                            )
                        else:
                            duration_to_transcribe = audio_duration_minutes
                            st.info(f"Audio is {audio_duration_minutes:.1f} minutes - will transcribe full duration")
                    
                    # Show what will be transcribed
                    if duration_to_transcribe < audio_duration_minutes:
                        st.info(f"📝 Will transcribe the first {duration_to_transcribe:.1f} minutes of {audio_duration_minutes:.1f} minutes total")
                    else:
                        st.info(f"📝 Will transcribe the entire audio file ({audio_duration_minutes:.1f} minutes)")
                    
                except Exception as e:
                    st.warning(f"Could not determine audio duration: {str(e)}")
                    duration_to_transcribe = None
                
                # Clean up temp file
                # os.unlink(temp_file_path)
            
            # Transcription controls
            st.markdown("#### 🚀 Transcription")
            
            col_btn, col_lang = st.columns([1, 2])
            with col_btn:
                transcribe_btn = st.button(
                    "🎯 Start Transcription", 
                    type="primary",
                    use_container_width=True                )
            with col_lang:
                st.info(f"Language: {selected_language}")
            
            if transcribe_btn:
                # Create BytesIO object from uploaded file
                audio_bytes = BytesIO(uploaded_file.read())
                language_code = languages[selected_language]
                  # Perform transcription with duration limit if specified
                duration_param = None
                if 'duration_to_transcribe' in locals() and duration_to_transcribe < audio_duration_minutes:
                    duration_param = duration_to_transcribe
                
                transcription, success, metadata = st.session_state.transcriber.transcribe_audio(
                    audio_bytes, uploaded_file.name, language_code, duration_param
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
            if 'transcribed_duration_minutes' in metadata:
                st.metric("🎵 Transcribed Duration", f"{metadata['transcribed_duration_minutes']:.1f} min")
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
