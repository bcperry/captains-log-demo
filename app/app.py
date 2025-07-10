import streamlit as st
from azure.identity import DefaultAzureCredential
import tempfile
import os
from typing import Tuple, Dict, Any
import logging
from pydub import AudioSegment
from io import BytesIO
import time
import json
import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv
from openai import AzureOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "0.6.0"

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
        
        # Configure speech recognition settings for better accuracy
        self.speech_config.speech_recognition_language = "en-US"  # Set default language
        self.speech_config.enable_audio_logging = True  # Enable audio logging for debugging
        self.speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs, "5000")
        self.speech_config.set_property(speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs, "10000")
        self.speech_config.set_property(speechsdk.PropertyId.Speech_SegmentationSilenceTimeoutMs, "2000")
        
        print(f"Speech SDK configured with language: {self.speech_config.speech_recognition_language}")
        print(f"Endpoint silence timeout: 5000ms")
        print(f"Initial silence timeout: 10000ms")
    
    def __reduce__(self):
        return AudioTranscriber, (self.speech_key, self.speech_region, self.speech_endpoint)

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
        
        # Calculate number of chunks more accurately
        audio_length_ms = len(audio)
        num_chunks = (audio_length_ms + chunk_duration_ms - 1) // chunk_duration_ms  # Ceiling division
        
        print(f"Audio length: {audio_length_ms}ms ({audio_length_ms/1000:.1f}s)")
        print(f"Chunk duration: {chunk_duration_ms}ms ({chunk_duration_ms/1000}s)")
        print(f"Calculated chunks: {num_chunks}")
        
        chunks = []
        for i in range(num_chunks):
            start = i * chunk_duration_ms
            end = min((i + 1) * chunk_duration_ms, audio_length_ms)  # Don't exceed audio length
            chunk = audio[start:end]
            
            # Validate chunk has content and minimum duration
            chunk_duration_s = len(chunk) / 1000
            print(f"Chunk {i+1}: {start}ms-{end}ms ({chunk_duration_s:.1f}s)")
            
            # Only add chunks that have reasonable duration (at least 0.5 seconds)
            if len(chunk) >= 500:  # 0.5 seconds minimum
                chunks.append(chunk)
            else:
                print(f"Skipping chunk {i+1} - too short ({chunk_duration_s:.1f}s)")
        
        return chunks
    
    def transcribe_without_ui(self, filename: str, duration_minutes=None):
        """Transcribe audio without UI elements (for caching)"""
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
            
            # Validate chunk has audio content
            if len(chunk) < 500:  # Less than 0.5 seconds
                logger.warning(f"Chunk {i} is too short ({len(chunk)}ms), skipping")
                continue
                
            # Check if chunk has actual audio content (not just silence)
            if chunk.max_possible_amplitude > 0 and chunk.rms > 100:  # Has some audio content
                logger.info(f"Chunk {i} has audio content (RMS: {chunk.rms}, Max: {chunk.max_possible_amplitude})")
            else:
                logger.warning(f"Chunk {i} appears to be silent (RMS: {chunk.rms})")

            # Create a temporary file to store the audio chunk
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio_file:
                try:
                    # Export chunk with proper audio settings
                    chunk.export(
                        temp_audio_file.name, 
                        format="wav",
                        parameters=["-ac", "1", "-ar", "16000"]  # Mono, 16kHz for better speech recognition
                    )
                    temp_audio_file.flush()  # Ensure the file is written to disk
                    
                    # Configure audio input
                    audio_config = speechsdk.audio.AudioConfig(filename=temp_audio_file.name)
                    speech_recognizer = speechsdk.SpeechRecognizer(
                        speech_config=self.speech_config, 
                        audio_config=audio_config
                    )
                    
                    # Add timeout and retry logic
                    logger.info(f"Starting recognition for chunk {i}")
                    transcription = speech_recognizer.recognize_once()
                    
                    # Handle transcription results
                    text = ""
                    if transcription.reason == speechsdk.ResultReason.RecognizedSpeech:
                        text = transcription.text
                        logger.info(f"Chunk {i} transcribed successfully: {len(text)} characters")
                    elif transcription.reason == speechsdk.ResultReason.NoMatch:
                        logger.warning(f"No speech could be recognized in chunk {i}")
                        if hasattr(transcription, 'no_match_details'):
                            logger.warning(f"No match details: {transcription.no_match_details}")
                        text = ""
                    elif transcription.reason == speechsdk.ResultReason.Canceled:
                        cancellation_details = transcription.cancellation_details
                        logger.error(f"Speech Recognition canceled for chunk {i}: {cancellation_details.reason}")
                        if cancellation_details.reason == speechsdk.CancellationReason.Error:
                            logger.error(f"Error in chunk {i}: {cancellation_details.error_details}")
                        text = ""
                    else:
                        logger.warning(f"Unexpected transcription result for chunk {i}: {transcription.reason}")
                        text = ""
                    
                    # Add transcribed text to full transcription
                    if text:
                        print(f"Chunk {i}: {text}")
                        full_transcription += text + " "
                    else:
                        print(f"Chunk {i}: [No speech detected]")
                        
                except Exception as e:
                    logger.error(f"Error processing chunk {i}: {str(e)}")
                    continue
                finally:
                    # Clean up temporary file
                    try:
                        temp_audio_file.close()
                        # os.unlink(temp_audio_file.name)  # Uncomment to delete temp files
                    except Exception as e:
                        logger.warning(f"Failed to clean up temp file for chunk {i}: {str(e)}")
                        
        return full_transcription.strip()

   
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
    @st.cache_data(ttl=3600)
    def transcribe_audio_cached(self, audio_bytes, filename, language_code, duration_minutes=None):
        """Cached version of transcription without UI elements"""
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
                
                # Add audio diagnostics
                print(f"\n=== Transcription Debug Info ===")
                print(f"File: {filename}")
                print(f"Temp file: {temp_file_path}")
                print(f"Language: {language_code}")
                print(f"Duration limit: {duration_minutes} minutes")
                
                # Run audio diagnostics
                try:
                    self.diagnose_audio_chunks(temp_file_path, duration_minutes)
                except Exception as e:
                    print(f"Audio diagnostics failed: {e}")
                
                transcription = self.transcribe_without_ui(temp_file_path, duration_minutes)
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
            logger.error(f"Transcription failed: {str(e)}")
            return f"Transcription failed: {str(e)}", False, {}
    
    def transcribe_audio(self, audio_bytes, filename, language_code, duration_minutes=None):
        """Non-cached wrapper that handles UI elements"""
        return self.transcribe_audio_cached(audio_bytes, filename, language_code, duration_minutes)
    
    def get_audio_duration_minutes(self, input_file):
        """Get the duration of audio file in minutes"""
        
        return len(AudioSegment.from_file(input_file)) / 1000 / 60


class AzureOpenAISummarizer:
    """Class to handle Azure OpenAI interactions for text summarization and action item extraction."""
    
    def __init__(self):
        # Load environment variables
        load_dotenv('.azure/captainslog/.env')
        
        self.openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
        self.openai_api_key = os.getenv('AZURE_OPENAI_KEY')
        self.openai_model = os.getenv('AZURE_OPENAI_MODEL_NAME')
        self.api_version = os.getenv('AZURE_OPENAI_API_VERSION')
        
        # Use managed identity if no API key is provided (preferred for Azure-hosted apps)
        self.client = AzureOpenAI(
            api_key=self.openai_api_key,
            api_version=self.api_version,
            azure_endpoint=self.openai_endpoint
        )
        
        
        logger.info(f"Azure OpenAI configured with endpoint: {self.openai_endpoint}")
        logger.info(f"Using model: {self.openai_model}")

    
    @st.cache_data(ttl=3600)
    def summarize_transcription(_self, transcription_text: str) -> Dict[str, Any]:
        """
        Summarize the transcription text and extract action items.
        
        Args:
            transcription_text: The text to summarize
            
        Returns:
            Dictionary containing summary, action items, and metadata
        """
        try:
            system_prompt = """You are an AI assistant that specializes in analyzing meeting transcriptions and audio content. 
            Your task is to provide a comprehensive summary and extract actionable items.
            
            Please analyze the provided transcription and return a JSON response with the following structure:
            {
                "summary": "A concise summary of the main topics and discussions",
                "key_points": ["List of key points discussed"],
                "action_items": [
                    {
                        "task": "Description of the action item",
                        "assignee": "Person responsible (if mentioned)",
                        "deadline": "Deadline if mentioned, otherwise null",
                        "priority": "high|medium|low based on context"
                    }
                ],
                "participants": ["List of participants mentioned"],
                "topics": ["Main topics covered"],
                "sentiment": "overall sentiment of the discussion (positive|neutral|negative)",
                "confidence": "How confident you are in the analysis (0.0-1.0)"
            }
            
            If no action items are found, return an empty array. Be specific and accurate in your analysis."""
            
            user_prompt = f"""Please analyze this transcription and provide a summary with action items:

            Transcription:
            {transcription_text}"""
            
            response = _self.client.chat.completions.create(
                model=_self.openai_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
                            )
            
            # Parse the JSON response
            analysis_result = json.loads(response.choices[0].message.content)
            
            # Add metadata
            analysis_result['processing_metadata'] = {
                'model_used': _self.openai_model,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                'input_length': len(transcription_text),
                'tokens_used': response.usage.total_tokens if response.usage else None
            }
            
            return analysis_result
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenAI response as JSON: {e}")
            
            # Try to extract JSON from response that might have markdown formatting
            try:
                response_content = response.choices[0].message.content
                logger.info(f"Attempting to clean response content: {response_content[:200]}...")
                
                # Remove markdown code block formatting
                if "```json" in response_content:
                    # Extract content between ```json and ```
                    start_marker = "```json"
                    end_marker = "```"
                    start_idx = response_content.find(start_marker) + len(start_marker)
                    end_idx = response_content.find(end_marker, start_idx)
                    
                    if start_idx > len(start_marker) - 1 and end_idx > start_idx:
                        cleaned_content = response_content[start_idx:end_idx].strip()
                        analysis_result = json.loads(cleaned_content)
                        
                        # Add metadata
                        analysis_result['processing_metadata'] = {
                            'model_used': _self.openai_model,
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'input_length': len(transcription_text),
                            'tokens_used': response.usage.total_tokens if response.usage else None,
                            'cleaned_response': True
                        }
                        
                        logger.info("Successfully parsed cleaned JSON response")
                        return analysis_result
                        
            except (json.JSONDecodeError, IndexError, AttributeError) as cleanup_error:
                logger.error(f"Failed to clean and parse response: {cleanup_error}")
            
            return {
                "error": "Failed to parse AI response",
                "summary": response.choices[0].message.content,
                "action_items": [],
                "key_points": [],
                "confidence": 0.0
            }
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return {
                "error": str(e),
                "summary": f"Failed to analyze transcription: {str(e)}",
                "action_items": [],
                "key_points": [],
                "confidence": 0.0
            }
    
    def test_connection(self) -> Tuple[bool, str]:
        """Test the Azure OpenAI connection."""
        try:
            response = self.client.chat.completions.create(
                model=self.openai_model,
                messages=[{"role": "user", "content": "Hello, this is a test message."}],
                max_tokens=10
            )
            return True, "Azure OpenAI connection successful"
        except Exception as e:
            return False, f"Azure OpenAI connection failed: {str(e)}"
    

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Captain's Log - Audio Transcription & AI Analysis",
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
        <h1>🎤 Captain's Log - Audio Transcription & AI Analysis</h1>
        <p>Upload an audio file to get AI-powered transcription and intelligent analysis using Azure Speech and OpenAI</p>
    </div>
    """, unsafe_allow_html=True)
      # Initialize transcriber and summarizer
    if 'transcriber' not in st.session_state:
        st.session_state.transcriber = AudioTranscriber()
    
    if 'summarizer' not in st.session_state:
        try:
            st.session_state.summarizer = AzureOpenAISummarizer()
            st.session_state.summarizer_available = True
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI: {e}")
            st.session_state.summarizer_available = False
            st.session_state.summarizer_error = str(e)
    
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
        
        # Azure OpenAI status
        if st.session_state.summarizer_available:
            st.success("✅ Azure OpenAI Connected")
        else:
            st.error("❌ Azure OpenAI Not Available")
            if hasattr(st.session_state, 'summarizer_error'):
                st.error(f"Error: {st.session_state.summarizer_error}")
            
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
        col_test1, col_test2 = st.columns(2)
        with col_test1:
            st.write(f"Captain's Log Version: {VERSION}")
        with col_test2:
            if st.button("🤖 Test OpenAI", use_container_width=True):
                if st.session_state.summarizer_available:
                    success, message = st.session_state.summarizer.test_connection()
                    if success:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
                else:
                    st.error("❌ OpenAI not initialized")
        
    
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
                
                # Show progress during transcription
                progress_placeholder = st.empty()
                with progress_placeholder.container():
                    st.info("🎯 Starting transcription...")
                    
                    # Calculate expected chunks for progress
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file_path = temp_file.name
                        
                        try:
                            # Get audio chunks to estimate progress
                            if duration_param:
                                audio = st.session_state.transcriber.trim_audio(temp_file_path, duration_param)
                            else:
                                audio = AudioSegment.from_file(temp_file_path)
                            
                            chunk_duration_ms = 30 * 1000  # 30 seconds
                            expected_chunks = (len(audio) // chunk_duration_ms) + 1
                            if len(audio) == chunk_duration_ms:
                                expected_chunks = 1
                            
                            # Show progress bar
                            progress_bar = st.progress(0)
                            progress_text = st.empty()
                            
                            # Start transcription
                            start_time = time.time()
                            
                            def update_progress(current_chunk, total_chunks):
                                progress = current_chunk / total_chunks
                                progress_bar.progress(progress)
                                progress_text.text(f"Processing chunk {current_chunk}/{total_chunks}")
                            
                            # Store progress callback in session state for the transcriber to use
                            st.session_state.progress_callback = update_progress
                            st.session_state.expected_chunks = expected_chunks
                            
                            transcription, success, metadata = st.session_state.transcriber.transcribe_audio(
                                audio_bytes, uploaded_file.name, language_code, duration_param
                            )
                            
                            # Complete progress
                            progress_bar.progress(1.0)
                            progress_text.text("Transcription completed!")
                            
                        except Exception as e:
                            st.error(f"Error during transcription: {str(e)}")
                            transcription, success, metadata = f"Transcription failed: {str(e)}", False, {}
                        finally:
                            # Clean up temp file
                            # os.unlink(temp_file_path)
                            pass
                
                # Clear progress display after a moment
                time.sleep(1)
                progress_placeholder.empty()
                
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
            st.success("✅ Transcription completed successfully!")
              # Display transcription
            st.markdown("#### Transcribed Text:")
            transcription_text = st.text_area(
                "Result:",
                value=st.session_state.last_transcription,
                height=200,
                help="Copy this text to use elsewhere",
                label_visibility="collapsed"
            )
            
            # AI Analysis Section
            if st.session_state.summarizer_available:
                st.markdown("#### 🤖 AI Analysis")
                
                col_analyze, col_status = st.columns([1, 2])
                with col_analyze:
                    analyze_btn = st.button(
                        "📊 Analyze & Summarize",
                        type="secondary",
                        use_container_width=True,
                        help="Generate summary and extract action items using Azure OpenAI"
                    )
                
                with col_status:
                    if analyze_btn:
                        with st.spinner("🤖 Analyzing transcription with Azure OpenAI..."):
                            analysis_result = st.session_state.summarizer.summarize_transcription(
                                st.session_state.last_transcription
                            )
                            st.session_state.last_analysis = analysis_result
                
                # Display analysis results
                if hasattr(st.session_state, 'last_analysis') and st.session_state.last_analysis:
                    analysis = st.session_state.last_analysis
                    
                    if 'error' not in analysis:
                        # Summary
                        if 'summary' in analysis:
                            st.markdown("##### 📋 Summary")
                            st.info(analysis['summary'])
                        
                        # Key Points
                        if 'key_points' in analysis and analysis['key_points']:
                            st.markdown("##### 🔑 Key Points")
                            for i, point in enumerate(analysis['key_points'], 1):
                                st.markdown(f"• {point}")
                        
                        # Action Items
                        if 'action_items' in analysis and analysis['action_items']:
                            st.markdown("##### ✅ Action Items")
                            for i, item in enumerate(analysis['action_items'], 1):
                                with st.expander(f"Action {i}: {item.get('task', 'No description')[:50]}..."):
                                    st.write(f"**Task:** {item.get('task', 'Not specified')}")
                                    if item.get('assignee'):
                                        st.write(f"**Assignee:** {item['assignee']}")
                                    if item.get('deadline'):
                                        st.write(f"**Deadline:** {item['deadline']}")
                                    if item.get('priority'):
                                        priority_color = {
                                            'high': '🔴', 
                                            'medium': '🟡', 
                                            'low': '🟢'
                                        }.get(item['priority'].lower(), '⚪')
                                        st.write(f"**Priority:** {priority_color} {item['priority'].title()}")
                        
                        # Additional Information
                        col_info1, col_info2, col_info3 = st.columns(3)
                        
                        with col_info1:
                            if 'participants' in analysis and analysis['participants']:
                                st.markdown("##### 👥 Participants")
                                for participant in analysis['participants']:
                                    st.markdown(f"• {participant}")
                        
                        with col_info2:
                            if 'topics' in analysis and analysis['topics']:
                                st.markdown("##### 📚 Topics")
                                for topic in analysis['topics']:
                                    st.markdown(f"• {topic}")
                        
                        with col_info3:
                            if 'sentiment' in analysis:
                                sentiment_emoji = {
                                    'positive': '😊',
                                    'neutral': '😐',
                                    'negative': '😔'
                                }.get(analysis['sentiment'].lower(), '🤔')
                                st.markdown("##### 💭 Sentiment")
                                st.markdown(f"{sentiment_emoji} {analysis['sentiment'].title()}")
                            
                            if 'confidence' in analysis:
                                confidence_pct = round(float(analysis['confidence']) * 100, 1)
                                st.markdown("##### 🎯 AI Confidence")
                                st.markdown(f"{confidence_pct}%")
                    else:
                        st.error(f"❌ Analysis failed: {analysis.get('error', 'Unknown error')}")
            else:
                st.warning("🤖 Azure OpenAI not available for analysis")
              # Action buttons
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.download_button(
                    label="💾 Download TXT",
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
                    label="📊 Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"{os.path.splitext(st.session_state.last_filename)[0]}_transcription.json",
                    mime="application/json",
                    use_container_width=True
                )
            with col3:
                # Download analysis if available
                if hasattr(st.session_state, 'last_analysis') and st.session_state.last_analysis:
                    analysis_export = {
                        "transcription": st.session_state.last_transcription,
                        "analysis": st.session_state.last_analysis,
                        "metadata": st.session_state.last_metadata,
                        "filename": st.session_state.last_filename,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    st.download_button(
                        label="🤖 Download Analysis",
                        data=json.dumps(analysis_export, indent=2),
                        file_name=f"{os.path.splitext(st.session_state.last_filename)[0]}_analysis.json",
                        mime="application/json",
                        use_container_width=True
                    )
                else:
                    st.button("🤖 Analysis N/A", disabled=True, use_container_width=True)
            with col4:
                if st.button("🔄 Clear Results", use_container_width=True):
                    for key in ['last_transcription', 'last_success', 'last_metadata', 'last_filename', 'last_analysis']:
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
