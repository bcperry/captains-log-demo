"""Captain's Log - Streamlit Frontend Application.

This Streamlit application serves as the main user interface for Captain's Log,
integrating with the FastAPI backend via REST API calls.

Features:
- User authentication flow with Azure Entra ID
- Audio file upload interface for transcription
- Display transcription results with speaker diarization
- Transcription history view and navigation
- Responsive layout for desktop and tablet
"""

import json
import os
from datetime import datetime
from typing import Any

import streamlit as st

from streamlit_client import APIClient, APIError

VERSION = "1.0.0"


def get_api_client() -> APIClient:
    """Get or create the API client from session state.

    Returns:
        APIClient instance with current access token
    """
    # Get backend URL from environment or default
    backend_url = os.getenv(
        "BACKEND_API_URL",
        os.getenv("API_BASE_URL", "http://localhost:8001"),
    )

    if "api_client" not in st.session_state:
        st.session_state.api_client = APIClient(base_url=backend_url)

    # Update access token if present
    if "access_token" in st.session_state and st.session_state.access_token:
        st.session_state.api_client.set_access_token(st.session_state.access_token)

    client: APIClient = st.session_state.api_client
    return client


def format_datetime(dt_str: str) -> str:
    """Format a datetime string for display.

    Args:
        dt_str: ISO format datetime string

    Returns:
        Human-readable datetime string
    """
    try:
        dt = datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return dt_str


def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string (e.g., "2.5 MB")
    """
    size_float = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size_float < 1024.0:
            return f"{size_float:.1f} {unit}"
        size_float /= 1024.0
    return f"{size_float:.1f} TB"


def check_backend_health() -> dict[str, Any]:
    """Check backend API health status.

    Returns:
        Health status dict or error dict
    """
    try:
        client = get_api_client()
        return client.get_ready()
    except APIError as e:
        return {"status": "error", "detail": e.detail}
    except Exception as e:
        return {"status": "error", "detail": str(e)}


def render_login_section() -> None:
    """Render the login/authentication section."""
    st.sidebar.markdown("## 🔐 Authentication")

    if "access_token" in st.session_state and st.session_state.access_token:
        # User is authenticated
        st.sidebar.success("✅ Authenticated")

        # Show user profile if available
        if "user_profile" in st.session_state:
            profile = st.session_state.user_profile
            st.sidebar.markdown(f"**{profile.get('name', 'User')}**")
            st.sidebar.markdown(f"_{profile.get('email', '')}_")

        if st.sidebar.button("🚪 Logout", use_container_width=True):
            st.session_state.access_token = None
            st.session_state.pop("user_profile", None)
            st.rerun()
    else:
        # User is not authenticated - show token input
        st.sidebar.info(
            "Enter your Azure Entra ID access token to authenticate. "
            "You can obtain a token using Azure CLI or the Azure Portal."
        )

        with st.sidebar.form("login_form"):
            token = st.text_area(
                "Access Token",
                height=100,
                placeholder="eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI...",
                help="Paste your Azure Entra ID JWT access token",
            )
            submit = st.form_submit_button("🔑 Authenticate", use_container_width=True)

            if submit and token:
                st.session_state.access_token = token.strip()
                # Try to fetch user profile to validate token
                try:
                    client = get_api_client()
                    profile = client.get_user_profile()
                    st.session_state.user_profile = profile
                    st.rerun()
                except APIError as e:
                    st.session_state.access_token = None
                    st.error(f"Authentication failed: {e.detail}")


def render_health_status() -> None:
    """Render backend health status in sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📊 Backend Status")

    health = check_backend_health()

    if health.get("status") == "ready":
        st.sidebar.success("✅ Backend Ready")

        # Show dependency status (list of dicts with name, healthy, message)
        deps = health.get("dependencies", [])
        for dep in deps:
            name = dep.get("name", "Unknown")
            if dep.get("healthy"):
                st.sidebar.markdown(f"✅ {name.replace('_', ' ').title()}")
            else:
                msg = dep.get("message", "Unavailable")
                st.sidebar.markdown(f"⚠️ {name.replace('_', ' ').title()}: {msg}")
    else:
        st.sidebar.error("❌ Backend Unavailable")
        detail = health.get("detail", "Cannot connect to backend")
        st.sidebar.markdown(f"_{detail}_")


def render_transcription_page() -> None:
    """Render the main transcription page."""
    st.markdown("## 🎤 Audio Transcription")

    # Check authentication
    if not st.session_state.get("access_token"):
        st.warning("⚠️ Please authenticate to use transcription features.")
        return

    # Transcription settings
    col1, col2 = st.columns([2, 1])

    with col1:
        # Language selection
        languages = {
            "English (US)": "en-US",
            "English (UK)": "en-GB",
            "Spanish": "es-ES",
            "French": "fr-FR",
            "German": "de-DE",
            "Italian": "it-IT",
            "Portuguese": "pt-BR",
            "Japanese": "ja-JP",
            "Chinese (Mandarin)": "zh-CN",
        }
        selected_language = st.selectbox(
            "🌍 Transcription Language",
            options=list(languages.keys()),
            index=0,
        )
        language_code = languages[selected_language]

    with col2:
        # Diarization option
        enable_diarization = st.checkbox(
            "👥 Enable Speaker Diarization",
            value=False,
            help="Identify different speakers in the audio",
        )

        if enable_diarization:
            max_speakers = st.slider(
                "Max Speakers",
                min_value=1,
                max_value=10,
                value=5,
                help="Maximum number of speakers to identify",
            )
        else:
            max_speakers = 5

    # File upload
    st.markdown("### 📁 Upload Audio File")

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a"],
        help="Supported formats: WAV, MP3, M4A. Maximum size: 25 MB",
    )

    if uploaded_file:
        # Show file info
        file_size = len(uploaded_file.getvalue())
        st.info(
            f"📄 **{uploaded_file.name}** | "
            f"Size: {format_file_size(file_size)} | "
            f"Type: {uploaded_file.type or 'Unknown'}"
        )

        # Transcription button
        col_btn, col_store = st.columns([1, 1])

        with col_btn:
            transcribe_clicked = st.button(
                "🚀 Start Transcription",
                type="primary",
                use_container_width=True,
            )

        with col_store:
            store_result = st.checkbox(
                "💾 Save to History",
                value=True,
                help="Store transcription result in your history",
            )

        if transcribe_clicked:
            with st.spinner("🔄 Transcribing audio..."):
                try:
                    client = get_api_client()
                    file_content = uploaded_file.getvalue()

                    if enable_diarization:
                        result = client.transcribe_audio_with_diarization(
                            file_content=file_content,
                            filename=uploaded_file.name,
                            language=language_code,
                            max_speakers=max_speakers,
                        )
                        st.session_state.last_transcription = result
                        st.session_state.last_transcription_type = "diarized"
                    else:
                        result = client.transcribe_audio(
                            file_content=file_content,
                            filename=uploaded_file.name,
                            language=language_code,
                            store=store_result,
                        )
                        st.session_state.last_transcription = result
                        st.session_state.last_transcription_type = "simple"

                    st.success("✅ Transcription completed!")

                except APIError as e:
                    st.error(f"❌ Transcription failed: {e.detail}")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # Display results
    if "last_transcription" in st.session_state:
        st.markdown("---")
        st.markdown("### 📝 Transcription Result")

        result = st.session_state.last_transcription
        trans_type = st.session_state.get("last_transcription_type", "simple")

        if trans_type == "diarized":
            # Display diarized transcription
            render_diarized_result(result)
        else:
            # Display simple transcription
            render_simple_result(result)


def render_simple_result(result: dict[str, Any]) -> None:
    """Render simple transcription result.

    Args:
        result: Transcription response from API
    """
    # Metadata
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📝 Characters", len(result.get("text", "")))
    with col2:
        st.metric("🔤 Words", len(result.get("text", "").split()))
    with col3:
        st.metric("📊 Format", result.get("audio_format", "").upper())

    # Transcribed text
    text = result.get("text", "")
    st.text_area(
        "Transcribed Text",
        value=text,
        height=200,
        label_visibility="collapsed",
    )

    # Download options
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="💾 Download TXT",
            data=text,
            file_name="transcription.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            label="📊 Download JSON",
            data=json.dumps(result, indent=2, default=str),
            file_name="transcription.json",
            mime="application/json",
            use_container_width=True,
        )


def render_diarized_result(result: dict[str, Any]) -> None:
    """Render diarized transcription result with speaker segments.

    Args:
        result: Diarized transcription response from API
    """
    # Metadata
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("👥 Speakers", result.get("speaker_count", 0))
    with col2:
        st.metric("📝 Segments", len(result.get("segments", [])))
    with col3:
        st.metric("🔤 Words", len(result.get("full_text", "").split()))
    with col4:
        st.metric("📊 Format", result.get("audio_format", "").upper())

    # Speaker segments with colors
    st.markdown("#### Speaker Segments")

    speaker_colors = {
        "Speaker_1": "🔵",
        "Speaker_2": "🟢",
        "Speaker_3": "🟠",
        "Speaker_4": "🟣",
        "Speaker_5": "🔴",
    }

    segments = result.get("segments", [])
    for segment in segments:
        speaker_id = segment.get("speaker_id", "Unknown")
        text = segment.get("text", "")
        start_ms = segment.get("start_time_ms", 0)
        end_ms = segment.get("end_time_ms", 0)

        # Format timestamp
        start_sec = start_ms / 1000.0
        end_sec = end_ms / 1000.0
        timestamp = f"[{start_sec:.1f}s - {end_sec:.1f}s]"

        # Get speaker icon
        icon = speaker_colors.get(speaker_id, "⚪")

        st.markdown(f"**{icon} {speaker_id}** {timestamp}")
        st.markdown(f"> {text}")
        st.markdown("")

    # Full text
    st.markdown("#### Full Transcription")
    full_text = result.get("full_text", "")
    st.text_area(
        "Full Transcription",
        value=full_text,
        height=150,
        label_visibility="collapsed",
    )

    # Download options
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="💾 Download TXT",
            data=full_text,
            file_name="transcription_diarized.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            label="📊 Download JSON",
            data=json.dumps(result, indent=2, default=str),
            file_name="transcription_diarized.json",
            mime="application/json",
            use_container_width=True,
        )


def render_history_page() -> None:
    """Render the transcription history page."""
    st.markdown("## 📚 Transcription History")

    # Check authentication
    if not st.session_state.get("access_token"):
        st.warning("⚠️ Please authenticate to view your transcription history.")
        return

    # Pagination
    page = st.session_state.get("history_page", 1)
    per_page = 10

    # Fetch transcriptions
    try:
        client = get_api_client()
        response = client.list_transcriptions(page=page, per_page=per_page)

        transcriptions = response.get("transcriptions", [])
        total = response.get("total", 0)
        total_pages = (total + per_page - 1) // per_page

        if not transcriptions:
            st.info("📭 No transcriptions yet. Upload an audio file to get started!")
            return

        st.markdown(f"**Showing {len(transcriptions)} of {total} transcriptions**")

        # Transcription list
        for trans in transcriptions:
            with st.expander(
                f"📝 {trans.get('id', 'Unknown')[:8]}... | "
                f"{format_datetime(trans.get('created_at', ''))} | "
                f"{trans.get('language', 'en-US')}"
            ):
                # Transcription details
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown(f"**Format:** {trans.get('audio_format', '').upper()}")
                with col2:
                    st.markdown(f"**Size:** {format_file_size(trans.get('file_size_bytes', 0))}")
                with col3:
                    if trans.get("has_diarization"):
                        st.markdown(f"**Speakers:** {trans.get('speaker_count', 0)}")
                    else:
                        st.markdown("**Type:** Simple")

                # Text preview
                text = trans.get("text", "")
                preview = text[:500] + "..." if len(text) > 500 else text
                st.text_area(
                    "Text",
                    value=preview,
                    height=100,
                    label_visibility="collapsed",
                    key=f"trans_{trans.get('id')}",
                )

                # Actions
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="💾 Download",
                        data=json.dumps(trans, indent=2, default=str),
                        file_name=f"transcription_{trans.get('id', 'unknown')}.json",
                        mime="application/json",
                        use_container_width=True,
                        key=f"download_{trans.get('id')}",
                    )
                with col2:
                    if st.button(
                        "🗑️ Delete",
                        use_container_width=True,
                        key=f"delete_{trans.get('id')}",
                    ):
                        try:
                            client.delete_transcription(trans.get("id", ""))
                            st.success("Deleted!")
                            st.rerun()
                        except APIError as e:
                            st.error(f"Delete failed: {e.detail}")

        # Pagination controls
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if page > 1:
                if st.button("⬅️ Previous"):
                    st.session_state.history_page = page - 1
                    st.rerun()
        with col2:
            st.markdown(f"**Page {page} of {max(1, total_pages)}**")
        with col3:
            if page < total_pages:
                if st.button("Next ➡️"):
                    st.session_state.history_page = page + 1
                    st.rerun()

    except APIError as e:
        st.error(f"❌ Failed to load history: {e.detail}")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")


def render_settings_page() -> None:
    """Render the user settings page."""
    st.markdown("## ⚙️ Settings")

    # Check authentication
    if not st.session_state.get("access_token"):
        st.warning("⚠️ Please authenticate to access settings.")
        return

    # User profile
    st.markdown("### 👤 User Profile")

    if "user_profile" in st.session_state:
        profile = st.session_state.user_profile

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Name:** {profile.get('name', 'Not set')}")
            st.markdown(f"**Email:** {profile.get('email', 'Not set')}")
        with col2:
            st.markdown(f"**Username:** {profile.get('preferred_username', 'Not set')}")
            created = profile.get("created_at", "")
            st.markdown(f"**Member since:** {format_datetime(created)}")

        # Preferences
        st.markdown("### 🎨 Preferences")

        prefs = profile.get("preferences", {})

        theme = st.selectbox(
            "Theme",
            options=["light", "dark"],
            index=0 if prefs.get("theme", "light") == "light" else 1,
        )

        default_lang = st.selectbox(
            "Default Language",
            options=["en-US", "en-GB", "es-ES", "fr-FR", "de-DE"],
            index=0,
        )

        auto_save = st.checkbox(
            "Auto-save transcriptions",
            value=prefs.get("auto_save_transcriptions", True),
        )

        if st.button("💾 Save Preferences", type="primary"):
            try:
                client = get_api_client()
                new_prefs = {
                    "theme": theme,
                    "default_language": default_lang,
                    "auto_save_transcriptions": auto_save,
                }
                updated = client.update_user_preferences(new_prefs)
                st.session_state.user_profile = updated
                st.success("✅ Preferences saved!")
            except APIError as e:
                st.error(f"❌ Failed to save: {e.detail}")
    else:
        st.info("Loading user profile...")
        try:
            client = get_api_client()
            profile = client.get_user_profile()
            st.session_state.user_profile = profile
            st.rerun()
        except APIError as e:
            st.error(f"❌ Failed to load profile: {e.detail}")


def main() -> None:
    """Main Streamlit application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="Captain's Log",
        page_icon="🎙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Header
    st.title("🎙️ Captain's Log")
    st.markdown(
        "AI-Powered Audio Transcription using Azure Speech Services • "
        f"v{VERSION}"
    )

    # Sidebar - Authentication
    render_login_section()

    # Sidebar - Backend health
    render_health_status()

    # Sidebar - Navigation
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 📍 Navigation")

    page = st.sidebar.radio(
        "Go to",
        options=["🎤 Transcribe", "📚 History", "⚙️ Settings"],
        label_visibility="collapsed",
    )

    # Main content
    st.markdown("---")

    if page == "🎤 Transcribe":
        render_transcription_page()
    elif page == "📚 History":
        render_history_page()
    elif page == "⚙️ Settings":
        render_settings_page()

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <small>
                Built with Streamlit and Azure Cognitive Services |
                Secure authentication with Azure Entra ID
            </small>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
