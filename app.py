import sys
import os
import time

# Proje ana dizinini Python yoluna ekler
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

try:
    from src.preprocessing import preprocess_audio, preprocess_audio_array
    from src.features import extract_pitch, extract_spectral_features
    from src.evaluation import (
        calculate_rtf,
        calculate_snr,
        calculate_energy_ratio,
        calculate_pitch_deviation,
    )
    from src.transformation import (
        change_gender_age,
        change_emotion,
        convert_speaker,
        generate_singing_voice,
    )
    from src.utils import audio_to_bytes, apply_spectral_tilt
except ImportError as e:
    st.error(f"Modül yükleme hatası: {e}")

try:
    from audio_recorder_streamlit import audio_recorder
    _mic_available = True
except ImportError:
    _mic_available = False

st.set_page_config(
    page_title="VoiceMorph Pro",
    page_icon="〰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = True  # True = Dark, False = Light

if "audio" not in st.session_state:
    st.session_state.audio = None
    st.session_state.sr = None
    st.session_state.processed_audio = None

if "gen_pitch" not in st.session_state:
    st.session_state.gen_pitch = 0
if "gen_formant" not in st.session_state:
    st.session_state.gen_formant = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────
is_dark = st.session_state.theme_mode

theme = {
    "bg": "#0b1020" if is_dark else "#f6f8fb",
    "surface": "#111827" if is_dark else "#ffffff",
    "surface_2": "#162033" if is_dark else "#f1f5f9",
    "surface_3": "#0f172a" if is_dark else "#eaf0f7",
    "text": "#f8fafc" if is_dark else "#0f172a",
    "muted": "#9ca3af" if is_dark else "#64748b",
    "border": "#243044" if is_dark else "#dbe3ee",
    "accent": "#3b82f6",
    "accent_2": "#22c55e",
    "danger": "#ef4444",
    "warning": "#f59e0b",
    "shadow": "0 18px 45px rgba(0,0,0,0.28)" if is_dark else "0 18px 45px rgba(15,23,42,0.10)",
    "plot_bg": "#0b1020" if is_dark else "#ffffff",
}

st.markdown(
    f"""
<style>
    :root {{
        --app-bg: {theme['bg']};
        --surface: {theme['surface']};
        --surface-2: {theme['surface_2']};
        --surface-3: {theme['surface_3']};
        --text: {theme['text']};
        --muted: {theme['muted']};
        --border: {theme['border']};
        --accent: {theme['accent']};
        --accent-2: {theme['accent_2']};
        --danger: {theme['danger']};
        --warning: {theme['warning']};
        --shadow: {theme['shadow']};
    }}

    * {{
        transition: background-color 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }}

    [data-testid="stAppViewContainer"] {{
        background:
            radial-gradient(circle at top left, rgba(59,130,246,0.16), transparent 34rem),
            radial-gradient(circle at top right, rgba(34,197,94,0.10), transparent 30rem),
            var(--app-bg);
        color: var(--text);
    }}

    [data-testid="stHeader"] {{
        background: transparent;
    }}

    [data-testid="stSidebar"] {{
        background: var(--surface);
        border-right: 1px solid var(--border);
    }}

    [data-testid="stSidebar"] > div:first-child {{
        padding-top: 1.2rem;
    }}

    .block-container {{
        padding-top: 1.4rem;
        padding-bottom: 3rem;
        max-width: 1440px;
    }}

    h1, h2, h3, h4, h5, h6, label, .stMarkdown {{
        color: var(--text);
    }}

    p, span, small, .stCaption, [data-testid="stMarkdownContainer"] p {{
        color: var(--muted);
    }}

    .hero-card {{
        background: linear-gradient(135deg, rgba(59,130,246,0.16), rgba(34,197,94,0.08)), var(--surface);
        border: 1.5px solid var(--border);
        border-radius: 28px;
        padding: 1.6rem 1.8rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.1rem;
    }}

    .brand-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
        flex-wrap: wrap;
    }}

    .brand-title {{
        display: flex;
        align-items: center;
        gap: 0.8rem;
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        color: var(--text);
    }}

    .brand-icon {{
        width: 48px;
        height: 48px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 16px;
        background: linear-gradient(135deg, var(--accent), #8b5cf6);
        color: #fff;
        box-shadow: 0 12px 28px rgba(59,130,246,0.26);
        font-size: 1.6rem;
        font-weight: 700;
    }}

    .brand-subtitle {{
        margin-top: 0.5rem;
        color: var(--muted);
        font-size: 1rem;
        max-width: 780px;
        line-height: 1.5;
    }}

    .steps {{
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
    }}

    .step-pill {{
        padding: 0.55rem 0.9rem;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.035);
        color: var(--muted);
        font-weight: 700;
        font-size: 0.8rem;
        transition: all 0.3s ease;
    }}

    .step-pill.active {{
        color: #fff;
        background: var(--accent);
        border-color: var(--accent);
        box-shadow: 0 8px 16px rgba(59,130,246,0.3);
    }}

    .section-card {{
        background: var(--surface);
        border: 1.2px solid var(--border);
        border-radius: 24px;
        padding: 1.3rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.2rem;
        transition: all 0.3s ease;
    }}

    .section-card:hover {{
        border-color: var(--accent);
        box-shadow: 0 20px 50px rgba(59,130,246,0.12);
    }}

    .section-title {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 0.8rem;
        margin-bottom: 0.9rem;
    }}

    .section-title h3 {{
        font-size: 0.95rem;
        margin: 0;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        color: var(--text);
        font-weight: 800;
    }}

    .badge {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        border-radius: 999px;
        padding: 0.3rem 0.65rem;
        font-size: 0.74rem;
        font-weight: 800;
        border: 1px solid var(--border);
        color: var(--muted);
        background: var(--surface-2);
        text-transform: uppercase;
    }}

    .badge.on {{
        color: #ecfdf5;
        background: rgba(34,197,94,0.20);
        border-color: rgba(34,197,94,0.45);
    }}

    .metric-grid {{
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.8rem;
        margin-top: 0.9rem;
    }}

    .metric-card {{
        background: var(--surface-2);
        border: 1.2px solid var(--border);
        border-radius: 18px;
        padding: 0.9rem;
        transition: all 0.3s ease;
    }}

    .metric-card:hover {{
        border-color: var(--accent);
        background: var(--surface-3);
    }}

    .metric-label {{
        color: var(--muted);
        font-size: 0.7rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    .metric-value {{
        color: var(--text);
        font-size: 1.1rem;
        font-weight: 800;
        margin-top: 0.3rem;
    }}

    .module-status-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        border: 1.2px solid var(--border);
        background: var(--surface-2);
        border-radius: 16px;
        padding: 0.7rem 0.9rem;
        margin-bottom: 0.6rem;
        font-size: 0.88rem;
        font-weight: 700;
        transition: all 0.3s ease;
    }}

    .module-status-row:hover {{
        border-color: var(--accent);
        background: var(--surface-3);
    }}

    .module-status-row strong {{
        color: var(--text);
    }}

    .status-on {{ color: var(--accent-2); font-weight: 900; }}
    .status-off {{ color: var(--muted); font-weight: 800; }}

    .empty-state {{
        min-height: 330px;
        border: 2px dashed var(--border);
        background: var(--surface-2);
        border-radius: 24px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 2.2rem;
        color: var(--muted);
    }}

    .empty-state-icon {{
        width: 64px;
        height: 64px;
        border-radius: 20px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(59,130,246,0.15);
        color: var(--accent);
        font-size: 2.2rem;
        margin-bottom: 1rem;
    }}

    .stExpander {{
        border: 1.2px solid var(--border) !important;
        border-radius: 18px !important;
        background: var(--surface) !important;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
        overflow: hidden;
        margin-bottom: 1rem !important;
    }}

    .stExpander summary {{
        font-weight: 800;
        color: var(--text);
        background: var(--surface-2);
        transition: all 0.3s ease;
        padding: 1rem !important;
        border-radius: 16px !important;
    }}

    .stExpander summary:hover {{
        background: var(--surface-3);
    }}

    div[data-testid="stFileUploader"] section {{
        border: 1.5px dashed var(--border);
        background: var(--surface-2);
        border-radius: 18px;
        transition: all 0.3s ease;
    }}

    div[data-testid="stFileUploader"] section:hover {{
        border-color: var(--accent);
        background: var(--surface-3);
    }}

    .stButton > button, .stDownloadButton > button {{
        border-radius: 14px !important;
        border: 1px solid var(--border) !important;
        font-weight: 800 !important;
        min-height: 2.8rem;
        transition: all 0.3s ease !important;
    }}

    .stButton > button:hover:not(:disabled), .stDownloadButton > button:hover:not(:disabled) {{
        border-color: var(--accent) !important;
        transform: translateY(-2px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15) !important;
    }}

    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, var(--accent), #2563eb) !important;
        border: none !important;
        box-shadow: 0 16px 28px rgba(59,130,246,0.24) !important;
        color: white !important;
    }}

    .stButton > button[kind="primary"]:hover {{
        box-shadow: 0 20px 40px rgba(59,130,246,0.32) !important;
    }}

    [data-testid="stAudio"] {{
        border-radius: 16px;
        overflow: hidden;
        border: 1.2px solid var(--border);
        background: var(--surface-2);
        padding: 0.8rem;
    }}

    input[type="text"], input[type="number"], textarea, select {{
        background: var(--surface-2) !important;
        border: 1.2px solid var(--border) !important;
        color: var(--text) !important;
        border-radius: 12px !important;
        padding: 0.7rem 0.9rem !important;
        transition: all 0.3s ease !important;
    }}

    input[type="text"]:focus, input[type="number"]:focus, textarea:focus, select:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.15) !important;
    }}

    input[type="checkbox"], input[type="radio"] {{
        accent-color: var(--accent) !important;
    }}

    .stSlider [data-baseweb="slider"] {{
        border-radius: 10px;
    }}

    .stToggle {{
        padding: 0.5rem 0;
    }}

    /* Microphone Icon Styling - FIX */
    [data-testid="stAudioRecorderUI"] {{
        background: var(--surface-2) !important;
        border: 1.2px solid var(--border) !important;
        border-radius: 18px !important;
        padding: 1rem !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
    }}

    [data-testid="stAudioRecorderUI"] button {{
        background: var(--surface) !important;
        border: 1.2px solid var(--border) !important;
        border-radius: 14px !important;
        transition: all 0.3s ease !important;
        min-height: 3rem;
    }}

    [data-testid="stAudioRecorderUI"] button:hover {{
        background: var(--surface-3) !important;
        border-color: var(--accent) !important;
        transform: scale(1.05);
    }}

    [data-testid="stAudioRecorderUI"] button svg {{
        color: var(--accent) !important;
    }}

    [data-testid="stAudioRecorderUI"] button[aria-label*="icone"] {{
        background: var(--accent) !important;
        border: none !important;
        color: white !important;
    }}

    [data-testid="stAudioRecorderUI"] button[aria-label*="icone"]:hover {{
        background: #2563eb !important;
        box-shadow: 0 12px 24px rgba(59,130,246,0.3) !important;
    }}

    /* Additional microphone icon fix for various selectors */
    .AudioRecorder {{
        background: var(--surface-2) !important;
    }}

    .AudioRecorder button {{
        background: var(--surface) !important;
        border: 1.2px solid var(--border) !important;
        color: var(--text) !important;
    }}

    .AudioRecorder button:hover {{
        background: var(--surface-3) !important;
        border-color: var(--accent) !important;
    }}

    /* Target the microphone icon itself */
    [role="button"] svg path {{
        fill: var(--accent) !important;
    }}

    button[class*="audio"] {{
        background: var(--surface) !important;
        border: 1.2px solid var(--border) !important;
    }}

    button[class*="mic"] {{
        background: var(--surface) !important;
        border: 1.2px solid var(--border) !important;
    }}

    hr {{ 
        border-color: var(--border);
        margin: 1.2rem 0;
    }}

    .stCaption {{
        font-size: 0.82rem;
        line-height: 1.4;
    }}

    label {{
        font-weight: 700 !important;
        font-size: 0.9rem !important;
    }}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def update_gen_preset():
    p = st.session_state.preset_gen
    if p == "Male to Female":
        st.session_state.gen_pitch = 5
        st.session_state.gen_formant = 1.15
    elif p == "Female to Male":
        st.session_state.gen_pitch = -12
        st.session_state.gen_formant = 1.00
    elif p == "Adult to Child":
        st.session_state.gen_pitch = 9
        st.session_state.gen_formant = 1.07
    else:
        st.session_state.gen_pitch = 0
        st.session_state.gen_formant = 1.00


def plot_waveform(y, sr, color, height=(8, 1.15)):
    fig, ax = plt.subplots(figsize=height)
    fig.patch.set_alpha(0)
    ax.set_facecolor(theme["plot_bg"])
    librosa.display.waveshow(y, sr=sr, color=color, ax=ax)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def module_badge(enabled):
    return "<span class='badge on'>● ON</span>" if enabled else "<span class='badge'>○ OFF</span>"

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="hero-card">
  <div class="brand-row">
    <div>
      <div class="brand-title">
        <span class="brand-icon">〰</span>
        <span>VoiceMorph <span style="color:{theme['accent']};">Pro</span></span>
      </div>
      <div class="brand-subtitle">
        Speech warping, voice conversion, noise reduction and singing voice generation in one modular DSP pipeline.
      </div>
    </div>
    <div class="steps">
      <span class="step-pill active">1 · Upload</span>
      <span class="step-pill">2 · Configure</span>
      <span class="step-pill">3 · Generate</span>
      <span class="step-pill">4 · Export</span>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# Theme toggle stays visible and unchanged in behavior
_, theme_col = st.columns([7, 1.4])
with theme_col:
    st.markdown(
        f"""
        <div style="display: flex; justify-content: flex-end; margin-top: -0.5rem;">
        """,
        unsafe_allow_html=True,
    )
    st.toggle("🌙 Dark", key="theme_mode")
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 〰 VoiceMorph Pro")
    st.caption("Configure input and transformation modules")
    st.markdown("---")

    st.markdown("#### 📁 Audio Input")
    noise_reduce = st.toggle(
        "🔇 Noise Reduction",
        value=False,
        help="Spectral Gating suppresses all types of noise, including sudden bursts of noise.",
    )

    if noise_reduce:
        noise_strength = st.slider(
            "Noise strength",
            0.1,
            1.0,
            0.75,
            0.05,
            help="Higher = more aggressive reduction",
        )
        stationary_noise = st.toggle(
            "Stationary noise mode",
            value=False,
            help="Off: sudden noises | On: constant noises like fan/hum",
        )
    else:
        noise_strength = 0.75
        stationary_noise = False

    uploaded_file = st.file_uploader("Load audio file", type=["wav", "mp3", "flac"])
    if uploaded_file is not None:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        audio, sr = preprocess_audio(
            "temp_audio.wav",
            noise_reduce=noise_reduce,
            noise_strength=noise_strength,
            stationary_noise=stationary_noise,
        )
        st.session_state.audio = audio
        st.session_state.sr = sr
        st.success("✓ Audio loaded")

    # ─ Professional Microphone Recording Interface ─
    if _mic_available:
        st.markdown(
            """
            <style>
                .mic-card {
                    background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(34,197,94,0.08));
                    border: 1.2px solid rgba(59,130,246,0.3);
                    border-radius: 20px;
                    padding: 1.8rem;
                    margin-bottom: 1.2rem;
                }
                
                .mic-header {
                    font-weight: 800;
                    font-size: 0.95rem;
                    color: var(--text);
                    margin-bottom: 1.2rem;
                    text-align: center;
                }
                
                .mic-visual {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 1.2rem;
                    flex-wrap: wrap;
                }
                
                .mic-icon-circle {
                    width: 88px;
                    height: 88px;
                    background: linear-gradient(135deg, var(--accent), #2563eb);
                    border-radius: 22px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 2.6rem;
                    color: white;
                    box-shadow: 0 14px 32px rgba(59,130,246,0.28);
                }
                
                .mic-info {
                    flex: 1;
                    min-width: 200px;
                }
                
                .mic-info-title {
                    color: var(--text);
                    font-weight: 700;
                    margin-bottom: 0.5rem;
                }
                
                .mic-info-text {
                    color: var(--muted);
                    font-size: 0.88rem;
                    line-height: 1.5;
                }
            </style>
            <div class="mic-card">
                <div class="mic-header">🎤 Voice Recording</div>
                <div class="mic-visual">
                    <div class="mic-icon-circle">🎙️</div>
                    <div class="mic-info">
                        <div class="mic-info-title">Ready to Record</div>
                        <div class="mic-info-text">Click the button below to start recording your voice.</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        mic_bytes = audio_recorder(
            text="🎙️ Click to Record",
            recording_color="#ef4444",
            neutral_color="#3b82f6",
            icon_name="microphone",
            icon_size="3x",
            pause_threshold=3.0,
            sample_rate=16000,
        )
        
        # Clean minimal styling for the recorder button
        st.markdown(
            """
            <script>
                // Hide the text "Click to Record" but keep the button
                setTimeout(function() {
                    const recorder = document.querySelector('[data-testid="stAudioRecorderUI"]');
                    if (recorder) {
                        // Hide all non-button children
                        recorder.querySelectorAll(':not(button)').forEach(el => {
                            if (el.tagName !== 'STYLE') el.style.display = 'none';
                        });
                        
                        // Style the button
                        const btn = recorder.querySelector('button');
                        if (btn) {
                            btn.innerHTML = '🎙️ Start Recording';
                            btn.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
                            btn.style.color = 'white';
                            btn.style.border = 'none';
                            btn.style.padding = '1rem 1.8rem';
                            btn.style.borderRadius = '14px';
                            btn.style.fontWeight = '800';
                            btn.style.fontSize = '1rem';
                            btn.style.boxShadow = '0 12px 28px rgba(239,68,68,0.28)';
                            btn.style.cursor = 'pointer';
                        }
                    }
                }, 300);
            </script>
            <style>
                div[data-testid="stAudioRecorderUI"] {
                    background: transparent !important;
                    border: none !important;
                    padding: 1.5rem 0 !important;
                    display: flex !important;
                    justify-content: center !important;
                    align-items: center !important;
                }
                
                div[data-testid="stAudioRecorderUI"] > * {
                    background: transparent !important;
                    border: none !important;
                }
                
                div[data-testid="stAudioRecorderUI"] button {
                    background: linear-gradient(135deg, #ef4444, #dc2626) !important;
                    border: none !important;
                    padding: 1rem 1.8rem !important;
                    border-radius: 14px !important;
                    color: white !important;
                    font-weight: 800 !important;
                    font-size: 1rem !important;
                    box-shadow: 0 12px 28px rgba(239,68,68,0.28) !important;
                    transition: all 0.3s ease !important;
                    cursor: pointer !important;
                }
                
                div[data-testid="stAudioRecorderUI"] button:hover {
                    transform: translateY(-3px) scale(1.02) !important;
                    box-shadow: 0 16px 40px rgba(239,68,68,0.36) !important;
                }
                
                div[data-testid="stAudioRecorderUI"] button:active {
                    transform: translateY(-1px) !important;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )
        
        if mic_bytes and len(mic_bytes) > 1000:
            with open("temp_mic.wav", "wb") as f:
                f.write(mic_bytes)
            try:
                audio_mic, sr_mic = preprocess_audio(
                    "temp_mic.wav",
                    noise_reduce=noise_reduce,
                    noise_strength=noise_strength,
                    stationary_noise=stationary_noise,
                )
                st.session_state.audio = audio_mic
                st.session_state.sr = sr_mic
                st.success("✓ Microphone recording is ready")
            except Exception as exc:
                st.error(f"✗ The record could not be processed: {exc}")
    else:
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(249,115,22,0.08));
                border: 1.2px solid rgba(239,68,68,0.3);
                border-radius: 20px;
                padding: 1.5rem;
                text-align: center;
                margin-bottom: 1.2rem;
            ">
                <div style="color: var(--text); font-weight: 800; margin-bottom: 0.5rem;">🎤 Microphone Not Available</div>
                <div style="color: var(--muted); font-size: 0.9rem; line-height: 1.5;">
                    Install <code style="background: var(--surface-2); padding: 0.3rem 0.6rem; border-radius: 8px; color: #ef4444;">audio-recorder-streamlit</code> to enable voice recording.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### Active Modules")

    emo_enabled = st.session_state.get("cb_emo", True)
    emo_type = st.session_state.get("emo_type", "Sad") if emo_enabled else "Off"
    gen_enabled = st.session_state.get("cb_gen", False)
    gen_type = st.session_state.get("preset_gen", "Manual") if gen_enabled else "Off"
    spk_enabled = st.session_state.get("cb_spk", False)
    sng_enabled = st.session_state.get("cb_sng", False)

    st.markdown(
        f"""
<div class="module-status-row"><strong>Emotion</strong><span class="{'status-on' if emo_enabled else 'status-off'}">{emo_type}</span></div>
<div class="module-status-row"><strong>Gender / Age</strong><span class="{'status-on' if gen_enabled else 'status-off'}">{gen_type}</span></div>
<div class="module-status-row"><strong>Speaker</strong><span class="{'status-on' if spk_enabled else 'status-off'}">{'On' if spk_enabled else 'Off'}</span></div>
<div class="module-status-row"><strong>Singing</strong><span class="{'status-on' if sng_enabled else 'status-off'}">{'On' if sng_enabled else 'Off'}</span></div>
""",
        unsafe_allow_html=True,
    )

    if st.session_state.audio is not None:
        duration = len(st.session_state.audio) / st.session_state.sr
        st.markdown("---")
        st.markdown("#### 📊 Current Audio")
        st.caption(f"**Sample Rate:** {st.session_state.sr} Hz")
        st.caption(f"**Duration:** {duration:.2f} seconds")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1.65, 1], gap="large")

with left_col:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'><h3>Input Signal</h3><span class='badge'>Waveform</span></div>",
        unsafe_allow_html=True,
    )
    if st.session_state.audio is not None:
        st.audio(st.session_state.audio, sample_rate=st.session_state.sr)
        st.pyplot(plot_waveform(st.session_state.audio, st.session_state.sr, theme["accent"], height=(8, 1.1)))
    else:
        st.markdown(
            """
<div class="empty-state" style="min-height: 210px;">
  <div class="empty-state-icon">↥</div>
  <strong>No audio loaded</strong>
  <span>Upload WAV, MP3, FLAC or record from microphone.</span>
</div>
""",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Transformation Modules")

    with st.expander("😊 Emotion conversion", expanded=True):
        st.checkbox("Enable Emotion conversion", key="cb_emo", value=True)

        def set_emotion_defaults():
            if st.session_state.emo_type == "Manual":
                return

            emotion_presets = {
                "Sad":       {"emo_pitch": -2, "emo_speed": 0.85, "emo_energy": 70,  "emo_tilt": -3},
                "Angry":     {"emo_pitch":  4, "emo_speed": 1.25, "emo_energy": 140, "emo_tilt":  3},
                "Excited":   {"emo_pitch":  6, "emo_speed": 1.20, "emo_energy": 130, "emo_tilt":  4},
                "Calm":      {"emo_pitch": -2, "emo_speed": 0.85, "emo_energy": 70,  "emo_tilt": -2},
                "Whispered": {"emo_pitch":  0, "emo_speed": 0.90, "emo_energy": 100, "emo_tilt":  0},
            }

            for key, value in emotion_presets.get(st.session_state.emo_type, {}).items():
                st.session_state[key] = value

        col_e1, col_e2 = st.columns([1, 1.25])
        with col_e1:
            st.selectbox(
                "Target Emotion",
                ["Sad", "Angry", "Excited", "Calm", "Whispered", "Manual"],
                key="emo_type",
                disabled=not st.session_state.cb_emo,
                on_change=set_emotion_defaults,
            )
            st.info("If you select Manual, the slider values ​​will be applied directly.")
        with col_e2:
            val_pitch = st.slider("Pitch shift (st)", -12, 12, -2, key="emo_pitch", disabled=not st.session_state.cb_emo)
            val_speed = st.slider("Speech rate (x)", 0.5, 2.0, 0.85, key="emo_speed", disabled=not st.session_state.cb_emo)
            val_energy = st.slider("Energy (%)", 0, 200, 70, key="emo_energy", disabled=not st.session_state.cb_emo)
            val_tilt = st.slider("Spectral tilt (dB)", -10, 10, -3, key="emo_tilt", disabled=not st.session_state.cb_emo)

    with st.expander("👤 Gender & age conversion"):
        st.checkbox("Enable Gender & age conversion", key="cb_gen", value=False)
        col_g1, col_g2 = st.columns([1, 1.25])
        with col_g1:
            st.selectbox(
                "Preset",
                ["Manual", "Male to Female", "Female to Male", "Adult to Child"],
                key="preset_gen",
                on_change=update_gen_preset,
                disabled=not st.session_state.cb_gen,
            )
        with col_g2:
            st.slider("Pitch Shift (st)", -12, 12, key="gen_pitch", disabled=not st.session_state.cb_gen)
            st.slider("Formant Shift", 0.50, 2.00, key="gen_formant", disabled=not st.session_state.cb_gen)

    with st.expander("👥 Speaker conversion"):
        st.checkbox("Enable Speaker conversion", key="cb_spk", value=False)
        if st.session_state.get("cb_spk", False):
            st.caption("Upload a reference speaker's audio file. The system will transform the input voice to match the reference speaker's characteristics.")
            col_spk1, col_spk2 = st.columns([2, 1])
            with col_spk1:
                ref_spk_file = st.file_uploader(
                    "Reference speaker audio (WAV/MP3/FLAC)",
                    type=["wav", "mp3", "flac"],
                    key="ref_spk_upload",
                )
            with col_spk2:
                if ref_spk_file is not None:
                    st.audio(ref_spk_file, format="audio/wav")
            if ref_spk_file is not None:
                with open("temp_ref_spk.wav", "wb") as f:
                    f.write(ref_spk_file.getbuffer())
                y_ref, sr_ref = librosa.load("temp_ref_spk.wav", sr=None, mono=True)
                st.session_state.ref_spk_audio = y_ref
                st.session_state.ref_spk_sr = sr_ref
                st.success("✓ Reference voice loaded successfully")
            else:
                st.info("Please upload a reference audio file to enable speaker conversion.")
        else:
            st.caption("Disabled - Enable to convert voice to match a reference speaker.")

    with st.expander("🎵 Singing voice generation"):
        st.checkbox("Enable Singing voice generation", key="cb_sng", value=False)
        if st.session_state.get("cb_sng", False):
            st.caption("Transform speech into singing by aligning pitch to a melody.")
            sng_input_type = st.radio(
                "Melody source",
                ["By default", "By note", "MIDI file"],
                key="sng_input_type",
                horizontal=True,
            )
            if sng_input_type == "By note":
                st.text_input(
                    "Enter notes (e.g., C4 D4 E4 F4 G4)",
                    value="C4 D4 E4 F4 G4 A4 B4 C5",
                    key="sng_melody_text",
                    help="Use note names like C4, D#4, etc. Separate with spaces.",
                )
            elif sng_input_type == "MIDI file":
                midi_file = st.file_uploader(
                    "Upload MIDI file",
                    type=["midi", "mid"],
                    key="sng_midi",
                )
                if midi_file is not None:
                    with open("temp_melody.mid", "wb") as f:
                        f.write(midi_file.getbuffer())
                    st.success("✓ MIDI file loaded")
            else:  # By default
                st.info("Using default melody (C major scale)")
        else:
            st.caption("Disabled - Enable to convert speech to singing.")

    generate_col1, generate_col2, generate_col3 = st.columns([1, 2.4, 1])
    with generate_col2:
        generate_clicked = st.button(
            "🚀 Generate Output",
            type="primary",
            use_container_width=True,
            help="Process audio with configured transformation modules"
        )

    if generate_clicked:
        if st.session_state.audio is None:
            st.error("No audio loaded!")
        else:
            with st.spinner("Processing audio..."):
                start_time = time.time()
                proc_audio = st.session_state.audio.copy()

                if st.session_state.get("cb_gen", False):
                    f0_r = 2.0 ** (st.session_state.gen_pitch / 12.0)
                    proc_audio = change_gender_age(
                        proc_audio,
                        st.session_state.sr,
                        f0_ratio=f0_r,
                        env_ratio=st.session_state.gen_formant,
                    )

                if st.session_state.get("cb_emo", False):
                    emo_str = st.session_state.emo_type.lower()
                    proc_audio = change_emotion(
                        proc_audio,
                        st.session_state.sr,
                        emotion=emo_str,
                        pitch_shift_st=st.session_state.emo_pitch,
                        speed_ratio=st.session_state.emo_speed,
                        energy_scale=st.session_state.emo_energy,
                    )
                    if st.session_state.emo_tilt != 0:
                        proc_audio = apply_spectral_tilt(
                            proc_audio,
                            st.session_state.sr,
                            tilt_db=st.session_state.emo_tilt,
                        )

                if st.session_state.get("cb_spk", False):
                    y_ref = st.session_state.get("ref_spk_audio", None)
                    sr_ref = st.session_state.get("ref_spk_sr", st.session_state.sr)
                    proc_audio = convert_speaker(
                        proc_audio,
                        st.session_state.sr,
                        y_ref=y_ref,
                        sr_ref=sr_ref,
                    )

                if st.session_state.get("cb_sng", False):
                    midi_path_arg = (
                        "temp_melody.mid"
                        if st.session_state.get("sng_input_type") == "MIDI file" and os.path.exists("temp_melody.mid")
                        else None
                    )
                    melody_text_arg = (
                        st.session_state.get("sng_melody_text", "")
                        if st.session_state.get("sng_input_type") == "By note"
                        else ""
                    )
                    proc_audio = generate_singing_voice(
                        proc_audio,
                        st.session_state.sr,
                        midi_path=midi_path_arg,
                        melody_text=melody_text_arg or None,
                    )

                # Normalize only if clipping would occur, so energy scaling remains audible.
                max_amp = np.max(np.abs(proc_audio))
                if max_amp > 1.0:
                    proc_audio = proc_audio / max_amp

                process_time = time.time() - start_time
                audio_duration = len(st.session_state.audio) / st.session_state.sr

                st.session_state.processed_audio = proc_audio
                st.session_state.metrics = {
                    "rtf": calculate_rtf(process_time, audio_duration),
                    "snr": calculate_snr(st.session_state.audio, proc_audio),
                    "energy": calculate_energy_ratio(st.session_state.audio, proc_audio),
                    "pitch_dev": calculate_pitch_deviation(st.session_state.audio, proc_audio, st.session_state.sr),
                    "time": process_time,
                }
                st.success("✓ Output generated successfully")

with right_col:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'><h3>Output Preview</h3><span class='badge'>Result</span></div>",
        unsafe_allow_html=True,
    )

    if st.session_state.processed_audio is None:
        st.markdown(
            """
<div class="empty-state">
  <div class="empty-state-icon">♪</div>
  <strong>Output will appear here</strong>
  <span>Configure modules and press Generate Output.</span>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        st.audio(st.session_state.processed_audio, sample_rate=st.session_state.sr)
        st.pyplot(plot_waveform(st.session_state.processed_audio, st.session_state.sr, theme["accent_2"], height=(6, 1.2)))

        try:
            wav_bytes = audio_to_bytes(st.session_state.processed_audio, st.session_state.sr)
            st.download_button(
                label="⬇ Download WAV",
                data=wav_bytes,
                file_name="voicemorph_output.wav",
                mime="audio/wav",
                use_container_width=True,
            )
        except Exception:
            st.button("⬇ Download WAV", use_container_width=True, disabled=True)

        if "metrics" in st.session_state:
            m = st.session_state.metrics
            snr_val = m["snr"]
            snr_str = f"{snr_val:.1f} dB" if snr_val != float("inf") else "∞ dB"
            pitch_dev = m.get("pitch_dev", 0.0)
            sign = "+" if pitch_dev >= 0 else ""

            st.markdown(
                f"""
<div class="metric-grid">
  <div class="metric-card"><div class="metric-label">Time</div><div class="metric-value">{m['time']:.2f} s</div></div>
  <div class="metric-card"><div class="metric-label">RTF</div><div class="metric-value">{m['rtf']:.3f}</div></div>
  <div class="metric-card"><div class="metric-label">SNR</div><div class="metric-value">{snr_str}</div></div>
  <div class="metric-card"><div class="metric-label">Energy</div><div class="metric-value">{m['energy']:.2f}×</div></div>
  <div class="metric-card"><div class="metric-label">Pitch</div><div class="metric-value">{sign}{pitch_dev:.1f} st</div></div>
</div>
""",
                unsafe_allow_html=True,
            )

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-title'><h3>Pipeline</h3><span class='badge'>DSP</span></div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
<div class="module-status-row"><strong>Noise Reduction</strong>{module_badge(noise_reduce)}</div>
<div class="module-status-row"><strong>Emotion</strong>{module_badge(st.session_state.get('cb_emo', True))}</div>
<div class="module-status-row"><strong>Gender / Age</strong>{module_badge(st.session_state.get('cb_gen', False))}</div>
<div class="module-status-row"><strong>Speaker</strong>{module_badge(st.session_state.get('cb_spk', False))}</div>
<div class="module-status-row"><strong>Singing</strong>{module_badge(st.session_state.get('cb_sng', False))}</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)
