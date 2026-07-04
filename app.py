import streamlit as st
import os
import tempfile
import re
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import librosa.display
from fpdf import FPDF
from engine import transcribe_audio, compute_semantic_analysis, process_audio_metrics

# Page Configuration
st.set_page_config(
    page_title="Voice-Based Concept Understanding Analyser (VBCUA)",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphic CSS Injection
st.markdown("""
<style>
    /* Import Premium Font */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Outfit', sans-serif;
    }
    h1, h2, h3, h4, h5, h6, p, label, .title-glow, .subtitle, textarea, input {
        font-family: 'Outfit', sans-serif !important;
    }
    
    .stApp {
        background: radial-gradient(circle at 5% 10%, rgba(56, 189, 248, 0.15), transparent 40%),
                    radial-gradient(circle at 95% 85%, rgba(99, 102, 241, 0.15), transparent 40%),
                    radial-gradient(circle at 50% 50%, rgba(139, 92, 246, 0.06), transparent 60%),
                    radial-gradient(rgba(255, 255, 255, 0.015) 1px, transparent 1px),
                    #080b16 !important;
        background-size: 100% 100%, 100% 100%, 100% 100%, 24px 24px, 100% 100% !important;
        color: #f3f4f6 !important;
    }
    
    /* Custom Scrollbars */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(7, 10, 19, 0.5);
    }
    ::-webkit-scrollbar-thumb {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(56, 189, 248, 0.3);
    }
    
    /* Header Glow */
    .title-glow {
        background: linear-gradient(135deg, #38bdf8 0%, #6366f1 50%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.6rem;
        font-weight: 800;
        text-shadow: 0 0 40px rgba(56, 189, 248, 0.12);
        margin-bottom: 0.2rem;
        text-align: center;
        letter-spacing: -0.5px;
    }
    
    .subtitle {
        color: #9ca3af;
        font-size: 1.05rem;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }
    
    /* Frosted Glass Container / Cards */
    .glass-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.03) 0%, rgba(255, 255, 255, 0.01) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.4), inset 0 1px 1px rgba(255, 255, 255, 0.1);
        margin-bottom: 20px;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .glass-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.2);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.15);
    }
    
    /* Glowing Translucent Metric Cards */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        margin-bottom: 24px;
        width: 100%;
    }
    
    .metric-card {
        flex: 1;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.02) 0%, rgba(255, 255, 255, 0.005) 100%);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 22px 16px;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2), inset 0 1px 0px rgba(255, 255, 255, 0.05);
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }
    
    .metric-card:hover {
        border-color: rgba(56, 189, 248, 0.25);
        transform: translateY(-3px);
        box-shadow: 0 12px 40px 0 rgba(56, 189, 248, 0.1);
    }
    
    .metric-title {
        color: #9ca3af;
        font-size: 0.8rem;
        font-weight: 600;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1.8px;
    }
    
    .metric-value {
        color: #38bdf8;
        font-size: 2.5rem;
        font-weight: 800;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.4);
        line-height: 1.1;
    }
    
    .metric-desc {
        color: #6b7280;
        font-size: 0.75rem;
        margin-top: 8px;
        letter-spacing: 0.2px;
    }
    
    /* Styled Badges for Qualitative Ratings */
    .badge-wrapper {
        margin-top: 12px;
        display: inline-block;
    }
    
    .badge {
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        padding: 6px 18px;
        border-radius: 50px;
        display: inline-block;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
    }
    
    .badge-strong {
        background-color: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.35);
        color: #10b981;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
    }
    
    .badge-moderate {
        background-color: rgba(245, 158, 11, 0.1);
        border: 1px solid rgba(245, 158, 11, 0.35);
        color: #f59e0b;
        box-shadow: 0 0 15px rgba(245, 158, 11, 0.2);
    }
    
    .badge-poor {
        background-color: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.35);
        color: #ef4444;
        box-shadow: 0 0 15px rgba(239, 68, 68, 0.2);
    }
    
    /* Sidebar glass panel */
    section[data-testid="stSidebar"] {
        background-color: rgba(9, 12, 22, 0.85) !important;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Inputs, textareas, selectboxes overrides */
    textarea, input {
        background-color: rgba(255, 255, 255, 0.02) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        transition: all 0.3s ease !important;
    }
    textarea:focus, input:focus {
        border-color: rgba(56, 189, 248, 0.4) !important;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.15) !important;
    }
    
    /* Streamlit input fields and upload styling overrides */
    div[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.005) !important;
        border: 1px dashed rgba(56, 189, 248, 0.2) !important;
        border-radius: 16px !important;
        padding: 15px !important;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
    }
    div[data-testid="stFileUploader"]:hover {
        border-color: rgba(56, 189, 248, 0.5) !important;
        background: rgba(255, 255, 255, 0.015) !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.08);
    }
    
    /* Cyber Button Styles (for st.download_button & st.button) */
    div.stButton > button, div.stDownloadButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 15px rgba(79, 70, 229, 0.25) !important;
        transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
        width: 100%;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(56, 189, 248, 0.4) !important;
        border-color: rgba(56, 189, 248, 0.4) !important;
    }
    div.stButton > button:active, div.stDownloadButton > button:active {
        transform: translateY(1px) !important;
    }
    
    /* Tabs Overrides */
    div[data-baseweb="tab-list"] {
        background-color: transparent !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08) !important;
        gap: 8px !important;
    }
    
    div[data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.005) !important;
        color: #9ca3af !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-bottom: none !important;
        border-top-left-radius: 12px !important;
        border-top-right-radius: 12px !important;
        padding: 12px 24px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    div[data-baseweb="tab"]:hover {
        color: #ffffff !important;
        background-color: rgba(255, 255, 255, 0.02) !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    div[data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(56, 189, 248, 0.06) !important;
        color: #38bdf8 !important;
        border-color: rgba(56, 189, 248, 0.25) !important;
        border-bottom: 2px solid #38bdf8 !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)


def sanitize_latin1(text):
    if not text:
        return ""
    # Map common curly quotes, dashes, etc. to standard ascii equivalents
    replacements = {
        '\u201c': '"', '\u201d': '"',  # Left/Right double quotes
        '\u2018': "'", '\u2019': "'",  # Left/Right single quotes
        '\u2013': '-', '\u2014': '-',  # En/Em dashes
        '\u2022': '*',                  # Bullet points
        '\xe9': 'e', '\xe1': 'a',      # Accents
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    # Encode to latin-1 and ignore any characters that can't be mapped
    return text.encode('latin-1', 'ignore').decode('latin-1')

class VBCUAReport(FPDF):
    def header(self):
        # Draw background band/border
        self.set_fill_color(7, 10, 19) # Deep dark background band #070a13
        self.rect(0, 0, 210, 30, 'F')
        
        self.set_text_color(255, 255, 255)
        self.set_font("Helvetica", "B", 15)
        self.set_y(6)
        self.cell(0, 8, "Voice-Based Concept Understanding Analyser", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "I", 9)
        self.cell(0, 6, "VBCUA AI-Powered Evaluation Report", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(10)
        
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()} | Generated dynamically by VBCUA Engine", align="C")

def generate_pdf_report(transcription, similarity_score, pause_ratio, avg_rms, duration, words_list, filler_count, reference_text, selected_concept, y, sr):
    pdf = VBCUAReport()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_y(38) # Move below header
    
    # Sanitise variables
    transcription = sanitize_latin1(transcription)
    reference_text = sanitize_latin1(reference_text)
    selected_concept = sanitize_latin1(selected_concept)
    
    # 1. Title/Header Info
    pdf.set_text_color(2, 132, 199) # printable deep cyan/blue
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "1. Executive Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    
    # Metadata / Scores Table
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 10)
    
    # Table headers
    pdf.cell(90, 8, "Evaluation Metric", border=1, align="L")
    pdf.cell(95, 8, "Computed Value / Grade", border=1, align="C", new_x="LMARGIN", new_y="NEXT")
    
    pdf.set_font("Helvetica", "", 10)
    
    # Determine grade
    if similarity_score >= 80.0:
        grade = "Strong Understanding"
    elif similarity_score >= 50.0:
        grade = "Moderate Understanding"
    else:
        grade = "Poor Understanding"
        
    metrics_data = [
        ("Concept Target", selected_concept),
        ("Match Similarity Score", f"{similarity_score:.1f}% ({grade})"),
        ("Silence Pause Ratio", f"{pause_ratio * 100:.1f}%"),
        ("Loudness Energy (RMS)", f"{avg_rms:.5f}"),
        ("Vocal Duration", f"{duration:.2f} seconds"),
        ("Total Filler Words Count", f"{filler_count} fillers"),
        ("Speaking Pace", f"{len(words_list) / duration:.2f} words/sec" if duration > 0 else "0.00 words/sec")
    ]
    
    for metric, val in metrics_data:
        pdf.cell(90, 8, metric, border=1, align="L")
        pdf.cell(95, 8, val, border=1, align="C", new_x="LMARGIN", new_y="NEXT")
        
    pdf.ln(8)
    
    # 2. Transcription section
    pdf.set_text_color(2, 132, 199)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "2. Speech-to-Text Transcription Output", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    
    pdf.set_text_color(30, 30, 30)
    pdf.set_font("Helvetica", "I", 10)
    # Using multi_cell to wrap long text
    pdf.multi_cell(0, 6, f'"{transcription}"' if transcription else '"No speech detected in audio file."', border=1)
    pdf.ln(8)
    
    # 3. Concept Verification Analysis
    pdf.set_text_color(2, 132, 199)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "3. Concept Verification & Mapping", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, f"Expected Reference Concept: {selected_concept}", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, reference_text)
    pdf.ln(4)
    
    pdf.set_font("Helvetica", "B", 10)
    pdf.cell(0, 6, "AI Qualitative Feedback & Analysis:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    if similarity_score >= 80.0:
        feedback = "Excellent conceptual alignment. The user's explanation matches all key semantic criteria and demonstrates high domain familiarity."
    elif similarity_score >= 50.0:
        feedback = "Moderate conceptual alignment. The speaker touched upon core terminology but missed significant structural explanations or contextual logic."
    else:
        feedback = "Weak conceptual alignment. Semantic density is low. The speaker either deviated from the subject or did not include necessary keyword qualifiers."
    pdf.multi_cell(0, 5, feedback)
    pdf.ln(8)
    
    # 4. Signal Profile (Waveform image)
    pdf.set_text_color(2, 132, 199)
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 10, "4. Fluency Signal Profile & Waveform", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    
    # Plot waveform to local file
    temp_img_path = ""
    try:
        fig, ax = plt.subplots(figsize=(8, 2.5))
        fig.patch.set_facecolor('#ffffff')
        ax.set_facecolor('#ffffff')
        librosa.display.waveshow(y, sr=sr, ax=ax, color='#0284c7', alpha=0.85) # Use solid sky blue for printable PDF
        ax.set_title("Time-Domain Audio Amplitude Waveform", color='#0f172a', fontsize=9, fontweight='bold')
        ax.set_xlabel("Time (Seconds)", color='#475569', fontsize=8)
        ax.set_ylabel("Amplitude", color='#475569', fontsize=8)
        ax.grid(True, color='#e2e8f0', linestyle='--', linewidth=0.5)
        for spine in ax.spines.values():
            spine.set_edgecolor('#cbd5e1')
        ax.tick_params(colors='#475569', which='both', labelsize=8)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
            temp_img_path = tmp_img.name
        
        fig.savefig(temp_img_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        
        # Insert image into PDF
        pdf.image(temp_img_path, w=180)
        pdf.ln(6)
    except Exception as img_err:
        pdf.set_text_color(220, 38, 38)
        pdf.cell(0, 6, f"[Error embedding waveform visualization: {img_err}]", new_x="LMARGIN", new_y="NEXT")
        pdf.set_text_color(0, 0, 0)
    finally:
        if temp_img_path and os.path.exists(temp_img_path):
            try:
                os.remove(temp_img_path)
            except Exception:
                pass
                
    # Return raw PDF bytes
    return pdf.output()

# ----------------- STATE ROUTING & WELCOME SCREEN -----------------
if "started" not in st.session_state:
    st.session_state.started = False

if not st.session_state.started:
    st.markdown("""
    <style>
        /* Hide sidebar toggle arrows on landing page */
        div[data-testid="collapsedControl"] {
            display: none !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Render Landing Page Title
    st.markdown('<div class="title-glow" style="margin-top: 15px; font-size: 2.6rem; background: linear-gradient(135deg, #00f2fe 0%, #4facfe 30%, #6366f1 70%, #a855f7 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-shadow: 0 0 35px rgba(0, 242, 254, 0.15);">Voice-Based Concept Understanding Analyser</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle" style="font-size: 1.0rem; margin-bottom: 25px; color: #a1a1aa; letter-spacing: 0.8px;">AI-Powered Speech Transcription & Cognitive Assessment Suite</div>', unsafe_allow_html=True)
    
    # Centered Hero Card
    st.markdown("""
    <div class="glass-card" style="max-width: 950px; margin: 0 auto; padding: 22px 30px; text-align: center; border: 1px solid rgba(56, 189, 248, 0.15); box-shadow: 0 15px 35px rgba(0, 0, 0, 0.6), inset 0 1px 1px rgba(255, 255, 255, 0.05); background: linear-gradient(135deg, rgba(255, 255, 255, 0.025) 0%, rgba(255, 255, 255, 0.005) 100%);">
        <h3 style="margin-bottom: 10px; background: linear-gradient(90deg, #ffffff 0%, #a1a1aa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700; font-size: 1.5rem; letter-spacing: -0.5px;">Assess Comprehension & Vocal Delivery Performance</h3>
        <p style="color: #cbd5e1; font-size: 0.95rem; line-height: 1.6; margin-bottom: 0px; font-weight: 300;">
            VBCUA merges advanced AI models and deep acoustic signal processing to grade how effectively you structure and present verbal definitions. Get real-time semantic similarity scoring, clarity audits, and speaking confidence profiles.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    
    # 3 Scenarios columns
    col_sc1, col_sc2, col_sc3 = st.columns(3)
    
    with col_sc1:
        st.markdown("""
        <div class="glass-card" style="height: 195px; text-align: center; padding: 20px 15px; border: 1px solid rgba(16, 185, 129, 0.25); background: linear-gradient(135deg, rgba(16, 185, 129, 0.04) 0%, rgba(255, 255, 255, 0.005) 100%);">
            <div style="font-size: 2.0rem; margin-bottom: 10px; filter: drop-shadow(0 0 10px rgba(16, 185, 129, 0.4));">🧬</div>
            <h4 style="color: #34d399; margin-bottom: 8px; font-weight: 600; font-size: 1.05rem; letter-spacing: 0.2px;">Scenario 1: Concept Mapping</h4>
            <p style="color: #9ca3af; font-size: 0.82rem; line-height: 1.5; font-weight: 300; margin-bottom: 0;">
                Compare speech transcriptions with target concepts using Sentence-BERT embeddings.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_sc2:
        st.markdown("""
        <div class="glass-card" style="height: 195px; text-align: center; padding: 20px 15px; border: 1px solid rgba(99, 102, 241, 0.25); background: linear-gradient(135deg, rgba(99, 102, 241, 0.04) 0%, rgba(255, 255, 255, 0.005) 100%);">
            <div style="font-size: 2.0rem; margin-bottom: 10px; filter: drop-shadow(0 0 10px rgba(99, 102, 241, 0.4));">📈</div>
            <h4 style="color: #818cf8; margin-bottom: 8px; font-weight: 600; font-size: 1.05rem; letter-spacing: 0.2px;">Scenario 2: Fluency Audits</h4>
            <p style="color: #9ca3af; font-size: 0.82rem; line-height: 1.5; font-weight: 300; margin-bottom: 0;">
                Analyse voice energy, silent interval pause ratios, and identify filler words using Librosa.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col_sc3:
        st.markdown("""
        <div class="glass-card" style="height: 195px; text-align: center; padding: 20px 15px; border: 1px solid rgba(168, 85, 247, 0.25); background: linear-gradient(135deg, rgba(168, 85, 247, 0.04) 0%, rgba(255, 255, 255, 0.005) 100%);">
            <div style="font-size: 2.0rem; margin-bottom: 10px; filter: drop-shadow(0 0 10px rgba(168, 85, 247, 0.4));">📄</div>
            <h4 style="color: #c084fc; margin-bottom: 8px; font-weight: 600; font-size: 1.05rem; letter-spacing: 0.2px;">Scenario 3: Reports</h4>
            <p style="color: #9ca3af; font-size: 0.82rem; line-height: 1.5; font-weight: 300; margin-bottom: 0;">
                Export transcripts, waveform charts, and qualitative AI feedback inside a PDF.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
    
    # Glowing Get Started button panel
    st.markdown('<div style="text-align: center; max-width: 320px; margin: 0 auto;">', unsafe_allow_html=True)
    if st.button("🚀 Get Started", key="get_started_btn"):
        st.session_state.started = True
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.stop()

# --- WORKSPACE MODE ---
# App Title & Description
st.markdown('<div class="title-glow">Voice-Based Concept Understanding Analyser</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A High-Fidelity Audio Analytics Dashboard using AI & Signal Processing</div>', unsafe_allow_html=True)

# ----------------- SIDEBAR CONFIGURATION -----------------
with st.sidebar:
    if st.button("<- Return Home", key="back_to_home_btn"):
        st.session_state.started = False
        st.rerun()
    st.markdown("---")
    st.markdown("### ⚙️ VBCUA Engine Config")
    st.markdown("Select a reference concept to verify the speaker's vocal understanding against, or construct your own target.")
    
    # Pre-populated concepts mapping
    prepopulated_concepts = {
        "Machine Learning": "Machine learning is a subset of artificial intelligence that involves training algorithms to learn patterns from data and make predictions or decisions without being explicitly programmed.",
        "Cloud Computing": "Cloud computing is the on-demand delivery of IT resources, including databases, storage, servers, and networking, over the internet with pay-as-you-go pricing.",
        "Custom Reference Concept": ""
    }
    
    selected_concept = st.selectbox(
        "Target Reference Concept:",
        options=list(prepopulated_concepts.keys()),
        index=0
    )
    
    # Text area for concept details
    if selected_concept == "Custom Reference Concept":
        reference_text = st.text_area(
            "Define Custom Target Concept:",
            value="",
            placeholder="Type your reference concept definition here...",
            height=180
        )
    else:
        # Show preset definition, but allow user to edit it if they wish
        reference_text = st.text_area(
            "Reference Concept Description:",
            value=prepopulated_concepts[selected_concept],
            height=180
        )
        
    st.markdown("---")
    st.markdown("### 🎙️ How to analyze")
    st.markdown("""
    1. Select or input the concept description.
    2. Upload a short audio recording (**WAV** or **MP3**) explaining that concept.
    3. The VBCUA pipeline executes:
       - **Librosa** extracts signal characteristics.
       - **Whisper** converts speech to text.
       - **Sentence-Transformers** calculates the concept alignment.
    """)
    
    # Visual Legend
    st.markdown("### 🏷️ Understanding Grades")
    st.markdown("""
    <div style='margin-bottom: 10px;'><span class='badge badge-strong'>Strong</span> 80% similarity or higher</div>
    <div style='margin-bottom: 10px;'><span class='badge badge-moderate'>Moderate</span> 50% to 79% similarity</div>
    <div style='margin-bottom: 10px;'><span class='badge badge-poor'>Poor</span> Below 50% similarity</div>
    """, unsafe_allow_html=True)


# ----------------- MAIN PROCESSING PIPELINE -----------------
# 1. File Uploader
st.markdown('<div class="glass-card">', unsafe_allow_html=True)
st.markdown("### 📥 Audio Upload Panel")
uploaded_file = st.file_uploader(
    "Drop your recording here (.wav, .mp3, .mpeg)",
    type=["wav", "mp3", "mpeg"],
    help="Supports WAV, MP3, and MPEG files."
)
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Validate file is not empty
    file_bytes = uploaded_file.getvalue()
    if len(file_bytes) == 0:
        st.error("Uploaded file is empty! Please upload a valid audio recording.")
    else:
        # Safe storage path inside a try-finally block to prevent memory leaks
        suffix = os.path.splitext(uploaded_file.name)[1].lower()
        if not suffix:
            suffix = ".wav"  # Default fallback
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name
            
        try:
            # Show Audio Player
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Playback Audio Clip")
            st.audio(file_bytes, format="audio/mpeg" if suffix in [".mp3", ".mpeg"] else "audio/wav")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Start Analysis
            with st.spinner("🚀 Running VBCUA Pipeline..."):
                # 1. Transcribe Audio using Whisper Tiny model in engine.py
                transcription = transcribe_audio(temp_file_path)
                
                # 2. Process Audio Fluency Metrics using Librosa in engine.py
                metrics = process_audio_metrics(temp_file_path, transcription)
                avg_rms = metrics["avg_rms"]
                pause_ratio = metrics["pause_ratio"]
                filler_count = metrics["filler_count"]
                duration = metrics["duration"]
                words_list = metrics["words_list"]
                y = metrics["y"]
                sr = metrics["sr"]
                
                # 3. Compute Semantic Similarity using SentenceTransformer in engine.py
                similarity_score = 0.0
                if not reference_text.strip():
                    st.warning("⚠️ Reference Concept is empty. Similarity score defaults to 0%. Please set reference concept in the sidebar.")
                elif not transcription:
                    st.warning("⚠️ Speech transcription was blank. Similarity score defaults to 0%.")
                else:
                    similarity_score = compute_semantic_analysis(transcription, reference_text)
            
            # ----------------- RENDER METRICS DASHBOARD -----------------
            # Determine rating badges
            if similarity_score >= 80.0:
                badge_html = '<div class="badge-wrapper"><span class="badge badge-strong">Strong Understanding</span></div>'
            elif similarity_score >= 50.0:
                badge_html = '<div class="badge-wrapper"><span class="badge badge-moderate">Moderate Understanding</span></div>'
            else:
                badge_html = '<div class="badge-wrapper"><span class="badge badge-poor">Poor Understanding</span></div>'
                
            # Create three columns of metric cards
            st.markdown(f"""
            <div class="metric-container">
                <div class="metric-card">
                    <div class="metric-title">Concept Match Similarity</div>
                    <div class="metric-value">{similarity_score:.1f}%</div>
                    {badge_html}
                </div>
                <div class="metric-card">
                    <div class="metric-title">Pause Ratio (Silent Interval)</div>
                    <div class="metric-value">{pause_ratio * 100:.1f}%</div>
                    <div class="metric-desc">Threshold: 30dB silence split</div>
                </div>
                <div class="metric-card">
                    <div class="metric-title">Loudness (RMS Energy)</div>
                    <div class="metric-value">{avg_rms:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- INTERACTIVE REPORTING & PDF EXPORT (Scenario 3) ---
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            pdf_col_left, pdf_col_right = st.columns([3, 1])
            with pdf_col_left:
                st.markdown("#### 📄 Export Performance Assessment Report")
                st.write("Generate and download a structured PDF report containing the time-domain waveform graph, full transcription, alignment grading metrics, and detailed fluency assessments.")
            with pdf_col_right:
                try:
                    pdf_bytes = generate_pdf_report(
                        transcription=transcription,
                        similarity_score=similarity_score,
                        pause_ratio=pause_ratio,
                        avg_rms=avg_rms,
                        duration=duration,
                        words_list=words_list,
                        filler_count=filler_count,
                        reference_text=reference_text,
                        selected_concept=selected_concept,
                        y=y,
                        sr=sr
                    )
                    st.download_button(
                        label="📥 Download PDF Report",
                        data=bytes(pdf_bytes),
                        file_name=f"VBCUA_Performance_Report_{selected_concept.replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                except Exception as pdf_err:
                    st.error(f"PDF Compile Error: {pdf_err}")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # ----------------- DASHBOARD DETAIL TABS -----------------
            tab1, tab2, tab3 = st.tabs([
                "💬 Linguistic Transcription", 
                "🧬 Concept Verification", 
                "📈 Fluency Signal Profile"
            ])
            
            with tab1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### Speech-to-Text Transcription Output")
                if transcription:
                    st.info(f'"{transcription}"')
                else:
                    st.warning("No speech could be deciphered in the audio file. Please ensure there is audible, clean talking.")
                
                # Filler word details
                st.markdown("#### Filler Word Analysis")
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.metric(label="Total Fillers", value=filler_count)
                with col2:
                    st.markdown("We scanned your audio transcription for the following filler items: `um`, `uh`, `like`, `ah`, `so`, `basically`.")
                    if filler_count > 0:
                        # Highlight matching words
                        filler_words = {"um", "uh", "like", "ah", "so", "basically"}
                        found_fillers = [w for w in words_list if w in filler_words]
                        st.markdown(f"**Identified filler tokens**: {', '.join([f'`{f}`' for f in set(found_fillers)])}")
                    else:
                        st.success("Clean speech! No common fillers were identified.")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with tab2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### Conceptual Mapping Analysis")
                
                c_col1, c_col2 = st.columns(2)
                with c_col1:
                    st.markdown("**Your Transcribed Speech**:")
                    st.write(transcription if transcription else "*No speech detected*")
                with c_col2:
                    st.markdown("**Expected Reference Description**:")
                    st.write(reference_text if reference_text.strip() else "*No reference set*")
                
                st.markdown("---")
                st.markdown("#### Qualitative Alignment Breakdown")
                if similarity_score >= 80.0:
                    st.success("🌟 **Excellent Conceptual Alignment**: Your explanation covers almost all key terms and semantic connections of the target concept. Great command shown!")
                elif similarity_score >= 50.0:
                    st.warning("⚠️ **Moderate Alignment**: You hit some core terms, but missed crucial context or detail. Try expanding on the structure and background details of the concept.")
                else:
                    st.error("❌ **Weak Alignment**: The semantics of your explanation do not align closely with the target reference concept. Check if you discussed the right topic or need to include more precise definitions.")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with tab3:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown("#### Audio Waveform Visualisation")
                
                # Generate clean visual graph matching visual styling
                fig, ax = plt.subplots(figsize=(12, 4))
                # Use a very dark figure & axes background to match the theme
                fig.patch.set_facecolor('#070a13')
                ax.set_facecolor('#0d1117')
                
                # Plot waveform
                librosa.display.waveshow(y, sr=sr, ax=ax, color='#38bdf8', alpha=0.85)
                
                # Styling plot aesthetics
                ax.set_title("Time-Domain Waveform (Amplitude)", color='#f3f4f6', fontsize=12, fontweight='bold', pad=15)
                ax.set_xlabel("Time (Seconds)", color='#9ca3af', fontsize=10)
                ax.set_ylabel("Amplitude Scale", color='#9ca3af', fontsize=10)
                
                # Muted grid: matplotlib requires float tuples or hex, NOT CSS rgba() strings
                ax.grid(True, color=(1.0, 1.0, 1.0, 0.07), linestyle='--', linewidth=0.5)
                for spine in ax.spines.values():
                    spine.set_edgecolor((1.0, 1.0, 1.0, 0.15))
                
                ax.tick_params(colors='#9ca3af', which='both', labelsize=9)
                
                # Display in Streamlit
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)  # Prevent memory leaks
                
                # Audio Signal Statistics Table
                st.markdown("#### Fluency Signal Metrics Breakdown")
                stats_df = {
                    "Metric Feature": [
                        "Audio Duration",
                        "Average RMS (Loudness Energy)",
                        "Vocal Pause Percentage",
                        "Speaking Rate (Approx. Words/Sec)"
                    ],
                    "Computed Value": [
                        f"{duration:.2f} seconds",
                        f"{avg_rms:.5f}",
                        f"{pause_ratio * 100:.2f}%",
                        f"{len(words_list) / duration:.2f} words/sec" if duration > 0 else "0.00 words/sec",
                    ],
                    "Fluency Interpretation": [
                        "Total length of audio track processed",
                        "Indicates loudness profile. Higher means more vocal power / microphone proximity",
                        "Percentage of silent gaps. Normal speech features 10% - 25% pauses",
                        "Standard English conversational pace is 2.0 - 2.5 words per second"
                    ]
                }
                st.table(stats_df)
                st.markdown('</div>', unsafe_allow_html=True)
                
        except Exception as err:
            st.error(f"⚠️ VBCUA Pipeline Failure: {err}")
        finally:
            # Delete temporary file immediately after analysis completion to prevent leaks
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception as del_err:
                    # Suppress or log deletion failure silently to user, but show developer warning if run locally
                    pass
else:
    # Welcome/Instructions display when no file is uploaded
    st.markdown('<div class="glass-card" style="text-align: center; padding: 40px 20px;">', unsafe_allow_html=True)
    st.markdown("### 🎙️ Ready to Analyse Your Speaking? ")
    st.markdown("""
    Please upload a **.wav** or **.mp3** voice clip using the uploader panel above.
    
    The engine will process your voice dynamics, convert speech to text, and verify if your definition 
    aligns conceptually with the selected target in the sidebar.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
