# voice-based-concept-understanding-analyser(VBCUA)
AI-powered voice based concept understanding analyser

VBCUA is an AI-powered voice diagnostics and cognitive understanding assessment dashboard. It uses state-of-the-art neural speech models and audio signal processing algorithms to transcribe spoken recordings, analyze speaking fluency metrics, and evaluate how accurately a speaker's definition matches a target reference concept.

The interface is built with a premium glassmorphic dark-theme design.

---

## Key Features

1. **Acoustic Signal Auditing (`librosa`)**
   - **Loudness**: Extracts the Root Mean Square (RMS) energy envelope.
   - **Silent Pause Ratio**: Splits speech signals dynamically at a `30dB` threshold to compute the precise ratio of active speech to silence.
   - **Visual Waveforms**: Plots signal amplitudes over time matching the dashboard's cyber aesthetics.

2. **Local AI Transcription (`openai-whisper`)**
   - Implements local speech-to-text using Whisper's `'tiny'` (or `'base'`) model.
   - Decodes audio files robustly in-memory, bypassing external system dependencies.

3. **Cognitive Similarity Assessment (`sentence-transformers`)**
   - Evaluates linguistic and conceptual alignment using the `'all-MiniLM-L6-v2'` sentence-embedding model.
   - Calculates the cosine similarity between the student's transcription and the master reference concept, bounded strictly between `0%` and `100%`.

4. **Speech Fluency & Filler Analysis**
   - Audits speaking pace (approximate words per second).
   - Scans text token outputs to detect and count hesitation filler words (`um`, `uh`, `like`, `ah`, `so`, `basically`).

5. **Dynamic PDF Reporting (`fpdf2`)**
   - Compiles transcripts, waveform graphs, alignment metrics, and qualitative AI grading feedback into a printable PDF report.

---

## File Structure

- [app.py](file:///d:/Ai%20Voice/app.py) - Streamlit dashboard UI, styling overrides, page layouts, and PDF compiler.
- [engine.py](file:///d:/Ai%20Voice/engine.py) - Decoupled backend execution module responsible for model loaders, Whisper transcription, embedding similarity, and audio signal processing.
- [requirements.txt](file:///d:/Ai%20Voice/requirements.txt) - List of application library dependencies.

---

## Installation & Setup

Ensure you have Python installed. Then follow these steps:

1. **Install Dependencies**
   Navigate to the project root directory and run:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**
   Launch the Streamlit server:
   ```bash
   streamlit run app.py
   ```
   The dashboard will automatically open in your default browser at `http://localhost:8501` (or next available port).

---

## Usage Workflow

1. **Select or Define Concept**: Choose a target reference concept from the sidebar dropdown (e.g., *Machine Learning*, *Cloud Computing*) or write your own custom definition.
2. **Upload Audio Recording**: Drop a `.wav` or `.mp3` recording explaining the chosen concept.
3. **Execution**: The VBCUA pipeline runs automatically:
   - Whisper transcribes the speech.
   - Sentence-Transformers evaluates semantic similarity.
   - Librosa analyzes vocal traits.
4. **Review Dashboard & Export**: Check the tabbed breakdown (Linguistic, Conceptual Alignment, Fluency Signal Profile) and download the generated PDF report.
