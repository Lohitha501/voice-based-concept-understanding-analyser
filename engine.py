import os
import re
import whisper
import librosa
import soundfile as sf
import numpy as np
from sentence_transformers import SentenceTransformer, util

# Thread-safe/module-level lazy loading of model resources
_whisper_model = None
_similarity_model = None

def get_whisper_model(model_name="tiny"):
    """
    Lazy loader for OpenAI Whisper model to cache it in memory.
    """
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(model_name)
    return _whisper_model

def get_similarity_model():
    """
    Lazy loader for Sentence-Transformers model to cache it in memory.
    """
    global _similarity_model
    if _similarity_model is None:
        _similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _similarity_model

def _load_audio_file(file_path):
    """
    Internal helper to load audio file robustly using librosa,
    falling back to soundfile if required.
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
    except Exception as e:
        try:
            data, sr_sf = sf.read(file_path)
            # Downmix to mono if multi-channel
            if len(data.shape) > 1:
                y = librosa.to_mono(data.T)
            else:
                y = data
            sr = sr_sf
        except Exception as sf_err:
            raise RuntimeError(
                f"Audio decoding failed. The format may be unsupported or corrupted.\n"
                f"Librosa error: {e}\nSoundfile error: {sf_err}"
            )
    if len(y) == 0:
        raise ValueError("Audio file contains no decodable sample data.")
    return y, sr

def transcribe_audio(file_path):
    """
    Loads OpenAI Whisper tiny model, transcribes the spoken audio into text,
    and returns the cleaned string.
    """
    try:
        # Load and decode audio robustly
        y, sr = _load_audio_file(file_path)
        
        # Resample to 16000Hz mono in-memory to bypass ffmpeg subprocess dependency
        y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
        
        # Load Whisper
        model = get_whisper_model("tiny")
        if model is None:
            raise RuntimeError("Failed to load OpenAI Whisper model.")
            
        transcribe_result = model.transcribe(y_16k)
        transcription = transcribe_result.get("text", "").strip()
        return transcription
    except Exception as e:
        raise RuntimeError(f"Transcription error: {e}")

def compute_semantic_analysis(student_text, master_text):
    """
    Calculates Cosine Similarity embedding score using 'all-MiniLM-L6-v2',
    bounded strictly between 0% and 100%.
    """
    try:
        if not student_text.strip() or not master_text.strip():
            return 0.0
            
        sim_model = get_similarity_model()
        if sim_model is None:
            raise RuntimeError("Failed to load Sentence-Transformer model.")
            
        emb_ref = sim_model.encode(master_text, convert_to_tensor=True)
        emb_trans = sim_model.encode(student_text, convert_to_tensor=True)
        cosine_val = float(util.cos_sim(emb_ref, emb_trans).item())
        
        # Strictly bound cosine similarity between 0% and 100%
        similarity_score = max(0.0, min(100.0, cosine_val * 100.0))
        return similarity_score
    except Exception as e:
        raise RuntimeError(f"Semantic similarity error: {e}")

def process_audio_metrics(file_path, transcript_text):
    """
    Calculates audio fluency metrics:
    - Loudness: Mean RMS energy.
    - Silent Pause Ratio: Using librosa.effects.split.
    - Filler Words: Count hesitation tokens.
    Returns a structured dictionary of these metrics.
    """
    try:
        y, sr = _load_audio_file(file_path)
        
        # Duration
        duration = len(y) / sr if sr > 0 else 0.0
        
        # 1. Average RMS Energy (Loudness)
        rms = librosa.feature.rms(y=y)
        avg_rms = float(np.mean(rms))
        
        # 2. Pause Ratio (using split at 30db threshold)
        total_samples = len(y)
        if total_samples > 0:
            intervals = librosa.effects.split(y, top_db=30)
            if len(intervals) > 0:
                non_silent_samples = sum(end - start for start, end in intervals)
                silent_samples = total_samples - non_silent_samples
                pause_ratio = max(0.0, min(1.0, silent_samples / total_samples))
            else:
                pause_ratio = 1.0
        else:
            pause_ratio = 0.0
            
        # 3. Filler words counting
        filler_words = {"um", "uh", "like", "ah", "so", "basically"}
        words_list = re.findall(r"\b[a-zA-Z']+\b", transcript_text.lower())
        filler_count = sum(1 for w in words_list if w in filler_words)
        
        return {
            "avg_rms": avg_rms,
            "pause_ratio": pause_ratio,
            "filler_count": filler_count,
            "duration": duration,
            "words_list": words_list,
            "y": y,
            "sr": sr
        }
    except Exception as e:
        raise RuntimeError(f"Audio metrics processing error: {e}")
