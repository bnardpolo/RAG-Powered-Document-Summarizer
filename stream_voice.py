from __future__ import annotations
from typing import Optional
import io
import tempfile
import streamlit as st


def _bytes_to_wav_16k_mono(audio_bytes: bytes, filename: str | None = None) -> bytes:
    """
    Normalize arbitrary audio bytes to mono 16 kHz WAV (PCM16).
    Prefers pydub (handles mp3/m4a/wav); falls back to soundfile.
    Returns WAV bytes; if all conversions fail, returns original bytes.
    """
    if not audio_bytes:
        return b""

    
    try:
        from pydub import AudioSegment
        file_like = io.BytesIO(audio_bytes)
        fmt = None
        if filename and "." in filename:
            fmt = filename.rsplit(".", 1)[-1].lower()
        seg = AudioSegment.from_file(file_like, format=fmt)
        seg = seg.set_channels(1).set_frame_rate(16000).set_sample_width(2)  # mono, 16k, 16-bit
        out = io.BytesIO()
        seg.export(out, format="wav")
        return out.getvalue()
    except Exception:
        pass

  
    try:
        import soundfile as sf
        import numpy as np
        data, sr = sf.read(io.BytesIO(audio_bytes), always_2d=True)
        # simple linear resample if needed (no heavy deps)
        if sr != 16000:
            import math
            ratio = 16000 / float(sr)
            new_len = int(math.ceil(data.shape[0] * ratio))
            x_old = np.linspace(0, 1, num=data.shape[0], endpoint=False)
            x_new = np.linspace(0, 1, num=new_len, endpoint=False)
            data = np.interp(x_new, x_old, data[:, 0]).reshape(-1, 1)
            sr = 16000
        mono = data[:, 0].astype("float32")
        mono = mono / (max(abs(mono.max()), abs(mono.min())) + 1e-9)
        pcm16 = (mono * 32767.0).astype("int16")
        out = io.BytesIO()
        sf.write(out, pcm16, sr, format="WAV", subtype="PCM_16")
        return out.getvalue()
    except Exception:
        pass

    return audio_bytes



def _transcribe_faster_whisper(wav_bytes: bytes) -> Optional[str]:
    try:
        from faster_whisper import WhisperModel
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            path = tmp.name
        model = WhisperModel("base", device="cpu")  # change to "small" for a bit more quality
        segments, _ = model.transcribe(path)
        text = " ".join(s.text.strip() for s in segments if getattr(s, "text", "").strip())
        return text.strip() or None
    except Exception:
        return None

def _transcribe_vosk(wav_bytes: bytes) -> Optional[str]:
    """
    Offline fallback using Vosk. Requires VOSK_MODEL_PATH env var pointing to a model dir.
    """
    try:
        import os, json, wave
        from vosk import Model, KaldiRecognizer
        model_path = os.getenv("VOSK_MODEL_PATH")
        if not model_path:
            return None
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            path = tmp.name
        wf = wave.open(path, "rb")
        if wf.getnchannels() != 1 or wf.getframerate() != 16000:
            return None
        rec = KaldiRecognizer(Model(model_path), wf.getframerate())
        rec.SetWords(True)
        out = []
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                try:
                    out.append(json.loads(rec.Result()).get("text", ""))
                except Exception:
                    pass
        try:
            out.append(json.loads(rec.FinalResult()).get("text", ""))
        except Exception:
            pass
        text = " ".join(t for t in out if t).strip()
        return text or None
    except Exception:
        return None

def transcribe_bytes(audio_bytes: bytes, filename: str | None = None) -> str:
    """
    Whisper-free transcription pipeline:
      1) normalize to mono 16 kHz WAV
      2) faster-whisper → Vosk fallback
    """
    if not audio_bytes:
        return ""
    wav = _bytes_to_wav_16k_mono(audio_bytes, filename=filename)
    text = _transcribe_faster_whisper(wav)
    if text:
        return text
    text = _transcribe_vosk(wav)
    return text or ""



def record_or_upload_audio() -> Optional[bytes]:
    """
    Primary: streamlit-mic-recorder (no aiortc/av).
    Fallback: file uploader. Returns raw bytes.
    """
    # 1) Mic recorder
    try:
        from streamlit_mic_recorder import mic_recorder
        st.write("Record audio")
        rec = mic_recorder(
            start_prompt="Start recording",
            stop_prompt="Stop",
            key="mic_rec",
            format="wav",         # returns WAV bytes already
            just_once=False
        )
        if rec and isinstance(rec, dict) and rec.get("bytes"):
            return rec["bytes"]
    except Exception:
        pass


    st.write("Or upload an audio file")
    up = st.file_uploader("Upload WAV/MP3/M4A", type=["wav", "mp3", "m4a"])
    if up:
        return up.read()
    return None
