"""
Offline Hindi voice recognition using Vosk.
Provides both live-mic and file-based transcription.
"""

import json, os, wave

_voice_dir = os.path.dirname(__file__)

MODEL_PATHS = {
    "hi": [
        os.path.join(_voice_dir, "vosk-model-hi-0.22"),
        os.path.join(_voice_dir, "vosk-model-hi-0.22", "vosk-model-hi-0.22"),
        "voice/vosk-model-hi-0.22"
    ],
    "en": [
        os.path.join(_voice_dir, "vosk-model-small-en-us-0.15"),
        "voice/vosk-model-small-en-us-0.15"
    ]
}

_models = {}

def _get_model(lang="en"):
    """Search for and lazy-load the Vosk model."""
    if lang not in _models:
        from vosk import Model
        options = MODEL_PATHS.get(lang, [])
        found_path = None
        for p in options:
            if os.path.isdir(p):
                found_path = p
                break
        
        if not found_path:
            raise FileNotFoundError(
                f"Vosk {lang} model not found. "
                f"Please download from alphacephei.com/vosk/models and unzip into 'voice/' folder."
            )
        
        print(f"Loading Vosk {lang} model from {found_path}...")
        _models[lang] = Model(found_path)
    return _models[lang]

def transcribe_audio_file(filepath: str, lang: str = "en") -> str:
    """
    Transcribe a WAV audio file to text using Vosk offline ASR.
    Accepts the file path that Gradio's gr.Audio returns.
    Returns the transcribed text string.
    """
    from vosk import KaldiRecognizer

    model = _get_model(lang)

    wf = wave.open(filepath, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
        raise ValueError("Audio must be mono 16-bit WAV")

    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    text_parts = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            result = json.loads(rec.Result())
            if result.get("text"):
                text_parts.append(result["text"])

    # Get final partial result
    final = json.loads(rec.FinalResult())
    if final.get("text"):
        text_parts.append(final["text"])

    wf.close()
    return " ".join(text_parts).strip()


def listen_mic(duration: int = 5, samplerate: int = 16000, lang: str = "en") -> str:
    """
    Record from microphone for `duration` seconds and transcribe.
    Requires sounddevice.
    """
    import queue
    import sounddevice as sd
    from vosk import KaldiRecognizer

    model = _get_model(lang)
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(bytes(indata))

    rec = KaldiRecognizer(model, samplerate)
    with sd.RawInputStream(samplerate=samplerate, blocksize=8000,
                           dtype="int16", channels=1, callback=callback):
        for _ in range(0, int(samplerate / 8000 * duration)):
            data = q.get()
            rec.AcceptWaveform(data)

    return json.loads(rec.FinalResult())["text"]