"""
FastAPI REST endpoint for the Rural Health Triage engine.
Run:  uvicorn api:app --host 0.0.0.0 --port 8000
"""

import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from engine.triage import TriageEngine
from engine.formatter import format_result
from voice.stt import transcribe_audio_file
import tempfile
import traceback
try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

app = FastAPI(
    title="Rural Health Triage API",
    description="Offline AI symptom triage — English + Hindi",
    version="1.0.0",
)

# ── Serve Static PWA ───────────────────────────────────────────────────────
# Serve static directories
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_pwa():
    """Serve the standalone frontend."""
    return FileResponse("index.html")


@app.get("/manifest.json")
def serve_manifest():
    return FileResponse("manifest.json")


@app.get("/service-worker.js")
def serve_service_worker():
    return FileResponse("service-worker.js")

# ── Engine Initialization ──────────────────────────────────────────────────
engine = TriageEngine()


class TriageRequest(BaseModel):
    symptoms: str = Field(..., description="Free-text symptoms (English or Hindi)")
    age: int = Field(30, ge=0, le=120, description="Patient age")
    gender: str = Field("Male", description="Male or Female")
    language: str = Field("en", description="'en' for English, 'hi' for Hindi")


class TriageResponse(BaseModel):
    urgency: str
    disease: str | None
    confidence: float
    detected_symptoms: list[str]
    advice: str
    explanation: str
    shap_text: str
    top3: list[dict]
    is_flag: bool
    formatted_markdown: str
    severity_score: int
    followup: str


@app.post("/triage", response_model=TriageResponse)
def triage(req: TriageRequest):
    """Run symptom triage and return structured result."""
    gender_int = 1 if req.gender.lower() in ("male", "पुरुष") else 0
    result = engine.predict(
        req.symptoms, age=req.age, gender=gender_int, lang=req.language
    )
    formatted = format_result(result, req.language)
    return TriageResponse(
        urgency=result["urgency"],
        disease=result["disease"],
        confidence=result["confidence"],
        detected_symptoms=result["detected"],
        advice=result["advice"],
        explanation=result["explanation"],
        shap_text=result.get("shap_text", ""),
        top3=result["top3"],
        is_flag=result["is_flag"],
        formatted_markdown=formatted,
        severity_score=result.get("severity_score", 0),
        followup=result.get("followup", ""),
    )


@app.post("/triage/voice")
async def triage_voice(file: UploadFile = File(...), lang: str = Form("en")):
    """Accept an audio blob from the PWA, run Vosk offline ASR, return text."""
    temp_raw = None
    temp_wav = None
    try:
        # 1. Save raw blob
        suffix = os.path.splitext(file.filename)[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            content = await file.read()
            f.write(content)
            temp_raw = f.name
        
        # 2. Convert to 16kHz mono WAV (Vosk requirement)
        if not AudioSegment:
            return {"error": "pydub not installed", "text": "Backend error: pydub missing"}
            
        fd, temp_wav = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        
        try:
            audio = AudioSegment.from_file(temp_raw)
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            audio.export(temp_wav, format="wav")
        except Exception as e:
            msg = "FFmpeg missing or invalid audio format."
            print(f"Conversion failed: {e}")
            return {"error": msg, "text": f"Error: {msg}"}

        # 3. Transcribe
        txt = transcribe_audio_file(temp_wav, lang=lang)
        
        return {"text": txt}
    except Exception as e:
        print(traceback.format_exc())
        return {"error": str(e), "text": ""}
    finally:
        if temp_raw and os.path.exists(temp_raw): os.remove(temp_raw)
        if temp_wav and os.path.exists(temp_wav): os.remove(temp_wav)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
