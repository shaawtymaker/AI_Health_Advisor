"""
Rural Health Triage Assistant — Gradio MVP
Offline AI symptom checker (English + Hindi)
Features: Text/Voice input, SHAP explainability, TTS output, bilingual UI
Run:  python app.py
"""

import json, os, textwrap, tempfile, traceback
import numpy as np
import gradio as gr
from xgboost import XGBClassifier


# --- NEW V4 IMPORTS ---
import json, os
from engine.triage import TriageEngine
from engine.formatter import format_result

with open(os.path.join(os.path.dirname(__file__), 'i18n', 'en.json'), encoding='utf-8') as f:
    UI_EN = json.load(f)
with open(os.path.join(os.path.dirname(__file__), 'i18n', 'hi.json'), encoding='utf-8') as f:
    UI_HI = json.load(f)
UI_TEXT = {'en': UI_EN, 'hi': UI_HI}

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TTS ENGINE  (plays directly via speakers — no file returned)          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def speak_advice(text: str, lang: str = "en"):
    """Speak the advice text aloud using pyttsx3 (runs in background thread)."""
    import threading
    def _speak():
        try:
            import pyttsx3
            tts = pyttsx3.init()
            voices = tts.getProperty("voices")
            for v in voices:
                name_lower = v.name.lower()
                if lang == "hi" and "hindi" in name_lower:
                    tts.setProperty("voice", v.id)
                    break
                elif lang == "en" and "english" in name_lower:
                    tts.setProperty("voice", v.id)
                    break
            tts.setProperty("rate", 150)
            clean = text.replace("**", "").replace("*", "").replace("#", "").replace("---", "")
            if len(clean) > 400:
                clean = clean[:400]
            tts.say(clean)
            tts.runAndWait()
        except Exception as e:
            print(f"[TTS] Error: {e}")
    threading.Thread(target=_speak, daemon=True).start()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  VOICE TRANSCRIPTION                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def transcribe_voice(audio_path: str, lang: str = "en") -> str:
    """Attempt offline transcription using Vosk. Return text or error."""
    if not audio_path:
        return ""
    try:
        from voice.stt import transcribe_audio_file
        return transcribe_audio_file(audio_path, lang=lang)
    except FileNotFoundError as e:
        return f"[Voice model not found: {e}]"
    except Exception as e:
        return f"[Voice error: {e}]"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GRADIO UI                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

engine = TriageEngine()

def process_audio_input(audio_path, current_text, language):
    """Triggered when recording stops. Transcribes and appends to textbox."""
    if not audio_path:
        return current_text
        
    lang = "hi" if language == "हिन्दी" else "en"
    
    import shutil
    stable_path = os.path.join(tempfile.gettempdir(), "triage_voice_input.wav")
    try:
        shutil.copy2(audio_path, stable_path)
        transcribed = transcribe_voice(stable_path, lang)
        
        if transcribed and not transcribed.startswith("["):
            # Append if there's already text, otherwise replace
            if current_text and current_text.strip():
                return f"{current_text} {transcribed}"
            return transcribed
        return current_text  # If failed, return what they had
    except Exception as e:
        print(f"[VOICE] Error in auto-transcribe: {e}")
        return current_text

def run_triage(symptoms_text, age, gender_choice, language):
    lang = "hi" if language == "हिन्दी" else "en"
    gender = 1 if gender_choice in ("Male", "पुरुष") else 0
    age = int(age) if age else 30

    print(f"\n{'='*50}")
    print(f"[TRIAGE] text='{symptoms_text}', age={age}, lang={lang}")

    # Voice is now pre-processed into symptoms_text
    result = engine.predict(symptoms_text, age=age, gender=gender, lang=lang)
    print(f"[TRIAGE] urgency={result['urgency']}, disease={result['disease']}, detected={result['detected']}")
    formatted = format_result(result, lang)

    # Speak advice aloud in background (non-blocking)
    if result.get("advice"):
        speak_advice(
            f"{result.get('disease', '')}. {result.get('advice', '')}",
            lang
        )

    return formatted


# ── Build the interface ───────────────────────────────────────────────────
EXAMPLE_CASES = [
    ["bukhar hai, sardard aur thandi lag rahi hai",     45, "Male",   "English"],
    ["high fever, rash, joint pain, eye pain",          32, "Female", "English"],
    ["naak beh raha hai, chheenk aa rahi hai",          25, "Female", "हिन्दी"],
    ["cough for 3 weeks, night sweats, weight loss",    55, "Male",   "English"],
    ["pet dard, ulti, dast, bahut kamzori",             40, "Female", "हिन्दी"],
    ["chest pain and sweating",                         60, "Male",   "English"],
    ["halka bukhar aur khansi",                         28, "Male",   "हिन्दी"],
]

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* ── Animated Background ────────────────────────────────────────── */
body, .gradio-container {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #0b3b42);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    font-family: 'Outfit', 'Noto Sans Devanagari', sans-serif !important;
    color: #edf2f4 !important;
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Container & Glassmorphism ─────────────────────────────────── */
.gradio-container {
    max-width: 850px !important;
    margin: 40px auto !important;
    padding: 30px !important;
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
}

/* ── Header ────────────────────────────────────────────────────── */
.gradio-container h1 {
    text-align: center;
    color: #ffffff !important;
    font-weight: 700;
    font-size: 2.4em;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    margin-bottom: 5px;
}
.gradio-container h3 {
    text-align: center;
    color: #cfd8dc !important;
    font-weight: 300;
    margin-top: 0;
}

/* ── Cards / Sections (The internal Gradio blocks) ─────────────── */
div.gr-box, div.gr-panel, div.gr-form, .gradio-html {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
}

/* Labels */
span.svelte-1gfkn6j {
    color: #e0e0e0 !important;
    font-weight: 500 !important;
}

/* ── Input fields ────────────────────────────────────────────── */
textarea, input[type="number"], select, .gr-input {
    background: rgba(0, 0, 0, 0.2) !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    font-size: 1.05em !important;
    transition: all 0.3s ease !important;
}
textarea:focus, input:focus, select:focus, .gr-input:focus {
    border-color: #48cae4 !important;
    box-shadow: 0 0 0 3px rgba(72, 202, 228, 0.3) !important;
    background: rgba(0, 0, 0, 0.4) !important;
    outline: none !important;
}

/* Placeholder color */
::placeholder {
    color: rgba(255,255,255,0.4) !important;
}

/* Dropdown list items inside select */
.gr-dropdown-list {
    background: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}
.gr-dropdown-list li {
    color: white !important;
}
.gr-dropdown-list li:hover {
    background: #334155 !important;
}

/* ── Buttons ─────────────────────────────────────────────────── */
button.primary {
    background: linear-gradient(135deg, #00b4d8, #0077b6) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 1.2em !important;
    padding: 14px 28px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(0, 119, 182, 0.4) !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
}
button.primary:hover {
    background: linear-gradient(135deg, #48cae4, #0096c7) !important;
    box-shadow: 0 6px 20px rgba(0, 119, 182, 0.6) !important;
    transform: translateY(-2px);
}
button.primary:active {
    transform: translateY(1px);
    box-shadow: 0 2px 10px rgba(0, 119, 182, 0.4) !important;
}

/* Audio component styling */
.audio-recorder {
    border-color: rgba(255,255,255,0.2) !important;
    background: rgba(0,0,0,0.1) !important;
}
button[aria-label="Record audio"] {
    background: rgba(239, 68, 68, 0.2) !important;
    border: 1px solid rgba(239, 68, 68, 0.5) !important;
}
button[aria-label="Record audio"]:hover {
    background: rgba(239, 68, 68, 0.4) !important;
}

/* ── Markdown Result Output ──────────────────────────────────── */
.output-markdown {
    color: #edf2f4 !important;
    font-size: 1.1em;
}
.output-markdown h2 {
    padding: 16px;
    border-radius: 10px;
    margin-top: 10px;
    font-weight: 700;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2);
}
.output-markdown h3 {
    color: #90e0ef !important;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding-bottom: 5px;
}
.output-markdown p {
    line-height: 1.6;
}

/* Examples rendering */
.gr-samples-gallery button {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e0e0e0 !important;
    transition: all 0.2s;
}
.gr-samples-gallery button:hover {
    background: rgba(72,202,228,0.2) !important;
    border-color: rgba(72,202,228,0.5) !important;
}

/* ── Footer ──────────────────────────────────────────────────── */
footer { 
    opacity: 0.5;
    text-align: center;
    border-top: 1px solid rgba(255,255,255,0.1) !important;
    padding-top: 15px;
}
/* Hide default Gradio footer logo */
footer .svelte-15x3029 {
    display: none !important;
}
"""

with gr.Blocks(css=CSS, title="🏥 Rural Health Triage") as demo:

    # ── Header ────────────────────────────────────────────────
    title_md    = gr.Markdown(UI_TEXT["en"]["title"])
    subtitle_md = gr.Markdown(UI_TEXT["en"]["subtitle"])

    # ── Language picker ───────────────────────────────────────
    with gr.Row():
        lang_dd = gr.Dropdown(
            ["English", "हिन्दी"], value="English",
            label=UI_TEXT["en"]["lang_label"], scale=1,
        )

    # ── Patient info ──────────────────────────────────────────
    with gr.Row():
        age_box = gr.Number(
            value=30, label=UI_TEXT["en"]["age_label"],
            minimum=0, maximum=120, precision=0, scale=1,
        )
        gender_dd = gr.Dropdown(
            UI_TEXT["en"]["gender_choices"], value="Male",
            label=UI_TEXT["en"]["gender_label"], scale=1,
        )

    # ── Symptom text input ────────────────────────────────────
    symptom_box = gr.Textbox(
        lines=3,
        placeholder=UI_TEXT["en"]["symptom_placeholder"],
        label=UI_TEXT["en"]["symptom_label"],
    )

    # ── Voice input ───────────────────────────────────────────
    voice_input = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label=UI_TEXT["en"]["voice_label"],
    )

    # ── Submit ────────────────────────────────────────────────
    submit_btn = gr.Button(
        UI_TEXT["en"]["submit"], variant="primary", size="lg",
    )

    # ── Outputs ───────────────────────────────────────────────
    output_md = gr.Markdown(label=UI_TEXT["en"]["result_label"])

    # ── Language toggle → update all labels ───────────────────
    def update_ui_language(language):
        lang = "hi" if language == "हिन्दी" else "en"
        t = UI_TEXT[lang]
        return (
            gr.update(value=t["title"]),           # title_md
            gr.update(value=t["subtitle"]),         # subtitle_md
            gr.update(label=t["age_label"]),         # age_box
            gr.update(label=t["gender_label"],       # gender_dd
                      choices=t["gender_choices"],
                      value=t["gender_choices"][0]),
            gr.update(label=t["symptom_label"],      # symptom_box
                      placeholder=t["symptom_placeholder"]),
            gr.update(label=t["voice_label"]),        # voice_input
            gr.update(value=t["submit"]),             # submit_btn
            gr.update(label=t["result_label"]),       # output_md
        )

    lang_dd.change(
        fn=update_ui_language,
        inputs=[lang_dd],
        outputs=[title_md, subtitle_md, age_box, gender_dd,
                 symptom_box, voice_input, submit_btn, output_md],
    )

    # ── Submit action ─────────────────────────────────────────
    # Transcript populated dynamically when user stops recording
    voice_input.stop_recording(
        fn=process_audio_input,
        inputs=[voice_input, symptom_box, lang_dd],
        outputs=[symptom_box]
    )

    submit_btn.click(
        fn=run_triage,
        inputs=[symptom_box, age_box, gender_dd, lang_dd],
        outputs=[output_md],
    )

    # ── Example cases ─────────────────────────────────────────
    gr.Examples(
        examples=EXAMPLE_CASES,
        inputs=[symptom_box, age_box, gender_dd, lang_dd],
        outputs=[output_md],
        fn=lambda s, a, g, l: run_triage(s, a, g, l),
        cache_examples=False,
        label=UI_TEXT["en"]["examples_label"],
    )

    # ── Footer ────────────────────────────────────────────────
    gr.Markdown(UI_TEXT["en"]["footer"])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)