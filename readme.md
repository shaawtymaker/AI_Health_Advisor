<p align="center">
  <img src="static/logo.png" alt="Rural Health Triage Logo" width="120"/>
</p>

<h1 align="center">🏥 Rural Health Triage Assistant</h1>

<p align="center">
  <strong>AI-powered, offline-first, dual-language (English + Hindi) symptom triage system built for rural healthcare.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Version-4.0-blue?style=for-the-badge" alt="Version"/>
  <img src="https://img.shields.io/badge/Offline-100%25-brightgreen?style=for-the-badge" alt="Offline"/>
  <img src="https://img.shields.io/badge/PWA-Ready-orange?style=for-the-badge" alt="PWA"/>
  <img src="https://img.shields.io/badge/Languages-English%20%2B%20Hindi-purple?style=for-the-badge" alt="Languages"/>
  <img src="https://img.shields.io/badge/Accuracy-91.4%25-green?style=for-the-badge" alt="Accuracy"/>
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Screenshots](#-screenshots)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [ML Model Details](#-ml-model-details)
- [Voice System](#-voice-system-stt--tts)
- [API Reference](#-api-reference)
- [Test Cases](#-test-cases)
- [Configuration](#-configuration)
- [Contributing](#-contributing)
- [Disclaimer](#%EF%B8%8F-disclaimer)
- [License](#-license)

---

## 🌍 Overview

The **Rural Health Triage Assistant** is a Progressive Web App (PWA) designed for healthcare workers and patients in rural and remote areas where:

- **Internet connectivity is unreliable or absent**
- **Trained medical professionals are scarce**
- **Patients speak Hindi or regional dialects**
- **Quick triage decisions can save lives**

The system runs an **XGBoost classifier** locally — no cloud, no API keys, no data leaves the device. A patient describes their symptoms in natural language (typed or spoken), and the system returns an urgency assessment, probable diagnosis, severity score, and actionable medical advice in their language.

---

## ✨ Key Features

### 🧠 AI-Powered Triage Engine
- **15 disease** classifier trained on WHO/CDC-aligned symptom profiles
- **91.4% accuracy** with 5-fold stratified cross-validation (93.0% CV mean)
- **SHAP explainability** — see which symptoms drove the prediction
- **Differential diagnosis** — top 3 possible conditions with confidence scores

### 🚨 Emergency Detection
- **Red-flag rules** catch life-threatening emergencies (heart attack, seizures, severe bleeding, unconsciousness, respiratory failure) and override the ML model
- **Severity scoring** (0–100) combines ML confidence, urgency tier, and red-flag detection

### 🗣️ Offline Voice Input (Speech-to-Text)
- **Vosk ASR engine** — fully offline, no internet required
- Supports **English** and **Hindi** transcription models
- **Audio normalization** — browser WebM/OGG blobs are auto-converted to 16kHz mono WAV via `pydub` + `static-ffmpeg`
- Pulsing microphone UI with real-time status indicators

### 🔊 Voice Announcements (Text-to-Speech)
- Results are **read aloud** using the Web Speech API
- Speaks the diagnosis, urgency level, and medical advice
- Automatic language selection (English `en-US` / Hindi `hi-IN`)
- Toggle play/stop with the 🔊 button on the results screen

### 🌐 Dual Language Support
- Full **English ↔ Hindi** runtime switching
- Bilingual keyword engine: understands `"bukhar"`, `"बुखार"`, and `"fever"` identically
- Localized medical advice, urgency labels, and follow-up questions

### 🤔 Dynamic Follow-Up Questions
- Detects **vague symptoms** (e.g., just "fever") and prompts for clarification
- Asks medically relevant follow-ups: *"How many days? Continuous or intermittent?"*
- Rule-based triggers defined in `data/followups.json`

### 🎨 Premium Glassmorphic PWA
- **3D Spline background** with transparent, frosted-glass UI cards
- Dark typography on light translucent panels for maximum readability
- Smooth page transitions, micro-animations, and hover effects
- Installable as a native-like app on mobile and desktop

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Browser (PWA)                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │ index.html│  │ app.js   │  │ styles.css           │  │
│  │ (3 pages)│  │ (logic)  │  │ (glassmorphism)      │  │
│  └──────────┘  └────┬─────┘  └──────────────────────┘  │
│                     │ fetch()                           │
└─────────────────────┼───────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────┐
│              FastAPI Backend (api.py)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ POST /triage │  │ POST         │  │ GET /         │  │
│  │ (text input) │  │ /triage/voice│  │ (serve PWA)  │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┘  │
│         │                 │                             │
│         ▼                 ▼                             │
│  ┌─────────────┐  ┌──────────────┐                     │
│  │ engine/     │  │ voice/stt.py │                     │
│  │ triage.py   │  │ (Vosk ASR)   │                     │
│  │ keywords.py │  └──────────────┘                     │
│  │ red_flags.py│                                       │
│  │ severity.py │                                       │
│  │ followup.py │                                       │
│  │ formatter.py│                                       │
│  └──────┬──────┘                                       │
│         │                                              │
│  ┌──────▼──────┐  ┌──────────────┐                     │
│  │ XGBoost     │  │ SHAP         │                     │
│  │ Classifier  │  │ Explainer    │                     │
│  └─────────────┘  └──────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **ML Engine** | XGBoost 2.0 | Multi-class disease classification |
| **Explainability** | SHAP | Feature importance for each prediction |
| **Data Balancing** | imbalanced-learn (SMOTE) | Class balance during training |
| **API Server** | FastAPI + Uvicorn | REST API with async support |
| **Voice (STT)** | Vosk | Offline speech-to-text (English + Hindi) |
| **Audio Processing** | pydub + static-ffmpeg | Browser audio → WAV conversion |
| **Voice (TTS)** | Web Speech API | Browser-native text-to-speech |
| **Frontend** | Vanilla HTML/CSS/JS | Zero-dependency PWA |
| **3D Graphics** | Spline | Animated background scene |
| **Data Format** | Pydantic | Request/response validation |

---

## 🚀 Installation

### Prerequisites

- **Python 3.9+** (tested on 3.11)
- **pip** (Python package manager)
- **~500 MB disk space** (for Vosk models)

### Step 1: Clone the Repository

```bash
git clone https://github.com/shaawtymaker/AI_Health_Advisor.git
cd AI_Health_Advisor
```

### Step 2: Install Python Dependencies

```bash
pip install -r requirements.txt
pip install pydub static-ffmpeg
```

### Step 3: Download Vosk Models (for Voice Input)

Run the automated setup script:

```bash
python scripts/setup_vosk.py
```

This downloads and extracts:
- `vosk-model-small-en-us-0.15` (~40 MB) — English
- `vosk-model-hi-0.22` (~450 MB) — Hindi

> **Note:** If you already have the models, place them in the `voice/` directory.

### Step 4: Run the Application

```bash
python api.py
```

The app launches at **`http://localhost:8000`** 🚀

---

## 📖 Usage

### Basic Workflow

1. **Select Language** — Choose English or Hindi on the home screen
2. **Start Triage** — Click the primary action button
3. **Enter Symptoms** — Type, use quick-select chips, or tap 🎤 to speak
4. **Set Age & Gender** — For demographic-adjusted predictions
5. **Evaluate** — Click "Evaluate Symptoms" to run the AI
6. **View Results** — See urgency level, diagnosis, severity score, and advice
7. **Listen** — Click 🔊 to hear the results read aloud

### Voice Input

1. Tap the **🎤 microphone** button
2. Speak your symptoms naturally (English or Hindi)
3. The button pulses red while recording
4. Tap again to stop — transcribed text appears in the input box
5. Edit if needed, then evaluate

### Example Inputs

| Input | Language | Expected Result |
|-------|----------|----------------|
| `"high fever, rash, joint pain, eye pain"` | English | 🔴 Dengue (RED) |
| `"bukhar hai, sardard aur thandi lag rahi hai"` | Hindi | 🔴 Malaria (RED) |
| `"cough for 3 weeks, night sweats, weight loss"` | English | 🟡 Tuberculosis (YELLOW) |
| `"naak beh raha hai, chheenk aa rahi hai"` | Hindi | 🟢 Common Cold (GREEN) |
| `"chest pain and sweating"` | English | 🔴 EMERGENCY (Red Flag) |
| `"behosh ho gaya, saans nahi aa rahi"` | Hindi | 🔴 EMERGENCY (Red Flag) |

---

## 📂 Project Structure

```
AI_Health_Advisor/
│
├── api.py                          # FastAPI entry point (run this)
├── train_model.py                  # Model training pipeline (XGBoost + SMOTE)
├── index.html                      # Main PWA HTML (3-page SPA)
├── manifest.json                   # PWA manifest for install
├── service-worker.js               # Offline caching service worker
├── requirements.txt                # Python dependencies
├── test_cases.json                 # 12 verified test scenarios
├── clinics.json                    # Nearby clinic data (stub)
│
├── engine/                         # 🧠 Core AI Engine
│   ├── __init__.py
│   ├── triage.py                   # TriageEngine class — prediction pipeline
│   ├── keywords.py                 # Bilingual symptom keyword mapping (34 symptoms)
│   ├── red_flags.py                # Emergency detection rules (5 rules)
│   ├── severity.py                 # Severity scorer (0–100)
│   ├── followup.py                 # Dynamic follow-up question engine
│   └── formatter.py                # Markdown result formatter
│
├── voice/                          # 🎙️ Offline Voice Engine
│   ├── __init__.py
│   ├── stt.py                      # Vosk speech-to-text (file + mic modes)
│   ├── vosk-model-small-en-us-0.15/ # English ASR model (~40 MB)
│   └── vosk-model-hi-0.22/         # Hindi ASR model (~450 MB)
│
├── models/                         # 📊 Trained ML Artifacts
│   ├── triage_model.json           # XGBoost model (JSON serialized)
│   ├── triage_model.onnx           # ONNX export for cross-platform inference
│   ├── metadata.json               # Feature names, diseases, urgency mapping
│   ├── evaluation_report.json      # Accuracy, F1, confusion matrix
│   └── training_data.csv           # Reproducible training dataset
│
├── data/                           # 📁 Knowledge Base & Datasets
│   ├── diseases.json               # Disease info + bilingual advice (15 diseases)
│   ├── followups.json              # Follow-up question triggers (5 symptoms)
│   ├── medical_knowledge_base.json # Extended medical reference
│   ├── Training.csv                # Primary training dataset
│   ├── Testing.csv                 # Held-out test set
│   └── Symptom2Disease.csv         # Kaggle symptom-disease mapping
│
├── static/                         # 🎨 Frontend Assets
│   ├── css/styles.css              # Glassmorphism design system
│   └── js/app.js                   # PWA logic (navigation, API, voice, TTS)
│
├── i18n/                           # 🌐 Internationalization
│   ├── en.json                     # English UI strings
│   └── hi.json                     # Hindi UI strings
│
└── scripts/                        # 🔧 Utilities
    └── setup_vosk.py               # Automated Vosk model downloader
```

---

## 🤖 ML Model Details

### Training Pipeline (`train_model.py`)

The model is trained on **medically-grounded synthetic data** aligned with WHO/CDC symptom profiles:

| Parameter | Value |
|-----------|-------|
| Algorithm | XGBoost (Gradient Boosting) |
| Estimators | 200 trees |
| Max Depth | 5 |
| Learning Rate | 0.1 |
| Features | 34 binary symptoms + age + gender = **36** |
| Classes | **15 diseases** |
| Training Samples | **22,410** |
| Test Samples | **3,417** |
| Test Accuracy | **91.4%** |
| CV Accuracy (5-fold) | **93.0% ± 0.4%** |
| Class Balancing | SMOTE oversampling |

### Diseases Covered

| Urgency | Disease | F1 Score |
|---------|---------|----------|
| 🔴 RED | Malaria | 0.946 |
| 🔴 RED | Dengue | 0.985 |
| 🔴 RED | Pneumonia | 0.914 |
| 🔴 RED | Heat Stroke | 0.970 |
| 🔴 RED | Cholera | 0.858 |
| 🟡 YELLOW | Typhoid | 0.885 |
| 🟡 YELLOW | Tuberculosis | 0.983 |
| 🟡 YELLOW | Gastroenteritis | 0.895 |
| 🟡 YELLOW | Anemia | 0.945 |
| 🟡 YELLOW | Jaundice | 0.902 |
| 🟡 YELLOW | UTI | 0.749 |
| 🟡 YELLOW | Asthma Attack | 0.885 |
| 🟢 GREEN | Common Cold | 0.959 |
| 🟢 GREEN | Flu | 0.874 |
| 🟢 GREEN | Chickenpox | 0.914 |

### Emergency Red Flags (Override Rules)

These bypass the ML model and immediately trigger 🔴 **RED** urgency:

| Rule | Trigger Keywords |
|------|-----------------|
| Loss of Consciousness | `unconscious`, `behosh`, `बेहोश`, `not responding` |
| Heart Attack Signs | `chest pain` + `sweating` (combo rule) |
| Seizures | `seizure`, `daura`, `दौरा`, `mirgi`, `मिर्गी` |
| Severe Bleeding | `severe bleeding`, `bahut khoon`, `बहुत खून` |
| Respiratory Failure | `cant breathe`, `saans nahi`, `सांस नहीं` |

### Severity Score (0–100)

```
Score = Base(urgency) + Confidence(×15) + SymptomCount(max 5)
```
- RED base = 80, YELLOW = 50, GREEN = 20
- Red-flag override → instant **100**

---

## 🎙️ Voice System (STT + TTS)

### Speech-to-Text (Vosk Offline ASR)

```
Browser Mic → MediaRecorder (WebM) → POST /triage/voice
    → pydub converts to 16kHz mono WAV
    → Vosk KaldiRecognizer transcribes
    → Text returned to frontend
```

**Models Required:**
| Language | Model | Size | Source |
|----------|-------|------|--------|
| English | `vosk-model-small-en-us-0.15` | ~40 MB | [Download](https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip) |
| Hindi | `vosk-model-hi-0.22` | ~450 MB | [Download](https://alphacephei.com/vosk/models/vosk-model-hi-0.22.zip) |

### Text-to-Speech (Web Speech API)

The browser's native `speechSynthesis` API reads the diagnosis and advice aloud:
- English voice: `en-US`
- Hindi voice: `hi-IN` (requires Hindi voice pack on the device)
- Controlled via the 🔊 button on the results page

---

## 📡 API Reference

### `POST /triage`

Text-based symptom triage.

**Request Body:**
```json
{
  "symptoms": "high fever, headache, chills",
  "age": 30,
  "gender": "Male",
  "language": "en"
}
```

**Response:**
```json
{
  "urgency": "RED",
  "disease": "malaria",
  "confidence": 0.87,
  "severity_score": 93,
  "detected_symptoms": ["fever", "high_fever", "headache", "chills"],
  "advice": "High risk of malaria. Go to hospital immediately for a blood test.",
  "explanation": "Based on symptoms: fever, high_fever, headache, chills",
  "shap_text": "**Top factors:** high_fever (+0.23), chills (+0.18), headache (+0.12)",
  "top3": [
    {"disease": "malaria", "prob": 0.87, "urgency": "RED"},
    {"disease": "dengue", "prob": 0.08, "urgency": "RED"},
    {"disease": "typhoid", "prob": 0.03, "urgency": "YELLOW"}
  ],
  "is_flag": false,
  "followup": "",
  "formatted_markdown": "..."
}
```

### `POST /triage/voice`

Voice-based symptom input. Accepts audio blob, returns transcribed text.

**Request:** `multipart/form-data`
- `file`: Audio blob (WebM/OGG/WAV)
- `lang`: `"en"` or `"hi"`

**Response:**
```json
{
  "text": "I have a high fever and a severe headache"
}
```

---

## 🧪 Test Cases

The project includes **12 verified test scenarios** in `test_cases.json`:

| # | Input | Expected | Urgency |
|---|-------|----------|---------|
| 1 | `bukhar hai, sardard aur thandi lag rahi hai` | Malaria/Dengue | 🔴 RED |
| 2 | `high fever, rash, joint pain, eye pain` | Dengue | 🔴 RED |
| 3 | `naak beh raha hai, chheenk aa rahi hai` | Common Cold | 🟢 GREEN |
| 4 | `cough for 3 weeks, night sweats, weight loss` | Tuberculosis | 🟡 YELLOW |
| 5 | `pet dard, ulti, dast, bahut kamzori` | Gastroenteritis | 🟡 YELLOW |
| 6 | `chest pain and sweating` | 🚨 Emergency | 🔴 RED |
| 7 | `halka bukhar aur khansi` | Cold/Flu | 🟢 GREEN |
| 8 | `behosh ho gaya, saans nahi aa rahi` | 🚨 Emergency | 🔴 RED |
| 9 | `tired, weak, pale skin, dizzy, no energy` | Anemia | 🟡 YELLOW |
| 10 | `high fever, confusion, dizziness, rapid breathing` | Heat Stroke | 🔴 RED |
| 11 | `daura pad gaya, mirgi jaisi haalat` | 🚨 Emergency | 🔴 RED |
| 12 | `fever, body ache, muscle pain, sore throat` | Flu | 🟢 GREEN |

---

## ⚙️ Configuration

### Extending the Knowledge Base

Add new diseases to `data/diseases.json`:
```json
{
  "new_disease": {
    "en": "Disease Name (English)",
    "hi": "रोग का नाम (Hindi)",
    "adv_en": "English medical advice...",
    "adv_hi": "Hindi चिकित्सा सलाह..."
  }
}
```

### Adding Follow-Up Questions

Edit `data/followups.json`:
```json
{
  "symptom_name": {
    "en": "Follow-up question in English?",
    "hi": "हिंदी में अनुवर्ती प्रश्न?",
    "trigger_if_missing": ["related_symptom_1", "related_symptom_2"]
  }
}
```

### Retraining the Model

```bash
python train_model.py
```

This regenerates `models/triage_model.json`, `models/metadata.json`, and `models/evaluation_report.json`.

---

## 🤝 Contributing

1. **Fork** the repository
2. Create a **feature branch** (`git checkout -b feature/new-disease`)
3. **Commit** your changes (`git commit -m "Add diabetes detection"`)
4. **Push** to the branch (`git push origin feature/new-disease`)
5. Open a **Pull Request**

### Areas for Contribution

- **New diseases** — Add symptom profiles in `train_model.py` and info in `diseases.json`
- **Regional languages** — Add Kannada, Telugu, Tamil, etc. to `engine/keywords.py`
- **Model improvements** — Experiment with different classifiers or feature engineering
- **UI/UX** — Accessibility improvements, new themes, mobile optimization

---

## ⚠️ Disclaimer

> **This application is NOT a medical device.**
>
> The Rural Health Triage Assistant is designed for **educational and informational purposes only**. It uses statistical models to suggest possibilities based on reported symptoms and should **never replace** the judgment of a qualified healthcare professional.
>
> **Always consult a doctor for medical emergencies.** If you or someone else is experiencing a life-threatening situation, call emergency services immediately.

---

## 📄 License

This project is open-source. See [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>Built with ❤️ for rural healthcare accessibility</strong>
  <br/>
  <em>100% Offline · No Data Leaves Your Device · Bilingual by Design</em>
</p>
