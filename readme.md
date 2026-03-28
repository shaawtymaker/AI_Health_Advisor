# 🏥 Rural Health Triage Assistant

> **AI-powered offline symptom checker for rural health workers and villagers.**
> Runs on any machine — no internet needed after setup. Directly supports **SDG 3** (Good Health & Well-being).

---

## ✨ Features

| Feature | Description |
|---|---|
| 🩺 **Symptom Triage** | XGBoost classifier diagnoses **15 diseases** with **91.5% accuracy** (93% CV) |
| 🌐 **Bilingual** | Full English + Hindi support — UI labels, advice, and output toggle dynamically |
| 🎤 **Voice Input** | Offline Hindi speech recognition via **Vosk ASR** (~50 MB model) |
| 🔊 **Text-to-Speech** | Advice read aloud via pyttsx3 (offline, no cloud) |
| 🧠 **Explainability** | **SHAP-powered** top-3 influential symptoms shown with every result |
| 🔴 **Red-Flag Rules** | Hybrid ML + rules: unconsciousness, seizures, chest pain+sweating → instant RED |
| 📞 **Ambulance Banner** | Prominent "Call 108" banner on emergency (RED) results |
| 🏥 **Nearest Clinics** | Offline clinic database with distance and phone numbers |
| 🔒 **Privacy** | 100% offline, no data stored, no cloud calls |
| 🔌 **REST API** | FastAPI `/triage` endpoint for frontend decoupling |

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Integrate Kaggle datasets (place CSVs in data/ first)
python integrate_kaggle.py

# 3. Train the model (generates hybrid real+synthetic dataset)
python train_model.py

# 4. Launch the UI at http://localhost:7860
python app.py
```

### Optional: REST API
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
# POST /triage  •  GET /health
```

---

## 📁 Project Structure

```
AI_Health_adviser/
├── app.py                  # Gradio web UI (main app)
├── api.py                  # FastAPI REST endpoint
├── train_model.py          # Model training pipeline (v3)
├── integrate_kaggle.py     # Maps Kaggle datasets → our 34 features
├── voice.py                # Vosk offline speech-to-text
├── clinics.json            # Clinic database (editable)
├── test_cases.json         # 12 structured test cases
├── requirements.txt        # Python dependencies
├── data/
│   ├── medical_knowledge_base.json  # WHO/CDC symptom profiles + vignettes
│   ├── Training.csv                 # Kaggle disease prediction (4920 rows)
│   ├── Testing.csv                  # Kaggle test set
│   ├── Disease_symptom_and_patient_profile_dataset.csv
│   ├── Symptom2Disease.csv          # NLP text descriptions
│   └── kaggle_mapped.csv           # Output: mapped to our features
├── models/
│   ├── triage_model.json   # Trained XGBoost model (~6.7 MB)
│   ├── metadata.json       # Feature/disease/urgency mappings
│   ├── evaluation_report.json  # Accuracy, F1, confusion matrix
│   └── training_data.csv   # Full training dataset for reproducibility
└── vosk-model-hi-0.22/     # Hindi ASR model (~50 MB, download separately)
```

---

## 🧪 Diseases Covered (15)

| Disease | Urgency | Key Symptoms |
|---|---|---|
| Malaria | 🔴 RED | Fever, chills, sweating, headache |
| Dengue | 🔴 RED | High fever, rash, joint/eye pain |
| Pneumonia | 🔴 RED | Cough, fever, breathing difficulty |
| Heat Stroke | 🔴 RED | High fever, confusion, dizziness |
| Cholera | 🔴 RED | Severe diarrhea, vomiting, dehydration |
| Typhoid | 🟡 YELLOW | Fever, abdominal pain, weakness |
| Tuberculosis | 🟡 YELLOW | Chronic cough, night sweats, weight loss |
| Gastroenteritis | 🟡 YELLOW | Diarrhea, vomiting, stomach pain |
| Anemia | 🟡 YELLOW | Fatigue, pale skin, dizziness |
| Jaundice | 🟡 YELLOW | Dark urine, fatigue, stomach pain |
| UTI | 🟡 YELLOW | Abdominal pain, dark urine, fever |
| Asthma Attack | 🟡 YELLOW | Breathing difficulty, cough, chest pain |
| Common Cold | 🟢 GREEN | Runny nose, sneezing, sore throat |
| Influenza | 🟢 GREEN | Fever, body ache, cough |
| Chickenpox | 🟢 GREEN | Rash, fever, fatigue |

---

## 📊 Model Performance

| Metric | Value |
|---|---|
| **Test Accuracy** | 91.5% |
| **CV Accuracy** | 93.0% ± 0.1% (5-fold stratified) |
| **Training data** | 17,083 samples (synthetic + Kaggle real) |
| **Model size** | ~6.7 MB (XGBoost JSON) |
| **Inference time** | <0.5s on CPU |
| **Emergency sensitivity** | ≥80% for all RED diseases |

### Data Sources
- **WHO/CDC** fact sheets for symptom probability profiles
- **Kaggle** Disease Prediction dataset (4,920 binary symptom rows)
- **India DHS NFHS-5** prevalence weights for realistic distribution
- **Clinical vignettes** from medical literature (17 bilingual cases)
- **SMOTE** oversampling for class balance

---

## 🔊 Voice Setup (Optional)

Download the Hindi Vosk model from [alphacephei.com/vosk/models](https://alphacephei.com/vosk/models) and place it at:
```
vosk-model-hi-0.22/vosk-model-hi-0.22/
```

---

## 🎯 SDG Alignment

- **SDG 3** — Good Health & Well-being: early diagnosis, reduced preventable deaths
- **SDG 9** — Innovation: affordable tech bridging urban/rural health gaps
- **SDG 10** — Reduced Inequalities: accessible to low-literacy users via voice + Hindi

---

## ⚠️ Disclaimer

This is a **health information tool**, not a medical device. It provides guidance only — always consult a health professional if symptoms worsen.
