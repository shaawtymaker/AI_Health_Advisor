"""
One-time script: generates synthetic patient data and trains the XGBoost triage model.
Run:  python train_model.py
"""

import json, os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# ── Symptom feature list (34 binary features + age + gender) ──────────────
SYMPTOMS = [
    "fever","high_fever","chills","headache","severe_headache",
    "body_ache","joint_pain","muscle_pain","cough","chronic_cough",
    "sore_throat","runny_nose","sneezing","difficulty_breathing",
    "rapid_breathing","chest_pain","abdominal_pain","nausea",
    "vomiting","diarrhea","rash","eye_pain","dark_urine",
    "blood_in_sputum","night_sweats","weight_loss","fatigue",
    "weakness","dizziness","sweating","confusion","bleeding",
    "pale_skin","dehydration_signs",
]
FEATURES = SYMPTOMS + ["age", "gender"]

# ── Disease definitions: symptom probabilities + triage level ─────────────
DISEASES = {
    "malaria": {
        "urgency": "RED",
        "p": {"fever":.95,"high_fever":.70,"chills":.90,"headache":.85,
              "sweating":.80,"body_ache":.70,"nausea":.50,"vomiting":.40,
              "dark_urine":.30,"fatigue":.60,"weakness":.50,"dizziness":.30},
    },
    "dengue": {
        "urgency": "RED",
        "p": {"fever":.95,"high_fever":.85,"severe_headache":.80,
              "eye_pain":.70,"joint_pain":.80,"muscle_pain":.75,
              "rash":.50,"nausea":.60,"vomiting":.40,"fatigue":.70,
              "weakness":.50,"bleeding":.15},
    },
    "typhoid": {
        "urgency": "YELLOW",
        "p": {"fever":.95,"high_fever":.60,"headache":.70,
              "abdominal_pain":.70,"weakness":.80,"fatigue":.70,
              "diarrhea":.40,"nausea":.40,"rash":.20,"body_ache":.50},
    },
    "tuberculosis": {
        "urgency": "YELLOW",
        "p": {"chronic_cough":.90,"cough":.95,"fever":.60,
              "night_sweats":.70,"weight_loss":.80,"blood_in_sputum":.40,
              "fatigue":.80,"chest_pain":.40,"weakness":.60},
    },
    "pneumonia": {
        "urgency": "RED",
        "p": {"cough":.90,"fever":.85,"high_fever":.50,
              "difficulty_breathing":.80,"rapid_breathing":.60,
              "chest_pain":.60,"fatigue":.70,"sweating":.30,
              "nausea":.20,"confusion":.15},
    },
    "common_cold": {
        "urgency": "GREEN",
        "p": {"runny_nose":.90,"sneezing":.85,"sore_throat":.70,
              "cough":.50,"fever":.30,"headache":.30,
              "body_ache":.20,"fatigue":.30},
    },
    "gastroenteritis": {
        "urgency": "YELLOW",
        "p": {"diarrhea":.90,"vomiting":.70,"abdominal_pain":.80,
              "nausea":.85,"fever":.40,"dehydration_signs":.40,
              "weakness":.50,"fatigue":.40},
    },
    "anemia": {
        "urgency": "YELLOW",
        "p": {"fatigue":.90,"weakness":.85,"pale_skin":.70,
              "dizziness":.60,"difficulty_breathing":.30,"headache":.30},
    },
    "heat_stroke": {
        "urgency": "RED",
        "p": {"high_fever":.90,"confusion":.70,"sweating":.30,
              "headache":.60,"nausea":.50,"vomiting":.30,
              "dizziness":.70,"rapid_breathing":.40,"weakness":.60},
    },
    "flu": {
        "urgency": "GREEN",
        "p": {"fever":.80,"high_fever":.30,"headache":.60,
              "body_ache":.70,"muscle_pain":.60,"cough":.60,
              "sore_throat":.50,"fatigue":.70,"runny_nose":.40,"chills":.40},
    },
}

DISEASE_NAMES = list(DISEASES.keys())

# ── Synthetic data generator ──────────────────────────────────────────────
def generate_dataset(n_per_disease: int = 500):
    rows, labels = [], []
    for idx, (name, info) in enumerate(DISEASES.items()):
        for _ in range(n_per_disease):
            row = {s: int(np.random.random() < info["p"].get(s, 0.05)) for s in SYMPTOMS}
            row["age"]    = np.random.randint(1, 80)
            row["gender"] = np.random.randint(0, 2)   # 0=F, 1=M
            rows.append(row)
            labels.append(idx)
    return pd.DataFrame(rows)[FEATURES], np.array(labels)

# ── Train & save ──────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Rural Health Triage — Model Training")
    print("=" * 55)

    X, y = generate_dataset(500)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"Train {len(X_tr)}  |  Test {len(X_te)}  |  Classes {len(DISEASE_NAMES)}")

    model = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.1,
        objective="multi:softprob", num_class=len(DISEASE_NAMES),
        eval_metric="mlogloss", random_state=42,
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    print(f"\nAccuracy: {accuracy_score(y_te, y_pred):.1%}\n")
    print(classification_report(y_te, y_pred, target_names=DISEASE_NAMES))

    os.makedirs("models", exist_ok=True)
    model.save_model("models/triage_model.json")

    meta = {
        "features": FEATURES,
        "symptoms": SYMPTOMS,
        "diseases": DISEASE_NAMES,
        "urgency": {n: d["urgency"] for n, d in DISEASES.items()},
    }
    with open("models/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    kb = os.path.getsize("models/triage_model.json") / 1024
    print(f"✅  Model saved  →  models/triage_model.json  ({kb:.0f} KB)")
    print("✅  Metadata     →  models/metadata.json")
    print("\nNext:  python app.py")

if __name__ == "__main__":
    main()