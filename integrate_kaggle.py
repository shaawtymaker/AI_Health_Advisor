"""
Kaggle Dataset Integration for Rural Health Triage Model
=========================================================
Maps real Kaggle symptom-disease data to our 34-feature triage model.

Data sources:
  1. Training.csv         — 4920 rows, 132 binary symptoms, 41 diseases
  2. Disease_symptom_and_patient_profile_dataset.csv — age, gender, basic symptoms
  3. Symptom2Disease.csv  — NLP text descriptions

This script:
  - Filters diseases that match our 15-disease triage set
  - Maps Kaggle symptom columns → our 34 binary features
  - Exports a clean CSV ready for train_model.py
"""

import pandas as pd
import numpy as np
import os, json

# Our 34 symptom features
OUR_SYMPTOMS = [
    "fever","high_fever","chills","headache","severe_headache",
    "body_ache","joint_pain","muscle_pain","cough","chronic_cough",
    "sore_throat","runny_nose","sneezing","difficulty_breathing",
    "rapid_breathing","chest_pain","abdominal_pain","nausea",
    "vomiting","diarrhea","rash","eye_pain","dark_urine",
    "blood_in_sputum","night_sweats","weight_loss","fatigue",
    "weakness","dizziness","sweating","confusion","bleeding",
    "pale_skin","dehydration_signs",
]
FEATURES = OUR_SYMPTOMS + ["age", "gender"]

OUR_DISEASES = [
    "malaria","dengue","pneumonia","heat_stroke","cholera",
    "typhoid","tuberculosis","gastroenteritis","anemia","jaundice",
    "urinary_tract_infection","asthma_attack","common_cold","flu","chickenpox",
]

# ═══════════════════════════════════════════════════════════════════════════
#  MAPPING: Kaggle symptom names → our 34 features
# ═══════════════════════════════════════════════════════════════════════════

KAGGLE_SYMPTOM_MAP = {
    # Kaggle column name → our feature name(s)
    # Many-to-one: multiple Kaggle columns can map to one of ours

    # Fever
    "high_fever":           "high_fever",
    "mild_fever":           "fever",
    # Chills & shivering
    "chills":               "chills",
    "shivering":            "chills",
    # Head
    "headache":             "headache",
    "pain_behind_the_eyes": "eye_pain",
    "dizziness":            "dizziness",
    "altered_sensorium":    "confusion",
    "coma":                 "confusion",
    # Body
    "joint_pain":           "joint_pain",
    "muscle_pain":          "muscle_pain",
    "back_pain":            "body_ache",
    "neck_pain":            "body_ache",
    "muscle_weakness":      "weakness",
    "muscle_wasting":       "weakness",
    "weakness_in_limbs":    "weakness",
    "weakness_of_one_body_side": "weakness",
    "fatigue":              "fatigue",
    "lethargy":             "fatigue",
    "malaise":              "fatigue",
    "restlessness":         "fatigue",
    "weight_loss":          "weight_loss",
    "sweating":             "sweating",
    # Respiratory
    "cough":                "cough",
    "breathlessness":       "difficulty_breathing",
    "fast_heart_rate":      "rapid_breathing",
    "chest_pain":           "chest_pain",
    "phlegm":               "cough",
    "mucoid_sputum":        "cough",
    "rusty_sputum":         "blood_in_sputum",
    "blood_in_sputum":      "blood_in_sputum",
    "throat_irritation":    "sore_throat",
    "patches_in_throat":    "sore_throat",
    "runny_nose":           "runny_nose",
    "continuous_sneezing":  "sneezing",
    "congestion":           "runny_nose",
    "sinus_pressure":       "runny_nose",
    # GI
    "stomach_pain":         "abdominal_pain",
    "abdominal_pain":       "abdominal_pain",
    "belly_pain":           "abdominal_pain",
    "nausea":               "nausea",
    "vomiting":             "vomiting",
    "diarrhoea":            "diarrhea",
    "dehydration":          "dehydration_signs",
    "sunken_eyes":          "dehydration_signs",
    # Skin
    "skin_rash":            "rash",
    "nodal_skin_eruptions": "rash",
    "red_spots_over_body":  "rash",
    "blister":              "rash",
    "itching":              "rash",
    "yellowish_skin":       "pale_skin",
    "dischromic _patches":  "pale_skin",
    # Urine
    "dark_urine":           "dark_urine",
    "yellow_urine":         "dark_urine",
    "yellowing_of_eyes":    "dark_urine",
    "burning_micturition":  "dark_urine",
    "bladder_discomfort":   "abdominal_pain",
    "continuous_feel_of_urine": "abdominal_pain",
    # Blood
    "stomach_bleeding":     "bleeding",
    "bloody_stool":         "bleeding",
    "bruising":             "bleeding",
    # Other
    "toxic_look_(typhos)":  "fever",
    "swelling_of_stomach":  "abdominal_pain",
    "loss_of_appetite":     "nausea",
    "palpitations":         "rapid_breathing",
    "cramps":               "muscle_pain",
    "stiff_neck":           "body_ache",
}

# Kaggle disease name → our disease name
KAGGLE_DISEASE_MAP = {
    "Malaria":                  "malaria",
    "Dengue":                   "dengue",
    "Typhoid":                  "typhoid",
    "Tuberculosis":             "tuberculosis",
    "Pneumonia":                "pneumonia",
    "Common Cold":              "common_cold",
    "Gastroenteritis":          "gastroenteritis",
    "Chicken pox":              "chickenpox",
    "Jaundice":                 "jaundice",
    "Urinary tract infection":  "urinary_tract_infection",
    "Bronchial Asthma":         "asthma_attack",
    "hepatitis A":              "jaundice",       # maps to jaundice (liver)
    "Hepatitis B":              "jaundice",
    "Hepatitis C":              "jaundice",
    "Hepatitis D":              "jaundice",
    "Hepatitis E":              "jaundice",
    "Allergy":                  "common_cold",    # mild allergy → cold-like
}

# Patient Profile dataset disease mapping
PP_DISEASE_MAP = {
    "Malaria":                     "malaria",
    "Dengue Fever":                "dengue",
    "Typhoid Fever":               "typhoid",
    "Tuberculosis":                "tuberculosis",
    "Pneumonia":                   "pneumonia",
    "Common Cold":                 "common_cold",
    "Gastroenteritis":             "gastroenteritis",
    "Chickenpox":                  "chickenpox",
    "Cholera":                     "cholera",
    "Anemia":                      "anemia",
    "Influenza":                   "flu",
    "Asthma":                      "asthma_attack",
    "Hepatitis":                   "jaundice",
    "Hepatitis B":                 "jaundice",
    "Urinary Tract Infection":     "urinary_tract_infection",
    "Urinary Tract Infection (UTI)": "urinary_tract_infection",
    "Bronchitis":                  "flu",          # similar presentation
}


# ═══════════════════════════════════════════════════════════════════════════
#  PROCESS Training.csv (132 binary symptoms → our 34 features)
# ═══════════════════════════════════════════════════════════════════════════

def process_training_csv(path):
    """Convert Kaggle Training.csv to our feature space."""
    df = pd.read_csv(path)
    if "Unnamed: 133" in df.columns:
        df.drop("Unnamed: 133", axis=1, inplace=True)

    converted = []
    skipped_diseases = set()

    for _, row in df.iterrows():
        disease_kaggle = row["prognosis"].strip()
        if disease_kaggle not in KAGGLE_DISEASE_MAP:
            skipped_diseases.add(disease_kaggle)
            continue

        our_disease = KAGGLE_DISEASE_MAP[disease_kaggle]
        our_row = {s: 0 for s in OUR_SYMPTOMS}

        # Map Kaggle binary columns to our features
        for kaggle_col, our_col in KAGGLE_SYMPTOM_MAP.items():
            if kaggle_col in df.columns and row.get(kaggle_col, 0) == 1:
                our_row[our_col] = 1

        # Infer fever from high_fever or mild_fever
        if our_row["high_fever"] == 1:
            our_row["fever"] = 1

        # Add age/gender (not in Training.csv — sample realistically)
        our_row["age"] = np.random.randint(5, 70)
        our_row["gender"] = np.random.randint(0, 2)
        our_row["_disease"] = our_disease

        converted.append(our_row)

    print(f"   Training.csv: {len(converted)} rows mapped, "
          f"{len(skipped_diseases)} diseases skipped")
    if skipped_diseases:
        print(f"   Skipped: {sorted(skipped_diseases)}")

    return pd.DataFrame(converted)


# ═══════════════════════════════════════════════════════════════════════════
#  PROCESS Disease_symptom_and_patient_profile_dataset.csv
# ═══════════════════════════════════════════════════════════════════════════

def process_patient_profile_csv(path):
    """Convert patient profile dataset — has age/gender but fewer symptoms."""
    df = pd.read_csv(path)
    converted = []

    for _, row in df.iterrows():
        disease = row["Disease"].strip()
        if disease not in PP_DISEASE_MAP:
            continue

        our_disease = PP_DISEASE_MAP[disease]
        our_row = {s: 0 for s in OUR_SYMPTOMS}

        # Map the few symptom columns
        if row.get("Fever", "No") == "Yes":
            our_row["fever"] = 1
        if row.get("Cough", "No") == "Yes":
            our_row["cough"] = 1
        if row.get("Fatigue", "No") == "Yes":
            our_row["fatigue"] = 1
        if row.get("Difficulty Breathing", "No") == "Yes":
            our_row["difficulty_breathing"] = 1

        # Add age/gender from dataset
        our_row["age"] = int(row.get("Age", 30))
        our_row["gender"] = 1 if row.get("Gender", "Male") == "Male" else 0
        our_row["_disease"] = our_disease

        converted.append(our_row)

    print(f"   Patient Profile: {len(converted)} rows mapped")
    return pd.DataFrame(converted)


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Kaggle Dataset Integration")
    print("=" * 60)

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    all_frames = []

    # 1. Training.csv
    training_path = os.path.join(data_dir, "Training.csv")
    if os.path.exists(training_path):
        df1 = process_training_csv(training_path)
        all_frames.append(df1)

    # 2. Testing.csv (same format as Training.csv)
    testing_path = os.path.join(data_dir, "Testing.csv")
    if os.path.exists(testing_path):
        df2 = process_training_csv(testing_path)
        all_frames.append(df2)

    # 3. Patient Profile
    pp_path = os.path.join(data_dir, "Disease_symptom_and_patient_profile_dataset.csv")
    if os.path.exists(pp_path):
        df3 = process_patient_profile_csv(pp_path)
        all_frames.append(df3)

    if not all_frames:
        print("No datasets found!")
        return

    # Combine all
    combined = pd.concat(all_frames, ignore_index=True)
    print(f"\n   Combined: {len(combined)} total rows")
    print(f"   Disease distribution:")
    for d in sorted(combined["_disease"].unique()):
        count = len(combined[combined["_disease"] == d])
        print(f"     {d:30s} {count:5d} rows")

    # Save
    out_path = os.path.join(data_dir, "kaggle_mapped.csv")
    combined.to_csv(out_path, index=False)
    print(f"\n   Saved: {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")
    print("   Ready for train_model.py!")


if __name__ == "__main__":
    main()
