"""Quick script to explore all datasets and find mappable columns/diseases."""
import pandas as pd
import os

DATA = "data"

# 1. Training.csv (Kaggle Disease Prediction)
print("=" * 60)
print("1. Training.csv (Kaggle binary symptoms)")
print("=" * 60)
df = pd.read_csv(os.path.join(DATA, "Training.csv"))
cols = [c for c in df.columns if c not in ["prognosis", "Unnamed: 133"]]
print(f"   Shape: {df.shape}")
print(f"   Symptom columns ({len(cols)}):")
for i in range(0, len(cols), 8):
    print(f"     {cols[i:i+8]}")
print(f"\n   Diseases ({df['prognosis'].nunique()}):")
for d in sorted(df["prognosis"].unique()):
    print(f"     - {d} ({len(df[df['prognosis']==d])} rows)")

# 2. Symptom2Disease.csv
print("\n" + "=" * 60)
print("2. Symptom2Disease.csv (NLP text)")
print("=" * 60)
df2 = pd.read_csv(os.path.join(DATA, "Symptom2Disease.csv"))
print(f"   Shape: {df2.shape}")
print(f"   Columns: {list(df2.columns)}")
print(f"   Diseases: {sorted(df2['label'].unique()) if 'label' in df2.columns else 'N/A'}")
print(f"   Sample row:\n     {df2.iloc[0].to_dict()}")

# 3. Disease_symptom_and_patient_profile_dataset.csv
print("\n" + "=" * 60)
print("3. Disease_symptom_and_patient_profile.csv")
print("=" * 60)
df3 = pd.read_csv(os.path.join(DATA, "Disease_symptom_and_patient_profile_dataset.csv"))
print(f"   Shape: {df3.shape}")
print(f"   Columns: {list(df3.columns)}")
print(f"   Diseases: {sorted(df3['Disease'].unique()) if 'Disease' in df3.columns else 'N/A'}")
print(f"   Sample:\n     {df3.iloc[0].to_dict()}")

# 4. sym_dis_matrix.csv
print("\n" + "=" * 60)
print("4. sym_dis_matrix.csv")
print("=" * 60)
df4 = pd.read_csv(os.path.join(DATA, "sym_dis_matrix.csv"))
print(f"   Shape: {df4.shape}")
print(f"   Columns (first 10): {list(df4.columns[:10])}")
print(f"   Sample:\n     {df4.iloc[0].to_dict()}")
