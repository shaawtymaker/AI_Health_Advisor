import os, json

_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_followups_file = os.path.join(_data_dir, "followups.json")

try:
    with open(_followups_file, "r", encoding="utf-8") as f:
        FOLLOWUPS = json.load(f)
except FileNotFoundError:
    FOLLOWUPS = {}

def get_followup_question(detected_symptoms: list[str], lang: str = "en") -> str:
    """
    Analyzes detected symptoms. If a primary symptom is present but 
    crucial clarifying symptoms are missing, return a follow-up string.
    Returns empty string if no follow-up is necessary.
    """
    # Look for the first symptom in the dictionary that matches what user inputted
    for sym in detected_symptoms:
        if sym in FOLLOWUPS:
            rule = FOLLOWUPS[sym]
            
            # Check if any of the 'trigger_if_missing' symptoms are ALREADY present
            # If none are present, it means the user was vague, so we ask the follow up.
            missing_deps = True
            for dep in rule.get("trigger_if_missing", []):
                if dep in detected_symptoms:
                    missing_deps = False
                    break
            
            if missing_deps:
                return rule.get(lang, "")
                
    return ""
