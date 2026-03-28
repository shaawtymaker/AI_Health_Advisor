"""Script to extract dictionaries from app.py to new structure."""
import json, os
from app import UI_TEXT, KEYWORD_MAP, RED_FLAG_RULES, DISEASE_INFO

# Write i18n
with open("i18n/en.json", "w", encoding="utf-8") as f:
    json.dump(UI_TEXT["en"], f, ensure_ascii=False, indent=2)

with open("i18n/hi.json", "w", encoding="utf-8") as f:
    json.dump(UI_TEXT["hi"], f, ensure_ascii=False, indent=2)

# Write Data
with open("data/diseases.json", "w", encoding="utf-8") as f:
    json.dump(DISEASE_INFO, f, ensure_ascii=False, indent=2)

# Write Keywords to engine/keywords.py
with open("engine/keywords.py", "w", encoding="utf-8") as f:
    f.write('KEYWORD_MAP = {\n')
    for k, v in KEYWORD_MAP.items():
        f.write(f'    "{k}": {json.dumps(v, ensure_ascii=False)},\n')
    f.write('}\n')

# Write Red Flags to engine/red_flags.py
# Red flags have a tuple for 'combo', json doesn't do tuples well without parsing back to lists, so keep it as python.
with open("engine/red_flags.py", "w", encoding="utf-8") as f:
    f.write('RED_FLAG_RULES = ' + repr(RED_FLAG_RULES) + '\n')
    
print("Extraction complete.")
