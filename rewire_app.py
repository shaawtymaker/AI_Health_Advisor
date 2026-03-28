import os
app_path = "app.py"

with open(app_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
skip = False

# We want to keep imports, then insert our new imports, then skip straight to line 460.
for i, line in enumerate(lines):
    # lines are 0-indexed
    if i == 12: # line 13: # ╔═════════════════════
        skip = True
        new_lines.append("\n# --- NEW V4 IMPORTS ---\n")
        new_lines.append("import json, os\n")
        new_lines.append("from engine.triage import TriageEngine\n")
        new_lines.append("from engine.formatter import format_result\n")
        new_lines.append("\n")
        new_lines.append("with open(os.path.join(os.path.dirname(__file__), 'i18n', 'en.json'), encoding='utf-8') as f:\n")
        new_lines.append("    UI_EN = json.load(f)\n")
        new_lines.append("with open(os.path.join(os.path.dirname(__file__), 'i18n', 'hi.json'), encoding='utf-8') as f:\n")
        new_lines.append("    UI_HI = json.load(f)\n")
        new_lines.append("UI_TEXT = {'en': UI_EN, 'hi': UI_HI}\n")
    
    if skip and i == 460: # line 461
        skip = False
        
    if not skip:
        # fix voice import
        if "from voice import" in line:
            line = line.replace("from voice import", "from voice.stt import")
            
        new_lines.append(line)

with open(app_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Re-wired app.py successfully!")
