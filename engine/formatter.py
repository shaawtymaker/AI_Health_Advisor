import os, json

_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_clinics_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "clinics.json")
try:
    with open(_clinics_file, "r", encoding="utf-8") as f:
        CLINICS = json.load(f)
except FileNotFoundError:
    CLINICS = []

URGENCY_STYLE = {
    "RED":    ("🔴", "#FF4444", "URGENT — Go to Hospital NOW",    "गंभीर — तुरंत अस्पताल जाएं"),
    "YELLOW": ("🟡", "#FFAA00", "See a Doctor Soon",              "जल्द डॉक्टर को दिखाएं"),
    "GREEN":  ("🟢", "#44BB44", "Home Care — Monitor",            "घर पर देखभाल — निगरानी रखें"),
    "NONE":   ("⚪", "#888888", "Unknown",                        "अज्ञात"),
}

def format_result(r: dict, lang: str) -> str:
    icon, color, label_en, label_hi = URGENCY_STYLE.get(
        r["urgency"], URGENCY_STYLE["NONE"]
    )
    label = label_en if lang == "en" else label_hi
    lines = []

    lines.append(f'<div style="background:{color}; color:white; padding:16px 20px; border-radius:12px; margin-bottom:16px; font-size:1.3em; font-weight:bold; text-align:center;">')
    lines.append(f'{icon}  {label}</div>\n')

    if r["disease"]:
        heading = "Likely condition" if lang == "en" else "संभावित बीमारी"
        lines.append(f"**{heading}:** {r['disease']}  (confidence {r['confidence']:.0%})\n")
    
    lines.append(f"*{r['explanation']}*\n")

    if r.get("shap_text"):
        lines.append(r["shap_text"] + "\n")

    adv_head = "💊 Advice" if lang == "en" else "💊 सलाह"
    lines.append(f"### {adv_head}\n{r['advice']}\n")

    if r["urgency"] == "RED":
        lines.append('<div style="background:#CC0000; color:white; padding:14px; border-radius:10px; text-align:center; font-size:1.2em; font-weight:bold; margin:12px 0;">')
        btn_text = '📞 CALL 108 AMBULANCE — EMERGENCY' if lang == "en" else '📞 108 एम्बुलेंस बुलाएं — आपातकालीन'
        lines.append(f'{btn_text}</div>\n')

    if r.get("top3"):
        diff_head = "Top possibilities" if lang == "en" else "संभावित बीमारियां"
        lines.append(f"### 📋 {diff_head}")
        for i, t in enumerate(r["top3"], 1):
            u_icon = URGENCY_STYLE.get(t["urgency"], URGENCY_STYLE["NONE"])[0]
            lines.append(f"{i}. {u_icon} **{t['disease']}** — {t['prob']:.0%}")
        lines.append("")

    clinic_head = "🏥 Nearest Clinics" if lang == "en" else "🏥 नज़दीकी स्वास्थ्य केंद्र"
    lines.append(f"### {clinic_head}")
    for c in sorted(CLINICS, key=lambda x: x.get("km", 99))[:3]:
        n = c["name"] if lang == "en" else c.get("name_hi", c["name"])
        lines.append(f"- **{n}** — {c.get('km','?')} km — ☎ {c.get('phone','')}  ({c.get('type','')})")
    
    return "\n".join(lines)
