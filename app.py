"""
Rural Health Triage Assistant — Gradio MVP
Offline AI symptom checker (English + Hindi)
Run:  python app.py
"""

import json, textwrap
import numpy as np
import gradio as gr
from xgboost import XGBClassifier

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  TRIAGE ENGINE                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Hindi + English keyword → symptom mapping ─────────────────────────────
KEYWORD_MAP = {
    "fever":                ["fever","bukhar","बुखार","taap","ताप","temperature"],
    "high_fever":           ["high fever","tez bukhar","तेज़ बुखार","39","40","104","103","102"],
    "chills":               ["chill","thandi","ठंड","kaanpna","काँपना","shiver","kamp"],
    "headache":             ["headache","sir dard","सिर दर्द","sardard","सरदर्द","head pain"],
    "severe_headache":      ["severe headache","bahut sir dard","बहुत सिर दर्द","tez sardard"],
    "body_ache":            ["body ache","badan dard","बदन दर्द","body pain","sharir dard","शरीर दर्द"],
    "joint_pain":           ["joint pain","jod dard","जोड़ दर्द","gathiya","गठिया","joint"],
    "muscle_pain":          ["muscle pain","maaspeshi dard","मांसपेशी दर्द","muscle"],
    "cough":                ["cough","khansi","खांसी","khaansi"],
    "chronic_cough":        ["chronic cough","purani khansi","पुरानी खांसी","lambi khansi","weeks cough","mahino se khansi"],
    "sore_throat":          ["sore throat","gala dard","गला दर्द","gala kharab","गला खराब","throat"],
    "runny_nose":           ["runny nose","naak behna","नाक बहना","naak beh raha","nose running"],
    "sneezing":             ["sneez","chheenk","छींक"],
    "difficulty_breathing": ["difficulty breathing","saans lene mein taklif","सांस लेने में तकलीफ","breathless","saans nahi","cant breathe"],
    "rapid_breathing":      ["rapid breath","tez saans","तेज़ सांस","fast breath"],
    "chest_pain":           ["chest pain","seena dard","सीना दर्द","chhati dard","छाती दर्द","chest"],
    "abdominal_pain":       ["stomach pain","pet dard","पेट दर्द","abdominal","pet mein dard","tummy"],
    "nausea":               ["nausea","ji machlana","जी मचलाना","ulti jaisa","मतली","queasy"],
    "vomiting":             ["vomit","ulti","उल्टी"],
    "diarrhea":             ["diarrhea","dast","दस्त","loose motion","पतला मल","loose stool","watery stool"],
    "rash":                 ["rash","daane","दाने","skin rash","चकत्ते","lal daane","spots on skin"],
    "eye_pain":             ["eye pain","aankh dard","आँख दर्द","aankh mein dard","eyes hurt"],
    "dark_urine":           ["dark urine","peela peshab","गहरा पेशाब","brown urine"],
    "blood_in_sputum":      ["blood cough","khoon khansi","खून खांसी","blood sputum","balgam mein khoon"],
    "night_sweats":         ["night sweat","raat ko paseena","रात को पसीना"],
    "weight_loss":          ["weight loss","vajan ghatna","वज़न घटना","patla hona","getting thin"],
    "fatigue":              ["tired","thakan","थकान","fatigue","kamzori","कमज़ोरी","exhausted"],
    "weakness":             ["weak","kamzor","कमज़ोर","no energy","shakti nahi"],
    "dizziness":            ["dizzy","chakkar","चक्कर","sir ghoomna","सिर घूमना","lightheaded"],
    "sweating":             ["sweat","paseena","पसीना","bahut paseena","profuse sweat"],
    "confusion":            ["confus","behoshi","samajh nahi","disoriented","confused"],
    "bleeding":             ["bleed","khoon","खून","khoon behna","blood coming"],
    "pale_skin":            ["pale","peela","पीला","safed","rang ud gaya","pallor"],
    "dehydration_signs":    ["dehydrat","paani ki kami","पानी की कमी","pyaas","प्यास","thirsty","dry mouth","sookha muh"],
}

# ── Red-flag rules (always → RED regardless of model) ─────────────────────
RED_FLAG_RULES = [
    {
        "kw": ["unconscious","behosh","बेहोश","loss of consciousness","hosh nahi","not responding","unresponsive"],
        "en": "Loss of consciousness detected",
        "hi": "बेहोशी — आपातकालीन स्थिति",
    },
    {
        "kw": ["chest pain+sweat","seena dard+paseena","सीना दर्द+पसीना"],
        "combo": (["chest pain","seena dard","सीना दर्द","chhati dard","छाती दर्द"],
                  ["sweat","paseena","पसीना"]),
        "en": "Chest pain with sweating — possible heart attack",
        "hi": "सीने में दर्द और पसीना — संभावित हार्ट अटैक",
    },
    {
        "kw": ["seizure","daura","दौरा","fits","mirgi","मिर्गी","convulsion"],
        "en": "Seizure / fits detected",
        "hi": "दौरा / मिर्गी — आपातकालीन",
    },
    {
        "kw": ["severe bleeding","bahut khoon","बहुत खून","heavy bleeding","profuse bleed"],
        "en": "Severe bleeding",
        "hi": "गंभीर रक्तस्राव — आपातकालीन",
    },
    {
        "kw": ["cant breathe","saans nahi","सांस नहीं","not breathing","saans band"],
        "en": "Cannot breathe — respiratory emergency",
        "hi": "सांस नहीं आ रही — श्वसन आपातकालीन",
    },
]

# ── Disease info (advice + display names) ─────────────────────────────────
DISEASE_INFO = {
    "malaria": {
        "en": "Malaria",  "hi": "मलेरिया",
        "adv_en": "High risk of malaria. Go to hospital immediately for a blood test. Use mosquito net.",
        "adv_hi": "मलेरिया का उच्च जोखिम। तुरंत अस्पताल जाएं और खून की जांच कराएं। मच्छरदानी का प्रयोग करें।",
    },
    "dengue": {
        "en": "Dengue",  "hi": "डेंगू",
        "adv_en": "Possible dengue fever. See doctor immediately. Drink plenty of fluids. Watch for bleeding or bruising.",
        "adv_hi": "संभावित डेंगू बुखार। तुरंत डॉक्टर को दिखाएं। खूब पानी पिएं। रक्तस्राव पर ध्यान दें।",
    },
    "typhoid": {
        "en": "Typhoid",  "hi": "टाइफाइड",
        "adv_en": "Possible typhoid. See a doctor soon for Widal/blood test. Drink only boiled water.",
        "adv_hi": "संभावित टाइफाइड। जल्द डॉक्टर को दिखाएं। केवल उबला पानी पिएं।",
    },
    "tuberculosis": {
        "en": "Tuberculosis (TB)",  "hi": "क्षय रोग (टीबी)",
        "adv_en": "Symptoms suggest TB. Visit clinic for sputum test. TB is curable with 6-month medicine.",
        "adv_hi": "लक्षण टीबी के हो सकते हैं। बलगम जांच के लिए क्लिनिक जाएं। 6 महीने की दवा से ठीक हो सकता है।",
    },
    "pneumonia": {
        "en": "Pneumonia",  "hi": "निमोनिया",
        "adv_en": "Possible pneumonia — breathing difficulty is serious. Seek medical care urgently.",
        "adv_hi": "संभावित निमोनिया — सांस की तकलीफ गंभीर है। तुरंत चिकित्सा सहायता लें।",
    },
    "common_cold": {
        "en": "Common Cold",  "hi": "सामान्य सर्दी",
        "adv_en": "Likely a common cold. Rest, drink warm fluids, take paracetamol if feverish. See doctor if no improvement in 3 days.",
        "adv_hi": "सामान्य सर्दी लगती है। आराम करें, गर्म पानी पिएं, बुखार हो तो पैरासिटामॉल लें। 3 दिन में ठीक न हो तो डॉक्टर को दिखाएं।",
    },
    "gastroenteritis": {
        "en": "Stomach Infection",  "hi": "पेट का संक्रमण",
        "adv_en": "Stomach infection likely. Drink ORS and boiled water. See doctor if vomiting persists or blood in stool.",
        "adv_hi": "पेट का संक्रमण हो सकता है। ORS और उबला पानी पिएं। उल्टी बंद न हो या मल में खून हो तो डॉक्टर को दिखाएं।",
    },
    "anemia": {
        "en": "Anemia",  "hi": "खून की कमी",
        "adv_en": "Signs of anemia (low blood). Eat iron-rich food (spinach, jaggery, eggs). Visit clinic for blood test.",
        "adv_hi": "खून की कमी के लक्षण। लोहे से भरपूर भोजन खाएं (पालक, गुड़, अंडे)। खून की जांच कराएं।",
    },
    "heat_stroke": {
        "en": "Heat Stroke",  "hi": "लू लगना",
        "adv_en": "Possible heat stroke. Move to shade, cool the body with wet cloth, give water. Go to hospital if confused.",
        "adv_hi": "लू लगने के लक्षण। छाया में जाएं, गीले कपड़े से शरीर ठंडा करें, पानी दें। बेहोशी हो तो अस्पताल जाएं।",
    },
    "flu": {
        "en": "Influenza (Flu)",  "hi": "फ्लू",
        "adv_en": "Likely flu. Rest, drink fluids, take paracetamol for fever. See doctor if symptoms worsen after 3 days.",
        "adv_hi": "फ्लू हो सकता है। आराम करें, तरल पदार्थ पिएं। 3 दिन बाद भी ठीक न हो तो डॉक्टर को दिखाएं।",
    },
}

# ── Static offline clinic database ────────────────────────────────────────
CLINICS = [
    {"name": "PHC Rampur",      "name_hi": "प्रा. स्वा. केंद्र रामपुर",    "km": 3,  "phone": "1800-180-1234", "type": "PHC"},
    {"name": "CHC Dhanpur",     "name_hi": "सा. स्वा. केंद्र धनपुर",      "km": 8,  "phone": "1800-180-5678", "type": "CHC"},
    {"name": "District Hospital","name_hi": "जिला अस्पताल",              "km": 15, "phone": "108 (Ambulance)","type": "Hospital"},
    {"name": "Sub-Centre Kheri","name_hi": "उप-केंद्र खेरी",              "km": 1.5,"phone": "9876543210",    "type": "Sub-Centre"},
]

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  ENGINE CLASS                                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class TriageEngine:
    def __init__(self, model_path="models/triage_model.json",
                 meta_path="models/metadata.json"):
        self.model = XGBClassifier()
        self.model.load_model(model_path)
        with open(meta_path) as f:
            meta = json.load(f)
        self.features    = meta["features"]
        self.symptoms    = meta["symptoms"]
        self.diseases    = meta["diseases"]
        self.urgency_map = meta["urgency"]

    # ── extract binary symptom vector from free text ──────────────────────
    def _extract(self, text: str):
        low = text.lower()
        hit = {}
        found = []
        for symptom, keywords in KEYWORD_MAP.items():
            match = any(kw in low for kw in keywords)
            hit[symptom] = int(match)
            if match:
                found.append(symptom)
        return hit, found

    # ── check red-flag rules ──────────────────────────────────────────────
    def _red_flags(self, text: str):
        low = text.lower()
        triggered = []
        for rule in RED_FLAG_RULES:
            if "combo" in rule:
                a_hit = any(k in low for k in rule["combo"][0])
                b_hit = any(k in low for k in rule["combo"][1])
                if a_hit and b_hit:
                    triggered.append(rule)
            else:
                if any(k in low for k in rule["kw"]):
                    triggered.append(rule)
        return triggered

    # ── main prediction ───────────────────────────────────────────────────
    def predict(self, text: str, age: int = 30, gender: int = 0,
                lang: str = "en"):
        flags = self._red_flags(text)
        sym_dict, detected = self._extract(text)
        sym_dict["age"]    = age
        sym_dict["gender"] = gender
        vec = np.array([[sym_dict.get(f, 0) for f in self.features]])

        # — red-flag override —
        if flags:
            msg = flags[0][lang]
            return dict(
                urgency="RED", detected=detected, confidence=1.0,
                disease="Emergency" if lang == "en" else "आपातकालीन",
                explanation=msg,
                advice="Call 108 ambulance or go to nearest hospital NOW."
                       if lang == "en"
                       else "108 एम्बुलेंस बुलाएं या तुरंत नज़दीकी अस्पताल जाएं।",
                top3=[], is_flag=True,
            )

        # — no symptoms detected —
        if not detected:
            return dict(
                urgency="NONE", detected=[], confidence=0,
                disease=None,
                explanation="No symptoms detected. Please describe what you feel."
                            if lang == "en"
                            else "कोई लक्षण नहीं मिला। कृपया बताएं क्या तकलीफ है।",
                advice="", top3=[], is_flag=False,
            )

        # — model inference —
        probs    = self.model.predict_proba(vec)[0]
        top3_idx = np.argsort(probs)[::-1][:3]
        best     = self.diseases[top3_idx[0]]
        urgency  = self.urgency_map[best]
        info     = DISEASE_INFO[best]

        top3 = []
        for i in top3_idx:
            d = self.diseases[i]
            di = DISEASE_INFO[d]
            top3.append({
                "disease": di[lang],
                "prob": float(probs[i]),
                "urgency": self.urgency_map[d],
            })

        detected_display = ", ".join(s.replace("_", " ") for s in detected[:6])
        expl = (f"Based on: {detected_display}"
                if lang == "en"
                else f"लक्षणों के आधार पर: {detected_display}")

        return dict(
            urgency=urgency, detected=detected,
            confidence=float(probs[top3_idx[0]]),
            disease=info[lang], explanation=expl,
            advice=info[f"adv_{lang}"], top3=top3, is_flag=False,
        )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  FORMAT OUTPUT                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

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
    lines.append(f"## {icon}  {label}")
    lines.append("")

    if r["disease"]:
        heading = "Likely condition" if lang == "en" else "संभावित बीमारी"
        lines.append(f"**{heading}:** {r['disease']}  "
                      f"(confidence {r['confidence']:.0%})")
    lines.append("")
    lines.append(f"*{r['explanation']}*")
    lines.append("")

    # advice box
    adv_head = "💊 Advice" if lang == "en" else "💊 सलाह"
    lines.append(f"### {adv_head}")
    lines.append(r["advice"])
    lines.append("")

    # top-3 differential
    if r["top3"]:
        diff_head = "Top possibilities" if lang == "en" else "संभावित बीमारियां"
        lines.append(f"### 📋 {diff_head}")
        for i, t in enumerate(r["top3"], 1):
            u_icon = URGENCY_STYLE.get(t["urgency"], URGENCY_STYLE["NONE"])[0]
            lines.append(f"{i}. {u_icon} **{t['disease']}** — {t['prob']:.0%}")
        lines.append("")

    # nearest clinics
    clinic_head = "🏥 Nearest Clinics" if lang == "en" else "🏥 नज़दीकी स्वास्थ्य केंद्र"
    lines.append(f"### {clinic_head}")
    for c in sorted(CLINICS, key=lambda x: x["km"])[:3]:
        n = c["name"] if lang == "en" else c["name_hi"]
        lines.append(f"- **{n}** — {c['km']} km — ☎ {c['phone']}  ({c['type']})")
    lines.append("")

    # disclaimer
    disc = ("⚠️ *This is a guide, not a diagnosis. "
            "Always consult a health worker if symptoms worsen.*"
            if lang == "en"
            else "⚠️ *यह केवल मार्गदर्शन है, निदान नहीं। "
                 "लक्षण बढ़ें तो स्वास्थ्य कर्मी से ज़रूर मिलें।*")
    lines.append(disc)

    return "\n".join(lines)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GRADIO UI                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

engine = TriageEngine()

def run_triage(symptoms_text, age, gender_choice, language):
    lang   = "hi" if language == "हिन्दी" else "en"
    gender = 1 if gender_choice in ("Male", "पुरुष") else 0
    age    = int(age) if age else 30

    result = engine.predict(symptoms_text, age=age, gender=gender, lang=lang)
    return format_result(result, lang)


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
.gradio-container { max-width: 700px !important; margin: auto; }
.output-markdown h2 { padding: 12px; border-radius: 8px; }
"""

with gr.Blocks(css=CSS, title="🏥 Rural Health Triage") as demo:

    gr.Markdown(
        "# 🏥 Rural Health Triage Assistant\n"
        "### AI-powered offline symptom checker · English + Hindi\n"
        "*Tell us your symptoms — get instant guidance.*\n"
        "---"
    )

    with gr.Row():
        lang_dd = gr.Dropdown(
            ["English", "हिन्दी"], value="English",
            label="Language / भाषा", scale=1,
        )

    with gr.Row():
        age_box = gr.Number(value=30, label="Age / उम्र", minimum=0,
                            maximum=120, precision=0, scale=1)
        gender_dd = gr.Dropdown(
            ["Male", "Female"], value="Male",
            label="Gender / लिंग", scale=1,
        )

    symptom_box = gr.Textbox(
        lines=3,
        placeholder="Describe symptoms… e.g. 'fever, headache, chills'\n"
                    "या हिंदी में बताएं… 'बुखार, सिर दर्द, उल्टी'",
        label="Symptoms / लक्षण 🩺",
    )

    submit_btn = gr.Button("🔍  Check / जांचें", variant="primary", size="lg")

    output_md = gr.Markdown(label="Result / परिणाम")

    submit_btn.click(
        fn=run_triage,
        inputs=[symptom_box, age_box, gender_dd, lang_dd],
        outputs=output_md,
    )

    gr.Examples(
        examples=EXAMPLE_CASES,
        inputs=[symptom_box, age_box, gender_dd, lang_dd],
        outputs=output_md,
        fn=run_triage,
        cache_examples=False,
        label="📝 Try these test cases",
    )

    gr.Markdown(
        "---\n"
        "*Model: XGBoost (<1 MB) · 100% offline · No data stored · "
        "Not a medical device — always consult a health professional.*"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)