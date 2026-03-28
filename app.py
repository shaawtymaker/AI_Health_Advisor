"""
Rural Health Triage Assistant — Gradio MVP
Offline AI symptom checker (English + Hindi)
Features: Text/Voice input, SHAP explainability, TTS output, bilingual UI
Run:  python app.py
"""

import json, os, textwrap, tempfile, traceback
import numpy as np
import gradio as gr
from xgboost import XGBClassifier

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  UI TEXT — BILINGUAL LABELS                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

UI_TEXT = {
    "en": {
        "title":          "# 🏥 Rural Health Triage Assistant",
        "subtitle":       "### AI-powered offline symptom checker · English + Hindi\n*Tell us your symptoms — get instant guidance.*\n---",
        "lang_label":     "Language / भाषा",
        "age_label":      "Age",
        "gender_label":   "Gender",
        "gender_choices": ["Male", "Female"],
        "symptom_label":  "Symptoms 🩺",
        "symptom_placeholder": "Describe symptoms… e.g. 'fever, headache, chills'",
        "voice_label":    "🎤 Voice Input (optional — record your symptoms)",
        "submit":         "🔍  Check",
        "result_label":   "Result",
        "audio_label":    "🔊 Listen to Advice",
        "examples_label": "📝 Try these test cases",
        "footer":         "---\n*Model: XGBoost (<1 MB) · 100% offline · No data stored · Not a medical device — always consult a health professional.*",
    },
    "hi": {
        "title":          "# 🏥 ग्रामीण स्वास्थ्य ट्राइएज सहायक",
        "subtitle":       "### AI-संचालित ऑफ़लाइन लक्षण जांचकर्ता · हिन्दी + English\n*अपने लक्षण बताएं — तुरंत मार्गदर्शन पाएं।*\n---",
        "lang_label":     "भाषा / Language",
        "age_label":      "उम्र",
        "gender_label":   "लिंग",
        "gender_choices": ["पुरुष", "महिला"],
        "symptom_label":  "लक्षण 🩺",
        "symptom_placeholder": "अपने लक्षण बताएं… जैसे 'बुखार, सिर दर्द, उल्टी'",
        "voice_label":    "🎤 आवाज़ से बताएं (वैकल्पिक — अपने लक्षण बोलें)",
        "submit":         "🔍  जांचें",
        "result_label":   "परिणाम",
        "audio_label":    "🔊 सलाह सुनें",
        "examples_label": "📝 ये उदाहरण आज़माएं",
        "footer":         "---\n*मॉडल: XGBoost (<1 MB) · 100% ऑफ़लाइन · कोई डेटा नहीं रखा जाता · यह चिकित्सा उपकरण नहीं है — हमेशा स्वास्थ्य पेशेवर से परामर्श करें।*",
    },
}

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
    "cholera": {
        "en": "Cholera",  "hi": "हैजा",
        "adv_en": "Possible cholera — severe dehydration risk. Drink ORS immediately. Go to hospital NOW for IV fluids.",
        "adv_hi": "संभावित हैजा — गंभीर निर्जलीकरण का खतरा। तुरंत ORS पिएं। IV तरल पदार्थ के लिए अभी अस्पताल जाएं।",
    },
    "jaundice": {
        "en": "Jaundice / Hepatitis",  "hi": "पीलिया",
        "adv_en": "Signs of jaundice/hepatitis. Avoid oily food, drink boiled water. See doctor for liver function test.",
        "adv_hi": "पीलिया के लक्षण। तला हुआ खाना बंद करें, उबला पानी पिएं। लिवर जांच के लिए डॉक्टर को दिखाएं।",
    },
    "urinary_tract_infection": {
        "en": "Urinary Tract Infection (UTI)",  "hi": "मूत्र मार्ग संक्रमण",
        "adv_en": "Possible UTI. Drink plenty of water. See doctor for urine test — antibiotics may be needed.",
        "adv_hi": "संभावित मूत्र संक्रमण। खूब पानी पिएं। पेशाब जांच के लिए डॉक्टर को दिखाएं।",
    },
    "asthma_attack": {
        "en": "Asthma Attack",  "hi": "दमा का दौरा",
        "adv_en": "Signs of asthma attack. Sit upright, use inhaler if available. Seek medical help if breathing doesn't improve.",
        "adv_hi": "दमा के दौरे के लक्षण। सीधे बैठें, इन्हेलर उपलब्ध हो तो उपयोग करें। सांस न सुधरे तो चिकित्सा सहायता लें।",
    },
    "chickenpox": {
        "en": "Chickenpox",  "hi": "चिकनपॉक्स (छोटी माता)",
        "adv_en": "Likely chickenpox. Keep rash clean, avoid scratching. Take paracetamol for fever. Stay isolated to prevent spread.",
        "adv_hi": "चिकनपॉक्स लगती है। दानों को साफ रखें, खुजाएं नहीं। बुखार हो तो पैरासिटामॉल लें। फैलने से रोकने के लिए अलग रहें।",
    },
}

# ── Load clinic database from external JSON ───────────────────────────────
CLINICS_PATH = os.path.join(os.path.dirname(__file__), "clinics.json")
with open(CLINICS_PATH, encoding="utf-8") as f:
    CLINICS = json.load(f)


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

        # SHAP explainer (lazy init)
        self._explainer = None

    def _get_explainer(self):
        if self._explainer is None:
            try:
                import shap
                self._explainer = shap.TreeExplainer(self.model)
            except ImportError:
                self._explainer = False  # Mark as unavailable
        return self._explainer

    # ── extract binary symptom vector from free text ──────────────────────
    @staticmethod
    def _keyword_match(keyword: str, text: str) -> bool:
        """Flexible keyword matching: for multi-word keywords, check if ALL
        words appear anywhere in the text (handles natural speech where
        extra words appear between symptom words, e.g. 'सिर में बहुत दर्द')."""
        if " " in keyword:
            return all(word in text for word in keyword.split())
        return keyword in text

    def _extract(self, text: str):
        low = text.lower()
        hit = {}
        found = []
        for symptom, keywords in KEYWORD_MAP.items():
            match = any(self._keyword_match(kw, low) for kw in keywords)
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

    # ── SHAP explanation ───────────────────────────────────────────────────
    def _explain(self, vec, predicted_class, lang="en"):
        explainer = self._get_explainer()
        if not explainer:
            return ""
        try:
            shap_values = explainer.shap_values(vec)
            # For multi-class, shap_values is a list of arrays (one per class)
            if isinstance(shap_values, list):
                vals = shap_values[predicted_class][0]
            else:
                vals = shap_values[0]

            # Get top 3 contributing features
            top_indices = np.argsort(np.abs(vals))[::-1][:3]
            top_features = []
            for i in top_indices:
                fname = self.features[i]
                if vals[i] != 0:
                    top_features.append(fname.replace("_", " "))

            if not top_features:
                return ""

            if lang == "en":
                return f"**🧠 Most influential factors:** {', '.join(top_features)}"
            else:
                return f"**🧠 सबसे प्रभावशाली कारक:** {', '.join(top_features)}"
        except Exception:
            return ""

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
                top3=[], is_flag=True, shap_text="",
            )

        # — no symptoms detected —
        if not detected:
            return dict(
                urgency="NONE", detected=[], confidence=0,
                disease=None,
                explanation="No symptoms detected. Please describe what you feel."
                            if lang == "en"
                            else "कोई लक्षण नहीं मिला। कृपया बताएं क्या तकलीफ है।",
                advice="", top3=[], is_flag=False, shap_text="",
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

        # SHAP explanation
        shap_text = self._explain(vec, top3_idx[0], lang)

        return dict(
            urgency=urgency, detected=detected,
            confidence=float(probs[top3_idx[0]]),
            disease=info[lang], explanation=expl,
            advice=info[f"adv_{lang}"], top3=top3, is_flag=False,
            shap_text=shap_text,
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

    # Urgency banner
    lines.append(f'<div style="background:{color}; color:white; padding:16px 20px; '
                 f'border-radius:12px; margin-bottom:16px; font-size:1.3em; '
                 f'font-weight:bold; text-align:center;">')
    lines.append(f'{icon}  {label}')
    lines.append('</div>')
    lines.append("")

    if r["disease"]:
        heading = "Likely condition" if lang == "en" else "संभावित बीमारी"
        lines.append(f"**{heading}:** {r['disease']}  "
                      f"(confidence {r['confidence']:.0%})")
    lines.append("")
    lines.append(f"*{r['explanation']}*")
    lines.append("")

    # SHAP explainability
    if r.get("shap_text"):
        lines.append(r["shap_text"])
        lines.append("")

    # advice box
    adv_head = "💊 Advice" if lang == "en" else "💊 सलाह"
    lines.append(f"### {adv_head}")
    lines.append(r["advice"])
    lines.append("")

    # 🚨 Ambulance button for RED urgency
    if r["urgency"] == "RED":
        if lang == "en":
            lines.append('<div style="background:#CC0000; color:white; padding:14px; '
                         'border-radius:10px; text-align:center; font-size:1.2em; '
                         'font-weight:bold; margin:12px 0;">')
            lines.append('📞 CALL 108 AMBULANCE — EMERGENCY')
            lines.append('</div>')
        else:
            lines.append('<div style="background:#CC0000; color:white; padding:14px; '
                         'border-radius:10px; text-align:center; font-size:1.2em; '
                         'font-weight:bold; margin:12px 0;">')
            lines.append('📞 108 एम्बुलेंस बुलाएं — आपातकालीन')
            lines.append('</div>')
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
# ║  TTS ENGINE  (plays directly via speakers — no file returned)          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def speak_advice(text: str, lang: str = "en"):
    """Speak the advice text aloud using pyttsx3 (runs in background thread)."""
    import threading
    def _speak():
        try:
            import pyttsx3
            tts = pyttsx3.init()
            voices = tts.getProperty("voices")
            for v in voices:
                name_lower = v.name.lower()
                if lang == "hi" and "hindi" in name_lower:
                    tts.setProperty("voice", v.id)
                    break
                elif lang == "en" and "english" in name_lower:
                    tts.setProperty("voice", v.id)
                    break
            tts.setProperty("rate", 150)
            clean = text.replace("**", "").replace("*", "").replace("#", "").replace("---", "")
            if len(clean) > 400:
                clean = clean[:400]
            tts.say(clean)
            tts.runAndWait()
        except Exception as e:
            print(f"[TTS] Error: {e}")
    threading.Thread(target=_speak, daemon=True).start()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  VOICE TRANSCRIPTION                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def transcribe_voice(audio_path: str, lang: str = "en") -> str:
    """Attempt offline transcription using Vosk. Return text or error."""
    if not audio_path:
        return ""
    try:
        from voice import transcribe_audio_file
        return transcribe_audio_file(audio_path, lang=lang)
    except FileNotFoundError as e:
        return f"[Voice model not found: {e}]"
    except Exception as e:
        return f"[Voice error: {e}]"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  GRADIO UI                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

engine = TriageEngine()

def process_audio_input(audio_path, current_text, language):
    """Triggered when recording stops. Transcribes and appends to textbox."""
    if not audio_path:
        return current_text
        
    lang = "hi" if language == "हिन्दी" else "en"
    
    import shutil
    stable_path = os.path.join(tempfile.gettempdir(), "triage_voice_input.wav")
    try:
        shutil.copy2(audio_path, stable_path)
        transcribed = transcribe_voice(stable_path, lang)
        
        if transcribed and not transcribed.startswith("["):
            # Append if there's already text, otherwise replace
            if current_text and current_text.strip():
                return f"{current_text} {transcribed}"
            return transcribed
        return current_text  # If failed, return what they had
    except Exception as e:
        print(f"[VOICE] Error in auto-transcribe: {e}")
        return current_text

def run_triage(symptoms_text, age, gender_choice, language):
    lang = "hi" if language == "हिन्दी" else "en"
    gender = 1 if gender_choice in ("Male", "पुरुष") else 0
    age = int(age) if age else 30

    print(f"\n{'='*50}")
    print(f"[TRIAGE] text='{symptoms_text}', age={age}, lang={lang}")

    # Voice is now pre-processed into symptoms_text
    result = engine.predict(symptoms_text, age=age, gender=gender, lang=lang)
    print(f"[TRIAGE] urgency={result['urgency']}, disease={result['disease']}, detected={result['detected']}")
    formatted = format_result(result, lang)

    # Speak advice aloud in background (non-blocking)
    if result.get("advice"):
        speak_advice(
            f"{result.get('disease', '')}. {result.get('advice', '')}",
            lang
        )

    return formatted


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
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

/* ── Animated Background ────────────────────────────────────────── */
body, .gradio-container {
    background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #0b3b42);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
    font-family: 'Outfit', 'Noto Sans Devanagari', sans-serif !important;
    color: #edf2f4 !important;
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* ── Container & Glassmorphism ─────────────────────────────────── */
.gradio-container {
    max-width: 850px !important;
    margin: 40px auto !important;
    padding: 30px !important;
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
}

/* ── Header ────────────────────────────────────────────────────── */
.gradio-container h1 {
    text-align: center;
    color: #ffffff !important;
    font-weight: 700;
    font-size: 2.4em;
    text-shadow: 0 2px 10px rgba(0,0,0,0.2);
    margin-bottom: 5px;
}
.gradio-container h3 {
    text-align: center;
    color: #cfd8dc !important;
    font-weight: 300;
    margin-top: 0;
}

/* ── Cards / Sections (The internal Gradio blocks) ─────────────── */
div.gr-box, div.gr-panel, div.gr-form, .gradio-html {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1) !important;
}

/* Labels */
span.svelte-1gfkn6j {
    color: #e0e0e0 !important;
    font-weight: 500 !important;
}

/* ── Input fields ────────────────────────────────────────────── */
textarea, input[type="number"], select, .gr-input {
    background: rgba(0, 0, 0, 0.2) !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    font-size: 1.05em !important;
    transition: all 0.3s ease !important;
}
textarea:focus, input:focus, select:focus, .gr-input:focus {
    border-color: #48cae4 !important;
    box-shadow: 0 0 0 3px rgba(72, 202, 228, 0.3) !important;
    background: rgba(0, 0, 0, 0.4) !important;
    outline: none !important;
}

/* Placeholder color */
::placeholder {
    color: rgba(255,255,255,0.4) !important;
}

/* Dropdown list items inside select */
.gr-dropdown-list {
    background: #1e293b !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}
.gr-dropdown-list li {
    color: white !important;
}
.gr-dropdown-list li:hover {
    background: #334155 !important;
}

/* ── Buttons ─────────────────────────────────────────────────── */
button.primary {
    background: linear-gradient(135deg, #00b4d8, #0077b6) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    font-size: 1.2em !important;
    padding: 14px 28px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 15px rgba(0, 119, 182, 0.4) !important;
    transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
}
button.primary:hover {
    background: linear-gradient(135deg, #48cae4, #0096c7) !important;
    box-shadow: 0 6px 20px rgba(0, 119, 182, 0.6) !important;
    transform: translateY(-2px);
}
button.primary:active {
    transform: translateY(1px);
    box-shadow: 0 2px 10px rgba(0, 119, 182, 0.4) !important;
}

/* Audio component styling */
.audio-recorder {
    border-color: rgba(255,255,255,0.2) !important;
    background: rgba(0,0,0,0.1) !important;
}
button[aria-label="Record audio"] {
    background: rgba(239, 68, 68, 0.2) !important;
    border: 1px solid rgba(239, 68, 68, 0.5) !important;
}
button[aria-label="Record audio"]:hover {
    background: rgba(239, 68, 68, 0.4) !important;
}

/* ── Markdown Result Output ──────────────────────────────────── */
.output-markdown {
    color: #edf2f4 !important;
    font-size: 1.1em;
}
.output-markdown h2 {
    padding: 16px;
    border-radius: 10px;
    margin-top: 10px;
    font-weight: 700;
    text-shadow: 0 1px 2px rgba(0,0,0,0.2);
}
.output-markdown h3 {
    color: #90e0ef !important;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    padding-bottom: 5px;
}
.output-markdown p {
    line-height: 1.6;
}

/* Examples rendering */
.gr-samples-gallery button {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #e0e0e0 !important;
    transition: all 0.2s;
}
.gr-samples-gallery button:hover {
    background: rgba(72,202,228,0.2) !important;
    border-color: rgba(72,202,228,0.5) !important;
}

/* ── Footer ──────────────────────────────────────────────────── */
footer { 
    opacity: 0.5;
    text-align: center;
    border-top: 1px solid rgba(255,255,255,0.1) !important;
    padding-top: 15px;
}
/* Hide default Gradio footer logo */
footer .svelte-15x3029 {
    display: none !important;
}
"""

with gr.Blocks(css=CSS, title="🏥 Rural Health Triage") as demo:

    # ── Header ────────────────────────────────────────────────
    title_md    = gr.Markdown(UI_TEXT["en"]["title"])
    subtitle_md = gr.Markdown(UI_TEXT["en"]["subtitle"])

    # ── Language picker ───────────────────────────────────────
    with gr.Row():
        lang_dd = gr.Dropdown(
            ["English", "हिन्दी"], value="English",
            label=UI_TEXT["en"]["lang_label"], scale=1,
        )

    # ── Patient info ──────────────────────────────────────────
    with gr.Row():
        age_box = gr.Number(
            value=30, label=UI_TEXT["en"]["age_label"],
            minimum=0, maximum=120, precision=0, scale=1,
        )
        gender_dd = gr.Dropdown(
            UI_TEXT["en"]["gender_choices"], value="Male",
            label=UI_TEXT["en"]["gender_label"], scale=1,
        )

    # ── Symptom text input ────────────────────────────────────
    symptom_box = gr.Textbox(
        lines=3,
        placeholder=UI_TEXT["en"]["symptom_placeholder"],
        label=UI_TEXT["en"]["symptom_label"],
    )

    # ── Voice input ───────────────────────────────────────────
    voice_input = gr.Audio(
        sources=["microphone"],
        type="filepath",
        label=UI_TEXT["en"]["voice_label"],
    )

    # ── Submit ────────────────────────────────────────────────
    submit_btn = gr.Button(
        UI_TEXT["en"]["submit"], variant="primary", size="lg",
    )

    # ── Outputs ───────────────────────────────────────────────
    output_md = gr.Markdown(label=UI_TEXT["en"]["result_label"])

    # ── Language toggle → update all labels ───────────────────
    def update_ui_language(language):
        lang = "hi" if language == "हिन्दी" else "en"
        t = UI_TEXT[lang]
        return (
            gr.update(value=t["title"]),           # title_md
            gr.update(value=t["subtitle"]),         # subtitle_md
            gr.update(label=t["age_label"]),         # age_box
            gr.update(label=t["gender_label"],       # gender_dd
                      choices=t["gender_choices"],
                      value=t["gender_choices"][0]),
            gr.update(label=t["symptom_label"],      # symptom_box
                      placeholder=t["symptom_placeholder"]),
            gr.update(label=t["voice_label"]),        # voice_input
            gr.update(value=t["submit"]),             # submit_btn
            gr.update(label=t["result_label"]),       # output_md
        )

    lang_dd.change(
        fn=update_ui_language,
        inputs=[lang_dd],
        outputs=[title_md, subtitle_md, age_box, gender_dd,
                 symptom_box, voice_input, submit_btn, output_md],
    )

    # ── Submit action ─────────────────────────────────────────
    # Transcript populated dynamically when user stops recording
    voice_input.stop_recording(
        fn=process_audio_input,
        inputs=[voice_input, symptom_box, lang_dd],
        outputs=[symptom_box]
    )

    submit_btn.click(
        fn=run_triage,
        inputs=[symptom_box, age_box, gender_dd, lang_dd],
        outputs=[output_md],
    )

    # ── Example cases ─────────────────────────────────────────
    gr.Examples(
        examples=EXAMPLE_CASES,
        inputs=[symptom_box, age_box, gender_dd, lang_dd],
        outputs=[output_md],
        fn=lambda s, a, g, l: run_triage(s, a, g, l),
        cache_examples=False,
        label=UI_TEXT["en"]["examples_label"],
    )

    # ── Footer ────────────────────────────────────────────────
    gr.Markdown(UI_TEXT["en"]["footer"])


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)