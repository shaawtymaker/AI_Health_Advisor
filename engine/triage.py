import os, json
import numpy as np
from xgboost import XGBClassifier
from engine.keywords import KEYWORD_MAP
from engine.red_flags import RED_FLAG_RULES
from engine.severity import calculate_severity
from engine.followup import get_followup_question

# Load diseases db dynamically
_data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_diseases_file = os.path.join(_data_dir, "diseases.json")
try:
    with open(_diseases_file, "r", encoding="utf-8") as f:
        DISEASE_INFO = json.load(f)
except FileNotFoundError:
    DISEASE_INFO = {}

class TriageEngine:
    def __init__(self, model_path="models/triage_model.json",
                 meta_path="models/metadata.json"):
        # Resolve absolute paths from root
        root = os.path.dirname(os.path.dirname(__file__))
        model_path = os.path.join(root, model_path)
        meta_path = os.path.join(root, meta_path)
        
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
            if isinstance(shap_values, list):
                vals = shap_values[predicted_class][0]
            else:
                vals = shap_values[0]

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
            msg = flags[0].get("hi" if lang == "hi" else "en", "Emergency Detected")
            return dict(
                urgency="RED", detected=detected, confidence=1.0,
                disease="Emergency" if lang == "en" else "आपातकालीन",
                explanation=msg,
                advice="Call 108 ambulance or go to nearest hospital NOW."
                       if lang == "en"
                       else "108 एम्बुलेंस बुलाएं या तुरंत नज़दीकी अस्पताल जाएं।",
                top3=[], is_flag=True, shap_text="",
                severity_score=100, followup="",
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
                severity_score=0, followup="",
            )

        # — model inference —
        probs    = self.model.predict_proba(vec)[0]
        top3_idx = np.argsort(probs)[::-1][:3]
        best     = self.diseases[top3_idx[0]]
        urgency  = self.urgency_map[best]
        info     = DISEASE_INFO.get(best, {})

        top3 = []
        for i in top3_idx:
            d = self.diseases[i]
            di = DISEASE_INFO.get(d, {})
            top3.append({
                "disease": di.get(lang, d),
                "prob": float(probs[i]),
                "urgency": self.urgency_map[d],
            })

        detected_display = ", ".join(s.replace("_", " ") for s in detected[:6])
        expl = (f"Based on: {detected_display}"
                if lang == "en"
                else f"लक्षणों के आधार पर: {detected_display}")

        shap_text = self._explain(vec, top3_idx[0], lang)
        
        # Calculate Advanced Severity & Context
        conf = float(probs[top3_idx[0]])
        severity_score = calculate_severity(urgency, conf, len(detected), False)
        followup = get_followup_question(detected, lang)

        return dict(
            urgency=urgency, detected=detected,
            confidence=conf,
            disease=info.get(lang, best), explanation=expl,
            advice=info.get(f"adv_{lang}", ""), top3=top3, is_flag=False,
            shap_text=shap_text,
            severity_score=severity_score,
            followup=followup,
        )
