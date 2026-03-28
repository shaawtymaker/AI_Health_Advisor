// Register Service Worker for PWA
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/service-worker.js')
            .then(reg => console.log('Service Worker Registered'))
            .catch(err => console.log('Service Worker Registration Failed', err));
    });
}

// DOM Elements
const pages = {
    splash: document.getElementById('page-splash'),
    input: document.getElementById('page-input'),
    result: document.getElementById('page-result'),
};

const btnStart = document.getElementById('btn-start');
const btnBack1 = document.getElementById('btn-back-1');
const btnRestart = document.getElementById('btn-restart');
const btnEvaluate = document.getElementById('btn-evaluate');
const btnMic = document.getElementById('btn-mic');
const micStatus = document.getElementById('mic-status');

const langSelect = document.getElementById('lang-select');
const symptomInput = document.getElementById('symptom-input');
const ageInput = document.getElementById('age-input');
const genderInput = document.getElementById('gender-input');
const loadingOverlay = document.getElementById('loading-overlay');

// ── State Management ──────────────────────────────────────────────
let currentLang = 'en';
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];

// Dictionary for simple UI swaps
const i18n = {
    "en": {
        splashSub: "AI-powered offline symptom checker. Tell us what's wrong, get immediate medical guidance.",
        btnStart: "Start Triage 🚀",
        lblAge: "Age",
        lblGender: "Gender",
        lblSymptoms: "What are your symptoms?",
        plhSymptoms: "e.g. High fever for 3 days, severe headache, and vomiting.",
        btnEval: "Evaluate Symptoms 🩺",
        loading: "Analyzing securely on device...",
        headAssess: "Assessment",
        headResult: "Triage Result",
        lblAdvice: "💊 Medical Advice",
        lblDiff: "📋 Other Possibilities",
        lblClinics: "🏥 Nearest Clinics",
    },
    "hi": {
        splashSub: "AI-संचालित ऑफ़लाइन लक्षण जांचकर्ता। अपनी समस्या बताएं, तुरंत चिकित्सा सलाह पाएं।",
        btnStart: "शुरू करें 🚀",
        lblAge: "उम्र",
        lblGender: "लिंग",
        lblSymptoms: "आपके लक्षण क्या हैं?",
        plhSymptoms: "जैसे: 3 दिन से तेज़ बुखार, सिर दर्द, और उल्टी।",
        btnEval: "लक्षण जांचें 🩺",
        loading: "डिवाइस पर सुरक्षित रूप से विश्लेषण कर रहा है...",
        headAssess: "लक्षण दर्ज करें",
        headResult: "परिणाम",
        lblAdvice: "💊 चिकित्सा सलाह",
        lblDiff: "📋 अन्य संभावनाएं",
        lblClinics: "🏥 नज़दीकी क्लिनिक",
    }
};

// ── Navigation ───────────────────────────────────────────────────
function showPage(pageId) {
    Object.values(pages).forEach(p => p.classList.add('hidden'));
    Object.values(pages).forEach(p => p.classList.remove('active'));
    
    setTimeout(() => {
        pages[pageId].classList.remove('hidden');
        // Force reflow
        void pages[pageId].offsetWidth; 
        pages[pageId].classList.add('active');
    }, 50);
}

btnStart.addEventListener('click', () => showPage('input'));
btnBack1.addEventListener('click', () => showPage('splash'));
btnRestart.addEventListener('click', () => {
    symptomInput.value = '';
    showPage('input');
});

// ── Translation ──────────────────────────────────────────────────
langSelect.addEventListener('change', (e) => {
    currentLang = e.target.value;
    const t = i18n[currentLang];
    
    document.getElementById('splash-subtitle').innerText = t.splashSub;
    btnStart.innerText = t.btnStart;
    document.getElementById('lbl-age').innerText = t.lblAge;
    document.getElementById('lbl-gender').innerText = t.lblGender;
    document.getElementById('lbl-symptoms').innerText = t.lblSymptoms;
    symptomInput.placeholder = t.plhSymptoms;
    btnEvaluate.innerText = t.btnEval;
    document.getElementById('loading-text').innerText = t.loading;
    document.getElementById('header-assessment').innerText = t.headAssess;
    document.getElementById('header-result').innerText = t.headResult;
    document.getElementById('lbl-advice').innerText = t.lblAdvice;
    document.getElementById('lbl-diff').innerText = t.lblDiff;
    document.getElementById('lbl-clinics').innerText = t.lblClinics;
});

// ── Chips ────────────────────────────────────────────────────────
document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', (e) => {
        const val = e.target.getAttribute('data-val');
        if (symptomInput.value.length > 0) {
            symptomInput.value += ', ' + val;
        } else {
            symptomInput.value = val;
        }
    });
});

// ── Helper: Markdown to simple HTML ──────────────────────────────
function parseMarkdown(md) {
    if(!md) return "";
    let html = md.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    html = html.replace(/\n/g, '<br>');
    return html;
}

// ── Triage API Call ──────────────────────────────────────────────
btnEvaluate.addEventListener('click', async () => {
    const text = symptomInput.value.trim();
    if (!text) {
        alert(currentLang === 'en' ? "Please enter your symptoms." : "कृपया अपने लक्षण दर्ज करें।");
        return;
    }

    loadingOverlay.classList.remove('hidden');

    try {
        const res = await fetch('http://127.0.0.1:8000/triage', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symptoms: text,
                age: parseInt(ageInput.value) || 30,
                gender: genderInput.value,
                language: currentLang
            })
        });

        if (!res.ok) throw new Error("API Error");
        const data = await res.json();
        
        renderResult(data);
        showPage('result');
        
    } catch (err) {
        console.error(err);
        alert("Make sure the backend is running via `uvicorn api:app`");
    } finally {
        loadingOverlay.classList.add('hidden');
    }
});

// ── Render Result ────────────────────────────────────────────────
function renderResult(data) {
    // Colors
    const colors = {
        "RED": "var(--danger)",
        "YELLOW": "var(--warn)",
        "GREEN": "var(--success)",
        "NONE": "#888888"
    };
    const color = colors[data.urgency] || colors["NONE"];
    
    // URgEnCY banner overrides if Followup is needed
    if(data.followup) {
        document.getElementById('followup-box').classList.remove('hidden');
        document.getElementById('followup-text').innerText = data.followup;
    } else {
        document.getElementById('followup-box').classList.add('hidden');
    }

    // Urgency Banner
    const banner = document.getElementById('urgency-banner');
    banner.style.backgroundColor = color;
    
    const icons = {"RED": "🔴", "YELLOW": "🟡", "GREEN": "🟢", "NONE": "⚪"};
    document.getElementById('urgency-icon').innerText = icons[data.urgency] || "⚪";
    
    const labelsEn = {"RED": "URGENT — GO TO HOSPITAL NOW", "YELLOW": "SEE DOCTOR SOON", "GREEN": "HOME CARE — MONITOR", "NONE": "UNKNOWN"};
    const labelsHi = {"RED": "गंभीर — तुरंत अस्पताल जाएं", "YELLOW": "जल्द डॉक्टर को दिखाएं", "GREEN": "घर पर देखभाल — निगरानी रखें", "NONE": "अज्ञात"};
    document.getElementById('urgency-label').innerText = currentLang === 'en' ? labelsEn[data.urgency] : labelsHi[data.urgency];

    // Card Details
    document.getElementById('disease-name').innerText = data.disease || (currentLang === 'en' ? "Uncertain" : "अनिश्चित");
    document.getElementById('disease-confidence').innerText = `${(data.confidence * 100).toFixed(0)}%`;
    document.getElementById('severity-score').innerText = data.severity_score || "0";
    document.getElementById('disease-explanation').innerHTML = parseMarkdown(data.explanation);
    
    const shapEl = document.getElementById('shap-factors');
    shapEl.innerHTML = parseMarkdown(data.shap_text);
    shapEl.style.display = data.shap_text ? "block" : "none";

    // Advice
    document.getElementById('advice-text').innerHTML = parseMarkdown(data.advice);

    // Differential
    const diffList = document.getElementById('differential-list');
    diffList.innerHTML = "";
    if (data.top3 && data.top3.length > 0) {
        document.getElementById('differential-box').style.display = 'block';
        data.top3.forEach(d => {
            const li = document.createElement('li');
            li.innerHTML = `${icons[d.urgency] || "⚪"} <strong>${d.disease}</strong> (${(d.prob * 100).toFixed(0)}%)`;
            diffList.appendChild(li);
        });
    } else {
        document.getElementById('differential-box').style.display = 'none';
    }

    // Overwrite clinics rendering (temporarily stubbed since API return isn't giving clinics directly yet in V4 base)
    const clinicsList = document.getElementById('clinics-list');
    clinicsList.innerHTML = "<li>" + (currentLang === 'en' ? "Loading local clinics..." : "स्थानीय क्लिनिक लोड हो रहे हैं...") + "</li>";
}

// ── Voice Recording API ──────────────────────────────────────────
btnMic.addEventListener('click', async () => {
    if (!isRecording) {
        try {
            // Request audio with constraints
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    channelCount: 1,
                    sampleRate: 16000,
                    echoCancellation: true
                } 
            });
            
            // Check supported mime types
            const mimeType = MediaRecorder.isTypeSupported('audio/webm') 
                             ? 'audio/webm' 
                             : 'audio/ogg';
                             
            mediaRecorder = new MediaRecorder(stream, { mimeType });
            audioChunks = [];

            mediaRecorder.ondataavailable = e => {
                if (e.data.size > 0) audioChunks.push(e.data);
            };

            mediaRecorder.onstop = async () => {
                // UI: Transcribing state
                micStatus.innerText = currentLang === 'en' ? "Transcribing..." : "लिख रहा है...";
                micStatus.classList.remove('hidden');
                
                const audioBlob = new Blob(audioChunks, { type: mimeType });
                const formData = new FormData();
                formData.append("file", audioBlob, `recording.${mimeType.split('/')[1]}`);
                formData.append("lang", currentLang);
                
                try {
                    const res = await fetch('/triage/voice', {
                        method: 'POST',
                        body: formData
                    });
                    const result = await res.json();
                    
                    if(result.text) {
                        const existing = symptomInput.value.trim();
                        symptomInput.value = existing ? existing + " " + result.text : result.text;
                        // Trigger pulse for user visual confirmation
                        symptomInput.style.borderColor = "var(--success)";
                        setTimeout(() => symptomInput.style.borderColor = "var(--border-glass)", 1000);
                    } else if (result.error) {
                        console.error(result.error);
                        alert(result.error);
                    }
                } catch (e) {
                    console.error(e);
                    alert("Voice transcription failed. Ensure backend is running.");
                } finally {
                    micStatus.classList.add('hidden');
                    btnMic.classList.remove('recording');
                }
            };

            // Start Recording
            mediaRecorder.start();
            isRecording = true;
            btnMic.classList.add('recording');
            micStatus.innerText = currentLang === 'en' ? "Listening..." : "सुन रहा है...";
            micStatus.classList.remove('hidden');

            // Timeout safety (20s max)
            setTimeout(() => {
                if(isRecording) btnMic.click();
            }, 20000);

        } catch (e) {
            console.error(e);
            alert("Microphone access denied or not supported.");
        }
    } else {
        // Stop Recording
        if(mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            mediaRecorder.stream.getTracks().forEach(t => t.stop());
        }
        isRecording = false;
    }
});
