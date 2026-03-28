import os
import requests
import zipfile
import io

MODELS = {
    "en": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
    "hi": "https://alphacephei.com/vosk/models/vosk-model-hi-0.22.zip"
}

VOICE_DIR = os.path.join(os.getcwd(), "voice")

def download_and_extract(lang, url):
    print(f"Downloading {lang} model from {url}...")
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        
        # Zip content
        z = zipfile.ZipFile(io.BytesIO(r.content))
        print(f"Extracting {lang} model into {VOICE_DIR}...")
        z.extractall(VOICE_DIR)
        print(f"Extraction of {lang} model complete.")
    except Exception as e:
        print(f"Failed to setup {lang} model: {e}")

if __name__ == "__main__":
    if not os.path.exists(VOICE_DIR):
        os.makedirs(VOICE_DIR)
        
    for lang, url in MODELS.items():
        download_and_extract(lang, url)
        
    print("\nVosk models setup complete. You can now use offline voice features.")
