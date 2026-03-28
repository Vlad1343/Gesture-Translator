"""Central configuration for gesture recognition and TTS."""

import os

try:
    from dotenv import load_dotenv
except ImportError:  # python-dotenv is optional
    load_dotenv = None

# Load environment variables from a .env file if available.
if load_dotenv:
    load_dotenv()

# ------------------------ MODEL / INFERENCE ------------------------
# Override these with environment variables if you need different values.
MODEL_PATH = os.getenv("MODEL_PATH", "models/classifier.pth")
DATA_DIR = os.getenv("DATA_DIR", "data/raw")
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "20"))
WINDOW_STRIDE = int(os.getenv("WINDOW_STRIDE", "2"))
PREDICTION_STABILITY = int(os.getenv("PREDICTION_STABILITY", "4"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.6"))
ANNOUNCEMENT_COOLDOWN_SECONDS = float(os.getenv("ANNOUNCEMENT_COOLDOWN_SECONDS", "3"))
MIN_TOP2_MARGIN = float(os.getenv("MIN_TOP2_MARGIN", "0.12"))
MAX_NORMALIZED_ENTROPY = float(os.getenv("MAX_NORMALIZED_ENTROPY", "0.65"))
PREDICTION_COOLDOWN_SECONDS = float(os.getenv("PREDICTION_COOLDOWN_SECONDS", "0.9"))

# ------------------------ TTS (ElevenLabs) ------------------------
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
# Set a different model if you prefer; v2 supports multilingual speech.
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
TTS_TIMEOUT_SECONDS = float(os.getenv("TTS_TIMEOUT_SECONDS", "10"))
# Avoid repeating the same gesture audio within this cooldown window (seconds).
TTS_REPEAT_COOLDOWN_SECONDS = float(os.getenv("TTS_REPEAT_COOLDOWN_SECONDS", "1.5"))
# Cache synthesized clips to avoid repeated network calls for common gestures.
TTS_CACHE_MAX_ITEMS = int(os.getenv("TTS_CACHE_MAX_ITEMS", "6"))

# Map gesture labels to spoken text. Adjust to match your label_map.json.
GESTURE_TO_TEXT = {
    "yes": "Yes",
    "no": "No",
    "sorry": "Sorry",
    "please-thank_you": "Please and thank you",
    "good-afternoon": "Good afternoon",
    "love": "We love Accelerate Me and Houser",
}

# Gestures in this set are eligible for announcements/inference smoothing.
ENABLED_GESTURES = set(GESTURE_TO_TEXT.keys())

# Fallback phrase if a gesture label is missing from GESTURE_TO_TEXT.
DEFAULT_GESTURE_TEXT = "Unknown gesture"
