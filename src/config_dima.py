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
MODEL_PATH = os.getenv("MODEL_PATH", "gesture_model.pth")
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "30"))
PREDICTION_STABILITY = int(os.getenv("PREDICTION_STABILITY", "5"))
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.8"))

# ------------------------ TTS (ElevenLabs) ------------------------
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "")
# Set a different model if you prefer; v2 supports multilingual speech.
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
TTS_TIMEOUT_SECONDS = float(os.getenv("TTS_TIMEOUT_SECONDS", "10"))
# Avoid repeating the same gesture audio within this cooldown window (seconds).
TTS_REPEAT_COOLDOWN_SECONDS = float(os.getenv("TTS_REPEAT_COOLDOWN_SECONDS", "1.5"))

# Map gesture labels to spoken text. Adjust to match your label_map.json.
GESTURE_TO_TEXT = {
    "yes": "Yes",
    "no": "No",
    "sorry": "Sorry",
    "please": "Please",
    "good afternoon": "Good afternoon",
}

# Fallback phrase if a gesture label is missing from GESTURE_TO_TEXT.
DEFAULT_GESTURE_TEXT = "Unknown gesture"