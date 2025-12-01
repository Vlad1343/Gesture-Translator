"""Lightweight ElevenLabs TTS helper for gesture labels."""

import logging
import os
import subprocess
import tempfile
import time
from typing import Optional

import requests

from config import (
    DEFAULT_GESTURE_TEXT,
    ELEVENLABS_API_KEY,
    ELEVENLABS_MODEL_ID,
    ELEVENLABS_VOICE_ID,
    GESTURE_TO_TEXT,
    TTS_REPEAT_COOLDOWN_SECONDS,
    TTS_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

ELEVENLABS_URL_TEMPLATE = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
AUDIO_MIME_TYPE = "audio/mpeg"

_last_spoken_text: Optional[str] = None
_last_spoken_at: Optional[float] = None


def speak_gesture(gesture: str) -> None:
    """Convert a gesture label to speech via ElevenLabs; errors are logged quietly."""
    text = GESTURE_TO_TEXT.get(gesture, DEFAULT_GESTURE_TEXT)
    if not text:
        logger.debug("Gesture '%s' not mapped to text; skipping TTS.", gesture)
        return

    global _last_spoken_text, _last_spoken_at
    now = time.monotonic()
    if _last_spoken_text == text and _last_spoken_at is not None:
        if now - _last_spoken_at < TTS_REPEAT_COOLDOWN_SECONDS:
            return

    if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
        logger.warning(
            "ElevenLabs not configured; set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID."
        )
        return

    try:
        audio_bytes = _fetch_tts_audio(text)
    except Exception as exc:  # noqa: BLE001
        logger.error("ElevenLabs TTS failed: %s", exc)
        return

    if not audio_bytes:
        logger.error("No audio returned for gesture '%s'.", gesture)
        return

    if not _play_audio_bytes(audio_bytes):
        logger.warning("Could not play audio for gesture '%s'.", gesture)
    else:
        _last_spoken_text = text
        _last_spoken_at = now


def _fetch_tts_audio(text: str) -> bytes:
    url = ELEVENLABS_URL_TEMPLATE.format(voice_id=ELEVENLABS_VOICE_ID)
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Accept": AUDIO_MIME_TYPE,
        "Content-Type": "application/json",
    }
    payload = {
        "text": text,
        "model_id": ELEVENLABS_MODEL_ID,
        "voice_settings": {
            "stability": 0.35,
            "similarity_boost": 0.75,
        },
    }

    response = requests.post(url, headers=headers, json=payload, timeout=TTS_TIMEOUT_SECONDS)
    response.raise_for_status()
    if not response.content:
        raise ValueError("Empty ElevenLabs response.")
    return response.content


def _play_audio_bytes(audio_bytes: bytes) -> bool:
    """Persist audio to a temp file and play it with available system tools."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
        tmp_file.write(audio_bytes)
        temp_path = tmp_file.name

    try:
        return _play_file(temp_path)
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            logger.debug("Could not delete temp audio file: %s", temp_path)


def _play_file(file_path: str) -> bool:
    # Try lightweight system players first to keep latency low.
    playback_commands = [
        ["afplay", file_path],  # macOS
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", file_path],
        ["mpg123", "-q", file_path],
    ]

    for cmd in playback_commands:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError:
            continue

    # Python fallback if no system player is available.
    try:
        from pydub import AudioSegment
        from pydub.playback import play

        audio = AudioSegment.from_file(file_path)
        play(audio)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("Audio playback fallback failed: %s", exc)
        return False