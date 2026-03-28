"""Lightweight ElevenLabs TTS helper for gesture labels."""

import logging
import os
import queue
import subprocess
import tempfile
import threading
import time
from collections import OrderedDict
from typing import Optional

import requests

from config_voice import (
    ANNOUNCEMENT_COOLDOWN_SECONDS,
    ELEVENLABS_API_KEY,
    ELEVENLABS_MODEL_ID,
    ELEVENLABS_VOICE_ID,
    GESTURE_TO_TEXT,
    TTS_CACHE_MAX_ITEMS,
    TTS_REPEAT_COOLDOWN_SECONDS,
    TTS_TIMEOUT_SECONDS,
)

logger = logging.getLogger(__name__)

ELEVENLABS_URL_TEMPLATE = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
AUDIO_MIME_TYPE = "audio/mpeg"

_last_spoken_text: Optional[str] = None
_last_spoken_at: Optional[float] = None
_last_enqueued_text: Optional[str] = None
_last_enqueued_at: Optional[float] = None
_state_lock = threading.Lock()
_audio_cache: "OrderedDict[str, bytes]" = OrderedDict()
_tts_queue: "queue.Queue[str]" = queue.Queue()
_worker_started = False


def speak_gesture(gesture: str) -> bool:
    """
    Convert a gesture label to speech via ElevenLabs, non-blocking.
    Enqueues playback on a background worker to avoid freezing the CV loop.
    Returns True when the audio request was accepted, False when throttled/skipped.
    """
    global _last_enqueued_text, _last_enqueued_at

    text = GESTURE_TO_TEXT.get(gesture)
    if not text:
        # Unknown gesture: do not play anything.
        logger.debug("Gesture '%s' not mapped to text; skipping TTS.", gesture)
        return False

    if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
        logger.warning(
            "ElevenLabs not configured; set ELEVENLABS_API_KEY and ELEVENLABS_VOICE_ID."
        )
        return False

    _start_worker_if_needed()

    now = time.monotonic()
    with _state_lock:
        # Enforce a hard cooldown between any announcements.
        if _last_spoken_at is not None:
            cooldown_elapsed = now - _last_spoken_at
            if cooldown_elapsed < ANNOUNCEMENT_COOLDOWN_SECONDS:
                return False

        if _last_spoken_text == text and _last_spoken_at is not None:
            if now - _last_spoken_at < TTS_REPEAT_COOLDOWN_SECONDS:
                return False
        if _last_enqueued_text == text and _last_enqueued_at is not None:
            if now - _last_enqueued_at < TTS_REPEAT_COOLDOWN_SECONDS:
                return False
        _last_enqueued_text = text
        _last_enqueued_at = now
        _clear_pending_queue()
    _tts_queue.put(text)
    return True


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


def _get_audio_bytes(text: str) -> bytes:
    """
    Return synthesized speech for the provided text, caching recent clips.
    The cache keeps latency low for repeated gestures by reusing MP3 bytes.
    """
    with _state_lock:
        cached = _audio_cache.get(text)
        if cached is not None:
            _audio_cache.move_to_end(text)
            return cached

    audio_bytes = _fetch_tts_audio(text)
    if not audio_bytes:
        return audio_bytes

    with _state_lock:
        _audio_cache[text] = audio_bytes
        while len(_audio_cache) > max(TTS_CACHE_MAX_ITEMS, 0):
            _audio_cache.popitem(last=False)
    return audio_bytes


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


def _worker_loop():
    global _last_spoken_text, _last_spoken_at
    while True:
        text = _tts_queue.get()
        try:
            audio_bytes = _get_audio_bytes(text)
            if not audio_bytes:
                logger.error("No audio returned for text '%s'.", text)
                continue

            if _play_audio_bytes(audio_bytes):
                with _state_lock:
                    _last_spoken_text = text
                    _last_spoken_at = time.monotonic()
            else:
                logger.warning("Could not play audio for text '%s'.", text)
        except Exception as exc:  # noqa: BLE001
            logger.error("ElevenLabs TTS failed for '%s': %s", text, exc)


def _clear_pending_queue() -> None:
    """Drop queued phrases to avoid long delays when gestures change quickly."""
    try:
        while True:
            _tts_queue.get_nowait()
    except queue.Empty:
        return


def _start_worker_if_needed():
    global _worker_started
    if _worker_started:
        return
    worker = threading.Thread(target=_worker_loop, daemon=True)
    worker.start()
    _worker_started = True
