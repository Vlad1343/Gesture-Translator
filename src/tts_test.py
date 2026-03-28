"""Manual tester for ElevenLabs TTS integration."""

import argparse
import time

from config_voice import GESTURE_TO_TEXT
from tts_elevenlabs import speak_gesture


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Play mapped gesture phrases via ElevenLabs TTS."
    )
    parser.add_argument(
        "gesture",
        nargs="?",
        help="Gesture label to speak once (must match your label_map).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="play_all",
        help="Speak every configured gesture in sequence.",
    )
    args = parser.parse_args()

    if args.play_all:
        gestures = list(GESTURE_TO_TEXT.keys())
    elif args.gesture:
        gestures = [args.gesture]
    else:
        parser.print_help()
        return 1

    for gesture in gestures:
        phrase = GESTURE_TO_TEXT.get(gesture, "Unknown gesture")
        print(f"Playing gesture '{gesture}' -> '{phrase}'")
        speak_gesture(gesture)
        # Small pause to avoid hitting cooldown between different gestures on slow systems.
        time.sleep(0.2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
