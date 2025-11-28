
import argparse
import json
import os
import time
import cv2
import numpy as np
import mediapipe as mp

from config import DATA_DIR, WINDOW_SIZE
from keypoints import extract_keypoints
from utils import draw_landmarks

mp_holistic = mp.solutions.holistic

with open("label_map.json", "r") as f:
    label_map = json.load(f)
GESTURES = [label_map[k] for k in sorted(label_map.keys(), key=int)]

# Default capture parameters; adjust MANUAL_* values below for quick tweaks
DEFAULT_REPS = 25
RECORD_SECONDS = 2.5
TRIM_HEAD = 4
TRIM_TAIL = 4

# Optional manual overrides so you can tweak behavior without CLI args.
# Example: MANUAL_GESTURES = ["hello"]; MANUAL_REPS = 10
MANUAL_GESTURES = ['please-thank_you']
MANUAL_REPS = 55


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def countdown(frame, text="Get ready", seconds=3):
    for i in range(seconds, 0, -1):
        display = frame.copy()
        cv2.putText(
            display,
            f"{text}: {i}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 0, 255),
            3,
        )
        cv2.imshow("Recording Gesture", display)
        cv2.waitKey(1000)


def next_index(gesture_dir):
    existing = [int(f.split(".")[0]) for f in os.listdir(gesture_dir) if f.endswith(".npy")]
    return max(existing) + 1 if existing else 0


def trim_sequence(frames):
    if len(frames) <= TRIM_HEAD + TRIM_TAIL:
        return frames
    return frames[TRIM_HEAD : len(frames) - TRIM_TAIL]


def record_gestures(selected_gestures, reps):
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        for gesture in selected_gestures:
            gesture_dir = os.path.join(DATA_DIR, gesture)
            ensure_dir(gesture_dir)

            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                countdown(frame, text=f"Prepare '{gesture}'", seconds=3)

            start_idx = next_index(gesture_dir)

            for rep in range(reps):
                print(f"Recording {gesture}, sample {start_idx + rep}")
                frames = []
                start_time = time.time()

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(rgb)

                    keypoints = extract_keypoints(results)
                    frames.append(keypoints)

                    frame = draw_landmarks(frame, results)
                    cv2.imshow("Recording Gesture", frame)

                    if time.time() - start_time > RECORD_SECONDS:
                        break
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                frames = trim_sequence(frames)
                if len(frames) < WINDOW_SIZE:
                    print("Sequence too short, discarded. Hold the pose a bit longer.")
                    continue

                rep_path = os.path.join(gesture_dir, f"{start_idx + rep}.npy")
                np.save(rep_path, np.array(frames, dtype=np.float32))

    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    parser = argparse.ArgumentParser(description="Record gesture samples one label at a time.")
    parser.add_argument(
        "-g",
        "--gesture",
        action="append",
        dest="gestures",
        help="Gesture label(s) to record. Repeat or comma-separate. Defaults to config/manual selection.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        help=f"Number of recordings per gesture (default {DEFAULT_REPS}).",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available gesture labels and exit.",
    )
    return parser.parse_args()


def _normalize_requested_list(requested):
    if not requested:
        return []
    if isinstance(requested, str):
        requested = [requested]
    normalized = []
    for entry in requested:
        normalized.extend([item.strip() for item in str(entry).split(",") if item.strip()])
    return normalized


def _resolve_gesture_list(cli_values, manual_values):
    for source in (cli_values, manual_values):
        normalized = _normalize_requested_list(source)
        if not normalized:
            continue

        invalid = [name for name in normalized if name not in GESTURES]
        if invalid:
            raise ValueError(f"Unknown gestures requested: {', '.join(invalid)}")

        seen = set()
        ordered = []
        for name in normalized:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered
    return GESTURES


def _resolve_reps(cli_reps):
    if cli_reps is not None:
        reps = cli_reps
    elif MANUAL_REPS is not None:
        reps = MANUAL_REPS
    else:
        reps = DEFAULT_REPS

    if reps < 1:
        raise ValueError("Number of repetitions must be positive.")
    return reps


if __name__ == "__main__":
    args = parse_args()

    if args.list:
        print("Available gestures:")
        for name in GESTURES:
            print(f" - {name}")
        raise SystemExit(0)

    try:
        target_gestures = _resolve_gesture_list(args.gestures, MANUAL_GESTURES)
        target_reps = _resolve_reps(args.reps)
    except ValueError as exc:
        print(exc)
        raise SystemExit(1)

    ensure_dir(DATA_DIR)
    print(f"Recording gestures: {', '.join(target_gestures)} | reps per gesture: {target_reps}")
    record_gestures(target_gestures, target_reps)
