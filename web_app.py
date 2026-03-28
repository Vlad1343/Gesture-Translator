from collections import Counter, deque
import json
import math
import os
import sys
from pathlib import Path
import threading
import time

import cv2
import mediapipe as mp
import torch
from flask import Flask, Response, jsonify, render_template, send_from_directory

# Paths
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"

# Ensure local src/ modules (config_voice, model, etc.) are importable when running from repo root.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_voice import (
    CONF_THRESHOLD,
    MAX_NORMALIZED_ENTROPY,
    MIN_TOP2_MARGIN,
    MODEL_PATH,
    PREDICTION_COOLDOWN_SECONDS,
    PREDICTION_STABILITY,
    WINDOW_SIZE,
)
from keypoints import KEYPOINT_VECTOR_LENGTH, extract_keypoints
from model import GestureLSTM
from tts_elevenlabs import speak_gesture
from utils import draw_landmarks


# ------------------------ CONFIG ------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Serve the built Vite frontend from frontend/dist (assets under /assets).
app = Flask(
    __name__,
    static_folder=str(FRONTEND_DIST / "assets"),
    static_url_path="/assets",
    template_folder=str(FRONTEND_DIST),
)


# ------------------------ LABELS ------------------------
with open("label_map.json", "r") as f:
    label_map = json.load(f)
idx_to_label = {int(k): v for k, v in label_map.items()}


# ------------------------ MODEL ------------------------
dummy_input = torch.zeros((1, WINDOW_SIZE, KEYPOINT_VECTOR_LENGTH))
input_size = dummy_input.size(2)
num_classes = len(idx_to_label)

model = GestureLSTM(input_size=input_size, num_classes=num_classes).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()


# ------------------------ BUFFERS ------------------------
frame_buffer = deque(maxlen=WINDOW_SIZE)
prediction_buffer = deque(maxlen=PREDICTION_STABILITY)
STOP_EVENT = threading.Event()
last_confirmed_at = 0.0
last_confirmed_label = None


def hands_present(results):
    """Return True if at least one hand is detected."""
    return results.left_hand_landmarks or results.right_hand_landmarks


def _decode_prediction_metrics(logits):
    """Extract confidence, class margin, and normalized entropy from logits."""
    probs = torch.softmax(logits, dim=1)
    topk = min(2, probs.size(1))
    top_probs, top_idxs = torch.topk(probs, k=topk, dim=1)

    confidence = top_probs[0, 0].item()
    pred_idx = top_idxs[0, 0].item()
    second_best = top_probs[0, 1].item() if topk > 1 else 0.0
    margin = confidence - second_best

    entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum(dim=1).item()
    normalized_entropy = entropy / math.log(max(probs.size(1), 2))
    return pred_idx, confidence, margin, normalized_entropy


def _should_accept_prediction(confidence, margin, normalized_entropy):
    """Gate uncertain predictions before adding them to temporal voting."""
    if confidence < CONF_THRESHOLD:
        return False
    if margin < MIN_TOP2_MARGIN:
        return False
    if normalized_entropy > MAX_NORMALIZED_ENTROPY:
        return False
    return True


def generate_frames():
    """Stream webcam frames with inference overlays as MJPEG."""
    global last_confirmed_at, last_confirmed_label
    STOP_EVENT.clear()
    cap = cv2.VideoCapture(0)
    mp_holistic = mp.solutions.holistic

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as holistic:
        while cap.isOpened() and not STOP_EVENT.is_set():
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            # Draw landmarks on frame
            frame = draw_landmarks(frame, results)

            # Prediction flow mirrors src/infer.py
            if hands_present(results):
                keypoints = extract_keypoints(results)
                frame_buffer.append(keypoints)

                if len(frame_buffer) == WINDOW_SIZE:
                    seq = torch.tensor([list(frame_buffer)], dtype=torch.float32).to(
                        DEVICE
                    )
                    with torch.no_grad():
                        logits = model(seq)
                        pred_idx, conf, margin, norm_entropy = _decode_prediction_metrics(
                            logits
                        )

                        if _should_accept_prediction(conf, margin, norm_entropy):
                            prediction_buffer.append(pred_idx)
                            most_common, count = Counter(prediction_buffer).most_common(
                                1
                            )[0]

                            if count >= PREDICTION_STABILITY:
                                gesture = idx_to_label[most_common]
                                now = time.monotonic()
                                can_emit = (
                                    gesture != last_confirmed_label
                                    or now - last_confirmed_at >= PREDICTION_COOLDOWN_SECONDS
                                )
                                if can_emit:
                                    print(
                                        "Detected gesture: "
                                        f"{gesture} conf={conf:.2f} margin={margin:.2f} "
                                        f"entropy={norm_entropy:.2f}"
                                    )
                                    if not speak_gesture(gesture):
                                        print(
                                            "Announcement skipped due to active cooldown."
                                        )
                                    last_confirmed_at = now
                                    last_confirmed_label = gesture
                                prediction_buffer.clear()
                        else:
                            prediction_buffer.clear()
            else:
                frame_buffer.clear()
                prediction_buffer.clear()
                last_confirmed_label = None
                last_confirmed_at = 0.0

            # Encode frame for MJPEG streaming
            ok, buffer = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            jpg_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
            )

    cap.release()


@app.route("/")
def index():
    # If the Vite build output is missing, fail fast with a clear message.
    if not FRONTEND_DIST.exists():
        return (
            "Frontend build not found. Run `cd frontend && npm install && npm run build` "
            "then restart the server.",
            500,
        )
    return render_template("index.html")


@app.route("/assets/<path:path>")
def assets(path):
    return send_from_directory(app.static_folder, path)


@app.route("/video_feed")
def video_feed():
    STOP_EVENT.clear()
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/healthz")
def healthz():
    return jsonify(
        {
            "status": "ok",
            "device": str(DEVICE),
            "num_classes": len(idx_to_label),
            "window_size": WINDOW_SIZE,
        }
    )


@app.route("/stop_infer", methods=["POST"])
def stop_infer():
    global last_confirmed_at, last_confirmed_label
    STOP_EVENT.set()
    frame_buffer.clear()
    prediction_buffer.clear()
    last_confirmed_at = 0.0
    last_confirmed_label = None
    return ("stopped", 200)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
