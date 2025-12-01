from collections import Counter, deque
import json
import os
import sys
from pathlib import Path
import threading

import cv2
import mediapipe as mp
import torch
from flask import Flask, Response, render_template, send_from_directory, request

# Paths
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"

# Ensure local src/ modules (config_voice, model, etc.) are importable when running from repo root.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from config_voice import CONF_THRESHOLD, MODEL_PATH, PREDICTION_STABILITY, WINDOW_SIZE
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


def hands_present(results):
    """Return True if at least one hand is detected."""
    return results.left_hand_landmarks or results.right_hand_landmarks


def generate_frames():
    """Stream webcam frames with inference overlays as MJPEG."""
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
                        probs = torch.softmax(logits, dim=1)
                        conf, pred_idx = torch.max(probs, dim=1)
                        conf = conf.item()
                        pred_idx = pred_idx.item()

                        if conf >= CONF_THRESHOLD:
                            prediction_buffer.append(pred_idx)
                            most_common, count = Counter(prediction_buffer).most_common(
                                1
                            )[0]

                            if count >= PREDICTION_STABILITY:
                                gesture = idx_to_label[most_common]
                                print(f"Detected gesture: {gesture}")
                                if not speak_gesture(gesture):
                                    print("Announcement skipped due to active cooldown.")
                                prediction_buffer.clear()
                        else:
                            prediction_buffer.clear()
            else:
                frame_buffer.clear()
                prediction_buffer.clear()

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


@app.route("/stop_infer", methods=["POST"])
def stop_infer():
    STOP_EVENT.set()
    frame_buffer.clear()
    prediction_buffer.clear()
    return ("stopped", 200)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
