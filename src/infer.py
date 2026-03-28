# src/infer.py
import cv2
import torch
from utils import draw_landmarks
from keypoints import extract_keypoints, KEYPOINT_VECTOR_LENGTH
from model import GestureLSTM
import mediapipe as mp
import json
import warnings
import time
from collections import deque, Counter

from tts_elevenlabs import speak_gesture
from config_voice import (
    ANNOUNCEMENT_COOLDOWN_SECONDS,
    CONF_THRESHOLD,
    ENABLED_GESTURES,
    MODEL_PATH,
    PREDICTION_STABILITY,
    WINDOW_SIZE,
)

# Silence noisy protobuf warnings from MediaPipe
warnings.filterwarnings(
    "ignore",
    message="SymbolDatabase.GetPrototype.*",
    category=UserWarning,
    module="google.protobuf.symbol_database",
)

# ------------------------ CONFIG ------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HISTORY_SIZE = 10           # frames to consider for smoothing

# ------------------------ LABELS ------------------------
with open("label_map.json", "r") as f:
    label_map = json.load(f)
idx_to_label = {int(k): v for k, v in label_map.items()}
num_classes = len(idx_to_label)

# ------------------------ MODEL ------------------------
dummy_input = torch.zeros((1, WINDOW_SIZE, KEYPOINT_VECTOR_LENGTH))
input_size = dummy_input.size(2)

model = GestureLSTM(input_size=input_size, num_classes=num_classes).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ------------------------ BUFFERS ------------------------
frame_buffer = deque(maxlen=WINDOW_SIZE)
gesture_history = deque(maxlen=HISTORY_SIZE)
last_announced = None
candidate_label = None
candidate_repeat_count = 0
announcement_cooldown_until = 0.0

# ------------------------ HELPERS ------------------------
def hands_present(results):
    """Check if at least one hand is visible."""
    return results.left_hand_landmarks or results.right_hand_landmarks

# ------------------------ WEBCAM LOOP ------------------------
mp_holistic = mp.solutions.holistic
cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    refine_face_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:

    print("Press 'q' to quit")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        # Draw landmarks
        frame = draw_landmarks(frame, results)
        cv2.imshow("Gesture Recognition", frame)

        # Reset buffers if no hands
        if not hands_present(results):
            frame_buffer.clear()
            gesture_history.clear()
            last_announced = None
            candidate_label = None
            candidate_repeat_count = 0
            announcement_cooldown_until = 0.0
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Hold off on new predictions for a short window after speaking a gesture.
        if time.monotonic() < announcement_cooldown_until:
            frame_buffer.clear()
            gesture_history.clear()
            candidate_label = None
            candidate_repeat_count = 0
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # Extract keypoints and append to buffer
        keypoints = extract_keypoints(results)
        frame_buffer.append(keypoints)

        # Predict gesture if buffer full
        if len(frame_buffer) == WINDOW_SIZE:
            seq = torch.tensor([list(frame_buffer)], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                logits = model(seq)
                probs = torch.softmax(logits, dim=1)
                conf, pred_idx = torch.max(probs, dim=1)
                conf = conf.item()
                pred_idx = pred_idx.item()

            # Only consider confident predictions
            if conf >= CONF_THRESHOLD:
                label = idx_to_label[pred_idx]

                if label not in ENABLED_GESTURES:
                    # Ignore gestures we do not want to announce.
                    candidate_label = None
                    candidate_repeat_count = 0
                    continue

                gesture_history.append(pred_idx)
                # Require a stable majority vote before announcing the gesture aloud.
                most_common, _ = Counter(gesture_history).most_common(1)[0]
                label = idx_to_label[most_common]

                if most_common == candidate_label:
                    candidate_repeat_count += 1
                else:
                    candidate_label = most_common
                    candidate_repeat_count = 1

                if candidate_repeat_count <= PREDICTION_STABILITY:
                    print(
                        f"Gesture candidate '{label}' "
                        f"({candidate_repeat_count}/{PREDICTION_STABILITY})"
                    )

                if candidate_repeat_count >= PREDICTION_STABILITY:
                    now = time.monotonic()
                    if most_common != last_announced:
                        print(f"Gesture confirmed: {label}")
                        if speak_gesture(label):
                            last_announced = most_common
                        else:
                            print("Announcement skipped due to active cooldown.")

                    announcement_cooldown_until = now + ANNOUNCEMENT_COOLDOWN_SECONDS

                    candidate_label = None
                    candidate_repeat_count = 0
                    gesture_history.clear()

        # Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
