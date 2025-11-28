# src/infer.py
import cv2
import torch
from utils import draw_landmarks
from keypoints import extract_keypoints, KEYPOINT_VECTOR_LENGTH
from model import GestureLSTM
import mediapipe as mp
import json
from collections import deque, Counter

from config import (
    CONF_THRESHOLD,
    MODEL_PATH,
    PREDICTION_STABILITY,
    WINDOW_SIZE,
)

# ------------------------ CONFIG ------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

        # Draw landmarks on screen
        frame = draw_landmarks(frame, results)
        cv2.imshow("Gesture Recognition", frame)

        # Skip prediction if no hands detected
        if not hands_present(results):
            frame_buffer.clear()
            prediction_buffer.clear()
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

                if conf >= CONF_THRESHOLD:
                    prediction_buffer.append(pred_idx)
                    most_common, count = Counter(prediction_buffer).most_common(1)[0]

                    if count >= PREDICTION_STABILITY:
                        gesture = idx_to_label[most_common]
                        print(f"Detected gesture: {gesture}")
                        prediction_buffer.clear()
                else:
                    prediction_buffer.clear()

        # Exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
