# src/infer.py
import cv2
import torch
import numpy as np
from utils import draw_landmarks
from keypoints import extract_keypoints
from model import GestureLSTM
import mediapipe as mp
import json
from collections import deque

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "models/classifier.pth"
WINDOW_SIZE = 20  # number of frames to consider for prediction

# load label map
with open("label_map.json", "r") as f:
    label_map = json.load(f)
idx_to_label = {int(k):v for k,v in label_map.items()}

mp_holistic = mp.solutions.holistic

# Load model
dummy_input = torch.zeros((1, WINDOW_SIZE, 1662))  # adjust 1662 to keypoints size
input_size = dummy_input.size(2)
num_classes = len(idx_to_label)
model = GestureLSTM(input_size=input_size, num_classes=num_classes).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# buffer for recent frames
frame_buffer = deque(maxlen=WINDOW_SIZE)

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

        # draw landmarks
        frame = draw_landmarks(frame, results)
        cv2.imshow("Gesture Recognition", frame)

        # extract keypoints
        keypoints = extract_keypoints(results)
        frame_buffer.append(keypoints)

        # predict gesture if enough frames
        if len(frame_buffer) == WINDOW_SIZE:
            seq = torch.tensor([list(frame_buffer)], dtype=torch.float32).to(DEVICE)
            with torch.no_grad():
                output = model(seq)
                pred_idx = torch.argmax(output, dim=1).item()
                gesture = idx_to_label[pred_idx]
                print(f"Detected gesture: {gesture}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
