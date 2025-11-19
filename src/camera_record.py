# src/camera_record.py
import os
import time
import cv2
import numpy as np
from keypoints import extract_keypoints
from utils import draw_landmarks
import mediapipe as mp

mp_holistic = mp.solutions.holistic

# GESTURES = ["yes", "no", "sorry", "thank_you", "hello"]
# REPS = 20
GESTURES = ["yes"]
REPS = 1
DATA_DIR = "data/raw"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# show countdown on frame
def countdown(frame, text="Get ready", seconds=3):
    for i in range(seconds, 0, -1):
        display = frame.copy()
        cv2.putText(display, f"{text}: {i}", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 3)
        cv2.imshow("Recording Gesture", display)
        cv2.waitKey(1000)

def record_gestures():
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as holistic:

        for gesture in GESTURES:
            gesture_dir = os.path.join(DATA_DIR, gesture)
            ensure_dir(gesture_dir)

            # show countdown before starting
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                countdown(frame, text=f"Prepare '{gesture}'", seconds=3)

            for rep in range(REPS):
                print(f"Recording {gesture}, repetition {rep+1}/{REPS}...")
                frames = []
                start_time = time.time()

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(rgb)

                    # extract keypoints
                    keypoints = extract_keypoints(results)
                    frames.append(keypoints)

                    # draw landmarks for visual feedback
                    frame = draw_landmarks(frame, results)
                    cv2.imshow("Recording Gesture", frame)

                    # stop after ~2 seconds per repetition
                    if time.time() - start_time > 2:
                        break

                    # allow quit with 'q'
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                        return

                # save repetition
                rep_path = os.path.join(gesture_dir, f"{rep}.npy")
                np.save(rep_path, np.array(frames))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ensure_dir(DATA_DIR)
    record_gestures()