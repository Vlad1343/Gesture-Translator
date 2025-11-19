# # src/capture.py
# import cv2
# import mediapipe as mp
# import numpy as np

# # MediaPipe solutions
# mp_holistic = mp.solutions.holistic
# mp_drawing = mp.solutions.drawing_utils

# # helper function to draw landmarks
# def draw_landmarks(frame, results):
#     # face
#     if results.face_landmarks:
#         mp_drawing.draw_landmarks(
#             frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
#             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
#             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1)
#         )
#     # hands
#     if results.left_hand_landmarks:
#         mp_drawing.draw_landmarks(
#             frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
#             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2)
#         )
#     if results.right_hand_landmarks:
#         mp_drawing.draw_landmarks(
#             frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
#             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
#             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2)
#         )
#     # pose
#     if results.pose_landmarks:
#         mp_drawing.draw_landmarks(
#             frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
#             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
#             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2)
#         )
#     return frame

# def main():
#     cap = cv2.VideoCapture(0)  # open webcam
#     with mp_holistic.Holistic(
#         static_image_mode=False,
#         model_complexity=1,
#         enable_segmentation=False,
#         refine_face_landmarks=True,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5
#     ) as holistic:

#         print("Press 'q' to quit")
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             frame = cv2.flip(frame, 1)  # mirror image
#             rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             results = holistic.process(rgb_frame)

#             # draw landmarks
#             frame = draw_landmarks(frame, results)

#             cv2.imshow("Gesture Capture", frame)

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()




import os
import time
import cv2
import numpy as np
from keypoints import extract_keypoints
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

GESTURES = ["yes", "no", "sorry", "thank_you", "hello"]
REPS = 20  # repetitions per gesture
DATA_DIR = "data/raw"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

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
            print(f"Get ready for gesture '{gesture}'. Press Enter to start.")
            input()
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

                    keypoints = extract_keypoints(results)
                    frames.append(keypoints)

                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                    if results.left_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                    if results.right_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                    cv2.imshow("Recording", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    # record ~2 seconds per repetition (adjust if needed)
                    if time.time() - start_time > 2:
                        break

                # save this repetition
                rep_path = os.path.join(gesture_dir, f"{rep}.npy")
                np.save(rep_path, np.array(frames))

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    ensure_dir(DATA_DIR)
    record_gestures()