# src/camera_live.py
import cv2
import mediapipe as mp
from utils import draw_landmarks

mp_holistic = mp.solutions.holistic

def main():
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
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb_frame)

            # draw landmarks
            frame = draw_landmarks(frame, results)

            cv2.imshow("Camera Live", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    