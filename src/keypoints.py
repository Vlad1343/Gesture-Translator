# 1563 values per frame

import cv2
import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# extract keypoints from a single frame
def extract_keypoints(results):
    # pose: 33 landmarks, each (x,y,z)
    pose = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33,3))
    
    # left hand: 21 landmarks
    lh = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21,3))
    
    # right hand: 21 landmarks
    rh = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21,3))
    
    # face: 468 landmarks
    face = np.array([[lmk.x, lmk.y, lmk.z] for lmk in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468,3))
    
    # flatten all together
    keypoints = np.concatenate([pose.flatten(), lh.flatten(), rh.flatten(), face.flatten()])
    return keypoints

# draw landmarks on frame (for live visual feedback)
def draw_landmarks(frame, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.face_landmarks:
        mp_drawing.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    return frame