import numpy as np
import mediapipe as mp

mp_holistic = mp.solutions.holistic

# number of values per frame (pose + both hands, no face mesh)
KEYPOINT_VECTOR_LENGTH = (33 + 21 + 21) * 3

# Mediapipe pose landmarks that correspond to facial features. We zero these out so
# the model never sees face-related information, keeping the face mesh purely visual.
FACE_POSE_LANDMARKS = [
    mp_holistic.PoseLandmark.NOSE.value,
    mp_holistic.PoseLandmark.LEFT_EYE_INNER.value,
    mp_holistic.PoseLandmark.LEFT_EYE.value,
    mp_holistic.PoseLandmark.LEFT_EYE_OUTER.value,
    mp_holistic.PoseLandmark.RIGHT_EYE_INNER.value,
    mp_holistic.PoseLandmark.RIGHT_EYE.value,
    mp_holistic.PoseLandmark.RIGHT_EYE_OUTER.value,
    mp_holistic.PoseLandmark.LEFT_EAR.value,
    mp_holistic.PoseLandmark.RIGHT_EAR.value,
    mp_holistic.PoseLandmark.MOUTH_LEFT.value,
    mp_holistic.PoseLandmark.MOUTH_RIGHT.value,
]


def _landmark_array(landmarks, expected):
    if not landmarks:
        return np.zeros((expected, 3), dtype=np.float32)
    coords = np.array([[lmk.x, lmk.y, lmk.z] for lmk in landmarks.landmark], dtype=np.float32)
    return coords


def _body_center_and_scale(pose_landmarks):
    if pose_landmarks is None:
        return np.zeros(3, dtype=np.float32), 1.0

    pose = np.array([[lmk.x, lmk.y, lmk.z] for lmk in pose_landmarks.landmark], dtype=np.float32)
    try:
        l_hip = pose[mp_holistic.PoseLandmark.LEFT_HIP.value]
        r_hip = pose[mp_holistic.PoseLandmark.RIGHT_HIP.value]
        l_shoulder = pose[mp_holistic.PoseLandmark.LEFT_SHOULDER.value]
        r_shoulder = pose[mp_holistic.PoseLandmark.RIGHT_SHOULDER.value]
    except IndexError:
        return np.zeros(3, dtype=np.float32), 1.0

    center = (l_hip + r_hip) / 2.0
    scale = np.linalg.norm(l_shoulder - r_shoulder)
    if not np.isfinite(scale) or scale < 1e-3:
        scale = 1.0
    return center, scale


def _normalize(coords, center, scale):
    if coords.size == 0:
        return coords
    return (coords - center) / scale


def extract_keypoints(results):
    pose_center, body_scale = _body_center_and_scale(results.pose_landmarks)

    pose = _landmark_array(results.pose_landmarks, 33)
    left_hand = _landmark_array(results.left_hand_landmarks, 21)
    right_hand = _landmark_array(results.right_hand_landmarks, 21)

    pose = _normalize(pose, pose_center, body_scale)
    left_hand = _normalize(left_hand, pose_center, body_scale)
    right_hand = _normalize(right_hand, pose_center, body_scale)

    # Discard face tracking points so only hands/body drive predictions.
    if pose.size:
        pose[FACE_POSE_LANDMARKS, :] = 0.0

    keypoints = np.concatenate([pose.flatten(), left_hand.flatten(), right_hand.flatten()])
    return keypoints.astype(np.float32)
