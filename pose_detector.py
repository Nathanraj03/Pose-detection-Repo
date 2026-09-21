"""
pose_detector.py
Wraps MediaPipe's modern Tasks API (PoseLandmarker) into a reusable class that:
- detects pose landmarks in a frame
- draws the skeleton overlay
- exposes landmark coordinates for downstream analysis (angles, movement)

Note: MediaPipe's older `mp.solutions.pose` API is deprecated / not bundled
in current PyPI wheels, so this uses the current `mediapipe.tasks` API
(PoseLandmarker), which requires a .task model file downloaded once and
cached locally.
"""

import os
import urllib.request
import cv2
import numpy as np
import mediapipe as mp

from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# POSE_CONNECTIONS: pairs of landmark indices to draw as skeleton edges
# (matches the 33-point MediaPipe Pose topology)
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),   # arms + shoulders
    (11, 23), (12, 24), (23, 24),                        # torso
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),    # left leg
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),    # right leg
    (15, 17), (15, 19), (15, 21),                        # left hand
    (16, 18), (16, 20), (16, 22),                        # right hand
    (0, 1), (1, 2), (2, 3), (3, 7),                       # face left
    (0, 4), (4, 5), (5, 6), (6, 8),                       # face right
    (9, 10),                                              # mouth
]


def ensure_model_downloaded():
    """Download the PoseLandmarker model file once and cache it locally."""
    if os.path.exists(MODEL_PATH):
        return
    os.makedirs(MODEL_DIR, exist_ok=True)
    print("Downloading pose landmarker model (one-time, ~5-30 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


class PoseDetector:
    def __init__(self,
                 model_path=MODEL_PATH,
                 num_poses=1,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 running_mode=RunningMode.VIDEO):

        ensure_model_downloaded()

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=running_mode,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        self.results = None
        self._frame_index = 0

    def find_pose(self, frame, draw=True):
        """
        Run pose detection on a BGR frame (from cv2.VideoCapture).
        Returns the (possibly annotated) frame.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        timestamp_ms = int(self._frame_index * (1000 / 30))  # assume ~30fps
        self._frame_index += 1

        self.results = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        if self.results.pose_landmarks and draw:
            self._draw_landmarks(frame, self.results.pose_landmarks[0])

        return frame

    def _draw_landmarks(self, frame, landmarks):
        h, w, _ = frame.shape
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        for start_idx, end_idx in POSE_CONNECTIONS:
            if start_idx < len(points) and end_idx < len(points):
                cv2.line(frame, points[start_idx], points[end_idx], (0, 255, 0), 2)

        for x, y in points:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

    def get_landmark_positions(self, frame):
        """
        Returns a list of [id, x_px, y_px, visibility] for every detected
        landmark (first detected person), converted to pixel coordinates.
        """
        landmark_list = []
        if self.results and self.results.pose_landmarks:
            h, w, _ = frame.shape
            for idx, lm in enumerate(self.results.pose_landmarks[0]):
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([idx, cx, cy, lm.visibility])
        return landmark_list

    def get_landmark_xy(self, frame, landmark_id):
        """Convenience method: pixel (x, y) for a single landmark id, or None."""
        positions = self.get_landmark_positions(frame)
        for lm_id, x, y, vis in positions:
            if lm_id == landmark_id:
                return (x, y)
        return None

    def close(self):
        self.landmarker.close()


# Reference: key MediaPipe Pose landmark indices used across the project
LANDMARKS = {
    "NOSE": 0,
    "LEFT_SHOULDER": 11, "RIGHT_SHOULDER": 12,
    "LEFT_ELBOW": 13, "RIGHT_ELBOW": 14,
    "LEFT_WRIST": 15, "RIGHT_WRIST": 16,
    "LEFT_HIP": 23, "RIGHT_HIP": 24,
    "LEFT_KNEE": 25, "RIGHT_KNEE": 26,
    "LEFT_ANKLE": 27, "RIGHT_ANKLE": 28,
}
