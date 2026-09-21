"""
utils.py
Helper functions for joint-angle calculation and simple movement/activity
analysis used by the pose detection system.
"""

import numpy as np
import math
import time
from collections import deque


def calculate_angle(a, b, c):
    """
    Calculate the angle (in degrees) at point b, formed by points a-b-c.
    Each point is (x, y) in pixel or normalized coordinates.
    Used for joint angles, e.g. elbow angle = shoulder, elbow, wrist.
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def calculate_distance(a, b):
    """Euclidean distance between two (x, y) points."""
    a = np.array(a)
    b = np.array(b)
    return np.linalg.norm(a - b)


class MovementAnalyzer:
    """
    Tracks landmark positions over time to compute simple motion metrics:
    - speed of key joints (pixels/second)
    - a rolling history for smoothing / trajectory drawing
    - basic activity classification (idle / moving / fast movement)
    """

    def __init__(self, history_len=30, fast_threshold=800, idle_threshold=50):
        self.history_len = history_len
        self.fast_threshold = fast_threshold   # px/sec considered "fast"
        self.idle_threshold = idle_threshold   # px/sec considered "idle"
        self.landmark_history = {}             # id -> deque of (x, y, t)

    def update(self, landmark_id, x, y):
        """Add a new position sample for a given landmark id."""
        if landmark_id not in self.landmark_history:
            self.landmark_history[landmark_id] = deque(maxlen=self.history_len)
        self.landmark_history[landmark_id].append((x, y, time.time()))

    def get_speed(self, landmark_id):
        """Return current speed (px/sec) for a landmark, based on last 2 samples."""
        hist = self.landmark_history.get(landmark_id)
        if not hist or len(hist) < 2:
            return 0.0

        (x1, y1, t1), (x2, y2, t2) = hist[-2], hist[-1]
        dt = t2 - t1
        if dt <= 0:
            return 0.0

        dist = math.hypot(x2 - x1, y2 - y1)
        return dist / dt

    def get_trajectory(self, landmark_id):
        """Return list of (x, y) points for drawing a motion trail."""
        hist = self.landmark_history.get(landmark_id)
        if not hist:
            return []
        return [(p[0], p[1]) for p in hist]

    def classify_activity(self, landmark_id):
        """Return a simple label: 'idle', 'moving', or 'fast movement'."""
        speed = self.get_speed(landmark_id)
        if speed < self.idle_threshold:
            return "Idle"
        elif speed < self.fast_threshold:
            return "Moving"
        else:
            return "Fast Movement"
