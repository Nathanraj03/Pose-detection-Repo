# Real-Time Human Pose Detection System

A real-time human pose detection and motion-tracking system built with
**OpenCV** (video capture/rendering) and **MediaPipe** (ML pose landmark
detection). Tracks 33 body landmarks per person, computes joint angles,
measures joint speed, classifies basic movement/activity, and overlays
everything live on the video feed.

## 1. File structure

```
pose_detection_system/
├── main.py              # Entry point: captures video, runs the pipeline, displays results
├── pose_detector.py      # PoseDetector class — wraps MediaPipe's PoseLandmarker model
├── utils.py               # Angle calculation + MovementAnalyzer (speed/activity/trail)
├── requirements.txt       # Python dependencies with pinned versions
├── models/                # Auto-created; stores the downloaded .task model file
│   └── pose_landmarker_lite.task   (downloaded automatically on first run)
└── outputs/                # Auto-created if you use --record; stores output.mp4
    └── output.mp4
```

**Why these exact files:**
- `pose_detector.py` isolates all MediaPipe-specific code (model loading,
  inference, drawing) behind one clean class, so `main.py` doesn't need to
  know how detection works internally.
- `utils.py` holds pure math/logic (angle geometry, speed tracking) that has
  nothing to do with video I/O — keeps it independently testable.
- `main.py` is the orchestrator: opens the camera/video, calls the detector
  frame-by-frame, and handles display/recording/CLI args.
- `models/` and `outputs/` are separated from source code so you can
  `.gitignore` large binary files (models, videos) without touching code.

## 2. Setup

```bash
cd pose_detection_system
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The first time you run the app, `pose_detector.py` automatically downloads
the MediaPipe Pose Landmarker model (`pose_landmarker_lite.task`, a few MB)
from Google's official model server into `models/`. This requires an
internet connection once; after that it's cached locally and runs fully
offline.

## 3. How it works (pipeline explained)

1. **Capture** — `cv2.VideoCapture` grabs frames from your webcam (or a
   video file if you pass `--source`).
2. **Detect** — Each BGR frame is converted to RGB and passed to MediaPipe's
   `PoseLandmarker` (`pose_detector.py`), which returns 33 normalized
   `(x, y, z, visibility)` landmarks describing the detected person's joints
   (shoulders, elbows, wrists, hips, knees, ankles, etc.).
3. **Draw skeleton** — Landmark pixel coordinates are connected with lines
   (bones) and circles (joints) and drawn directly onto the frame.
4. **Angle analysis** — `utils.calculate_angle()` uses the law of cosines
   (via `atan2`) on three points (e.g. shoulder–elbow–wrist) to compute a
   joint angle in degrees — useful for form-checking (e.g. squat depth,
   punch extension, rep counting).
5. **Movement analysis** — `utils.MovementAnalyzer` keeps a short rolling
   history of a chosen joint's position with timestamps, computes
   px/second speed between the last two samples, and classifies it as
   `Idle`, `Moving`, or `Fast Movement` against configurable thresholds. It
   also stores a trail of recent points to draw a motion trace.
6. **Display/record** — FPS, angle, and activity text are overlaid with
   `cv2.putText`; the annotated frame is shown in a window and, if
   `--record` is passed, written to `outputs/output.mp4`.

## 4. Running it

```bash
# Webcam, live window
python main.py

# Run on a video file instead of webcam
python main.py --source path/to/video.mp4

# Also save the annotated output video
python main.py --record

# Track/analyze a different joint (see pose_detector.LANDMARKS for names)
python main.py --track-joint LEFT_KNEE
```

Press **`q`** to quit the live window.

## 5. Extending it

- **Rep counting**: track an angle (e.g. elbow) over time and count when it
  crosses two thresholds (extended → flexed → extended = 1 rep).
- **Multi-person tracking**: raise `num_poses` in `PoseDetector.__init__`.
- **Export landmark data**: call `detector.get_landmark_positions(frame)`
  each frame and append to a CSV/JSON for offline analysis in pandas.
- **Different activities**: adjust `fast_threshold` / `idle_threshold` in
  `MovementAnalyzer` to match your camera resolution and use case (sports
  drills vs. physical-therapy range-of-motion vs. security/idle detection).

## 6. Notes

- This uses MediaPipe's current **Tasks API** (`PoseLandmarker`), not the
  older `mp.solutions.pose` API, which is deprecated and no longer bundled
  in current MediaPipe PyPI releases.
- Model variants: `pose_landmarker_lite` (fastest, used here),
  `_full`, and `_heavy` (most accurate, slowest) — swap the `MODEL_URL` in
  `pose_detector.py` if you want higher accuracy at the cost of speed.
