"""
main.py
Real-time human pose detection system using OpenCV + MediaPipe.

Features:
- Live webcam (or video file) capture
- Skeleton overlay via MediaPipe Pose
- Joint angle calculation (e.g. elbow angle)
- Per-joint speed + simple activity classification (Idle / Moving / Fast)
- Motion trail for a tracked joint
- FPS counter
- Optional recording of the annotated output to outputs/output.mp4

Usage:
    python main.py                      # webcam
    python main.py --source video.mp4   # video file
    python main.py --record             # also save annotated video
"""

import cv2
import time
import argparse
import os

from pose_detector import PoseDetector, LANDMARKS
from utils import calculate_angle, MovementAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time Pose Detection System")
    parser.add_argument("--source", type=str, default="0",
                         help="Video source: '0' for webcam, or path to a video file")
    parser.add_argument("--record", action="store_true",
                         help="Save annotated output video to outputs/output.mp4")
    parser.add_argument("--track-joint", type=str, default="RIGHT_WRIST",
                         help="Landmark name (see pose_detector.LANDMARKS) to track for speed/trail")
    return parser.parse_args()


def main():
    args = parse_args()
    source = 0 if args.source == "0" else args.source

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open video source: {source}")
        return

    detector = PoseDetector()
    analyzer = MovementAnalyzer()
    tracked_id = LANDMARKS.get(args.track_joint.upper(), LANDMARKS["RIGHT_WRIST"])

    writer = None
    if args.record:
        os.makedirs("outputs", exist_ok=True)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter("outputs/output.mp4", fourcc, 20.0, (w, h))

    prev_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = detector.find_pose(frame, draw=True)
        landmarks = detector.get_landmark_positions(frame)

        if landmarks:
            lm_dict = {lm[0]: (lm[1], lm[2]) for lm in landmarks}

            # --- Joint angle example: right elbow angle ---
            if all(i in lm_dict for i in
                   [LANDMARKS["RIGHT_SHOULDER"], LANDMARKS["RIGHT_ELBOW"], LANDMARKS["RIGHT_WRIST"]]):
                shoulder = lm_dict[LANDMARKS["RIGHT_SHOULDER"]]
                elbow = lm_dict[LANDMARKS["RIGHT_ELBOW"]]
                wrist = lm_dict[LANDMARKS["RIGHT_WRIST"]]
                angle = calculate_angle(shoulder, elbow, wrist)

                cv2.putText(frame, f"Elbow: {int(angle)} deg", (elbow[0] - 40, elbow[1] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            # --- Movement analysis on tracked joint ---
            if tracked_id in lm_dict:
                tx, ty = lm_dict[tracked_id]
                analyzer.update(tracked_id, tx, ty)
                speed = analyzer.get_speed(tracked_id)
                activity = analyzer.classify_activity(tracked_id)

                cv2.putText(frame, f"{args.track_joint}: {activity} ({int(speed)} px/s)",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Draw motion trail
                trail = analyzer.get_trajectory(tracked_id)
                for i in range(1, len(trail)):
                    cv2.line(frame, trail[i - 1], trail[i], (0, 200, 255), 2)

        # --- FPS counter ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Pose Detection System", frame)

        if writer is not None:
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer is not None:
        writer.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
