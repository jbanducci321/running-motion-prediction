"""Live demo: webcam feed (with the raw MediaPipe 33-point overlay) next to
a real-time VPython rig of skeleton_adapter's canonical 17-joint skeleton.

Point of this one, unlike live_test_mediapipe.py, is continuous live
verification - no captured image to inspect afterward, just watch the two
windows and see whether the standardized skeleton tracks you correctly in
real time.

Same webcam + VPython rig approach as ../../3D-Tracking-Demo/body_demo.py,
adapted to drive skeleton_adapter's canonical format instead of raw
MediaPipe landmarks.

Controls: stand in front of the webcam, press 'q' in the webcam window to quit.

Usage: python scripts/live_demo.py [camera_index]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import mediapipe as mp  # noqa: E402
import numpy as np  # noqa: E402
from mediapipe.tasks.python import (
    BaseOptions,  # noqa: E402
    vision,  # noqa: E402
)
from vpython import canvas, color, cylinder, rate, sphere, vector  # noqa: E402

from skeleton_adapter.adapters import from_mediapipe  # noqa: E402
from skeleton_adapter.skeleton import SKELETON  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "data" / "models" / "pose_landmarker_lite.task"

RAW_POSE_CONNECTIONS = [(c.start, c.end) for c in vision.PoseLandmarksConnections.POSE_LANDMARKS]
CANONICAL_BONES = SKELETON.bones()

# skeleton_adapter's canonical joints come out root-centered, in millimeters,
# in MediaPipe's raw axis convention (x=right, y=down, z=depth). Flip y/z so
# the VPython scene reads with "up" actually up, and scale mm down to a
# comfortable VPython scene size.
MM_TO_WORLD = 0.03


def build_rig():
    scene = canvas(title="Canonical 17-Joint Rig", width=800, height=600, background=color.black)
    joints = [
        sphere(canvas=scene, pos=vector(0, 0, 0), radius=0.3, color=color.orange)
        for _ in range(SKELETON.num_joints)
    ]
    bones = [
        cylinder(canvas=scene, pos=vector(0, 0, 0), axis=vector(0, 0, 0),
                 radius=0.12, color=color.white)
        for _ in CANONICAL_BONES
    ]
    return joints, bones


def canonical_to_vector(joint_mm: np.ndarray) -> vector:
    x, y, z = joint_mm
    return vector(x * MM_TO_WORLD, -y * MM_TO_WORLD, -z * MM_TO_WORLD)


def update_rig(joints, bones, canonical_joints: np.ndarray) -> None:
    positions = [canonical_to_vector(j) for j in canonical_joints]
    for joint, pos in zip(joints, positions):
        joint.pos = pos
    for bone, (child, parent) in zip(bones, CANONICAL_BONES):
        bone.pos = positions[parent]
        bone.axis = positions[child] - positions[parent]


def draw_raw_landmarks_on_frame(frame, landmarks_2d) -> None:
    h, w = frame.shape[:2]
    points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks_2d]
    for i, j in RAW_POSE_CONNECTIONS:
        cv2.line(frame, points[i], points[j], (0, 255, 0), 2)
    for x, y in points:
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)


def main() -> None:
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"missing pose landmarker model at {MODEL_PATH}")

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {camera_index}")

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    landmarker = vision.PoseLandmarker.create_from_options(options)
    joints, bones = build_rig()
    start_time = time.time()

    try:
        while True:
            rate(30)

            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((time.time() - start_time) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                draw_raw_landmarks_on_frame(frame, result.pose_landmarks[0])
                world_xyz = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_world_landmarks[0]])
                canonical = from_mediapipe(world_xyz)
                update_rig(joints, bones, canonical)

            cv2.imshow("Webcam - raw MediaPipe overlay (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
