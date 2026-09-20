"""One-shot live test: captures a single frame from a real webcam, runs
MediaPipe PoseLandmarker on it, and saves both the raw 2D detection overlay
and the resulting canonical 17-joint skeleton from skeleton_adapter.

Not the real-time capture loop planned for a later step - just a quick
sanity check that MediaPipe can track a real person on this hardware, and
that from_mediapipe() produces a sane-looking skeleton from real (not
synthetic) data.

Usage: python scripts/live_test_mediapipe.py [camera_index] [countdown_seconds]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from mediapipe import Image, ImageFormat  # noqa: E402
from mediapipe.tasks.python import (
    BaseOptions,  # noqa: E402
    vision,  # noqa: E402
)

from skeleton_adapter.adapters import from_mediapipe  # noqa: E402
from skeleton_adapter.skeleton import SKELETON  # noqa: E402
from skeleton_adapter.visualize import plot_skeletons  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO_ROOT / "data" / "models" / "pose_landmarker_lite.task"
OUTPUT_DIR = REPO_ROOT / "data"


def capture_frame(camera_index: int, countdown_seconds: int) -> np.ndarray:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {camera_index}")
    try:
        deadline = time.time() + countdown_seconds
        while time.time() < deadline:
            cap.read()  # keep the buffer fresh / let auto-exposure settle
            remaining = max(0, int(deadline - time.time()))
            print(f"\rget in frame... {remaining + 1}s ", end="", flush=True)
            time.sleep(0.2)
        print()
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("failed to read a frame from the camera")
        return frame
    finally:
        cap.release()


def main() -> None:
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    countdown_seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"missing pose landmarker model at {MODEL_PATH}")

    frame_bgr = capture_frame(camera_index, countdown_seconds)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
    )
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        result = landmarker.detect(mp_image)

    if not result.pose_landmarks:
        raw_frame_path = OUTPUT_DIR / "live_test_raw_frame.png"
        cv2.imwrite(str(raw_frame_path), frame_bgr)
        print("No person detected in the captured frame.")
        print(f"Saved the raw frame to {raw_frame_path} for a look.")
        return

    landmarks_2d = result.pose_landmarks[0]
    world_landmarks = result.pose_world_landmarks[0]

    height, width = frame_bgr.shape[:2]
    annotated = frame_bgr.copy()
    points_px = [(int(lm.x * width), int(lm.y * height)) for lm in landmarks_2d]
    for conn in vision.PoseLandmarksConnections.POSE_LANDMARKS:
        cv2.line(annotated, points_px[conn.start], points_px[conn.end], (0, 165, 255), 2)
    for x, y in points_px:
        cv2.circle(annotated, (x, y), 4, (255, 120, 0), -1)
    annotated_path = OUTPUT_DIR / "live_test_annotated.png"
    cv2.imwrite(str(annotated_path), annotated)
    print(f"Saved 2D detection overlay to {annotated_path}")

    world_xyz = np.array([[lm.x, lm.y, lm.z] for lm in world_landmarks])
    canonical = from_mediapipe(world_xyz)

    print("\nBone lengths (mm) - sanity check that's independent of plot viewing angle:")
    for child, parent in SKELETON.bones():
        length = np.linalg.norm(canonical[child] - canonical[parent])
        parent_name, child_name = SKELETON.joint_names[parent], SKELETON.joint_names[child]
        print(f"  {parent_name:>10} -> {child_name:<10} {length:6.0f}")

    skeleton_path = OUTPUT_DIR / "live_test_skeleton.png"
    fig = plot_skeletons({"MediaPipe (live)": canonical})
    fig.savefig(skeleton_path, dpi=150)
    print(f"Saved canonical skeleton plot to {skeleton_path}")


if __name__ == "__main__":
    main()
