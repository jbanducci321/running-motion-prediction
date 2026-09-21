"""Live demo for the single-camera pipeline: webcam feed (raw YOLOv8-pose 2D
overlay) next to a real-time VPython rig of skeleton_adapter's canonical
17-joint skeleton, driven by VideoPose3D.

Unlike live_demo.py (MediaPipe), this one has a real, visible lag: VideoPose3D
is a temporal model with a 243-frame receptive field, so it needs a rolling
window of past AND future frames around whatever it's predicting - there's
no way to get its most-accurate output for the instant "now". This script
keeps a rolling buffer of the last ~243 frames of 2D keypoints and always
predicts the pose for the *middle* of that buffer, so the 3D rig always
shows you from a few seconds ago, not live. The webcam window's 2D overlay
is genuinely live (that part has no such constraint); only the 3D rig
lags. The lag shrinks as the buffer fills at startup, then stays roughly
constant (about half the buffer's time span).

Controls: stand in front of the webcam, press 'q' in the webcam window to quit.

Usage: python scripts/live_demo_videopose3d.py [camera_index]
"""
from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402
from vpython import canvas, color, cylinder, rate, sphere, vector  # noqa: E402

from single_camera.coco_utils import draw_2d_skeleton  # noqa: E402
from single_camera.inference import load_model, predict_3d_sequence  # noqa: E402
from skeleton_adapter.adapters import from_videopose3d  # noqa: E402
from skeleton_adapter.skeleton import SKELETON  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
VIDEOPOSE3D_CHECKPOINT = REPO_ROOT / "data" / "models" / "pretrained_h36m_detectron_coco.bin"
YOLO_POSE_WEIGHTS = REPO_ROOT / "data" / "models" / "yolov8n-pose.pt"

# Re-running VideoPose3D's temporal model over the whole padded buffer every
# single frame is the actual bottleneck (not YOLO) - throttling how often
# the rig updates keeps the webcam window responsive without changing what
# ends up in the buffer (still one real detection per frame either way).
INFER_EVERY_N_FRAMES = 6

CANONICAL_BONES = SKELETON.bones()

# skeleton_adapter's canonical joints come out root-centered, in millimeters,
# in image-plane-ish axes (x=right, y=down, z=depth). Flip y/z so the VPython
# scene reads with "up" actually up, matching live_demo.py's convention.
MM_TO_WORLD = 0.03


def build_rig():
    scene = canvas(title="VideoPose3D Rig (delayed)", width=800, height=600, background=color.black)
    joints = [
        sphere(canvas=scene, pos=vector(0, 0, 0), radius=0.3, color=color.cyan)
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


def main() -> None:
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    for path in (VIDEOPOSE3D_CHECKPOINT, YOLO_POSE_WEIGHTS):
        if not path.exists():
            raise FileNotFoundError(f"missing model file at {path}")

    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {camera_index}")

    yolo = YOLO(str(YOLO_POSE_WEIGHTS))
    model = load_model(VIDEOPOSE3D_CHECKPOINT)
    receptive_field = model.receptive_field()
    joints, bones = build_rig()

    keypoints_buffer: deque = deque(maxlen=receptive_field)
    last_valid_keypoints = None
    frame_count = 0
    fps_timer = time.time()

    try:
        while True:
            rate(30)

            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            result = yolo.predict(frame, verbose=False)[0]
            if len(result.boxes) > 0:
                best = result.boxes.conf.argmax().item()
                last_valid_keypoints = result.keypoints.xy[best].numpy()
                frame = draw_2d_skeleton(frame, last_valid_keypoints)

            if last_valid_keypoints is not None:
                keypoints_buffer.append(last_valid_keypoints)

                if frame_count % INFER_EVERY_N_FRAMES == 0:
                    buffered = np.stack(keypoints_buffer)
                    poses_3d = predict_3d_sequence(model, buffered, frame_w, frame_h)
                    center = len(poses_3d) // 2
                    canonical = from_videopose3d(poses_3d[center])
                    update_rig(joints, bones, canonical)

            frame_count += 1
            if frame_count % 30 == 0:
                fps = 30 / (time.time() - fps_timer)
                fps_timer = time.time()
                lag_seconds = (len(keypoints_buffer) / 2) / max(fps, 1e-6)
                print(f"\r~{fps:.1f} fps, buffer {len(keypoints_buffer)}/{receptive_field}, "
                      f"~{lag_seconds:.1f}s lag on the rig  ", end="", flush=True)

            cv2.imshow("Webcam - live YOLOv8-pose overlay (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        print()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
