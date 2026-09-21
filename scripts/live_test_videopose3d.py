"""One-shot test for the single-camera (VideoPose3D) pipeline: records a
short burst of frames from a real webcam, runs YOLOv8-pose per frame to get
2D COCO keypoints, lifts them to 3D with the pretrained VideoPose3D model,
and saves both a 2D detection overlay and the resulting canonical 17-joint
skeleton from skeleton_adapter.

Unlike live_test_mediapipe.py, this isn't a single-frame test: VideoPose3D
is a temporal model with a 243-frame receptive field (it lifts a *sequence*
of 2D poses, not one frame), so a short burst is recorded and edge-padded
up to that receptive field, matching VideoPose3D's own approach for video
edges. The 3D pose shown is for the middle frame of the recorded burst.

2D detector: Ultralytics YOLOv8-pose, not the official Detectron2 (painful
to install on Windows) - same COCO-17 keypoint order the pretrained
checkpoint expects, but not the exact detector it was calibrated against.
Fine for a functional test; revisit before final published accuracy numbers.

Usage: python scripts/live_test_videopose3d.py [camera_index] [countdown_seconds] [num_frames]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from single_camera.inference import load_model, predict_3d_sequence  # noqa: E402
from skeleton_adapter.adapters import from_videopose3d  # noqa: E402
from skeleton_adapter.skeleton import SKELETON  # noqa: E402
from skeleton_adapter.visualize import plot_skeletons  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "data"
VIDEOPOSE3D_CHECKPOINT = OUTPUT_DIR / "models" / "pretrained_h36m_detectron_coco.bin"
YOLO_POSE_WEIGHTS = OUTPUT_DIR / "models" / "yolov8n-pose.pt"


def record_burst(camera_index: int, countdown_seconds: int, num_frames: int) -> list:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera index {camera_index}")
    try:
        deadline = time.time() + countdown_seconds
        while time.time() < deadline:
            cap.read()
            remaining = max(0, int(deadline - time.time()))
            print(f"\rget in frame... {remaining + 1}s ", end="", flush=True)
            time.sleep(0.2)
        print()

        frames = []
        print(f"recording {num_frames} frames...")
        for _ in range(num_frames):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("failed to read a frame from the camera")
            frames.append(frame)
        return frames
    finally:
        cap.release()


def detect_2d_keypoints(yolo: YOLO, frames: list) -> np.ndarray:
    """Runs YOLOv8-pose on each frame. Returns (num_frames, 17, 2) pixel
    coordinates, using the highest-confidence detected person per frame
    (matching VideoPose3D's own custom-video handling), and holding the
    previous frame's keypoints if a frame has no detection at all.
    """
    keypoints = np.zeros((len(frames), 17, 2), dtype="float32")
    last_valid = None
    for i, frame in enumerate(frames):
        result = yolo.predict(frame, verbose=False)[0]
        if len(result.boxes) == 0:
            if last_valid is None:
                raise RuntimeError(f"no person in frame {i}, no earlier frame to fall back on")
            keypoints[i] = last_valid
            continue
        best = result.boxes.conf.argmax().item()
        keypoints[i] = result.keypoints.xy[best].numpy()
        last_valid = keypoints[i]
    return keypoints


def draw_2d_skeleton(frame, keypoints_px: np.ndarray):
    # Standard COCO-17 skeleton connections.
    coco_bones = [
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
    ]
    annotated = frame.copy()
    points = [(int(x), int(y)) for x, y in keypoints_px]
    for i, j in coco_bones:
        cv2.line(annotated, points[i], points[j], (0, 165, 255), 2)
    for x, y in points:
        cv2.circle(annotated, (x, y), 4, (255, 120, 0), -1)
    return annotated


def main() -> None:
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    countdown_seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    num_frames = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    for path in (VIDEOPOSE3D_CHECKPOINT, YOLO_POSE_WEIGHTS):
        if not path.exists():
            raise FileNotFoundError(f"missing model file at {path}")

    frames = record_burst(camera_index, countdown_seconds, num_frames)
    frame_h, frame_w = frames[0].shape[:2]

    yolo = YOLO(str(YOLO_POSE_WEIGHTS))
    keypoints_2d = detect_2d_keypoints(yolo, frames)

    model = load_model(VIDEOPOSE3D_CHECKPOINT)
    poses_3d = predict_3d_sequence(model, keypoints_2d, frame_w, frame_h)

    mid = len(frames) // 2
    annotated = draw_2d_skeleton(frames[mid], keypoints_2d[mid])
    annotated_path = OUTPUT_DIR / "live_test_videopose3d_annotated.png"
    cv2.imwrite(str(annotated_path), annotated)
    print(f"Saved 2D detection overlay (middle frame) to {annotated_path}")

    # VideoPose3D's output is already in H36M's mm scale (see adapters.py),
    # but it's a network prediction, not ground truth - printing bone
    # lengths is a quick, angle-independent sanity check either way.
    canonical = from_videopose3d(poses_3d[mid])
    print("\nBone lengths (mm) - sanity check that's independent of plot viewing angle:")
    for child, parent in SKELETON.bones():
        length = np.linalg.norm(canonical[child] - canonical[parent])
        parent_name, child_name = SKELETON.joint_names[parent], SKELETON.joint_names[child]
        print(f"  {parent_name:>10} -> {child_name:<10} {length:6.0f}")

    skeleton_path = OUTPUT_DIR / "live_test_videopose3d_skeleton.png"
    fig = plot_skeletons({"VideoPose3D (live)": canonical})
    fig.savefig(skeleton_path, dpi=150)
    print(f"Saved canonical skeleton plot to {skeleton_path}")


if __name__ == "__main__":
    main()
