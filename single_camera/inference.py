"""Wraps the vendored VideoPose3D TemporalModel for 2D-COCO -> 3D-H36M
inference on arbitrary (short) keypoint sequences.

The official 2D detector (Detectron2) is impractical to install on Windows,
so this project uses Ultralytics YOLOv8-pose instead - it outputs the same
COCO-17 keypoint order the pretrained checkpoint expects, though it wasn't
the exact detector the checkpoint was calibrated against. Fine for
functional testing; revisit before final published accuracy numbers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from single_camera.videopose3d_model import TemporalModel, normalize_screen_coordinates

# Matches the official checkpoint: `-arc 3,3,3,3,3` per VideoPose3D's
# INFERENCE.md, trained on Detectron-COCO keypoints, outputs H36M 17-joint
# 3D positions (see facebookresearch/VideoPose3D INFERENCE.md).
FILTER_WIDTHS = [3, 3, 3, 3, 3]
NUM_JOINTS_COCO = 17


def load_model(checkpoint_path: Path | str) -> TemporalModel:
    model = TemporalModel(
        num_joints_in=NUM_JOINTS_COCO,
        in_features=2,
        num_joints_out=17,
        filter_widths=FILTER_WIDTHS,
        causal=False,
        dropout=0.25,
        channels=1024,
        dense=False,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_pos"])
    model.eval()
    return model


def predict_3d_sequence(
    model: TemporalModel,
    keypoints_2d_px: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """keypoints_2d_px: (num_frames, 17, 2) COCO-order pixel coordinates.

    Returns (num_frames, 17, 3) H36M-order 3D positions in millimeters,
    root-relative (VideoPose3D doesn't regress global trajectory), one per
    input frame - ready to hand to skeleton_adapter.from_videopose3d(),
    which expects millimeters like the rest of the canonical format.

    Short sequences are edge-padded up to the model's receptive field
    (243 frames for the official checkpoint) the same way VideoPose3D's own
    UnchunkedGenerator does for real video edges - this works even for a
    single frame, since edge-replication just gives the temporal
    convolutions a (less informative) valid window to operate on.
    """
    keypoints_2d_px = np.asarray(keypoints_2d_px, dtype="float32")
    if keypoints_2d_px.ndim != 3 or keypoints_2d_px.shape[1:] != (NUM_JOINTS_COCO, 2):
        raise ValueError(f"expected (num_frames, 17, 2) keypoints, got {keypoints_2d_px.shape}")

    normalized = normalize_screen_coordinates(keypoints_2d_px, w=frame_width, h=frame_height)
    normalized = normalized.astype("float32")

    pad = model.receptive_field() // 2
    padded = np.pad(normalized, ((pad, pad), (0, 0), (0, 0)), mode="edge")

    with torch.no_grad():
        output = model(torch.from_numpy(padded).unsqueeze(0))

    # Empirically verified (not documented in VideoPose3D's own docs): this
    # checkpoint's raw output is in meters, not H36M's native millimeters -
    # max joint-to-hip magnitude comes out ~0.6, matching real body scale in
    # meters. Convert here so downstream code gets consistent millimeters.
    meters_to_mm = 1000.0
    return output.squeeze(0).numpy() * meters_to_mm
