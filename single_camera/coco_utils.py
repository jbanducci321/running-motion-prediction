"""Small shared helpers for working with COCO-17 2D keypoints (YOLOv8-pose's
output format), used by both the one-shot test and the live demo.
"""
from __future__ import annotations

import cv2
import numpy as np

# Standard COCO-17 skeleton connections.
COCO_BONES = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6),
]


def draw_2d_skeleton(frame: np.ndarray, keypoints_px: np.ndarray) -> np.ndarray:
    annotated = frame.copy()
    points = [(int(x), int(y)) for x, y in keypoints_px]
    for i, j in COCO_BONES:
        cv2.line(annotated, points[i], points[j], (0, 165, 255), 2)
    for x, y in points:
        cv2.circle(annotated, (x, y), 4, (255, 120, 0), -1)
    return annotated
