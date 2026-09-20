"""Converts H36M, VideoPose3D, and MediaPipe pose output into one canonical
17-joint skeleton format (see configs/skeleton.yaml).

Joint order and parent hierarchy match VideoPose3D's own h36m_skeleton
definition (facebookresearch/VideoPose3D: common/h36m_dataset.py's
h36m_skeleton + Human36mDataset.remove_joints), not hand-derived.
"""
from __future__ import annotations

import numpy as np

from skeleton_adapter.mediapipe_landmarks import MP
from skeleton_adapter.skeleton import SKELETON

# Indices into H36M's full 32-joint raw skeleton that make up the canonical
# 17-joint set, in canonical order. Derived by simulating VideoPose3D's own
# Human36mDataset(remove_static_joints=True) joint removal against its
# h36m_skeleton parent list - see configs/skeleton.yaml's header comment.
H36M_32_TO_17 = [0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27]

# Canonical joint indices, for readability below.
(HIP, RHIP, RKNEE, RFOOT, LHIP, LKNEE, LFOOT, SPINE, THORAX, NECK, HEAD,
 LSHOULDER, LELBOW, LWRIST, RSHOULDER, RELBOW, RWRIST) = range(17)

_METERS_TO_MM = 1000.0


def _center_on_root(joints: np.ndarray) -> np.ndarray:
    return joints - joints[SKELETON.root_joint]


def from_h36m(joints_32: np.ndarray) -> np.ndarray:
    """joints_32: (32, 3) raw H36M mocap joints, in millimeters.

    Returns (17, 3) canonical joints, root-centered, in millimeters.
    """
    joints_32 = np.asarray(joints_32, dtype=float)
    if joints_32.shape != (32, 3):
        raise ValueError(f"expected (32, 3) H36M joints, got {joints_32.shape}")
    joints_17 = joints_32[H36M_32_TO_17]
    return _center_on_root(joints_17)


def from_videopose3d(joints_17: np.ndarray) -> np.ndarray:
    """joints_17: (17, 3) VideoPose3D output, already in the canonical H36M
    17-joint order and in millimeters.

    Returns the same joints, root-centered.
    """
    joints_17 = np.asarray(joints_17, dtype=float)
    if joints_17.shape != (17, 3):
        raise ValueError(f"expected (17, 3) VideoPose3D joints, got {joints_17.shape}")
    return _center_on_root(joints_17)


def from_mediapipe(world_landmarks: np.ndarray) -> np.ndarray:
    """world_landmarks: (33, 3) MediaPipe PoseLandmarker `world_landmarks`
    output (metric 3D, meters).

    Returns (17, 3) canonical joints, root-centered, in millimeters.

    MediaPipe has no pelvis/thorax/spine/neck markers, so those are
    synthesized: pelvis = hip midpoint, thorax = shoulder midpoint, spine =
    midpoint of pelvis/thorax, head = the nose landmark. Neck isn't covered
    by the original project plan; it's synthesized here as the midpoint of
    thorax/head, for consistency with how spine is derived - revisit if
    that turns out to bias the neck bone length noticeably.
    """
    lm = np.asarray(world_landmarks, dtype=float)
    if lm.shape != (33, 3):
        raise ValueError(f"expected (33, 3) MediaPipe world landmarks, got {lm.shape}")

    pelvis = (lm[MP.LEFT_HIP] + lm[MP.RIGHT_HIP]) / 2
    thorax = (lm[MP.LEFT_SHOULDER] + lm[MP.RIGHT_SHOULDER]) / 2
    head = lm[MP.NOSE]

    joints = np.zeros((17, 3), dtype=float)
    joints[HIP] = pelvis
    joints[RHIP] = lm[MP.RIGHT_HIP]
    joints[RKNEE] = lm[MP.RIGHT_KNEE]
    joints[RFOOT] = lm[MP.RIGHT_ANKLE]
    joints[LHIP] = lm[MP.LEFT_HIP]
    joints[LKNEE] = lm[MP.LEFT_KNEE]
    joints[LFOOT] = lm[MP.LEFT_ANKLE]
    joints[SPINE] = (pelvis + thorax) / 2
    joints[THORAX] = thorax
    joints[NECK] = (thorax + head) / 2
    joints[HEAD] = head
    joints[LSHOULDER] = lm[MP.LEFT_SHOULDER]
    joints[LELBOW] = lm[MP.LEFT_ELBOW]
    joints[LWRIST] = lm[MP.LEFT_WRIST]
    joints[RSHOULDER] = lm[MP.RIGHT_SHOULDER]
    joints[RELBOW] = lm[MP.RIGHT_ELBOW]
    joints[RWRIST] = lm[MP.RIGHT_WRIST]

    joints *= _METERS_TO_MM
    return _center_on_root(joints)


def normalize_scale(joints_17: np.ndarray) -> np.ndarray:
    """Scale-normalizes an already root-centered canonical skeleton so its
    trunk length (thorax-to-pelvis distance, a proxy for shoulder-to-hip
    distance) is 1.0.

    Opt-in - not applied inside the from_*() adapters, since some callers
    (e.g. the validation plots) want to see real bone lengths.
    """
    joints_17 = np.asarray(joints_17, dtype=float)
    trunk_length = np.linalg.norm(joints_17[THORAX] - joints_17[HIP])
    if trunk_length == 0:
        raise ValueError("cannot scale-normalize a skeleton with zero trunk length")
    return joints_17 / trunk_length
