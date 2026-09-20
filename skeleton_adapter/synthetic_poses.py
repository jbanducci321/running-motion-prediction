"""Hand-built T-pose fixtures in each source's raw format, all representing
the same pose, for validating skeleton_adapter without any real capture
hardware or dataset access.

Coordinates use an arbitrary z-up convention (+x = subject's right,
+z = up) purely for visual sanity-checking bone connectivity - they are
not meant to match H36M's real camera-relative coordinate system.
"""
from __future__ import annotations

import numpy as np

from skeleton_adapter.mediapipe_landmarks import MP, NUM_LANDMARKS

# name -> (x, y, z) in millimeters, root (pelvis) at the origin.
_T_POSE_MM = {
    "Hip": (0, 0, 0),
    "RHip": (90, 0, 0),
    "RKnee": (90, 0, -450),
    "RFoot": (90, 0, -900),
    "LHip": (-90, 0, 0),
    "LKnee": (-90, 0, -450),
    "LFoot": (-90, 0, -900),
    "Spine": (0, 0, 200),
    "Thorax": (0, 0, 400),
    "Neck/Nose": (0, 0, 450),
    "Head": (0, 0, 550),
    "RShoulder": (180, 0, 400),
    "RElbow": (450, 0, 400),
    "RWrist": (700, 0, 400),
    "LShoulder": (-180, 0, 400),
    "LElbow": (-450, 0, 400),
    "LWrist": (-700, 0, 400),
}

# Original (pre-removal) index of each named joint in H36M's full 32-joint
# raw skeleton - matches H36M_32_TO_17 in adapters.py.
_H36M_32_INDEX = {
    "Hip": 0, "RHip": 1, "RKnee": 2, "RFoot": 3,
    "LHip": 6, "LKnee": 7, "LFoot": 8,
    "Spine": 12, "Thorax": 13, "Neck/Nose": 14, "Head": 15,
    "LShoulder": 17, "LElbow": 18, "LWrist": 19,
    "RShoulder": 25, "RElbow": 26, "RWrist": 27,
}


def t_pose_h36m_32() -> np.ndarray:
    """(32, 3) raw H36M array. Only the 17 joints skeleton_adapter reads are
    populated; the rest (fingers, duplicate "site" markers, etc.) are zero
    since from_h36m() never looks at them.
    """
    joints = np.zeros((32, 3), dtype=float)
    for name, idx in _H36M_32_INDEX.items():
        joints[idx] = _T_POSE_MM[name]
    return joints


def t_pose_videopose3d_17() -> np.ndarray:
    """(17, 3) array already in canonical H36M order, as VideoPose3D itself
    would output.
    """
    from skeleton_adapter.skeleton import SKELETON

    joints = np.zeros((17, 3), dtype=float)
    for i, name in enumerate(SKELETON.joint_names):
        joints[i] = _T_POSE_MM[name]
    return joints


def t_pose_mediapipe_33() -> np.ndarray:
    """(33, 3) MediaPipe `world_landmarks`-shaped array, in meters. Only the
    12 directly-mapped joints + nose are populated; the rest (eyes, ears,
    mouth, fingers, heels, foot index) are zero since from_mediapipe() never
    reads them.
    """
    joints = np.zeros((NUM_LANDMARKS, 3), dtype=float)
    mm_to_m = 1 / 1000
    joints[MP.NOSE] = np.array(_T_POSE_MM["Head"]) * mm_to_m
    joints[MP.LEFT_SHOULDER] = np.array(_T_POSE_MM["LShoulder"]) * mm_to_m
    joints[MP.RIGHT_SHOULDER] = np.array(_T_POSE_MM["RShoulder"]) * mm_to_m
    joints[MP.LEFT_ELBOW] = np.array(_T_POSE_MM["LElbow"]) * mm_to_m
    joints[MP.RIGHT_ELBOW] = np.array(_T_POSE_MM["RElbow"]) * mm_to_m
    joints[MP.LEFT_WRIST] = np.array(_T_POSE_MM["LWrist"]) * mm_to_m
    joints[MP.RIGHT_WRIST] = np.array(_T_POSE_MM["RWrist"]) * mm_to_m
    joints[MP.LEFT_HIP] = np.array(_T_POSE_MM["LHip"]) * mm_to_m
    joints[MP.RIGHT_HIP] = np.array(_T_POSE_MM["RHip"]) * mm_to_m
    joints[MP.LEFT_KNEE] = np.array(_T_POSE_MM["LKnee"]) * mm_to_m
    joints[MP.RIGHT_KNEE] = np.array(_T_POSE_MM["RKnee"]) * mm_to_m
    joints[MP.LEFT_ANKLE] = np.array(_T_POSE_MM["LFoot"]) * mm_to_m
    joints[MP.RIGHT_ANKLE] = np.array(_T_POSE_MM["RFoot"]) * mm_to_m
    return joints
