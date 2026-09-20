"""Static side-by-side plotting of canonical 17-joint skeletons, for
visually validating skeleton_adapter output - wrong joint mappings show up
as a bone connecting to the wrong place or an impossible bone length.
"""
from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from skeleton_adapter.skeleton import SKELETON


def plot_skeletons(named_joints: dict[str, np.ndarray]) -> Figure:
    """named_joints: e.g. {"H36M": (17,3) array, "VideoPose3D": ..., "MediaPipe": ...}.

    Returns a matplotlib Figure with one 3D subplot per entry, sharing axis
    limits so bone lengths are visually comparable across sources.
    """
    all_points = np.concatenate(list(named_joints.values()), axis=0)
    center = all_points.mean(axis=0)
    radius = np.abs(all_points - center).max() * 1.1 + 1e-6

    fig = Figure(figsize=(5 * len(named_joints), 5))
    for i, (name, joints) in enumerate(named_joints.items(), start=1):
        joints = np.asarray(joints, dtype=float)
        ax = fig.add_subplot(1, len(named_joints), i, projection="3d")
        ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], c="tab:blue")
        for child, parent in SKELETON.bones():
            xs, ys, zs = zip(joints[child], joints[parent])
            ax.plot(xs, ys, zs, c="tab:orange")
        ax.set_title(name)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)
    fig.tight_layout()
    return fig
