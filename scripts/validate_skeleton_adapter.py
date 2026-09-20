"""Runnable sanity check for skeleton_adapter: builds the same T-pose in
each source's raw format, runs it through the adapters, and plots all three
side by side. Uses synthetic data only - no camera or dataset needed.

Usage: python scripts/validate_skeleton_adapter.py [output_path.png]
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skeleton_adapter.adapters import from_h36m, from_mediapipe, from_videopose3d  # noqa: E402
from skeleton_adapter.synthetic_poses import (
    t_pose_h36m_32,
    t_pose_mediapipe_33,
    t_pose_videopose3d_17,
)
from skeleton_adapter.visualize import plot_skeletons


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_output = repo_root / "data" / "skeleton_adapter_validation.png"
    output_path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_output

    fig = plot_skeletons(
        {
            "H36M": from_h36m(t_pose_h36m_32()),
            "VideoPose3D": from_videopose3d(t_pose_videopose3d_17()),
            "MediaPipe": from_mediapipe(t_pose_mediapipe_33()),
        }
    )
    fig.savefig(output_path, dpi=150)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
