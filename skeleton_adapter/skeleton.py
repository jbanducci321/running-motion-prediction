"""Loads the canonical skeleton definition from configs/skeleton.yaml."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "skeleton.yaml"


@dataclass(frozen=True)
class CanonicalSkeleton:
    unit: str
    joint_names: list[str]
    parents: np.ndarray
    joints_left: list[int]
    joints_right: list[int]
    root_joint: int

    @property
    def num_joints(self) -> int:
        return len(self.joint_names)

    def bones(self) -> list[tuple[int, int]]:
        """(child, parent) index pairs, excluding the root."""
        return [(i, p) for i, p in enumerate(self.parents) if p != -1]


def load_skeleton(config_path: Path | str = _DEFAULT_CONFIG_PATH) -> CanonicalSkeleton:
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    joints = sorted(raw["joints"], key=lambda j: j["id"])
    expected_ids = list(range(len(joints)))
    actual_ids = [j["id"] for j in joints]
    if actual_ids != expected_ids:
        raise ValueError(f"skeleton.yaml joint ids must be 0..N-1 with no gaps, got {actual_ids}")

    return CanonicalSkeleton(
        unit=raw["unit"],
        joint_names=[j["name"] for j in joints],
        parents=np.array([j["parent"] for j in joints], dtype=int),
        joints_left=list(raw["joints_left"]),
        joints_right=list(raw["joints_right"]),
        root_joint=raw["root_joint"],
    )


SKELETON = load_skeleton()
