"""Reference trajectories from the paper."""

from __future__ import annotations

import numpy as np


def trajectory(name: str, period_s: float = 40.0):
    key = name.strip().lower().replace(" ", "")

    def traj1(t: float) -> np.ndarray:
        w = 2.0 * np.pi / period_s
        return np.array([0.240 + 0.020 * np.cos(w * t), 0.060 * np.sin(w * t), 0.0])

    def traj2(t: float) -> np.ndarray:
        w = 2.0 * np.pi / period_s
        return np.array([0.240, 0.0, np.deg2rad(30.0) * np.sin(w * t)])

    def traj3(t: float) -> np.ndarray:
        w = 2.0 * np.pi / period_s
        return np.array(
            [
                0.240 + 0.020 * np.cos(w * t),
                0.060 * np.sin(w * t),
                np.deg2rad(30.0) * np.sin(w * t),
            ]
        )

    options = {
        "traj1": traj1,
        "trajectory1": traj1,
        "position": traj1,
        "traj2": traj2,
        "trajectory2": traj2,
        "attitude": traj2,
        "traj3": traj3,
        "trajectory3": traj3,
        "hybrid": traj3,
    }
    if key not in options:
        raise ValueError(f"Unknown trajectory {name!r}")
    return options[key]

