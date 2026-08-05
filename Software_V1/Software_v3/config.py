"""Configuration objects for the Zhai-style continuum controller."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class PaperParameters:
    """Default parameters from Zhai et al. 2025, expressed in SI units."""

    segment_lengths_m: np.ndarray = field(
        default_factory=lambda: np.array([0.09, 0.09, 0.09], dtype=float)
    )
    tendon_offsets_m: np.ndarray = field(
        default_factory=lambda: np.array([0.005, 0.0035, 0.002], dtype=float)
    )
    alpha: float = 1.0
    beta: float = 0.5
    gamma: float = 0.5
    sigma_p: float = 0.35
    sigma_psi: float = 2.5
    q_scale: float = 1.0
    r_scale: float = 0.5
    control_hz: float = 20.0
    feedback_hz: float = 30.0
    filter_cutoff_hz: float = 2.0
    max_u_step_m: float = 0.001
    max_abs_u_m: float = 0.06

    @property
    def q_matrix(self) -> np.ndarray:
        return self.q_scale * np.eye(9)

    def r_matrix(self, du: np.ndarray, min_variance: float = 1e-9) -> np.ndarray:
        variance = self.r_scale * float(np.linalg.norm(du) ** 2)
        return max(variance, min_variance) * np.eye(3)
