"""Planar 3-segment PCC kinematics from Zhai et al. 2025.

All internal units are SI:
- tendon length changes in metres,
- position in metres,
- attitude in radians.
"""

from __future__ import annotations

import numpy as np

from Software_v3.config import PaperParameters


def _as_vec3(value: np.ndarray | list[float] | tuple[float, float, float], name: str) -> np.ndarray:
    out = np.asarray(value, dtype=float).reshape(3)
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain finite numbers")
    return out


def pcc_angles(
    u: np.ndarray | list[float] | tuple[float, float, float],
    offsets_m: np.ndarray | None = None,
) -> np.ndarray:
    """Solve Eqs. 7-9 for segment bending angles.

    Parameters
    ----------
    u:
        ``[Delta l1, Delta l2, Delta l3]`` in metres.
    offsets_m:
        ``[r1, r2, r3]`` tendon offsets in metres.
    """

    params = PaperParameters()
    r = _as_vec3(offsets_m if offsets_m is not None else params.tendon_offsets_m, "offsets_m")
    dl1, dl2, dl3 = _as_vec3(u, "u")
    if abs(r[0]) < 1e-12:
        raise ValueError("r1 must be nonzero for the paper actuation model")

    theta1 = dl1 / r[0]
    theta2 = (dl2 - r[1] * theta1) / r[0]
    theta3 = (dl3 - r[2] * theta1 - r[1] * theta2) / r[0]
    return np.array([theta1, theta2, theta3], dtype=float)


def _segment_translation(length_m: float, theta: float, heading: float) -> tuple[float, float]:
    """Return one PCC segment translation in the base frame."""

    if abs(theta) < 1e-8:
        return length_m * np.cos(heading), length_m * np.sin(heading)

    next_heading = heading + theta
    scale = length_m / theta
    dx = scale * (np.sin(next_heading) - np.sin(heading))
    dy = scale * (np.cos(heading) - np.cos(next_heading))
    return dx, dy


def forward_kinematics(
    u: np.ndarray | list[float] | tuple[float, float, float],
    lengths_m: np.ndarray | None = None,
    offsets_m: np.ndarray | None = None,
) -> np.ndarray:
    """Compute ``[x, y, psi]`` from paper PCC kinematics."""

    params = PaperParameters()
    lengths = _as_vec3(lengths_m if lengths_m is not None else params.segment_lengths_m, "lengths_m")
    theta = pcc_angles(u, offsets_m)

    x = 0.0
    y = 0.0
    heading = 0.0
    for length_m, bend in zip(lengths, theta):
        dx, dy = _segment_translation(float(length_m), float(bend), heading)
        x += dx
        y += dy
        heading += float(bend)
    return np.array([x, y, heading], dtype=float)


def model_jacobian(
    u: np.ndarray | list[float] | tuple[float, float, float],
    lengths_m: np.ndarray | None = None,
    offsets_m: np.ndarray | None = None,
    step_m: float = 1e-6,
) -> np.ndarray:
    """Numerically compute ``d[x, y, psi] / d[Delta l1, Delta l2, Delta l3]``.

    Central differences are used because they are simple, stable around the
    zero-curvature home pose, and easy to validate against finite hardware data.
    """

    u0 = _as_vec3(u, "u")
    jac = np.zeros((3, 3), dtype=float)
    for col in range(3):
        du = np.zeros(3, dtype=float)
        du[col] = step_m
        plus = forward_kinematics(u0 + du, lengths_m, offsets_m)
        minus = forward_kinematics(u0 - du, lengths_m, offsets_m)
        jac[:, col] = (plus - minus) / (2.0 * step_m)
    return jac


def u_to_antagonistic_motors(u: np.ndarray, unit: str = "mm") -> np.ndarray:
    """Map 3 paper inputs to 6 antagonistic motor displacements.

    For each segment pair, motor A pulls by ``+Delta l_i`` and motor B releases
    by ``-Delta l_i``. Returned values are millimetres by default for GUI/hardware
    compatibility.
    """

    scale = 1000.0 if unit == "mm" else 1.0
    u = _as_vec3(u, "u") * scale
    return np.array([u[0], -u[0], u[1], -u[1], u[2], -u[2]], dtype=float)


def antagonistic_motors_to_u(displacements: np.ndarray, unit: str = "mm") -> np.ndarray:
    """Convert 6 antagonistic motor displacements to the 3 paper inputs."""

    values = np.asarray(displacements, dtype=float).reshape(6)
    scale = 0.001 if unit == "mm" else 1.0
    return 0.5 * np.array(
        [values[0] - values[1], values[2] - values[3], values[4] - values[5]],
        dtype=float,
    ) * scale

