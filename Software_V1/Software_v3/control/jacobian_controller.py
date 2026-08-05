"""Jacobian-based controller with paper-style online compensation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Software_v3.config import PaperParameters
from Software_v3.estimation import JacobianErrorKalmanFilter
from Software_v3.model import model_jacobian


@dataclass
class ControlStep:
    u_next: np.ndarray
    pose_error: np.ndarray
    model_jacobian: np.ndarray
    estimated_jacobian: np.ndarray
    constrained_jacobian: np.ndarray
    delta_jacobian: np.ndarray
    innovation: np.ndarray


class JacobianController:
    """Closed-loop controller described by Eqs. 19-22 in the paper."""

    def __init__(self, params: PaperParameters | None = None, use_kalman: bool = True):
        self.params = params or PaperParameters()
        self.use_kalman = use_kalman
        self.estimator = JacobianErrorKalmanFilter(self.params)
        self.constrained_jacobian = model_jacobian(
            np.zeros(3),
            self.params.segment_lengths_m,
            self.params.tendon_offsets_m,
        )

    def reset(self) -> None:
        self.estimator.reset()
        self.constrained_jacobian = model_jacobian(
            np.zeros(3),
            self.params.segment_lengths_m,
            self.params.tendon_offsets_m,
        )

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def _apply_constraints(self, estimated_jacobian: np.ndarray) -> np.ndarray:
        prev = self.constrained_jacobian
        out = prev.copy()

        jp_est = estimated_jacobian[:2, :]
        jp_prev = prev[:2, :]
        diff_p = float(np.linalg.norm(jp_est - jp_prev, ord="fro"))
        if diff_p <= self.params.sigma_p or diff_p < 1e-12:
            out[:2, :] = jp_est
        else:
            out[:2, :] = jp_prev + self.params.sigma_p * (jp_est - jp_prev) / diff_p

        jpsi_est = estimated_jacobian[2:3, :]
        jpsi_prev = prev[2:3, :]
        diff_psi = float(np.linalg.norm(jpsi_est - jpsi_prev, ord="fro"))
        if diff_psi <= self.params.sigma_psi or diff_psi < 1e-12:
            out[2:3, :] = jpsi_est
        else:
            out[2:3, :] = jpsi_prev + self.params.sigma_psi * (jpsi_est - jpsi_prev) / diff_psi

        self.constrained_jacobian = out
        return out

    def _damped_pseudoinverse(self, jacobian: np.ndarray) -> np.ndarray:
        return np.linalg.solve(
            jacobian.T @ jacobian + self.params.alpha * np.eye(3),
            jacobian.T,
        )

    def _limit_command(self, u_curr: np.ndarray, u_candidate: np.ndarray) -> np.ndarray:
        """Apply safety/rate limits that keep the quasi-static assumption valid."""

        delta = np.asarray(u_candidate, dtype=float).reshape(3) - u_curr
        step_norm = float(np.linalg.norm(delta))
        if step_norm > self.params.max_u_step_m > 0.0:
            delta = delta * (self.params.max_u_step_m / step_norm)

        limited = u_curr + delta
        if self.params.max_abs_u_m > 0.0:
            limited = np.clip(limited, -self.params.max_abs_u_m, self.params.max_abs_u_m)
        return limited

    def step(self, u_curr: np.ndarray, pose_ref: np.ndarray, pose_measured: np.ndarray) -> ControlStep:
        u_curr = np.asarray(u_curr, dtype=float).reshape(3)
        pose_ref = np.asarray(pose_ref, dtype=float).reshape(3)
        pose_measured = np.asarray(pose_measured, dtype=float).reshape(3)

        mJ = model_jacobian(
            u_curr,
            self.params.segment_lengths_m,
            self.params.tendon_offsets_m,
        )

        if self.use_kalman:
            update = self.estimator.update(u_curr, pose_measured, mJ)
            eJ = update.estimated_jacobian
            delta_J = update.delta_jacobian
            innovation = update.innovation
        else:
            eJ = mJ
            delta_J = np.zeros((3, 3), dtype=float)
            innovation = np.zeros(3, dtype=float)

        cJ = self._apply_constraints(eJ)
        error = pose_ref - pose_measured
        error[2] = self._wrap_angle(float(error[2]))

        u_candidate = u_curr + self.params.beta * (self._damped_pseudoinverse(cJ) @ error)
        u_next = self._limit_command(u_curr, u_candidate)

        return ControlStep(
            u_next=u_next,
            pose_error=error,
            model_jacobian=mJ,
            estimated_jacobian=eJ,
            constrained_jacobian=cJ,
            delta_jacobian=delta_J,
            innovation=innovation,
        )
