"""Kalman estimator for online Jacobian error compensation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from Software_v3.config import PaperParameters


def vec_rowwise(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=float).reshape(9, order="C")


def unvec_rowwise(vector: np.ndarray) -> np.ndarray:
    return np.asarray(vector, dtype=float).reshape((3, 3), order="C")


@dataclass
class KalmanUpdate:
    model_jacobian: np.ndarray
    delta_jacobian: np.ndarray
    estimated_jacobian: np.ndarray
    innovation: np.ndarray
    gain: np.ndarray


class JacobianErrorKalmanFilter:
    """9-state Kalman filter where the state is ``vec(delta_J)``."""

    def __init__(self, params: PaperParameters | None = None):
        self.params = params or PaperParameters()
        self.xi = np.zeros(9, dtype=float)
        self.P = np.zeros((9, 9), dtype=float)
        self.pose_prev: np.ndarray | None = None
        self.u_prev: np.ndarray | None = None

    @staticmethod
    def build_measurement_matrix(du: np.ndarray) -> np.ndarray:
        du = np.asarray(du, dtype=float).reshape(3)
        H = np.zeros((3, 9), dtype=float)
        H[0, 0:3] = du
        H[1, 3:6] = du
        H[2, 6:9] = du
        return H

    def reset(self) -> None:
        self.xi[:] = 0.0
        self.P[:, :] = 0.0
        self.pose_prev = None
        self.u_prev = None

    def update(
        self,
        u_curr: np.ndarray,
        pose_curr: np.ndarray,
        model_jacobian: np.ndarray,
    ) -> KalmanUpdate:
        """Run one predict/update cycle and return the compensated Jacobian."""

        u_curr = np.asarray(u_curr, dtype=float).reshape(3)
        pose_curr = np.asarray(pose_curr, dtype=float).reshape(3)
        model_jacobian = np.asarray(model_jacobian, dtype=float).reshape((3, 3))

        if self.u_prev is None or self.pose_prev is None:
            self.u_prev = u_curr.copy()
            self.pose_prev = pose_curr.copy()
            estimated = model_jacobian + unvec_rowwise(self.xi)
            return KalmanUpdate(
                model_jacobian=model_jacobian,
                delta_jacobian=unvec_rowwise(self.xi),
                estimated_jacobian=estimated,
                innovation=np.zeros(3),
                gain=np.zeros((9, 3)),
            )

        gamma = self.params.gamma
        du = u_curr - self.u_prev
        dpose = pose_curr - self.pose_prev
        H = self.build_measurement_matrix(du)

        xi_prior = gamma * self.xi
        P_prior = (gamma**2) * self.P + self.params.q_matrix
        R = self.params.r_matrix(du)

        predicted_dpose = H @ (vec_rowwise(model_jacobian) + xi_prior)
        innovation = dpose - predicted_dpose

        S = H @ P_prior @ H.T + R
        gain = P_prior @ H.T @ np.linalg.pinv(S)

        self.xi = xi_prior + gain @ innovation
        self.P = (np.eye(9) - gain @ H) @ P_prior
        self.u_prev = u_curr.copy()
        self.pose_prev = pose_curr.copy()

        delta = unvec_rowwise(self.xi)
        estimated = model_jacobian + delta
        return KalmanUpdate(
            model_jacobian=model_jacobian,
            delta_jacobian=delta,
            estimated_jacobian=estimated,
            innovation=innovation,
            gain=gain,
        )

