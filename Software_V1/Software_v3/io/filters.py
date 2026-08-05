"""Filtering helpers for pose and actuation signals."""

from __future__ import annotations

import numpy as np


class LowPassFilterBank:
    """Multi-channel low-pass filter with SciPy Butterworth when available.

    If SciPy is unavailable, this falls back to four cascaded first-order
    exponential filters. The fallback keeps the same interface and is adequate
    for simulation and early integration tests.
    """

    def __init__(self, cutoff_hz: float, sample_hz: float, channels: int, order: int = 4):
        self.cutoff_hz = float(cutoff_hz)
        self.sample_hz = float(sample_hz)
        self.channels = int(channels)
        self.order = int(order)
        self._mode = "ema"
        self._state = np.zeros((self.order, self.channels), dtype=float)
        self._initialized = False

        try:
            from scipy.signal import butter, lfilter_zi  # type: ignore

            wn = self.cutoff_hz / (0.5 * self.sample_hz)
            self._b, self._a = butter(self.order, wn, btype="low")
            zi = lfilter_zi(self._b, self._a)
            self._zi = np.tile(zi[:, None], (1, self.channels))
            self._mode = "scipy"
        except Exception:
            dt = 1.0 / self.sample_hz
            tau = 1.0 / (2.0 * np.pi * self.cutoff_hz)
            self._ema_alpha = dt / (tau + dt)

    def reset(self, value: np.ndarray | None = None) -> None:
        base = np.zeros(self.channels, dtype=float) if value is None else np.asarray(value, dtype=float).reshape(self.channels)
        if self._mode == "scipy":
            self._zi = self._zi * 0.0 + base
        else:
            self._state[:, :] = base
        self._initialized = value is not None

    def update(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=float).reshape(self.channels)
        if not self._initialized:
            self.reset(value)
            return value.copy()

        if self._mode == "scipy":
            from scipy.signal import lfilter  # type: ignore

            y, self._zi = lfilter(self._b, self._a, value.reshape(1, -1), axis=0, zi=self._zi)
            return y[-1].copy()

        current = value
        for i in range(self.order):
            self._state[i] += self._ema_alpha * (current - self._state[i])
            current = self._state[i]
        return current.copy()

