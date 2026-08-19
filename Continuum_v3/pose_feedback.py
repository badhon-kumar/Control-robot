"""
Real end-effector pose feedback for Continuum_v3.

The controller expects pose as [x_m, y_m, psi_rad]. This module receives UDP
JSON packets from a camera/tracker process, converts units, applies the same
kind of low-pass filtering used in the paper, and exposes the latest valid pose.

Expected UDP JSON examples:
    {"x_mm": 240.0, "y_mm": 10.0, "psi_deg": 5.0}
    {"x_m": 0.240, "y_m": 0.010, "psi_rad": 0.0873}

Optional fields:
    "confidence": 0.0..1.0
    "timestamp": tracker-side timestamp
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class PoseSample:
    pose: np.ndarray
    received_at: float
    confidence: float = 1.0
    raw: str = ""

    @property
    def age_s(self) -> float:
        return time.time() - self.received_at


class ButterworthLowPass4:
    """Fourth-order low-pass filter, with a dependency-free fallback."""

    def __init__(self, cutoff_hz: float = 2.0, sample_hz: float = 30.0, channels: int = 3):
        self.cutoff_hz = float(cutoff_hz)
        self.sample_hz = float(sample_hz)
        self.channels = int(channels)
        self._ready = False
        self._mode = "ema"
        self._state = np.zeros((4, self.channels), dtype=float)

        try:
            from scipy.signal import butter, lfilter_zi  # type: ignore

            wn = self.cutoff_hz / (0.5 * self.sample_hz)
            self._b, self._a = butter(4, wn, btype="low")
            zi = lfilter_zi(self._b, self._a)
            self._zi = np.tile(zi[:, None], (1, self.channels))
            self._mode = "scipy"
        except Exception:
            dt = 1.0 / self.sample_hz
            tau = 1.0 / (2.0 * math.pi * self.cutoff_hz)
            self._alpha = dt / (tau + dt)

    def reset(self, value: np.ndarray) -> None:
        value = np.asarray(value, dtype=float).reshape(self.channels)
        if self._mode == "scipy":
            self._zi = self._zi * 0.0 + value
        else:
            self._state[:, :] = value
        self._ready = True

    def update(self, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=float).reshape(self.channels)
        if not self._ready:
            self.reset(value)
            return value.copy()

        if self._mode == "scipy":
            from scipy.signal import lfilter  # type: ignore

            y, self._zi = lfilter(self._b, self._a, value.reshape(1, -1), axis=0, zi=self._zi)
            return y[-1].copy()

        current = value
        for i in range(4):
            self._state[i] += self._alpha * (current - self._state[i])
            current = self._state[i]
        return current.copy()


class UdpPoseFeedbackReceiver:
    """Background UDP receiver for end-effector pose measurements."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5005,
        max_age_s: float = 0.25,
        min_confidence: float = 0.0,
        filter_cutoff_hz: float = 2.0,
        filter_sample_hz: float = 30.0,
    ):
        self.host = host
        self.port = int(port)
        self.max_age_s = float(max_age_s)
        self.min_confidence = float(min_confidence)
        self._filter = ButterworthLowPass4(filter_cutoff_hz, filter_sample_hz, channels=3)
        self._lock = threading.Lock()
        self._sample: Optional[PoseSample] = None
        self._last_error = ""
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="UdpPoseFeedbackReceiver", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._sock:
            try:
                self._sock.close()
            except OSError:
                pass
        if self._thread:
            self._thread.join(timeout=0.5)
        self._thread = None

    def latest(self) -> Optional[PoseSample]:
        with self._lock:
            if self._sample is None:
                return None
            if self._sample.age_s > self.max_age_s:
                return None
            if self._sample.confidence < self.min_confidence:
                return None
            return PoseSample(
                pose=self._sample.pose.copy(),
                received_at=self._sample.received_at,
                confidence=self._sample.confidence,
                raw=self._sample.raw,
            )

    def status(self) -> Tuple[str, str]:
        sample = self.latest()
        if sample is not None:
            return "ok", f"pose age {sample.age_s * 1000:.0f} ms, conf {sample.confidence:.2f}"
        with self._lock:
            if self._last_error:
                return "warn", self._last_error
            if self._sample is None:
                return "warn", "waiting for UDP pose"
            return "warn", f"stale pose age {self._sample.age_s * 1000:.0f} ms"

    def _run(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.1)
        try:
            sock.bind((self.host, self.port))
        except OSError as exc:
            with self._lock:
                self._last_error = f"UDP bind failed on {self.host}:{self.port}: {exc}"
            try:
                sock.close()
            except OSError:
                pass
            return
        self._sock = sock

        while not self._stop.is_set():
            try:
                data, _addr = sock.recvfrom(4096)
            except socket.timeout:
                continue
            except OSError:
                break

            raw = data.decode("utf-8", errors="replace").strip()
            try:
                pose, confidence = self._parse_payload(raw)
                pose_f = self._filter.update(pose)
                sample = PoseSample(pose=pose_f, received_at=time.time(), confidence=confidence, raw=raw)
                with self._lock:
                    self._sample = sample
                    self._last_error = ""
            except Exception as exc:
                with self._lock:
                    self._last_error = f"bad pose packet: {exc}"

    @staticmethod
    def _parse_payload(raw: str) -> Tuple[np.ndarray, float]:
        obj = json.loads(raw)
        if "x_m" in obj:
            x = float(obj["x_m"])
        else:
            x = float(obj["x_mm"]) / 1000.0

        if "y_m" in obj:
            y = float(obj["y_m"])
        else:
            y = float(obj["y_mm"]) / 1000.0

        if "psi_rad" in obj:
            psi = float(obj["psi_rad"])
        else:
            psi = math.radians(float(obj.get("psi_deg", 0.0)))

        confidence = float(obj.get("confidence", 1.0))
        pose = np.array([x, y, psi], dtype=float)
        if not np.all(np.isfinite(pose)):
            raise ValueError("pose values must be finite")
        return pose, confidence
