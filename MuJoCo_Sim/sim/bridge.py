"""
Controller <-> plant bridge.

Two ways to connect the existing Kalman/PCC controller to the MuJoCo plant:

  closed_loop()   in-process. The experiment script owns both objects and calls
                  them directly. Fast, deterministic, and what Phases 5 and 6
                  use for generating results.

  PoseUdpPublisher  out-of-process. Publishes the plant's tip pose as the JSON
                  packets pose_feedback.py already parses, so the existing GUI
                  can be driven by MuJoCo without modifying a single line of
                  Continuum_v3. Used for demos.

Nothing in Continuum_v3 is modified by either path.

Author: Badhon Kumar
"""

import json
import os
import socket
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "controller"))

from continuum_ellipse import KalmanJacobianController


def make_controller(use_kalman=True, gamma=0.5, beta=0.5, alpha=1.0,
                    sigma_p=0.35, sigma_psi=2.5):
    """
    The paper's controller, unmodified, with its published defaults.

    Note these gains were tuned against a PCC plant. Against MuJoCo they may
    need retuning - Phase 5 checks this rather than assuming it.
    """
    return KalmanJacobianController(
        gamma=gamma, beta=beta, alpha=alpha,
        sigma_p=sigma_p, sigma_psi=sigma_psi,
        use_kalman=use_kalman,
    )


def closed_loop(plant, controller, ref_fn, n_steps, dt,
                u0=None, settle_time=1.5, noisy=False, u_limit=0.012,
                on_step=None):
    """
    Run the control loop: reference -> controller -> plant -> measured pose.

    ref_fn(t) -> [x, y, psi] reference pose at time t.

    Returns a dict of per-step logs. Each entry is recorded every step so that
    metrics can be computed over whole trajectories rather than sampled at one
    instant (the Phase 0 lesson).

    u_limit clamps the commanded tendon displacement to the physical travel of
    the mechanism. Without it a transient controller excursion can command an
    unreachable configuration, and what gets reported is the clamp rather than
    the control law.

    settle_time is a BUDGET, not a fixed cost: plant.step() returns as soon as
    the arm is quasi-static, so a generous budget is nearly free while a mean
    one silently biases everything. At 0.15 s only 2 of 200 steps actually
    reached equilibrium, meaning the controller was reading the arm mid-swing -
    which breaks the quasi-static assumption the paper's method rests on.
    Always check the `settled` fraction in the log before trusting a run.
    """
    u = np.zeros(3) if u0 is None else np.asarray(u0, float).copy()
    log = {k: [] for k in ("t", "ref", "pose", "u", "err", "err_norm",
                           "settled", "clamped", "K_norm")}

    for i in range(n_steps):
        t = i * dt
        ref = np.asarray(ref_fn(t), float)

        pose = plant.step(u, settle_time=settle_time, noisy=noisy)
        if plant.diverged:
            log["diverged_at"] = i
            break

        err = ref - pose
        u_next = controller.compute_control(u, ref, pose)

        clamped = bool(np.any(np.abs(u_next) > u_limit))
        u_next = np.clip(u_next, -u_limit, u_limit)

        log["t"].append(t)
        log["ref"].append(ref.copy())
        log["pose"].append(pose.copy())
        log["u"].append(u.copy())
        log["err"].append(err.copy())
        log["err_norm"].append(float(np.hypot(err[0], err[1])))
        log["settled"].append(bool(getattr(plant, "settled", False)))
        log["clamped"].append(clamped)
        log["K_norm"].append(float(getattr(controller, "last_K_norm", 0.0)))

        if on_step is not None:
            on_step(i, t, ref, pose, u)

        u = u_next

    for k in ("t", "err_norm", "K_norm"):
        log[k] = np.asarray(log[k])
    for k in ("ref", "pose", "u", "err"):
        log[k] = np.asarray(log[k])
    return log


def tracking_stats(log, warmup=0):
    """RMS / mean / max tip error over a run, excluding an initial transient."""
    e = log["err_norm"][warmup:]
    if len(e) == 0:
        return {"rms_mm": float("nan"), "mean_mm": float("nan"),
                "max_mm": float("nan"), "final_mm": float("nan")}
    return {
        "rms_mm": float(np.sqrt(np.mean(e ** 2)) * 1000),
        "mean_mm": float(np.mean(e) * 1000),
        "max_mm": float(np.max(e) * 1000),
        "final_mm": float(e[-1] * 1000),
    }


class PoseUdpPublisher:
    """
    Publishes plant tip pose in the exact JSON schema pose_feedback.py accepts,
    so the existing GUI can consume MuJoCo as if it were the real sensor.
    """

    def __init__(self, host="127.0.0.1", port=5005):
        self.addr = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, pose, confidence=1.0):
        x, y, psi = float(pose[0]), float(pose[1]), float(pose[2])
        pkt = {"x_m": x, "y_m": y, "psi_rad": psi, "confidence": confidence}
        self.sock.sendto(json.dumps(pkt).encode("utf-8"), self.addr)
        return pkt

    def close(self):
        self.sock.close()
