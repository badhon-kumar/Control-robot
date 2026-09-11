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
import socket

import numpy as np

from . import vendor  # noqa: F401  - puts the vendored controller/ on sys.path
from .realism import RealismConfig, RealismLayer
from continuum_ellipse import KalmanJacobianController, forward_kinematics


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


def initial_u_for_pose(target_pose, u_limit=0.012):
    """
    Find a bounded PCC inverse-kinematic seed for the starting pose.

    This avoids the singular straight-arm problem: at u = 0 the local Jacobian
    cannot reduce x, so a pure feedback pre-roll leaves Fig. 6 starting at
    270 mm instead of the paper's 260 mm first point.
    """
    target = np.asarray(target_pose, float)
    scale = np.array([1000.0, 1000.0, 180.0 / np.pi])

    def cost(u):
        e = (forward_kinematics(np.asarray(u, float)) - target) * scale
        return float(e @ e)

    starts = [
        np.zeros(3),
        np.array([0.002, -0.004, 0.002]),
        np.array([-0.002, 0.004, -0.002]),
        np.array([0.005, -0.010, 0.005]),
        np.array([-0.005, 0.010, -0.005]),
        np.array([0.008, -0.012, 0.008]),
        np.array([-0.008, 0.012, -0.008]),
    ]
    best = starts[0].copy()
    best_cost = cost(best)
    for start in starts:
        u = np.clip(start.astype(float), -u_limit, u_limit)
        step = 0.004
        for _ in range(80):
            improved = False
            base = cost(u)
            for j in range(3):
                for direction in (-1.0, 1.0):
                    trial = u.copy()
                    trial[j] = np.clip(trial[j] + direction * step,
                                       -u_limit, u_limit)
                    c = cost(trial)
                    if c < base:
                        u = trial
                        base = c
                        improved = True
            if not improved:
                step *= 0.5
        c = cost(u)
        if c < best_cost:
            best = u.copy()
            best_cost = c
    return best


def preposition_to_pose(plant, target_pose, dt=0.05, n_steps=160,
                        settle_time=1.5, realism: RealismConfig | None = None,
                        u_limit=0.012):
    """
    Move the plant close to `target_pose` before a logged experiment begins.

    The paper figures start from the commanded trajectory, while the MuJoCo arm
    naturally resets straight at x = total length. This helper performs an
    unlogged settling move, then returns the tendon command that holds the plant
    near the first reference point.
    """
    ctrl = make_controller(use_kalman=True)
    layer = RealismLayer(realism)
    target = np.asarray(target_pose, float)
    u_ctrl = initial_u_for_pose(target, u_limit=u_limit)
    layer.reset(initial_pose=plant.tip_pose(), initial_u=u_ctrl)
    u_plant = layer.command_to_plant(u_ctrl)

    for _ in range(max(1, int(n_steps))):
        true_pose = plant.step(u_plant, settle_time=settle_time, noisy=False)
        if plant.diverged:
            break
        pose = layer.measure(true_pose)
        u_ctrl = np.clip(ctrl.compute_control(u_ctrl, target, pose),
                         -u_limit, u_limit)
        u_plant = np.clip(layer.command_to_plant(u_ctrl), -u_limit, u_limit)

    plant.step(u_plant, settle_time=settle_time, noisy=False)
    return u_ctrl.copy(), u_plant.copy()


def closed_loop(plant, controller, ref_fn, n_steps, dt,
                u0=None, u_plant0=None, settle_time=1.5, noisy=False,
                u_limit=0.012, on_step=None,
                realism: RealismConfig | None = None):
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
    u_ctrl = np.zeros(3) if u0 is None else np.asarray(u0, float).copy()
    layer = RealismLayer(realism)
    layer.reset(initial_pose=plant.tip_pose(), initial_u=u_ctrl,
                initial_applied_u=u_plant0)
    u_plant = (np.asarray(u_plant0, float).copy()
               if u_plant0 is not None else layer.command_to_plant(u_ctrl))
    log = {k: [] for k in ("t", "ref", "pose", "measured_pose", "u", "u_plant",
                           "err", "err_norm", "settled", "clamped", "K_norm")}

    for i in range(n_steps):
        t = i * dt
        ref = np.asarray(ref_fn(t), float)

        true_pose = plant.step(u_plant, settle_time=settle_time, noisy=False)
        if plant.diverged:
            log["diverged_at"] = i
            break

        pose = plant.tip_pose(noisy=True) if noisy else layer.measure(true_pose)
        err = ref - pose
        u_next = controller.compute_control(u_ctrl, ref, pose)

        clamped = bool(np.any(np.abs(u_next) > u_limit))
        u_next = np.clip(u_next, -u_limit, u_limit)

        log["t"].append(t)
        log["ref"].append(ref.copy())
        log["pose"].append(true_pose.copy())
        log["measured_pose"].append(pose.copy())
        log["u"].append(u_ctrl.copy())
        log["u_plant"].append(u_plant.copy())
        log["err"].append(err.copy())
        log["err_norm"].append(float(np.hypot(err[0], err[1])))
        log["settled"].append(bool(getattr(plant, "settled", False)))
        log["clamped"].append(clamped)
        log["K_norm"].append(float(getattr(controller, "last_K_norm", 0.0)))

        if on_step is not None:
            on_step(i, t, ref, pose, u_ctrl)

        u_ctrl = u_next
        u_plant = layer.command_to_plant(u_ctrl)

    for k in ("t", "err_norm", "K_norm"):
        log[k] = np.asarray(log[k])
    for k in ("ref", "pose", "measured_pose", "u", "u_plant", "err"):
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
