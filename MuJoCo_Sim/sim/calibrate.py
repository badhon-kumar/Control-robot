"""
Phase 3 calibration utilities.

IMPORTANT - what "calibration" means here, and what it must NOT mean.

PLAN.md originally proposed fitting joint stiffness to minimise the MuJoCo-vs-PCC
discrepancy. That objective is wrong for this project. The whole reason the
MuJoCo plant exists is that it does NOT share the controller's model; driving the
discrepancy to zero would rebuild the tautology described in PLAN.md section 1,
just with more machinery. A perfectly "calibrated" plant in that sense would make
Phase 5 meaningless.

So the objective here is PHYSICAL PLAUSIBILITY, not agreement with PCC:

  * stiffness is chosen so the tendon forces needed to span the workspace are
    realistic for this mechanism (order 10-30 N), which is how JOINT_STIFFNESS
    was derived in params.py;
  * the resulting MuJoCo-vs-PCC mismatch is then MEASURED and reported as the
    model-error budget - the input to Phase 5, not something to be removed.

fit_stiffness() below fits against *measured tip poses*. It is the right tool the
day hardware data exists (see PLAN.md open question 2). It is deliberately NOT
pointed at PCC-generated targets.

Author: Badhon Kumar
"""

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "controller"))

from sim import params as P
from sim.plant import ContinuumPlant

from continuum_ellipse import forward_kinematics


def workspace_grid(amp=0.004, n=3):
    """
    Control inputs spanning the reachable workspace.

    Includes single-segment sweeps (which expose per-segment behaviour) and a
    coarse 3-D grid (which exposes coupling).
    """
    us = []
    for s in range(P.N_SEG):
        for v in np.linspace(-amp, amp, 2 * n + 1):
            u = [0.0, 0.0, 0.0]
            u[s] = float(v)
            us.append(u)
    for a in np.linspace(-amp, amp, n):
        for b in np.linspace(-amp, amp, n):
            for c in np.linspace(-amp, amp, n):
                us.append([float(a), float(b), float(c)])
    return us


def sweep(plant, us, settle_time=2.0):
    """
    Drive the plant over a list of control inputs and record, for each:
    the MuJoCo tip pose, the PCC-predicted tip pose, and the tendon forces.

    Each point starts from rest so results are path-independent and directly
    comparable; a continuum arm under tendon load can otherwise carry history.
    """
    sim, pcc, forces, angles = [], [], [], []
    for u in us:
        plant.reset()
        p = plant.step(u, settle_time=settle_time)
        sim.append(p)
        pcc.append(forward_kinematics(np.asarray(u, float)))
        forces.append(np.asarray(plant.data.actuator_force).copy())
        angles.append(plant.segment_angles())
    return (np.array(sim), np.array(pcc), np.array(forces), np.array(angles),
            np.array(us))


def residual_stats(sim, pcc):
    """Tip position/attitude discrepancy between plant and controller model."""
    dp = np.hypot(sim[:, 0] - pcc[:, 0], sim[:, 1] - pcc[:, 1])
    dpsi = np.abs(sim[:, 2] - pcc[:, 2])
    return {
        "rms_mm": float(np.sqrt(np.mean(dp ** 2)) * 1000),
        "mean_mm": float(np.mean(dp) * 1000),
        "max_mm": float(np.max(dp) * 1000),
        "rms_psi_deg": float(np.degrees(np.sqrt(np.mean(dpsi ** 2)))),
        "max_psi_deg": float(np.degrees(np.max(dpsi))),
    }


def fit_stiffness(measured_u, measured_pose, scales0=(1.0, 1.0, 1.0),
                  bounds=(0.2, 5.0), iters=3, grid=7, **plant_kwargs):
    """
    Fit per-segment joint-stiffness multipliers so the plant reproduces
    MEASURED tip poses.

    measured_u    : list of control inputs, metres
    measured_pose : matching measured tip poses [x, y, psi], SI

    Coordinate descent on a shrinking grid - the objective is cheap but noisy
    (each evaluation rebuilds and settles the model), so a derivative-free
    search is more robust here than a gradient method.

    Point this at real hardware data. Pointing it at PCC-generated poses would
    defeat the purpose of having an independent plant - see the module docstring.
    """
    scales = list(scales0)
    lo, hi = bounds

    def cost(sc):
        plant = ContinuumPlant(seg_stiffness_scale=list(sc), **plant_kwargs)
        err = []
        for u, target in zip(measured_u, measured_pose):
            plant.reset()
            p = plant.step(u)
            err.append(np.hypot(p[0] - target[0], p[1] - target[1]))
        return float(np.sqrt(np.mean(np.square(err))))

    best = cost(scales)
    span = (hi - lo) / 2.0
    for _ in range(iters):
        for s in range(P.N_SEG):
            centre = scales[s]
            for cand in np.linspace(max(lo, centre - span),
                                    min(hi, centre + span), grid):
                trial = list(scales)
                trial[s] = float(cand)
                c = cost(trial)
                if c < best:
                    best, scales = c, trial
        span /= 2.0
    return scales, best
