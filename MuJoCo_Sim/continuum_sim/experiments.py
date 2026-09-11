"""
Every experiment reproduced from Zhai et al. 2025, defined in one place.

THIS IS THE FILE YOU EDIT TO ADD A FIGURE. Nothing else needs touching: the
runner, the plotting and the command line all read from the registry below.

An experiment differs from its neighbours in only three ways - the reference it
tracks, which channels are worth plotting, and the condition the plant is in.
Those are data, so they live here as data rather than as another script.

    python run.py fig7                  open one experiment in MuJoCo
    python run.py figures               list what is available

Currently reproduced:

    fig6    Traj. 1 - position tracking round an ellipse, attitude held level.
            Panels C (position and attitude vs time) and D (position error).

    fig7    Traj. 2 - attitude tracking, position commanded to stay put.
            Panels B (position and attitude vs time) and C (attitude error).

    fig12   Payload disturbance - hold a pose, hang a weight on the end disk,
            watch the controller recover. Panels D to F, plus MaxAE results.

GRAVITY IS PER EXPERIMENT, and that is deliberate.

The paper figures are now treated as a vertical bending-plane setup, so gravity
acts in the plotted y direction: gravity = (0, -9.81, 0). This lets Fig. 6 and
Fig. 7 show gravity-induced sag/trajectory-dependent error, and it is required
for Fig. 12 because a hung payload would otherwise produce no in-plane motion.

Author: Badhon Kumar
"""

import math
from dataclasses import dataclass, field
from typing import Callable, Tuple

import numpy as np

# ── Reference trajectories ───────────────────────────────────────────────────
#
# Defined here rather than in the vendored controller. continuum_ellipse.py is
# byte-locked (see controller/README.md) and knows only Traj. 1, but a reference
# trajectory is an experiment definition, not controller code, so this is its
# proper home regardless.

def traj1_ellipse(T=40.0):
    """
    Fig. 6, Traj. 1: an ellipse in position, attitude held horizontal.

        xr = 240 + 20 cos(2 pi t / T) mm
        yr =       60 sin(2 pi t / T) mm
        psi_r = 0
    """
    def fn(t):
        return np.array([0.240 + 0.020 * math.cos(2 * math.pi * t / T),
                         0.060 * math.sin(2 * math.pi * t / T),
                         0.0])
    return fn


def traj2_attitude(T=40.0, hold=(0.240, 0.0)):
    """
    Fig. 7, Traj. 2: attitude swings, position commanded to stay put.

        xr, yr = constant
        psi_r  = 30 deg sin(2 pi t / T)

    The hold position is taken from the paper's own Fig. 7B axes, which are
    centred on x = 240 mm and y = 0 mm.
    """
    def fn(t):
        return np.array([hold[0], hold[1],
                         math.radians(30.0) * math.sin(2 * math.pi * t / T)])
    return fn


def hold_pose(pose=(0.240, 0.0, 0.0)):
    """Fig. 12: a constant setpoint. The disturbance is the payload, not the reference."""
    p = np.asarray(pose, float)
    return lambda t: p.copy()


# ── The registry ─────────────────────────────────────────────────────────────

@dataclass
class Experiment:
    """One reproducible experiment. Add an entry, get a command."""
    key: str
    figure: str                     # which figure of the paper this is
    title: str
    ref_fn: Callable                # t -> [x, y, psi] in SI
    duration_s: float
    ctrl_dt: float = 0.2
    gravity: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    kind: str = "tracking"          # "tracking" (A/B pair) or "disturbance"

    # Which channels the ERROR figure shows. Fig. 6D is position only; Fig. 7C
    # is attitude only. Plotting all three regardless would bury the point.
    error_channels: Tuple[int, ...] = (0, 1, 2)
    time_ylims: Tuple[Tuple[float, float], ...] = ()
    error_ylim: Tuple[float, float] = ()
    path_figure: bool = True        # an x-y plot only makes sense if x-y moves

    # Disturbance experiments only.
    payloads_kg: Tuple[float, ...] = ()
    payload_names: Tuple[str, ...] = ()
    drop_s: float = 3.0             # when the weight is hung

    notes: str = ""

    @property
    def n_steps(self):
        return max(1, int(round(self.duration_s / self.ctrl_dt)))


REGISTRY = {

    "fig6": Experiment(
        key="fig6",
        figure="Fig. 6",
        title="Traj. 1 - position tracking, attitude held level",
        ref_fn=traj1_ellipse(T=40.0),
        duration_s=40.0,
        ctrl_dt=0.05,
        gravity=(0.0, -9.81, 0.0),
        error_channels=(0, 1),      # 6D plots x and y error
        time_ylims=((215.0, 265.0), (-65.0, 65.0), (-5.0, 5.0)),
        error_ylim=(-10.0, 10.0),
        notes="Panels C and D. In-plane gravity enabled.",
    ),

    "fig7": Experiment(
        key="fig7",
        figure="Fig. 7",
        title="Traj. 2 - attitude tracking, position held stationary",
        ref_fn=traj2_attitude(T=40.0),
        duration_s=40.0,
        gravity=(0.0, -9.81, 0.0),
        error_channels=(2,),        # 7C plots attitude error only
        error_ylim=(-10.0, 10.0),
        path_figure=False,          # the position is meant not to move
        notes="Panels B and C. In-plane gravity enabled.",
    ),

    "fig12": Experiment(
        key="fig12",
        figure="Fig. 12",
        title="Payload disturbance while holding a stationary pose",
        ref_fn=hold_pose((0.240, 0.0, 0.0)),
        duration_s=15.0,
        ctrl_dt=0.15,
        gravity=(0.0, -9.81, 0.0),  # see the module docstring
        kind="disturbance",
        payloads_kg=(0.0056, 0.0128, 0.0175),
        payload_names=("Tape core (5.6 g)",
                       "Corner bracket (12.8 g)",
                       "USB converter (17.5 g)"),
        drop_s=3.0,
        path_figure=False,
        notes="Panels D to F, plus Table 4 (MaxAE). Gravity IN the bending "
              "plane - a hung weight does nothing otherwise. See module docstring.",
    ),
}


def get(key):
    if key not in REGISTRY:
        raise SystemExit(f"Unknown experiment '{key}'. "
                         f"Choose from: {', '.join(REGISTRY)}, or 'all'.")
    return REGISTRY[key]
