"""
Reproduces panels C and D of Fig. 6 (Zhai et al. 2025) from the MuJoCo plant.

  C - time plots of x, y, psi          (Reference / PCC model / Proposed)
  D - position tracking errors         (measurement minus reference)

Both conditions are the SAME controller; only the online Jacobian compensation
differs, exactly as in the paper's comparison:

    Proposed  = KalmanJacobianController(use_kalman=True)
    PCC model = KalmanJacobianController(use_kalman=False)

Nothing new is computed here. bridge.closed_loop() already logs the full 3-vector
pose and error every step - the existing phase-4 CSV simply dropped the psi
column. This script reads the same log and plots all three channels.

    python run.py fig6
    python run.py fig6 --steps 400 --paper-limits

Outputs:
    outputs/figures/fig6C_time_plots.png
    outputs/figures/fig6D_tracking_errors.png
    outputs/logs/fig6_traj1_proposed.csv
    outputs/logs/fig6_traj1_pcc.csv

Author: Badhon Kumar
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from continuum_sim import bridge as B
from continuum_sim import vendor  # noqa: F401
from continuum_sim.paths import FIGURES as FIG_DIR
from continuum_sim.paths import LOGS as LOG_DIR
from continuum_sim.plant import ContinuumPlant
from continuum_ellipse import make_trajectory

# Paper's Fig. 6 axis limits, so a side-by-side comparison is honest about scale.
PAPER_C = {"x": (215, 265), "y": (-65, 65), "psi": (-5, 5)}
PAPER_D = (-10, 10)

# Colours follow the paper's ordering: reference dashed black, then the two runs.
C_REF = "#000000"
C_PCC = "#D95F02"
C_PRO = "#1B6CA8"


def run(use_kalman, ref_fn, n_steps, dt, settle_time):
    """One closed-loop run. Returns (t, ref, pose, err) in mm / degrees."""
    plant = ContinuumPlant()
    plant.reset()
    ctrl = B.make_controller(use_kalman=use_kalman)

    t0 = time.time()
    log = B.closed_loop(plant, ctrl, ref_fn, n_steps=n_steps, dt=dt,
                        settle_time=settle_time)
    label = "Proposed" if use_kalman else "PCC model"
    if "diverged_at" in log:
        raise RuntimeError(f"{label} run diverged at step {log['diverged_at']}")

    settled = int(np.sum(log["settled"]))
    print(f"  {label:10s}  {len(log['t'])} steps, "
          f"{settled}/{len(log['t'])} settled, {time.time()-t0:.1f} s wall")
    if settled < 0.9 * len(log["t"]):
        print(f"  {'':10s}  WARNING: only {100*settled/len(log['t']):.0f}% of "
              f"steps reached quasi-static equilibrium; the plotted pose is "
              f"partly mid-transient. Raise --settle.")

    # SI -> the paper's units. Rows are [x, y, psi].
    scale = np.array([1000.0, 1000.0, 180.0 / np.pi])
    return (log["t"],
            log["ref"] * scale,
            log["pose"] * scale,
            log["err"] * scale)


def per_axis_stats(err):
    """RMSE and MAE per channel, as reported in the paper's Table 2."""
    return {
        "rmse": np.sqrt(np.mean(err ** 2, axis=0)),
        "mae": np.mean(np.abs(err), axis=0),
    }


def write_csv(path, t, ref, pose, err):
    hdr = ("t_s,ref_x_mm,ref_y_mm,ref_psi_deg,"
           "pose_x_mm,pose_y_mm,pose_psi_deg,"
           "err_x_mm,err_y_mm,err_psi_deg")
    rows = np.column_stack([t, ref, pose, err])
    np.savetxt(path, rows, delimiter=",", header=hdr, comments="", fmt="%.4f")


def panel_c(t, ref, pose_pcc, pose_pro, paper_limits, out):
    """Fig. 6C - x(t), y(t), psi(t), all three curves per axis."""
    fig, axes = plt.subplots(3, 1, figsize=(6.2, 6.4), sharex=True)
    rows = [("x (mm)", 0, "x"), ("y (mm)", 1, "y"), (r"$\psi$ ($\degree$)", 2, "psi")]

    for ax, (ylabel, i, key) in zip(axes, rows):
        ax.plot(t, ref[:, i], "--", color=C_REF, lw=1.4, label="Reference")
        ax.plot(t, pose_pcc[:, i], "-", color=C_PCC, lw=1.4, label="PCC model")
        ax.plot(t, pose_pro[:, i], "-", color=C_PRO, lw=1.4, label="Proposed")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, lw=0.5)
        ax.margins(x=0)
        if paper_limits:
            ax.set_ylim(*PAPER_C[key])

    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=3)
    fig.suptitle("Fig. 6C  -  Position and attitude vs time (Traj. 1, MuJoCo plant)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def panel_d(t, err_pcc, err_pro, paper_limits, out):
    """Fig. 6D - x and y tracking error (measurement minus reference)."""
    fig, axes = plt.subplots(2, 1, figsize=(6.2, 4.6), sharex=True)

    for ax, (ylabel, i) in zip(axes, [("x error (mm)", 0), ("y error (mm)", 1)]):
        ax.axhline(0.0, color=C_REF, lw=0.8, ls="--", alpha=0.6)
        ax.plot(t, err_pcc[:, i], "-", color=C_PCC, lw=1.4, label="PCC model")
        ax.plot(t, err_pro[:, i], "-", color=C_PRO, lw=1.4, label="Proposed")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, lw=0.5)
        ax.margins(x=0)
        if paper_limits:
            ax.set_ylim(*PAPER_D)

    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
    fig.suptitle("Fig. 6D  -  Position tracking error (Traj. 1, MuJoCo plant)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--period", type=float, default=40.0,
                    help="ellipse period T in seconds (paper: 40)")
    ap.add_argument("--steps", type=int, default=200,
                    help="control steps over one period")
    ap.add_argument("--settle", type=float, default=1.5,
                    help="quasi-static settle budget per control step")
    ap.add_argument("--warmup", type=int, default=40,
                    help="steps excluded from RMSE/MAE (initial approach)")
    ap.add_argument("--paper-limits", action="store_true",
                    help="clamp axes to the paper's Fig. 6 ranges")
    a = ap.parse_args()

    T = a.period
    dt = T / a.steps
    ref_fn = make_trajectory("Traj 1", T=T)

    print(f"\nTraj. 1 ellipse, T = {T:g} s, {a.steps} steps, dt = {dt:.3f} s")
    t, ref, pose_pro, err_pro = run(True, ref_fn, a.steps, dt, a.settle)
    _, _, pose_pcc, err_pcc = run(False, ref_fn, a.steps, dt, a.settle)

    write_csv(LOG_DIR / "fig6_traj1_proposed.csv",
              t, ref, pose_pro, err_pro)
    write_csv(LOG_DIR / "fig6_traj1_pcc.csv",
              t, ref, pose_pcc, err_pcc)

    print(f"\nTable 2 row, this plant (warm-up {a.warmup} steps excluded):")
    print(f"  {'':10s}  {'x/mm':>12s} {'y/mm':>12s} {'psi/deg':>12s}   (RMSE/MAE)")
    for label, e in (("Proposed", err_pro), ("PCC model", err_pcc)):
        s = per_axis_stats(e[a.warmup:])
        cells = "".join(f" {s['rmse'][i]:6.1f}/{s['mae'][i]:<5.1f}" for i in range(3))
        print(f"  {label:10s} {cells}")

    print()
    panel_c(t, ref, pose_pcc, pose_pro, a.paper_limits,
            FIG_DIR / "fig6C_time_plots.png")
    panel_d(t, err_pcc, err_pro, a.paper_limits,
            FIG_DIR / "fig6D_tracking_errors.png")
    print()


if __name__ == "__main__":
    main()
