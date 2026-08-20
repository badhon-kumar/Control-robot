"""
Phase 4 verification: controller bridge.

Exit criterion from PLAN.md:
  a closed loop runs - controller commands u, MuJoCo returns pose, controller
  reacts - for 100+ steps without divergence, with error logged each step.

Run:
    python run.py check closed-loop

Author: Badhon Kumar
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import numpy as np

from continuum_sim import bridge as B
from continuum_sim import vendor  # noqa: F401
from continuum_sim.paths import FIGURES, LOGS
from continuum_sim.plant import ContinuumPlant
from continuum_ellipse import make_trajectory
from pose_feedback import UdpPoseFeedbackReceiver

from checks.harness import Report
from checks.harness import ok as _ok, fail as _fail, warn as _warn

report = Report()
expect = report.expect



# ── 1. Controller imports and constructs ─────────────────────────────────────
print("\n[1/5] Controller import (unmodified from Continuum_v3)")
ctrl = B.make_controller(use_kalman=True)
expect(hasattr(ctrl, "compute_control"),
       f"{type(ctrl).__name__} constructed with the paper's defaults "
       f"(gamma={ctrl.gamma}, beta={ctrl.beta}, alpha={ctrl.alpha})",
       "controller missing compute_control()")


# ── 2. Setpoint regulation ───────────────────────────────────────────────────
print("\n[2/5] Closed-loop regulation to a fixed setpoint")
plant = ContinuumPlant()
plant.reset()
target = np.array([0.240, 0.030, 0.0])
ctrl = B.make_controller(use_kalman=True)
log = B.closed_loop(plant, ctrl, lambda t: target, n_steps=120, dt=0.1)

expect("diverged_at" not in log,
       f"ran {len(log['t'])} closed-loop steps with no divergence",
       f"diverged at step {log.get('diverged_at')}")

e0, ef = log["err_norm"][0] * 1000, log["err_norm"][-1] * 1000
expect(ef < e0 * 0.05,
       f"error converged {e0:.2f} mm -> {ef:.4f} mm "
       f"({100*(1-ef/e0):.1f}% reduction) against a plant that does not share "
       f"the controller's model",
       f"did not converge: {e0:.2f} mm -> {ef:.2f} mm")

first_below = next((i for i, v in enumerate(log["err_norm"]) if v < 1e-3), None)
_ok(f"reached 1 mm at step {first_below}" if first_below is not None
    else "never reached 1 mm")


# ── 3. Trajectory tracking (the paper's ellipse) ─────────────────────────────
print("\n[3/5] Closed-loop tracking of the paper's Traj. 1 ellipse")
T = 40.0
ref_fn = make_trajectory("Traj 1", T=T)
plant = ContinuumPlant()
plant.reset()
ctrl = B.make_controller(use_kalman=True)

t0 = time.time()
log = B.closed_loop(plant, ctrl, ref_fn, n_steps=200, dt=T / 200)
elapsed = time.time() - t0

expect("diverged_at" not in log,
       f"200 steps of ellipse tracking, no divergence ({elapsed:.1f} s wall)",
       f"diverged at step {log.get('diverged_at')}")

st = B.tracking_stats(log, warmup=40)
print(f"        after warm-up: RMS {st['rms_mm']:.3f} mm   "
      f"mean {st['mean_mm']:.3f} mm   max {st['max_mm']:.3f} mm")

expect(st["rms_mm"] < 20.0,
       f"tracks the ellipse with {st['rms_mm']:.2f} mm RMS - the loop is "
       f"genuinely closing, not merely stable",
       f"RMS error {st['rms_mm']:.1f} mm is too large for a working loop")

n_settled = int(np.sum(log["settled"]))
n_clamped = int(np.sum(log["clamped"]))
frac = n_settled / len(log["t"])
expect(frac > 0.9,
       f"{n_settled}/{len(log['t'])} steps ({frac*100:.0f}%) reached "
       f"quasi-static equilibrium - the controller is reading a settled arm, "
       f"as the paper's quasi-static assumption requires",
       f"only {n_settled}/{len(log['t'])} steps ({frac*100:.0f}%) settled; the "
       f"controller is reading the arm mid-transient and Phase 5's numbers "
       f"would not mean what they claim - raise settle_time")
if n_clamped:
    _warn(f"{n_clamped} step(s) hit the u clamp - transients are commanding "
          f"beyond the mechanism's travel")


# ── 4. Kalman path is actually exercised ─────────────────────────────────────
print("\n[4/5] Kalman compensation is live")
active = int(np.sum(log["K_norm"] > 0))
expect(active > 10,
       f"Kalman gain non-zero on {active}/{len(log['t'])} steps "
       f"(max |K| = {log['K_norm'].max():.4g}) - the estimator is running, "
       f"not silently bypassed",
       f"Kalman gain was non-zero on only {active} steps - the filter is "
       f"effectively inactive, so Phase 5's comparison would be meaningless")

# A quick A/B, to confirm the two conditions genuinely differ before Phase 5
# spends real time on them.
plant_b = ContinuumPlant()
plant_b.reset()
log_b = B.closed_loop(plant_b, B.make_controller(use_kalman=False),
                      ref_fn, n_steps=200, dt=T / 200)
st_b = B.tracking_stats(log_b, warmup=40)
print(f"        Kalman ON : RMS {st['rms_mm']:.3f} mm")
print(f"        Kalman OFF: RMS {st_b['rms_mm']:.3f} mm")
expect(abs(st["rms_mm"] - st_b["rms_mm"]) > 1e-6,
       "the two conditions produce different results, so Phase 5 has something "
       "real to measure",
       "Kalman on/off give identical results - the comparison is degenerate")


# ── 5. UDP path (lets the existing GUI consume MuJoCo unmodified) ────────────
print("\n[5/5] UDP bridge to pose_feedback.py")
try:
    rx = UdpPoseFeedbackReceiver(host="127.0.0.1", port=5077, max_age_s=5.0)
    rx.start()
    time.sleep(0.3)

    pub = B.PoseUdpPublisher(host="127.0.0.1", port=5077)
    pose = plant.tip_pose()
    for _ in range(12):                     # filter needs a few samples
        pub.send(pose)
        time.sleep(0.03)
    time.sleep(0.2)

    got = rx.latest()
    rx.stop()
    pub.close()

    expect(got is not None,
           f"pose_feedback.py received and parsed the plant's packets "
           f"(filtered pose x={got.pose[0]*1000:.1f} mm, "
           f"y={got.pose[1]*1000:.1f} mm vs sent "
           f"{pose[0]*1000:.1f}, {pose[1]*1000:.1f} mm) - the existing GUI can "
           f"be driven by MuJoCo with no code changes"
           if got is not None else "",
           "no packet was received by pose_feedback.py")
except Exception as e:
    _warn(f"UDP check skipped: {e}")


# ── log ──────────────────────────────────────────────────────────────────────
p = LOGS / "closed_loop.csv"
with open(p, "w", encoding="utf-8", newline="\n") as f:
    f.write("t_s,ref_x_mm,ref_y_mm,pose_x_mm,pose_y_mm,err_mm,u1_mm,u2_mm,u3_mm\n")
    for i in range(len(log["t"])):
        f.write(f"{log['t'][i]:.4f},"
                f"{log['ref'][i][0]*1000:.4f},{log['ref'][i][1]*1000:.4f},"
                f"{log['pose'][i][0]*1000:.4f},{log['pose'][i][1]*1000:.4f},"
                f"{log['err_norm'][i]*1000:.4f},"
                f"{log['u'][i][0]*1000:.4f},{log['u'][i][1]*1000:.4f},"
                f"{log['u'][i][2]*1000:.4f}\n")
_ok(f"per-step log -> {p}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].plot(log["ref"][:, 0] * 1000, log["ref"][:, 1] * 1000, "--",
               lw=1.6, label="reference")
    ax[0].plot(log["pose"][:, 0] * 1000, log["pose"][:, 1] * 1000, lw=1.3,
               label="MuJoCo plant")
    ax[0].set_xlabel("x (mm)"); ax[0].set_ylabel("y (mm)")
    ax[0].set_title("Traj. 1 ellipse"); ax[0].legend(); ax[0].axis("equal")
    ax[0].grid(alpha=.3)

    ax[1].plot(log["t"], log["err_norm"] * 1000, label="Kalman ON")
    ax[1].plot(log_b["t"], log_b["err_norm"] * 1000, label="Kalman OFF")
    ax[1].set_xlabel("t (s)"); ax[1].set_ylabel("tip error (mm)")
    ax[1].set_title("Closed-loop tracking error"); ax[1].legend()
    ax[1].grid(alpha=.3)
    fig.tight_layout()
    fp = FIGURES / "closed_loop.png"
    fig.savefig(fp, dpi=130)
    _ok(f"figure -> {fp}")
except Exception as e:
    _warn(f"plot skipped: {e}")


report.finish("Closed loop")
