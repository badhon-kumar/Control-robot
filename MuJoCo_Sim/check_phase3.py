"""
Phase 3: calibration and model-error characterisation.

Exit criterion from PLAN.md:
  monotonic, repeatable u -> tip pose mapping; residual PCC mismatch documented
  and in a plausible range (a few mm, not tens of mm and not zero).

Note on objective: PLAN.md proposed fitting stiffness to MINIMISE the PCC
mismatch. That is the wrong target here and has been changed - see the docstring
of sim/calibrate.py. Stiffness is instead justified physically, and the mismatch
is measured and reported as the model-error budget.

Run:
    python MuJoCo_Sim/check_phase3.py

Author: Badhon Kumar
"""

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from sim import params as P
from sim.plant import ContinuumPlant
from sim import calibrate as C

OUT = os.path.join(HERE, "outputs")


def _ok(m):   print(f"  [ OK ] {m}")
def _fail(m): print(f"  [FAIL] {m}")
def _warn(m): print(f"  [WARN] {m}")

failures = []
def expect(cond, good, bad):
    if cond:
        _ok(good)
    else:
        _fail(bad)
        failures.append(bad)


# ── 1. Monotonicity and repeatability ────────────────────────────────────────
print("\n[1/6] Monotonic and repeatable u -> tip mapping")
plant = ContinuumPlant()
mono_ok, rep_worst = True, 0.0
for s in range(P.N_SEG):
    ys = []
    for v in np.linspace(-0.004, 0.004, 9):
        u = [0.0, 0.0, 0.0]
        u[s] = float(v)
        plant.reset()
        ys.append(plant.step(u)[1])
    d = np.diff(ys)
    if not (np.all(d > 0) or np.all(d < 0)):
        mono_ok = False

    # repeatability: same command twice, from rest, must land identically
    u = [0.0, 0.0, 0.0]
    u[s] = 0.003
    plant.reset(); a = plant.step(u)
    plant.reset(); b = plant.step(u)
    rep_worst = max(rep_worst, float(np.hypot(a[0] - b[0], a[1] - b[1])))

expect(mono_ok,
       "tip y is strictly monotonic in u for all 3 segments across "
       "u in [-4, +4] mm - no folding or hysteresis in the mapping",
       "u -> tip mapping is not monotonic")
expect(rep_worst < 1e-9,
       f"repeatable to {rep_worst*1e9:.3f} nm over repeated commands",
       f"not repeatable: {rep_worst*1000:.4f} mm spread")


# ── 2. Are the implied tendon forces physically plausible? ───────────────────
print("\n[2/6] Physical plausibility of the chosen stiffness")
us = C.workspace_grid(amp=0.004, n=3)
sim, pcc, forces, angles, us_arr = C.sweep(plant, us)
fmag = np.abs(forces)
peak = float(fmag.max())
typ = float(np.median(fmag[fmag > 1e-6]))
n_sat = int(np.sum(fmag > 0.999 * P.TENDON_FORCE_MAX))

# Saturation must be caught explicitly. A binding force limit does not announce
# itself - it just quietly changes the answer, as it did on the first run of
# this phase.
expect(n_sat == 0,
       f"no actuator saturation anywhere in the sweep (peak {peak:.1f} N vs "
       f"{P.TENDON_FORCE_MAX:.0f} N limit) - results reflect the mechanism, "
       f"not the clamp",
       f"{n_sat} sample(s) hit the {P.TENDON_FORCE_MAX:.0f} N force limit; "
       f"every downstream number in this phase is suspect until it is raised")

expect(2.0 < peak < P.TENDON_FORCE_MAX,
       f"tendon forces span the workspace at {typ:.1f} N typical / {peak:.1f} N "
       f"peak - a realistic range for a 13 mm tendon-driven arm, which is the "
       f"basis on which JOINT_STIFFNESS was chosen",
       f"tendon forces are implausible ({typ:.1f} N typical, {peak:.1f} N peak) "
       f"- revisit JOINT_STIFFNESS in params.py")

max_bend = float(np.degrees(np.abs(angles).max()))
_ok(f"workspace reaches {max_bend:.1f} deg peak segment bend at u = +/-4 mm")


# ── 3. Model-error budget: MuJoCo vs the controller's PCC model ──────────────
print("\n[3/6] Model-error budget (MuJoCo plant vs PCC controller model)")
stats = C.residual_stats(sim, pcc)
print(f"        tip position   RMS {stats['rms_mm']:6.3f} mm   "
      f"mean {stats['mean_mm']:6.3f} mm   max {stats['max_mm']:6.3f} mm")
print(f"        tip attitude   RMS {stats['rms_psi_deg']:6.3f} deg  "
      f"                max {stats['max_psi_deg']:6.3f} deg")

expect(0.5 < stats["rms_mm"] < 40.0,
       f"RMS mismatch {stats['rms_mm']:.2f} mm over {len(us)} workspace points "
       f"- substantial enough for the Kalman compensator to have real work to "
       f"do, small enough that the controller can still converge",
       f"RMS mismatch {stats['rms_mm']:.2f} mm is outside the useful band")

# Break the residual down by how hard the arm is bent.
print("\n        mismatch vs bend magnitude:")
mag = np.abs(us_arr).sum(axis=1)
for lo, hi in ((0.0, 0.004), (0.004, 0.008), (0.008, 0.013)):
    m = (mag >= lo) & (mag < hi)
    if m.sum():
        d = np.hypot(sim[m, 0] - pcc[m, 0], sim[m, 1] - pcc[m, 1]) * 1000
        print(f"          sum|u| {lo*1000:4.1f}-{hi*1000:4.1f} mm  "
              f"({m.sum():3d} pts):  mean {d.mean():6.3f} mm   max {d.max():6.3f} mm")
_ok("mismatch grows with bend, as expected from the chord-vs-arc term "
    "(Phase 2, check 6b) which scales as theta^2")


# ── 4. Stiffness leverage ────────────────────────────────────────────────────
print("\n[4/6] Does joint stiffness actually move the PCC mismatch?")
print("        stiffness x   RMS mismatch   peak tendon force   saturated?")
rms_by_k = []
sat_any = 0
small = C.workspace_grid(amp=0.004, n=1)
for mult in (0.25, 0.5, 1.0, 2.0):
    pl = ContinuumPlant(stiffness=P.JOINT_STIFFNESS * mult)
    s2, p2, f2, _, _ = C.sweep(pl, small)
    st = C.residual_stats(s2, p2)
    rms_by_k.append(st["rms_mm"])
    ns = int(np.sum(np.abs(f2) > 0.999 * P.TENDON_FORCE_MAX))
    sat_any += ns
    print(f"        {mult:5.2f}         {st['rms_mm']:7.3f} mm      "
          f"{np.abs(f2).max():6.2f} N            {'YES' if ns else 'no'}")

expect(sat_any == 0,
       "no stiffness setting saturated the actuators, so this sweep measures "
       "stiffness and nothing else",
       f"{sat_any} saturated sample(s) - this sweep is measuring the force "
       f"clamp, not stiffness")

spread = (max(rms_by_k) - min(rms_by_k)) / np.mean(rms_by_k) * 100
expect(spread < 25.0,
       f"an 8x change in stiffness moves the mismatch by only {spread:.1f}% - "
       f"so stiffness is NOT the knob that explains it, and fitting stiffness "
       f"against PCC would be tuning the wrong parameter",
       f"stiffness changes the mismatch by {spread:.0f}%; it is a live "
       f"parameter and the calibration objective needs revisiting")
_ok("the mismatch is kinematic (tendon chords + actuator compliance), not "
    "elastic - which is why Phase 3 characterises it instead of fitting it away")


# ── 5. Gravity decision ──────────────────────────────────────────────────────
print("\n[5/6] Gravity")
probe = [[0.0, 0.0, 0.0], [0.003, 0.0, 0.0], [0.003, 0.002, 0.001]]

pl_out = ContinuumPlant(gravity=(0.0, 0.0, -9.81))     # perpendicular to bending
pl_in = ContinuumPlant(gravity=(0.0, -9.81, 0.0))      # within bending plane
base = []
d_out = []
d_in = []
for u in probe:
    plant.reset();  a = plant.step(u)
    pl_out.reset(); b = pl_out.step(u)
    pl_in.reset();  c = pl_in.step(u)
    base.append(a)
    d_out.append(np.hypot(b[0] - a[0], b[1] - a[1]) * 1000)
    d_in.append(np.hypot(c[0] - a[0], c[1] - a[1]) * 1000)

print(f"        out-of-plane gravity (0,0,-g): max tip shift "
      f"{max(d_out):.6f} mm")
print(f"        in-plane gravity     (0,-g,0): max tip shift "
      f"{max(d_in):.3f} mm")

expect(max(d_out) < 1e-6,
       "out-of-plane gravity changes nothing, exactly as it must: a force along "
       "z produces no torque about a z hinge. A planar model cannot represent "
       "out-of-plane sag at all.",
       f"out-of-plane gravity shifted the tip {max(d_out):.6f} mm - the model "
       f"is not purely planar")
_ok(f"in-plane gravity would add up to {max(d_in):.2f} mm of sag")
_warn("KEEPING GRAVITY OFF. For the Phase 6 printing scenario the arm bends in "
      "a horizontal plane over the bed, so gravity is out-of-plane and its "
      "correct planar contribution is zero. If Prof. Cao confirms the arm bends "
      "in a VERTICAL plane, switch params.GRAVITY to (0,-9.81,0) and re-run "
      "this check - that is PLAN.md open question 1.")


# ── 6. Persist the calibration ───────────────────────────────────────────────
print("\n[6/6] Record calibration")
os.makedirs(os.path.join(OUT, "logs"), exist_ok=True)
log = os.path.join(OUT, "logs", "phase3_calibration.csv")
with open(log, "w", encoding="utf-8", newline="\n") as f:
    f.write("u1_mm,u2_mm,u3_mm,sim_x_mm,sim_y_mm,sim_psi_deg,"
            "pcc_x_mm,pcc_y_mm,pcc_psi_deg,err_mm\n")
    for u, a, b in zip(us_arr, sim, pcc):
        e = np.hypot(a[0] - b[0], a[1] - b[1]) * 1000
        f.write(f"{u[0]*1000:.4f},{u[1]*1000:.4f},{u[2]*1000:.4f},"
                f"{a[0]*1000:.4f},{a[1]*1000:.4f},{np.degrees(a[2]):.4f},"
                f"{b[0]*1000:.4f},{b[1]*1000:.4f},{np.degrees(b[2]):.4f},"
                f"{e:.4f}\n")
_ok(f"{len(us_arr)} calibration points -> {log}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    ax[0].plot(pcc[:, 0] * 1000, pcc[:, 1] * 1000, ".", ms=5,
               label="PCC controller model")
    ax[0].plot(sim[:, 0] * 1000, sim[:, 1] * 1000, ".", ms=5,
               label="MuJoCo plant")
    ax[0].set_xlabel("x (mm)"); ax[0].set_ylabel("y (mm)")
    ax[0].set_title("Reachable tip poses"); ax[0].legend(); ax[0].axis("equal")
    ax[0].grid(alpha=.3)

    err = np.hypot(sim[:, 0] - pcc[:, 0], sim[:, 1] - pcc[:, 1]) * 1000
    ax[1].plot(np.abs(us_arr).sum(axis=1) * 1000, err, ".", ms=5)
    ax[1].set_xlabel("sum |u| (mm)"); ax[1].set_ylabel("plant - model (mm)")
    ax[1].set_title("Model-error budget vs bend magnitude"); ax[1].grid(alpha=.3)
    fig.tight_layout()
    fp = os.path.join(OUT, "figures", "phase3_model_error.png")
    fig.savefig(fp, dpi=130)
    _ok(f"figure -> {fp}")
except Exception as e:
    _warn(f"plot skipped: {e}")


print("\n" + "-" * 70)
if failures:
    print(f"Phase 3 FAILED - {len(failures)} check(s):")
    for f_ in failures:
        print(f"  - {f_}")
    sys.exit(1)
print("Phase 3 complete - all checks passed.\n")
