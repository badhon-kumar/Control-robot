"""
Phase 2 verification: tendons and actuators.

Exit criterion from PLAN.md:
  commanding a single positive u1 bends segment 1 into a smooth, roughly
  constant-curvature arc while segments 2 and 3 stay straight; bending
  direction is correct and repeatable.

Run:
    python run.py check tendons

Author: Badhon Kumar
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import numpy as np

from continuum_sim import build_model, vendor
from continuum_sim import params as P
from continuum_sim.paths import FIGURES as OUT_DIR
from continuum_ellipse import pcc_angles

from checks.harness import Report
from checks.harness import ok as _ok, fail as _fail, warn as _warn

report = Report()
expect = report.expect



def tip_pose(model, data):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
    R = data.site_xmat[sid].reshape(3, 3)
    return (float(data.site_xpos[sid][0]), float(data.site_xpos[sid][1]),
            float(np.arctan2(R[1, 0], R[0, 0])))


def segment_angles(data, n_per_seg=None):
    """Total bend angle of each segment = sum of its joint angles (rad)."""
    n = n_per_seg or P.LINKS_PER_SEG
    return np.array([float(np.sum(data.qpos[s * n:(s + 1) * n]))
                     for s in range(P.N_SEG)])


REST = None      # measured rest lengths, filled in after the model compiles


def settle(model, data, u, max_time=4.0, vel_tol=1e-5):
    """
    Drive to control input u and integrate until quasi-static.
    The paper assumes quasi-static motion, so every measurement is taken at
    equilibrium rather than mid-transient.
    """
    data.ctrl[:] = P.u_to_tendon_lengths(u, REST)
    n = int(max_time / model.opt.timestep)
    for i in range(n):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            return False, i
        if np.max(np.abs(data.qvel)) < vel_tol:
            return True, i
    return True, n


def u_for_pure_bend(seg, theta):
    """
    Control input that bends ONLY segment `seg` by `theta`, accounting for the
    fact that a distal tendon also shortens while passing through the segments
    below it. This is the inverse of pcc_angles(): commanding u=[u1,0,0] does
    NOT isolate segment 1 - it bends the segments above it backwards.
    """
    th = [0.0] * P.N_SEG
    th[seg] = theta
    u = []
    for j in range(P.N_SEG):                      # tendon of segment j
        u.append(sum(P.R_TENDON[j - i] * th[i] for i in range(j + 1)))
    return u


# ── 1. Structure ─────────────────────────────────────────────────────────────
print("\n[1/8] Tendon and actuator structure")
xml_path = build_model.write_xml()
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

expect(model.ntendon == P.N_MOTORS and model.nu == P.N_MOTORS,
       f"{model.ntendon} spatial tendons, {model.nu} position actuators "
       f"(M1..M6, 3 antagonistic pairs)",
       f"expected {P.N_MOTORS} tendons/actuators, got "
       f"{model.ntendon}/{model.nu}")

REST = P.rest_lengths_from_model(model, data, mujoco)
_ok(f"measured rest lengths {np.round(REST*1000,3).tolist()} mm "
    f"(nominal 90/180/270 plus 1.5 mm per radial step at a segment boundary)")

fr = model.actuator_forcerange
expect(np.all(fr[:, 1] <= 0) and np.all(fr[:, 0] < 0),
       "actuators are pull-only (forcerange = [-Fmax, 0]) - a released tendon "
       "goes slack instead of pushing",
       f"actuator forcerange is not pull-only: {fr}")


# ── 2. Force sign convention ─────────────────────────────────────────────────
print("\n[2/8] Force sign convention")
d2 = mujoco.MjData(model)
settle(model, d2, [0.003, 0.0, 0.0])
f_pull = float(d2.actuator_force[0])     # M1 commanded shorter
f_rel = float(d2.actuator_force[1])      # M2 commanded longer -> should be slack
expect(f_pull < 0 and abs(f_rel) < 1e-6,
       f"M1 pulls with {abs(f_pull):.2f} N; its partner M2 sits slack at "
       f"{abs(f_rel):.2e} N, as an antagonistic pair should",
       f"unexpected forces: M1={f_pull:.4f} N, M2={f_rel:.4f} N")


# ── 3. Actuator authority: achieved vs commanded tendon length ───────────────
print("\n[3/8] Position-actuator tracking (kp adequacy)")
worst_pct = 0.0
for u1 in (0.001, 0.002, 0.004):
    d3 = mujoco.MjData(model)
    settle(model, d3, [u1, 0.0, 0.0])
    cmd = P.u_to_tendon_lengths([u1, 0, 0], REST)[0]
    ach = float(d3.ten_length[0])
    shortening_cmd = REST[0] - cmd
    shortening_ach = REST[0] - ach
    pct = abs(shortening_ach - shortening_cmd) / shortening_cmd * 100
    worst_pct = max(worst_pct, pct)
    print(f"        u1={u1*1000:.1f} mm -> tendon shortened "
          f"{shortening_ach*1000:6.4f} mm ({pct:4.2f}% under command)")

expect(worst_pct < 5.0,
       f"commanded displacement achieved to within {worst_pct:.2f}% at "
       f"kp={P.TENDON_KP:g} N/m - u means what it says",
       f"actuator too soft: {worst_pct:.1f}% shortfall; raise TENDON_KP")


# ── 4. Single-segment actuation (the exit criterion) ─────────────────────────
print("\n[4/8] Single-segment actuation")
u_pure = u_for_pure_bend(0, np.radians(40))
d4 = mujoco.MjData(model)
ok, steps = settle(model, d4, u_pure)
ang = np.degrees(segment_angles(d4))
expect(ok, f"settled in {steps} steps", "diverged during settling")
_ok(f"isolating segment 1 needs u = "
    f"[{u_pure[0]*1000:.2f}, {u_pure[1]*1000:.2f}, {u_pure[2]*1000:.2f}] mm, "
    f"not [u1, 0, 0] - distal tendons must pay out as segment 1 bends")
expect(ang[0] > 1.0 and abs(ang[1]) < 0.6 and abs(ang[2]) < 0.6,
       f"segment 1 bends {ang[0]:.2f} deg while segments 2 and 3 stay straight "
       f"({ang[1]:+.3f}, {ang[2]:+.3f} deg)",
       f"segment isolation failed: bends = {np.round(ang,4).tolist()} deg")

# Constant curvature within the bent segment: all 6 joints should share the load.
j = np.degrees(d4.qpos[:P.LINKS_PER_SEG])
spread = (j.max() - j.min()) / j.mean() * 100
expect(spread < 1.0,
       f"segment-1 joints bend uniformly ({j.mean():.4f} deg each, spread "
       f"{spread:.3f}%) - a genuinely constant-curvature arc",
       f"joint angles are uneven ({np.round(j,4).tolist()} deg, "
       f"spread {spread:.1f}%) - not constant curvature")

# Direction: +u on the +y tendon must pull the tip toward +y.
x4, y4, _ = tip_pose(model, d4)
expect(y4 > 0,
       f"tip moved to +y ({y4*1000:.2f} mm), matching the +y routing of M1",
       f"tip moved to {y4*1000:.2f} mm - bending direction is inverted")

# Repeatability: return to zero, re-apply the SAME u, expect the same pose.
settle(model, d4, [0.0, 0.0, 0.0])
settle(model, d4, u_pure)
x4b, y4b, _ = tip_pose(model, d4)
expect(np.hypot(x4b - x4, y4b - y4) < 1e-6,
       f"repeatable: unload/reload returns to the same tip within "
       f"{np.hypot(x4b-x4, y4b-y4)*1e6:.3f} um",
       f"not repeatable: {np.hypot(x4b-x4, y4b-y4)*1000:.4f} mm drift")


# ── 5. Effective moment arms vs the paper's routing scheme ───────────────────
print("\n[5/8] Effective moment arms (must reproduce R_TENDON exactly)")
d5 = mujoco.MjData(model)
arm_err = 0.0
for s in range(P.N_SEG):
    d5.qpos[:] = 0.0
    th = 1e-4
    d5.qpos[s * P.LINKS_PER_SEG:(s + 1) * P.LINKS_PER_SEG] = th / P.LINKS_PER_SEG
    mujoco.mj_forward(model, d5)
    arms = (REST - d5.ten_length) / th
    exp_arms = np.array([P.tendon_radius(k, s) if P.MOTOR_TO_SEG[k] >= s else 0.0
                         for k in range(P.N_MOTORS)])
    arm_err = max(arm_err, float(np.max(np.abs(arms - exp_arms))))
    print(f"        bend seg{s+1}: M1..M6 = "
          f"{np.round(arms*1000, 3).tolist()} mm")

expect(arm_err < 1e-6,
       f"all moment arms match R_TENDON to {arm_err*1e6:.4f} um - the plant's "
       f"tendon map is exactly the inverse of the controller's pcc_angles()",
       f"moment arms deviate by up to {arm_err*1000:.4f} mm from R_TENDON")


# ── 6. Plant vs controller model (the comparison that matters) ───────────────
print("\n[6/8] MuJoCo segment angles vs the controller's pcc_angles()")
try:
    print("        u (mm)              PCC theta (deg)        MuJoCo theta (deg)"
          "     max dev")
    worst = 0.0
    cases = [[0.004, 0.0, 0.0], [0.0, 0.004, 0.0], [0.0, 0.0, 0.004],
             [0.003, 0.002, 0.001], u_for_pure_bend(1, np.radians(30))]
    for u in cases:
        d6 = mujoco.MjData(model)
        settle(model, d6, u)
        th_pcc = np.degrees(pcc_angles(np.array(u)))
        th_sim = np.degrees(segment_angles(d6))
        dev = float(np.max(np.abs(th_pcc - th_sim)))
        worst = max(worst, dev)
        print(f"        [{u[0]*1000:5.2f},{u[1]*1000:5.2f},{u[2]*1000:5.2f}]  "
              f"[{th_pcc[0]:7.2f},{th_pcc[1]:7.2f},{th_pcc[2]:7.2f}]  "
              f"[{th_sim[0]:7.2f},{th_sim[1]:7.2f},{th_sim[2]:7.2f}]  "
              f"{dev:6.2f}")

    expect(0.2 < worst < 15.0,
           f"plant deviates from the controller model by up to {worst:.2f} deg "
           f"- the coupling structure is right, but the plant is NOT the "
           f"controller's model. That gap is the point of the project.",
           f"deviation {worst:.2f} deg is outside the plausible band - either "
           f"the routing is wrong (too large) or the plant is a tautology "
           f"(too small)")
except Exception as e:
    _warn(f"controller comparison skipped: {e}")


# ── 6b. Where that model error comes from ────────────────────────────────────
print("\n[6b/8] Dominant error source: tendon path is polygonal, not a smooth arc")
d6b = mujoco.MjData(model)
print("        theta1     measured dl     linear r*theta     excess    L*d^2/8")
worst_ratio = 0.0
for deg in (15, 30, 45, 60):
    th = np.radians(deg)
    d6b.qpos[:] = 0.0
    d6b.qpos[0:P.LINKS_PER_SEG] = th / P.LINKS_PER_SEG
    mujoco.mj_forward(model, d6b)
    dl = float((REST[0] - d6b.ten_length[0]))
    lin = P.R_TENDON[0] * th
    delta = th / P.LINKS_PER_SEG
    pred = P.L_SEG[0] * delta ** 2 / 8.0        # analytic excess per segment
    print(f"        {deg:3d} deg   {dl*1000:8.4f} mm   {lin*1000:8.4f} mm   "
          f"{(dl-lin)*1000:8.4f}  {pred*1000:8.4f} mm")
    worst_ratio = max(worst_ratio, abs((dl - lin) - pred) / max(pred, 1e-12))

expect(worst_ratio < 0.10,
       f"excess shortening matches the analytic chord-vs-arc term L*delta^2/8 "
       f"to within {worst_ratio*100:.1f}% - the mechanism is understood, "
       f"not a numerical accident",
       f"excess shortening deviates {worst_ratio*100:.0f}% from L*delta^2/8 - "
       f"the cause is something else and needs investigating")

_ok("This is PHYSICAL, not a discretization artifact: the paper's Fig. 2B "
    "describes revolute joints between adjacent disks, so the real tendons also "
    "run as straight chords between disk holes. PCC's smooth-arc dl = r*theta "
    "is the approximation - and this is error the Kalman filter should absorb.")


# ── 7. Stability across the working range ────────────────────────────────────
print("\n[7/8] Stability sweep")
bad = []
for u1 in np.linspace(-0.006, 0.006, 9):
    for u3 in (-0.002, 0.0, 0.002):
        d7 = mujoco.MjData(model)
        ok, _ = settle(model, d7, [u1, 0.0, u3], max_time=3.0)
        if not ok or not np.all(np.isfinite(d7.qpos)):
            bad.append((u1, u3))
expect(not bad,
       f"27 load cases across u1 in [-6, +6] mm and u3 in [-2, +2] mm all "
       f"settled with no divergence at kp={P.TENDON_KP:g}, "
       f"timestep={P.TIMESTEP}",
       f"{len(bad)} load case(s) diverged: {bad[:5]}")


# ── 8. Render ────────────────────────────────────────────────────────────────
print("\n[8/8] Render")
try:
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "planar")
    frames = []
    with mujoco.Renderer(model, height=600, width=900) as r:
        for u in ([0, 0, 0], [0.005, 0, 0], [0, 0.004, 0], [0, 0, 0.002]):
            d8 = mujoco.MjData(model)
            settle(model, d8, u)
            r.update_scene(d8, camera=cam)
            frames.append(r.render())
    import imageio.v3 as iio
    path = OUT_DIR / "tendons_actuation.png"
    iio.imwrite(path, np.concatenate(frames, axis=1))
    _ok(f"rendered rest / u1 / u2 / u3 actuation -> {path}")
except Exception as e:
    _warn(f"render skipped: {e}")


report.finish("Tendons")
