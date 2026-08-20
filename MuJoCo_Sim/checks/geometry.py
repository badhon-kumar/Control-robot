"""
Phase 1 verification: planar arm model (geometry only, no tendons).

Exit criteria from PLAN.md:
  - the arm is a 270 mm chain of 18 links
  - manually setting joint angles bends it smoothly
  - the tip site position matches hand-computed geometry

Checks performed:
  1. structural counts and mass
  2. parameters agree with Continuum_v3/continuum_ellipse.py (no silent drift)
  3. undeformed tip lands exactly at (0.270, 0, 0)
  4. MuJoCo tip vs. an independently computed rigid-chain formula
     -> validates that the model was BUILT as intended
  5. rigid chain vs. the continuous PCC arc
     -> quantifies the discretization error the controller will see
  6. discretization sensitivity (6 / 12 / 24 links per segment)
  7. dynamic stability at the chosen stiffness/damping/armature
  8. render the bent arm

Run:
    python run.py check geometry

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
import continuum_ellipse as ce

from checks.harness import Report
from checks.harness import ok as _ok, fail as _fail, warn as _warn

report = Report()
expect = report.expect



def tip_pose(model, data):
    """Tip (x, y, psi) from MuJoCo. psi is rotation about +z of the tip site."""
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
    x, y = data.site_xpos[sid][0], data.site_xpos[sid][1]
    R = data.site_xmat[sid].reshape(3, 3)
    return float(x), float(y), float(np.arctan2(R[1, 0], R[0, 0]))


# ── 1. Structure ─────────────────────────────────────────────────────────────
print("\n[1/8] Model structure")
xml_path = build_model.write_xml()
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

expect(model.njnt == P.N_LINKS,
       f"{model.njnt} hinge joints ({P.N_SEG} segments x {P.LINKS_PER_SEG} links)",
       f"expected {P.N_LINKS} joints, got {model.njnt}")
expect(model.nq == P.N_LINKS,
       f"{model.nq} DOF - all planar, one rotation each",
       f"expected {P.N_LINKS} DOF, got {model.nq}")

axes = model.jnt_axis[:model.njnt]
expect(np.allclose(axes, np.array([0.0, 0.0, 1.0])),
       "every hinge axis is +z, so motion is confined to the x-y plane",
       "some hinge axes are not +z - the model is not planar")

_ok(f"total mass {sum(model.body_mass)*1000:.1f} g "
    f"({model.ngeom} geoms, {model.nsite} sites)")


# ── 2. Vendored controller: integrity, and drift vs upstream ─────────────────
#
# The controller lives in controller/ so MuJoCo_Sim is portable. That means two
# copies exist and could silently diverge, which would invalidate every result
# without raising an error. So verify BOTH:
#   a) the vendored files still match MANIFEST.sha256  (always checkable)
#   b) they still match a sibling Continuum_v3/, if one is present
# Both are implemented in continuum_sim.vendor, which sync_controller.py shares.
print("\n[2/8] Vendored controller integrity")

intact, problems = vendor.check_integrity()
if not vendor.read_manifest():
    _warn(problems[0])
else:
    expect(intact,
           f"all {len(vendor.read_manifest())} vendored files match "
           f"MANIFEST.sha256 - the controller under test is byte-identical to "
           f"what was vendored",
           f"vendored controller has been altered: {', '.join(problems)}. "
           f"Re-run sync_controller.py or restore the files.")

status, drift = vendor.check_drift()
if status == "drift":
    _warn(f"UPSTREAM DRIFT: {', '.join(drift)} differ from Continuum_v3/. "
          f"The simulation is running an OLD controller. "
          f"Run sync_controller.py to update.")
elif status == "clean":
    _ok("vendored copies are identical to the sibling Continuum_v3/")
else:
    _ok("no sibling Continuum_v3/ present - running fully standalone")

print("\n      geometry constants vs the controller")
try:
    expect(list(ce.L_SEG) == list(P.L_SEG),
           f"L_SEG matches: {P.L_SEG}",
           f"L_SEG drift: controller {ce.L_SEG} vs params {P.L_SEG}")
    expect(list(ce.R_TENDON) == list(P.R_TENDON),
           f"R_TENDON matches: {P.R_TENDON}",
           f"R_TENDON drift: controller {ce.R_TENDON} vs params {P.R_TENDON}")
    expect(ce.SPACER_DISKS_PER_SEG == P.SPACER_DISKS_PER_SEG
           and ce.END_DISKS_PER_SEG == P.END_DISKS_PER_SEG,
           f"disk counts match: {P.SPACER_DISKS_PER_SEG} spacer + "
           f"{P.END_DISKS_PER_SEG} end per segment",
           "disk counts differ from the controller module")
    n_disks_ctrl = len(ce.disk_layout_mm())
    expect(n_disks_ctrl == P.N_LINKS + 1,
           f"disk layout matches: {n_disks_ctrl} disks = 1 base + {P.N_LINKS} link disks",
           f"disk layout mismatch: controller {n_disks_ctrl} vs model {P.N_LINKS + 1}")
except Exception as e:
    _warn(f"controller module not importable ({e}) - skipped cross-check")


# ── 3. Undeformed geometry ───────────────────────────────────────────────────
print("\n[3/8] Undeformed pose")
data.qpos[:] = 0.0
mujoco.mj_forward(model, data)
x0, y0, psi0 = tip_pose(model, data)

expect(abs(x0 - P.TOTAL_LEN) < 1e-9 and abs(y0) < 1e-9 and abs(psi0) < 1e-9,
       f"tip at ({x0*1000:.6f}, {y0*1000:.6f}) mm, psi={np.degrees(psi0):.6f} deg "
       f"- exactly the {P.TOTAL_LEN*1000:.0f} mm design length",
       f"tip at ({x0*1000:.4f}, {y0*1000:.4f}) mm, expected ({P.TOTAL_LEN*1000:.0f}, 0)")


# ── 4. MuJoCo vs. independent rigid-chain formula ────────────────────────────
print("\n[4/8] MuJoCo vs. independent rigid-chain kinematics")
worst = 0.0
for theta_deg in (5, 15, 30, 45, 60):
    theta = np.radians(theta_deg)               # per-segment bend
    delta = theta / P.LINKS_PER_SEG             # per-joint angle
    data.qpos[:] = delta
    mujoco.mj_forward(model, data)
    xm, ym, pm = tip_pose(model, data)
    xc, yc, pc = P.chain_tip_pose([delta] * P.N_LINKS)
    err = np.hypot(xm - xc, ym - yc)
    worst = max(worst, err)

expect(worst < 1e-9,
       f"agrees with the analytic chain to {worst*1e6:.4f} um across 5-60 deg "
       f"per-segment bends - model is built as intended",
       f"disagrees with analytic chain by up to {worst*1000:.4f} mm")


# ── 5. Discretization error vs. the continuous PCC arc ───────────────────────
print("\n[5/8] Rigid chain vs. continuous PCC arc (discretization error)")
print("        per-seg bend |   PCC arc tip (mm)   |  MuJoCo tip (mm)  |  error")
for theta_deg in (15, 30, 45, 60, 90):
    theta = np.radians(theta_deg)
    delta = theta / P.LINKS_PER_SEG
    data.qpos[:] = delta
    mujoco.mj_forward(model, data)
    xm, ym, _ = tip_pose(model, data)
    xa, ya, _ = P.arc_tip_pose([theta] * P.N_SEG)
    err = np.hypot(xm - xa, ym - ya) * 1000
    print(f"        {theta_deg:6.0f} deg   | "
          f"({xa*1000:7.2f},{ya*1000:7.2f}) | "
          f"({xm*1000:7.2f},{ym*1000:7.2f}) | {err:6.3f} mm")

# Worst case over the working range must stay well inside the paper's
# tracking-error scale, or the discretization itself would dominate results.
theta = np.radians(60)
data.qpos[:] = theta / P.LINKS_PER_SEG
mujoco.mj_forward(model, data)
xm, ym, _ = tip_pose(model, data)
xa, ya, _ = P.arc_tip_pose([theta] * P.N_SEG)
err60 = np.hypot(xm - xa, ym - ya) * 1000
expect(err60 < 1.0,
       f"discretization error {err60:.3f} mm at 60 deg/segment - "
       f"small relative to expected tracking errors",
       f"discretization error {err60:.3f} mm is too large; increase LINKS_PER_SEG")


# ── 6. Discretization sensitivity (PLAN.md risk item) ────────────────────────
# Capped at 24 links/segment: MJCF nests one body per link, and 48/segment
# (144 nested bodies) overflows the XML parser's stack and hard-crashes the
# process (0xC00000FD). 24/segment = 72 nested bodies is safely below that.
print("\n[6/8] Sensitivity to link count (60 deg per segment)")
theta = np.radians(60)
xa, ya, _ = P.arc_tip_pose([theta] * P.N_SEG)
errs = {}
for nps in (6, 12, 24):
    m2 = mujoco.MjModel.from_xml_string(build_model.build_xml(links_per_seg=nps))
    d2 = mujoco.MjData(m2)
    d2.qpos[:] = theta / nps
    mujoco.mj_forward(m2, d2)
    x2, y2, _ = tip_pose(m2, d2)
    e = np.hypot(x2 - xa, y2 - ya) * 1000
    errs[nps] = e
    tag = "  <- chosen" if nps == P.LINKS_PER_SEG else ""
    print(f"        {nps:2d} links/segment ({nps*P.N_SEG:3d} total): "
          f"error {e:6.4f} mm{tag}")

# Second order means error should fall ~4x per doubling. Verifying rather than
# asserting: a ~2x ratio would mean the hinges are misplaced again.
r1, r2 = errs[6] / errs[12], errs[12] / errs[24]
expect(r1 > 3.0 and r2 > 3.0,
       f"error falls {r1:.2f}x then {r2:.2f}x per doubling - second-order "
       f"convergence confirms hinges are correctly centred",
       f"error falls only {r1:.2f}x/{r2:.2f}x per doubling - first-order, "
       f"hinge placement is biased")


# ── 7. Dynamic stability ─────────────────────────────────────────────────────
print("\n[7/8] Dynamic stability (elastic relaxation, gravity off)")
data2 = mujoco.MjData(model)
# Neutralise the actuators first. A fresh MjData has ctrl = 0, which for a
# position actuator on tendon LENGTH means "command zero length" - i.e. haul the
# tendon in as far as ctrlrange allows. That is a ~20 mm pull, not a released
# arm, and it silently turned this free-relaxation test into a hard-load test
# the moment Phase 2 added tendons to the model.
rest = P.rest_lengths_from_model(model, data2, mujoco)
data2.ctrl[:] = rest
data2.qpos[:] = np.radians(45) / P.LINKS_PER_SEG      # release from a bent pose
mujoco.mj_forward(model, data2)
x_start, y_start, _ = tip_pose(model, data2)

n_steps = int(3.0 / P.TIMESTEP)
blew_up = False
for _ in range(n_steps):
    mujoco.mj_step(model, data2)
    if not np.all(np.isfinite(data2.qpos)):
        blew_up = True
        break

if blew_up:
    expect(False, "", "simulation diverged - reduce timestep or raise armature/damping")
else:
    resid = float(np.max(np.abs(data2.qpos)))
    xr, yr, _ = tip_pose(model, data2)
    expect(resid < np.radians(0.5),
           f"released from 45 deg/segment, relaxed to straight in 3.0 s "
           f"(max residual joint angle {np.degrees(resid):.4f} deg)",
           f"did not settle: max residual joint angle {np.degrees(resid):.3f} deg")
    _ok(f"tip returned ({x_start*1000:.1f},{y_start*1000:.1f}) -> "
        f"({xr*1000:.1f},{yr*1000:.1f}) mm, no divergence over {n_steps} steps")


# ── 8. Render ────────────────────────────────────────────────────────────────
print("\n[8/8] Render")
try:
    cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "planar")
    frames = []
    with mujoco.Renderer(model, height=600, width=900) as r:
        for theta_deg in (0, 30, 60):
            data.qpos[:] = np.radians(theta_deg) / P.LINKS_PER_SEG
            mujoco.mj_forward(model, data)
            r.update_scene(data, camera=cam)
            frames.append(r.render())
    strip = np.concatenate(frames, axis=1)
    path = OUT_DIR / "geometry_arm.png"
    import imageio.v3 as iio
    iio.imwrite(path, strip)
    _ok(f"rendered straight / 30 deg / 60 deg poses -> {path}")
except Exception as e:
    _warn(f"render skipped: {e}")


# ── Summary ──────────────────────────────────────────────────────────────────
report.finish("Geometry")
