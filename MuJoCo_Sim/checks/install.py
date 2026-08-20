"""
Phase 0 verification for the MuJoCo continuum-arm simulation.

Confirms the toolchain works before any robot modelling starts:

  1. mujoco / numpy / matplotlib / imageio import cleanly
  2. an MJCF string compiles into mj_model + mj_data
  3. the mj_step integration loop runs and produces correct physics
  4. a spatial tendon + position actuator can bend a hinge chain
     (smoke test for Phase 2 - tendons are the core mechanism of this project)
  5. offscreen rendering works, so Phase 7 video output is possible

Run:
    python run.py check install

Author: Badhon Kumar
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from checks.harness import ok as _ok, fail as _fail, warn as _warn
from continuum_sim.paths import FIGURES as OUT_DIR


# ── 1. Imports ────────────────────────────────────────────────────────────────
print("\n[1/5] Imports")
try:
    import mujoco
    import numpy as np
    _ok(f"mujoco {mujoco.__version__}")
    _ok(f"numpy  {np.__version__}")
except Exception as e:
    _fail(f"core import failed: {e}")
    sys.exit(1)

for name in ("matplotlib", "imageio", "imageio_ffmpeg"):
    try:
        mod = __import__(name)
        _ok(f"{name} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        _warn(f"{name} unavailable ({e}) - needed for Phase 5/7 output only")


# ── 2 & 3. Minimal MJCF: pendulum falls under gravity ────────────────────────
print("\n[2/5] Compile minimal MJCF (2 bodies, 1 hinge)")

PENDULUM_XML = """
<mujoco model="phase0_pendulum">
  <option timestep="0.002" gravity="0 0 -9.81">
    <flag energy="enable"/>
  </option>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="1 1 0.1" rgba="0.9 0.9 0.9 1"/>
    <body name="anchor" pos="0 0 0.5">
      <geom name="hub" type="sphere" size="0.02" rgba="0.3 0.3 0.3 1"/>
      <body name="link" pos="0 0 0">
        <joint name="hinge" type="hinge" axis="0 1 0"/>
        <geom name="rod" type="capsule" fromto="0 0 0 0.25 0 0" size="0.012"
              rgba="0.15 0.45 0.75 1"/>
        <site name="tip" pos="0.25 0 0" size="0.008" rgba="1 0 0 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

try:
    model = mujoco.MjModel.from_xml_string(PENDULUM_XML)
    data = mujoco.MjData(model)
    _ok(f"compiled: {model.nbody} bodies, {model.njnt} joint(s), {model.nq} DOF")
except Exception as e:
    _fail(f"MJCF compile failed: {e}")
    sys.exit(1)

print("\n[3/5] Physics loop (mj_step)")
tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
mujoco.mj_forward(model, data)
z_start = float(data.site_xpos[tip_id][2])

# The pendulum is undamped, so it oscillates. Sampling a single instant is
# meaningless - it may be caught mid-swing back at the start. Track the
# extremes over a full period instead.
z_min = z_start
angle_max = 0.0
energy_0 = None

for _ in range(500):                      # 500 * 0.002 s = 1.0 s
    mujoco.mj_step(model, data)
    z_min = min(z_min, float(data.site_xpos[tip_id][2]))
    angle_max = max(angle_max, abs(float(data.qpos[0])))
    e = float(data.energy[0] + data.energy[1])
    if energy_0 is None:
        energy_0 = e
    energy_now = e

rod_len = 0.25
if z_min < z_start - 0.9 * rod_len:
    _ok(f"tip swung down under gravity: z {z_start:.4f} m -> {z_min:.4f} m "
        f"(drop {(z_start - z_min)*1000:.1f} mm of {rod_len*1000:.0f} mm rod)")
    _ok(f"peak swing {np.degrees(angle_max):.1f} deg "
        f"(expected ~180 deg for a rod released horizontally)")
else:
    _fail(f"tip did not swing down (z {z_start:.4f} -> min {z_min:.4f}) "
          f"- check gravity/joint axis")

if energy_0 is not None and abs(energy_0) > 1e-9:
    drift = abs(energy_now - energy_0) / abs(energy_0)
    if drift < 0.05:
        _ok(f"energy conserved to {drift*100:.2f}% - integrator is stable")
    else:
        _warn(f"energy drifted {drift*100:.1f}% - reduce timestep in later phases")


# ── 4. Spatial tendon smoke test (de-risks Phase 2) ──────────────────────────
print("\n[4/5] Spatial tendon + position actuator")

TENDON_XML = """
<mujoco model="phase0_tendon">
  <option timestep="0.001" gravity="0 0 0"/>
  <worldbody>
    <light pos="0 0 1"/>
    <site name="s_base" pos="0 0.005 0" size="0.002"/>
    <body name="l1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1" stiffness="0.05" damping="0.01"/>
      <geom type="capsule" fromto="0 0 0 0.045 0 0" size="0.0065" rgba="0.2 0.6 0.4 1"/>
      <site name="s1" pos="0.045 0.005 0" size="0.002"/>
      <body name="l2" pos="0.045 0 0">
        <joint name="j2" type="hinge" axis="0 0 1" stiffness="0.05" damping="0.01"/>
        <geom type="capsule" fromto="0 0 0 0.045 0 0" size="0.0065" rgba="0.2 0.6 0.4 1"/>
        <site name="s2" pos="0.045 0.005 0" size="0.002"/>
        <site name="s_tip" pos="0.045 0 0" size="0.003" rgba="1 0 0 1"/>
      </body>
    </body>
  </worldbody>

  <tendon>
    <spatial name="t1" width="0.0008" rgba="0.9 0.2 0.2 1">
      <site site="s_base"/>
      <site site="s1"/>
      <site site="s2"/>
    </spatial>
  </tendon>

  <actuator>
    <position name="a1" tendon="t1" kp="80" ctrlrange="-0.02 0.02"/>
  </actuator>
</mujoco>
"""

try:
    tmodel = mujoco.MjModel.from_xml_string(TENDON_XML)
    tdata = mujoco.MjData(tmodel)
    _ok(f"compiled: {tmodel.ntendon} tendon(s), {tmodel.nu} actuator(s)")

    tip2 = mujoco.mj_name2id(tmodel, mujoco.mjtObj.mjOBJ_SITE, "s_tip")
    mujoco.mj_forward(tmodel, tdata)
    rest_len = float(tdata.ten_length[0])
    y_rest = float(tdata.site_xpos[tip2][1])

    # Command the tendon shorter than its rest length -> the chain should bend.
    tdata.ctrl[0] = rest_len - 0.004
    for _ in range(4000):
        mujoco.mj_step(tmodel, tdata)

    y_pull = float(tdata.site_xpos[tip2][1])
    j1, j2 = np.degrees(tdata.qpos[:2])

    if abs(y_pull - y_rest) > 1e-4:
        _ok(f"tendon rest length {rest_len*1000:.2f} mm")
        _ok(f"pulling 4.0 mm moved tip y: {y_rest*1000:+.3f} -> {y_pull*1000:+.3f} mm")
        _ok(f"joints bent: j1={j1:+.2f} deg, j2={j2:+.2f} deg")
    else:
        _warn("tendon pull produced no motion - revisit actuator kp/stiffness in Phase 2")
except Exception as e:
    _fail(f"tendon test failed: {e}")


# ── 5. Offscreen rendering (needed for Phase 7 video) ────────────────────────
print("\n[5/5] Offscreen rendering")
try:
    with mujoco.Renderer(model, height=480, width=640) as renderer:
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        pixels = renderer.render()

    png_path = OUT_DIR / "toolchain_render.png"
    try:
        import imageio.v3 as iio
        iio.imwrite(png_path, pixels)
        _ok(f"rendered {pixels.shape[1]}x{pixels.shape[0]} frame -> {png_path}")
    except Exception:
        _ok(f"rendered {pixels.shape[1]}x{pixels.shape[0]} frame (not written to disk)")
except Exception as e:
    _warn(f"offscreen render unavailable: {e}")
    _warn("physics is fine; only Phase 7 video export is affected. "
          "Try setting MUJOCO_GL=glfw or updating GPU drivers.")

print("\nToolchain check complete.\n")
