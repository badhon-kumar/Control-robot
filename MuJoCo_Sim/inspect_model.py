"""
Inspect the continuum-arm model before running anything automatic.

Prints a structural report you can check line-by-line against the paper, then
either opens the interactive viewer or writes a static contact sheet.

    python MuJoCo_Sim/inspect_model.py                  # report + interactive viewer
    python MuJoCo_Sim/inspect_model.py --u 3,1,0        # open in a bent pose (mm)
    python MuJoCo_Sim/inspect_model.py --shot           # report + PNG sheet, no window
    python MuJoCo_Sim/inspect_model.py --report         # report only

The viewer is opened with ctrl set to the measured REST LENGTHS. That matters:
these are position actuators on tendon LENGTH, so MuJoCo's default ctrl = 0 means
"reel every tendon all the way in" - about a 20 mm pull on all six at once - and
the arm appears violently contorted. Launching the raw XML with
`python -m mujoco.viewer` will show exactly that, and it is not a modelling bug.

Author: Badhon Kumar
"""

import argparse
import os
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "models"))

from sim import params as P
import build_model

OUT = os.path.join(HERE, "outputs", "figures")


# ── structural report ────────────────────────────────────────────────────────
def report(model, data, rest):
    def hdr(t):
        print(f"\n{t}\n" + "-" * len(t))

    hdr("BODIES / JOINTS")
    print(f"  bodies              {model.nbody}  (world + base + {P.N_LINKS} links)")
    print(f"  hinge joints        {model.njnt}   all axis +z -> motion is planar")
    print(f"  DOF                 {model.nq}")
    print(f"  geoms / sites       {model.ngeom} / {model.nsite}")
    print(f"  total mass          {sum(model.body_mass)*1000:.2f} g")
    g = ", ".join(f"{v:g}" for v in model.opt.gravity)
    print(f"  timestep / gravity  {model.opt.timestep} s / ({g}) m/s^2")

    hdr("GEOMETRY (vs Continuum_v3/README.md)")
    print(f"  segments            {P.N_SEG} x {P.L_SEG[0]*1000:.0f} mm "
          f"= {P.TOTAL_LEN*1000:.0f} mm total")
    print(f"  links per segment   {P.LINKS_PER_SEG}  "
          f"({P.SPACER_DISKS_PER_SEG} spacer + {P.END_DISKS_PER_SEG} end disk)")
    print(f"  axial disk pitch    {P.LINK_LEN*1000:.1f} mm")
    print(f"  body diameter       {P.BODY_DIAM*1000:.1f} mm "
          f"(disk radius {P.DISK_RADIUS*1000:.2f} mm)")
    print(f"  backbone radius     {P.BACKBONE_RADIUS*1000:.2f} mm")

    hdr("TENDON ROUTING  (radius in mm, by which segment it passes through)")
    print("   motor  seg  |  in seg1   in seg2   in seg3  | terminates | rest mm")
    for k in range(P.N_MOTORS):
        seg = P.MOTOR_TO_SEG[k]
        cells = []
        for j in range(P.N_SEG):
            cells.append(f"{P.tendon_radius(k, j)*1000:+7.2f}" if j <= seg
                         else "      -")
        print(f"   M{k+1}      {seg+1}   | {cells[0]}  {cells[1]}  {cells[2]}  "
              f"|  seg {seg+1} end | {rest[k]*1000:7.2f}")
    print("\n  Expect 5.00 in its own segment, 3.50 one segment back, 2.00 two back")
    print("  (paper Eqs. 7-9 / Fig. 2C). Odd motors +y, even motors -y.")

    hdr("EFFECTIVE MOMENT ARMS  (measured, d(shortening)/d(theta), mm)")
    d = mujoco.MjData(model)
    d.ctrl[:] = rest
    print("            M1      M2      M3      M4      M5      M6")
    for s in range(P.N_SEG):
        d.qpos[:] = 0.0
        th = 1e-4
        d.qpos[s*P.LINKS_PER_SEG:(s+1)*P.LINKS_PER_SEG] = th / P.LINKS_PER_SEG
        mujoco.mj_forward(model, d)
        arms = (rest - d.ten_length) / th * 1000
        print(f"  bend seg{s+1}  " + "  ".join(f"{a:+6.3f}" for a in arms))

    hdr("ACTUATORS")
    fr = model.actuator_forcerange
    print(f"  type                position, on tendon length")
    print(f"  kp                  {P.TENDON_KP:,.0f} N/m")
    print(f"  forcerange          [{fr[0,0]:.0f}, {fr[0,1]:.0f}] N  "
          f"-> pull-only; a released tendon goes slack, never pushes")
    print(f"  ctrl = tendon LENGTH in metres (NOT displacement)")
    print(f"  rest ctrl values    " +
          ", ".join(f"M{k+1}={rest[k]:.4f}" for k in range(P.N_MOTORS)))

    hdr("ELASTICITY")
    print(f"  joint stiffness     {P.JOINT_STIFFNESS} N*m/rad   (derived, "
          f"Phase 3 verified forces 9.9 N typical / 31 N peak)")
    print(f"  joint damping       {P.JOINT_DAMPING} N*m*s/rad")
    print(f"  joint armature      {P.JOINT_ARMATURE}  (fictitious; removes the "
          f"actuator buzz, does not bias statics)")
    print(f"  contacts            {'ON' if P.ENABLE_CONTACTS else 'OFF'}  "
          f"(scope is tip-path only)")

    hdr("SANITY CHECKS")
    d2 = mujoco.MjData(model)
    d2.ctrl[:] = rest
    mujoco.mj_forward(model, d2)
    tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
    x, y = d2.site_xpos[tip][0], d2.site_xpos[tip][1]
    ok = abs(x - P.TOTAL_LEN) < 1e-9 and abs(y) < 1e-9
    print(f"  {'PASS' if ok else 'FAIL'}  undeformed tip at "
          f"({x*1000:.6f}, {y*1000:.6f}) mm, expected ({P.TOTAL_LEN*1000:.0f}, 0)")
    axes_ok = np.allclose(model.jnt_axis[:model.njnt], [0, 0, 1])
    print(f"  {'PASS' if axes_ok else 'FAIL'}  all hinge axes +z (planar motion)")
    pull_ok = np.all(fr[:, 1] <= 0) and np.all(fr[:, 0] < 0)
    print(f"  {'PASS' if pull_ok else 'FAIL'}  all actuators pull-only")
    print()


# ── static contact sheet ─────────────────────────────────────────────────────
def _cam(lookat, dist, az=90.0, el=-89.9):
    c = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(c)
    c.lookat[:] = lookat
    c.distance = dist
    c.azimuth = az
    c.elevation = el
    return c


def contact_sheet(model, data, rest, path):
    """Overview + close-ups of the routing detail that is easiest to get wrong."""
    import imageio.v3 as iio

    def settle(u):
        d = mujoco.MjData(model)
        d.ctrl[:] = P.u_to_tendon_lengths(np.array(u), rest)
        for _ in range(1200):
            mujoco.mj_step(model, d)
        return d

    # Frame the bent pose from its own extent so nothing falls off the edge.
    u_bent = [0.003, 0.001, 0.0]
    db = settle(u_bent)
    xs = db.site_xpos[:, 0]
    ys = db.site_xpos[:, 1]
    mid = [(xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2, 0]
    span = 1.35 * max(xs.max() - xs.min(), ys.max() - ys.min())

    views = [
        # top-down: the bending plane, seen face-on
        ("whole arm at rest (top-down)", [0.0, 0.0, 0.0],
         _cam([P.TOTAL_LEN / 2, 0, 0], 0.33)),
        (f"bent, u={[round(v*1000,1) for v in u_bent]} mm", u_bent,
         _cam(mid, span)),
        ("seg1/seg2 boundary: radius steps 3.5 -> 5.0 mm", [0.0, 0.0, 0.0],
         _cam([P.L_SEG[0], 0, 0], 0.055)),
    ]

    # Looking down the arm's axis, all 18 opaque disks stack into a single
    # circle and hide every tendon. Fade the disks for that one view so the
    # routing pattern is actually visible.
    disk_ids = [g for g in range(model.ngeom)
                if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                .startswith(("disk", "base_disk"))]

    frames = []
    with mujoco.Renderer(model, height=560, width=760) as r:
        for label, u, cam in views:
            d = settle(u)
            end_on = "end-on" in label
            if end_on:
                saved = model.geom_rgba[disk_ids].copy()
                model.geom_rgba[disk_ids, 3] = 0.12
            r.update_scene(d, camera=cam)
            frames.append(r.render())
            if end_on:
                model.geom_rgba[disk_ids] = saved
            print(f"    rendered: {label}")

    frames.append(_routing_schematic(frames[0].shape[1], frames[0].shape[0]))
    print("    rendered: routing cross-sections (schematic)")

    top = np.concatenate(frames[:2], axis=1)
    bot = np.concatenate([frames[3], frames[2]], axis=1)
    iio.imwrite(path, np.concatenate([top, bot], axis=0))
    return path


def _routing_schematic(width, height):
    """
    Cross-section of a disk as seen looking down the arm, one panel per segment,
    showing which tendons pass through and at what radius.

    Drawn rather than rendered: looking down the real 3-D model, 18 disks and 6
    tendons stack into an unreadable pile, and perspective makes radii impossible
    to judge by eye. This is the view that can actually be checked against the
    paper.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, P.N_SEG, figsize=(width / 100, height / 100),
                             dpi=100, facecolor="black")
    for j, ax in enumerate(axes):
        ax.set_facecolor("black")
        R = P.DISK_RADIUS * 1000
        ax.add_patch(plt.Circle((0, 0), R, fill=False, ec="0.55", lw=1.6))
        ax.add_patch(plt.Circle((0, 0), P.BACKBONE_RADIUS * 1000,
                                fc="0.4", ec="0.7", lw=1.0))
        for k in range(P.N_MOTORS):
            if P.MOTOR_TO_SEG[k] < j:
                continue                      # already terminated below here
            r = P.tendon_radius(k, j) * 1000
            col = P.SEG_COLORS[P.MOTOR_TO_SEG[k]].split()
            col = (float(col[0]), float(col[1]), float(col[2]))
            ax.plot(0, r, "o", ms=11, color=col, mec="white", mew=1.0, zorder=3)
            ax.annotate(f"M{k+1}  {abs(r):.1f}", (0, r), (0.9, r),
                        color="white", fontsize=8, va="center",
                        textcoords="data")
        ax.set_xlim(-R * 1.25, R * 2.6)
        ax.set_ylim(-R * 1.35, R * 1.35)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"through segment {j+1}", color="white", fontsize=10)

    fig.suptitle("Tendon routing radii (mm) - looking down the arm axis",
                 color="white", fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    if buf.shape[0] != height or buf.shape[1] != width:
        out = np.zeros((height, width, 3), dtype=buf.dtype)
        h, w = min(height, buf.shape[0]), min(width, buf.shape[1])
        out[:h, :w] = buf[:h, :w]
        buf = out
    return buf


# ── interactive ──────────────────────────────────────────────────────────────
# Manual-driving limit, in metres of tendon displacement.
#
# NOT an actuator limit - the model is stable and unsaturated out to +/-20 mm
# (peak 53 N against the 80 N forcerange). The binding constraint is the
# backbone: u = 12 mm bends a segment 120 deg, i.e. a 43 mm curvature radius and
# ~3.5% surface strain at the 1.5 mm backbone radius, which is inside NiTi's
# superelastic range. By 20 mm the segment is at 190 deg and ~5.5% strain, which
# is at the material edge and not a pose this arm would ever be driven to.
#
# Note contacts are disabled, so nothing stops the disks interpenetrating at
# extreme bends - the model will happily show you a physically impossible pose.
U_LIMIT = 0.012


def viewer(model, data, rest, u, u_limit=U_LIMIT):
    """
    Interactive viewer with ANTAGONISTIC PAIR control.

    The MJCF has 6 independent tendons, faithful to the real robot's 6 motors.
    The antagonistic rule (paper Eq. 6: one tendon shortens by dl, its partner
    lengthens by dl) lives in u_to_tendon_lengths(), which the viewer's own
    Control sliders bypass entirely. Driving a single slider therefore makes both
    tendons of that pair fight each other: measured, M1 alone at -3 mm bends
    segment 1 by only 3.65 deg with BOTH actuators saturated at 80 N, versus
    32.56 deg and a slack partner when the pair is driven properly.
    """
    import time
    import mujoco.viewer

    state = {"u": np.asarray(u, float).copy(), "step": 0.0005, "paired": True,
             "report": 0}

    def status():
        ang = np.degrees([np.sum(data.qpos[s*P.LINKS_PER_SEG:(s+1)*P.LINKS_PER_SEG])
                          for s in range(P.N_SEG)])
        tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip")
        x, y = data.site_xpos[tip][0] * 1000, data.site_xpos[tip][1] * 1000
        f = data.actuator_force
        print(f"  u = [{state['u'][0]*1000:+6.2f},{state['u'][1]*1000:+6.2f},"
              f"{state['u'][2]*1000:+6.2f}] mm  |  bend = "
              f"[{ang[0]:+6.1f},{ang[1]:+6.1f},{ang[2]:+6.1f}] deg  |  "
              f"tip = ({x:7.2f},{y:7.2f}) mm  |  max |F| = "
              f"{np.abs(f).max():5.1f} N")

    def key_cb(keycode):
        ch = chr(keycode).upper() if 0 < keycode < 0x110000 else ""
        inc = {"Q": (0, +1), "A": (0, -1),
               "W": (1, +1), "S": (1, -1),
               "E": (2, +1), "D": (2, -1)}
        if ch in inc:
            i, sgn = inc[ch]
            want = state["u"][i] + sgn * state["step"]
            got = float(np.clip(want, -u_limit, u_limit))
            # Say so out loud. Clamping silently is why this looked like the
            # keys had stopped working.
            if abs(want - got) > 1e-12:
                print(f"  u{i+1} is at the {u_limit*1000:.0f} mm limit - "
                      f"raise it with --limit if you want to go further")
            state["u"][i] = got
            state["report"] = 400
        elif ch == "R":
            state["u"][:] = 0.0
            state["report"] = 400
        elif ch == "]":
            state["step"] = min(state["step"] * 2, 0.002)
            print(f"  step size -> {state['step']*1000:.2f} mm")
        elif ch == "[":
            state["step"] = max(state["step"] / 2, 0.000125)
            print(f"  step size -> {state['step']*1000:.3f} mm")
        elif ch == "P":
            status()
        elif ch == "M":
            state["paired"] = not state["paired"]
            print(f"  mode -> {'PAIRED (keys drive u)' if state['paired'] else 'MANUAL (sliders free, pairing NOT enforced)'}")

    data.ctrl[:] = P.u_to_tendon_lengths(state["u"], rest)
    mujoco.mj_forward(model, data)

    print("""
Opening viewer.

  SEGMENT CONTROL - moves each antagonistic PAIR correctly (paper Eq. 6)
    Q / A     segment 1  bend + / -      (M1 pulls, M2 releases equally)
    W / S     segment 2  bend + / -      (M3 / M4)
    E / D     segment 3  bend + / -      (M5 / M6)
    R         reset all to straight
    [ / ]     halve / double the step size
    P         print bend angles, tip pose and tendon forces
    M         toggle PAIRED <-> MANUAL (frees the sliders; pairing not enforced)

  CAMERA
    drag orbit  |  scroll zoom  |  right-drag pan  |  Tab side panels

  While in PAIRED mode this script owns ctrl, so the Control sliders will be
  overwritten each step - that is deliberate, it is what keeps the tendons
  antagonistic. Press M if you want to drive the six tendons raw.
""")
    print(f"  step size {state['step']*1000:.2f} mm, u limit "
          f"+/-{u_limit*1000:.0f} mm  (--limit to change)\n")

    with mujoco.viewer.launch_passive(model, data,
                                      key_callback=key_cb) as v:
        while v.is_running():
            t0 = time.time()
            if state["paired"]:
                data.ctrl[:] = P.u_to_tendon_lengths(state["u"], rest)
            mujoco.mj_step(model, data)
            v.sync()

            if state["report"]:
                state["report"] -= 1
                if state["report"] == 0:
                    status()

            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u", default="0,0,0",
                    help="tendon displacement u1,u2,u3 in mm (default 0,0,0)")
    ap.add_argument("--shot", action="store_true",
                    help="write a static PNG sheet instead of opening a window")
    ap.add_argument("--report", action="store_true", help="report only")
    ap.add_argument("--limit", type=float, default=U_LIMIT * 1000,
                    help="max |u| per segment in mm for manual driving "
                         f"(default {U_LIMIT*1000:.0f}; the model stays stable "
                         f"and unsaturated to ~20)")
    a = ap.parse_args()

    u = [float(v) / 1000.0 for v in a.u.split(",")]

    xml = build_model.write_xml()
    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    rest = P.rest_lengths_from_model(model, data, mujoco)

    print("=" * 72)
    print("  CONTINUUM ARM - MODEL INSPECTION")
    print("=" * 72)
    report(model, data, rest)

    if a.report:
        return
    if a.shot:
        os.makedirs(OUT, exist_ok=True)
        p = contact_sheet(model, data, rest,
                          os.path.join(OUT, "setup_inspection.png"))
        print(f"  sheet -> {p}\n")
        return
    viewer(model, data, rest, u, u_limit=a.limit / 1000.0)


if __name__ == "__main__":
    main()
