"""
Live closed-loop run: the controller drives the MuJoCo arm in the viewer.

Shows the reference path and the tip's actual path together, so tracking error
is visible directly rather than only in a plot afterwards.

    python MuJoCo_Sim/run_live.py                    # starts on the ellipse
    python MuJoCo_Sim/run_live.py --speed 8          # 8x real time
    python MuJoCo_Sim/run_live.py --no-kalman
    python MuJoCo_Sim/run_live.py --gcode triangle.gcode

The arm starts STRAIGHT AND STATIONARY with the reference path already drawn, so
nothing moves until S is pressed. N / B browse the ellipse plus every .gcode file
in MuJoCo_Sim/gcode/, and switching preserves the run state - browse as much as
you like while idle, or swap shapes mid-run without relaunching. S stops a
running arm where it is, and reruns a finished one.

The control loop matches check_phase4.py exactly: the controller only reads the
arm once it is quasi-static, which is the assumption the paper's method rests on.
Physics runs continuously between control updates, so the motion you see is the
arm's real transient, not interpolation.

G-code goes through the vendored gcode_trajectory.py unchanged, so every command
that module supports (G0-G4, arcs, splines, units, absolute/relative) works here.
Note it yields a toolpath only - travel moves and extruding moves are not
distinguished, so the trail shows the whole path. Separating them, and rendering
a deposited bead, is Phase 6.

Author: Badhon Kumar
"""

import argparse
import glob
import os
import sys
import time

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "controller"))

from sim import params as P
from sim.plant import ContinuumPlant
from sim import bridge as B
from continuum_ellipse import make_trajectory
from gcode_trajectory import load_gcode_file

# Toolpaths live with the simulation, in MuJoCo_Sim/gcode/. Drop a .gcode file
# there and it appears in the menu on the next launch - no code change needed.
#
# The parser is the vendored copy of the hardware's own gcode_trajectory.py (see
# controller/README.md), so what the simulation follows is exactly what the
# hardware would follow.
GCODE_DIR = os.path.join(HERE, "gcode")

U_LIMIT = 0.012
FRAME = 1.0 / 60.0
TRAIL_MAX = 3000

REF_RGBA = (0.35, 0.75, 1.00, 0.55)
TIP_RGBA = (1.00, 0.45, 0.10, 0.95)


FONT = mujoco.mjtFontScale.mjFONTSCALE_150
G_TL = mujoco.mjtGridPos.mjGRID_TOPLEFT
G_TR = mujoco.mjtGridPos.mjGRID_TOPRIGHT
G_BL = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT


def draw_overlay(v, st, trajs):
    """
    In-viewer menu and status, drawn with the viewer's text-overlay API.

    The passive viewer has no widget API - no buttons or dropdowns can be added
    - so selection stays on the keyboard. This at least puts the list, the
    current selection and the live error inside the window, instead of leaving
    them in the terminal behind it.
    """
    idx = st["idx"]
    marks = "\n".join((">" if i == idx else " ") + f" {i+1}"
                      for i in range(len(trajs)))
    names = "\n".join(n for n, _ in trajs)

    e = np.array(st["err"])
    rms = np.sqrt(np.mean(e ** 2)) * 1000 if len(e) else 0.0
    last = e[-1] * 1000 if len(e) else 0.0
    state = ("READY - press S" if st["hold"] and st["k"] == 0 else
             "HOLD" if st["hold"] else
             "done" if st["done"] else "running")

    stat_l = "step\nerror\nRMS\nKalman\nspeed\nstate"
    stat_r = (f"{st['k']}/{st['n_ctrl']}\n"
              f"{last:.2f} mm\n"
              f"{rms:.2f} mm\n"
              f"{'ON' if st['kalman'] else 'OFF'}\n"
              f"{st['speed']:.1f}x\n"
              f"{state}")

    keys_l = "S\nN / B\nK\nR\nT\n[ / ]"
    keys_r = ("start / stop\nnext / prev shape\nKalman on-off\nrestart\n"
              "trail\nspeed")

    v.set_texts([
        (FONT, G_TL, marks, names),
        (FONT, G_TR, stat_l, stat_r),
        (FONT, G_BL, keys_l, keys_r),
    ])


def _add_sphere(scn, pos, radius, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        np.array([pos[0], pos[1], 0.0]),
        np.eye(3).flatten(),
        np.array(rgba, np.float32),
    )
    scn.ngeom += 1


def discover_trajectories():
    """The paper ellipse, plus every .gcode file in MuJoCo_Sim/gcode/."""
    out = [("Traj. 1 ellipse (paper)", None)]
    for p in sorted(glob.glob(os.path.join(GCODE_DIR, "*.gcode"))):
        out.append((os.path.basename(p), p))
    return out


def build_reference(entry, period, cycles, ctrl_dt_override):
    """
    Turn a trajectory entry into (ref_fn, ref_pts, total_t, ctrl_dt, n_ctrl).

    ctrl_dt defaults to 0.2 s of trajectory time per control update, which is
    what the ellipse uses at T=40 over 200 steps - keeping it equal across
    trajectories means differences in tracking error are due to the SHAPE and
    not to how often the controller was allowed to act.
    """
    label, path = entry
    ctrl_dt = ctrl_dt_override or 0.2

    if path is None:
        ref_fn = make_trajectory("Traj 1", T=period)
        total_t = cycles * period
        ref_pts = [ref_fn(i * period / 240.0)[:2] for i in range(241)]
        label = f"{label}, T={period:.0f} s"
    else:
        prog = load_gcode_file(path)
        for w in prog.warnings:
            print(f"  [gcode warning] {w}")
        dur = prog.duration_s
        if cycles > 1:
            # pose_at() clamps past the end, so repeat by wrapping time. Fine
            # for closed paths; an open path jumps back to its start each time.
            ref_fn = lambda t: prog.pose_at(t % dur)
        else:
            ref_fn = prog.pose_at
        total_t = dur * cycles
        ref_pts = [(x / 1000.0, y / 1000.0) for x, y in prog.xy_points_mm()]
        label = f"{label} ({len(prog.points)} pts, {dur:.1f} s)"

    return ref_fn, ref_pts, total_t, ctrl_dt, max(1, int(total_t / ctrl_dt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speed", type=float, default=6.0,
                    help="playback speed vs real time (default 6)")
    ap.add_argument("--gcode", default=None,
                    help="open on this .gcode file instead of the ellipse; "
                         "a bare name is looked up in MuJoCo_Sim/gcode/")
    ap.add_argument("--autostart", action="store_true",
                    help="begin tracking immediately instead of waiting for S")
    ap.add_argument("--period", type=float, default=40.0,
                    help="ellipse period T in seconds (default 40)")
    ap.add_argument("--cycles", type=float, default=3.0)
    ap.add_argument("--ctrl-dt", type=float, default=None,
                    help="seconds of trajectory time per control update "
                         "(default 0.2)")
    ap.add_argument("--no-kalman", action="store_true")
    a = ap.parse_args()

    import mujoco.viewer

    plant = ContinuumPlant()
    model, data = plant.model, plant.data
    dt = model.opt.timestep

    trajs = discover_trajectories()
    start = 0
    if a.gcode:
        want = os.path.basename(a.gcode)
        match = [i for i, (n, _) in enumerate(trajs) if n == want]
        if not match:
            print(f"G-code file not found: {a.gcode}\nAvailable:")
            for n, _ in trajs[1:]:
                print(f"  {n}")
            return
        start = match[0]

    st = {
        "u": np.zeros(3),
        "kalman": not a.no_kalman,
        "ctrl": None,
        "k": 0,
        "traj_t": 0.0,
        "trail": [],
        "show_trail": True,
        "speed": a.speed,
        "err": [],
        "hold": False,
        "done": False,
        "idx": start,
        "ref_fn": None, "ref_pts": [], "total_t": 0.0,
        "ctrl_dt": 0.2, "n_ctrl": 1, "label": "",
    }

    def load(idx, announce=True, hold=False):
        """
        Select a trajectory. `hold=True` loads it without starting, so the
        reference path is previewed with the arm still straight - that is the
        launch state, so nothing moves until it is asked to.
        """
        st["idx"] = idx % len(trajs)
        (st["ref_fn"], st["ref_pts"], st["total_t"],
         st["ctrl_dt"], st["n_ctrl"]) = build_reference(
            trajs[st["idx"]], a.period, a.cycles, a.ctrl_dt)
        st["label"] = trajs[st["idx"]][0]
        if announce:
            print(f"\n  >> [{st['idx']+1}/{len(trajs)}] {st['label']}"
                  f"   {st['total_t']:.0f} s, {st['n_ctrl']} control steps"
                  + ("   [ready - press S to start]" if hold else ""))
        restart(hold=hold)

    def restart(hold=False):
        plant.reset()
        st["u"] = np.zeros(3)
        st["ctrl"] = B.make_controller(use_kalman=st["kalman"])
        st["k"] = 0
        st["traj_t"] = 0.0
        st["trail"].clear()
        st["err"].clear()
        st["done"] = False
        st["hold"] = hold

    def summary(tag):
        e = np.array(st["err"])
        if len(e) < 5:
            return
        warm = e[len(e) // 5:]
        print(f"  {tag}  {st['label']}  |  Kalman "
              f"{'ON' if st['kalman'] else 'OFF'}  |  RMS "
              f"{np.sqrt(np.mean(warm**2))*1000:.3f} mm  max "
              f"{warm.max()*1000:.3f} mm")

    def key_cb(keycode):
        ch = chr(keycode).upper() if 0 < keycode < 0x110000 else ""
        # Switching preserves the run state: browse freely while idle without
        # anything starting, and swap shapes mid-run without having to restart.
        if ch == "N":
            load(st["idx"] + 1, hold=st["hold"])
        elif ch == "B":
            load(st["idx"] - 1, hold=st["hold"])
        elif ch == "S":
            if st["done"]:
                restart()                      # S after a finished run reruns it
                print("  restarted")
            else:
                st["hold"] = not st["hold"]
                print("  STOPPED - arm holding position (S to resume)"
                      if st["hold"] else "  running")
        elif ch == "K":
            st["kalman"] = not st["kalman"]
            print(f"  Kalman -> {'ON' if st['kalman'] else 'OFF'} (restarting)")
            restart(hold=st["hold"])           # keep whatever state we were in
        elif ch == "R":
            restart(hold=st["hold"])
            print("  restarted")
        elif ch == "T":
            st["show_trail"] = not st["show_trail"]
        elif ch == "]":
            st["speed"] = min(st["speed"] * 1.5, 60.0)
            print(f"  speed -> {st['speed']:.1f}x")
        elif ch == "[":
            st["speed"] = max(st["speed"] / 1.5, 0.25)
            print(f"  speed -> {st['speed']:.1f}x")
        elif ch == "P":
            e = np.array(st["err"])
            if len(e):
                print(f"  step {st['k']}/{st['n_ctrl']}  RMS "
                      f"{np.sqrt(np.mean(e**2))*1000:.3f} mm  last "
                      f"{e[-1]*1000:.3f} mm  max {e.max()*1000:.3f} mm")

    print("\nLive closed-loop tracking\n")
    print("  Available trajectories (switch any time with N / B):")
    for i, (n, _) in enumerate(trajs):
        print(f"    {i+1:2d}. {n}")
    print(f"""
  S       START / stop the arm              N / B   next / previous trajectory
  K       toggle Kalman compensation        R       restart this run
  T       show/hide the tip trail           P       print running error
  [ / ]   slower / faster                   Space   pause physics (viewer)

  blue dots = reference path    orange trail = actual tip path
  speed {a.speed:.1f}x real time, Kalman {'ON' if st['kalman'] else 'OFF'}

{"  Starting immediately (--autostart)."
 if a.autostart else
 "  The arm starts STRAIGHT AND STATIONARY with the reference path shown."
 "\n  Browse shapes with N / B, then press S to run the one you picked."}
""")

    load(start, hold=not a.autostart)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as v:
        settled_for, since_update = 0, 0
        prev_tip = plant.tip_pose()[:2]

        while v.is_running():
            t0 = time.time()
            substeps = max(1, int(st["speed"] * FRAME / dt))

            for _ in range(substeps):
                mujoco.mj_step(model, data)
                since_update += 1

                cur = plant.tip_pose()[:2]
                settled_for = (settled_for + 1
                               if np.hypot(*(cur - prev_tip)) < plant.pos_tol
                               else 0)
                prev_tip = cur

                if len(st["trail"]) == 0 or \
                        np.hypot(*(cur - st["trail"][-1])) > 3e-4:
                    st["trail"].append(cur.copy())
                    if len(st["trail"]) > TRAIL_MAX:
                        st["trail"].pop(0)

                # Control update: only once the arm is quasi-static, with a
                # hard cap so a stubborn transient cannot stall the run.
                ready = settled_for >= plant.stable_steps or since_update > 900
                if ready and not st["done"] and not st["hold"]:
                    pose = plant.tip_pose()
                    ref = np.asarray(st["ref_fn"](st["traj_t"]), float)
                    st["err"].append(float(np.hypot(ref[0] - pose[0],
                                                    ref[1] - pose[1])))
                    u_next = st["ctrl"].compute_control(st["u"], ref, pose)
                    st["u"] = np.clip(u_next, -U_LIMIT, U_LIMIT)
                    data.ctrl[:] = P.u_to_tendon_lengths(st["u"], plant.rest)

                    st["traj_t"] += st["ctrl_dt"]
                    st["k"] += 1
                    settled_for, since_update = 0, 0

                    if st["k"] % 25 == 0:
                        e = np.array(st["err"])
                        print(f"  step {st['k']:4d}/{st['n_ctrl']}   err "
                              f"{e[-1]*1000:6.2f} mm   RMS "
                              f"{np.sqrt(np.mean(e**2))*1000:6.2f} mm")
                    if st["k"] >= st["n_ctrl"]:
                        st["done"] = True
                        print()
                        summary("FINISHED")
                        print("  N next shape  |  R rerun  |  K flip Kalman\n")

            scn = v.user_scn
            scn.ngeom = 0
            for p in st["ref_pts"]:
                _add_sphere(scn, p, 0.0016, REF_RGBA)
            if st["show_trail"]:
                for p in st["trail"]:
                    _add_sphere(scn, p, 0.0020, TIP_RGBA)

            draw_overlay(v, st, trajs)
            v.sync()
            rest = FRAME - (time.time() - t0)
            if rest > 0:
                time.sleep(rest)


if __name__ == "__main__":
    main()
