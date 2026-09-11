"""
Interactive viewer for the paper experiments.

    python run.py figures
    python run.py fig6
    python run.py fig7
    python run.py fig12
    python run.py fig6 --real
    python run.py fig6 --real --save

The viewer is the only place that creates saveable data for these experiments.
Press S to run. In Fig. 12, press P when you want to attach the payload. When a pass finishes, it is kept in memory. Press G to write the
completed pass(es) to outputs/<figure>/, overwriting the previous contents of
that folder. If nothing has finished, G saves nothing.

Author: Badhon Kumar
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer
import numpy as np

from continuum_sim import bridge as B
from continuum_sim import experiments as X
from continuum_sim import figures
from continuum_sim import params as P
from continuum_sim import vendor  # noqa: F401 - puts controller/ on sys.path
from continuum_sim.paths import OUTPUTS
from continuum_sim.plant import ContinuumPlant
from continuum_sim.realism import RealismLayer, PRESETS, describe, preset

U_LIMIT = 0.012
FRAME = 1.0 / 60.0
TRAIL_MAX = 3000

REF_RGBA = (0.35, 0.75, 1.00, 0.55)
TIP_RGBA = (1.00, 0.45, 0.10, 0.95)

CAM_AZIMUTH = 90
CAM_ELEVATION = -89
CAM_LOOKAT = (0.135, 0.0, 0.0)
CAM_DISTANCE = 0.44

FONT = mujoco.mjtFontScale.mjFONTSCALE_150
G_TL = mujoco.mjtGridPos.mjGRID_TOPLEFT
G_TR = mujoco.mjtGridPos.mjGRID_TOPRIGHT
G_BL = mujoco.mjtGridPos.mjGRID_BOTTOMLEFT


def add_sphere(scn, pos, radius, rgba):
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


def reference_points(exp, n=241):
    ts = np.linspace(0.0, exp.duration_s, n)
    return [exp.ref_fn(t)[:2] for t in ts]


def set_payload_mass(plant, mass_kg):
    body_id = mujoco.mj_name2id(plant.model, mujoco.mjtObj.mjOBJ_BODY, "payload")
    if body_id < 0:
        return False
    plant.model.body_mass[body_id] = float(mass_kg)
    mujoco.mj_setConst(plant.model, plant.data)
    return True


def build_plant(exp, payload_kg=0.0):
    # Fig. 12 needs a visible payload body, but it starts with zero mass and is
    # switched on by pressing P so the curve is hold -> manual load -> recovery.
    plant = ContinuumPlant(gravity=exp.gravity, payload_kg=max(0.0, payload_kg))
    if payload_kg > 0:
        set_payload_mass(plant, 0.0)
    return plant


def run_label(exp, use_kalman, payload_i):
    if exp.kind == "disturbance":
        return exp.payload_names[payload_i]
    return "Proposed" if use_kalman else "PCC model"


def make_completed_run(exp, st):
    return figures.make_run(
        st["label"],
        np.asarray(st["t"]),
        np.asarray(st["ref"]),
        np.asarray(st["pose"]),
        meta={
            "kalman": st["kalman"],
            "realism": st["realism_name"],
            "realism_details": describe(st["realism"].config),
            "payload_kg": st["payload_kg"],
            "payload_index": st["payload_i"],
            "payload_drop_s": st["payload_drop_s"],
        },
    )


def save_completed(exp, completed):
    runs = list(completed.values())
    if not runs:
        print("  Nothing saved: finish a run first.")
        return None
    warmup = 0
    outdir = OUTPUTS / exp.key
    return figures.save(exp, runs, outdir, warmup=warmup)


def run_headless(exp, use_kalman, realism_name, realism_config):
    """Run one complete tracking pass without opening the MuJoCo viewer."""
    plant = build_plant(exp)
    target0 = np.asarray(exp.ref_fn(0.0), float)
    u0, u_plant0 = B.preposition_to_pose(
        plant,
        target0,
        dt=exp.ctrl_dt,
        n_steps=max(40, int(round(8.0 / exp.ctrl_dt))),
        realism=None,
        u_limit=U_LIMIT,
    )
    ctrl = B.make_controller(use_kalman=use_kalman)
    log = B.closed_loop(
        plant,
        ctrl,
        exp.ref_fn,
        n_steps=exp.n_steps,
        dt=exp.ctrl_dt,
        u0=u0,
        u_plant0=u_plant0,
        realism=realism_config,
        u_limit=U_LIMIT,
    )
    return figures.make_run(
        run_label(exp, use_kalman, 0),
        log["t"],
        log["ref"],
        log["measured_pose"],
        meta={
            "kalman": use_kalman,
            "realism": realism_name,
            "realism_details": describe(realism_config),
            "prepositioned_to_first_reference": True,
            "payload_kg": 0.0,
            "payload_index": 0,
            "payload_drop_s": None,
        },
    )


def save_headless(exp, realism_name):
    if exp.kind != "tracking":
        raise SystemExit("--save is currently for tracking experiments")
    realism_config = preset(realism_name)
    runs = [
        run_headless(exp, True, realism_name, realism_config),
        run_headless(exp, False, realism_name, realism_config),
    ]
    return figures.save(exp, runs, OUTPUTS / exp.key, warmup=0)


def overlay(viewer, exp, st, completed):
    last = st["last_err"]
    err = np.asarray(st["err_norm"])
    warm = err[len(err) // 5:] if len(err) else err
    rms = np.sqrt(np.mean(warm ** 2)) * 1000.0 if len(warm) else 0.0

    if exp.kind == "disturbance" and not st["payload_on"] and not st["hold"]:
        state = "HOLDING - press P"
    elif st["hold"] and st["k"] == 0:
        state = "READY - press S"
    elif st["hold"]:
        state = "HOLD"
    elif st["done"]:
        state = "DONE"
    else:
        state = "RUNNING"

    saved = st["saved"] or "-"
    complete_names = ", ".join(completed) if completed else "-"
    viewer.set_texts([
        (FONT, G_TL,
         "experiment\nrun\ncompleted\nsaved" + ("\npayload" if exp.kind == "disturbance" else ""),
         f"{exp.figure}: {exp.title}\n{st['label']}\n{complete_names}\n{saved}"
         + (f"\n{st['payload_kg'] * 1000:.1f} g" if exp.kind == "disturbance" else "")),
        (FONT, G_TR,
         "step\nerr x\nerr y\nerr psi\nRMS xy\nspeed\nstate",
         f"{st['k']}/{exp.n_steps}\n"
         f"{last[0] * 1000:+.2f} mm\n"
         f"{last[1] * 1000:+.2f} mm\n"
         f"{np.degrees(last[2]):+.2f} deg\n"
         f"{rms:.2f} mm\n"
         f"{st['speed']:.1f}x\n{state}"),
        (FONT, G_BL,
         "S\nP\nG\nR\nK\n1/2/3\nT\n[ / ]",
         "start / stop\nattach payload\nsave completed\nrestart\nProposed/PCC\npayload mass\ntrail\nspeed"),
    ])


def print_help(exp, speed, realism_name, realism_config):
    print(f"""
{exp.figure}: {exp.title}

  S       start / stop
  P       Fig. 12 only: attach the selected payload now
  G       save completed run(s) to outputs/{exp.key}/
  R       restart current choice
  K       tracking only: toggle Proposed / PCC model and restart
  1/2/3   Fig. 12 only: choose payload and restart
  T       show / hide trail
  [ / ]   slower / faster

  Nothing is written until a run finishes and you press G.
  For Fig. 12: press S, let the arm reach the fixed pose, then press P.
  Saving overwrites outputs/{exp.key}/, so only the latest saved result remains.
  Playback speed starts at {speed:.1f}x.
  Realism: {realism_name} ({describe(realism_config)}).
""")


def main():
    ap = argparse.ArgumentParser(description="Open a paper experiment in MuJoCo.")
    ap.add_argument("experiment", nargs="?", default="list",
                    help="fig6 | fig7 | fig12 | list")
    ap.add_argument("--speed", type=float, default=6.0,
                    help="viewer playback speed multiplier")
    ap.add_argument("--autorun", action="store_true",
                    help="start immediately instead of waiting for S")
    ap.add_argument("--realism", choices=sorted(PRESETS), default="off",
                    help="feedback-loop imperfections preset (default off)")
    ap.add_argument("--save", action="store_true",
                    help="run headless and save the paper-style comparison")
    args = ap.parse_args()

    if args.experiment == "list":
        print("\nAvailable paper experiments:\n")
        for key, exp in X.REGISTRY.items():
            print(f"  {key:6s} {exp.figure:7s} {exp.title}")
        print()
        return

    exp = X.get(args.experiment)
    if args.save:
        save_headless(exp, args.realism)
        return

    ref_pts = reference_points(exp)
    completed = {}

    st = {
        "plant": build_plant(exp, max(exp.payloads_kg) if exp.payloads_kg else 0.0),
        "ctrl": None,
        "realism": RealismLayer(preset(args.realism)),
        "realism_name": args.realism,
        "u": np.zeros(3),
        "u_plant": np.zeros(3),
        "k": 0,
        "traj_t": 0.0,
        "t": [],
        "ref": [],
        "pose": [],
        "err_norm": [],
        "last_err": np.zeros(3),
        "trail": [],
        "show_trail": True,
        "hold": not args.autorun,
        "done": False,
        "kalman": True,
        "speed": args.speed,
        "saved": None,
        "payload_i": 0,
        "payload_kg": exp.payloads_kg[0] if exp.payloads_kg else 0.0,
        "payload_on": False,
        "payload_drop_s": None,
        "label": "",
        "generation": 0,
    }

    def restart(hold=None):
        if hold is None:
            hold = st["hold"]
        payload_kg = exp.payloads_kg[st["payload_i"]] if exp.payloads_kg else 0.0
        st["plant"].reset()
        if exp.kind == "disturbance":
            set_payload_mass(st["plant"], 0.0)
        st["ctrl"] = B.make_controller(use_kalman=st["kalman"])
        st["u"] = np.zeros(3)
        st["u_plant"] = np.zeros(3)
        if exp.kind == "tracking":
            st["u"], st["u_plant"] = B.preposition_to_pose(
                st["plant"],
                exp.ref_fn(0.0),
                dt=exp.ctrl_dt,
                n_steps=max(40, int(round(8.0 / exp.ctrl_dt))),
                realism=None,
                u_limit=U_LIMIT,
            )
        st["realism"].reset(initial_pose=st["plant"].tip_pose(), initial_u=st["u"],
                            initial_applied_u=st["u_plant"])
        st["plant"].data.ctrl[:] = P.u_to_tendon_lengths(st["u_plant"], st["plant"].rest)
        st["k"] = 0
        st["traj_t"] = 0.0
        st["t"].clear()
        st["ref"].clear()
        st["pose"].clear()
        st["err_norm"].clear()
        st["last_err"] = np.zeros(3)
        st["trail"].clear()
        st["hold"] = hold
        st["done"] = False
        st["payload_kg"] = payload_kg
        st["payload_on"] = False
        st["payload_drop_s"] = None
        st["label"] = run_label(exp, st["kalman"], st["payload_i"])
        st["generation"] += 1

    def attach_payload():
        if exp.kind != "disturbance" or st["payload_on"] or st["done"]:
            return
        set_payload_mass(st["plant"], st["payload_kg"])
        st["payload_on"] = True
        st["payload_drop_s"] = st["traj_t"]
        st["hold"] = False
        print(f"  Payload attached at t = {st['payload_drop_s']:.2f} s; compensating.")

    def finish_run():
        run = make_completed_run(exp, st)
        completed[st["label"]] = run
        st["done"] = True
        print(f"\n  Finished: {st['label']}")
        print("  Press G to save, R to rerun, or choose another condition.\n")

    def key_cb(keycode):
        ch = chr(keycode).upper() if 0 < keycode < 0x110000 else ""
        if ch == "S":
            if st["done"]:
                restart(hold=False)
            else:
                st["hold"] = not st["hold"]
        elif ch == "G":
            out = save_completed(exp, completed)
            if out is not None:
                st["saved"] = str(out.relative_to(OUTPUTS.parent))
        elif ch == "P":
            attach_payload()
        elif ch == "R":
            restart()
        elif ch == "K" and exp.kind == "tracking":
            st["kalman"] = not st["kalman"]
            restart()
            print(f"  Running {'Proposed' if st['kalman'] else 'PCC model'}")
        elif ch in ("1", "2", "3") and exp.kind == "disturbance":
            i = int(ch) - 1
            if i < len(exp.payloads_kg):
                st["payload_i"] = i
                restart()
                print(f"  Payload: {st['label']}")
        elif ch == "T":
            st["show_trail"] = not st["show_trail"]
        elif ch == "[":
            st["speed"] = max(0.25, st["speed"] / 1.25)
        elif ch == "]":
            st["speed"] = min(50.0, st["speed"] * 1.25)

    print_help(exp, args.speed, args.realism, st["realism"].config)
    restart(hold=not args.autorun)

    plant = st["plant"]
    model, data = plant.model, plant.data

    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as v:
        v.cam.azimuth = CAM_AZIMUTH
        v.cam.elevation = CAM_ELEVATION
        v.cam.lookat[:] = CAM_LOOKAT
        v.cam.distance = CAM_DISTANCE

        settled_for = 0
        since_update = 0
        prev_tip = plant.tip_pose()[:2]
        seen_generation = st["generation"]

        while v.is_running():
            plant = st["plant"]
            if seen_generation != st["generation"]:
                settled_for = 0
                since_update = 0
                prev_tip = plant.tip_pose()[:2]
                seen_generation = st["generation"]
            t0 = time.time()
            substeps = max(1, int(st["speed"] * FRAME / model.opt.timestep))

            for _ in range(substeps):
                mujoco.mj_step(model, data)
                since_update += 1

                cur = plant.tip_pose()[:2]
                settled_for = (
                    settled_for + 1
                    if np.hypot(*(cur - prev_tip)) < plant.pos_tol
                    else 0
                )
                prev_tip = cur

                if not st["trail"] or np.hypot(*(cur - st["trail"][-1])) > 3e-4:
                    st["trail"].append(cur.copy())
                    if len(st["trail"]) > TRAIL_MAX:
                        st["trail"].pop(0)

                ready = settled_for >= plant.stable_steps or since_update > 900
                if ready and not st["hold"] and not st["done"]:
                    ref = np.asarray(exp.ref_fn(st["traj_t"]), float)
                    true_pose = plant.tip_pose()
                    pose = st["realism"].measure(true_pose)
                    err = pose - ref
                    err[2] = (err[2] + np.pi) % (2.0 * np.pi) - np.pi

                    st["t"].append(st["traj_t"])
                    st["ref"].append(ref.copy())
                    st["pose"].append(pose.copy())
                    st["err_norm"].append(float(np.hypot(err[0], err[1])))
                    st["last_err"] = err

                    u_next = st["ctrl"].compute_control(st["u"], ref, pose)
                    st["u"] = np.clip(u_next, -U_LIMIT, U_LIMIT)
                    st["u_plant"] = np.clip(
                        st["realism"].command_to_plant(st["u"]),
                        -U_LIMIT,
                        U_LIMIT,
                    )
                    data.ctrl[:] = P.u_to_tendon_lengths(st["u_plant"], plant.rest)

                    st["traj_t"] += exp.ctrl_dt
                    st["k"] += 1
                    settled_for = 0
                    since_update = 0

                    if exp.kind == "disturbance":
                        recovery_s = max(exp.ctrl_dt, exp.duration_s - exp.drop_s)
                        if (st["payload_on"]
                                and st["traj_t"] >= st["payload_drop_s"] + recovery_s):
                            finish_run()
                    elif st["k"] >= exp.n_steps:
                        finish_run()

            scn = v.user_scn
            scn.ngeom = 0
            for p in ref_pts:
                add_sphere(scn, p, 0.0016, REF_RGBA)
            if st["show_trail"]:
                for p in st["trail"]:
                    add_sphere(scn, p, 0.0020, TIP_RGBA)

            overlay(v, exp, st, completed)
            v.sync()
            rest = FRAME - (time.time() - t0)
            if rest > 0:
                time.sleep(rest)


if __name__ == "__main__":
    main()
