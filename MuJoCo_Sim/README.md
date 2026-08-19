# MuJoCo_Sim — planar continuum manipulator simulation

**Author:** Badhon Kumar

Physics simulation of the 3-segment tendon-driven continuum manipulator, used as
an **independent plant** for closed-loop tip control and for reproducing a G-code
printing process.

This folder is **self-contained**. It needs Python, the packages in
`requirements.txt`, and nothing else — no sibling folders.

---

## Install and check

```powershell
python -m pip install -r MuJoCo_Sim/requirements.txt
python MuJoCo_Sim/check_install.py
```

## Look at the model

```powershell
python MuJoCo_Sim/inspect_model.py            # report + interactive viewer
python MuJoCo_Sim/inspect_model.py --report   # structural report only
python MuJoCo_Sim/inspect_model.py --shot     # static PNG contact sheet
```

Bend it by hand with **Q/A**, **W/S**, **E/D** (one antagonistic pair each),
**R** to straighten, **P** to print bend angles / tip pose / tendon forces.

## Watch the controller drive it

```powershell
python MuJoCo_Sim/run_live.py                        # opens on the paper ellipse
python MuJoCo_Sim/run_live.py --gcode triangle.gcode
python MuJoCo_Sim/run_live.py --speed 15 --no-kalman
python MuJoCo_Sim/run_live.py --autostart            # skip the ready state
```

The arm opens **straight and stationary** with the reference path already drawn —
nothing moves until you press **S**. Browse shapes with **N / B** (switching
preserves the run state, so you can look through them all while idle, or swap
mid-run), then **S** to run the one you picked.

**S** start/stop · **N / B** switch shape · **K** Kalman on/off · **R** restart ·
**T** trail · **P** error · **[ ]** speed. The trajectory list, live error and key
legend are drawn inside the viewer.

## Verify each phase

```powershell
python MuJoCo_Sim/check_phase1.py     # model geometry
python MuJoCo_Sim/check_phase2.py     # tendons and actuators
python MuJoCo_Sim/check_phase3.py     # calibration / model-error budget
python MuJoCo_Sim/check_phase4.py     # closed loop
```

---

## Layout

```text
MuJoCo_Sim/
├── PLAN.md                 phase plan, decisions and results
├── README.md               this file
├── requirements.txt
├── check_install.py        Phase 0 - toolchain
├── check_phase1..4.py      per-phase verification
├── inspect_model.py        structural report + manual-drive viewer
├── run_live.py             live closed-loop tracking viewer
├── sync_controller.py      refresh controller/ from a Continuum_v3/
├── controller/             VENDORED copies - do not edit (see its README)
│   ├── continuum_ellipse.py    the controller under test
│   ├── gcode_trajectory.py     G-code parser
│   ├── pose_feedback.py        UDP pose receiver
│   └── MANIFEST.sha256         integrity hashes
├── gcode/                  toolpaths: circle, ellipse, square, triangle
├── models/
│   ├── build_model.py      generates the MJCF from sim/params.py
│   └── continuum_planar.xml    GENERATED - do not hand-edit
├── sim/
│   ├── params.py           single source of geometry / material parameters
│   ├── plant.py            MuJoCo plant: step(u) -> tip pose
│   ├── bridge.py           controller <-> plant, plus the UDP publisher
│   └── calibrate.py        sweeps, model-error stats, stiffness fitting
└── outputs/                figures, logs, video
```

---

## The vendored controller

`controller/` holds byte-identical copies of three files from `Continuum_v3/`.
They are copies, not rewrites, because the whole point of this project is to test
**the real controller** against a plant that does not share its model.

Two copies can drift, so `check_phase1.py` step 2 checks both:

- vendored files still match `MANIFEST.sha256` (always);
- if a sibling `Continuum_v3/` exists, that the copies still match it, warning
  loudly if not.

After editing the original, re-sync:

```powershell
python MuJoCo_Sim/sync_controller.py
```

> `continuum_ellipse.py` imports `tkinter` at module level, so a Python build
> without tkinter cannot import it even headlessly. It ships with standard
> CPython on Windows and macOS.

## Adding a shape

Drop a `.gcode` file into `gcode/`. It appears in `run_live.py`'s menu on the
next launch — no code change.

---

## Where things stand

Phases 0–4 complete and verified. Closed-loop tracking of the paper's Traj. 1
ellipse runs at **2.75 mm RMS** against a plant whose model error is **4.75 mm
RMS**. See `PLAN.md` for the full record, including the open Phase 5 concern:
Kalman compensation currently buys only ~3%, because the error is dominated by
lag rather than Jacobian error.
