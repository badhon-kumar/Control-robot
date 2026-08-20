# MuJoCo_Sim — planar continuum manipulator simulation

**Author:** Badhon Kumar

Physics simulation of the 3-segment tendon-driven continuum manipulator, used as
an **independent plant** for closed-loop tip control and for reproducing a G-code
printing process.

This folder is **self-contained**. It needs Python, the packages in
`requirements.txt`, and nothing else — no sibling folders.

---

## Everything runs through `run.py`

```powershell
python -m pip install -r MuJoCo_Sim/requirements.txt
cd MuJoCo_Sim
python run.py                     # list every command
```

| Command | What it does |
| :--- | :--- |
| `python run.py check all` | run the whole verification suite |
| `python run.py check geometry` | one check: `install` `geometry` `tendons` `calibration` `closed-loop` |
| `python run.py inspect` | structural report + manual-drive viewer |
| `python run.py live` | live closed-loop tracking viewer |
| `python run.py fig6` | reproduce Fig. 6C and 6D from the paper |
| `python run.py build` | regenerate `models/continuum_planar.xml` |
| `python run.py sync` | refresh `controller/` from a sibling `Continuum_v3/` |

Arguments after the command are passed straight through:

```powershell
python run.py live --gcode triangle.gcode --speed 8
python run.py live --no-kalman --autostart
python run.py inspect --report          # report only, no viewer
python run.py inspect --shot            # static PNG contact sheet
python run.py fig6 --steps 400 --paper-limits
python run.py sync --check              # report differences, change nothing
```

Every target is also runnable on its own — `python checks/geometry.py`,
`python tools/run_live.py` — so the dispatcher is a convenience, not a dependency.

## Driving the arm by hand

`python run.py inspect` opens the viewer with **Q/A**, **W/S**, **E/D** bending
one antagonistic tendon pair each, **R** to straighten, **P** to print bend
angles / tip pose / tendon forces.

## Watching the controller drive it

`python run.py live` opens **straight and stationary** with the reference path
already drawn — nothing moves until you press **S**. Browse shapes with **N / B**
(switching preserves run state, so you can look through them all while idle, or
swap mid-run), then **S** to run the one you picked.

**S** start/stop · **N / B** switch shape · **K** Kalman on/off · **R** restart ·
**T** trail · **P** error · **[ ]** speed. The trajectory list, live error and key
legend are drawn inside the viewer.

> MuJoCo's passive viewer has no plotting widget, so time-series plots cannot be
> drawn inside it. `python run.py fig6` renders them with matplotlib instead.

---

## Layout

```text
MuJoCo_Sim/
├── PLAN.md                 phase plan, decisions and results
├── README.md               this file
├── run.py                  single entry point for every command
├── requirements.txt
├── continuum_sim/          the simulation package
│   ├── params.py           single source of geometry / material parameters
│   ├── build_model.py      generates the MJCF from params.py
│   ├── plant.py            MuJoCo plant: step(u) -> tip pose
│   ├── bridge.py           controller <-> plant, plus the UDP publisher
│   ├── calibrate.py        sweeps, model-error stats, stiffness fitting
│   ├── paths.py            every directory, defined once
│   └── vendor.py           controller/ access and integrity checking
├── checks/                 verification, named for what each one tests
│   ├── harness.py          shared pass/fail reporting
│   ├── install.py          toolchain: mujoco, rendering, tendon smoke test
│   ├── geometry.py         arm structure, vendored integrity, PCC error
│   ├── tendons.py          tendon routing and actuation vs the controller
│   ├── calibration.py      plant-vs-PCC model error budget, stiffness
│   └── closed_loop.py      the controller drives the plant, ellipse tracking
├── tools/
│   ├── inspect_model.py    structural report + manual-drive viewer
│   ├── run_live.py         live closed-loop tracking viewer
│   ├── make_fig6.py        reproduces Fig. 6C and 6D
│   └── sync_controller.py  refresh controller/ from a Continuum_v3/
├── controller/             VENDORED copies - do not edit (see its README)
│   ├── continuum_ellipse.py    the controller under test
│   ├── gcode_trajectory.py     G-code parser
│   ├── pose_feedback.py        UDP pose receiver
│   └── MANIFEST.sha256         integrity hashes
├── gcode/                  toolpaths: circle, ellipse, square, triangle
├── models/
│   └── continuum_planar.xml    GENERATED - do not hand-edit
└── outputs/                figures, logs, video
```

Two rules the layout enforces:

- **`continuum_sim/params.py` is the only place geometry is written down.** The
  MJCF is generated from it, and `checks/geometry.py` cross-checks it against the
  controller's own constants so the two cannot silently drift.
- **`continuum_sim/paths.py` is the only place a directory is written down.** It
  creates the output folders on import, so nothing else calls `os.makedirs`.

---

## The vendored controller

`controller/` holds byte-identical copies of three files from `Continuum_v3/`.
They are copies, not rewrites, because the whole point of this project is to test
**the real controller** against a plant that does not share its model.

Two copies can drift, so `checks/geometry.py` step 2 checks both:

- vendored files still match `MANIFEST.sha256` (always);
- if a sibling `Continuum_v3/` exists, that the copies still match it, warning
  loudly if not.

Both checks live in `continuum_sim/vendor.py`, shared with `tools/sync_controller.py`
so the two can never disagree about what "up to date" means.

After editing the original, re-sync:

```powershell
python run.py sync
```

> `continuum_ellipse.py` imports `tkinter` at module level, so a Python build
> without tkinter cannot import it even headlessly. It ships with standard
> CPython on Windows and macOS.

## Adding a shape

Drop a `.gcode` file into `gcode/`. It appears in `run.py live`'s menu on the
next launch — no code change.

---

## Where things stand

Phases 0–4 complete and verified; `python run.py check all` passes 5/5.
Closed-loop tracking of the paper's Traj. 1 ellipse runs at **2.75 mm RMS**
against a plant whose model error is **4.75 mm RMS**.

`python run.py fig6` reproduces panels C and D of the paper's Fig. 6. On this
plant the compensation's benefit shows up almost entirely in attitude
(ψ RMSE 0.1° vs 0.4°) rather than in x/y, because the plant's PCC mismatch is
predominantly an attitude error — see `PLAN.md` for the open Phase 5 concern:
Kalman compensation currently buys only ~3%, because the error is dominated by
lag rather than Jacobian error.
