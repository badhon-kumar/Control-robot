# MuJoCo Simulation Plan — Planar Continuum Manipulator

**Author:** Badhon Kumar
**Created:** August 2026

Physics-based simulation of the 3-segment tendon-driven continuum manipulator, used as an
independent plant for closed-loop tip control and for reproducing a G-code printing process.

Based on Zhai et al. 2025, *"Model-Based Control of a Continuum Manipulator with Online
Jacobian Error Compensation Using Kalman Filtering"*, and on the existing controller in
`controller/continuum_ellipse.py` (vendored from `Continuum_v3/`).

---

## 1. Why this exists

The current `--simulation` mode in `continuum_ellipse.py` is **not a simulation of the robot**.
It is a plot of the controller's own equations:

| Role | What currently provides it |
| :--- | :--- |
| Controller model | PCC forward kinematics + PCC Jacobian |
| "Plant" | The same PCC forward kinematics |
| Visualizer | The same PCC forward kinematics |

Because the plant and the controller share one model, the tracking error is near zero by
construction and the online Kalman Jacobian compensation — the actual contribution of the
paper — has no model error to compensate. The result proves nothing.

MuJoCo supplies an **independent plant**: a discretized elastic backbone driven by real
spatial tendons, with its own dynamics, gravity and friction. It will *not* match PCC exactly.
That mismatch is the whole point: it is what the Kalman compensator is supposed to remove.

**Success = showing that Kalman-compensated tracking beats uncompensated tracking against a
plant that does not share the controller's model.**

---

## 2. Scope (agreed)

| Decision | Choice |
| :--- | :--- |
| Dimensionality | **2D planar** — motion confined to one plane, as the physical manipulator moves in 2 directions |
| Printing fidelity | **Tip path only** — nozzle trace / deposited bead visualization, no material contact physics |
| Deliverables | **Both** — MP4 video of the printing run *and* tracking-error plots |
| Approach | Step by step; each phase has an exit criterion before moving on |

Out of scope for now: 3D bending, material extrusion physics, layer-by-layer stacking,
thermal effects, hardware-in-the-loop.

---

## 3. Robot parameters (from `continuum_ellipse.py`)

| Parameter | Symbol | Value | Source |
| :--- | :--- | :--- | :--- |
| Segments | — | 3 | `L_SEG` |
| Segment length | L₁,L₂,L₃ | 0.09 m each | `L_SEG` |
| Total length | — | 0.270 m | `TOTAL_LEN_MM` |
| Tendon radius | r₁,r₂,r₃ | 0.005, 0.0035, 0.002 m | `R_TENDON` |
| Body diameter | — | 0.013 m | README |
| Spacer disks / segment | — | 5 | `SPACER_DISKS_PER_SEG` |
| End disks / segment | — | 1 | `END_DISKS_PER_SEG` |
| Axial disk pitch | — | 90 / 6 = 15 mm | `disk_layout_mm()` |
| Base disk | — | 1, at s = 0 | `disk_layout_mm()` |
| Motors | M1..M6 | 3 antagonistic pairs | `MOTOR_TO_SEG` |

Note the **tapered tendon radius** (5 → 3.5 → 2 mm). Tendon attachment sites must be placed
at the correct per-segment radius, not a single constant.

### Conventions

- **Units:** MJCF in SI (metres, radians). Convert only at the GUI/CSV boundary.
- **Control input:** `u ∈ ℝ³` — one signed tendon displacement per segment. Positive `u_i`
  pulls the odd motor, the antagonistic partner releases by the same amount.
- **Pose:** `p = [x, y, ψ] ∈ ℝ³`. Jacobian is 3×3 and square — no pseudoinverse
  rank problems in the planar case.
- **Frame:** base at origin, undeformed arm along **+x**, bending in the **x–y** plane,
  all hinge axes along **+z**.

---

## 4. Planned file layout

```text
MuJoCo_Sim/                     SELF-CONTAINED - no sibling folders needed
├── PLAN.md                     this file
├── README.md                   how to run everything
├── run.py                      single entry point for every command
├── requirements.txt
├── continuum_sim/              the simulation package
│   ├── params.py               single source of geometry parameters
│   ├── build_model.py          generates the MJCF from params.py
│   ├── plant.py                MuJoCo plant: step(u) -> tip pose
│   ├── bridge.py               controller <-> plant + UDP publisher
│   ├── calibrate.py            sweeps, error stats, stiffness fitting
│   ├── paths.py                every directory, defined once
│   └── vendor.py               controller/ access + integrity checking
├── checks/                     verification, named for what they test
│   ├── harness.py              shared pass/fail reporting
│   ├── install.py              toolchain            (was check_install)
│   ├── geometry.py             arm structure        (was check_phase1)
│   ├── tendons.py              tendons/actuators    (was check_phase2)
│   ├── calibration.py          model-error budget   (was check_phase3)
│   └── closed_loop.py          closed loop          (was check_phase4)
├── tools/
│   ├── inspect_model.py        structural report + manual-drive viewer
│   ├── run_live.py             live closed-loop tracking viewer
│   ├── make_fig6.py            reproduces Fig. 6C and 6D from the paper
│   └── tools/sync_controller.py      refresh controller/ from a Continuum_v3/
├── controller/                 VENDORED, byte-identical, do not edit
│   ├── continuum_ellipse.py    the controller under test
│   ├── gcode_trajectory.py     G-code parser
│   ├── pose_feedback.py        UDP pose receiver
│   ├── MANIFEST.sha256         integrity hashes
│   └── README.md               provenance and re-sync instructions
├── gcode/                      circle, ellipse, square, triangle
├── models/
│   └── continuum_planar.xml    GENERATED - do not hand-edit
└── outputs/
    ├── figures/
    ├── logs/
    └── video/
```

The checks are named for what they verify rather than for the phase that built
them; the phase numbering below is the build history, and the mapping is in the
tree above. Run them with `python run.py check all`, or individually.

The controller is **vendored** rather than imported from `Continuum_v3/` so this
folder is portable. Copies can drift, so `checks/geometry.py` step 2 verifies the
vendored files against `MANIFEST.sha256`, and additionally against a sibling
`Continuum_v3/` when one is present. See `controller/README.md`.

---

## 5. Phases

### Phase 0 — Environment ✅ COMPLETE

1. ~~`pip install mujoco numpy matplotlib imageio imageio-ffmpeg`~~
2. ~~Verify install~~
3. ~~Write `requirements.txt` with pinned versions~~
4. ~~Work through one minimal MJCF by hand (2 bodies, 1 hinge)~~

**Exit criterion met.** Verified by `python run.py check install` — all 5 checks pass.

Verified environment:

| Component | Version |
| :--- | :--- |
| Python | 3.14.6 (`C:\Python314\python.exe`) |
| mujoco | 3.11.0 |
| numpy | 2.5.1 |
| matplotlib | 3.11.1 |
| imageio / imageio-ffmpeg | 2.37.4 / 0.6.0 |

Results worth carrying forward:

- **Physics validated analytically** — rod released horizontally reaches exactly 180.0°
  peak swing and drops 250.0 mm (= rod length), energy conserved to 0.02% at a 2 ms
  timestep with the default integrator. Phase 1 can safely start at `timestep="0.002"`.
- **Spatial tendons work as needed for Phase 2** — a 4 mm pull on a 2-link chain bent both
  hinges by an equal +26.67°, i.e. naturally constant-curvature. This is the exact
  mechanism the real arm uses, and equal joint angles under equal tendon moment arms is
  what makes the PCC comparison in Phase 3 meaningful.
- **Offscreen rendering works** (640×480 PNG written), so Phase 7 MP4 export is viable
  on this machine with no extra GPU/driver setup.

> Test-design note: an undamped pendulum oscillates, so checking its state at one fixed
> time is meaningless — the first version of this test sampled t = 1.0 s and caught the rod
> swinging back through its start, reporting a misleading 0.5°. The check now tracks the
> extremes over a full period. Apply the same care to Phase 3/5 metrics: log trajectories
> and report min/max/RMS, never a single sampled instant.

---

### Phase 1 — Planar arm model (geometry only, no tendons) ✅ COMPLETE

1. ~~Generate the kinematic chain: 18 links, 6 per segment, 15 mm each~~
2. ~~Each link: hinge about +z with stiffness/damping, backbone geom, disk geom~~
3. ~~Base disk welded to world~~
4. ~~`<site name="tip">` at the distal face of link 18~~
5. ~~Gravity disabled for now~~

**Exit criterion met.** Verified by `python run.py check geometry` — 8/8 checks pass.

Built: `continuum_sim/build_model.py` (generator), `models/continuum_planar.xml` (generated —
do not hand-edit), `continuum_continuum_sim/params.py` (single source of geometry), `checks/geometry.py`.

Model: 20 bodies, 18 hinges (all +z, so motion is provably planar), 37 geoms, 5 sites,
total mass 20.9 g. Undeformed tip lands at exactly (270.000000, 0.000000) mm.

#### Key modelling decision: hinges sit at link *midpoints*

The first build placed each hinge at its link's proximal end — the obvious choice, and
wrong. It leaves link 1 already rotated by the full joint angle while the true arc still
has tangent 0, and that half-angle bias accumulates down the chain:

| | Hinges at proximal end | Hinges centred |
| :--- | :--- | :--- |
| Tip error vs PCC arc @ 60°/seg | **15.006 mm** | **0.437 mm** |
| Convergence order | 1st (÷2 per doubling) | 2nd (÷4 per doubling) |

Centring the hinges turns the tip sum into the trapezoidal rule over the arc. Measured
error now falls **exactly 4.00× per doubling** (0.437 → 0.109 → 0.027 mm at 6/12/24 links
per segment), confirming the construction is right rather than merely tuned.

This mattered: a 15 mm systematic bias would have swamped the very tracking errors
Phase 5 exists to measure. **Any change to link placement must re-run check 6** — a ~2×
ratio instead of ~4× means the hinges have been biased again.

#### Discretization error at the chosen 18 links

| Per-segment bend | Tip error vs continuous PCC arc |
| :--- | :--- |
| 15° | 0.042 mm |
| 30° | 0.154 mm |
| 45° | 0.302 mm |
| 60° | 0.437 mm |
| 90° | 0.463 mm |

The G-code samples work near 15–30°/segment, so expected error is **under 0.16 mm** —
comfortably below the tracking errors of interest. 18 links is enough; 36 buys 0.1 mm for
double the compute.

#### Other Phase 1 findings

- **MJCF nesting limit.** One nested `<body>` per link means 48 links/segment (144 nested
  bodies) overflows the XML parser stack and *hard-crashes the process* (`0xC00000FD`) —
  not a catchable Python exception. 24/segment (72 bodies) is safe; the sensitivity sweep
  is capped there.
- **Integrator is stable** at the chosen stiffness/damping/armature: released from
  45°/segment the arm relaxes to straight in 3.0 s with 0.0000° residual, no divergence
  over 1500 steps.
- **Contacts disabled globally** (`contype/conaffinity = 0`). Scope is tip-path only, so
  no collision is needed; this also removes adjacent-disk self-collision and speeds the
  solve. The Phase 6 print bed will be visual-only.
- **Parameter drift is now guarded.** `checks/geometry.py` step 2 imports
  `continuum_ellipse.py` and asserts `L_SEG`, `R_TENDON` and the disk counts still match.
  That import is clean and `__main__`-guarded (no GUI opens), which also de-risks the
  Phase 4 controller import.
- **Joint stiffness 0.6 N·m/rad is a derived placeholder, not a fit** — derivation is in
  `continuum_continuum_sim/params.py`. Phase 3 replaces it.

---

### Phase 2 — Tendons and actuators ✅ COMPLETE

**Exit criterion met** (revised — see below). `python run.py check tendons`, 8/8 pass.
6 spatial tendons, 6 pull-only position actuators, measured rest lengths 90 / 181.5 / 273 mm.

#### Correction: `R_TENDON` is indexed by segments-back, not by segment

The plan said tendon sites go at "that segment's radius (±5 / ±3.5 / ±2 mm)". That reading
was wrong. `pcc_angles()` in `continuum_ellipse.py` (paper Eqs. 7–9) divides **every**
segment by `r[0]` and uses `r[1]`, `r[2]` only as coupling coefficients:

```python
theta1 = dl1 / r[0]
theta2 = (dl2 - r[1]*theta1) / r[0]
theta3 = (dl3 - r[2]*theta1 - r[1]*theta2) / r[0]
```

So a tendon runs at **5 mm inside its own segment, 3.5 mm one segment back, 2 mm two back**
— the paper's scheme for keeping the three pairs clear of each other ("offsets … adjusted
along the backbone to avoid interaction and friction between tendons"). Measured moment
arms now reproduce this exactly (5.000 / 3.500 / 2.000 mm).

Getting this wrong would have made plant and controller disagree *by construction*, and the
disagreement would have looked like physics.

#### Correction: the exit criterion itself was wrong

"Commanding `u₁` bends segment 1 while segments 2 and 3 stay straight" is false for this
mechanism, and the paper says so: tendons are *"attached only to the end disk of the
respective segment and slide through all preceding disks"* (Fig. 2C). Bending segment 1
shortens every distal tendon's path through it, so holding those tendons at fixed length
forces the segments above to bend back. Isolating segment 1 by 40° actually requires
`u = [3.49, 2.44, 1.40] mm`, not `[u₁, 0, 0]`. With that command, segments 2 and 3 hold to
±0.000°.

#### Model error is real, and its dominant term is understood

Tendon shortening is **not** linear in θ. Measured kinematically (no forces):

| θ₁ | measured dl | linear r·θ | excess | analytic L·δ²/8 |
| :--- | :--- | :--- | :--- | :--- |
| 15° | 1.3303 mm | 1.3090 mm | 0.0213 | 0.0214 mm |
| 30° | 2.7028 mm | 2.6180 mm | 0.0848 | 0.0857 mm |
| 45° | 4.1169 mm | 3.9270 mm | 0.1899 | 0.1928 mm |
| 60° | 5.5718 mm | 5.2360 mm | 0.3358 | 0.3427 mm |

Agreement with the closed-form chord-vs-arc term is within 2%, so this is understood, not
a numerical accident. It is **physical, not a discretization artifact**: the paper's
Fig. 2B describes *"the revolute joint between adjacent disks"*, so the real tendons also
run as straight chords between disk holes. PCC's smooth-arc `dl = r·θ` is the
approximation — this is error the Kalman filter should absorb.

Net effect: the plant deviates from `pcc_angles()` by up to **5.21°** of segment angle.

Other decisions: actuators are `forcerange = [-Fmax, 0]` so a released tendon goes slack at
zero force rather than pushing; the radial step at each segment boundary is placed at a
single axial station (a second hole in the same disk) so every span keeps a constant
radius — spreading it across a span costs 2.4% of the own-segment moment arm.

---

### Phase 3 — Calibration ✅ COMPLETE

**Exit criterion met.** `python run.py check calibration`, 6/6 pass.
Built `continuum_continuum_sim/plant.py`, `continuum_continuum_sim/calibrate.py`.

#### The plan's objective was wrong and has been changed

PLAN.md said *"fit joint stiffness … to minimise the discrepancy"* against PCC. **That
target is self-defeating.** The MuJoCo plant exists precisely because it does not share the
controller's model; driving the discrepancy to zero rebuilds the section-1 tautology with
more machinery, and would make Phase 5 meaningless.

Changed to: justify stiffness **physically**, then *measure* the mismatch and report it as
the model-error budget. `fit_stiffness()` exists in `continuum_continuum_sim/calibrate.py` and fits against
**measured tip poses** — the right tool the day hardware data exists (open question 2). It
is deliberately not pointed at PCC-generated targets.

#### Model-error budget (the Phase 5 input)

| Metric | Tip position | Tip attitude |
| :--- | :--- | :--- |
| RMS | **4.75 mm** | 3.82° |
| mean | 3.24 mm | — |
| max | 13.69 mm | 8.29° |

| Bend magnitude (sum of \|u\|) | mean error | max error |
| :--- | :--- | :--- |
| 0–4 mm | 0.475 mm | 1.695 mm |
| 4–8 mm | 1.769 mm | 2.108 mm |
| 8–13 mm | 6.345 mm | 13.685 mm |

Error grows with bend, consistent with the θ²-scaling chord term above.

#### Stiffness is not the knob

An 8× stiffness change moves the mismatch by only **21.8%** (3.58 → 4.43 mm), confirming
the mismatch is kinematic (tendon chords + actuator compliance), not elastic. Stiffness is
instead justified by force: spanning the workspace demands **9.9 N typical / 31 N peak**,
realistic for a 13 mm tendon-driven arm.

> **Saturation trap.** The first run reported a 168% stiffness sensitivity. That was a
> binding 60 N `forcerange`, not physics — the true figure is 8.7%. A force clamp does not
> announce itself, it just changes the answer. `TENDON_FORCE_MAX` is now 80 N and
> `checks/calibration.py` tests for saturation explicitly.

#### Gravity: staying OFF

Out-of-plane gravity `(0,0,-g)` shifts the tip by **0.000000 mm** — exactly as it must,
since a z-force exerts no torque about a z hinge. In-plane gravity `(0,-g,0)` would add up
to 4.6 mm of sag. For the Phase 6 printing scenario the arm bends in a horizontal plane
over the bed, so gravity is out-of-plane and its correct planar contribution is zero.
**If Prof. Cao confirms a vertical bending plane, set `params.GRAVITY = (0,-9.81,0)` and
re-run.** A planar model cannot represent out-of-plane sag at all.

---

### Phase 4 — Controller bridge ✅ COMPLETE

**Exit criterion met.** `python run.py check closed-loop`, 5/5 pass. Built `continuum_continuum_sim/bridge.py`.

- **Regulation:** 42.43 mm → 0.0000 mm, reaching 1 mm by step 10.
- **Ellipse tracking:** 200 steps, no divergence, **2.75 mm RMS** after warm-up.
- **100% of steps reach quasi-static equilibrium**, as the paper's assumption requires.
- **UDP path verified live:** `pose_feedback.py` receives and parses the plant's packets,
  so the existing GUI can be driven by MuJoCo with **no changes to `Continuum_v3`**.

#### Two traps found here, both of which would have corrupted Phase 5

**1. Settling was measured on the wrong observable.** `max|qvel|` plateaus at ~0.25 rad/s
and never decays, so only 14% of steps counted as "settled" — for an arm that had actually
stopped. Equilibrium is now judged on **tip displacement**, the observable the controller
consumes and the one that genuinely converges.

**2. That plateau was a real 75 µm/step tip buzz.** The stiff actuator mode is
ω ≈ √(kp·r²/armature) ≈ 790 rad/s, giving ω·dt ≈ 1.58 at a 2 ms timestep — past clean
integration. It was injecting 75 µm of noise straight into the pose the Kalman filter reads.
`JOINT_ARMATURE` 1e-5 → **1e-4** removes it entirely at zero compute cost. Verified not to
bias statics: armature 1e-4/4e-4/1e-3 at dt=2 ms and 1e-5 at dt=0.5 ms all settle to the
same tip y = 97.8660 mm; the buzzing case reported 97.8921 mm, so the 26 µm gap was the
artifact, not the fix.

> Also fixed: a fresh `MjData` has `ctrl = 0`, which for a position actuator on tendon
> *length* commands a ~20 mm pull, not a released arm. This silently turned Phase 1's
> free-relaxation test into a hard-load test the moment Phase 2 added tendons. Any new test
> creating raw `MjData` must set `ctrl = rest` first.

#### Open concern for Phase 5

Kalman ON gives **2.748 mm** RMS vs OFF **2.830 mm** — only a **3%** improvement. The
project's stated success criterion needs better than that. Phase 5 must address it before
concluding anything: retune `gamma`/`beta`/`alpha` against this plant (they were tuned
against a PCC plant), and note the error is dominated by a periodic *lag* term — the control
law `u += β·J⁺e` has no feedforward, so it trails a moving reference regardless of Jacobian
quality. A slower trajectory or a feedforward term may be needed for the comparison to
isolate what the Kalman filter actually contributes.

---

### Phase 5 — Reproduce the paper (ellipse tracking)

1. Implement the paper's Traj. 1 ellipse reference with horizontal attitude, matching the
   `TrajectoryControlPanel` parameters (`Period T`, `Cycles`).
2. Run three conditions on the identical trajectory and seed:
   - **A:** PCC-only control, no Kalman compensation (`use_kalman=False`)
   - **B:** PCC + Kalman Jacobian compensation (`use_kalman=True`)
   - **C:** optional — B with added measurement noise, for robustness
3. Log per-step: reference pose, actual pose, error norm, `u`, and the Kalman diagnostics
   already exposed (`last_mJ`, `last_eJ`, `last_cJ`, `last_K_norm`).
4. Produce plots: tracking error vs time, x–y path overlay (reference vs actual), and a
   summary table of RMS / max error per condition.
5. Sanity-check the controller gains under the new plant — `gamma`, `beta`, `alpha`,
   `sigma_p`, `sigma_psi` were tuned against a PCC plant and will likely need retuning.

**Exit criterion:** condition B measurably outperforms condition A on RMS tip error.
If it does not, the finding is still reportable — but first check calibration, gains, and
the quasi-static settling time before concluding anything.

---

### Phase 6 — Reproduce the printing process

1. Reuse the existing G-code parser (`gcode_trajectory.py`, `load_gcode_file`) — do not
   rewrite it. Start with `sample_square_path.gcode`, then `sample_circle_path.gcode`.
2. Attach a nozzle body/geom at the tip site so it is visually obvious the arm is a toolhead.
3. Add a print bed plane to the scene for visual reference.
4. Track extrusion state per G-code segment (`G0` = travel, `G1`/`G2`/`G3` = extrude).
5. **Bead rendering:** append a small sphere/capsule geom at the tip position each timestep
   while extruding. Pre-allocate a fixed pool of geoms in the MJCF (MuJoCo models are static
   after compile) or draw the trail via the viewer's user-scene geoms.
6. Log the deposited path so print accuracy can be measured against the commanded G-code
   path — this is the printing-quality metric.

**Exit criterion:** the arm traces the square and circle paths, the deposited bead is visible
and follows the nozzle, and commanded-vs-deposited deviation is quantified.

---

### Phase 7 — Deliverables

1. **Video:** offscreen render to MP4 via `mujoco.Renderer` + `imageio`. Target ~30 fps,
   a fixed informative camera angle, showing one full print of the square and the circle.
2. **Plots:**
   - tracking error vs time, Kalman on vs off
   - reference vs actual x–y path overlay
   - MuJoCo-vs-PCC model mismatch from Phase 3
   - commanded-vs-deposited print path
3. **README.md** for this folder: install, how to run each experiment, what each output is.
4. **Report section:** method (discretized-link approximation, calibration procedure),
   results, and an explicit limitations paragraph.

---

## 6. Known risks

| Risk | Mitigation |
| :--- | :--- |
| MuJoCo has no true Cosserat rod; 18 rigid links is an approximation | Standard practice — state it explicitly in the report; check link-count sensitivity (12 vs 18 vs 24) |
| Stiff tendons + position actuators can go numerically unstable | Reduce `timestep`, raise damping, or switch solver; validate with a settling test |
| Controller gains tuned against a PCC plant may not suit MuJoCo | Retune in Phase 5; log the gains actually used |
| Quasi-static assumption in the paper | Keep trajectory periods long; verify the arm settles between control steps |
| Tendon friction is absent by default | Note as a limitation; optionally add tendon `frictionloss` later |
| Scope creep into 3D / material physics | Explicitly deferred — see Section 2 |

---

## 7. Open questions for Prof. Cao

1. Should the planar model include gravity in-plane, or is the physical arm horizontal
   (gravity out-of-plane, negligible for planar motion)?
2. Is calibrating MuJoCo against the *PCC model* acceptable, or should it be calibrated
   against *measured hardware data* if any exists?
3. For the printing deliverable, is a single-layer 2D trace sufficient, or is a multi-layer
   visualization expected later?

---

## 8. Progress

- [x] Phase 0 — Environment *(complete — `checks/install.py`, 5/5 pass)*
- [x] Phase 1 — Planar arm model *(complete — `checks/geometry.py`, 8/8 pass)*
- [x] Phase 2 — Tendons and actuators *(complete — checks/tendons.py, 8/8)*
- [x] Phase 3 — Calibration *(complete — checks/calibration.py, 6/6)*
- [x] Phase 4 — Controller bridge *(complete — checks/closed_loop.py, 5/5)*
- [ ] Phase 5 — Ellipse tracking reproduction
- [ ] Phase 6 — Printing process reproduction
- [ ] Phase 7 — Deliverables
