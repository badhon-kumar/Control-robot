# What I Built, and How — a plain-language guide

**For:** preparing to explain this work to Prof. Cao
**Covers:** `Continuum_v3/` (the control software) and `MuJoCo_Sim/` (the physics simulation)

Read sections 1–4 to understand the story. Section 9 is the 60-second version.
Section 10 is a Q&A rehearsal. Section 11 is the live demo.

---

## 1. The one-sentence summary

> I built a physics simulation of the continuum robot in MuJoCo that acts as an
> **independent test bench** for the paper's controller — so that for the first time
> the controller is being tested against a robot that does **not** share its own
> assumptions.

If you say nothing else, say that. Everything below is the detail behind it.

---

## 2. Background: what the robot and the paper are

### 2.1 The robot

A **tendon-driven continuum manipulator** — think of an elephant trunk or a snake
arm. No rigid joints, no elbows. It is a flexible rod that bends when you pull
cables (tendons) running inside it.

| Property | Value |
| :--- | :--- |
| Segments | 3 |
| Length per segment | 90 mm (270 mm total) |
| Body diameter | 13 mm |
| Motors | 6, in 3 antagonistic pairs (M1/M2, M3/M4, M5/M6) |
| Disks per segment | 5 spacer + 1 end disk (+ 1 base disk) |
| Tendon routing radius | 5 mm / 3.5 mm / 2 mm |

**Antagonistic pair** = two cables on opposite sides of one segment. If M1 pulls
by 3 mm, M2 releases by 3 mm, and segment 1 bends. So although there are 6 motors,
there are really only **3 independent commands** — call them `u = [u₁, u₂, u₃]`,
in millimetres of cable displacement, one per segment.

The tip of the arm is described by **3 numbers**: `p = [x, y, ψ]` — position in
the plane plus the tip's pointing angle. So 3 inputs, 3 outputs — a **square**
problem, which is convenient.

### 2.2 The paper (Zhai et al. 2025)

*"Model-Based Control of a Continuum Manipulator with Online Jacobian Error
Compensation Using Kalman Filtering"*, Cyborg and Bionic Systems, 2025.
Yujia Zhai, Jihao Xu, Hangjie Mo, Chunqi Zhang, Dong Sun (City University of Hong Kong).

The paper's argument, in plain terms:

1. **Model-based control** of a continuum robot uses the *PCC model* (piecewise
   constant curvature — assume each segment bends into a perfect circular arc).
   It is fast and simple, but it is **wrong** in reality: real rods have friction,
   the cables run as straight chords between disks, the material is not ideal.
2. **Data-driven control** (neural networks etc.) learns the real behaviour, but
   needs a training dataset collected in advance.
3. **The paper's idea = do both.** Start from the PCC model, then use a **Kalman
   filter running online** to estimate how wrong the model is *right now*, and
   correct it on the fly. No pre-training, no dataset.

That "how wrong the model is" quantity is an error in the **Jacobian**.

### 2.3 What a Jacobian is (say it this way)

The Jacobian `J` is a 3×3 table that answers one question:

> *If I pull cable 1 by 1 mm, how far does the tip move in x, in y, and how much
> does the tip angle change?* — and the same for cables 2 and 3.

In symbols: `Δp = J · Δu`. Tip motion = Jacobian × cable motion.

If you know `J`, you can invert the question: *I want the tip to move HERE, so how
much cable do I pull?* That is the whole controller:

```
u_next = u + β · J† · (reference_pose − measured_pose)
```

- `(reference − measured)` = the error you want to remove
- `J†` = damped pseudo-inverse of J — "inverting" the table, with a damping term
  `α` so it stays stable
- `β` = gain, how aggressively you correct (0.5 by default — half-steps, for safety)

**The catch:** if `J` comes from the PCC model and the real robot's true `J` is
different, every correction is aimed slightly in the wrong direction.

### 2.4 The paper's contribution, simply

Define `δJ = J_true − J_model` — the Jacobian error. That is 9 unknown numbers
(a 3×3 table flattened into a 9-vector called `ξ`).

Every control step gives you a free measurement: you commanded `Δu` and you
observed the tip actually move `Δp`. That gives 3 equations. 3 equations, 9
unknowns — not enough on its own. So a **Kalman filter** accumulates the
information over many steps, with a **fading factor γ = 0.5** so old measurements
decay away (because as the arm moves, the true Jacobian changes too).

Then a **constraint step** (`σ_p = 0.35`, `σ_ψ = 2.5`) limits how much the
estimated Jacobian is allowed to jump between consecutive steps, so sensor noise
cannot make the controller lurch. That constraint is the paper's second
contribution.

All of this is in `Continuum_v3/continuum_ellipse.py`, class
`KalmanJacobianController` — `_kalman_step()` is Eqs. 12–19, `_apply_constraints()`
is Eq. 20, `compute_control()` is Eqs. 21–22.

---

## 3. The problem I found — this is the heart of the work

The existing software (`Continuum_v3/continuum_ellipse.py`) had a `--simulation`
mode. It looked like it worked: the tracking error was almost zero, beautiful plots.

**But it proved nothing.** Look at what was providing each role:

| Role | What provided it |
| :--- | :--- |
| The controller's internal model | PCC forward kinematics + PCC Jacobian |
| The "robot" being controlled | **the same** PCC forward kinematics |
| The visualizer | **the same** PCC forward kinematics |

The controller was being tested against a copy of its own equations. Of course the
error was near zero — it is a **tautology**, like marking your own exam with your
own answer key.

Worse: the paper's entire contribution is the Kalman filter that removes **model
error**. In that setup there *was* no model error. So the one thing the software was
supposed to demonstrate was the one thing it could not possibly demonstrate.

**That is why the MuJoCo simulation exists.** MuJoCo gives an *independent plant*:
a real physics engine, with its own elastic backbone, its own cables, its own
forces. It will **not** agree with PCC — and that disagreement is exactly the thing
the Kalman filter is supposed to eat.

> **Success criterion:** show that Kalman-compensated tracking beats uncompensated
> tracking, against a plant that does not share the controller's model.

This framing is the single most important thing to be able to defend. If Prof. Cao
asks *"why simulate at all?"* — the answer is not "to make a video". It is: **the
previous simulation could not falsify anything, so it could not validate anything.**

---

## 4. What I actually built

Two folders. Keep them straight:

### `Continuum_v3/` — the control software (existing, was already there)
- `continuum_ellipse.py` (3,600 lines) — GUI, PCC kinematics, the Kalman Jacobian
  controller, RS-485 motor driver for the real LKM motors, 3D visualizer,
  manual jog, CSV sequencer, G-code trajectory panel.
- `gcode_trajectory.py` — a G-code parser (G0–G4, arcs, splines, units,
  absolute/relative), so the arm can follow a printing toolpath.
- `pose_feedback.py` — receives measured tip pose over UDP as JSON (from a camera
  or EM tracker), 4th-order low-pass filter.

### `MuJoCo_Sim/` — the physics simulation (this is the new work)

```
MuJoCo_Sim/
├── PLAN.md              the full engineering record — phases, decisions, results
├── README.md            how to run everything
├── check_install.py     Phase 0 verification
├── check_phase1..4.py   one verification script per phase
├── inspect_model.py     look at the model, bend it by hand with the keyboard
├── run_live.py          watch the controller drive the arm, live
├── sync_controller.py   re-copy the controller from Continuum_v3
├── controller/          VENDORED byte-identical copies of the 3 controller files
├── gcode/               circle, ellipse, square, triangle toolpaths
├── models/build_model.py  writes the MJCF XML from the parameters
├── sim/params.py        single source of every geometry/material number
├── sim/plant.py         the plant: step(u) → tip pose
├── sim/bridge.py        controller ↔ plant, plus a UDP publisher
└── sim/calibrate.py     sweeps, model-error statistics, stiffness fitting
```

**Why the controller is *vendored* (copied, not rewritten):** the whole point is
to test *the real controller*. If I rewrote it, I would be testing my rewrite.
The copies are checked against SHA-256 hashes (`MANIFEST.sha256`) on every run, and
against the original `Continuum_v3/` folder when it is present, so the two cannot
silently drift apart.

**How the physics model works:** MuJoCo has no true flexible-rod (Cosserat)
element, so a continuum arm is approximated as a chain of **18 short rigid links**
(6 per segment, 15 mm each — one per physical disk), connected by **hinges with
torsional springs**. A stiff spring chain behaves like an elastic rod. Then 6
**spatial tendons** are routed through sites at the correct radius on every disk,
driven by 6 **position actuators** that command cable length.

Nothing in this model comes from the PCC equations. It bends because of forces.

---

## 5. Phase by phase — what was done and what was learned

I worked in 7 phases, each with a written **exit criterion** and an automated
verification script. **Phases 0–4 are complete and all their checks pass.**
Phases 5–7 remain.

### Phase 0 — Environment ✅ (`check_install.py`, 5/5)

Installed MuJoCo 3.11 / numpy / matplotlib / imageio on Python 3.14.6, and
validated the physics analytically before trusting it: a rod released
horizontally reaches exactly 180.0° peak swing, drops exactly its own length,
and conserves energy to **0.02%** at a 2 ms timestep.

Also verified early that spatial tendons produce *naturally constant curvature* —
a 4 mm pull on a 2-link chain bent both hinges by an equal +26.67°. That matters,
because it means comparing against PCC later is a fair comparison.

> **Lesson recorded (and it recurs):** the first version of the pendulum test
> sampled the state at one fixed time, t = 1.0 s, and caught the rod swinging back
> through its start — reporting a misleading 0.5°. **Never judge a dynamic system
> from one sampled instant.** Log the whole trajectory and report min/max/RMS.

### Phase 1 — The arm geometry ✅ (`check_phase1.py`, 8/8)

Built the 18-link chain: 20 bodies, 18 hinges (all about +z, so motion is
*provably* planar), 37 geoms, total mass 20.9 g. Undeformed tip lands at exactly
(270.000000, 0.000000) mm.

**The key modelling decision — hinges at link *midpoints*.** My first build put
each hinge at the *start* of its link, which is the obvious choice and is **wrong**:
it leaves link 1 already rotated by the full joint angle while the true arc still
has zero tangent, and that half-angle bias accumulates down the chain.

| | Hinges at start | Hinges centred |
| :--- | :--- | :--- |
| Tip error vs true arc at 60°/seg | **15.006 mm** | **0.437 mm** |
| Convergence order | 1st (÷2 per doubling) | 2nd (÷4 per doubling) |

Centring the hinges makes the tip calculation a **trapezoidal rule** over the arc.
The measured error now falls **exactly 4.00× per doubling** of link count
(0.437 → 0.109 → 0.027 mm at 6/12/24 links per segment). That exact 4.00× is the
proof the construction is *correct*, not merely *tuned* — you cannot fake
second-order convergence.

This mattered enormously: a 15 mm systematic bias would have completely swamped the
2–5 mm tracking errors the whole project exists to measure.

**Discretization error at 18 links:** 0.042 mm at 15°/segment, 0.154 at 30°,
0.437 at 60°. The G-code paths work near 15–30°, so the expected geometric error is
**under 0.16 mm** — negligible relative to what we are measuring. 18 links is enough.

Other findings: nesting more than ~24 links/segment **hard-crashes** the XML parser
(stack overflow, `0xC00000FD`, not a catchable exception); contacts are disabled
globally since the scope is tip-path only; joint stiffness 0.6 N·m/rad started as a
*derived* placeholder (from EI/ds, implying E ≈ 2.3 GPa — a plausible polymer
backbone), not a guess.

### Phase 2 — Tendons and actuators ✅ (`check_phase2.py`, 8/8)

Added 6 spatial tendons and 6 pull-only position actuators. Measured rest lengths:
90 / 181.5 / 273 mm.

**Correction 1 — what the three radii actually mean.** I initially read
"5 / 3.5 / 2 mm" as "segment 1's tendon at 5 mm, segment 2's at 3.5 mm...". That is
wrong. The controller's own equations (paper Eqs. 7–9) say otherwise:

```python
theta1 = dl1 / r[0]
theta2 = (dl2 - r[1]*theta1) / r[0]
theta3 = (dl3 - r[2]*theta1 - r[1]*theta2) / r[0]
```

Every segment is divided by `r[0]`. So the radii are indexed by **how many segments
back** the cable currently is: a tendon runs at **5 mm inside its own segment,
3.5 mm one segment back, 2 mm two segments back**. That is the paper's scheme for
keeping the three cable pairs from rubbing against each other (Fig. 2C). My measured
moment arms now reproduce it exactly: 5.000 / 3.500 / 2.000 mm.

Getting this wrong would have made the plant and the controller disagree **by
construction** — and that disagreement would have looked like physics.

**Correction 2 — my own exit criterion was wrong.** I had written "commanding u₁
should bend segment 1 while segments 2 and 3 stay straight." That is false for this
mechanism, and the paper says so: the tendons *"are attached only to the end disk of
the respective segment and slide through all preceding disks"*. Bending segment 1
shortens the path of every cable passing through it, so holding those cables at fixed
length **forces the upper segments to bend back**. Isolating segment 1 by 40°
actually requires `u = [3.49, 2.44, 1.40] mm`. With that, segments 2 and 3 hold to
±0.000°.

**Model error is real and I understand where it comes from.** Cable shortening is
**not** linear in bend angle:

| θ₁ | measured Δl | linear r·θ | excess | closed form L·δ²/8 |
| :--- | :--- | :--- | :--- | :--- |
| 15° | 1.3303 mm | 1.3090 mm | 0.0213 | 0.0214 mm |
| 30° | 2.7028 | 2.6180 | 0.0848 | 0.0857 |
| 45° | 4.1169 | 3.9270 | 0.1899 | 0.1928 |
| 60° | 5.5718 | 5.2360 | 0.3358 | 0.3427 |

Agreement with the analytic chord-vs-arc term is within **2%**, so this is understood
physics, not a numerical accident. And it is **physical, not a discretization
artifact**: the paper's Fig. 2B says there is a revolute joint between adjacent disks,
so the real cables also run as straight chords between disk holes. PCC's smooth-arc
`Δl = r·θ` is the approximation. **This is precisely the error the Kalman filter
should absorb.**

Net effect: the plant deviates from the controller's `pcc_angles()` by up to **5.21°**
of segment angle.

### Phase 3 — Calibration ✅ (`check_phase3.py`, 6/6)

**I changed the plan's objective, deliberately.** The plan originally said "fit joint
stiffness to minimise the discrepancy against PCC." **That target is self-defeating** —
it rebuilds the Section 3 tautology with more machinery. If I tune the plant until it
agrees with PCC, there is no model error left and Phase 5 measures nothing.

New objective: **justify the stiffness physically, then measure the mismatch and
report it as a budget.**

Stiffness is justified by force: spanning the workspace demands **9.9 N typical /
31 N peak** cable tension — realistic for a 13 mm tendon-driven arm. And stiffness is
demonstrably *not* the important knob: an **8× stiffness change moves the mismatch by
only 21.8%**, confirming the mismatch is kinematic (cable chords, actuator compliance),
not elastic.

**The model-error budget — this is the key input to Phase 5:**

| Metric | Tip position | Tip attitude |
| :--- | :--- | :--- |
| RMS | **4.75 mm** | 3.82° |
| mean | 3.24 mm | — |
| max | 13.69 mm | 8.29° |

And it grows with bend, as the θ² chord term predicts: 0.475 mm mean error for small
bends, 6.345 mm for large ones.

> **Trap caught here.** The first run reported a **168%** sensitivity to stiffness.
> That was not physics — it was a 60 N force limit binding at the workspace corners.
> The true figure is **8.7%**. *A saturating actuator does not announce itself; it
> just changes your answer.* The limit is now 80 N and `check_phase3.py` tests for
> saturation explicitly.

**Gravity — an honest result.** Out-of-plane gravity `(0,0,−g)` shifts the tip by
**exactly 0.000000 mm**, as it must: a z-force exerts no torque about a z hinge.
In-plane gravity would add up to 4.6 mm of sag. For the printing scenario the arm
bends in a horizontal plane over the bed, so gravity is out-of-plane and its
contribution is genuinely zero. **This is open question 1 for Prof. Cao** — if the
physical arm bends in a *vertical* plane, I set `params.GRAVITY = (0,−9.81,0)` and
re-run everything.

### Phase 4 — Connecting the controller ✅ (`check_phase4.py`, 5/5)

Built `sim/bridge.py`, which closes the loop: reference → controller → plant →
measured pose → controller. The controller is imported **unmodified** from the
vendored copy.

Results (all re-verified today, they reproduce exactly):

- **Regulation:** error 42.43 mm → **0.0000 mm**, reaching 1 mm by step 10.
- **Ellipse tracking:** 200 steps, no divergence, **2.75 mm RMS**.
- **100% of steps reached quasi-static equilibrium** — the controller is reading a
  settled arm, which is the assumption the paper's method rests on.
- **Kalman gain non-zero on 198/200 steps** (max |K| = 3.48e4) — the estimator is
  genuinely running, not silently bypassed.
- **UDP path verified live:** `pose_feedback.py` received and correctly parsed the
  plant's packets. **So the existing GUI can be driven by MuJoCo with zero changes
  to `Continuum_v3`.**

> **Two more traps found here, both of which would have corrupted Phase 5.**
>
> **1. I was measuring settling on the wrong observable.** `max|qvel|` plateaus at
> ~0.25 rad/s and never decays, so only 14% of steps counted as "settled" — for an arm
> that had completely stopped. Equilibrium is now judged on **tip displacement**: the
> observable the controller actually consumes, and the one that genuinely converges.
>
> **2. That plateau was a real 75 µm/step tip buzz.** The stiff actuator mode is
> ω ≈ √(kp·r²/armature) ≈ 790 rad/s, so ω·dt ≈ 1.58 at a 2 ms timestep — past clean
> integration. It was injecting 75 µm of noise straight into the pose the Kalman filter
> reads. Raising `JOINT_ARMATURE` from 1e-5 to **1e-4** removes it entirely at zero
> compute cost, and I verified it does not bias the statics: four different
> armature/timestep combinations all settle to the same tip y = 97.8660 mm, while the
> buzzing case reported 97.8921 mm — so the 26 µm gap was the artifact, not the fix.
>
> Also fixed: a fresh MuJoCo `MjData` has `ctrl = 0`, which for a position actuator on
> cable *length* commands a ~20 mm pull, not a released arm. That had silently turned a
> free-relaxation test into a hard-load test.

---

## 6. The one honest problem — be upfront about this

| Condition | Tip tracking RMS |
| :--- | :--- |
| Kalman **ON** | 2.748 mm |
| Kalman **OFF** | 2.830 mm |

**Only a ~3% improvement.** The project's stated success criterion needs better
than that. Do **not** hide this — walking in and saying "it works, 3% better" is
much weaker than saying "here is the number, here is why, here is my plan."

**Why I believe it is small, and it is not because the filter is broken:**

The error is dominated by **lag, not by Jacobian error**. Look at the control law:

```
u_next = u + β · J† · error
```

There is **no feedforward term**. It only reacts to error that has already
happened, so it inherently *trails* a moving reference — no matter how perfect the
Jacobian is. Against a moving ellipse, most of the 2.75 mm is that periodic lag, and
the Kalman filter cannot remove lag. It can only remove the part of the error caused
by aiming in the wrong direction.

**My plan for Phase 5** (this is the answer to give if he pushes):

1. **Retune the gains against this plant.** `γ`, `β`, `α`, `σ_p`, `σ_ψ` were tuned
   against a PCC plant — i.e. tuned in the tautology. There is no reason they are
   right here.
2. **Slow the trajectory down** (increase period T). The paper's method assumes
   quasi-static motion. A slower reference shrinks the lag term, which lets the
   Jacobian-error term become visible.
3. **Or add a feedforward term** so lag is removed by construction and the comparison
   isolates what the Kalman filter alone contributes.
4. Push the arm into **larger bends**, where the model error is 6.3 mm rather than
   0.5 mm — the compensator has more to work with there.

Note the structural point in my favour: the model error budget is **4.75 mm RMS** but
the closed-loop tracking error is **2.75 mm**. The feedback loop is already absorbing
most of the model error. That is expected — feedback is good at steady offsets. The
Kalman filter's job is the *remainder*, which is a smaller target than the headline
4.75 mm suggests.

---

## 7. Numbers worth memorising

| Thing | Number |
| :--- | :--- |
| Arm | 3 segments × 90 mm = 270 mm, 13 mm diameter |
| Cable radii | 5 / 3.5 / 2 mm (by segments-back, not by segment) |
| Model | 18 rigid links, 15 mm each, 18 hinges, 20.9 g |
| Timestep | 2 ms, implicitfast integrator |
| Discretization error | 0.44 mm at 60°/seg; **< 0.16 mm** in the working range |
| Convergence | exactly **4.00×** per doubling — second order |
| Model error budget | **4.75 mm RMS**, 3.82°, max 13.69 mm |
| Cable forces | 9.9 N typical, 31 N peak, 80 N limit |
| Stiffness sensitivity | 8× stiffness → only 21.8% mismatch change |
| Regulation | 42.43 mm → 0.0000 mm in ~10 steps |
| Tracking (ellipse) | **2.75 mm RMS** Kalman ON, 2.83 mm OFF |
| Quasi-static | **100%** of control steps reached equilibrium |
| Verification | Phases 0–4: 5/5, 8/8, 8/8, 6/6, 5/5 — all pass |

---

## 8. What is not done yet

| Phase | Status | What it is |
| :--- | :--- | :--- |
| 5 — Ellipse reproduction | **next** | Run Kalman ON vs OFF vs noisy on the paper's Traj. 1, retune gains, produce error plots and an RMS table |
| 6 — Printing process | not started | Attach a nozzle at the tip, add a print bed, track extrusion state per G-code command, render the deposited bead, measure commanded-vs-deposited deviation |
| 7 — Deliverables | not started | MP4 video of a print, the four result plots, report section with an explicit limitations paragraph |

**Three open questions I need answered by Prof. Cao:**

1. **Gravity.** Does the physical arm bend in a horizontal plane (gravity
   out-of-plane, contribution exactly zero — my current assumption) or a vertical one
   (up to 4.6 mm of sag, and a planar model cannot represent out-of-plane sag at all)?
2. **Calibration target.** Is there any **measured hardware data**? `fit_stiffness()`
   is already written and waiting for it. I deliberately did not point it at
   PCC-generated targets, for the tautology reason above.
3. **Printing scope.** Is a single-layer 2D trace sufficient, or is multi-layer
   expected later?

---

## 9. The 60-second version

> The existing simulation used the PCC model as both the controller **and** the robot,
> so the tracking error was near zero by construction and the paper's Kalman
> compensator had no model error to compensate — it could not prove anything.
>
> So I built an independent plant in MuJoCo: the arm as 18 rigid links with torsional
> springs, driven by six real routed cables, with its own dynamics and forces. Nothing
> in it comes from PCC.
>
> I verified it in phases, each with an automated check. The geometry is second-order
> accurate — error falls exactly 4× per doubling of link count, which proves the
> construction rather than tuning it. I then measured the mismatch against PCC rather
> than tuning it away: **4.75 mm RMS**, dominated by a cable chord-versus-arc term that
> matches the closed-form L·δ²/8 to within 2%. That mismatch *is* the point — it is what
> the Kalman filter should absorb.
>
> The unmodified controller now closes the loop on this plant: it regulates 42 mm of
> error to zero, and tracks the paper's ellipse at **2.75 mm RMS**, with 100% of steps
> quasi-static as the paper assumes. The Kalman filter is live and measurably different
> from off — but currently only 3% better, because the residual error is dominated by
> tracking lag, not Jacobian error. The control law has no feedforward term, so it
> trails a moving reference regardless of Jacobian quality. Phase 5 addresses that:
> retune the gains against this plant instead of against PCC, slow the trajectory, and
> test at larger bends where the model error is 6 mm rather than 0.5 mm.

---

## 10. Likely questions, and what to say

**"Why MuJoCo and not a Cosserat rod model or FEM?"**
Cosserat and FEM are more accurate but computationally expensive — the paper itself
says so in its introduction, and that cost is exactly why the paper uses PCC in the
first place. I need to run thousands of closed-loop steps. A discretized rigid-link
chain is standard practice, it is the "pseudo-rigid body" approach the paper cites as
approach #1, and I *quantified* its error rather than assuming it: 0.44 mm at 60°,
under 0.16 mm in the working range — an order of magnitude below the effects I am
measuring. It is a stated limitation in the report.

**"Is 18 links enough?"**
Yes, and I tested it rather than assumed it. Error falls exactly 4.00× per doubling,
so at 18 links it is 0.44 mm worst case; 36 links buys 0.1 mm for double the compute.
And a hard limit: beyond ~24 links per segment the nested MJCF crashes the XML parser.

**"How do you know the model is right?"**
Four independent ways. (1) The undeformed tip is at exactly 270.000000 mm. (2) The
convergence is exactly second-order, which cannot be faked by tuning. (3) The measured
cable moment arms come out at exactly 5.000 / 3.500 / 2.000 mm. (4) The nonlinear part
of the cable shortening matches the closed-form chord term to within 2%. Plus every
phase has an automated script that re-checks all of it.

**"Isn't a 4.75 mm model error too big?"**
That is not a defect, it is the experimental condition. If it were zero I would be back
to testing the controller against itself. And it is *understood*: it is dominated by the
cable chord-versus-arc term, which is real physics — the paper's own Fig. 2B shows
revolute joints between disks, so the real cables run as chords too.

**"Why does the Kalman filter only help by 3%?"**
→ Section 6. Say it plainly, then give the four-point plan.

**"Did you modify the controller?"**
No — deliberately. It is copied byte-for-byte and verified by SHA-256 on every run.
If I rewrote it I would be testing my rewrite, not the paper's method. The bridge also
speaks the same UDP JSON protocol the GUI already uses, so the existing software can be
driven by MuJoCo with zero code changes.

**"How does this connect to the real robot?"**
Three ways, all already in place. The parameters come from the controller source and
are cross-checked automatically so they cannot drift. The G-code parser is the
hardware's own parser, so the simulation follows exactly what the hardware would follow.
And the pose feedback path is the same UDP protocol a camera or EM tracker would use —
so the simulation is a drop-in stand-in for the sensor.

**"What if I told you the arm bends vertically?"**
Then I set `GRAVITY = (0,−9.81,0)` and re-run — one line, and the checks re-validate
everything. I measured that in-plane gravity would add up to 4.6 mm of sag. I would also
flag that a *planar* model cannot represent out-of-plane sag at all, so if the arm bends
vertically we should also discuss whether the 2D restriction still holds.

---

## 11. If you want to show something live

```powershell
# Look at the model, bend it by hand with the keyboard
python MuJoCo_Sim/inspect_model.py
#   Q/A, W/S, E/D bend the three segments; R straightens; P prints angles/pose/forces

# Watch the controller drive the arm on the paper's ellipse
python MuJoCo_Sim/run_live.py

#   Opens STRAIGHT AND STILL with the reference path drawn - good for talking
#   over. Press S when you want it to move; N/B browse without starting anything.

# Follow a G-code toolpath instead
python MuJoCo_Sim/run_live.py --gcode triangle.gcode

# Kalman on vs off, side by side
python MuJoCo_Sim/run_live.py --no-kalman
#   In the viewer: S start/stop, N/B switch shape, K toggles Kalman,
#   T trail, [ ] speed, R restart

# Re-run the verification in front of him — this is the strongest move
python MuJoCo_Sim/check_phase1.py     # geometry           8/8
python MuJoCo_Sim/check_phase2.py     # tendons/actuators  8/8
python MuJoCo_Sim/check_phase3.py     # model-error budget 6/6
python MuJoCo_Sim/check_phase4.py     # closed loop        5/5
```

Existing figures are already in `MuJoCo_Sim/outputs/figures/` (`phase1_arm.png`,
`phase2_tendons.png`, `phase3_model_error.png`, `phase4_closed_loop.png`) and the
per-step logs in `MuJoCo_Sim/outputs/logs/`.

---

## 12. The three things to lead with

If the meeting is short, these are the three points that carry the most weight — they
are all about **judgement**, which is what a supervisor is actually assessing:

1. **I found that the previous simulation was circular and could not validate the
   paper's contribution**, and I fixed that with an independent physics plant.
2. **I changed the calibration objective when I realised the plan's own objective was
   self-defeating** — I measure the model mismatch instead of tuning it away, because
   removing it would recreate the same circularity.
3. **I caught four measurement artifacts that would each have produced confident,
   wrong numbers**: a saturating force limit reporting 168% instead of 8.7%; a settling
   test on the wrong observable reporting 14% instead of 100%; a numerical buzz injecting
   75 µm of fake noise into the sensor; and a hinge placement error worth 15 mm of tip
   bias. Everything since is verified by scripts so none of them can come back silently.
