# UPGRADES — Aligning `continuum_ellipse.py` with Zhai et al. 2025

Gap analysis of [`continuum_ellipse.py`](continuum_ellipse.py) and [`pose_feedback.py`](pose_feedback.py)
against:

> Zhai Y, Xu J, Mo H, Zhang C, Sun D. *Model-Based Control of a Continuum Manipulator with
> Online Jacobian Error Compensation Using Kalman Filtering.* Cyborg Bionic Syst. 2025;6:0339.
> https://doi.org/10.34133/cbsystems.0339

Equation and table numbers below refer to the paper. Source text: `../paper_zhai_2025.txt`.

---

## Contents

- [Status summary](#status-summary)
- [Already faithful — do not change](#already-faithful--do-not-change)
- [P0 — Blockers](#p0--blockers)
- [P1 — Missing experiments](#p1--missing-experiments)
- [P2 — Signal processing](#p2--signal-processing)
- [P3 — Correctness and instrumentation](#p3--correctness-and-instrumentation)
- [Implementation order](#implementation-order)
- [Acceptance targets](#acceptance-targets)

---

## Status summary

| # | Item | Priority | Status |
|---|------|----------|--------|
| 1 | Ground-truth plant model distinct from controller model | P0 | ☐ |
| 2 | Kalman `R`/`S` singularity when `Δu → 0` | P0 | ☐ |
| 3 | Traj. 2 and Traj. 3 | P1 | ☐ |
| 4 | PCC-vs-Proposed comparison run + Table 2 output | P1 | ☐ |
| 5 | Butterworth filter on actuation inputs | P2 | ☐ |
| 6 | MaxAE metric + payload disturbance (Tables 4–5) | P1 | ☐ |
| 7 | Parameter sweep driver (Table 3) | P1 | ☐ |
| 8 | Jacobian / δJ time-history plots | P3 | ☐ |
| 9 | Constraint wrongly applied to PCC-only baseline | P3 | ☐ |
| 10 | Angle wrapping on ψ error | P3 | ☐ |
| 11 | Virtual clock drift in control loop | P3 | ☐ |
| 12 | `Q` hardcoded, not configurable | P3 | ☐ |
| 13 | Marker tracker (UDP sender) missing | P2 | ☐ |
| 14 | No filtering path in simulation mode | P2 | ☐ |
| 15 | CSV export drops ψ and Jacobian | P3 | ☐ |

**The math core is correct. What is missing is everything that makes the paper's claims testable.**

---

## Already faithful — do not change

These were verified line-by-line against the paper and are correct:

- **PCC angle solve (Eqs. 7–9)** — [`continuum_ellipse.py:517-528`](continuum_ellipse.py#L517-L528)
  `θ₁ = Δl₁/r₁`, `θ₂ = (Δl₂ − r₂θ₁)/r₁`, `θ₃ = (Δl₃ − r₃θ₁ − r₂θ₂)/r₁`.
- **Forward kinematics (Eqs. 3–5)** — [`continuum_ellipse.py:554-587`](continuum_ellipse.py#L554-L587)
  Arc integrals match exactly, and the `|θ| < 1e-9` L'Hôpital branch correctly handles the
  straight-segment limit the paper leaves undefined.
- **Kalman recursion (Eqs. 11–19)** — [`continuum_ellipse.py:777-821`](continuum_ellipse.py#L777-L821)
  Row-stacked `∨` operator, 3×9 measurement matrix `H` of Eq. 15, and the innovation
  `Δpose − H(ᵐJ∨ + ξ⁻)` are all correct and correctly ordered.
- **Jacobian constraint (Eq. 20)** — [`continuum_ellipse.py:823-848`](continuum_ellipse.py#L823-L848)
  Frobenius norm, correct position/attitude split at rows `0:2` / `2:3`.
- **Damped pseudoinverse (Eq. 22)** — [`continuum_ellipse.py:850-855`](continuum_ellipse.py#L850-L855)
- **Table 1 parameters** — `L`, `r`, `α=1`, `β=0.5`, `γ=0.5`, `σ_p=0.35`, `σ_ψ=2.5`,
  `ξ₀=0`, `P₀=0`, `Q=I₉`, `R=0.5‖Δu‖²I₃`, 20 Hz control rate. All present and matching.

Optional refinement, not a gap: `model_jacobian` uses central finite differences
([`continuum_ellipse.py:590-604`](continuum_ellipse.py#L590-L604), 6 FK evals/step). Numerically fine
at `h=1e-7`; a closed-form analytical Jacobian would be faster and exact.

---

## P0 — Blockers

### 1. The Kalman filter has nothing to estimate

**This one invalidates every other result.**

[`continuum_ellipse.py:2849-2850`](continuum_ellipse.py#L2849-L2850):

```python
if self._simulated:
    return forward_kinematics(self._u_curr), "model", "simulation model pose"
```

In simulation the "measurement" **is the controller's own PCC model**. The paper's entire premise is
`δJ = J − ᵐJ ≠ 0`, caused by modeling inaccuracy. Here `δJ ≡ 0` by construction: the innovation is
numerically zero, `ξ` stays at zero, and **"Proposed" and "PCC only" produce identical trajectories**.
Table 2 cannot be reproduced.

`ModifiedPCCParams` ([`continuum_ellipse.py:266-277`](continuum_ellipse.py#L266-L277)) is applied to the
**controller** ([`continuum_ellipse.py:2841`](continuum_ellipse.py#L2841)) while the plant stays nominal —
backwards from its own docstring, which describes it as a *calibration* layer, i.e. the more accurate model.

**Implement**

A ground-truth plant function `plant_forward_kinematics(u)` used *only* by `_read_pose_for_control`,
never by `KalmanJacobianController`. It must contain effects the PCC model structurally cannot represent:

- **Gravity sag** on distal segments — y-biased, growing with cumulative bend angle.
  This is what produces the 30–40 mm y-deviations of Table 4.
- **Tendon friction / hysteresis** — Coulomb deadband on `Δl` direction reversal.
- **Backbone extension and inter-segment coupling** — `θᵢ` depending on `θⱼ, j < i`,
  beyond what the `r`-matrix of Eqs. 7–9 captures.
- **Per-segment effective length ≠ 0.09 m.**

Repoint `ModifiedPCCParams` to drive the **plant**; give the controller the nominal PCC. Then
`use_kalman=False` vs `True` becomes a meaningful comparison.

---

### 2. `R → 0` when the robot is stationary

[`continuum_ellipse.py:793-808`](continuum_ellipse.py#L793-L808):

```python
R = max(0.5 * du_norm_sq, 1e-12) * np.eye(3)
S = H @ P_minus @ H.T + R
K = P_minus @ H.T @ np.linalg.inv(S)
```

Under Traj. 2 the position is held stationary, so `Δu → 0` drives **both** `H → 0` and
`R → 1e-12·I`. `S` collapses to ~`1e-12`; `np.linalg.inv` returns ~`1e12` garbage rather than
raising, so the `except np.linalg.LinAlgError` never fires and `ξ` receives a huge bogus correction.
The `1e-12` floor makes this *worse*, not better — it guarantees a near-singular `S` instead of a
well-conditioned large one.

**Implement**

```python
DU_MIN = 1e-6                                   # metres, excitation threshold
if np.linalg.norm(du) < DU_MIN:                 # no excitation → predict only
    self.xi, self.P = xi_minus, P_minus
    return mJ + xi_minus.reshape(3, 3)

R = (0.5 * du_norm_sq + R_FLOOR) * np.eye(3)    # R_FLOOR ~ measurement variance
S = H @ P_minus @ H.T + R
K = np.linalg.solve(S.T, (P_minus @ H.T).T).T   # never inv()
```

Also add:
- symmetrisation `self.P = 0.5 * (P_plus + P_plus.T)` after Eq. 18
- Joseph form of Eq. 18 for numerical stability over long runs

---

## P1 — Missing experiments

### 3. Traj. 2 and Traj. 3 do not exist

[`continuum_ellipse.py:900-916`](continuum_ellipse.py#L900-L916) implements only Traj. 1;
`TRAJECTORY_NAMES` lists one trajectory plus G-code. The `att_hold` checkbox
([`continuum_ellipse.py:2941-2942`](continuum_ellipse.py#L2941-L2942)) forces `ψᵣ=0`, a partial
stand-in for Traj. 1 only.

**Implement** in `make_trajectory`:

| Trajectory | xᵣ (mm) | yᵣ (mm) | ψᵣ |
|---|---|---|---|
| Traj. 1 — position, level attitude | `240 + 20·cos(2πt/T)` | `60·sin(2πt/T)` | `0` |
| Traj. 2 — attitude, stationary position | `240` (const) | `0` (const) | `30°·sin(2πt/T)` |
| Traj. 3 — hybrid | `240 + 20·cos(2πt/T)` | `60·sin(2πt/T)` | `30°·sin(2πt/T)` |

Default `T = 40 s`. Traj. 2 exercises the stationary-`Δu` path of item 2, and is where the paper's
improvement is largest (2.1° → 1.5° ψ RMSE).

---

### 4. No PCC-vs-Proposed overlay, no Table 2

Figs. 6B / 7B / 8B all plot **Reference / PCC model / Proposed on one axis**, and Table 2 tabulates
both methods side by side. The GUI runs one method at a time; `_redraw_traj` plots ref-vs-actual only,
and `_print_summary` ([`continuum_ellipse.py:3018-3026`](continuum_ellipse.py#L3018-L3026)) prints a
single method's column.

**Implement**

A "Run comparison" action that:
1. Executes the selected trajectory with `use_kalman=False` from a clean reset.
2. Re-executes it with `use_kalman=True` from an identical reset state.
3. Retains both histories and overlays three curves (reference, PCC, proposed).
4. Prints the full 6-column Table 2 (RMSE/MAE × x/y/ψ × both methods).

---

### 6. No disturbance injection, no MaxAE (Tables 4–5)

Metrics at [`continuum_ellipse.py:3064-3086`](continuum_ellipse.py#L3064-L3086) compute RMSE and MAE
only. The robustness section reports **MaxAE** under hung payloads.

**Implement**

- `maxae_x` / `maxae_y` / `maxae_psi` alongside the existing metrics, surfaced in the metrics panel
  and in `_print_summary`.
- A payload mass field plus an "apply at t =" trigger that adds a mass-dependent gravitational
  deflection to the **plant** (item 1) mid-run. Recovery from that step disturbance is the headline
  robustness result.

Paper reference values (Table 4, stationary pose):

| Payload | x MaxAE | y MaxAE | ψ MaxAE |
|---|---|---|---|
| Tape core (5.6 g) | 1.6 mm | 16.0 mm | 4.8° |
| Corner bracket (12.8 g) | 3.4 mm | 30.5 mm | 7.4° |
| USB converter (17.5 g) | 5.3 mm | 38.9 mm | 12.5° |

---

### 7. No parameter sweep (Table 3)

Table 3 requires 8 runs of Traj. 3:

| Run | Parameters |
|---|---|
| Baseline | `T=40`, `γ=0.5`, `σ_p=0.35`, `σ_ψ=2.5` |
| Period | `T = 20 s`, `10 s`, `5 s` |
| Fading rate | `γ = 0`, `γ = 1` |
| Thresholds | `σ_p=1.4, σ_ψ=10` (4×) and `σ_p=0.0875, σ_ψ=0.625` (¼×) |

All knobs are already exposed as `StringVar`s, so this is a batch driver looping over
`_make_controller()` and collecting metrics — no new physics.

> Note: the `T = 5 s` row is exactly the case corrupted by item 11 (clock drift). Fix 11 before
> trusting these numbers.

---

## P2 — Signal processing

### 5. Actuation inputs are never filtered

Paper, Experimental setup:

> *"a 4th-order Butterworth low-pass filter with a cutoff frequency of 2 Hz was employed to
> preprocess the end-effector pose measurements **and the actuation inputs** used in the Jacobian
> estimation."*

`ButterworthLowPass4` exists at [`pose_feedback.py:42-90`](pose_feedback.py#L42-L90) but is wired only
to the pose stream. `du = u_curr - self.u_prev` at
[`continuum_ellipse.py:784`](continuum_ellipse.py#L784) uses raw commands.

**Implement** a second `ButterworthLowPass4(cutoff_hz=2.0, sample_hz=20.0, channels=3)` inside
`KalmanJacobianController`, filtering `u` before `du` is formed.

> The sample rate here must be the **control** rate (20 Hz), not the camera rate (30 Hz) used by the
> pose filter.

---

### 14. No filtering path in simulation mode

The receiver — and therefore the pose filter — is constructed only when `not self._sim`
([`continuum_ellipse.py:3328-3333`](continuum_ellipse.py#L3328-L3333)). The "Add measurement noise"
option at [`continuum_ellipse.py:2946-2947`](continuum_ellipse.py#L2946-L2947) therefore feeds
unfiltered noise straight into the estimator — harsher than the paper's setup and unrepresentative
of hardware behaviour.

**Implement** a `ButterworthLowPass4` on the simulated pose path so both modes share identical
filtering.

---

### 13. Marker tracker (UDP sender) missing

[`pose_feedback.py`](pose_feedback.py) is only the UDP **receiver**. The paper's sender does not exist
in the repo: OpenCV blob detection of 3 reflective IR markers at 30 Hz from a 640×480 infrared webcam,
yielding `(x, y, ψ)` from the marker triangle.

**Implement** `marker_tracker.py` if hardware validation is planned — required for any real-robot run,
not needed for simulation work.

---

## P3 — Correctness and instrumentation

### 9. Constraint applied to the PCC-only baseline

[`continuum_ellipse.py:878`](continuum_ellipse.py#L878) runs `_apply_constraints` unconditionally.
When `use_kalman=False`, `eJ = ᵐJ`, and the `σ_ψ = 2.5 rad/m` rate limit throttles an attitude
Jacobian whose entries are ~200 rad/m (Fig. 6E). The baseline is slew-limited in a way the paper's
baseline is not — biasing the comparison in the proposed method's favour.

```python
cJ = self._apply_constraints(eJ) if self.use_kalman else eJ
```

### 10. No angle wrapping on ψ

`error = pose_ref - pose_curr` at [`continuum_ellipse.py:883`](continuum_ellipse.py#L883) treats ψ
linearly. Harmless at ±30°, but wrap to `(−π, π]` before the pseudoinverse.

### 11. Virtual clock drift

[`continuum_ellipse.py:2996`](continuum_ellipse.py#L2996) advances `t += dt` unconditionally while
sleeping only the remainder. If a step overruns — likely at `T = 5 s` with GUI redraws active — the
reference is evaluated at a virtual time diverged from wall time, silently corrupting the fast rows of
Table 3. Use `t = time.perf_counter() - t0`.

### 12. `Q` hardcoded

`Q = np.eye(9)` is buried inside `_kalman_step`
([`continuum_ellipse.py:789`](continuum_ellipse.py#L789)). Table 1 lists `Q` and the `R` scale as model
parameters — move both to constructor arguments.

### 8. No Jacobian time-history plots

Figs. 6E / 7D / 8E plot all 9 Jacobian elements vs. time; Figs. S1–S3 plot the estimated `δJ`.
`_update_jac_display` ([`continuum_ellipse.py:3089`](continuum_ellipse.py#L3089)) writes a text label.

**Implement** per-step logging of `last_mJ`, `last_cJ` and `xi.reshape(3,3)`, plus a 3×3 subplot grid.
This is the primary visual evidence that the compensation is doing anything.

### 15. CSV export drops ψ and Jacobian

[`continuum_ellipse.py:3043`](continuum_ellipse.py#L3043) writes only `t, ref_x, ref_y, act_x, act_y, err`.
Add `ψᵣ`, `ψ`, and the 9 Jacobian elements so figures can be regenerated offline.

---

## Implementation order

```
1. Ground-truth plant model              ← nothing else is measurable without this
2. R/S singularity fix                   ← only reachable once the plant differs from the model
3. Traj. 2 and Traj. 3
4. Comparison run + Table 2
────────────────────────────────────────  paper's main result reproduced
5. Butterworth on actuation inputs
6. MaxAE + payload disturbance (Tables 4–5)
7. Parameter sweep driver (Table 3)
8. Jacobian / δJ history plots
────────────────────────────────────────  supporting figures
```

Items 1 and 2 should be done together: item 2's failure mode is only observable once the plant
actually differs from the model.

Items 9–12 and 15 are small and independent; fold them in wherever convenient, but do **9** before
item 4 and **11** before item 7, since each biases the results of the experiment that follows it.

---

## Acceptance targets

Reproduction of Table 2 (paper values):

| Experiment | Proposed x/y/ψ (RMSE) | PCC model x/y/ψ (RMSE) |
|---|---|---|
| Traj. 1 | 1.1 mm / 2.1 mm / 1.1° | 1.6 mm / 2.3 mm / 1.4° |
| Traj. 2 | 0.6 mm / 0.8 mm / 1.5° | 1.3 mm / 0.9 mm / 2.1° |
| Traj. 3 | 0.9 mm / 1.9 mm / 1.3° | 1.7 mm / 2.1 mm / 1.6° |

Absolute values depend on the plant model chosen in item 1 and will not match exactly. The
qualitative results that **must** hold:

- Proposed beats PCC-only on nearly every metric.
- Tracking error grows sharply as `T` decreases (quasi-static assumption breaks down).
- `γ = 1` is clearly worse than the baseline; `γ = 0` is close to but not better than it overall.
- Small `σ_p`/`σ_ψ` → sluggish tracking deviation; large → noisy oscillation.
