# Software V3 Model Improvement Workflow

This document defines the implementation plan for improving the continuum robot control software while staying aligned with Zhai et al. 2025, "Model-Based Control of a Continuum Manipulator with Online Jacobian Error Compensation Using Kalman Filtering."

## Goal

Build a V3 control stack that reproduces the paper's core method more faithfully than V2:

- Planar 3-segment PCC kinematic model.
- Jacobian-based closed-loop control.
- Online Jacobian error compensation using a 9-state Kalman filter.
- Constrained/rate-limited Jacobian updates for stability.
- Filtered real-time pose and actuation feedback.
- Clear validation against paper trajectories and parameter studies.

## Paper-Faithful Baseline

Before adding improvements, V3 should first reproduce the baseline method:

1. Use actuation vector:
   `u = [Delta l1, Delta l2, Delta l3]^T`

2. Convert actuation to segment bending:
   `r1 * theta1 = Delta l1`
   `r2 * theta1 + r1 * theta2 = Delta l2`
   `r3 * theta1 + r2 * theta2 + r1 * theta3 = Delta l3`

3. Compute end-effector pose:
   `pose = [x, y, psi]^T`
   where `psi = theta1 + theta2 + theta3`.

4. Compute PCC model Jacobian:
   `J_model = d[x, y, psi] / d[Delta l1, Delta l2, Delta l3]`

5. Estimate model error:
   `delta_J = J_actual - J_model`

6. Use Kalman filter state:
   `xi = vec(delta_J)`, with shape `9x1`.

7. Compute compensated Jacobian:
   `J_est = J_model + delta_J_est`

8. Apply Jacobian constraint:
   limit position Jacobian update by `sigma_p`.
   limit attitude Jacobian update by `sigma_psi`.

9. Apply closed-loop control:
   `u_next = u + beta * pinv_damped(J_constrained) * (pose_ref - pose_measured)`

## Default Parameters

Use the paper values as the starting baseline:

| Parameter | Value |
| --- | --- |
| `L1, L2, L3` | `0.09, 0.09, 0.09 m` |
| `r1, r2, r3` | `0.005, 0.0035, 0.002 m` |
| `alpha` | `1` |
| `beta` | `0.5` |
| `gamma` | `0.5` |
| `sigma_p` | `0.35` |
| `sigma_psi` | `2.5` |
| `xi0` | `zeros(9, 1)` |
| `P0` | `zeros(9, 9)` |
| `Qk` | `I9` |
| `Rk` | `0.5 * norm(u_k - u_prev)^2 * I3` |
| Control loop | `20 Hz` |
| Pose feedback | `30 Hz` |
| Low-pass filter | 4th-order Butterworth, `2 Hz` cutoff |

## Proposed V3 Architecture

Separate the software into small testable modules:

1. `model/pcc_kinematics.py`
   - Convert tendon-pair length changes to bending angles.
   - Compute forward pose `[x, y, psi]`.
   - Compute analytical or numerical PCC Jacobian.
   - Handle near-zero bending angles safely.

2. `estimation/jacobian_kalman.py`
   - Maintain `xi`, `P`, `Q`, `R`.
   - Build measurement matrix `H`.
   - Update `delta_J` from measured pose changes and actuation changes.
   - Reconstruct `delta_J` from row-vectorized state.

3. `control/jacobian_controller.py`
   - Apply Jacobian compensation.
   - Apply position and attitude Jacobian constraints.
   - Compute damped pseudoinverse.
   - Generate next actuation command.

4. `io/pose_feedback.py`
   - Receive or simulate visual pose feedback.
   - Normalize units and angle convention.
   - Apply low-pass filtering.

5. `io/motor_interface.py`
   - Convert desired `Delta l` commands to motor/capstan commands.
   - Enforce motor limits, velocity limits, and safe stop.

6. `experiments/`
   - Reproduce paper trajectories.
   - Log reference pose, measured pose, command, Jacobian, Kalman state, and errors.
   - Export CSV and plots for RMSE/MAE/MaxAE.

## Implementation Phases

### Phase 1: Audit V2

- Identify where V2 currently computes PCC pose.
- Identify where V2 computes Jacobian or inverse Jacobian.
- Identify whether V2 already has Kalman Jacobian compensation.
- Identify whether V2 filters pose and actuation inputs before estimation.
- Identify unit conventions: meters vs millimeters, radians vs degrees.
- Identify sign conventions for tendon length, motor rotation, and image coordinates.

Deliverable:

- A short V2 audit note listing paper-aligned parts and mismatches.

### Phase 2: Build Paper Baseline in V3

- Copy only necessary stable code from V2.
- Implement clean PCC kinematics.
- Implement paper trajectories:
  - Traj. 1: `x_r = 240 + 20 cos(2*pi*t/T) mm`, `y_r = 60 sin(2*pi*t/T) mm`, horizontal attitude.
  - Traj. 2: stationary position, `psi_r = 30 deg sin(2*pi*t/T)`.
  - Traj. 3: combined position and attitude tracking.
- Implement controller with PCC Jacobian only.
- Verify expected behavior before adding Kalman compensation.

Deliverable:

- A runnable baseline that can track references using only `J_model`.

### Phase 3: Add Online Jacobian Error Compensation

- Add Kalman prediction:
  - `xi_prior = gamma * xi_prev`
  - `P_prior = gamma^2 * P_prev + Q`

- Add measurement update:
  - measurement is `pose_k - pose_prev`
  - model prediction uses `H * (vec(J_model) + xi_prior)`
  - update `xi` and `P` with Kalman gain.

- Reconstruct:
  - `delta_J_est = unvec_rowwise(xi)`
  - `J_est = J_model + delta_J_est`

Deliverable:

- Logged `J_model`, `delta_J_est`, and `J_est` during trajectory tracking.

### Phase 4: Add Jacobian Constraints

- Split `J_est` into:
  - position Jacobian: first two rows.
  - attitude Jacobian: third row.

- Apply Frobenius-norm update limits:
  - if `norm(Jp_est - Jp_prev_constrained) <= sigma_p`, accept it.
  - otherwise move from previous constrained value toward estimate by exactly `sigma_p`.
  - repeat similarly for `Jpsi` with `sigma_psi`.

Deliverable:

- Smooth constrained Jacobian logs with no large frame-to-frame jumps.

### Phase 5: Filtering and Timing

- Add 4th-order Butterworth filter at `2 Hz`.
- Filter measured pose before Kalman update.
- Filter actuation input before Kalman update.
- Keep controller loop near `20 Hz`.
- Handle missing/delayed pose packets gracefully.

Deliverable:

- Stable controller loop with timestamped logs and measured loop frequency.

### Phase 6: Validation Against Paper

Run the same validation cases as the paper:

- Traj. 1, 2, and 3 with `T = 40 s`.
- Traj. 3 with `T = 20 s`, `10 s`, and `5 s`.
- Gamma comparison: `gamma = 0`, `0.5`, `1`.
- Threshold comparison:
  - small: `sigma_p = 0.0875`, `sigma_psi = 0.625`.
  - baseline: `sigma_p = 0.35`, `sigma_psi = 2.5`.
  - large: `sigma_p = 1.4`, `sigma_psi = 10`.
- Disturbance/payload recovery if hardware setup allows.

Metrics:

- RMSE.
- MAE.
- MaxAE.
- Jacobian smoothness.
- Recovery time after disturbance.

Deliverable:

- A validation report comparing V3 results against paper trends.

## Model Improvements After Baseline

Only after the baseline works, consider these improvements:

1. Better numerical Jacobian handling
   - Use analytical Jacobian if stable.
   - Fall back to central finite differences near singular cases.
   - Add tests around `theta -> 0`.

2. Unit-safe control
   - Internally use SI units.
   - Convert camera output and GUI display at boundaries only.
   - Keep `psi` internally in radians.

3. Adaptive noise tuning
   - Start with paper `Q` and `R`.
   - Increase `R` when pose feedback is noisy or marker confidence is low.
   - Increase `Q` when payload/contact disturbance is detected.

4. Safety limits
   - Clamp tendon length commands.
   - Clamp motor velocity and acceleration.
   - Stop on pose loss, singular Jacobian, or command jump.

5. Calibration layer
   - Estimate actual tendon offsets `r1, r2, r3`.
   - Estimate capstan radius and motor-to-length scale.
   - Record camera-to-robot coordinate transform.

6. Optional dynamic extension
   - The paper explicitly neglects dynamics.
   - Do not tune V3 for fast trajectories without adding a dynamic model or velocity-aware compensation.

## Wrong-Direction Checklist

Treat these as warning signs:

- The controller uses PCC only and has no online Jacobian residual update.
- The Kalman filter estimates pose instead of Jacobian error.
- Pose feedback is not used in the estimator.
- Actuation and pose units are mixed.
- `psi` degrees and radians are mixed.
- The controller is tested only on one trajectory.
- High-speed tracking is expected without modeling dynamics.
- The constrained Jacobian step is skipped.
- Raw noisy measurements are fed directly into the Kalman update.
- Results are judged visually without RMSE, MAE, and MaxAE.

## Immediate Next Steps

1. Audit `Software_V2/continuum_ellipse.py` against this workflow.
2. Decide whether V3 should copy V2 code first or start as a clean module layout.
3. Implement PCC kinematics tests.
4. Implement the paper controller in simulation mode before touching hardware.
5. Add hardware integration only after logged simulation behavior matches paper trends.
