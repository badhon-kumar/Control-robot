# Zhai 2025 Replication Context

This note captures the working context from:

- `d:\Abroad\Meeting\Prof Cao\zhai_2025_Model-Based Control of a Continuum.pdf`
- `Software_V2/continuum_ellipse.py`

## Paper Target

Paper: Yujia Zhai, Jihao Xu, Hangjie Mo, Chunqi Zhang, Dong Sun, "Model-Based Control of a Continuum Manipulator with Online Jacobian Error Compensation Using Kalman Filtering", Cyborg and Bionic Systems, 2025, Article 0339.

Goal of the paper: improve tendon-driven continuum manipulator trajectory tracking by combining a PCC model Jacobian with online Kalman-filter estimation of the Jacobian error. No offline training data is required.

Main assumptions:

- Robot is quasi-static.
- Motion between adjacent control steps is small.
- Dynamics are neglected.
- End-effector pose feedback is available in real time.

## Robot Geometry And Parameters

The paper robot is planar, tendon-driven, and has 3 continuum segments.

- Each segment has 5 spacer disks and 1 end disk.
- Total length: 270 mm.
- Segment lengths: `L1 = L2 = L3 = 0.09 m`.
- Disk diameter: 13 mm.
- Three antagonistic tendon pairs actuate the three segments.
- Tendon offsets: `r1 = 0.005 m`, `r2 = 0.0035 m`, `r3 = 0.002 m`.
- Paper actuation vector: `u = [Delta l1, Delta l2, Delta l3]^T`.

Table 1 controller defaults:

- `alpha = 1`
- `beta = 0.5`
- `gamma = 0.5`
- `sigma_p = 0.35`
- `sigma_psi = 2.5`
- `xi0 = zeros(9, 1)`
- `P0 = zeros(9, 9)`
- `Qk = I9`
- `Rk = 0.5 * ||u_k - u_{k-1}||^2 * I3`

## PCC Model

The paper uses piecewise constant curvature:

- End-effector pose: `[x, y, psi]`.
- `psi = theta1 + theta2 + theta3`.
- The forward kinematics integrate each segment as a circular arc.
- Tendon-to-angle relationships:
  - `r1 * theta1 = Delta l1`
  - `r2 * theta1 + r1 * theta2 = Delta l2`
  - `r3 * theta1 + r2 * theta2 + r1 * theta3 = Delta l3`
- Model Jacobian:
  - `mJ = d[x, y, psi] / d[Delta l1, Delta l2, Delta l3]`

## Kalman Jacobian Error Compensation

The actual Jacobian is treated as:

- `J = mJ + deltaJ`
- Kalman state: `xi = vec(deltaJ)` using row-stacked matrix vectorization.

State transition:

- `xi_k^- = gamma * xi_{k-1}^+`
- `P_k^- = gamma^2 * P_{k-1}^+ + Q_{k-1}`

Measurement model:

- Pose delta between two steps is predicted from actuation delta and the compensated Jacobian.
- `H_k` is a 3 x 9 block matrix built from `(u_k - u_{k-1})`.
- Kalman gain:
  - `K_k = P_k^- H_k^T (H_k P_k^- H_k^T + R_k)^-1`

Posterior:

- `xi_k^+ = xi_k^- + K_k * innovation`
- `P_k^+ = (I9 - K_k H_k) P_k^-`
- `eJ_k = mJ_k + reshape(xi_k^+)`

Constrained Jacobian:

- The estimated Jacobian is split into position rows and attitude row.
- Position update is limited by `sigma_p`.
- Attitude update is limited by `sigma_psi`.
- This produces `cJ`, the constrained Jacobian.

Control law:

- `u_{k+1} = u_k + beta * cJ_dagger * ([p_r, psi_r] - [p, psi])`
- Damped pseudoinverse:
  - `cJ_dagger = (cJ^T cJ + alpha I3)^-1 cJ^T`

## Paper Trajectories And Reported Results

Traj. 1:

- Position tracking with horizontal attitude.
- `xr = 240 + 20*cos(2*pi*t/T) mm`
- `yr = 60*sin(2*pi*t/T) mm`
- `psi_r = 0`
- `T = 40 s`
- Proposed RMSE/MAE: x `1.1/0.9 mm`, y `2.1/1.8 mm`, psi `1.1/0.8 deg`.
- PCC-only RMSE/MAE: x `1.6/1.4 mm`, y `2.3/1.9 mm`, psi `1.4/1.1 deg`.

Traj. 2:

- Attitude tracking while keeping position stationary.
- `psi_r = 30 deg * sin(2*pi*t/T)`, `T = 40 s`.
- Proposed RMSE/MAE: x `0.6/0.4 mm`, y `0.8/0.6 mm`, psi `1.5/1.3 deg`.
- PCC-only RMSE/MAE: x `1.3/1.0 mm`, y `0.9/0.6 mm`, psi `2.1/1.9 deg`.

Traj. 3:

- Combined Traj. 1 position and Traj. 2 attitude.
- Proposed RMSE/MAE: x `0.9/0.8 mm`, y `1.9/1.7 mm`, psi `1.3/1.1 deg`.
- PCC-only RMSE/MAE: x `1.7/1.4 mm`, y `2.1/1.7 mm`, psi `1.6/1.4 deg`.

Important experimental detail:

- Real pose feedback comes from 3 reflective end-effector markers observed by an infrared camera.
- Camera feedback is 30 Hz and sent to the controller via UDP.
- A 4th-order Butterworth low-pass filter with 2 Hz cutoff preprocesses pose measurements and actuation inputs for Jacobian estimation.

## What continuum_ellipse.py Already Implements

Implemented:

- Hardware wrapper for six LKMTECH MF5015 motors over RS-485.
- Paper geometry constants:
  - `L_SEG = [0.09, 0.09, 0.09]`
  - `R_TENDON = [0.005, 0.0035, 0.002]`
  - `TOTAL_LEN_MM = 270`
  - disk layout with 5 spacer disks plus 1 end disk per segment.
- PCC angle solve in `pcc_angles()`.
- PCC forward kinematics in `forward_kinematics()`.
- Numeric model Jacobian in `model_jacobian()`.
- Motor displacement conversion:
  - `u_to_motor_disps_mm()`: converts 3 tendon deltas to six equal-and-opposite motor displacements.
  - `motor_disps_mm_to_u()`: inverse conversion.
- `KalmanJacobianController` with:
  - 9-state Jacobian error vector.
  - Kalman prediction/update.
  - constrained Jacobian update.
  - damped pseudoinverse.
  - paper control law.
- Traj. 1 ellipse in `make_trajectory()`.
- `TrajectoryControlPanel` GUI for running ellipse tracking, plotting reference/actual path, plotting error, exporting CSV, and printing Table-2-style metrics.

## Current Implementation Gaps For True Replication

The current code is a strong skeleton for Traj. 1, but it is not yet a full experimental replication.

- Real hardware mode still uses `forward_kinematics(self._u_curr)` as the actual pose. The paper uses external visual pose feedback from 3 markers.
- There is no camera/UDP pose feedback integration in `continuum_ellipse.py`.
- There is no 4th-order Butterworth 2 Hz preprocessing of pose and actuation values for Jacobian estimation.
- Only Traj. 1 is exposed. Traj. 2 and Traj. 3 are not implemented in the trajectory selector.
- Simulation uses the same PCC model as the controller plant, so Kalman compensation cannot demonstrate real model-error correction unless a perturbed plant, sensor feedback, or recorded experimental data is introduced.
- The motor protocol implementation may need verification against the exact LKMTECH MF5015 RS-485 protocol before real hardware use.
- Some GUI display strings show mojibake in terminal extraction, likely due to encoding display rather than necessarily file logic.

## Practical Next Steps

Likely replication path:

1. Add a measured-pose input abstraction: simulated FK, UDP camera pose, or recorded CSV replay.
2. Feed measured pose, not commanded-state FK, into the Kalman update and metrics.
3. Add optional low-pass filtering for pose and actuation.
4. Add Traj. 2 and Traj. 3.
5. Add a plant-model mismatch simulation mode to validate Kalman compensation without hardware.
6. Compare proposed vs PCC-only using identical trajectories and export Table-2-style metrics.
