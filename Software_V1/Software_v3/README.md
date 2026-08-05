# Software V3

V3 is a clean, testable implementation of the Zhai et al. 2025 control method:

- PCC forward kinematics for the 3-segment planar continuum manipulator.
- Numerical PCC model Jacobian.
- 9-state Kalman filter for online Jacobian error compensation.
- Constrained Jacobian updates.
- Damped pseudoinverse control.
- Paper trajectory simulation and metric export.

## Code Structure

```text
Software_v3/
|-- README.md                         # V3 overview and run commands
|-- MODEL_IMPROVEMENT_WORKFLOW.md     # Paper-aligned implementation roadmap
|-- config.py                         # Paper parameters and controller safety limits
|-- model/
|   |-- pcc_kinematics.py             # PCC angles, forward kinematics, model Jacobian, motor/u mapping
|-- estimation/
|   |-- jacobian_kalman.py            # 9-state Kalman filter for online Jacobian error estimation
|-- control/
|   |-- jacobian_controller.py        # Jacobian compensation, constraints, damped inverse control
|-- io/
|   |-- filters.py                    # Low-pass filtering for pose and actuation signals
|-- experiments/
|   |-- trajectories.py               # Paper Traj. 1, Traj. 2, and Traj. 3 references
|   |-- simulate_paper_controller.py  # Hardware-free smoke simulation and metric export
```

Run the smoke simulation from the repository root:

```powershell
python -m Software_v3.experiments.simulate_paper_controller --trajectory traj3 --period 40
```

Export a CSV log:

```powershell
python -m Software_v3.experiments.simulate_paper_controller --trajectory traj3 --period 40 --csv Software_v3/experiments/traj3_smoke.csv
```

The simulation currently uses the PCC model as the virtual plant. That is intentional for the first V3 checkpoint: it verifies controller plumbing and metric logging before hardware feedback and calibration are connected.

The smoke-test metrics are not expected to reproduce the paper values yet. They are used to verify that the controller runs, logs, and stays bounded. Paper-level validation requires real or calibrated feedback, hardware travel limits, camera pose input, and the full experiment protocol in `MODEL_IMPROVEMENT_WORKFLOW.md`.
