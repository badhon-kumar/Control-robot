# Continuum Manipulator Software V2

This folder contains two GUI programs for the 3-segment planar tendon-driven continuum manipulator based on Zhai et al. 2025, "Model-Based Control of a Continuum Manipulator with Online Jacobian Error Compensation Using Kalman Filtering."

## Programs

| File | Purpose | Hardware interface |
| :--- | :--- | :--- |
| `continuum_gui_v1.py` | Manual motor jog, CSV state sequencing, capstan/body visualization | CAN bus via `can_interface.py` / `robot_controller.py` |
| `continuum_ellipse.py` | Ellipse trajectory tracking with PCC + Kalman Jacobian compensation | RS-485 serial |

Both programs use the same robot concept: 3 segments, 6 tendons, and 3 antagonistic tendon pairs.

## User Guide

For a plain-language explanation of the adjustable GUI parameters, including `Period T`, `Cycles`, `Motor speed`, Kalman controller parameters, control rate, and measurement noise settings, see:

```text
USER_GUIDE.md
```

Paper mechanism used in the visualizers:

- Total manipulator length: `270 mm`.
- Segment length: `90 mm` each.
- Manipulator diameter: `13 mm`.
- Each segment has `5` spacer disks and `1` end disk.
- There is also an additional base disk.

| Segment | Motors | Rule |
| :--- | :--- | :--- |
| Segment 1 | M1 / M2 | If one pulls, the other releases by the same amount. |
| Segment 2 | M3 / M4 | If one pulls, the other releases by the same amount. |
| Segment 3 | M5 / M6 | If one pulls, the other releases by the same amount. |

---

## Install

Install dependencies from the folder:

```powershell
pip install -r Software_V2/requirements.txt
```

`tkinter` is also required, but it is included with most standard Python installations.

---

## Run `continuum_gui_v1.py`

Use this GUI for manual motor control, CSV state playback, and visualization.

Simulation mode:

```powershell
python Software_V2/continuum_gui_v1.py
```

Real hardware:

```powershell
python Software_V2/continuum_gui_v1.py --real --port COM3
```

Optional CAN bus type:

```powershell
python Software_V2/continuum_gui_v1.py --real --port COM3 --bustype slcan
```

Arguments:

| Argument | Description |
| :--- | :--- |
| none | Run in simulation mode. |
| `--real` | Connect to real hardware. |
| `--port COM3` | Select hardware port. Default is `COM3`. |
| `--bustype slcan` | Select CAN backend. Options: `slcan`, `pcan`, `kvaser`, `socketcan`. |

### Manual Jog Rules

Manual Jog gives access to all 6 motors, but operator commands are positive pull magnitudes only.

- Press `+` to continuously increase the selected motor's pull.
- Press `+S` to apply one positive step.
- Enter a positive value in `Pull mm` and press `SEND`.
- The selected motor pulls by that amount.
- Its antagonistic partner automatically releases by the same amount.
- `HOME ALL` / `ZERO ALL` returns all motor commands to `0`.

Do not enter negative values. The GUI creates the negative paired command internally.

### CSV Sequencer Rules

CSV files should also use positive-only commands. For each antagonistic pair, only one motor in the pair should have a positive displacement in a row; the partner should be `0`.

Valid example:

```csv
state_name,delay_s,m1_disp_mm,m1_speed_mms,m2_disp_mm,m2_speed_mms
S1_M1_Pull,2.0,30,25,0,25
S1_M2_Pull,2.0,0,25,30,25
```

Invalid example:

```csv
Bad_Row,2.0,30,25,30,25
```

The invalid row commands both motors of the same antagonistic pair at the same time.

`sample_states.csv` already follows the positive-only convention.

CSV panel visibility:

- Before loading CSV: file controls and compact playback area.
- After loading CSV/sample: state table and timing panel appear.
- During playback: `RUN` and `STEP` are hidden, while `PAUSE` and `ABORT` are shown.
- After completion or abort: the panel returns to idle controls.

### Visualization

- Capstan panel shows signed motor commands: positive means pull, negative means release.
- Side view shows the continuum body, disks, and internal tendon routing.
- Top view shows tendon paths inside the manipulator through disk holes, matching the paper's routing concept instead of external cables.
- Pose HUD displays tip position, attitude, and segment bending angles.

---

## Run `continuum_ellipse.py`

Use this GUI for closed-loop ellipse trajectory tracking with the PCC model and online Kalman Jacobian error compensation.

Simulation mode:

```powershell
python Software_V2/continuum_ellipse.py
```

List serial ports:

```powershell
python Software_V2/continuum_ellipse.py --list-ports
```

Real hardware:

```powershell
python Software_V2/continuum_ellipse.py --real --port COM3 --baud 115200
```

Arguments:

| Argument | Description |
| :--- | :--- |
| none | Run in simulation mode. |
| `--real` | Connect to real hardware through RS-485. |
| `--port COM3` | Select serial port. Default is `COM3`. |
| `--baud 115200` | Select RS-485 baud rate. Default is `115200`. |
| `--list-ports` | List available serial ports and exit. |

### Ellipse Tracking Panel

- `START TRACKING` runs the reference ellipse trajectory.
- `STOP` stops the active tracking run.
- The reference trajectory follows Zhai et al. Traj. 1 style ellipse tracking with horizontal attitude.
- The controller uses PCC kinematics with optional online Kalman Jacobian error compensation.
- Trajectory history can be exported to CSV after a run.

Useful parameters:

- `Period T (s)`: full ellipse cycle time. Larger values are slower and usually more accurate.
- `Cycles`: number of ellipse cycles to execute.
- `Motor speed (mm/s)`: capstan/tendon speed limit.
- `gamma`: Kalman fading rate.
- `beta`: controller gain.
- `alpha`: damped pseudoinverse regularization.
- `sigma_p`, `sigma_psi`: Jacobian update constraints.
- `Control rate (Hz)`: controller update rate.

The paper assumes quasi-static motion, so slow trajectories are safer and should track better.

---

## CSV Format for `continuum_gui_v1.py`

The expected CSV columns are:

```csv
state_name,delay_s,
m1_disp_mm,m1_speed_mms,
m2_disp_mm,m2_speed_mms,
m3_disp_mm,m3_speed_mms,
m4_disp_mm,m4_speed_mms,
m5_disp_mm,m5_speed_mms,
m6_disp_mm,m6_speed_mms
```

Rules:

- All `m*_disp_mm` values should be `>= 0`.
- For each pair `(M1,M2)`, `(M3,M4)`, `(M5,M6)`, command at most one motor with a positive displacement per row.
- Use `0` for the partner motor.
- Speeds are positive in `mm/s`.

---

## Safety Notes

- Home the system before running manual, CSV, or trajectory commands.
- Use small pull values first when testing a new tendon routing or motor direction.
- Use `E-STOP`, `STOP ALL`, or trajectory `STOP` if motion is unexpected.
- Keep early tests slow. The model and paper assume quasi-static motion.
