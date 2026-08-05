# Continuum Manipulator Continuum_v3

This folder starts from `Software_V2/continuum_ellipse.py` and adds the first real end-effector feedback path for the 3-segment planar tendon-driven continuum manipulator based on Zhai et al. 2025, "Model-Based Control of a Continuum Manipulator with Online Jacobian Error Compensation Using Kalman Filtering."

## Programs

| File | Purpose | Hardware interface |
| :--- | :--- | :--- |
| `continuum_ellipse.py` | Ellipse and G-code trajectory tracking with PCC + Kalman Jacobian compensation and UDP end-effector pose feedback | RS-485 serial |
| `pose_feedback.py` | UDP pose receiver and 4th-order low-pass filtering for measured `[x, y, psi]` | UDP JSON packets |
| `sample_square_path.gcode` | Example square G-code path for simulation/testing | Local file |
| `sample_circle_path.gcode` | Example circular G-code path using `G3` arcs | Local file |
| `G_CODE_GUIDE.md` | Plain-language guide to supported G-code commands | Documentation |

The robot concept remains the same: 3 segments, 6 tendons, and 3 antagonistic tendon pairs.

## User Guide

For a plain-language explanation of the adjustable GUI parameters, including `Period T`, `Cycles`, `Motor speed`, Kalman controller parameters, control rate, and measurement noise settings, see:

```text
USER_GUIDE.md
```

For simple G-code syntax and examples, see:

```text
G_CODE_GUIDE.md
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
pip install -r Continuum_v3/requirements.txt
```

`tkinter` is also required, but it is included with most standard Python installations.

---

## Run `continuum_ellipse.py`

Simulation mode:

```powershell
python Continuum_v3/continuum_ellipse.py
```

Real hardware with UDP pose feedback enabled:

```powershell
python Continuum_v3/continuum_ellipse.py --real --port COM3 --pose-host 127.0.0.1 --pose-port 5005
```

Real hardware without UDP pose feedback, for fallback/debug only:

```powershell
python Continuum_v3/continuum_ellipse.py --real --port COM3 --no-pose-udp
```

Arguments:

| Argument | Description |
| :--- | :--- |
| none | Run in simulation mode. |
| `--real` | Connect to real hardware. |
| `--port COM3` | Select hardware port. Default is `COM3`. |
| `--baud 115200` | Select RS-485 baud rate. |
| `--pose-host 127.0.0.1` | UDP host/IP for measured end-effector pose feedback. |
| `--pose-port 5005` | UDP port for measured end-effector pose feedback. |
| `--no-pose-udp` | Disable UDP measured pose feedback in real hardware mode. |

## UDP Pose Feedback

The controller expects fresh measured end-effector pose packets at around `30 Hz`. Send JSON over UDP:

```json
{"x_mm": 240.0, "y_mm": 10.0, "psi_deg": 5.0, "confidence": 1.0}
```

or in SI units:

```json
{"x_m": 0.240, "y_m": 0.010, "psi_rad": 0.0873, "confidence": 1.0}
```

When a fresh UDP pose is available, real hardware mode uses it as `pose_curr` in the Kalman Jacobian update and control law. If the pose is stale or missing, the GUI warns and temporarily falls back to the PCC model pose so the run does not crash.

## G-code Path Following

In simulation mode:

1. Run the GUI:

   ```powershell
   python Continuum_v3/continuum_ellipse.py
   ```

2. In the trajectory panel, set `Type` to `G-code path`.
3. Choose a `.gcode` file from the dropdown, or use `Browse File` to pick one from any folder.
4. Press `Load G-code`.
5. Press `START TRACKING`.

Supported motion commands include `G0`, `G1`, `G2`, `G3`, `G4`, `G20`, `G21`, `G90`, and `G91`. Position uses `X`/`Y`; attitude uses `A` in degrees. Circular arcs use `I`/`J` center offsets.

The G-code parser is built into `continuum_ellipse.py`, so no separate helper script needs to be run.

Example:

```gcode
G21
G90
F600
G0 X260 Y0 A0
G1 X250 Y35 A5
G1 X230 Y35 A0
G1 X220 Y0 A-5
```

Arc example:

```gcode
G0 X260 Y20 A0
G3 X220 Y20 I-20 J0 A0
G3 X260 Y20 I20 J0 A0
```

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
python Continuum_v3/continuum_ellipse.py
```

List serial ports:

```powershell
python Continuum_v3/continuum_ellipse.py --list-ports
```

Real hardware:

```powershell
python Continuum_v3/continuum_ellipse.py --real --port COM3 --baud 115200
```

Arguments:

| Argument | Description |
| :--- | :--- |
| none | Run in simulation mode. |
| `--real` | Connect to real hardware through RS-485. |
| `--port COM3` | Select serial port. Default is `COM3`. |
| `--baud 115200` | Select RS-485 baud rate. Default is `115200`. |
| `--pose-host 127.0.0.1` | UDP host/IP for measured end-effector pose feedback. |
| `--pose-port 5005` | UDP port for measured end-effector pose feedback. |
| `--no-pose-udp` | Disable measured UDP pose feedback in real hardware mode. |
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

## CSV Format for Legacy State Playback

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
