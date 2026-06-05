# Continuum Manipulator GUI User Guide

## Purpose

This guide explains the adjustable parameters in the continuum manipulator GUI, especially the ellipse trajectory tracking controls in `continuum_ellipse.py`.

The GUI is based on the model-based control method from Zhai et al. 2025. The robot is a 3-segment tendon-driven continuum manipulator. Each segment is actuated by an antagonistic motor pair: when one tendon is pulled, the opposite tendon is released by the same amount.

The goal of the ellipse tracking mode is to move the manipulator tip along the reference path:

```text
x = 240 + 20 cos(2*pi*t/T) mm
y =  60 sin(2*pi*t/T) mm
psi = 0
```

This means the desired end-effector motion is:

```text
X range: 220 mm to 260 mm
Y range: -60 mm to +60 mm
Tip attitude psi: held near 0 degrees
```

## Quick Start Values

These are good starting values for a smooth simulation:

| Parameter           |                        Recommended value |
| ------------------- | ---------------------------------------: |
| Period T            |                                     40 s |
| Cycles              |                                        1 |
| Motor speed         |                                  40 mm/s |
| Gamma               |                                      0.5 |
| Beta                |                                      0.5 |
| Alpha               |                                      1.0 |
| Sigma P             |                                     0.35 |
| Sigma Psi           |                                      2.5 |
| Control rate        |                                    20 Hz |
| Kalman compensation |                                  Enabled |
| Measurement noise   | Disabled first, then enabled for testing |
| Noise sigma         |                                 0.0005 m |

## Parameter Reference

### Trajectory Type

**What it is**

This selects the desired path that the robot tip should follow.

Currently, the main supported trajectory is:

```text
Traj 1 - Ellipse
```

**Function**

It tells the controller where the end-effector should move over time.

**Example**

For the ellipse trajectory, the tip tries to move between:

```text
X = 220 mm to 260 mm
Y = -60 mm to +60 mm
```

**Why it is necessary**

Without a selected trajectory, the controller has no target path to follow.

---

### Period T

**What it is**

`Period T` is the time required to complete one full ellipse.

Unit:

```text
seconds
```

**Function**

It controls how fast the desired ellipse is traced.

**Example**

```text
T = 40 s
```

The robot tip takes 40 seconds to complete one ellipse.

```text
T = 20 s
```

The same ellipse is completed in 20 seconds, so the motion is twice as fast.

**If you increase it**

- The robot moves slower.
- The motors need less aggressive motion.
- Tracking is usually easier and smoother.
- The simulation or hardware is less likely to oscillate.

**If you decrease it**

- The robot moves faster.
- Motors must react more quickly.
- Tracking error may increase.
- The robot may look less smooth.

**Why it is necessary**

It defines the speed of the experiment.

**Practical advice**

Start with:

```text
T = 40 s
```

If the motion is stable, you can try reducing it gradually.

---

### Cycles

**What it is**

`Cycles` is the number of times the ellipse is repeated.

**Function**

It controls the duration of the experiment.

**Example**

```text
Cycles = 1
```

The robot completes the ellipse once.

```text
Cycles = 3
```

The robot repeats the same ellipse three times.

**If you increase it**

- The experiment runs longer.
- More data is collected.
- You can better observe repeated tracking performance.

**If you decrease it**

- The experiment ends sooner.
- Less tracking data is collected.

**Why it is necessary**

It is useful for comparing tracking accuracy over repeated runs.

---

### Motor Speed

**What it is**

`Motor speed` is the maximum commanded motor displacement speed during trajectory tracking.

Unit:

```text
mm/s
```

**Function**

It limits how quickly the capstan motors move the tendons.

**Example**

```text
Motor speed = 40 mm/s
```

The motors are commanded to move at up to 40 mm/s.

**If you increase it**

- Motors can reach commanded positions faster.
- Faster trajectories become easier to follow.
- Motion may become more aggressive.
- On real hardware, high values may increase mechanical stress.

**If you decrease it**

- Motion becomes smoother and safer.
- The robot may lag behind the desired trajectory.
- Tracking error may increase for fast trajectories.

**Why it is necessary**

The controller calculates target tendon displacements, but the motors still need a speed command to reach those targets.

**Practical advice**

Use moderate speed first:

```text
40 mm/s
```

If the robot cannot keep up, increase slowly.

---

### Gamma - Fading Rate

**What it is**

`Gamma` is used in the Kalman-based Jacobian error compensation.

Simple meaning:

```text
It controls how much old correction information is remembered.
```

**Function**

The controller starts with a mathematical PCC model. Real continuum robots are not perfectly identical to the model. The Kalman filter estimates the model error online. Gamma controls how quickly old error estimates fade.

**Example**

```text
Gamma = 0.5
```

Old correction information fades fairly quickly.

```text
Gamma = 0.9
```

Old correction information is remembered longer.

**If you increase it**

- The controller remembers previous corrections longer.
- This can help if the robot behavior is consistent.
- It may also keep wrong old corrections for too long.

**If you decrease it**

- Old correction information fades faster.
- The controller adapts more freshly.
- It may become less stable if it forgets useful information too quickly.

**Why it is necessary**

It helps balance memory and adaptation in the Kalman correction.

**Practical advice**

Start with:

```text
Gamma = 0.5
```

If correction feels too unstable, try a slightly higher value. If correction feels too slow to adapt, try a slightly lower value.

---

### Beta - Controller Gain

**What it is**

`Beta` controls how strongly the controller reacts to tracking error.

Simple meaning:

```text
It decides how large the correction step should be.
```

**Function**

When the tip is away from the desired point, the controller computes a correction. Beta scales that correction.

**Example**

If the tip is 10 mm away from the target:

- Small beta: the controller makes a small correction.
- Large beta: the controller makes a stronger correction.

**If you increase it**

- The robot reacts faster.
- Error may reduce more quickly.
- Too high a value may cause overshoot or oscillation.

**If you decrease it**

- Motion becomes smoother.
- Correction becomes slower.
- The robot may lag behind the ellipse.

**Why it is necessary**

It balances tracking speed and stability.

**Practical advice**

Start with:

```text
Beta = 0.5
```

If the robot is too slow, increase beta slightly. If it oscillates, decrease beta.

---

### Alpha - Regularisation

**What it is**

`Alpha` is the damping or regularisation term in the controller's Jacobian inverse calculation.

Simple meaning:

```text
It prevents unstable or extremely large motor corrections.
```

**Function**

The controller uses a Jacobian to convert desired tip motion into tendon displacement commands. Sometimes this inverse calculation can become sensitive. Alpha makes the calculation safer.

**If you increase it**

- The controller becomes more conservative.
- Motor commands become smaller and smoother.
- Tracking may become slower.

**If you decrease it**

- The controller becomes more aggressive.
- It may track more strongly if the model is good.
- Too low a value can create unstable or large corrections.

**Why it is necessary**

Continuum robot kinematics can be sensitive. Alpha protects the controller from unstable inverse calculations.

**Practical advice**

Start with:

```text
Alpha = 1.0
```

Increase it if the motor commands look too aggressive. Decrease it only if tracking is too weak and stable.

---

### Sigma P - Position Threshold

**What it is**

`Sigma P` limits how much the Kalman-corrected position Jacobian is allowed to change.

Position means:

```text
x and y tip position
```

**Function**

It prevents the Kalman filter from making unrealistic changes to the position part of the model.

**Example**

Suppose the model predicts that a tendon change moves the tip by a small amount, but noisy data suggests a very large movement. Sigma P limits how much the controller is allowed to believe that sudden change.

**If you increase it**

- The position Jacobian can change more freely.
- The controller may adapt faster.
- Too high a value may allow noisy or unrealistic corrections.

**If you decrease it**

- The correction is more restricted.
- The controller becomes safer and smoother.
- Adaptation may become too slow.

**Why it is necessary**

It protects the controller from unrealistic position-model updates.

**Practical advice**

Start with:

```text
Sigma P = 0.35
```

If adaptation is too weak, increase slightly. If correction looks noisy, decrease slightly.

---

### Sigma Psi - Attitude Threshold

**What it is**

`Sigma Psi` limits how much the Kalman-corrected attitude Jacobian is allowed to change.

Attitude means:

```text
psi, the end-effector orientation angle
```

**Function**

The ellipse trajectory tries to keep:

```text
psi = 0
```

Sigma Psi controls how much the controller is allowed to update the model related to that angle.

**If you increase it**

- The attitude model can change more freely.
- The controller may adapt faster to orientation error.
- Too high a value can make orientation correction unstable.

**If you decrease it**

- The attitude correction is more restricted.
- The controller becomes safer.
- It may adapt too slowly to orientation error.

**Why it is necessary**

It prevents unrealistic changes to the orientation part of the Jacobian.

**Practical advice**

Start with:

```text
Sigma Psi = 2.5
```

Only tune it if the orientation tracking looks unstable or too slow.

---

### Control Rate

**What it is**

`Control rate` is how many times per second the controller updates.

Unit:

```text
Hz
```

**Function**

Each update cycle does this:

1. Computes the desired tip position.
2. Estimates the current tip position.
3. Computes the tracking error.
4. Calculates new motor/tendon commands.

**Example**

```text
Control rate = 20 Hz
```

The controller updates 20 times per second.

That means one update every:

```text
1 / 20 = 0.05 s
```

**If you increase it**

- The controller reacts more frequently.
- Motion can become smoother in simulation.
- More computation is required.
- Real hardware communication may struggle if the rate is too high.

**If you decrease it**

- Less computation and communication load.
- Motion may become less smooth.
- Tracking error may increase.

**Why it is necessary**

The controller works in discrete time. It needs a fixed update rate.

**Practical advice**

Start with:

```text
20 Hz
```

Use higher values only if the computer and hardware communication can keep up.

---

### Enable Kalman Compensation

**What it is**

This turns the online Jacobian error compensation on or off.

**Function**

When enabled, the controller uses:

```text
PCC model + Kalman correction
```

When disabled, the controller uses:

```text
PCC model only
```

**If enabled**

- The controller can compensate for model error.
- This better represents the method described in the paper.
- Tracking may improve when the model is imperfect.

**If disabled**

- The controller uses only the theoretical PCC model.
- This is simpler.
- It is useful for comparison.
- Tracking error may be larger.

**Why it is necessary**

The paper's main contribution is using online Jacobian error compensation to improve control accuracy.

**Practical example**

PCC only:

```text
I fully trust the mathematical model.
```

PCC + Kalman:

```text
I use the mathematical model, but I correct it using observed behavior.
```

---

### Add Measurement Noise

**What it is**

This option adds artificial noise to the simulated measurement of the robot pose.

This is mainly for simulation testing.

**Function**

In ideal simulation, the controller can read the exact pose from the mathematical model. Real experiments are different. In the paper, the end-effector is measured using an infrared camera, and camera measurements are never perfectly noise-free.

This option imitates that real measurement uncertainty.

**Example**

True simulated tip position:

```text
x = 240.0 mm
y = 30.0 mm
```

With measurement noise, the controller may receive:

```text
x = 240.4 mm
y = 29.7 mm
```

**If enabled**

- Simulation becomes more realistic.
- Tracking may become less perfect.
- It tests whether the controller can handle noisy measurements.

**If disabled**

- Simulation is cleaner and ideal.
- Easier to understand basic controller behavior.
- Less realistic compared with camera-based experiments.

**Why it is necessary**

It helps test robustness before using real camera feedback.

**Practical advice**

First test with measurement noise disabled. After the controller behaves well, enable noise to test robustness.

---

### Noise Sigma

**What it is**

`Noise sigma` controls the size of the artificial measurement noise.

Unit:

```text
meters
```

Default example:

```text
0.0005 m = 0.5 mm
```

**Function**

It sets how much random error is added to the measured pose during simulation.

**Examples**

| Noise sigma | Meaning                   |
| ----------: | ------------------------- |
|    0.0001 m | 0.1 mm noise, very clean  |
|    0.0005 m | 0.5 mm noise, moderate    |
|    0.0020 m | 2.0 mm noise, quite noisy |

**If you increase it**

- Measurement becomes noisier.
- Tracking error may increase.
- The Kalman filter has a harder job.
- Useful for robustness testing.

**If you decrease it**

- Measurement becomes cleaner.
- Tracking becomes smoother.
- Simulation becomes less realistic.

**Why it is necessary**

It lets the user simulate different camera measurement qualities.

**Practical advice**

Start with:

```text
0.0005
```

If you want an ideal simulation, disable measurement noise instead of setting this to zero.

## How The Parameters Work Together

### Motion Parameters

These control the experiment itself:

```text
Period T
Cycles
Motor speed
```

Simple interpretation:

```text
How fast, how long, and how quickly the motors are allowed to move.
```

### Controller Parameters

These control the mathematical behavior of the controller:

```text
Gamma
Beta
Alpha
Sigma P
Sigma Psi
Control rate
Kalman compensation
```

Simple interpretation:

```text
How strongly and safely the controller reacts to tracking error.
```

### Simulation Realism Parameters

These control how realistic the simulated measurement is:

```text
Add measurement noise
Noise sigma
```

Simple interpretation:

```text
How imperfect the simulated camera measurement should be.
```

## Common Tuning Examples

### The robot moves too slowly

Try:

```text
Increase Beta slightly.
Increase Motor speed slightly.
Increase Period T only if you want the trajectory itself to be slower.
```

### The robot oscillates or overshoots

Try:

```text
Decrease Beta.
Increase Alpha.
Decrease Motor speed.
Increase Period T.
```

### Tracking error is high during fast motion

Try:

```text
Increase Motor speed.
Increase Control rate if hardware can support it.
Increase Period T to make the trajectory easier.
```

### Kalman correction looks noisy

Try:

```text
Decrease Sigma P.
Decrease Sigma Psi.
Increase Alpha.
Reduce measurement noise.
```

### Simulation is too ideal

Try:

```text
Enable Add measurement noise.
Use Noise sigma = 0.0005.
```

## Important Notes

### Antagonistic motor pairs

The manipulator uses three antagonistic motor pairs:

```text
Segment 1: M1 and M2
Segment 2: M3 and M4
Segment 3: M5 and M6
```

When one motor pulls a tendon, its paired motor releases the opposite tendon by the same amount.

Example:

```text
M1 = +10 mm
M2 = -10 mm
```

This represents one tendon shortening while the opposite tendon elongates equally.

### Spacer disk geometry

The model uses the paper geometry:

```text
3 segments
5 spacer disks per segment
1 end disk per segment
Total length = 270 mm
Segment length = 90 mm
Disk pitch = 15 mm
Manipulator diameter = 13 mm
```

### Top and side views

The side view shows the actual planar bending in the X-Y plane.

The top view shows the X-Z projection. Because this manipulator is planar in the current model:

```text
Z = 0
```

So the top view is mainly used to confirm that the manipulator does not move out of plane.

## Suggested Workflow

1. Start in simulation mode.
2. Use the recommended starting values.
3. Keep measurement noise disabled first.
4. Press `START TRACKING`.
5. Observe the ellipse path, side view, top view, motor panel, and tracking error.
6. Enable measurement noise and repeat the test.
7. Compare RMSE and MAE values.
8. Only after stable simulation, try hardware mode with conservative speed and period values.

## Safety Notes For Hardware Use

- Start with slow motion.
- Use a larger period, such as `T = 40 s` or higher.
- Use moderate motor speed.
- Watch the motor displacement values.
- Stop immediately if the robot moves unexpectedly.
- Do not use aggressive beta or very high motor speed until the system is verified.
