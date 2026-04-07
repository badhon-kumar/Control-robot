"""
kinematics.py
=============
Forward and Inverse Kinematics for a 6-DOF Robot Arm
Using Denavit-Hartenberg (DH) Parameters + Jacobian Pseudoinverse IK

THEORY OVERVIEW:
────────────────
Forward Kinematics (FK):
    Given 6 joint angles → compute end-effector position (X, Y, Z) and orientation.
    Uses the DH convention: multiply 6 homogeneous transformation matrices T1×T2×...×T6.

Inverse Kinematics (IK):
    Given target (X, Y, Z) → compute the 6 joint angles needed.
    Uses iterative Jacobian pseudoinverse method:
      1. Compute current end-effector position via FK
      2. Compute error between current and target
      3. Compute Jacobian matrix (how each joint affects end-effector position)
      4. Use pseudoinverse to find joint angle updates: Δθ = J⁺ × Δx
      5. Apply joint angle updates, repeat until converged

IMPORTANT — DH Parameters:
────────────────────────────
These values MUST match your physical robot's link lengths and joint geometry.
The default values below are representative for a medium-sized 6-DOF arm.
Measure your actual robot and update the DH_PARAMS table before real use.

DH Parameter convention (modified DH):
    theta : joint angle (variable, set by motor) [radians]
    d     : link offset along previous z-axis [meters]
    a     : link length along previous x-axis [meters]
    alpha : twist angle between z-axes [radians]
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# DH PARAMETER TABLE
# Update these values to match your physical robot!
# ─────────────────────────────────────────────────────────
#
# Format per joint: [theta_offset_rad, d_m, a_m, alpha_rad]
#
#   theta_offset : added to the motor angle (home position offset)
#   d            : link offset along Z (metres)
#   a            : link length along X (metres)  ← measure this physically!
#   alpha        : twist between Z axes (radians)
#
# Example dimensions below approximate a ~600mm reach arm.
# Replace with your actual measurements.

DH_PARAMS = [
    # theta_offset   d        a       alpha
    [  0.0,         0.127,   0.0,    np.pi/2  ],  # Joint 1 — Base rotation
    [ -np.pi/2,     0.0,     0.300,  0.0      ],  # Joint 2 — Shoulder
    [  0.0,         0.0,     0.250,  0.0      ],  # Joint 3 — Elbow
    [  0.0,         0.102,   0.0,    np.pi/2  ],  # Joint 4 — Wrist 1
    [  0.0,         0.102,   0.0,   -np.pi/2  ],  # Joint 5 — Wrist 2
    [  0.0,         0.060,   0.0,    0.0      ],  # Joint 6 — Tool
]

# Joint limits in radians (must match DEFAULT_JOINT_CONFIG in robot_controller.py)
JOINT_LIMITS_RAD = [
    (-np.pi,       np.pi    ),   # Joint 1: ±180°
    (-np.pi/2,     np.pi/2  ),   # Joint 2: ±90°
    (-2*np.pi/3,   2*np.pi/3),   # Joint 3: ±120°
    (-np.pi,       np.pi    ),   # Joint 4: ±180°
    (-np.pi/2,     np.pi/2  ),   # Joint 5: ±90°
    (-2*np.pi,     2*np.pi  ),   # Joint 6: ±360°
]

# IK solver configuration
IK_MAX_ITERATIONS  = 200      # Max iterations before giving up
IK_TOLERANCE_M     = 0.0005   # 0.5mm position accuracy
IK_STEP_SIZE       = 0.5      # How aggressively to step (0–1, lower = more stable)
IK_DAMPING         = 0.01     # Levenberg-Marquardt damping (prevents singularities)


# ─────────────────────────────────────────────────────────
# RESULT DATA CLASSES
# ─────────────────────────────────────────────────────────

@dataclass
class FKResult:
    """Result from forward kinematics calculation."""
    position_m: np.ndarray        # [x, y, z] in metres
    rotation:   np.ndarray        # 3×3 rotation matrix
    transform:  np.ndarray        # Full 4×4 homogeneous transform
    joint_positions: list         # [x,y,z] of each joint (for visualisation)

    @property
    def x(self) -> float: return float(self.position_m[0])
    @property
    def y(self) -> float: return float(self.position_m[1])
    @property
    def z(self) -> float: return float(self.position_m[2])

    def position_mm(self) -> np.ndarray:
        return self.position_m * 1000.0

    def __str__(self):
        p = self.position_m * 1000
        return f"FK: X={p[0]:+.1f}mm  Y={p[1]:+.1f}mm  Z={p[2]:+.1f}mm"


@dataclass
class IKResult:
    """Result from inverse kinematics calculation."""
    success:      bool
    joint_angles_rad: np.ndarray    # 6 joint angles in radians
    joint_angles_deg: np.ndarray    # 6 joint angles in degrees
    iterations:   int
    final_error_m: float            # Achieved position error (metres)
    message:      str

    def __str__(self):
        if self.success:
            angles = "  ".join(f"J{i+1}:{a:+.1f}°" for i,a in enumerate(self.joint_angles_deg))
            return f"IK OK [{self.iterations} iters, err={self.final_error_m*1000:.2f}mm]: {angles}"
        return f"IK FAILED [{self.iterations} iters]: {self.message}"


# ─────────────────────────────────────────────────────────
# DH TRANSFORMATION MATRIX
# ─────────────────────────────────────────────────────────

def dh_matrix(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    """
    Compute a single DH homogeneous transformation matrix (4×4).

    This represents the transformation from frame i-1 to frame i.

    Parameters
    ----------
    theta : float   Joint angle (radians) — variable, set by motor
    d     : float   Link offset (metres)
    a     : float   Link length (metres)
    alpha : float   Twist angle (radians)
    """
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct,   -st*ca,   st*sa,   a*ct],
        [st,    ct*ca,  -ct*sa,   a*st],
        [0.0,   sa,      ca,      d   ],
        [0.0,   0.0,     0.0,     1.0 ],
    ])


# ─────────────────────────────────────────────────────────
# FORWARD KINEMATICS
# ─────────────────────────────────────────────────────────

def forward_kinematics(joint_angles_deg: list | np.ndarray) -> FKResult:
    """
    Compute end-effector position from 6 joint angles.

    Parameters
    ----------
    joint_angles_deg : list or ndarray of 6 floats (degrees)

    Returns
    -------
    FKResult with position [x, y, z] in metres and full transform matrix.
    """
    angles_rad = np.deg2rad(joint_angles_deg)
    T = np.eye(4)
    joint_positions = [[0.0, 0.0, 0.0]]   # base origin

    for i, (theta_off, d, a, alpha) in enumerate(DH_PARAMS):
        theta = angles_rad[i] + theta_off
        Ti = dh_matrix(theta, d, a, alpha)
        T = T @ Ti
        joint_positions.append([T[0,3], T[1,3], T[2,3]])

    return FKResult(
        position_m=T[:3, 3].copy(),
        rotation=T[:3, :3].copy(),
        transform=T.copy(),
        joint_positions=joint_positions,
    )


# ─────────────────────────────────────────────────────────
# JACOBIAN COMPUTATION
# ─────────────────────────────────────────────────────────

def compute_jacobian(joint_angles_deg: np.ndarray, delta: float = 0.001) -> np.ndarray:
    """
    Compute the 3×6 position Jacobian numerically using finite differences.

    The Jacobian J maps joint velocities to end-effector velocities:
        dx/dt = J × dθ/dt

    Each column i is: ∂[x,y,z]/∂θi  (how joint i affects end-effector position)

    Parameters
    ----------
    joint_angles_deg : ndarray of 6 floats (degrees)
    delta : float   Finite difference step (degrees)

    Returns
    -------
    J : ndarray shape (3, 6)
    """
    J = np.zeros((3, 6))
    base_pos = forward_kinematics(joint_angles_deg).position_m

    for i in range(6):
        perturbed = joint_angles_deg.copy()
        perturbed[i] += delta
        perturbed_pos = forward_kinematics(perturbed).position_m
        J[:, i] = (perturbed_pos - base_pos) / np.deg2rad(delta)

    return J


# ─────────────────────────────────────────────────────────
# INVERSE KINEMATICS
# ─────────────────────────────────────────────────────────

def inverse_kinematics(
    target_position_m: np.ndarray,
    initial_angles_deg: np.ndarray,
    max_iterations: int = IK_MAX_ITERATIONS,
    tolerance_m: float = IK_TOLERANCE_M,
    step_size: float = IK_STEP_SIZE,
    damping: float = IK_DAMPING,
) -> IKResult:
    """
    Solve inverse kinematics using damped least squares (Levenberg-Marquardt).

    Finds joint angles such that FK(angles) ≈ target_position.

    Parameters
    ----------
    target_position_m  : ndarray [x, y, z] in metres
    initial_angles_deg : ndarray of 6 starting joint angles (degrees)
                         Use current motor positions for best convergence.
    max_iterations     : int
    tolerance_m        : float  Acceptable position error (metres)
    step_size          : float  Step scale factor (0–1)
    damping            : float  Damping factor λ for numerical stability

    Returns
    -------
    IKResult
    """
    angles = initial_angles_deg.copy().astype(float)
    target = np.array(target_position_m, dtype=float)

    for iteration in range(max_iterations):
        # Current end-effector position
        current_pos = forward_kinematics(angles).position_m

        # Position error vector
        error = target - current_pos
        error_norm = np.linalg.norm(error)

        # Check convergence
        if error_norm < tolerance_m:
            return IKResult(
                success=True,
                joint_angles_rad=np.deg2rad(angles),
                joint_angles_deg=angles.copy(),
                iterations=iteration + 1,
                final_error_m=float(error_norm),
                message=f"Converged in {iteration+1} iterations",
            )

        # Compute Jacobian
        J = compute_jacobian(angles)

        # Damped least squares pseudoinverse: J⁺ = Jᵀ(JJᵀ + λ²I)⁻¹
        JJt = J @ J.T
        damped = JJt + (damping ** 2) * np.eye(3)
        J_pinv = J.T @ np.linalg.inv(damped)   # shape (6, 3)

        # Joint angle update in degrees
        delta_angles_rad = J_pinv @ error
        delta_angles_deg = np.rad2deg(delta_angles_rad) * step_size

        # Apply update
        angles += delta_angles_deg

        # Clamp to joint limits
        for i, (lo, hi) in enumerate(JOINT_LIMITS_RAD):
            lo_deg, hi_deg = np.degrees(lo), np.degrees(hi)
            angles[i] = np.clip(angles[i], lo_deg, hi_deg)

    # Did not converge
    final_pos  = forward_kinematics(angles).position_m
    final_error = float(np.linalg.norm(target - final_pos))

    return IKResult(
        success=False,
        joint_angles_rad=np.deg2rad(angles),
        joint_angles_deg=angles.copy(),
        iterations=max_iterations,
        final_error_m=final_error,
        message=f"Did not converge. Error={final_error*1000:.1f}mm. "
                f"Target may be out of reach or near a singularity.",
    )


# ─────────────────────────────────────────────────────────
# CARTESIAN MOTION PLANNER
# ─────────────────────────────────────────────────────────

@dataclass
class CartesianMotionPlan:
    """
    Complete motion plan for a Cartesian displacement move.

    Contains all information needed to execute the move:
    - Target joint angles for each motor
    - Required speed for each motor (so all arrive at the same time)
    - Validation results
    """
    success:            bool
    message:            str

    # Inputs
    start_position_mm:  np.ndarray       # [x,y,z] start (mm)
    target_position_mm: np.ndarray       # [x,y,z] target (mm)
    displacement_mm:    np.ndarray       # [dx,dy,dz] (mm)
    time_s:             float            # Requested motion duration (s)

    # IK solution
    start_angles_deg:   np.ndarray       # 6 starting joint angles (deg)
    target_angles_deg:  np.ndarray       # 6 target joint angles (deg)

    # Per-joint motion parameters
    angle_changes_deg:  np.ndarray       # |Δθ| per joint (deg)
    speeds_dps:         np.ndarray       # Required speed per joint (deg/s)

    # Quality metrics
    ik_error_mm:        float            # IK position accuracy (mm)
    max_speed_dps:      float            # Fastest joint (deg/s)
    reachable:          bool             # Is target within robot workspace?

    def summary(self) -> str:
        if not self.success:
            return f"PLAN FAILED: {self.message}"
        lines = [
            f"Cartesian Motion Plan",
            f"  Start   : X={self.start_position_mm[0]:+.1f}  Y={self.start_position_mm[1]:+.1f}  Z={self.start_position_mm[2]:+.1f} mm",
            f"  Target  : X={self.target_position_mm[0]:+.1f}  Y={self.target_position_mm[1]:+.1f}  Z={self.target_position_mm[2]:+.1f} mm",
            f"  Time    : {self.time_s:.2f} s",
            f"  IK error: {self.ik_error_mm:.2f} mm",
            f"  Joint speeds:",
        ]
        for i in range(6):
            lines.append(
                f"    J{i+1}: {self.start_angles_deg[i]:+6.1f}° → "
                f"{self.target_angles_deg[i]:+6.1f}°  "
                f"(Δ={self.angle_changes_deg[i]:5.1f}°  "
                f"@ {self.speeds_dps[i]:6.1f}°/s)"
            )
        lines.append(f"  Fastest joint: {self.max_speed_dps:.1f}°/s")
        return "\n".join(lines)


# Per-joint hardware speed limits (deg/s) — matches DEFAULT_JOINT_CONFIG
JOINT_SPEED_LIMITS_DPS = [120.0, 90.0, 90.0, 180.0, 180.0, 200.0]
MIN_SPEED_DPS = 1.0   # Minimum meaningful speed


def plan_cartesian_motion(
    displacement_mm: list | np.ndarray,
    time_s: float,
    current_angles_deg: list | np.ndarray,
    mode: str = "relative",
    absolute_target_mm: list | np.ndarray = None,
) -> CartesianMotionPlan:
    """
    Plan a Cartesian motion: compute per-joint speeds for time-parameterized motion.

    The key principle:
        speed_j = |angle_change_j| / time_s

    Every joint gets a different speed, but ALL joints finish at exactly time_s.

    Parameters
    ----------
    displacement_mm : [dx, dy, dz] relative displacement in mm (used if mode='relative')
    time_s          : motion duration in seconds (independent variable, user-controlled)
    current_angles_deg : current 6 joint angles in degrees
    mode            : 'relative' (displace from current) or 'absolute' (go to exact point)
    absolute_target_mm : [x, y, z] in mm — used only when mode='absolute'

    Returns
    -------
    CartesianMotionPlan
    """
    if time_s <= 0:
        return _failed_plan("Time must be greater than 0.", current_angles_deg)

    current_angles = np.array(current_angles_deg, dtype=float)

    # ── Compute start position via FK ──────────────────────────────────────
    fk_start = forward_kinematics(current_angles)
    start_mm  = fk_start.position_m * 1000.0

    # ── Determine target position ─────────────────────────────────────────
    if mode == "absolute":
        if absolute_target_mm is None:
            return _failed_plan("Absolute target not provided.", current_angles)
        target_mm = np.array(absolute_target_mm, dtype=float)
        disp_mm   = target_mm - start_mm
    else:
        disp_mm   = np.array(displacement_mm, dtype=float)
        target_mm = start_mm + disp_mm

    target_m = target_mm / 1000.0

    # ── Reachability check (simple sphere check) ──────────────────────────
    # Max reach ≈ sum of all link lengths
    max_reach_m = sum(abs(p[2]) + abs(p[3]) for p in DH_PARAMS)
    dist_from_base = np.linalg.norm(target_m)
    reachable = dist_from_base < max_reach_m * 0.95   # 5% safety margin

    if not reachable:
        return _failed_plan(
            f"Target is out of reach. Distance={dist_from_base*1000:.0f}mm, "
            f"Max reach≈{max_reach_m*1000:.0f}mm.",
            current_angles,
        )

    # ── Solve IK ──────────────────────────────────────────────────────────
    logger.info(f"IK: Solving for target {target_mm} mm...")
    ik = inverse_kinematics(target_m, current_angles)

    if not ik.success and ik.final_error_m > 0.005:   # Allow up to 5mm if not fully converged
        return _failed_plan(f"IK failed: {ik.message}", current_angles)

    target_angles = ik.joint_angles_deg

    # ── Compute per-joint speeds ──────────────────────────────────────────
    angle_changes = np.abs(target_angles - current_angles)
    speeds = angle_changes / time_s           # degrees per second per joint

    # Clamp to hardware limits
    speed_violations = []
    for i in range(6):
        limit = JOINT_SPEED_LIMITS_DPS[i]
        if speeds[i] > limit:
            speed_violations.append(
                f"J{i+1} needs {speeds[i]:.1f}°/s but limit is {limit:.1f}°/s"
            )
        speeds[i] = max(MIN_SPEED_DPS, min(speeds[i], limit))

    if speed_violations:
        violation_str = "; ".join(speed_violations)
        logger.warning(f"Speed limit exceeded — motion will be slower than requested: {violation_str}")

    plan = CartesianMotionPlan(
        success=True,
        message="Plan OK" + (f" [SPEED CLAMPED: {speed_violations[0]}]" if speed_violations else ""),
        start_position_mm=start_mm,
        target_position_mm=target_mm,
        displacement_mm=disp_mm,
        time_s=time_s,
        start_angles_deg=current_angles.copy(),
        target_angles_deg=target_angles,
        angle_changes_deg=angle_changes,
        speeds_dps=speeds,
        ik_error_mm=ik.final_error_m * 1000.0,
        max_speed_dps=float(np.max(speeds)),
        reachable=reachable,
    )

    logger.info(f"Motion plan ready:\n{plan.summary()}")
    return plan


def _failed_plan(msg: str, current_angles: np.ndarray) -> CartesianMotionPlan:
    zero6 = np.zeros(6)
    cur   = np.array(current_angles, dtype=float)
    return CartesianMotionPlan(
        success=False, message=msg,
        start_position_mm=zero6, target_position_mm=zero6, displacement_mm=zero6,
        time_s=0, start_angles_deg=cur, target_angles_deg=cur,
        angle_changes_deg=zero6, speeds_dps=zero6,
        ik_error_mm=0, max_speed_dps=0, reachable=False,
    )


# ─────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== Forward Kinematics Test ===")
    home = [0, 0, 0, 0, 0, 0]
    fk = forward_kinematics(home)
    print(f"Home position: {fk}")

    print("\n=== Cartesian Motion Plan Test ===")
    plan = plan_cartesian_motion(
        displacement_mm=[50, 0, -30],
        time_s=3.0,
        current_angles_deg=home,
    )
    print(plan.summary())
