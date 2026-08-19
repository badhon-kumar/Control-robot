"""
Single source of geometry / material parameters for the MuJoCo continuum model.

Geometry values mirror `Continuum_v3/continuum_ellipse.py` (L_SEG, R_TENDON,
SPACER_DISKS_PER_SEG, END_DISKS_PER_SEG) and the paper mechanism described in
Continuum_v3/README.md. `check_phase1.py` cross-checks these against the
controller module when it is importable, so the two cannot silently drift apart.

All values SI (metres, radians, kg).

Author: Badhon Kumar
"""

import math

# ── Kinematic geometry (mirrors continuum_ellipse.py) ────────────────────────
N_SEG = 3
L_SEG = [0.09, 0.09, 0.09]              # segment lengths, m           (L_SEG)
TOTAL_LEN = sum(L_SEG)                  # 0.270 m

# R_TENDON is indexed by HOW MANY SEGMENTS BACK the tendon currently is, not by
# segment number. A tendon runs at R_TENDON[0] = 5 mm inside its own segment,
# at R_TENDON[1] = 3.5 mm through the segment one below it, and at
# R_TENDON[2] = 2 mm two below.
#
# This is not an assumption - it is forced by pcc_angles() in
# continuum_ellipse.py (paper Eqs. 7-9), which divides EVERY segment by r[0]
# and uses r[1], r[2] only as coupling coefficients:
#     theta1 = dl1 / r[0]
#     theta2 = (dl2 - r[1]*theta1) / r[0]
#     theta3 = (dl3 - r[2]*theta1 - r[1]*theta2) / r[0]
# i.e. dl2 = r[1]*theta1 + r[0]*theta2, so segment 2's tendon has a 3.5 mm
# moment arm in segment 1 and a 5 mm arm in its own segment.
#
# It also matches the paper's Fig. 2C: tendons "are attached only to the end
# disk of the respective segment and slide through all preceding disks via
# specifically designed holes", with "offsets ... adjusted along the backbone
# to avoid interaction and friction between tendons".
R_TENDON = [0.005, 0.0035, 0.002]       # m, indexed by segments-back

SPACER_DISKS_PER_SEG = 5                # (SPACER_DISKS_PER_SEG)
END_DISKS_PER_SEG = 1                   # (END_DISKS_PER_SEG)
DISKS_PER_SEG = SPACER_DISKS_PER_SEG + END_DISKS_PER_SEG   # 6

# One rigid link per disk, so link boundaries land exactly on physical disks.
LINKS_PER_SEG = DISKS_PER_SEG           # 6
N_LINKS = N_SEG * LINKS_PER_SEG         # 18
LINK_LEN = L_SEG[0] / LINKS_PER_SEG     # 0.015 m axial disk pitch

# ── Physical cross-section ───────────────────────────────────────────────────
BODY_DIAM = 0.013                       # manipulator diameter, m (README)
DISK_RADIUS = BODY_DIAM / 2             # 0.0065 m
DISK_THICK = 0.0015                     # spacer disk thickness, m
BACKBONE_RADIUS = 0.0015                # central backbone rod radius, m
BASE_DISK_RADIUS = 0.009
BASE_DISK_THICK = 0.003

# Backbone radius must stay inside the smallest tendon routing radius, otherwise
# segment-3 tendons would be routed through the middle of the backbone geom.
assert BACKBONE_RADIUS < min(R_TENDON), "backbone too thick for segment-3 tendon radius"
assert max(R_TENDON) < DISK_RADIUS, "tendon radius must fit inside the disks"

# ── Materials ────────────────────────────────────────────────────────────────
BACKBONE_DENSITY = 6450.0               # NiTi, kg/m^3
DISK_DENSITY = 1400.0                   # printed polymer, kg/m^3

# ── Elastic properties — PLACEHOLDER, fitted in Phase 3 ──────────────────────
#
# Derived, not guessed: a discretized elastic rod of flexural rigidity EI split
# into links of length ds behaves as torsional springs of k = EI / ds.
# Choosing k so that a full-workspace bend (segment angle ~1 rad, i.e. ~0.167 rad
# per joint) needs a realistic tendon force of ~20 N at the segment-1 moment arm
# of 5 mm gives k = (20 N * 0.005 m) / 0.167 rad = 0.6 N*m/rad, which implies
# EI = 0.009 N*m^2 and E ~ 2.3 GPa for this backbone radius - a plausible
# polymer/PEEK backbone. Phase 3 replaces this with a fitted value.
JOINT_STIFFNESS = 0.6                   # N*m/rad, per joint
JOINT_DAMPING = 0.02                    # N*m*s/rad

# Fictitious rotor inertia added to every joint. The distal links have real
# inertia ~1e-7 kg*m^2, which against k=0.6 gives ~2500 rad/s natural frequency
# and would need an impractically small timestep. Armature raises the effective
# inertia for integration stability. It does NOT affect static equilibrium, so
# under the paper's quasi-static assumption it does not bias any result.
#
# Sized from the ACTUATOR mode, which is the stiffest one in the model:
# omega ~ sqrt(kp*r^2 / armature). At the original 1e-5 that is ~790 rad/s, so
# omega*dt ~ 1.58 at a 2 ms timestep - past clean integration. The arm then
# sustained a 75 um/step tip buzz that never decayed, which both defeated any
# settling test and injected 75 um of noise into the pose the Kalman filter
# consumes. At 1e-4 the jitter is exactly zero.
#
# Verified not to bias statics: armature 1e-4, 4e-4 and 1e-3 at dt=2 ms and
# armature 1e-5 at dt=0.5 ms all settle to the same tip y = 97.8660 mm, whereas
# the buzzing 1e-5 / 2 ms case reported 97.8921 mm - the 26 um discrepancy was
# the artifact, not the fix.
JOINT_ARMATURE = 1e-4

# ── Tendons and actuators (Phase 2) ──────────────────────────────────────────
#
# Motor numbering follows continuum_ellipse.py exactly:
#   M1/M2 -> segment 1, M3/M4 -> segment 2, M5/M6 -> segment 3
#   odd motors (M1,M3,M5) route on +y, even motors on -y
# so u_i > 0 pulls the odd motor of pair i and releases its partner.
N_MOTORS = 6
MOTOR_TO_SEG = [0, 0, 1, 1, 2, 2]
MOTOR_SIGN = [+1, -1, +1, -1, +1, -1]

TENDON_WIDTH = 0.0006                   # visual only, m

# Position actuator on tendon length: ctrl IS the commanded tendon length, so
# it models a position-controlled capstan (the LKM motors' native mode).
# kp must be stiff relative to the arm's effective tendon stiffness
# k/(n*r^2) ~ 4000 N/m for segment 1, or commanded displacement is not achieved.
# Verified against achieved-vs-commanded length in check_phase2.py.
TENDON_KP = 250000.0                    # N/m

# Tendons pull, never push. A position actuator produces f = kp*(ctrl - len),
# so shortening (ctrl < len) gives NEGATIVE force - hence a pull-only actuator
# is forcerange = [-Fmax, 0]. A released tendon then simply goes slack at zero
# force, which is the physically correct antagonistic behaviour.
#
# Sized from measurement, not guessed: with the clamp lifted, spanning the print
# workspace demands a peak of 34.7 N (median 9.9 N) at nominal stiffness. 80 N
# leaves headroom to ~2x stiffness while staying plausible for a capstan drive.
#
# This limit MUST NOT bind during normal operation. An earlier 60 N value
# saturated at the workspace corners and silently corrupted the Phase 3
# stiffness study - it reported a 168% sensitivity to stiffness that was really
# just a clipped actuator (the true figure is 8.7%). check_phase3.py now tests
# for saturation explicitly so that failure mode cannot recur unnoticed.
TENDON_FORCE_MAX = 80.0                 # N
TENDON_CTRL_MARGIN = 0.02               # m of slack/pull allowed either side


def tendon_radius(motor: int, disk_segment: int) -> float:
    """
    Signed routing radius of motor `motor` where it passes through segment
    `disk_segment`. Uses the segments-back indexing of R_TENDON.
    """
    back = MOTOR_TO_SEG[motor] - disk_segment
    if back < 0:
        raise ValueError(f"motor {motor+1} terminates before segment {disk_segment+1}")
    return R_TENDON[back] * MOTOR_SIGN[motor]


def nominal_rest_length(k: int) -> float:
    """
    Nominal undeformed tendon length: base to its segment's end disk.

    The true rest length is slightly longer because the routing radius steps at
    each segment boundary, adding a small radial jog. Always prefer the value
    read from the compiled model at qpos = 0 (see rest_lengths_from_model);
    this is only a sanity reference.
    """
    return sum(L_SEG[: MOTOR_TO_SEG[k] + 1])


def rest_lengths_from_model(model, data, mujoco):
    """Exact undeformed tendon lengths, measured from the compiled model."""
    import numpy as _np
    qpos = data.qpos.copy()
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)
    rest = data.ten_length.copy()
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    return _np.asarray(rest)


def u_to_tendon_lengths(u, rest):
    """
    Map control input u (3 signed per-segment tendon displacements, metres) to
    the 6 commanded tendon lengths.

    Positive u_i pulls the odd motor of pair i by u_i and releases the even
    partner by the same amount - the antagonistic rule enforced by the GUI in
    continuum_ellipse.py. Enforced here in the wrapper rather than in the MJCF
    so the model stays a plain 6-tendon mechanism.

    `rest` must be the measured rest lengths, not the nominal ones.
    """
    return [rest[k] - MOTOR_SIGN[k] * u[MOTOR_TO_SEG[k]] for k in range(N_MOTORS)]


# ── Simulation options ───────────────────────────────────────────────────────
TIMESTEP = 0.002
# Phase 1 is calibrated purely elastically, so gravity starts disabled.
# See PLAN.md Phase 1 step 5 before enabling "0 -9.81 0".
GRAVITY = (0.0, 0.0, 0.0)

# Contacts are disabled everywhere: the agreed scope is tip-path only, with no
# material contact physics. This also removes self-collision between adjacent
# disks and makes the solve faster and more stable.
ENABLE_CONTACTS = False

# ── Visual ───────────────────────────────────────────────────────────────────
SEG_COLORS = [
    "0.16 0.50 0.73 1",                 # segment 1 - blue
    "0.18 0.62 0.44 1",                 # segment 2 - green
    "0.85 0.45 0.13 1",                 # segment 3 - orange
]
DISK_COLOR = "0.35 0.38 0.42 1"
BASE_COLOR = "0.20 0.22 0.25 1"
TIP_COLOR = "0.85 0.15 0.15 1"


def segment_of_link(i: int) -> int:
    """0-based segment index for 0-based link index i."""
    return i // LINKS_PER_SEG


def is_end_disk(i: int) -> bool:
    """True if link i carries a segment end disk (rather than a spacer disk)."""
    return (i % LINKS_PER_SEG) == (LINKS_PER_SEG - 1)


def arc_tip_pose(theta_per_seg):
    """
    Analytic planar PCC tip pose for the ideal continuous arm.

    theta_per_seg: iterable of total bending angle per segment (rad).
    Returns (x, y, psi) in metres/radians. A segment of length L bent through
    angle t is a circular arc of curvature t/L; t -> 0 degenerates to a
    straight segment, handled by the small-angle branch.
    """
    x = y = psi = 0.0
    for L, t in zip(L_SEG, theta_per_seg):
        if abs(t) < 1e-9:
            dx, dy = L, 0.0
        else:
            dx = L * math.sin(t) / t
            dy = L * (1.0 - math.cos(t)) / t
        # rotate the segment-local offset into the accumulated frame
        c, s = math.cos(psi), math.sin(psi)
        x += c * dx - s * dy
        y += s * dx + c * dy
        psi += t
    return x, y, psi


def chain_tip_pose(joint_angles, link_len: float = None):
    """
    Exact tip pose of the discretized rigid-link chain, computed independently
    of MuJoCo. Used to verify the generated model is built as intended.

    Mirrors the midpoint construction in build_model.py: each hinge sits at the
    CENTRE of its link, so the chain starts and ends with a half-link.

    Why the midpoint matters: with hinges at the proximal end of each link, the
    first link is already rotated by the full joint angle while the true arc
    still has tangent 0. That half-angle bias accumulates along the chain and
    makes the model only first-order accurate - measured at 15 mm tip error for
    18 links at 60 deg/segment, which would have swamped the very tracking
    errors this project exists to measure. Centring the hinges turns the sum
    into the trapezoidal rule, which is second-order: error drops ~4x per
    doubling of link count instead of ~2x.
    """
    L = LINK_LEN if link_len is None else link_len
    n = len(joint_angles)
    x, y, psi = L / 2.0, 0.0, 0.0       # first half-link along the base tangent
    for i, a in enumerate(joint_angles):
        psi += a
        if i < n - 1:
            x += L * math.cos(psi)
            y += L * math.sin(psi)
    x += (L / 2.0) * math.cos(psi)      # final half-link
    y += (L / 2.0) * math.sin(psi)
    return x, y, psi
