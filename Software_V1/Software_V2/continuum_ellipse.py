"""
continuum_ellipse_v2.py
═══════════════════════════════════════════════════════════════════════════════
Tendon-Driven Continuum Manipulator · Ellipse Trajectory Control GUI
3-Segment · 6 Tendons · RS-485 (LKMTECH MF5015) · PCC + Kalman Control

Implements Traj. 1 (Fig. 6) from:
  Zhai et al. 2025 — "Model-Based Control of a Continuum Manipulator with
  Online Jacobian Error Compensation Using Kalman Filtering"
  DOI: 10.34133/cbsystems.0339

  xr = 240 + 20·cos(2π/T · t)  mm
  yr =  60·sin(2π/T · t)        mm
  ψr = 0  (horizontal attitude held)

HARDWARE
────────
  Motors: LKMTECH MF5015 brushless with integrated driver
  Bus:    RS-485 via USB adapter (python: serial)

  Run (simulation):   python continuum_ellipse_v2.py
  Run (real HW):      python continuum_ellipse_v2.py --real --port COM3
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading, time, logging, queue, csv, os, math, argparse, io, copy
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Callable
import numpy as np

# ── Optional hardware import ──────────────────────────────────────────────────
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# ══════════════════════════════════════════════════════════════════════════════
# ROBOT PHYSICAL PARAMETERS  (Table 1, Zhai et al. 2025)
# ══════════════════════════════════════════════════════════════════════════════

L_SEG        = [0.09, 0.09, 0.09]          # L₁,L₂,L₃ (metres)
R_TENDON     = [0.005, 0.0035, 0.002]       # r₁,r₂,r₃ (metres)
SEG_LEN_MM   = [L * 1000.0 for L in L_SEG]  # mm equivalents for visualizer
TOTAL_LEN_MM = sum(SEG_LEN_MM)              # 3 × 90 mm = 270 mm
DISK_DIAMETER_MM = 13.0
SPACER_DISKS_PER_SEG = 5
END_DISKS_PER_SEG = 1
CAPSTAN_R_MM = [15.0] * 6                   # drum radii (mm)
MAX_DISP_MM  = 120.0                        # max motor displacement (mm)
MAX_SPEED_MMS= 80.0
MAX_DELTA_L  = 0.060                        # max |Δlᵢ| in metres

MOTOR_TO_SEG  = [0, 0, 1, 1, 2, 2]
MOTOR_IS_POS  = [True, False, True, False, True, False]
PAIR_MATES    = [1, 0, 3, 2, 5, 4]
SEG_COLORS    = ["#00ddb8", "#3a9ef0", "#f5a623"]
SEG_NAMES     = ["Segment 1", "Segment 2", "Segment 3"]
MOTOR_COLORS  = [SEG_COLORS[0], SEG_COLORS[0],
                 SEG_COLORS[1], SEG_COLORS[1],
                 SEG_COLORS[2], SEG_COLORS[2]]
MOTOR_SHORT   = ["M1","M2","M3","M4","M5","M6"]


def disk_layout_mm():
    """
    Paper geometry: each of the 3 segments has 5 spacer disks and 1 end disk.
    Segment length is 90 mm, so the axial disk pitch is 90 / 6 = 15 mm.
    Returns (arc_length_mm, segment_index, kind).
    """
    layout = [(0.0, -1, "base")]
    start = 0.0
    disks_per_segment = SPACER_DISKS_PER_SEG + END_DISKS_PER_SEG
    for seg_i, seg_len in enumerate(SEG_LEN_MM):
        pitch = seg_len / disks_per_segment
        for disk_i in range(1, disks_per_segment + 1):
            kind = "end" if disk_i == disks_per_segment else "spacer"
            layout.append((start + disk_i * pitch, seg_i, kind))
        start += seg_len
    return layout

# ── Theme  (off-white light palette) ─────────────────────────────────────────
BG     = "#f7f8fa"; PANEL  = "#edf0f4"; CARD   = "#ffffff"; INPUT  = "#f9fafb"
BORDER = "#c8d0da"; ACCENT = "#0f766e"; BLUE   = "#2563eb"; RED    = "#dc2626"
YELLOW = "#d97706"; GREEN  = "#16a34a"; ORANGE = "#ea580c"; MUTED  = "#5a6a7e"
TEXT   = "#0d1b2a"; DIM    = "#7a8699"; CBKG   = "#eef1f5"
FNT    = "Segoe UI"; FNT_H = "Segoe UI"

_lq: queue.Queue = queue.Queue()
class _QH(logging.Handler):
    def emit(self, r): _lq.put(self.format(r))

# ══════════════════════════════════════════════════════════════════════════════
# LKMTECH MF5015 MOTOR DRIVER  (RS-485)
# ══════════════════════════════════════════════════════════════════════════════

class LKMMotor:
    """
    Minimal driver for LKMTECH MF5015 via RS-485.

    Command format (simplified from LKMTECH protocol):
      Position+Speed command: 0xA4
      Byte layout: [0xA4, 0x00, speed_lo, speed_hi, angle_lo, angle_mid_lo, angle_mid_hi, angle_hi]
      Angle unit: 0.01 degree/LSB  → multiply degrees × 100
      Speed unit: 1 dps/LSB

    Feedback: motor replies with 8-byte packet:
      [0xA4, temp, torque_lo, torque_hi, speed_lo, speed_hi, angle_lo, angle_hi]
    """

    CMD_POSITION_SPEED = 0xA4
    CMD_STOP           = 0x81
    CMD_READ_STATE     = 0x9C
    PACKET_LEN         = 8

    def __init__(self, motor_id: int, bus: "RS485Bus"):
        self.motor_id   = motor_id   # 1-based
        self.bus        = bus
        self.position_deg = 0.0
        self.velocity_dps = 0.0
        self.temperature  = 0
        self.torque_raw   = 0
        self._lock        = threading.Lock()

    def _build_packet(self, cmd: int, data: bytes) -> bytes:
        """Build LKMTECH 8-byte packet with motor ID header."""
        # Header: 0x3E, motor_id, data_len, cmd
        # Simplified: direct command packet
        payload = bytes([0x3E, self.motor_id, len(data) + 1, cmd]) + data
        checksum = sum(payload) & 0xFF
        return payload + bytes([checksum])

    def set_position(self, position_deg: float, max_speed_dps: float = 500.0,
                     wait: bool = False):
        """Send position+speed command."""
        angle_raw = int(position_deg * 100) & 0xFFFFFFFF
        speed_raw = max(1, min(int(max_speed_dps), 32767))
        data = bytes([
            speed_raw & 0xFF, (speed_raw >> 8) & 0xFF,
            angle_raw & 0xFF, (angle_raw >> 8) & 0xFF,
            (angle_raw >> 16) & 0xFF, (angle_raw >> 24) & 0xFF,
        ])
        pkt = self._build_packet(self.CMD_POSITION_SPEED, data)
        reply = self.bus.send_receive(pkt, expect_bytes=self.PACKET_LEN + 5)
        if reply:
            self._parse_feedback(reply)

    def stop(self):
        """Send stop command."""
        pkt = self._build_packet(self.CMD_STOP, b'')
        self.bus.send_receive(pkt, expect_bytes=0)

    def read_state(self):
        """Request and parse feedback."""
        pkt = self._build_packet(self.CMD_READ_STATE, b'')
        reply = self.bus.send_receive(pkt, expect_bytes=self.PACKET_LEN + 5)
        if reply:
            self._parse_feedback(reply)

    def _parse_feedback(self, data: bytes):
        """Parse LKMTECH feedback packet."""
        try:
            # Find 0x3E header in reply
            idx = data.find(0x3E)
            if idx < 0 or len(data) < idx + 9:
                return
            self.temperature  = data[idx + 4]
            torq_raw          = int.from_bytes(data[idx+5:idx+7], 'little', signed=True)
            speed_raw         = int.from_bytes(data[idx+7:idx+9], 'little', signed=True)
            with self._lock:
                self.velocity_dps = float(speed_raw)
                self.torque_raw   = torq_raw
        except Exception:
            pass

    def get_feedback(self):
        with self._lock:
            return {
                "position_deg": self.position_deg,
                "velocity_dps": self.velocity_dps,
                "temperature":  self.temperature,
            }


class RS485Bus:
    """Thread-safe RS-485 serial bus for LKMTECH motors."""

    def __init__(self, port: str, baud: int = 115200, timeout: float = 0.05,
                 simulated: bool = False):
        self.port      = port
        self.baud      = baud
        self.timeout   = timeout
        self.simulated = simulated
        self._ser      = None
        self._lock     = threading.Lock()
        self._sim_positions = [0.0] * 6   # simulated motor angles

    def open(self):
        if self.simulated:
            return
        if not HAS_SERIAL:
            raise RuntimeError("pyserial not installed. Run: pip install pyserial")
        self._ser = serial.Serial(
            self.port, self.baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout
        )

    def close(self):
        if self._ser and self._ser.is_open:
            self._ser.close()

    def send_receive(self, packet: bytes, expect_bytes: int = 13) -> Optional[bytes]:
        if self.simulated:
            return None
        with self._lock:
            try:
                self._ser.write(packet)
                if expect_bytes > 0:
                    return self._ser.read(expect_bytes)
            except Exception as e:
                logging.warning(f"RS485 error: {e}")
        return None

    def is_open(self) -> bool:
        if self.simulated:
            return True
        return self._ser is not None and self._ser.is_open


class RobotController:
    """
    High-level controller: manages 6 LKMMotor instances.
    Converts mm displacement → degrees via capstan radius.
    """

    def __init__(self, bus: RS485Bus, capstan_radii_mm: List[float] = None):
        self.bus    = bus
        self.radii  = capstan_radii_mm or list(CAPSTAN_R_MM)
        self.motors = [LKMMotor(i + 1, bus) for i in range(6)]
        self._running = False

    def start(self):
        self.bus.open()
        self._running = True

    def close(self):
        self._running = False
        for m in self.motors:
            try: m.stop()
            except: pass
        self.bus.close()

    def mm_to_deg(self, mm: float, motor_idx: int) -> float:
        r = self.radii[motor_idx]
        return (mm / r) * (180.0 / math.pi)

    def mms_to_dps(self, mms: float, motor_idx: int) -> float:
        r = self.radii[motor_idx]
        return (mms / r) * (180.0 / math.pi)

    def set_motor_mm(self, motor_idx: int, disp_mm: float,
                     speed_mms: float = 20.0, wait: bool = False):
        """Command motor by tendon displacement in mm."""
        if not self._running:
            return
        deg   = self.mm_to_deg(disp_mm, motor_idx)
        dps   = self.mms_to_dps(speed_mms, motor_idx)
        self.motors[motor_idx].set_position(deg, dps, wait=wait)

    def set_all_mm(self, disps_mm: List[float], speeds_mms: List[float]):
        """Command all 6 motors simultaneously (non-blocking)."""
        threads = []
        for i in range(6):
            t = threading.Thread(
                target=self.set_motor_mm,
                args=(i, disps_mm[i], speeds_mms[i]),
                daemon=True
            )
            threads.append(t)
            t.start()
        # Don't join — fire and forget for real-time control

    def stop_all(self):
        for m in self.motors:
            try: m.stop()
            except: pass

    def get_all_feedback(self):
        fb = {}
        for m in self.motors:
            fb[m.motor_id] = m.get_feedback()
        return fb


# ══════════════════════════════════════════════════════════════════════════════
# PCC KINEMATICS (Zhai et al. 2025, Eqs. 1–9)
# ══════════════════════════════════════════════════════════════════════════════

def pcc_angles(u: np.ndarray,
               r: List[float] = R_TENDON) -> np.ndarray:
    """
    Solve for bending angles from tendon differentials (Eqs. 7–9).
    u = [Δl₁, Δl₂, Δl₃] in metres.
    Returns [θ₁, θ₂, θ₃] in radians.
    """
    dl1, dl2, dl3 = u
    theta1 = dl1 / r[0]
    theta2 = (dl2 - r[1] * theta1) / r[0]
    theta3 = (dl3 - r[2] * theta1 - r[1] * theta2) / r[0]
    return np.array([theta1, theta2, theta3])


def forward_kinematics(u: np.ndarray,
                        L: List[float] = L_SEG,
                        r: List[float] = R_TENDON) -> np.ndarray:
    """
    PCC forward kinematics (Eqs. 3–5).
    u = [Δl₁, Δl₂, Δl₃] in metres.
    Returns [x, y, ψ] in metres and radians.
    """
    th = pcc_angles(u, r)
    x, y = 0.0, 0.0
    cum = 0.0   # cumulative bending angle from X-axis

    for i in range(3):
        ti  = th[i]
        Li  = L[i]
        s0  = cum
        s1  = cum + ti
        if abs(ti) < 1e-9:
            # L'Hôpital limit: arc → straight segment along current tangent
            # X is along arm (cos), Y is bending (sin)
            dx = Li * math.cos(s0)
            dy = Li * math.sin(s0)
        else:
            # Arc integral: x=∫cos(s)ds, y=∫sin(s)ds over [s0,s1]
            dx = (Li / ti) * (math.sin(s1) - math.sin(s0))
            dy = (Li / ti) * (math.cos(s0) - math.cos(s1))
        x  += dx
        y  += dy
        cum = s1

    psi = float(cum)
    return np.array([x, y, psi])


def model_jacobian(u: np.ndarray,
                   L: List[float] = L_SEG,
                   r: List[float] = R_TENDON) -> np.ndarray:
    """
    Numeric Jacobian ᵐJ ∈ ℝ³ˣ³ = ∂[x,y,ψ]/∂[Δl₁,Δl₂,Δl₃] (Eq. 10).
    """
    h   = 1e-7
    mJ  = np.zeros((3, 3))
    for j in range(3):
        u_p = u.copy(); u_p[j] += h
        u_m = u.copy(); u_m[j] -= h
        mJ[:, j] = (forward_kinematics(u_p, L, r) -
                    forward_kinematics(u_m, L, r)) / (2 * h)
    return mJ


def compute_pcc_kinematics_mm(disps_mm: List[float]) -> dict:
    """
    Full kinematics for the visualizer — returns body arc points + pose.
    disps_mm: 6-motor displacements in mm (M1..M6).
    """
    # Convert signed antagonistic motor-pair displacement to tendon differential.
    # A valid pair is [+d, -d], so Δl = (M_pull - M_release) / 2.
    dl = np.array([
        0.5 * (disps_mm[0] - disps_mm[1]) / 1000.0,
        0.5 * (disps_mm[2] - disps_mm[3]) / 1000.0,
        0.5 * (disps_mm[4] - disps_mm[5]) / 1000.0,
    ])
    dl = np.clip(dl, -MAX_DELTA_L, MAX_DELTA_L)

    pose = forward_kinematics(dl)
    th   = pcc_angles(dl)

    # Build body arc using the EXACT analytical PCC arc formula.
    # Each segment is a constant-curvature arc: the same sin/cos integrals used
    # in forward_kinematics, sampled at N_PTS_SEG sub-steps per segment.
    # This guarantees all_pts[-1] == (tip_x_mm, tip_y_mm) to floating-point precision,
    # so the orange tip dot in the side view always lands on the actual EE position.
    N_PTS_SEG = 40          # sub-steps per segment  (120 arc points total)
    anchors_mm = [SEG_LEN_MM[0],
                  SEG_LEN_MM[0] + SEG_LEN_MM[1],
                  sum(SEG_LEN_MM)]

    all_pts, curvature, pt_seg = [], [], []
    x2, y2, cum_angle = 0.0, 0.0, 0.0

    for seg_i in range(3):
        Li  = SEG_LEN_MM[seg_i]
        ti  = float(th[seg_i])          # total bending angle of this segment (rad)
        s0  = cum_angle                 # tangent angle at segment start
        # curvature magnitude for colour mapping (rad/mm), clamped
        ki  = ti / Li if Li > 0 else 0.0
        lim = math.radians(150) / Li
        ki  = max(-lim, min(lim, ki))

        for k in range(N_PTS_SEG + 1):
            frac = k / N_PTS_SEG
            s_k  = s0 + frac * ti
            if abs(ti) < 1e-9:
                xk = x2 + frac * Li * math.cos(s0)
                yk = y2 + frac * Li * math.sin(s0)
            else:
                xk = x2 + (Li / ti) * (math.sin(s_k) - math.sin(s0))
                yk = y2 + (Li / ti) * (math.cos(s0)  - math.cos(s_k))
            # Skip duplicate boundary point between consecutive segments
            if seg_i > 0 and k == 0:
                continue
            all_pts.append((xk, yk))
            curvature.append(ki)
            pt_seg.append(seg_i)

        # Advance accumulated position/angle to end of segment
        s1 = s0 + ti
        if abs(ti) < 1e-9:
            x2 += Li * math.cos(s0)
            y2 += Li * math.sin(s0)
        else:
            x2 += (Li / ti) * (math.sin(s1) - math.sin(s0))
            y2 += (Li / ti) * (math.cos(s0) - math.cos(s1))
        cum_angle = s1

    # Segment boundary indices
    b0 = N_PTS_SEG                       # last point of segment 1
    b1 = N_PTS_SEG * 2                   # last point of segment 2
    end_pts = [all_pts[b0], all_pts[b1], all_pts[-1]]

    thetas_rad = [float(t) for t in th]
    psi_deg    = math.degrees(float(pose[2]))

    seg_pts = [
        all_pts[:b0 + 1],
        all_pts[b0: b1 + 1],
        all_pts[b1:],
    ]

    tip_x_mm = pose[0] * 1000.0
    tip_y_mm = pose[1] * 1000.0

    return {
        "all_pts":    all_pts,
        "pt_seg":     pt_seg,
        "curvature":  curvature,
        "end_pts":    end_pts,
        "thetas":     thetas_rad,
        "tip_x":      tip_x_mm,
        "tip_y":      tip_y_mm,
        "psi":        pose[2],
        "seg_points": seg_pts,
    }


# ══════════════════════════════════════════════════════════════════════════════
# KALMAN-FILTER JACOBIAN CONTROLLER  (Zhai et al. 2025)
# ══════════════════════════════════════════════════════════════════════════════

class KalmanJacobianController:
    """
    Hybrid PCC + online Kalman Jacobian error compensation.

    All quantities in SI units (metres, radians).
    """

    def __init__(self,
                 L: List[float] = None,
                 r: List[float] = None,
                 gamma: float   = 0.5,
                 sigma_p: float = 0.35,
                 sigma_psi: float = 2.5,
                 alpha: float   = 1.0,
                 beta: float    = 0.5,
                 use_kalman: bool = True):

        self.L        = L   or list(L_SEG)
        self.r        = r   or list(R_TENDON)
        self.gamma    = gamma
        self.sigma_p  = sigma_p
        self.sigma_psi= sigma_psi
        self.alpha    = alpha
        self.beta     = beta
        self.use_kalman = use_kalman

        # Kalman state
        self.xi   = np.zeros(9)    # ξ = vec(δJ)
        self.P    = np.zeros((9, 9))

        # Previous step storage
        self.u_prev    = np.zeros(3)
        self.pose_prev = np.zeros(3)

        # Constrained Jacobian (initialise to PCC Jacobian at home)
        self.cJ_prev = model_jacobian(np.zeros(3), self.L, self.r)

        # Diagnostics (updated each step)
        self.last_mJ     = np.zeros((3, 3))
        self.last_eJ     = np.zeros((3, 3))
        self.last_cJ     = np.zeros((3, 3))
        self.last_error  = np.zeros(3)
        self.last_K_norm = 0.0

        self._first_step = True

    def reset(self):
        self.xi        = np.zeros(9)
        self.P         = np.zeros((9, 9))
        self.u_prev    = np.zeros(3)
        self.pose_prev = np.zeros(3)
        self.cJ_prev   = model_jacobian(np.zeros(3), self.L, self.r)
        self._first_step = True

    # ── Core functions ────────────────────────────────────────────────────────

    def _fk(self, u): return forward_kinematics(u, self.L, self.r)
    def _mj(self, u): return model_jacobian(u, self.L, self.r)

    def _build_H(self, du: np.ndarray) -> np.ndarray:
        """Build 3×9 measurement matrix Hₖ (Eq. 15)."""
        z = np.zeros(3)
        H = np.block([
            [du.reshape(1, 3), z.reshape(1, 3), z.reshape(1, 3)],
            [z.reshape(1, 3), du.reshape(1, 3), z.reshape(1, 3)],
            [z.reshape(1, 3), z.reshape(1, 3), du.reshape(1, 3)],
        ])
        return H   # shape (3, 9)

    def _kalman_step(self, u_curr: np.ndarray, pose_curr: np.ndarray,
                     mJ: np.ndarray) -> np.ndarray:
        """
        Run one Kalman predict+update cycle.
        Returns estimated Jacobian ᵉJ (3×3).
        """
        gamma  = self.gamma
        du     = u_curr - self.u_prev
        dpose  = pose_curr - self.pose_prev

        # ── Predict (Eqs. 12–13) ──────────────────────────────────────────────
        xi_minus = gamma * self.xi
        Q        = np.eye(9)
        P_minus  = gamma**2 * self.P + Q

        # ── Update (Eqs. 14–18) ───────────────────────────────────────────────
        du_norm_sq = float(np.dot(du, du))
        R          = max(0.5 * du_norm_sq, 1e-12) * np.eye(3)

        H          = self._build_H(du)
        mJ_vec     = mJ.flatten()              # row-stacked 9-vector

        # Innovation: y = dpose - H·(mJ_vec + xi_minus)
        y_hat = H @ (mJ_vec + xi_minus)
        innov = dpose - y_hat

        # Kalman gain: K = P⁻·Hᵀ·(H·P⁻·Hᵀ + R)⁻¹  (Eq. 16)
        S = H @ P_minus @ H.T + R
        try:
            K = P_minus @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = np.zeros((9, 3))

        # Posterior (Eqs. 17–18)
        xi_plus = xi_minus + K @ innov
        P_plus  = (np.eye(9) - K @ H) @ P_minus

        self.xi             = xi_plus
        self.P              = P_plus
        self.last_K_norm    = float(np.linalg.norm(K))

        # Reconstruct estimated Jacobian (Eq. 19)
        delta_J = xi_plus.reshape(3, 3)
        eJ      = mJ + delta_J
        return eJ

    def _apply_constraints(self, eJ: np.ndarray) -> np.ndarray:
        """
        Constrain Jacobian update magnitude (Eq. 20).
        """
        cJ   = self.cJ_prev.copy()
        eJp  = eJ[:2, :]
        eJps = eJ[2:3, :]
        cJp  = cJ[:2, :]
        cJps = cJ[2:3, :]

        # Position Jacobian
        diff_p = np.linalg.norm(eJp - cJp, 'fro')
        if diff_p <= self.sigma_p or diff_p < 1e-12:
            new_Jp = eJp
        else:
            new_Jp = cJp + self.sigma_p * (eJp - cJp) / diff_p

        # Attitude Jacobian
        diff_ps = np.linalg.norm(eJps - cJps, 'fro')
        if diff_ps <= self.sigma_psi or diff_ps < 1e-12:
            new_Jps = eJps
        else:
            new_Jps = cJps + self.sigma_psi * (eJps - cJps) / diff_ps

        cJ_new = np.vstack([new_Jp, new_Jps])
        return cJ_new

    def _damped_pinv(self, cJ: np.ndarray) -> np.ndarray:
        """Damped pseudoinverse (Eq. 22): (cJᵀ·cJ + α·I)⁻¹·cJᵀ"""
        return np.linalg.solve(
            cJ.T @ cJ + self.alpha * np.eye(3),
            cJ.T
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_control(self, u_curr: np.ndarray,
                        pose_ref: np.ndarray,
                        pose_curr: np.ndarray) -> np.ndarray:
        """
        Full control pipeline → returns u_next (3-vector, metres).

        pose_ref  : [xr, yr, ψr]
        pose_curr : [x,  y,  ψ]
        """
        mJ = self._mj(u_curr)
        self.last_mJ = mJ.copy()

        if self.use_kalman and not self._first_step:
            eJ = self._kalman_step(u_curr, pose_curr, mJ)
        else:
            eJ = mJ.copy()

        self.last_eJ = eJ.copy()

        cJ = self._apply_constraints(eJ)
        self.last_cJ     = cJ.copy()
        self.cJ_prev     = cJ.copy()

        # Control law (Eq. 21)
        error     = pose_ref - pose_curr
        self.last_error = error.copy()
        cJ_dagger = self._damped_pinv(cJ)
        u_next    = u_curr + self.beta * (cJ_dagger @ error)

        # Save for next Kalman step
        self.u_prev    = u_curr.copy()
        self.pose_prev = pose_curr.copy()
        self._first_step = False

        return u_next


# ══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def make_trajectory(name: str, T: float = 40.0) -> Callable[[float], np.ndarray]:
    """Returns a function t → [xr, yr, ψr] (metres, radians).
    Only Traj 1 — Ellipse is supported (Zhai et al. 2025, Fig. 6).
    """
    # Traj 1: xr = 240+20·cos(2π/T·t) mm,  yr = 60·sin(2π/T·t) mm,  ψr = 0
    def fn(t):
        xr  = 0.240 + 0.020 * math.cos(2 * math.pi * t / T)
        yr  = 0.060 * math.sin(2 * math.pi * t / T)
        psi = 0.0
        return np.array([xr, yr, psi])
    return fn


TRAJECTORY_NAMES = [
    "Traj 1 — Ellipse (pos only)",
]


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TendonState:
    name:    str
    delay_s: float
    motors:  list = field(default_factory=lambda: [(0.0, 20.0)] * 6)

    def summary(self):
        parts = [f"M{i+1}:{self.motors[i][0]:+.1f}@{self.motors[i][1]:.0f}"
                 for i in range(6)]
        return f"[{self.name}]  " + "  ".join(parts)


def parse_states_csv(path: str):
    states, errors = [], []
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if "state_name" not in (reader.fieldnames or []):
                return [], "Missing required column: 'state_name'"
            for row_num, row in enumerate(reader, start=2):
                name  = row.get("state_name", f"State {row_num}").strip()
                try:   delay = max(0.0, float(row.get("delay_s", "1.0") or "1.0"))
                except: delay = 1.0
                motors = []
                for m in range(1, 7):
                    try:    disp  = float(row.get(f"m{m}_disp_mm",   "0") or "0")
                    except: disp  = 0.0
                    try:    speed = max(0.1, min(float(row.get(f"m{m}_speed_mms", "20") or "20"), MAX_SPEED_MMS))
                    except: speed = 20.0
                    disp = max(-MAX_DISP_MM, min(MAX_DISP_MM, disp))
                    motors.append((disp, speed))
                motors = normalize_antagonistic_motor_state(motors)
                states.append(TendonState(name=name, delay_s=delay, motors=motors))
    except FileNotFoundError:
        return [], f"File not found: {path}"
    except Exception as e:
        return [], f"CSV error: {e}"
    return states, "\n".join(errors)


def make_sample_states():
    raw = [
        ("Home",          1.5, (0,20),(0,20),(0,20),(0,20),(0,20),(0,20)),
        ("S1 Bend +30mm", 2.0, (30,25),(0,25),(0,20),(0,20),(0,20),(0,20)),
        ("S1 Bend -30mm", 2.0, (0,25),(30,25),(0,20),(0,20),(0,20),(0,20)),
        ("S2 Bend +30mm", 2.0, (0,20),(0,20),(30,25),(0,25),(0,20),(0,20)),
        ("S2 Bend -30mm", 2.0, (0,20),(0,20),(0,25),(30,25),(0,20),(0,20)),
        ("S3 Bend +30mm", 2.0, (0,20),(0,20),(0,20),(0,20),(30,25),(0,25)),
        ("S3 Bend -30mm", 2.0, (0,20),(0,20),(0,20),(0,20),(0,25),(30,25)),
        ("Forward Arc",   2.5, (20,20),(0,20),(20,20),(0,20),(20,20),(0,20)),
        ("Reverse Arc",   2.5, (0,20),(20,20),(0,20),(20,20),(0,20),(20,20)),
        ("Home",          1.0, (0,30),(0,30),(0,30),(0,30),(0,30),(0,30)),
    ]
    out = []
    for r in raw:
        name, delay = r[0], r[1]
        motors = [(d, s) for d, s in r[2:]]
        motors = normalize_antagonistic_motor_state(motors)
        out.append(TendonState(name=name, delay_s=delay, motors=motors))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: motor disp <-> u conversion
# ══════════════════════════════════════════════════════════════════════════════

def u_to_motor_disps_mm(u: np.ndarray) -> List[float]:
    """
    Convert 3-vector u [Δl₁,Δl₂,Δl₃] (metres) to 6 motor displacements (mm).
    Positive Δlᵢ → M_pos[i] pulls, M_neg[i] releases the same amount.
    Negative Δlᵢ → M_neg[i] pulls, M_pos[i] releases the same amount.
    """
    disps = [0.0] * 6
    for i in range(3):
        dl_mm = max(-MAX_DISP_MM, min(MAX_DISP_MM, float(u[i]) * 1000.0))
        pos_idx = i * 2
        neg_idx = i * 2 + 1
        disps[pos_idx] = dl_mm
        disps[neg_idx] = -dl_mm
    return disps


def motor_disps_mm_to_u(disps_mm: List[float]) -> np.ndarray:
    """Inverse: 6 motor displacements (mm) → u [Δl₁,Δl₂,Δl₃] (metres)."""
    u = np.zeros(3)
    for i in range(3):
        pos_idx = i * 2
        neg_idx = i * 2 + 1
        u[i] = 0.5 * (disps_mm[pos_idx] - disps_mm[neg_idx]) / 1000.0
    return u


def normalize_antagonistic_motor_state(motors: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Enforce the paper's antagonistic pair rule for M1/M2, M3/M4, M5/M6.
    Legacy CSV rows that command only one positive motor are converted to
    signed equal-and-opposite pairs.
    """
    out = [(0.0, 20.0)] * 6
    for seg_i in range(3):
        pos_idx = seg_i * 2
        neg_idx = pos_idx + 1
        d_pos, s_pos = motors[pos_idx]
        d_neg, s_neg = motors[neg_idx]
        if d_pos >= 0.0 and d_neg >= 0.0:
            pair_delta = d_pos if d_pos >= d_neg else -d_neg
        else:
            pair_delta = 0.5 * (d_pos - d_neg)
        pair_delta = max(-MAX_DISP_MM, min(MAX_DISP_MM, pair_delta))
        pair_speed = max(0.1, min(MAX_SPEED_MMS, max(float(s_pos), float(s_neg))))
        out[pos_idx] = (pair_delta, pair_speed)
        out[neg_idx] = (-pair_delta, pair_speed)
    return out


def mm_to_deg(mm: float, motor_idx: int = 0) -> float:
    return (mm / CAPSTAN_R_MM[motor_idx]) * (180.0 / math.pi)

def deg_to_mm(deg: float, motor_idx: int = 0) -> float:
    return deg * CAPSTAN_R_MM[motor_idx] * (math.pi / 180.0)

def mms_to_dps(mms: float, motor_idx: int = 0) -> float:
    return (mms / CAPSTAN_R_MM[motor_idx]) * (180.0 / math.pi)


# ══════════════════════════════════════════════════════════════════════════════
# CAPSTAN PANEL
# ══════════════════════════════════════════════════════════════════════════════

class CapstanPanel(tk.Frame):
    DRUM_R = 26

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._disps  = [0.0] * 6
        self._speeds = [0.0] * 6
        self._canvases, self._disp_lbls, self._speed_lbls = [], [], []
        self._build()

    def _build(self):
        hf = tk.Frame(self, bg=BG)
        hf.pack(fill="x", pady=(8, 4))
        tk.Label(hf, text="CAPSTAN DRUMS", font=(FNT_H, 12, "bold"),
                 fg=ACCENT, bg=BG).pack(side="left")
        tk.Label(hf, text="  ·  live tendon displacement per motor",
                 font=(FNT, 10), fg=MUTED, bg=BG).pack(side="left")

        grp = tk.Frame(self, bg=BG)
        grp.pack(fill="x")
        for sc, sn in zip(SEG_COLORS, SEG_NAMES):
            g = tk.Frame(grp, bg=BG)
            g.pack(side="left", expand=True, fill="x")
            tk.Frame(g, bg=sc, height=2).pack(fill="x")
            tk.Label(g, text=f"── {sn} ──", font=(FNT, 10, "bold"),
                     fg=sc, bg=BG).pack(pady=2)

        grid = tk.Frame(self, bg=BG)
        grid.pack(fill="x")

        for i in range(6):
            col = MOTOR_COLORS[i]
            sc  = SEG_COLORS[MOTOR_TO_SEG[i]]
            cell = tk.Frame(grid, bg=CARD,
                            highlightbackground=BORDER, highlightthickness=1)
            cell.grid(row=0, column=i, padx=3, pady=2, sticky="nsew")
            grid.columnconfigure(i, weight=1)

            tk.Frame(cell, bg=sc, height=3).pack(fill="x")
            tk.Label(cell, text=f"M{i+1}", font=(FNT, 14, "bold"),
                     fg=col, bg=CARD).pack(pady=(5, 0))
            cv = tk.Canvas(cell, width=100, height=66, bg=CARD, highlightthickness=0)
            cv.pack(padx=4, pady=3)
            self._canvases.append(cv)
            dl = tk.Label(cell, text="  0.0", font=(FNT, 13, "bold"),
                          fg=col, bg=CARD, anchor="center")
            dl.pack()
            self._disp_lbls.append(dl)
            vl = tk.Label(cell, text="0 mm/s", font=(FNT, 9), fg=MUTED, bg=CARD)
            vl.pack(pady=(0, 5))
            self._speed_lbls.append(vl)

        self.after(60, self._draw_all)

    def _draw_all(self):
        for i in range(6): self._draw(i)

    def _draw(self, i: int):
        cv  = self._canvases[i]
        col = MOTOR_COLORS[i]
        w, h = 100, 66
        cv.delete("all")
        d     = self._disps[i]
        ratio = (d + MAX_DISP_MM) / (2 * MAX_DISP_MM)
        ratio = max(0.0, min(1.0, ratio))
        bar_w = int(w * ratio)
        cv.create_rectangle(0, h - 14, w, h, fill=INPUT, outline="")
        if bar_w > 0:
            cv.create_rectangle(0, h - 14, bar_w, h,
                                fill=col if d >= 0 else "#d1d5db", outline="")
        cv.create_line(w//2, h-14, w//2, h, fill=BORDER, dash=(2,2))
        cv.create_text(w//2, h-6, text=f"{d:+.1f}mm",
                       fill=TEXT, font=(FNT, 9), anchor="center")
        dr = self.DRUM_R
        cx_, cy_ = w//2, 28
        cv.create_oval(cx_-dr, cy_-dr, cx_+dr, cy_+dr,
                       fill="#f3f4f6", outline=col, width=2)
        ang = math.radians(d / CAPSTAN_R_MM[i] * (180/math.pi) % 360)
        ix  = cx_ + int(dr*0.7*math.cos(ang - math.pi/2))
        iy  = cy_ + int(dr*0.7*math.sin(ang - math.pi/2))
        cv.create_line(cx_, cy_, ix, iy, fill=col, width=2, capstyle=tk.ROUND)
        cv.create_oval(cx_-3, cy_-3, cx_+3, cy_+3, fill=col, outline="")
        if abs(self._speeds[i]) > 0.1:
            frac = min(1.0, abs(self._speeds[i]) / MAX_SPEED_MMS)
            cv.create_arc(cx_-dr+3, cy_-dr+3, cx_+dr-3, cy_+dr-3,
                          start=90, extent=-frac*270,
                          outline=YELLOW, style="arc", width=2)

    def update_all(self, disps, speeds, targets=None):
        for i in range(6):
            self._disps[i]  = disps[i]
            self._speeds[i] = speeds[i]
            self._disp_lbls[i].config(text=f"{disps[i]:+.1f}mm")
            self._speed_lbls[i].config(text=f"{speeds[i]:.0f}mm/s")
        self._draw_all()


# ══════════════════════════════════════════════════════════════════════════════
# CONTINUUM SHAPE VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════

class ContinuumVisualizer(tk.Frame):
    BODY_RADII  = [11, 8, 6]
    TENDON_TERM = [0, 0, 1, 1, 2, 2]
    TENDON_ANGLES_DEG = [150, 330, 90, 270, 210, 30]

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._disps    = [0.0] * 6
        self._side_zoom = 1.0
        self._top_zoom = 1.0
        self._iso_zoom = 1.0
        self._iso_yaw = -38.0
        self._iso_pitch = 32.0
        self._iso_drag = None
        self._cv_side  = None
        self._cv_top   = None
        self._cv_iso   = None
        self._cv_heat  = None
        self._pose_widgets = {}
        self._iso_zoom_lbl = None
        self._build()
        self.after(120, self._redraw)

    def _build(self):
        hf = tk.Frame(self, bg=BG)
        hf.pack(fill="x", pady=(4, 2))
        tk.Label(hf, text="CONTINUUM SHAPE", font=(FNT_H, 12, "bold"),
                 fg=BLUE, bg=BG).pack(side="left")
        tk.Label(hf, text="  ·  PCC kinematics  ·  3 segments  ·  mm",
                 font=(FNT, 10), fg=MUTED, bg=BG).pack(side="left")

        views = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=BG, sashwidth=7,
                               sashrelief=tk.RAISED, bd=0, opaqueresize=True)
        views.pack(fill="both", expand=True)

        # Side view
        sc = tk.Frame(views, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        views.add(sc, minsize=260, stretch="always")
        side_hdr = tk.Frame(sc, bg=PANEL)
        side_hdr.pack(fill="x")
        tk.Label(side_hdr, text=" SIDE VIEW  (X–Y plane:  X=arm length, Y=bending)",
                 font=(FNT, 9, "bold"), fg=BLUE, bg=PANEL).pack(side="left")
        self._side_zoom_lbl = tk.Label(side_hdr, text="100%",
                 font=(FNT, 8), fg=MUTED, bg=PANEL, width=5)
        self._side_zoom_lbl.pack(side="right", padx=2)
        tk.Button(side_hdr, text="⟳", font=(FNT, 8), bg=PANEL, fg=MUTED,
                  relief="flat", cursor="hand2", padx=2,
                  command=self._side_zoom_reset).pack(side="right")
        tk.Button(side_hdr, text="＋", font=(FNT, 8), bg=PANEL, fg=MUTED,
                  relief="flat", cursor="hand2", padx=2,
                  command=self._side_zoom_in).pack(side="right")
        tk.Button(side_hdr, text="－", font=(FNT, 8), bg=PANEL, fg=MUTED,
                  relief="flat", cursor="hand2", padx=2,
                  command=self._side_zoom_out).pack(side="right")
        self._cv_side = tk.Canvas(sc, bg=CBKG, highlightthickness=0)
        self._cv_side.pack(fill="both", expand=True)
        self._cv_side.bind("<Configure>", lambda e: self._draw_side())
        self._cv_side.bind("<MouseWheel>", self._on_side_scroll)

        # Top view
        tc = tk.Frame(views, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        views.add(tc, minsize=240, stretch="always")
        top_hdr = tk.Frame(tc, bg=PANEL)
        top_hdr.pack(fill="x")
        tk.Label(top_hdr, text=" TOP VIEW  (X–Z plane:  Z=0 always — planar robot)",
                 font=(FNT, 9, "bold"), fg=ORANGE, bg=PANEL).pack(side="left")
        self._top_zoom_lbl = tk.Label(top_hdr, text="100%",
                 font=(FNT, 8), fg=MUTED, bg=PANEL, width=5)
        self._top_zoom_lbl.pack(side="right", padx=2)
        tk.Button(top_hdr, text="⟳", font=(FNT, 8), bg=PANEL, fg=MUTED,
                  relief="flat", cursor="hand2", padx=2,
                  command=self._top_zoom_reset).pack(side="right")
        tk.Button(top_hdr, text="＋", font=(FNT, 8), bg=PANEL, fg=MUTED,
                  relief="flat", cursor="hand2", padx=2,
                  command=self._top_zoom_in).pack(side="right")
        tk.Button(top_hdr, text="－", font=(FNT, 8), bg=PANEL, fg=MUTED,
                  relief="flat", cursor="hand2", padx=2,
                  command=self._top_zoom_out).pack(side="right")
        self._cv_top = tk.Canvas(tc, bg=CBKG, highlightthickness=0)
        self._cv_top.pack(fill="both", expand=True)
        self._cv_top.bind("<Configure>", lambda e: self._draw_top())
        self._cv_top.bind("<MouseWheel>", self._on_top_scroll)

        # Isometric view
        ic = tk.Frame(views, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        views.add(ic, minsize=260, stretch="always")
        iso_hdr = tk.Frame(ic, bg=PANEL)
        iso_hdr.pack(fill="x")
        tk.Label(iso_hdr, text=" ISOMETRIC VIEW",
                 font=(FNT, 9, "bold"), fg=ACCENT, bg=PANEL).pack(side="left")
        for label, yaw, pitch in [
            ("ISO", -38.0, 32.0),
            ("FRONT", 0.0, 0.0),
            ("TOP", -90.0, 89.0),
            ("RIGHT", -90.0, 0.0),
        ]:
            tk.Button(iso_hdr, text=label, font=(FNT, 8, "bold"),
                      bg=INPUT, fg=TEXT, relief="flat", cursor="hand2",
                      padx=4, pady=1,
                      command=lambda y=yaw, p=pitch: self._set_iso_view(y, p)
                      ).pack(side="right", padx=(1,0), pady=1)
        self._iso_zoom_lbl = tk.Label(iso_hdr, text="100%",
                                      font=(FNT, 8), fg=MUTED, bg=PANEL, width=5)
        self._iso_zoom_lbl.pack(side="right", padx=2)
        self._cv_iso = tk.Canvas(ic, bg=CBKG, highlightthickness=0)
        self._cv_iso.pack(fill="both", expand=True)
        self._cv_iso.bind("<Configure>", lambda e: self._draw_iso())
        self._cv_iso.bind("<MouseWheel>", self._on_iso_scroll)
        self._cv_iso.bind("<ButtonPress-1>", self._on_iso_press)
        self._cv_iso.bind("<B1-Motion>", self._on_iso_drag)

        # Pose HUD
        pc = tk.Frame(views, bg=CARD, highlightbackground=BORDER, highlightthickness=1,
                      width=158)
        pc.pack_propagate(False)
        views.add(pc, minsize=135, stretch="never")
        tk.Label(pc, text=" END-EFFECTOR", font=(FNT, 9, "bold"),
                 fg=ORANGE, bg=PANEL).pack(fill="x")

        for key, label, unit, col in [
            ("tip_x","X","mm",ACCENT), ("tip_y","Y","mm",ACCENT),
            ("psi","ψ","deg",YELLOW), ("th1","θ₁","deg",SEG_COLORS[0]),
            ("th2","θ₂","deg",SEG_COLORS[1]), ("th3","θ₃","deg",SEG_COLORS[2]),
            ("len","L","mm",MUTED),
        ]:
            pf = tk.Frame(pc, bg=CARD)
            pf.pack(fill="x", padx=6, pady=3)
            tk.Label(pf, text=label, font=(FNT,10), fg=MUTED, bg=CARD,
                     anchor="w", width=3).pack(side="left")
            lv = tk.Label(pf, text="---", font=(FNT,13,"bold"),
                          fg=col, bg=CARD, anchor="e", width=7)
            lv.pack(side="right")
            tk.Label(pf, text=unit, font=(FNT,9), fg=DIM, bg=CARD).pack(side="right")
            self._pose_widgets[key] = lv
            tk.Frame(pc, bg=BORDER, height=1).pack(fill="x", padx=4)

        # Heatmap
        hc = tk.Frame(self, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        hc.pack(fill="x", pady=(3,0))
        tk.Label(hc, text=" MOTOR LOAD HEATMAP  (blue=release · red=pull)",
                 font=(FNT, 9, "bold"), fg=MUTED, bg=PANEL).pack(fill="x")
        self._cv_heat = tk.Canvas(hc, height=52, bg=CBKG, highlightthickness=0)
        self._cv_heat.pack(fill="x")
        self._cv_heat.bind("<Configure>", lambda e: self._draw_heat())

    def set_displacements(self, disps):
        self._disps = list(disps)
        self._redraw()

    def _redraw(self):
        self._draw_side()
        self._draw_top()
        self._draw_iso()
        self._draw_heat()
        self._update_pose()

    def _on_side_scroll(self, event):
        self._side_zoom *= 1.12 if event.delta > 0 else (1/1.12)
        self._side_zoom  = max(0.2, min(6.0, self._side_zoom))
        self._side_zoom_lbl.config(text=f"{int(self._side_zoom*100)}%")
        self._draw_side()
        return "break"

    def _on_top_scroll(self, event):
        self._top_zoom *= 1.12 if event.delta > 0 else (1/1.12)
        self._top_zoom  = max(0.4, min(4.0, self._top_zoom))
        self._top_zoom_lbl.config(text=f"{int(self._top_zoom*100)}%")
        self._draw_top()
        return "break"

    def _side_zoom_in(self):
        self._side_zoom = min(6.0, self._side_zoom * 1.25)
        self._side_zoom_lbl.config(text=f"{int(self._side_zoom*100)}%")
        self._draw_side()

    def _side_zoom_out(self):
        self._side_zoom = max(0.2, self._side_zoom / 1.25)
        self._side_zoom_lbl.config(text=f"{int(self._side_zoom*100)}%")
        self._draw_side()

    def _side_zoom_reset(self):
        self._side_zoom = 1.0
        self._side_zoom_lbl.config(text="100%")
        self._draw_side()

    def _top_zoom_in(self):
        self._top_zoom = min(4.0, self._top_zoom * 1.25)
        self._top_zoom_lbl.config(text=f"{int(self._top_zoom*100)}%")
        self._draw_top()

    def _top_zoom_out(self):
        self._top_zoom = max(0.2, self._top_zoom / 1.25)
        self._top_zoom_lbl.config(text=f"{int(self._top_zoom*100)}%")
        self._draw_top()

    def _top_zoom_reset(self):
        self._top_zoom = 1.0
        self._top_zoom_lbl.config(text="100%")
        self._draw_top()

    def _set_iso_view(self, yaw, pitch):
        self._iso_yaw = float(yaw)
        self._iso_pitch = max(-89.0, min(89.0, float(pitch)))
        self._draw_iso()

    def _on_iso_scroll(self, event):
        self._iso_zoom *= 1.12 if event.delta > 0 else (1 / 1.12)
        self._iso_zoom = max(0.35, min(5.0, self._iso_zoom))
        if self._iso_zoom_lbl is not None:
            self._iso_zoom_lbl.config(text=f"{int(self._iso_zoom * 100)}%")
        self._draw_iso()
        return "break"

    def _on_iso_press(self, event):
        self._iso_drag = (event.x, event.y, self._iso_yaw, self._iso_pitch)

    def _on_iso_drag(self, event):
        if not self._iso_drag:
            return
        x0, y0, yaw0, pitch0 = self._iso_drag
        self._iso_yaw = yaw0 + (event.x - x0) * 0.35
        self._iso_pitch = max(-89.0, min(89.0, pitch0 - (event.y - y0) * 0.25))
        self._draw_iso()

    @staticmethod
    def _grid(cv, w, h, step=38):
        for x in range(0, w, step):
            cv.create_line(x, 0, x, h, fill="#dde3eb", width=1)
        for y in range(0, h, step):
            cv.create_line(0, y, w, y, fill="#dde3eb", width=1)

    @staticmethod
    def _sample_index_for_s_mm(s_mm: float, n_points: int) -> int:
        samples_per_segment = (n_points - 1) // len(SEG_LEN_MM)
        if s_mm <= 0:
            return 0
        seg_start = 0.0
        for seg_i, seg_len in enumerate(SEG_LEN_MM):
            seg_end = seg_start + seg_len
            if s_mm <= seg_end + 1e-9:
                frac = 0.0 if seg_len <= 0 else (s_mm - seg_start) / seg_len
                idx = seg_i * samples_per_segment + int(round(frac * samples_per_segment))
                return max(0, min(n_points - 1, idx))
            seg_start = seg_end
        return n_points - 1

    def _draw_side_disks(self, cv, all_pts, px):
        n_points = len(all_pts)
        for s_mm, seg_i, kind in disk_layout_mm():
            if kind == "base":
                continue
            idx = self._sample_index_for_s_mm(s_mm, n_points)
            x_mm, y_mm = all_pts[idx]
            p0 = all_pts[max(0, idx - 1)]
            p1 = all_pts[min(n_points - 1, idx + 1)]
            tx, ty = p1[0] - p0[0], p1[1] - p0[1]
            mag = math.hypot(tx, ty) or 1.0
            nx, ny = -ty / mag, tx / mag
            radius = DISK_DIAMETER_MM * 0.5
            x0, y0 = px(x_mm - nx * radius, y_mm - ny * radius)
            x1, y1 = px(x_mm + nx * radius, y_mm + ny * radius)
            cx, cy = px(x_mm, y_mm)
            outline = SEG_COLORS[seg_i]
            if kind == "end":
                cv.create_line(x0, y0, x1, y1, fill=outline, width=4, capstyle=tk.ROUND)
                cv.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                               fill=CARD, outline=outline, width=2)
            else:
                cv.create_line(x0, y0, x1, y1, fill="#f8fafc", width=3, capstyle=tk.ROUND)
                cv.create_line(x0, y0, x1, y1, fill=MUTED, width=1, capstyle=tk.ROUND)

    def _draw_top_disks(self, cv, px):
        radius = DISK_DIAMETER_MM * 0.5
        for s_mm, seg_i, kind in disk_layout_mm():
            if kind == "base":
                continue
            x0, z0 = px(s_mm, -radius)
            x1, z1 = px(s_mm, radius)
            cx, cz = px(s_mm, 0.0)
            outline = SEG_COLORS[seg_i]
            if kind == "end":
                cv.create_line(x0, z0, x1, z1, fill=outline, width=4, capstyle=tk.ROUND)
                cv.create_oval(cx - 4, cz - 4, cx + 4, cz + 4,
                               fill=CARD, outline=outline, width=2)
            else:
                cv.create_line(x0, z0, x1, z1, fill="#f8fafc", width=3, capstyle=tk.ROUND)
                cv.create_line(x0, z0, x1, z1, fill=MUTED, width=1, capstyle=tk.ROUND)

    def _draw_iso(self):
        cv = self._cv_iso
        if cv is None:
            return
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 40 or h < 40:
            return
        cv.delete("all")

        kin = compute_pcc_kinematics_mm(self._disps)
        all_pts = kin["all_pts"]
        pt_seg = kin["pt_seg"]
        n_pts = len(all_pts)
        if n_pts < 2:
            return

        yaw = math.radians(self._iso_yaw)
        pitch = math.radians(self._iso_pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)

        def rotate(p3):
            x, y, z = p3
            z = -z
            x1 = cy * x - sy * y
            y1 = sy * x + cy * y
            z1 = z
            y2 = cp * y1 - sp * z1
            z2 = sp * y1 + cp * z1
            return x1, y2, z2

        scale = min(w, h) * 0.72 / TOTAL_LEN_MM * self._iso_zoom
        base_screen_x = w * 0.42
        base_screen_y = h * 0.78
        margin = 30

        base_corners = [(-20.0, 0.0, -16.0), (20.0, 0.0, -16.0),
                        (20.0, 0.0, 16.0), (-20.0, 0.0, 16.0)]
        fit_points = base_corners + [(x, y, 0.0) for x, y in all_pts]
        rotated_fit = [rotate(p3) for p3 in fit_points]
        rx_vals = [p[0] for p in rotated_fit]
        ry_vals = [p[1] for p in rotated_fit]
        span_x = max(rx_vals) - min(rx_vals)
        span_y = max(ry_vals) - min(ry_vals)
        if span_x > 1e-9 and span_x * scale > (w - 2 * margin):
            scale *= (w - 2 * margin) / (span_x * scale)
        if span_y > 1e-9 and span_y * scale > (h - 2 * margin):
            scale *= (h - 2 * margin) / (span_y * scale)

        screen_x = [base_screen_x + rx * scale for rx in rx_vals]
        screen_y = [base_screen_y - ry * scale for ry in ry_vals]
        if min(screen_x) < margin:
            base_screen_x += margin - min(screen_x)
        if max(screen_x) > w - margin:
            base_screen_x -= max(screen_x) - (w - margin)
        if min(screen_y) < margin:
            base_screen_y += margin - min(screen_y)
        if max(screen_y) > h - margin:
            base_screen_y -= max(screen_y) - (h - margin)

        def project(p3):
            rx, ry, rz = rotate(p3)
            return base_screen_x + rx * scale, base_screen_y - ry * scale, rz

        def basis_at(k):
            k0 = max(0, k - 1)
            k1 = min(n_pts - 1, k + 1)
            dx = all_pts[k1][0] - all_pts[k0][0]
            dy = all_pts[k1][1] - all_pts[k0][1]
            ln = math.hypot(dx, dy) or 1.0
            normal = (-dy / ln, dx / ln, 0.0)
            binormal = (0.0, 0.0, 1.0)
            return normal, binormal

        for gx in range(0, w, 42):
            cv.create_line(gx, 0, gx, h, fill="#e5eaf0", width=1)
        for gy in range(0, h, 42):
            cv.create_line(0, gy, w, gy, fill="#e5eaf0", width=1)

        plate_pts = []
        for p3 in base_corners:
            sx, sy_, _ = project(p3)
            plate_pts += [sx, sy_]
        cv.create_polygon(*plate_pts, fill="#dde3ea", outline=MUTED, width=1)
        bx, by, _ = project((0.0, 0.0, 0.0))
        cv.create_text(bx, by + 14, text="BASE", fill=MUTED,
                       font=(FNT, 8, "bold"), anchor="n")

        body_w = max(5, int(DISK_DIAMETER_MM * scale * 0.86))
        for k in range(1, n_pts):
            x0, y0, _ = project((all_pts[k - 1][0], all_pts[k - 1][1], 0.0))
            x1, y1, _ = project((all_pts[k][0], all_pts[k][1], 0.0))
            cv.create_line(x0, y0 + 2, x1, y1 + 2,
                           fill="#b9c3cf", width=body_w + 5,
                           capstyle=tk.ROUND)
        for k in range(1, n_pts):
            seg = pt_seg[k]
            taper = 1.0 - 0.28 * (k / n_pts)
            x0, y0, _ = project((all_pts[k - 1][0], all_pts[k - 1][1], 0.0))
            x1, y1, _ = project((all_pts[k][0], all_pts[k][1], 0.0))
            cv.create_line(x0, y0, x1, y1,
                           fill=SEG_COLORS[seg],
                           width=max(4, int(body_w * taper)),
                           capstyle=tk.ROUND)

        disk_items = []
        for s_mm, disk_seg, disk_kind in disk_layout_mm():
            k = self._sample_index_for_s_mm(s_mm, n_pts)
            normal, binormal = basis_at(k)
            cx0, cy0 = all_pts[k]
            ring = []
            depths = []
            for q in range(24):
                a = 2.0 * math.pi * q / 24
                rr = DISK_DIAMETER_MM * 0.5
                p3 = (
                    cx0 + normal[0] * rr * math.cos(a) + binormal[0] * rr * math.sin(a),
                    cy0 + normal[1] * rr * math.cos(a) + binormal[1] * rr * math.sin(a),
                    normal[2] * rr * math.cos(a) + binormal[2] * rr * math.sin(a),
                )
                sx, sy_, dep = project(p3)
                ring += [sx, sy_]
                depths.append(dep)
            disk_items.append((sum(depths) / len(depths), disk_kind, disk_seg, ring, k))

        for _, disk_kind, disk_seg, ring, k in sorted(disk_items, key=lambda item: item[0]):
            is_end = disk_kind == "end"
            is_base = disk_kind == "base"
            seg_col = SEG_COLORS[disk_seg] if disk_seg >= 0 else MUTED
            fill = "#fff7ed" if is_end else "#f8fafc"
            if is_base:
                fill = "#e5e7eb"
            edge = seg_col if (is_end or is_base) else "#94a3b8"
            cv.create_polygon(*ring, fill=fill, outline=edge, width=2 if is_end else 1)
            cxp, cyp, _ = project((all_pts[k][0], all_pts[k][1], 0.0))
            cv.create_oval(cxp - 1.8, cyp - 1.8, cxp + 1.8, cyp + 1.8,
                           fill=edge, outline="")

        tip_x, tip_y, _ = project((kin["tip_x"], kin["tip_y"], 0.0))
        cv.create_oval(tip_x - 8, tip_y - 8, tip_x + 8, tip_y + 8,
                       fill=ORANGE, outline=TEXT, width=2)
        cv.create_text(tip_x + 12, tip_y, text="EE", fill=ORANGE,
                       font=(FNT, 9, "bold"), anchor="w")

        ox, oy = 30, h - 30
        axes = [((35.0, 0.0, 0.0), "X", BLUE),
                ((0.0, 35.0, 0.0), "Y", GREEN),
                ((0.0, 0.0, 35.0), "Z", RED)]
        r0 = rotate((0.0, 0.0, 0.0))
        for vec, label, col in axes:
            r1 = rotate(vec)
            dx = r1[0] - r0[0]
            dy = -(r1[1] - r0[1])
            ln = math.hypot(dx, dy) or 1.0
            ex = ox + dx / ln * 26
            ey = oy + dy / ln * 26
            cv.create_line(ox, oy, ex, ey, fill=col, width=2,
                           arrow=tk.LAST, arrowshape=(7, 9, 3))
            cv.create_text(ex + 4, ey, text=label, fill=col,
                           font=(FNT, 8, "bold"), anchor="w")

        cv.create_text(w // 2, h - 6,
                       text=f"yaw {self._iso_yaw:+.0f} deg   pitch {self._iso_pitch:+.0f} deg",
                       fill=DIM, font=(FNT, 8), anchor="s")

    def _draw_side(self):
        """
        SIDE VIEW — X-Y plane.
        X = along arm length  (base = 0 mm, tip ≈ 270 mm at rest)
        Y = bending direction (Y=0 = straight horizontal, +Y = up, -Y = down)
        Ellipse zone: X 220–260 mm, Y ±60 mm
        """
        cv = self._cv_side
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 30 or h < 30: return
        cv.delete("all")

        kin     = compute_pcc_kinematics_mm(self._disps)
        all_pts = kin["all_pts"]
        pt_seg  = kin["pt_seg"]
        N       = len(all_pts)

        # ── Coordinate system ────────────────────────────────────────────────
        # World range: X = -50..310 mm  (arm body can reach X=-35 when bent)
        #              Y = -230..+230 mm (arm body arcs to ±205mm during tracking)
        X_LO, X_HI = -50.0,  310.0
        Y_LO, Y_HI = -230.0, 230.0
        mg_l, mg_r, mg_t, mg_b = 42, 12, 12, 28   # canvas margins (px)

        plot_w = w - mg_l - mg_r
        plot_h = h - mg_t - mg_b
        scl_x  = plot_w  / (X_HI - X_LO) * self._side_zoom
        scl_y  = plot_h  / (Y_HI - Y_LO) * self._side_zoom

        # Use same scale for both axes so shape is undistorted
        scl    = min(scl_x, scl_y)
        # Origin pixel (world 0,0 = base of robot)
        ox = mg_l + int((0 - X_LO) * scl)
        oy = mg_t + int((Y_HI - 0) * scl)

        def px(xmm, ymm):
            return ox + xmm * scl, oy - ymm * scl

        def in_canvas(cx, cy):
            return -20 < cx < w+20 and -20 < cy < h+20

        # ── Grid ─────────────────────────────────────────────────────────────
        for xv in range(0, 310, 30):
            gx, _ = px(xv, 0)
            if 0 < gx < w:
                cv.create_line(gx, mg_t, gx, h-mg_b, fill="#e2e8f0", width=1)
        for yv in range(-210, 211, 30):
            _, gy = px(0, yv)
            if 0 < gy < h:
                cv.create_line(mg_l, gy, w-mg_r, gy, fill="#e2e8f0", width=1)

        # ── Ellipse operating zone (shaded band) ─────────────────────────────
        ex0, _ = px(220, 0); ex1, _ = px(260, 0)
        _, ey0  = px(0, 60);  _, ey1  = px(0, -60)
        cv.create_rectangle(ex0, ey0, ex1, ey1,
                            fill="#e8f5e9", outline="#a5d6a7", width=1, dash=(3,3))
        # Label inside the box (below its top edge) so it's never clipped
        cv.create_text((ex0+ex1)//2, ey0+10, text="ellipse tip zone",
                       fill="#4caf50", font=(FNT, 7), anchor="n")

        # ── Axes ─────────────────────────────────────────────────────────────
        # X axis (Y=0 line)
        ax0, ay = px(X_LO, 0); ax1, _ = px(X_HI, 0)
        cv.create_line(ax0, ay, ax1, ay, fill=MUTED, width=1)
        # Y axis (X=0 line)
        bx, by0 = px(0, Y_HI); _, by1 = px(0, Y_LO)
        cv.create_line(bx, by0, bx, by1, fill=MUTED, width=1)

        # ── Tick marks + labels ───────────────────────────────────────────────
        for xv in range(0, 310, 30):
            gx, _ = px(xv, 0)
            if mg_l < gx < w - mg_r:
                cv.create_line(gx, ay-3, gx, ay+3, fill=MUTED, width=1)
                cv.create_text(gx, ay+5, text=str(xv), fill=DIM,
                               font=(FNT, 7), anchor="n")
        for yv in range(-210, 211, 30):
            _, gy = px(0, yv)
            if mg_t < gy < h - mg_b:
                cv.create_line(bx-3, gy, bx+3, gy, fill=MUTED, width=1)
                if yv != 0:
                    cv.create_text(bx-5, gy, text=str(yv), fill=DIM,
                                   font=(FNT, 7), anchor="e")

        # Axis labels
        cv.create_text(w - mg_r - 2, ay - 4, text="X (mm) →",
                       fill=MUTED, font=(FNT, 8, "bold"), anchor="e")
        cv.create_text(bx + 3, mg_t + 2, text="↑ Y (mm)",
                       fill=MUTED, font=(FNT, 8, "bold"), anchor="nw")

        # ── Base bracket ──────────────────────────────────────────────────────
        bh = int(15 * scl / 2.0) + 6
        cv.create_rectangle(ox - 10, oy - bh, ox + 4, oy + bh,
                            fill="#e5e7eb", outline=TEXT, width=1)
        for k2 in range(-bh, bh, 8):
            cv.create_line(ox - 10, oy + k2, ox - 16, oy + k2 + 6, fill=MUTED, width=1)

        # ── Shadow tube ───────────────────────────────────────────────────────
        for k in range(1, N):
            x0c, y0c = px(*all_pts[k-1]); x1c, y1c = px(*all_pts[k])
            cv.create_line(x0c, y0c, x1c, y1c, fill="#d1d5db", width=22, capstyle=tk.ROUND)

        # ── Coloured body ─────────────────────────────────────────────────────
        for k in range(1, N):
            s_idx = pt_seg[k]
            t_g   = k / N
            br    = int(self.BODY_RADII[0] + t_g*(self.BODY_RADII[2]-self.BODY_RADII[0]))
            x0c, y0c = px(*all_pts[k-1]); x1c, y1c = px(*all_pts[k])
            cv.create_line(x0c, y0c, x1c, y1c, fill=SEG_COLORS[s_idx],
                           width=max(4, br*2), capstyle=tk.ROUND)

        # ── Paper spacer/end disks: 5 spacer disks + 1 end disk per segment ──
        self._draw_side_disks(cv, all_pts, px)

        # ── Tip marker ────────────────────────────────────────────────────────
        arc_tip_x, arc_tip_y = all_pts[-1]
        tip_px, tip_py = px(arc_tip_x, arc_tip_y)
        ch = 14
        cv.create_line(tip_px-ch, tip_py, tip_px+ch, tip_py, fill=ORANGE, dash=(3,3), width=1)
        cv.create_line(tip_px, tip_py-ch, tip_px, tip_py+ch, fill=ORANGE, dash=(3,3), width=1)
        cv.create_oval(tip_px-8, tip_py-8, tip_px+8, tip_py+8,
                       fill=ORANGE, outline=TEXT, width=2)
        cv.create_text(tip_px+12, tip_py-1, text="EE", fill=ORANGE,
                       font=(FNT, 9, "bold"), anchor="w")

        # ── Attitude arrow ────────────────────────────────────────────────────
        psi = kin["psi"]
        al  = 30
        cv.create_line(tip_px, tip_py,
                       tip_px + al*math.cos(psi), tip_py - al*math.sin(psi),
                       fill=YELLOW, width=2, arrow=tk.LAST, arrowshape=(7,9,3))

        # ── Status bar ────────────────────────────────────────────────────────
        cv.create_text(w//2, h - 4,
                       text=f"EE  x={arc_tip_x:.1f} mm    y={arc_tip_y:+.1f} mm    ψ={math.degrees(psi):+.1f}°",
                       fill=TEXT, font=(FNT, 9, "bold"), anchor="s")

    def _draw_top(self):
        """
        TOP VIEW — X-Z plane, rotated 90 degrees anticlockwise on screen.
        X = arm length, drawn vertically from base to tip.
        Z = out-of-plane, drawn horizontally. Z remains 0 for this planar robot.
        """
        cv = self._cv_top
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 30 or h < 30: return
        cv.delete("all")

        kin     = compute_pcc_kinematics_mm(self._disps)
        all_pts = kin["all_pts"]
        pt_seg  = kin["pt_seg"]
        N       = len(all_pts)

        # ── Coordinate system ────────────────────────────────────────────────
        X_LO, X_HI = -10.0, 310.0
        Z_LO, Z_HI = -60.0,  60.0   # Z range shown (always 0, but give context)
        mg_l, mg_r, mg_t, mg_b = 36, 36, 16, 34

        plot_w = w - mg_l - mg_r
        plot_h = h - mg_t - mg_b
        scl_x  = plot_h / (X_HI - X_LO) * self._top_zoom
        scl_z  = plot_w / (Z_HI - Z_LO) * self._top_zoom
        scl    = min(scl_x, scl_z)

        ox = mg_l + plot_w / 2.0              # Z=0 pixel (centre)
        oy = mg_t + (X_HI - 0.0) * scl        # X=0 pixel near bottom

        def px(xmm, zmm=0.0):
            return ox - zmm * scl, oy - xmm * scl

        # ── Grid ─────────────────────────────────────────────────────────────
        for xv in range(0, 310, 30):
            _, gy = px(xv, 0)
            if 0 < gy < h:
                cv.create_line(mg_l, gy, w-mg_r, gy, fill="#e2e8f0", width=1)
        for zv in range(-60, 61, 30):
            gx, _ = px(0, zv)
            if 0 < gx < w:
                lw = 2 if zv == 0 else 1
                cv.create_line(gx, mg_t, gx, h-mg_b,
                               fill="#c8d8e8" if zv == 0 else "#e2e8f0", width=lw)

        # ── Ellipse target zone, rotated with the top view ───────────────────
        # The real trajectory is in X-Y; top view shows the same X window and a
        # small Z envelope so the target region is still evident.
        p_a = px(220, -20)
        p_b = px(260,  20)
        ex_l, ex_r = min(p_a[0], p_b[0]), max(p_a[0], p_b[0])
        ey_t, ey_b = min(p_a[1], p_b[1]), max(p_a[1], p_b[1])
        cv.create_rectangle(ex_l, ey_t, ex_r, ey_b,
                            fill="#e8f5e9", outline="#4caf50", width=2, dash=(3,3))
        cv.create_text((ex_l+ex_r)//2, ey_t+10, text="ellipse tip zone",
                       fill="#4caf50", font=(FNT, 7), anchor="n")

        # ── Z=0 line (arm rest axis) ──────────────────────────────────────────
        zx, zy0 = px(X_LO, 0); _, zy1 = px(X_HI, 0)
        cv.create_line(zx, zy0, zx, zy1, fill="#90a4ae", width=1, dash=(5,3))

        # ── Axes ─────────────────────────────────────────────────────────────
        # X axis
        ax, ay0 = px(X_LO, 0); _, ay1 = px(X_HI, 0)
        cv.create_line(ax, ay0, ax, ay1, fill=MUTED, width=1)
        # Z axis
        zax0, zay = px(0, Z_HI); zax1, _ = px(0, Z_LO)
        cv.create_line(zax0, zay, zax1, zay, fill=MUTED, width=1)

        # ── Tick marks + labels ───────────────────────────────────────────────
        for xv in range(0, 310, 30):
            _, gy = px(xv, 0)
            if mg_t < gy < h - mg_b:
                cv.create_line(ax-3, gy, ax+3, gy, fill=MUTED, width=1)
                cv.create_text(ax-6, gy, text=str(xv), fill=DIM,
                               font=(FNT, 7), anchor="e")
        for zv in range(-60, 61, 30):
            gx, _ = px(0, zv)
            if mg_l < gx < w - mg_r:
                cv.create_line(gx, zay-3, gx, zay+3, fill=MUTED, width=1)
                if zv != 0:
                    cv.create_text(gx, zay+6, text=str(zv), fill=DIM,
                                   font=(FNT, 7), anchor="n")

        cv.create_text(ax + 6, mg_t + 2, text="X (mm)",
                       fill=MUTED, font=(FNT, 8, "bold"), anchor="nw")
        cv.create_text(w - mg_r - 2, zay - 4, text="Z (mm)",
                       fill=MUTED, font=(FNT, 8, "bold"), anchor="e")

        # ── Z=0 label ─────────────────────────────────────────────────────────
        cv.create_text(ax + 4, zay - 4, text="Z = 0",
                       fill="#607d8b", font=(FNT, 8, "italic"), anchor="sw")

        # ── Base plate ────────────────────────────────────────────────────────
        plate_r = 14
        base_x, base_y = px(0, 0)
        cv.create_oval(base_x - plate_r, base_y - plate_r,
                       base_x + plate_r, base_y + plate_r,
                       fill="#e5e7eb", outline=MUTED, width=1)

        # ── Arm (always on Z=0 line) ──────────────────────────────────────────
        for k in range(1, N):
            x0, z0 = px(all_pts[k-1][0], 0.0)
            x1, z1 = px(all_pts[k][0],   0.0)
            cv.create_line(x0, z0, x1, z1, fill="#d1d5db", width=12, capstyle=tk.ROUND)
        for k in range(1, N):
            s_idx = pt_seg[k]
            t_g   = k / N
            br    = int(self.BODY_RADII[0] + t_g*(self.BODY_RADII[2] - self.BODY_RADII[0]))
            x0, z0 = px(all_pts[k-1][0], 0.0)
            x1, z1 = px(all_pts[k][0],   0.0)
            cv.create_line(x0, z0, x1, z1, fill=SEG_COLORS[s_idx],
                           width=max(3, br), capstyle=tk.ROUND)

        # ── Paper spacer/end disks: 5 spacer disks + 1 end disk per segment ──
        self._draw_top_disks(cv, px)

        # ── Tip marker ────────────────────────────────────────────────────────
        arc_tip_x = all_pts[-1][0]
        tip_px, tip_pz = px(arc_tip_x, 0.0)
        cv.create_oval(tip_px-8, tip_pz-8, tip_px+8, tip_pz+8,
                       fill=ORANGE, outline=TEXT, width=2)
        cv.create_text(tip_px + 12, tip_pz, text="EE", fill=ORANGE,
                       font=(FNT, 9, "bold"), anchor="w")

        # ── Annotation ────────────────────────────────────────────────────────
        cv.create_text(w//2, h - 4,
                       text=f"EE  x={arc_tip_x:.1f} mm    z=0.0 mm  (planar — bending is in Y direction only)",
                       fill=TEXT, font=(FNT, 9, "bold"), anchor="s")

    def _draw_heat(self):
        cv = self._cv_heat
        w  = cv.winfo_width()
        if w < 30: return
        cv.delete("all")
        h  = 52; bw = w//6
        for i in range(6):
            x0 = i*bw; x1 = (i+1)*bw
            d  = self._disps[i]
            norm = (d+MAX_DISP_MM)/(2*MAX_DISP_MM)
            norm = max(0.0, min(1.0, norm))
            if norm > 0.5:
                t2 = (norm-0.5)*2
                rr,gg,bb = int(0x18+t2*(0xe8-0x18)), int(0x38*(1-t2)), int(0x38*(1-t2))
            else:
                t2 = (0.5-norm)*2
                rr,gg,bb = int(0x18*(1-t2)), int(0x38*(1-t2)), int(0x18+t2*(0xd8-0x18))
            heat_col = f"#{max(rr,0):02x}{max(gg,0):02x}{max(bb,0):02x}"
            cv.create_rectangle(x0,0,x1,h, fill=heat_col, outline=BORDER)
            cv.create_rectangle(x0,0,x1,3, fill=SEG_COLORS[MOTOR_TO_SEG[i]], outline="")
            cv.create_text((x0+x1)//2, 11, text=f"M{i+1}",
                           fill=MOTOR_COLORS[i], font=(FNT,9,"bold"))
            cv.create_text((x0+x1)//2, 27, text=f"{d:+.1f}mm",
                           fill=TEXT, font=(FNT,11,"bold"))

    def _update_pose(self):
        kin = compute_pcc_kinematics_mm(self._disps)
        self._pose_widgets["tip_x"].config(text=f"{kin['tip_x']:+.2f}")
        self._pose_widgets["tip_y"].config(text=f"{kin['tip_y']:.2f}")
        self._pose_widgets["psi"].config(text=f"{math.degrees(kin['psi']):+.1f}")
        for j, key in enumerate(["th1","th2","th3"]):
            self._pose_widgets[key].config(text=f"{math.degrees(kin['thetas'][j]):+.1f}")
        self._pose_widgets["len"].config(text=f"{TOTAL_LEN_MM:.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# MANUAL JOG PANEL
# ══════════════════════════════════════════════════════════════════════════════

class ManualJogPanel(tk.Frame):

    def __init__(self, parent, get_robot, get_disps, apply_disp_fn, log_fn, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._get_robot  = get_robot
        self._get_disps  = get_disps
        self._apply_disp = apply_disp_fn
        self._log        = log_fn
        self._disp_vars  = [tk.StringVar(value="0.0") for _ in range(6)]
        self._speed_vars = [tk.StringVar(value="20.0") for _ in range(6)]
        self._step_var   = tk.StringVar(value="0.5")
        self._jog_jobs   = {}
        self._bar_canvases, self._bar_fills, self._bar_lbls = [], [], []
        self._live_lbls  = []
        self._build()

    def _build(self):
        tb = tk.Frame(self, bg=PANEL)
        tb.pack(fill="x")
        tk.Label(tb, text="MANUAL JOG", font=(FNT_H,12,"bold"),
                 fg=ACCENT, bg=PANEL).pack(side="left", padx=14, pady=10)
        tk.Label(tb, text="STEP:", font=(FNT,11), fg=MUTED, bg=PANEL
                 ).pack(side="left", padx=(14,4))
        for s in ["0.1","0.5","1","2","5"]:
            tk.Radiobutton(tb, text=f"{s}mm", variable=self._step_var, value=s,
                           font=(FNT,11,"bold"), bg=PANEL, fg=TEXT,
                           selectcolor=INPUT, activebackground=PANEL,
                           indicatoron=False, relief="flat", cursor="hand2",
                           padx=7, pady=3).pack(side="left", padx=2)
        for txt,c,fg2,cmd in [
            ("⚠ E-STOP",RED,TEXT,self._estop),
            ("STOP ALL",YELLOW,BG,self._stop_all),
            ("ZERO ALL",MUTED,TEXT,self._zero_all),
        ]:
            tk.Button(tb, text=txt, font=(FNT,11,"bold"),
                      bg=c, fg=fg2, relief="flat", cursor="hand2",
                      padx=8, pady=4, command=cmd).pack(side="right", padx=3)

        hdr = tk.Frame(self, bg=PANEL)
        hdr.pack(fill="x")
        for t,w in [("",4),("Motor",7),("Displacement bar",24),
                    ("Live mm",9),("  Jog",10),("Step",9),
                    ("Target mm",10),("Speed mm/s",11)]:
            tk.Label(hdr, text=t, font=(FNT,10,"bold"), fg=MUTED, bg=PANEL,
                     width=w, anchor="w").pack(side="left", padx=2, pady=3)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        for seg in range(3):
            sc_ = SEG_COLORS[seg]
            sf  = tk.Frame(self, bg=BG)
            sf.pack(fill="x")
            tk.Frame(sf, bg=sc_, width=4).pack(side="left", fill="y")
            tk.Label(sf, text=f"  ─── {SEG_NAMES[seg]} (θ{seg+1})  ───",
                     font=(FNT,10,"bold"), fg=sc_, bg=BG).pack(
                     side="left", padx=6, pady=2)
            for j in range(2):
                self._build_row(seg*2+j)

    def _build_row(self, i: int):
        col = MOTOR_COLORS[i]
        bg  = BG if i%2==0 else CARD
        row = tk.Frame(self, bg=bg)
        row.pack(fill="x", pady=1)
        tk.Frame(row, bg=col, width=4).pack(side="left", fill="y")
        tk.Label(row, text=f"M{i+1}", font=(FNT,9,"bold"), fg=col, bg=bg,
                 width=5).pack(side="left", padx=3)

        c = tk.Canvas(row, width=155, height=26, bg=bg, highlightthickness=0)
        c.pack(side="left", padx=4)
        c.create_rectangle(2,9,153,19, fill=INPUT, outline="")
        c.create_line(77,5,77,23, fill=BORDER, width=1)
        bar_fill = c.create_rectangle(77,9,77,19, fill=col, outline="")
        bar_lbl  = c.create_text(77,14, text="0.0", fill=TEXT, font=(FNT,9))
        self._bar_canvases.append(c)
        self._bar_fills.append(bar_fill)
        self._bar_lbls.append(bar_lbl)

        ll = tk.Label(row, text="  0.0", font=(FNT,13,"bold"), fg=col, bg=bg, width=7)
        ll.pack(side="left", padx=4)
        self._live_lbls.append(ll)

        b_neg = tk.Button(row, text=" − ", font=(FNT,14,"bold"),
                          bg=INPUT, fg=col, relief="flat", cursor="hand2", padx=8, pady=2)
        b_pos = tk.Button(row, text=" + ", font=(FNT,14,"bold"),
                          bg=INPUT, fg=col, relief="flat", cursor="hand2", padx=8, pady=2)
        b_neg.pack(side="left", padx=1)
        b_pos.pack(side="left", padx=1)
        b_neg.bind("<ButtonPress-1>",   lambda e,idx=i: self._jog_start(idx,-1))
        b_neg.bind("<ButtonRelease-1>", lambda e,idx=i: self._jog_stop(idx))
        b_pos.bind("<ButtonPress-1>",   lambda e,idx=i: self._jog_start(idx,+1))
        b_pos.bind("<ButtonRelease-1>", lambda e,idx=i: self._jog_stop(idx))

        tk.Button(row, text="−S", font=(FNT,10,"bold"), bg=INPUT, fg=DIM,
                  relief="flat", cursor="hand2", padx=5, pady=1,
                  command=lambda idx=i: self._step_jog(idx,-1)).pack(side="left",padx=1)
        tk.Button(row, text="+S", font=(FNT,10,"bold"), bg=INPUT, fg=DIM,
                  relief="flat", cursor="hand2", padx=5, pady=1,
                  command=lambda idx=i: self._step_jog(idx,+1)).pack(side="left",padx=1)

        e = tk.Entry(row, textvariable=self._disp_vars[i], width=8,
                     font=(FNT,12), bg=INPUT, fg=TEXT,
                     insertbackground=TEXT, relief="flat",
                     highlightbackground=BORDER, highlightthickness=1)
        e.pack(side="left", padx=5)
        e.bind("<Return>",   lambda ev,idx=i: self._send(idx))
        e.bind("<KP_Enter>", lambda ev,idx=i: self._send(idx))
        tk.Button(row, text="SEND", font=(FNT,10,"bold"),
                  bg=ACCENT, fg=BG, relief="flat", cursor="hand2", padx=6, pady=2,
                  command=lambda idx=i: self._send(idx)).pack(side="left",padx=2)
        tk.Entry(row, textvariable=self._speed_vars[i], width=7,
                 font=(FNT,12), bg=INPUT, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 highlightbackground=BORDER, highlightthickness=1).pack(side="left",padx=4)

    def _get_speed(self, i):
        try:   return max(0.1, min(float(self._speed_vars[i].get()), MAX_SPEED_MMS))
        except: return 20.0
    def _get_step(self):
        try:   return float(self._step_var.get())
        except: return 5.0

    def _send(self, i):
        try:
            d = max(-MAX_DISP_MM, min(float(self._disp_vars[i].get()), MAX_DISP_MM))
        except ValueError:
            self._log(f"M{i+1}: invalid value", "warn"); return
        self._apply_disp(i, d, self._get_speed(i))

    def _jog_start(self, i, sign):
        self._jog_stop(i)
        def tick():
            disps = self._get_disps()
            nd = max(-MAX_DISP_MM, min(MAX_DISP_MM, disps[i]+sign*self._get_step()*0.25))
            self._apply_disp(i, nd, self._get_speed(i))
            self._disp_vars[i].set(f"{nd:.2f}")
            self._jog_jobs[i] = self.after(100, tick)
        tick()

    def _jog_stop(self, i):
        if i in self._jog_jobs:
            self.after_cancel(self._jog_jobs[i]); del self._jog_jobs[i]

    def _step_jog(self, i, sign):
        disps = self._get_disps()
        nd = max(-MAX_DISP_MM, min(MAX_DISP_MM, disps[i]+sign*self._get_step()))
        self._apply_disp(i, nd, self._get_speed(i))
        self._disp_vars[i].set(f"{nd:.2f}")

    def _estop(self):
        for i in range(6): self._jog_stop(i)
        self._zero_all()
        self._log("[E-STOP]", "err")

    def _stop_all(self):
        for i in range(6): self._jog_stop(i)

    def _zero_all(self):
        for i in range(6):
            self._apply_disp(i, 0.0, 20.0)
            self._disp_vars[i].set("0.0")

    def update_disp(self, idx1: int, disp: float):
        i  = idx1-1
        c  = self._bar_canvases[i]
        bw = 155
        ratio   = (disp+MAX_DISP_MM)/(2*MAX_DISP_MM)
        ratio   = max(0.0, min(1.0, ratio))
        fill_x  = 2+int((bw-4)*ratio)
        centre  = 77
        col     = MOTOR_COLORS[i]
        c.coords(self._bar_fills[i], min(centre,fill_x),9,max(centre,fill_x),19)
        c.itemconfig(self._bar_fills[i], fill=col if disp>=0 else "#d1d5db")
        c.itemconfig(self._bar_lbls[i], text=f"{disp:+.1f}")
        self._live_lbls[i].config(text=f"{disp:+.1f}")
        self._disp_vars[i].set(f"{disp:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# CSV STATE PANEL
# ══════════════════════════════════════════════════════════════════════════════

class CsvStatePanel(tk.Frame):

    def __init__(self, parent, get_robot, apply_state_fn, log_fn, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._get_robot   = get_robot
        self._apply_state = apply_state_fn
        self._log         = log_fn
        self._states: List[TendonState] = []
        self._filepath    = tk.StringVar()
        self._loop_var    = tk.BooleanVar(value=False)
        self._delay_var   = tk.StringVar(value="0")
        self._wait_var    = tk.BooleanVar(value=True)
        self._running     = False
        self._paused      = False
        self._abort_flag  = threading.Event()
        self._current_step= -1
        self._row_frames  = []
        self._timing_log: List[Tuple] = []
        self._seq_start   = self._step_start = 0.0
        self._phase       = "idle"
        self._sw_job      = None
        self._build()

    def _build(self):
        fc = tk.Frame(self, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        fc.pack(fill="x", pady=(0,4))
        tk.Label(fc, text="CSV STATE FILE", font=(FNT_H,12,"bold"),
                 fg=ACCENT, bg=CARD).pack(side="left", padx=10, pady=8)
        tk.Entry(fc, textvariable=self._filepath, font=(FNT,12), bg=INPUT, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 highlightbackground=BORDER, highlightthickness=1,
                 width=36).pack(side="left", padx=(0,6))
        for txt,col,fg2,cmd in [
            ("BROWSE",BLUE,BG,self._browse),
            ("LOAD FILE",ACCENT,BG,self._load_file),
            ("SAMPLE DATA",INPUT,TEXT,self._load_sample),
        ]:
            tk.Button(fc, text=txt, font=(FNT,11,"bold"),
                      bg=col, fg=fg2, relief="flat", cursor="hand2",
                      padx=8, pady=4, command=cmd).pack(side="left", padx=3)

        pc = tk.Frame(self, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        pc.pack(fill="x", pady=(0,4))
        ctrl = tk.Frame(pc, bg=CARD)
        ctrl.pack(fill="x", padx=8, pady=8)

        self.btn_run   = tk.Button(ctrl, text="▶  RUN", font=(FNT,13,"bold"),
                                   bg=GREEN, fg=BG, relief="flat", cursor="hand2",
                                   padx=16, pady=6, command=self._run, state="disabled")
        self.btn_run.pack(side="left", padx=(0,4))
        self.btn_step  = tk.Button(ctrl, text="⏭  STEP", font=(FNT,13,"bold"),
                                   bg=BLUE, fg=BG, relief="flat", cursor="hand2",
                                   padx=14, pady=6, command=self._step_once, state="disabled")
        self.btn_step.pack(side="left", padx=(0,4))
        self.btn_pause = tk.Button(ctrl, text="⏸  PAUSE", font=(FNT,13,"bold"),
                                   bg=YELLOW, fg=BG, relief="flat", cursor="hand2",
                                   padx=14, pady=6, command=self._pause, state="disabled")
        self.btn_pause.pack(side="left", padx=(0,4))
        self.btn_abort = tk.Button(ctrl, text="■  ABORT", font=(FNT,13,"bold"),
                                   bg=RED, fg=TEXT, relief="flat", cursor="hand2",
                                   padx=14, pady=6, command=self._abort, state="disabled")
        self.btn_abort.pack(side="left", padx=(0,14))
        tk.Label(ctrl, text="DELAY OVR:", font=(FNT,11), fg=MUTED, bg=CARD).pack(side="left")
        tk.Entry(ctrl, textvariable=self._delay_var, width=5, font=(FNT,12), bg=INPUT, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 highlightbackground=BORDER, highlightthickness=1).pack(side="left", padx=(3,10))
        tk.Checkbutton(ctrl, text="Wait motors", variable=self._wait_var,
                       font=(FNT,11), bg=CARD, fg=TEXT, selectcolor=INPUT,
                       activebackground=CARD).pack(side="left", padx=(0,10))
        tk.Checkbutton(ctrl, text="Loop", variable=self._loop_var,
                       font=(FNT,11), bg=CARD, fg=TEXT, selectcolor=INPUT,
                       activebackground=CARD).pack(side="left")
        self.lbl_status = tk.Label(ctrl, text="No states loaded.",
                                   font=(FNT,11), fg=MUTED, bg=CARD)
        self.lbl_status.pack(side="right", padx=10)

        style = ttk.Style()
        style.configure("Seq.Horizontal.TProgressbar", troughcolor=INPUT, background=GREEN)
        self.prog_var = tk.DoubleVar(value=0)
        ttk.Progressbar(pc, variable=self.prog_var, maximum=100,
                        style="Seq.Horizontal.TProgressbar").pack(fill="x", padx=8, pady=(0,6))

        bot = tk.Frame(self, bg=BG)
        bot.pack(fill="x", pady=(0,4))
        tbl_outer = tk.Frame(bot, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        tbl_outer.pack(side="left", fill="both", expand=True)
        tk.Label(tbl_outer, text=" STATE TABLE  (click row to jump)",
                 font=(FNT,9,"bold"), fg=MUTED, bg=PANEL).pack(fill="x")
        tbl_cv = tk.Canvas(tbl_outer, bg=CARD, highlightthickness=0, height=200)
        tbl_sb = ttk.Scrollbar(tbl_outer, orient="vertical", command=tbl_cv.yview)
        self._tbl_inner = tk.Frame(tbl_cv, bg=CARD)
        self._tbl_inner.bind("<Configure>",
            lambda e: tbl_cv.configure(scrollregion=tbl_cv.bbox("all")))
        tbl_cv.create_window((0,0), window=self._tbl_inner, anchor="nw")
        tbl_cv.configure(yscrollcommand=tbl_sb.set)
        tbl_sb.pack(side="right", fill="y")
        tbl_cv.pack(side="left", fill="both", expand=True)
        self._build_table_header()

    def _browse(self):
        path = filedialog.askopenfilename(title="Select State CSV",
                                          filetypes=[("CSV","*.csv"),("All","*.*")])
        if path: self._filepath.set(path)

    def _load_file(self):
        path = self._filepath.get().strip()
        if not path: messagebox.showwarning("No File","Enter or browse a CSV path."); return
        states, err = parse_states_csv(path)
        if not states: messagebox.showerror("Parse Error", err or "No states found."); return
        self._states = states
        self._populate_table()
        self._enable_run()
        self.lbl_status.config(text=f"{len(states)} states loaded", fg=GREEN)
        if err: self._log(f"[CSV] Warnings: {err}", "warn")

    def _load_sample(self):
        self._states = make_sample_states()
        self._populate_table(); self._enable_run()
        self.lbl_status.config(text=f"Sample data loaded ({len(self._states)} states)", fg=GREEN)

    def _build_table_header(self):
        hdr = tk.Frame(self._tbl_inner, bg=PANEL)
        hdr.pack(fill="x")
        cols = [("#",3),("State Name",14),("Delay",6)]
        for i in range(6): cols.append((MOTOR_SHORT[i],13))
        for t,w in cols:
            tk.Label(hdr, text=t, font=(FNT,9,"bold"), fg=MUTED, bg=PANEL,
                     width=w, anchor="w").pack(side="left", padx=2, pady=2)

    def _populate_table(self):
        for w in list(self._tbl_inner.winfo_children())[1:]: w.destroy()
        self._row_frames.clear()
        for idx, state in enumerate(self._states):
            bg = CARD if idx%2==0 else INPUT
            row = tk.Frame(self._tbl_inner, bg=bg, cursor="hand2")
            row.pack(fill="x")
            row.bind("<Button-1>", lambda e,i=idx: self._jump_to(i))
            self._row_frames.append(row)
            tk.Label(row, text=str(idx+1), font=(FNT,9), fg=MUTED, bg=bg,
                     width=3, anchor="w").pack(side="left", padx=2, pady=2)
            tk.Label(row, text=state.name[:14], font=(FNT,9), fg=TEXT, bg=bg,
                     width=14, anchor="w").pack(side="left", padx=2)
            tk.Label(row, text=f"{state.delay_s:.1f}s", font=(FNT,9),
                     fg=YELLOW, bg=bg, width=6, anchor="w").pack(side="left", padx=2)
            for i in range(6):
                d,s = state.motors[i]
                col = MOTOR_COLORS[i] if abs(d)>0.5 else MUTED
                tk.Label(row, text=f"{d:+.0f}@{s:.0f}", font=(FNT,9),
                         fg=col, bg=bg, width=13, anchor="w").pack(side="left", padx=2)

    def _jump_to(self, idx: int):
        if not self._states or self._running: return
        state = self._states[idx]
        self._apply_state(state)
        self._highlight_row(idx)

    def _highlight_row(self, idx):
        for i, fr in enumerate(self._row_frames):
            bg = GREEN if i==idx else (CARD if i%2==0 else INPUT)
            fr.config(bg=bg)
            for w in fr.winfo_children():
                try: w.config(bg=bg)
                except: pass

    def _enable_run(self):
        self.btn_run.config(state="normal")
        self.btn_step.config(state="normal")

    def _run(self):
        if self._running or not self._states: return
        self._running = True; self._paused = False
        self._abort_flag.clear()
        self.btn_run.config(state="disabled")
        self.btn_step.config(state="disabled")
        self.btn_pause.config(state="normal", text="⏸  PAUSE")
        self.btn_abort.config(state="normal")
        self.prog_var.set(0)
        self._seq_start = time.time()
        self._sw_job    = self.after(100, self._tick_sw)
        threading.Thread(target=self._worker, daemon=True).start()

    def _step_once(self):
        if self._running or not self._states: return
        idx = (self._current_step+1) % len(self._states)
        self._current_step = idx
        state = self._states[idx]
        self._apply_state(state)
        self._highlight_row(idx)

    def _pause(self):
        if self._paused:
            self._paused = False
            self.btn_pause.config(text="⏸  PAUSE", bg=YELLOW)
        else:
            self._paused = True
            self.btn_pause.config(text="▶  RESUME", bg=GREEN)

    def _abort(self): self._abort_flag.set(); self._paused = False

    def _worker(self):
        total = len(self._states)
        run_n = 0
        while True:
            run_n += 1
            for idx, state in enumerate(self._states):
                if self._abort_flag.is_set():
                    self.after(0, self._done, "Aborted."); return
                while self._paused:
                    if self._abort_flag.is_set():
                        self.after(0, self._done, "Aborted."); return
                    time.sleep(0.05)
                self._current_step = idx
                self.after(0, self._highlight_row, idx)
                self.after(0, self.prog_var.set, idx/total*100)
                self.after(0, self._apply_state, state)
                t_step = time.time()
                if self._wait_var.get():
                    max_d = max(abs(m[0]) for m in state.motors)
                    max_s = max(m[1] for m in state.motors)
                    est   = (max_d/max_s+0.3) if max_s > 0 else 1.0
                    t_end = time.time()+est
                    while time.time() < t_end:
                        if self._abort_flag.is_set(): break
                        time.sleep(0.04)
                try:    gui_delay = float(self._delay_var.get())
                except: gui_delay = 0.0
                delay = gui_delay if gui_delay > 0 else state.delay_s
                if delay > 0:
                    t0 = time.time()
                    while time.time()-t0 < delay:
                        if self._abort_flag.is_set():
                            self.after(0, self._done, "Aborted."); return
                        time.sleep(0.04)
            self.after(0, self.prog_var.set, 100)
            if not self._loop_var.get() or self._abort_flag.is_set(): break
            time.sleep(0.1)
        self.after(0, self._done, f"Complete ({run_n} run{'s' if run_n>1 else ''}).")

    def _done(self, msg):
        self._running = False; self._paused = False
        if self._sw_job: self.after_cancel(self._sw_job); self._sw_job = None
        self.btn_run.config(state="normal")
        self.btn_step.config(state="normal")
        self.btn_pause.config(state="disabled", text="⏸  PAUSE", bg=YELLOW)
        self.btn_abort.config(state="disabled")
        self.lbl_status.config(text=msg, fg=DIM)

    def _tick_sw(self):
        if not self._running: return
        elapsed = time.time()-self._seq_start
        m = int(elapsed//60); s = elapsed%60
        self._sw_job = self.after(100, self._tick_sw)


# ══════════════════════════════════════════════════════════════════════════════
# TRAJECTORY CONTROL PANEL  (NEW — implements Zhai et al. 2025)
# ══════════════════════════════════════════════════════════════════════════════

class TrajectoryControlPanel(tk.Frame):
    """
    Closed-loop trajectory tracking with online Kalman Jacobian compensation.
    Implements the full algorithm from Zhai et al. 2025.
    Works in both simulation and real hardware modes.
    """

    PLOT_W = 280
    PLOT_H = 320
    ERR_H  = 120

    def __init__(self, parent, get_robot, set_disps_fn, log_fn, simulated=True, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._get_robot   = get_robot
        self._set_disps   = set_disps_fn   # fn(disps_mm[6], speeds_mm/s[6])
        self._log         = log_fn
        self._simulated   = simulated

        # Controller state
        self._controller  : Optional[KalmanJacobianController] = None
        self._running      = False
        self._thread       = None
        self._abort_flag   = threading.Event()
        self._u_curr       = np.zeros(3)
        self._disps_mm     = [0.0] * 6

        # Trajectory history (for plotting)
        self._ref_hist    = []   # list of [xr,yr]
        self._act_hist    = []   # list of [x,y]
        self._err_hist    = []   # list of scalar |e|
        self._t_hist      = []
        self._rmse_x = self._rmse_y = self._rmse_psi = 0.0
        self._mae_x  = self._mae_y  = self._mae_psi  = 0.0
        self._err_x_hist = []; self._err_y_hist = []; self._err_psi_hist = []

        # Parameter variables
        self._traj_var  = tk.StringVar(value=TRAJECTORY_NAMES[0])
        self._T_var     = tk.StringVar(value="40.0")
        self._rate_var  = tk.StringVar(value="20")
        self._gamma_var = tk.StringVar(value="0.5")
        self._beta_var  = tk.StringVar(value="0.5")
        self._alpha_var = tk.StringVar(value="1.0")
        self._sigp_var  = tk.StringVar(value="0.35")
        self._sigps_var = tk.StringVar(value="2.5")
        self._kalman_var= tk.BooleanVar(value=True)
        self._noise_var = tk.BooleanVar(value=False)
        self._noise_sig = tk.StringVar(value="0.0005")
        self._speed_var = tk.StringVar(value="40.0")   # motor speed mm/s during tracking
        self._cycles_var= tk.StringVar(value="1")
        self._atthold_var= tk.BooleanVar(value=False)
        self._start_buttons = []
        self._stop_buttons  = []

        self._build()

    def _build(self):
        # ── Header ───────────────────────────────────────────────────────────
        hf = tk.Frame(self, bg=PANEL)
        hf.pack(fill="x")
        tk.Label(hf, text="ELLIPSE TRAJECTORY TRACKING", font=(FNT_H,15,"bold"),
                 fg=ACCENT, bg=PANEL).pack(side="left", padx=14, pady=12)
        tk.Label(hf, text="Traj. 1  ·  Zhai et al. 2025  ·  xr=240+20·cos(ωt)mm   yr=60·sin(ωt)mm",
                 font=(FNT,11), fg=MUTED, bg=PANEL).pack(side="left")
        hb = tk.Frame(hf, bg=PANEL)
        hb.pack(side="right", padx=10, pady=6)
        self.btn_start = tk.Button(hb, text="▶ START", font=(FNT,12,"bold"),
                                   bg=GREEN, fg=BG, relief="flat", cursor="hand2",
                                   padx=16, pady=7, command=self._start)
        self.btn_start.pack(side="left", padx=(0, 5))
        self._start_buttons.append(self.btn_start)
        self.btn_stop = tk.Button(hb, text="■ STOP", font=(FNT,12,"bold"),
                                  bg=RED, fg=BG, relief="flat", cursor="hand2",
                                  padx=16, pady=7, command=self._stop, state="disabled")
        self.btn_stop.pack(side="left")
        self._stop_buttons.append(self.btn_stop)

        # ── Main layout: resizable control pane | resizable plot pane ────────
        main = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=BG, sashwidth=7,
                              sashrelief=tk.RAISED, bd=0, opaqueresize=True)
        main.pack(fill="both", expand=True, padx=4, pady=4)

        # LEFT — configuration
        lf = tk.Frame(main, bg=BG, width=380)
        lf.pack_propagate(False)
        main.add(lf, minsize=380, stretch="always")

        # Trajectory selection
        tc = tk.LabelFrame(lf, text="  Trajectory  ", font=(FNT,11,"bold"),
                           fg=BLUE, bg=CARD, relief="flat",
                           highlightbackground=BORDER, highlightthickness=1)
        tc.pack(fill="x", pady=(0,4), padx=2)
        tk.Label(tc, text="Type:", font=(FNT,12), fg=MUTED, bg=CARD).grid(
            row=0, column=0, sticky="w", padx=8, pady=5)
        ttk.Combobox(tc, textvariable=self._traj_var,
                     values=TRAJECTORY_NAMES, state="readonly",
                     font=(FNT,12), width=28).grid(row=0, column=1, padx=4, pady=5)
        for r,(lbl,var) in enumerate([("Period T (s):",self._T_var),
                                       ("Cycles:",self._cycles_var),
                                       ("Motor speed (mm/s):",self._speed_var),
                                       ("Control rate (Hz):",self._rate_var)], 1):
            tk.Label(tc, text=lbl, font=(FNT,12), fg=MUTED, bg=CARD).grid(
                row=r, column=0, sticky="w", padx=8, pady=4)
            tk.Entry(tc, textvariable=var, width=10, font=(FNT,13), bg=INPUT, fg=TEXT,
                     insertbackground=TEXT, relief="flat",
                     highlightbackground=BORDER, highlightthickness=1).grid(
                row=r, column=1, sticky="w", padx=4, pady=4)

        # Primary controls stay near the top; no scrolling required.
        bf = tk.Frame(lf, bg=BG)
        bf.pack(fill="x", pady=(6, 6), padx=2)
        btn_start_big = tk.Button(bf, text="▶   START TRACKING", font=(FNT,15,"bold"),
                                  bg=GREEN, fg=BG, relief="flat", cursor="hand2",
                                  padx=20, pady=12, command=self._start)
        btn_start_big.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self._start_buttons.append(btn_start_big)
        btn_stop_big = tk.Button(bf, text="■   STOP", font=(FNT,15,"bold"),
                                 bg=RED, fg=BG, relief="flat", cursor="hand2",
                                 padx=20, pady=12, command=self._stop, state="disabled")
        btn_stop_big.pack(side="left", fill="x", expand=True, padx=(4, 0))
        self._stop_buttons.append(btn_stop_big)
        tk.Button(lf, text="↺  RESET PLOTS", font=(FNT,12,"bold"),
                  bg=PANEL, fg=MUTED, relief="flat", cursor="hand2",
                  padx=14, pady=8, command=self._reset).pack(fill="x", pady=(0, 6), padx=2)

        options_wrap = tk.Frame(lf, bg=BG)
        options_wrap.pack(fill="both", expand=True)
        opt_canvas = tk.Canvas(options_wrap, bg=BG, highlightthickness=0)
        opt_scroll = ttk.Scrollbar(options_wrap, orient="vertical", command=opt_canvas.yview)
        opt_canvas.configure(yscrollcommand=opt_scroll.set)
        opt_scroll.pack(side="right", fill="y")
        opt_canvas.pack(side="left", fill="both", expand=True)
        opt_panel = tk.Frame(opt_canvas, bg=BG)
        opt_window = opt_canvas.create_window((0, 0), window=opt_panel, anchor="nw")

        def _sync_options_scroll(_event=None):
            opt_canvas.configure(scrollregion=opt_canvas.bbox("all"))

        def _fit_options_width(event):
            opt_canvas.itemconfigure(opt_window, width=event.width)

        def _scroll_options(event):
            opt_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
            return "break"

        opt_panel.bind("<Configure>", _sync_options_scroll)
        opt_canvas.bind("<Configure>", _fit_options_width)
        opt_canvas.bind("<MouseWheel>", _scroll_options)
        opt_panel.bind("<MouseWheel>", _scroll_options)
        opt_canvas.bind("<Enter>", lambda _e: opt_canvas.bind_all("<MouseWheel>", _scroll_options))
        opt_canvas.bind("<Leave>", lambda _e: opt_canvas.unbind_all("<MouseWheel>"))

        # (attitude held at ψ=0 for ellipse trajectory)

        # Advanced controller tuning: hidden by default, expandable on demand.
        adv_section = tk.Frame(opt_panel, bg=BG)
        adv_section.pack(fill="x", pady=(0,4), padx=2)
        adv_state = {"open": False}
        cc = tk.LabelFrame(adv_section, text="  Tuning Values  ",
                           font=(FNT,11,"bold"), fg=ACCENT, bg=CARD, relief="flat",
                           highlightbackground=BORDER, highlightthickness=1)

        def _toggle_advanced():
            adv_state["open"] = not adv_state["open"]
            if adv_state["open"]:
                adv_btn.config(text="▼  Advanced Controller Parameters")
                cc.pack(fill="x", pady=(2,0))
            else:
                adv_btn.config(text="▶  Advanced Controller Parameters")
                cc.pack_forget()

        adv_btn = tk.Button(adv_section, text="▶  Advanced Controller Parameters",
                            font=(FNT,11,"bold"), bg=PANEL, fg=ACCENT,
                            relief="flat", cursor="hand2", anchor="w",
                            padx=10, pady=7, command=_toggle_advanced)
        adv_btn.pack(fill="x")
        tk.Label(cc, text="Normally keep these fixed unless tuning the controller.",
                 font=(FNT,9), fg=MUTED, bg=CARD).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(5,2))
        params = [
            ("γ  (fading rate):",     self._gamma_var),
            ("β  (controller gain):", self._beta_var),
            ("α  (regularisation):",  self._alpha_var),
            ("σₚ (pos threshold):",   self._sigp_var),
            ("σᵩ (att threshold):",   self._sigps_var),
        ]
        for r,(lbl,var) in enumerate(params):
            tk.Label(cc, text=lbl, font=(FNT,12), fg=MUTED, bg=CARD).grid(
                row=r+1, column=0, sticky="w", padx=8, pady=3)
            tk.Entry(cc, textvariable=var, width=10, font=(FNT,12), bg=INPUT, fg=TEXT,
                     insertbackground=TEXT, relief="flat",
                     highlightbackground=BORDER, highlightthickness=1).grid(
                row=r+1, column=1, sticky="w", padx=4, pady=3)

        # Simulation options
        sc2 = tk.LabelFrame(opt_panel, text="  Simulation Options  ",
                            font=(FNT,10,"bold"), fg=MUTED, bg=CARD, relief="flat",
                            highlightbackground=BORDER, highlightthickness=1)
        sc2.pack(fill="x", pady=(0,4), padx=2)
        tk.Checkbutton(sc2, text="Enable Kalman compensation",
                       variable=self._kalman_var, font=(FNT,11,"bold"),
                       fg=GREEN, bg=CARD, selectcolor=INPUT, activebackground=CARD).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=8, pady=3)
        tk.Checkbutton(sc2, text="Add measurement noise",
                       variable=self._noise_var, font=(FNT,10), bg=CARD, fg=TEXT,
                       selectcolor=INPUT, activebackground=CARD).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=8, pady=3)
        tk.Label(sc2, text="Noise σ (m):", font=(FNT,10), fg=MUTED, bg=CARD).grid(
            row=2, column=0, sticky="w", padx=8, pady=2)
        tk.Entry(sc2, textvariable=self._noise_sig, width=10, font=(FNT,11), bg=INPUT, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 highlightbackground=BORDER, highlightthickness=1).grid(
            row=2, column=1, sticky="w", padx=4, pady=2)

        # Metrics display
        mf = tk.LabelFrame(opt_panel, text="  Live Metrics  ",
                           font=(FNT,11,"bold"), fg=ORANGE, bg=CARD, relief="flat",
                           highlightbackground=BORDER, highlightthickness=1)
        mf.pack(fill="x", pady=(0,4), padx=2)
        self._metric_lbls = {}
        for r,(key,lbl,col) in enumerate([
            ("rmse_x",  "RMSE x (mm):",  ACCENT),
            ("rmse_y",  "RMSE y (mm):",  ACCENT),
            ("rmse_psi","RMSE ψ (°):",   YELLOW),
            ("mae_x",   "MAE  x (mm):",  BLUE),
            ("mae_y",   "MAE  y (mm):",  BLUE),
            ("mae_psi", "MAE  ψ (°):",   BLUE),
            ("err_now",  "Current |e|:", RED),
            ("steps",    "Steps:",        MUTED),
        ]):
            tk.Label(mf, text=lbl, font=(FNT,12), fg=MUTED, bg=CARD).grid(
                row=r, column=0, sticky="w", padx=8, pady=2)
            lv = tk.Label(mf, text="—", font=(FNT,13,"bold"), fg=col, bg=CARD, width=10)
            lv.grid(row=r, column=1, sticky="w", padx=4)
            self._metric_lbls[key] = lv

        # Jacobian info
        jf = tk.LabelFrame(opt_panel, text="  Jacobian Info  ",
                           font=(FNT,10,"bold"), fg=SEG_COLORS[0], bg=CARD, relief="flat",
                           highlightbackground=BORDER, highlightthickness=1)
        jf.pack(fill="x", pady=(0,4), padx=2)
        self._jac_lbl = tk.Label(jf, text="—", font=(FNT, 9), fg=MUTED, bg=CARD,
                                 justify="left", anchor="w")
        self._jac_lbl.pack(fill="x", padx=8, pady=4)

        # ── Quick Actions (fill bottom-left space) ───────────────────────────
        qa = tk.LabelFrame(opt_panel, text="  Quick Actions  ",
                           font=(FNT,11,"bold"), fg=MUTED, bg=CARD, relief="flat",
                           highlightbackground=BORDER, highlightthickness=1)
        qa.pack(fill="x", pady=(0,4), padx=2)
        qa_row1 = tk.Frame(qa, bg=CARD)
        qa_row1.pack(fill="x", padx=6, pady=(6,3))
        tk.Button(qa_row1, text="💾  Export CSV", font=(FNT,12,"bold"),
                  bg=BLUE, fg=BG, relief="flat", cursor="hand2",
                  padx=12, pady=10, command=self._export_csv).pack(side="left", fill="x", expand=True, padx=(0,4))
        tk.Button(qa_row1, text="🏠  Home Motors", font=(FNT,12,"bold"),
                  bg=PANEL, fg=TEXT, relief="flat", cursor="hand2",
                  padx=12, pady=10, command=lambda: self._set_disps([0.0]*6,[20.0]*6)).pack(side="left", fill="x", expand=True)
        qa_row2 = tk.Frame(qa, bg=CARD)
        qa_row2.pack(fill="x", padx=6, pady=(3,8))
        tk.Button(qa_row2, text="⟳  1 Cycle Fast", font=(FNT,12,"bold"),
                  bg=ORANGE, fg=BG, relief="flat", cursor="hand2",
                  padx=12, pady=10, command=self._run_one_fast).pack(side="left", fill="x", expand=True, padx=(0,4))
        tk.Button(qa_row2, text="📋  Print Summary", font=(FNT,12,"bold"),
                  bg=PANEL, fg=TEXT, relief="flat", cursor="hand2",
                  padx=12, pady=10, command=self._print_summary).pack(side="left", fill="x", expand=True)
        rf = tk.Frame(main, bg=BG)
        main.add(rf, minsize=250, stretch="always")

        plot_pane = tk.PanedWindow(rf, orient=tk.VERTICAL, bg=BG, sashwidth=7,
                                   sashrelief=tk.RAISED, bd=0, opaqueresize=True)
        plot_pane.pack(fill="both", expand=True)

        traj_frame = tk.Frame(plot_pane, bg=BG)
        err_frame = tk.Frame(plot_pane, bg=BG)
        plot_pane.add(traj_frame, minsize=180, stretch="always")
        plot_pane.add(err_frame, minsize=90, stretch="never")

        # Trajectory plot — fixed scale matching Fig. 6 (x: 215–265 mm, y: ±65 mm)
        tp_lbl = tk.Label(traj_frame, text="END-EFFECTOR TRAJECTORY  (mm)",
                          font=(FNT,10,"bold"), fg=BLUE, bg=PANEL)
        tp_lbl.pack(fill="x")
        self._cv_traj = tk.Canvas(traj_frame, bg=CBKG, highlightthickness=1,
                                  highlightbackground=BORDER,
                                  width=self.PLOT_W, height=self.PLOT_H)
        self._cv_traj.pack(fill="both", expand=True)
        self._cv_traj.bind("<Configure>", lambda e: self._redraw_traj())

        # Error plot
        tk.Label(err_frame, text="TRACKING ERROR MAGNITUDE  (mm)",
                 font=(FNT,12,"bold"), fg=RED, bg=PANEL).pack(fill="x")
        self._cv_err = tk.Canvas(err_frame, bg=CBKG, highlightthickness=1,
                                 highlightbackground=BORDER,
                                 width=self.PLOT_W, height=self.ERR_H)
        self._cv_err.pack(fill="both", expand=True)
        self._cv_err.bind("<Configure>", lambda e: self._redraw_err())

        # Status bar
        self._status_lbl = tk.Label(rf, text="Ready. Press START to begin ellipse tracking.",
                                    font=(FNT,11), fg=MUTED, bg=PANEL)
        self._status_lbl.pack(fill="x")

    # ── Parameter helpers ─────────────────────────────────────────────────────

    def _get_float(self, var: tk.StringVar, default: float) -> float:
        try:   return float(var.get())
        except: return default

    def _set_run_controls(self, running: bool):
        start_state = "disabled" if running else "normal"
        stop_state = "normal" if running else "disabled"
        for btn in self._start_buttons:
            btn.config(state=start_state)
        for btn in self._stop_buttons:
            btn.config(state=stop_state)

    def _make_controller(self) -> KalmanJacobianController:
        return KalmanJacobianController(
            L          = list(L_SEG),
            r          = list(R_TENDON),
            gamma      = self._get_float(self._gamma_var, 0.5),
            sigma_p    = self._get_float(self._sigp_var,  0.35),
            sigma_psi  = self._get_float(self._sigps_var, 2.5),
            alpha      = self._get_float(self._alpha_var, 1.0),
            beta       = self._get_float(self._beta_var,  0.5),
            use_kalman = bool(self._kalman_var.get()),
        )

    # ── Start / Stop / Reset ─────────────────────────────────────────────────

    def _start(self):
        if self._running: return
        self._controller = self._make_controller()
        self._u_curr     = np.zeros(3)
        self._ref_hist.clear(); self._act_hist.clear()
        self._err_hist.clear(); self._t_hist.clear()
        self._err_x_hist.clear(); self._err_y_hist.clear(); self._err_psi_hist.clear()
        self._rmse_x = self._rmse_y = self._rmse_psi = 0.0
        self._mae_x  = self._mae_y  = self._mae_psi  = 0.0

        self._abort_flag.clear()
        self._running = True
        self._set_run_controls(True)
        self._status_lbl.config(text="Running…", fg=GREEN)
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
        self._log("[TRAJ] Trajectory tracking started")

    def _stop(self):
        self._abort_flag.set()
        self._running = False
        self._set_run_controls(False)
        self._status_lbl.config(text="Stopped.", fg=YELLOW)
        self._log("[TRAJ] Stopped")
        # Home motors
        self._set_disps([0.0]*6, [20.0]*6)

    def _reset(self):
        if self._running: self._stop()
        self._ref_hist.clear(); self._act_hist.clear()
        self._err_hist.clear(); self._t_hist.clear()
        self._err_x_hist.clear(); self._err_y_hist.clear(); self._err_psi_hist.clear()
        self._u_curr = np.zeros(3)
        self._disps_mm = [0.0]*6
        for key in self._metric_lbls: self._metric_lbls[key].config(text="—")
        self._jac_lbl.config(text="—")
        self._redraw_traj(); self._redraw_err()
        self._log("[TRAJ] Reset")

    # ── Control loop (background thread) ─────────────────────────────────────

    def _control_loop(self):
        rate     = max(1.0, self._get_float(self._rate_var, 20.0))
        dt       = 1.0 / rate
        T        = max(1.0, self._get_float(self._T_var, 40.0))
        cycles   = max(1, int(self._get_float(self._cycles_var, 1.0)))
        traj_fn  = make_trajectory(self._traj_var.get(), T)
        add_noise= bool(self._noise_var.get())
        noise_s  = self._get_float(self._noise_sig, 0.0005)
        att_hold = bool(self._atthold_var.get())
        mot_spd  = self._get_float(self._speed_var, 40.0)
        total_dur= T * cycles

        t = 0.0
        step = 0

        while not self._abort_flag.is_set() and t <= total_dur:
            t_start = time.perf_counter()

            # ── 1. Reference pose ─────────────────────────────────────────────
            pose_ref = traj_fn(t)
            if att_hold:
                pose_ref[2] = 0.0

            # ── 2. Actual pose ────────────────────────────────────────────────
            if self._simulated:
                # Simulation: PCC model IS the plant + optional noise
                pose_curr = forward_kinematics(self._u_curr)
                if add_noise:
                    pose_curr += np.random.normal(0, noise_s, 3)
            else:
                # Real hardware: read from forward kinematics of commanded state
                # (Feedback from motor encoders could be used here if available)
                pose_curr = forward_kinematics(self._u_curr)

            # ── 3. Control update ─────────────────────────────────────────────
            u_next = self._controller.compute_control(
                self._u_curr, pose_ref, pose_curr)
            u_next = np.clip(u_next, -MAX_DELTA_L, MAX_DELTA_L)

            # ── 4. Convert to motor displacements ────────────────────────────
            disps_mm = u_to_motor_disps_mm(u_next)
            disps_mm = [max(-MAX_DISP_MM, min(MAX_DISP_MM, d)) for d in disps_mm]
            speeds   = [mot_spd] * 6

            # ── 5. Send to motors ─────────────────────────────────────────────
            self._u_curr   = u_next.copy()
            self._disps_mm = list(disps_mm)
            self._set_disps(disps_mm, speeds)

            # ── 6. Record history ─────────────────────────────────────────────
            err = pose_ref - pose_curr
            self._ref_hist.append([pose_ref[0]*1000, pose_ref[1]*1000])
            self._act_hist.append([pose_curr[0]*1000, pose_curr[1]*1000])
            self._err_hist.append(math.hypot(err[0]*1000, err[1]*1000))
            self._err_x_hist.append(abs(err[0]*1000))
            self._err_y_hist.append(abs(err[1]*1000))
            self._err_psi_hist.append(abs(math.degrees(err[2])))
            self._t_hist.append(t)
            step += 1

            # ── 7. Update metrics ─────────────────────────────────────────────
            if step % 5 == 0 or step == 1:
                self._update_metrics(step, err)
                self._update_jac_display()

            # ── 8. Update GUI plots (every 3 steps) ──────────────────────────
            if step % 3 == 0:
                self.after(0, self._redraw_traj)
                self.after(0, self._redraw_err)

            # Status bar
            if step % 20 == 0:
                self.after(0, self._status_lbl.config,
                           {"text": f"t={t:.1f}s / {total_dur:.0f}s   "
                                    f"step {step}   "
                                    f"|e|={self._err_hist[-1]:.2f}mm",
                            "fg": GREEN})

            t += dt

            # Precise timing
            elapsed = time.perf_counter() - t_start
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

        # Done
        self.after(0, self._on_done, step)

    def _on_done(self, steps: int):
        self._running = False
        self._set_run_controls(False)
        msg = (f"Complete — {steps} steps  |  "
               f"RMSE: x={self._rmse_x:.2f}mm  y={self._rmse_y:.2f}mm  "
               f"ψ={self._rmse_psi:.2f}°")
        self._status_lbl.config(text=msg, fg=ACCENT)
        self._log(f"[TRAJ] {msg}")
        # Print Table 2 equivalent
        self._print_summary()

    def _print_summary(self):
        self._log("─"*60)
        self._log("  RESULTS (Zhai et al. 2025 Table 2 format)")
        self._log(f"  Method: {'Proposed (PCC+Kalman)' if self._kalman_var.get() else 'PCC only'}")
        self._log(f"  {'':12s}  {'RMSE':>10s}  {'MAE':>10s}")
        self._log(f"  {'x (mm)':12s}  {self._rmse_x:>10.3f}  {self._mae_x:>10.3f}")
        self._log(f"  {'y (mm)':12s}  {self._rmse_y:>10.3f}  {self._mae_y:>10.3f}")
        self._log(f"  {'ψ (deg)':12s}  {self._rmse_psi:>10.3f}  {self._mae_psi:>10.3f}")
        self._log("─"*60)

    def _export_csv(self):
        """Export trajectory history to CSV."""
        if not self._t_hist:
            self._log("[CSV] No data to export — run a trajectory first.", "warn")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Export trajectory data")
        if not path:
            return
        try:
            with open(path, "w", newline="") as f:
                import csv as _csv
                wr = _csv.writer(f)
                wr.writerow(["t_s","ref_x_mm","ref_y_mm","act_x_mm","act_y_mm","err_mm"])
                for i, t in enumerate(self._t_hist):
                    rx, ry = (self._ref_hist[i] if i < len(self._ref_hist) else (0,0))
                    ax, ay = (self._act_hist[i] if i < len(self._act_hist) else (0,0))
                    er     = self._err_hist[i] if i < len(self._err_hist) else 0.0
                    wr.writerow([f"{t:.4f}", f"{rx:.4f}", f"{ry:.4f}",
                                 f"{ax:.4f}", f"{ay:.4f}", f"{er:.4f}"])
            self._log(f"[CSV] Exported {len(self._t_hist)} rows → {os.path.basename(path)}", "ok")
        except Exception as e:
            self._log(f"[CSV] Export failed: {e}", "err")

    def _run_one_fast(self):
        """Quick-start: 1 cycle at faster period."""
        if self._running:
            return
        self._cycles_var.set("1")
        self._T_var.set("20.0")
        self._rate_var.set("25")
        self._start()
        self._log("[QUICK] 1-cycle fast run started (T=20s, 25 Hz)", "ok")

    def _update_metrics(self, step: int, err: np.ndarray):
        if not self._err_x_hist: return
        ex  = np.array(self._err_x_hist)
        ey  = np.array(self._err_y_hist)
        eps = np.array(self._err_psi_hist)
        self._rmse_x   = float(np.sqrt(np.mean(ex**2)))
        self._rmse_y   = float(np.sqrt(np.mean(ey**2)))
        self._rmse_psi = float(np.sqrt(np.mean(eps**2)))
        self._mae_x    = float(np.mean(np.abs(ex)))
        self._mae_y    = float(np.mean(np.abs(ey)))
        self._mae_psi  = float(np.mean(np.abs(eps)))
        err_now = math.hypot(err[0]*1000, err[1]*1000)

        def upd(key, val):
            self.after(0, self._metric_lbls[key].config, {"text": f"{val:.3f}"})

        upd("rmse_x",  self._rmse_x)
        upd("rmse_y",  self._rmse_y)
        upd("rmse_psi",self._rmse_psi)
        upd("mae_x",   self._mae_x)
        upd("mae_y",   self._mae_y)
        upd("mae_psi", self._mae_psi)
        upd("err_now", err_now)
        self.after(0, self._metric_lbls["steps"].config, {"text": str(step)})

    def _update_jac_display(self):
        ctrl = self._controller
        if ctrl is None: return
        mJ = ctrl.last_mJ; cJ = ctrl.last_cJ
        txt = (f"ᵐJ row1: [{mJ[0,0]:6.2f} {mJ[0,1]:6.2f} {mJ[0,2]:6.2f}]\n"
               f"ᶜJ row1: [{cJ[0,0]:6.2f} {cJ[0,1]:6.2f} {cJ[0,2]:6.2f}]\n"
               f"K_norm: {ctrl.last_K_norm:.4f}")
        self.after(0, self._jac_lbl.config, {"text": txt})

    # ── Plotting ──────────────────────────────────────────────────────────────

    def _redraw_traj(self):
        cv = self._cv_traj
        w  = cv.winfo_width();  h = cv.winfo_height()
        if w < 30 or h < 30: return
        cv.delete("all")

        # ── Coordinate system ─────────────────────────────────────────────────
        # World range: X = 210..285 mm  — covers rest position (270mm) + ellipse (220..260)
        #              Y = -75..+75 mm  — covers ellipse semi-major (±60) with margin
        # The arm rests at X=270mm; ellipse starts at X=260mm so the initial dot
        # is always within this range.
        X_LO, X_HI = 210.0, 285.0
        Y_LO, Y_HI = -75.0,  75.0
        mg_l, mg_r, mg_t, mg_b = 42, 12, 14, 28

        plot_w = w - mg_l - mg_r
        plot_h = h - mg_t - mg_b
        scl_x  = plot_w / (X_HI - X_LO)
        scl_y  = plot_h / (Y_HI - Y_LO)
        scl    = min(scl_x, scl_y)

        ox = mg_l + int((240 - X_LO) * scl)   # X=240 (ellipse centre) anchor
        oy = mg_t + int((Y_HI - 0)   * scl)   # Y=0 anchor

        def to_px(xm, ym):
            return mg_l + (xm - X_LO)*scl, mg_t + (Y_HI - ym)*scl

        # ── Grid ─────────────────────────────────────────────────────────────
        for xv in range(210, 286, 10):
            gx, _ = to_px(xv, 0)
            if mg_l < gx < w - mg_r:
                lw = 2 if xv == 240 else 1
                col = "#c8d8e8" if xv == 240 else "#e2e8f0"
                cv.create_line(gx, mg_t, gx, h-mg_b, fill=col, width=lw)
        for yv in range(-70, 71, 10):
            _, gy = to_px(0, yv)
            if mg_t < gy < h - mg_b:
                lw = 2 if yv == 0 else 1
                col = "#c8d8e8" if yv == 0 else "#e2e8f0"
                cv.create_line(mg_l, gy, w-mg_r, gy, fill=col, width=lw)

        # ── Ellipse bounding box — marks the reference ellipse extents ────────
        # Ellipse: x = 240±20 = 220..260,  y = ±60
        bx0, _ = to_px(220, 0); bx1, _ = to_px(260, 0)
        _, by0  = to_px(0,  60); _, by1  = to_px(0, -60)
        cv.create_rectangle(bx0, by0, bx1, by1,
                            fill="#fafff8", outline="#81c784", width=1, dash=(4,3))
        # Dimension labels placed safely inside the box
        cv.create_text((bx0+bx1)//2, by0 + 8, text="reference ellipse  x:220–260 mm  y:±60 mm",
                       fill="#4caf50", font=(FNT, 7), anchor="n")

        # Mark the arm rest position (X=270, Y=0) with a small indicator
        rx0, ry0 = to_px(270, 0)
        cv.create_line(rx0-6, ry0, rx0+6, ry0, fill="#90a4ae", width=1, dash=(2,2))
        cv.create_line(rx0, ry0-6, rx0, ry0+6, fill="#90a4ae", width=1, dash=(2,2))
        cv.create_text(rx0+4, ry0-8, text="rest\n270mm", fill="#90a4ae",
                       font=(FNT, 6), anchor="sw")

        # ── Axes ─────────────────────────────────────────────────────────────
        ax0, ay = to_px(X_LO, 0); ax1, _ = to_px(X_HI, 0)
        cv.create_line(ax0, ay, ax1, ay, fill=MUTED, width=1)
        bx2, by2_0 = to_px(240, Y_HI); _, by2_1 = to_px(240, Y_LO)
        cv.create_line(bx2, by2_0, bx2, by2_1, fill=MUTED, width=1)

        # ── Tick marks + labels ───────────────────────────────────────────────
        for xv in range(210, 286, 10):
            gx, _ = to_px(xv, 0)
            if mg_l < gx < w - mg_r:
                cv.create_line(gx, ay-3, gx, ay+3, fill=MUTED, width=1)
                cv.create_text(gx, ay+5, text=str(xv), fill=DIM,
                               font=(FNT, 7), anchor="n")
        for yv in range(-70, 71, 10):
            _, gy = to_px(0, yv)
            if mg_t < gy < h - mg_b:
                cv.create_line(bx2-3, gy, bx2+3, gy, fill=MUTED, width=1)
                if yv != 0 and yv % 20 == 0:
                    cv.create_text(bx2 - 5, gy, text=str(yv), fill=DIM,
                                   font=(FNT, 7), anchor="e")

        cv.create_text(w - mg_r - 2, ay - 4, text="X (mm) →",
                       fill=MUTED, font=(FNT, 8, "bold"), anchor="e")
        cv.create_text(bx2 + 3, mg_t + 2, text="↑ Y (mm)",
                       fill=MUTED, font=(FNT, 8, "bold"), anchor="nw")

        # Ellipse centre label
        ecx, ecy = to_px(240, 0)
        cv.create_text(ecx+3, ecy-4, text="centre\n(240, 0)",
                       fill=DIM, font=(FNT, 7), anchor="sw")

        # ── Reference ellipse ─────────────────────────────────────────────────
        T_val = max(1.0, self._get_float(self._T_var, 40.0))
        ell_pts = []
        for k in range(201):
            t_ = k / 200 * T_val
            xr_ = 240 + 20 * math.cos(2*math.pi*t_/T_val)
            yr_ =  60 * math.sin(2*math.pi*t_/T_val)
            ell_pts.append(to_px(xr_, yr_))
        flat_ell = [v for p in ell_pts for v in p]
        cv.create_line(*flat_ell, fill="#78909c", width=2, dash=(6,4), smooth=True)

        # ── Actual path ───────────────────────────────────────────────────────
        if len(self._act_hist) >= 2:
            pts  = [to_px(*p) for p in self._act_hist]
            flat = [v for p in pts for v in p]
            cv.create_line(*flat, fill=ACCENT, width=2, smooth=True)

        # Current tip dot
        if self._act_hist:
            px_, py_ = to_px(*self._act_hist[-1])
            cv.create_oval(px_-7, py_-7, px_+7, py_+7,
                           fill=ORANGE, outline=TEXT, width=2)

        # Current reference dot
        if self._ref_hist:
            rx_, ry_ = to_px(*self._ref_hist[-1])
            cv.create_oval(rx_-4, ry_-4, rx_+4, ry_+4,
                           fill="#ffffff", outline="#78909c", width=2)

        # ── Legend ────────────────────────────────────────────────────────────
        cv.create_line(mg_l, h-10, mg_l+24, h-10, fill="#78909c", width=2, dash=(6,4))
        cv.create_text(mg_l+28, h-10, text="Reference", fill=MUTED, font=(FNT,9), anchor="w")
        cv.create_line(mg_l+110, h-10, mg_l+134, h-10, fill=ACCENT, width=2)
        cv.create_text(mg_l+138, h-10, text="Actual tip", fill=ACCENT, font=(FNT,9), anchor="w")

        # Step counter
        if self._t_hist:
            cv.create_text(w-mg_r-2, h-4, fill=DIM, font=(FNT,8), anchor="se",
                           text=f"n={len(self._t_hist)}  t={self._t_hist[-1]:.1f}s")

    def _redraw_err(self):
        cv = self._cv_err
        w  = cv.winfo_width(); h = cv.winfo_height()
        if w < 30 or h < 30: return
        cv.delete("all")

        for x in range(0, w, 40):
            cv.create_line(x,0,x,h, fill="#d8dfe8", width=1)
        for y in range(0, h, 20):
            cv.create_line(0,y,w,y, fill="#d8dfe8", width=1)

        if not self._err_hist:
            cv.create_text(w//2, h//2, text="Error plot will appear here",
                           fill=DIM, font=(FNT,10), anchor="center")
            return

        margin = 20
        max_e  = max(max(self._err_hist), 1.0)
        N      = len(self._err_hist)

        def to_px(i, e):
            px_ = margin + (i/(max(N-1,1))) * (w-2*margin)
            py_ = (h-margin) - (e/max_e) * (h-2*margin)
            return px_, py_

        # Error line
        if N >= 2:
            pts = [to_px(i,e) for i,e in enumerate(self._err_hist)]
            flat = [v for p in pts for v in p]
            cv.create_line(*flat, fill=RED, width=1.5, smooth=True)

        # Max error label
        cv.create_text(w-4, margin, text=f"{max_e:.2f}mm",
                       fill=RED, font=(FNT,9), anchor="ne")
        cv.create_text(w-4, h-margin, text="0",
                       fill=DIM, font=(FNT,8), anchor="se")
        cv.create_text(4, h-4, text=f"n={N}", fill=DIM, font=(FNT,8), anchor="sw")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class ContinuumGUI(tk.Tk):

    def __init__(self, simulated=True, port="COM3", baud=115200):
        super().__init__()
        self.title("Continuum Manipulator  ·  Ellipse Trajectory Control  ·  Zhai et al. 2025")
        self.configure(bg=BG)
        self.minsize(1400, 820)
        try:    self.state("zoomed")
        except: self.geometry("1600x960")

        self._sim    = simulated
        self._port   = port
        self._baud   = baud
        self._disps  = [0.0] * 6
        self._speeds = [0.0] * 6
        self._connected = False
        self._mon_job   = None
        self.robot: Optional[RobotController] = None
        self.bus:   Optional[RS485Bus]        = None

        self._build()
        self._start_monitor()
        self._poll_log()

        if simulated:
            self.after(300, self._connect)

    def _build(self):
        # Top bar
        top = tk.Frame(self, bg=PANEL)
        top.pack(fill="x")
        tk.Label(top,
                 text="⬡  CONTINUUM MANIPULATOR  ·  ELLIPSE TRAJECTORY  ·  PCC + KALMAN",
                 font=(FNT_H,15,"bold"), fg=ACCENT, bg=PANEL).pack(side="left", padx=16, pady=12)
        tk.Label(top, text="  Zhai et al. 2025  ·  Traj. 1  ·  3-segment  ·  270 mm",
                 font=(FNT,11), fg=MUTED, bg=PANEL).pack(side="left")

        self._lbl_c = tk.Label(top, text="● DISCONNECTED",
                               font=(FNT,13,"bold"), fg=RED, bg=PANEL)
        self._lbl_c.pack(side="right", padx=12)
        mode = "SIM" if self._sim else f"HW · {self._port} · {self._baud}baud"
        tk.Label(top, text=mode, font=(FNT,12), fg=MUTED, bg=PANEL).pack(side="right", padx=4)
        self._mkbtn(top,"DISCONNECT",PANEL,RED,  self._disconnect).pack(side="right",padx=3)
        self._mkbtn(top,"CONNECT",  ACCENT,BG,   self._connect   ).pack(side="right",padx=3)

        # Body: resizable trajectory-control pane | resizable visualization pane
        body = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg=BG, sashwidth=8,
                              sashrelief=tk.RAISED, bd=0, opaqueresize=True)
        body.pack(fill="both", expand=True, padx=8, pady=6)

        # Left: trajectory panel only (no tab switching)
        left_outer = tk.Frame(body, bg=BG, width=720)
        left_outer.pack_propagate(False)
        body.add(left_outer, minsize=460, stretch="always")

        tbar = tk.Frame(left_outer, bg=BG)
        tbar.pack(fill="x", pady=(0,6))
        tk.Label(tbar, text="TRAJECTORY CONTROL", font=(FNT,13,"bold"),
                 fg=ACCENT, bg=BG).pack(side="left", padx=4)
        self._mkbtn(tbar,"  🏠  HOME ALL",GREEN,BG,
                    self._home_all).pack(side="right", padx=(6,0))

        # Trajectory panel (only panel)
        self._traj_panel = TrajectoryControlPanel(
            left_outer,
            get_robot    = lambda: self.robot,
            set_disps_fn = self._apply_disps_from_ctrl,
            log_fn       = self._log,
            simulated    = self._sim,
        )
        self._traj_panel.pack(fill="both", expand=True)

        # Right: resizable motor pane | resizable continuum-view pane
        right = tk.PanedWindow(body, orient=tk.VERTICAL, bg=BG, sashwidth=8,
                               sashrelief=tk.RAISED, bd=0, opaqueresize=True)
        body.add(right, minsize=430, stretch="always")

        motor_pane = tk.Frame(right, bg=BG)
        self._capstan_panel = CapstanPanel(motor_pane)
        self._capstan_panel.pack(fill="both", expand=True)
        right.add(motor_pane, minsize=120, stretch="never")

        viz_pane = tk.Frame(right, bg=BG)
        self._cont_viz = ContinuumVisualizer(viz_pane)
        self._cont_viz.pack(fill="both", expand=True)
        right.add(viz_pane, minsize=260, stretch="always")

        # Log
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        lh = tk.Frame(self, bg=PANEL)
        lh.pack(fill="x")
        tk.Label(lh, text="SYSTEM LOG", font=(FNT,10), fg=MUTED,
                 bg=PANEL, pady=3).pack(side="left", padx=12)
        tk.Button(lh, text="CLEAR", font=(FNT,10), bg=PANEL, fg=MUTED,
                  relief="flat", cursor="hand2",
                  command=self._clear_log).pack(side="right", padx=12)
        self._lbox = scrolledtext.ScrolledText(
            self, height=4, font=(FNT,10),
            bg=PANEL, fg=MUTED, relief="flat", state="disabled", wrap="word")
        self._lbox.pack(fill="x")
        for t,c in [("ok",GREEN),("warn",YELLOW),("err",RED),("info",DIM)]:
            self._lbox.tag_config(t, foreground=c)

    # ── Motion API ────────────────────────────────────────────────────────────

    def _apply_disps_from_ctrl(self, disps_mm: List[float], speeds: List[float]):
        """Called by TrajectoryControlPanel on each control step."""
        motors = normalize_antagonistic_motor_state(list(zip(disps_mm, speeds)))
        disps_mm = [m[0] for m in motors]
        speeds = [m[1] for m in motors]
        self._disps  = list(disps_mm)
        self._speeds = list(speeds)
        self._capstan_panel.update_all(disps_mm, speeds)
        self._cont_viz.set_displacements(disps_mm)
        if self.robot and not self._sim:
            try:
                self.robot.set_all_mm(disps_mm, speeds)
            except Exception as e:
                self._log(f"HW error: {e}", "err")

    def _home_all(self):
        for i in range(6):
            self._disps[i] = 0.0
            self._speeds[i] = 20.0
        self._capstan_panel.update_all(self._disps, self._speeds)
        self._cont_viz.set_displacements(self._disps)
        if self.robot and not self._sim:
            try: self.robot.set_all_mm(self._disps, self._speeds)
            except: pass
        self._log("[HOME] All motors → 0.0 mm", "ok")

    # ── Connection ────────────────────────────────────────────────────────────

    def _connect(self):
        threading.Thread(target=self._connect_bg, daemon=True).start()

    def _connect_bg(self):
        try:
            self.bus   = RS485Bus(self._port, baud=self._baud,
                                  simulated=self._sim)
            self.robot = RobotController(self.bus)
            self.robot.start()
            self._connected = True
            self.after(0, self._conn_ok)
        except Exception as e:
            self.after(0, lambda: self._conn_fail(str(e)))

    def _conn_ok(self):
        mode = "SIMULATION" if self._sim else f"CONNECTED  ({self._port})"
        self._lbl_c.config(text=f"● {mode}", fg=YELLOW if self._sim else GREEN)
        self._log(f"{'Simulation' if self._sim else 'Hardware'} ready — 6 motors active.", "ok")

    def _conn_fail(self, err):
        self._lbl_c.config(text="● FAILED", fg=RED)
        self._log(f"Connection failed: {err}", "err")
        messagebox.showerror("Connection Failed", err)

    def _disconnect(self):
        self._stop_monitor()
        if self.robot:
            try: self.robot.close()
            except: pass
            self.robot = None
        self.bus = None
        self._connected = False
        self._lbl_c.config(text="● DISCONNECTED", fg=RED)
        self._log("Disconnected.")

    # ── Monitor ───────────────────────────────────────────────────────────────

    def _start_monitor(self):
        self._mon_job = self.after(300, self._monitor)

    def _stop_monitor(self):
        if self._mon_job:
            self.after_cancel(self._mon_job); self._mon_job = None

    def _monitor(self):
        # Sync visualizer with current displacements
        self._capstan_panel.update_all(self._disps, self._speeds)
        self._cont_viz.set_displacements(self._disps)

        # If real hardware, poll motor feedback
        if self.robot and not self._sim and self._connected:
            try:
                fb = self.robot.get_all_feedback()
                for i, m in enumerate(self.robot.motors):
                    f = fb.get(m.motor_id)
                    if f and abs(f["velocity_dps"]) > 0.1:
                        v_mm = deg_to_mm(abs(f["velocity_dps"]), i)
                        self._speeds[i] = v_mm
            except Exception:
                pass

        self._mon_job = self.after(200, self._monitor)

    # ── Log ───────────────────────────────────────────────────────────────────

    def _log(self, msg, tag="info"):
        ts = time.strftime("%H:%M:%S")
        self._lbox.config(state="normal")
        self._lbox.insert("end", f"{ts}  {msg}\n", tag)
        self._lbox.see("end")
        self._lbox.config(state="disabled")

    def _poll_log(self):
        while not _lq.empty():
            m = _lq.get_nowait()
            t = "warn" if "WARNING" in m else ("err" if "ERROR" in m else "info")
            self._log(m, t)
        self.after(150, self._poll_log)

    def _clear_log(self):
        self._lbox.config(state="normal")
        self._lbox.delete("1.0","end")
        self._lbox.config(state="disabled")

    @staticmethod
    def _mkbtn(parent, text, bg, fg, cmd):
        return tk.Button(parent, text=text, font=(FNT,12,"bold"),
                         bg=bg, fg=fg, activebackground=bg,
                         relief="flat", cursor="hand2",
                         command=cmd, padx=12, pady=6)

    def _close(self):
        self._stop_monitor()
        if self.robot:
            try: self.robot.close()
            except: pass
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Continuum Manipulator GUI v2 — PCC + Kalman Control")
    ap.add_argument("--real",   action="store_true",
                    help="Connect to real hardware via RS-485")
    ap.add_argument("--port",   default="COM3",
                    help="Serial port (e.g. COM3, /dev/ttyUSB0, /dev/ttyACM0)")
    ap.add_argument("--baud",   default=115200, type=int,
                    help="Baud rate for RS-485 (default: 115200)")
    ap.add_argument("--list-ports", action="store_true",
                    help="List available serial ports and exit")
    args = ap.parse_args()

    if args.list_ports:
        if HAS_SERIAL:
            ports = serial.tools.list_ports.comports()
            print("Available serial ports:")
            for p in ports:
                print(f"  {p.device:20s}  {p.description}")
        else:
            print("pyserial not installed. Run: pip install pyserial")
        raise SystemExit(0)

    app = ContinuumGUI(
        simulated = not args.real,
        port      = args.port,
        baud      = args.baud,
    )
    app.protocol("WM_DELETE_WINDOW", app._close)
    app.mainloop()
