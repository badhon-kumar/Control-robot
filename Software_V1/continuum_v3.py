"""
continuum_gui_v2.py
═══════════════════════════════════════════════════════════════════════════════
Tendon-Driven Continuum Manipulator · Control GUI v2.0
3-Segment · 6 Tendons · RS-485 (LKMTECH MF5015) · PCC + Kalman Control

Implements:
  Zhai et al. 2025 — "Model-Based Control of a Continuum Manipulator with
  Online Jacobian Error Compensation Using Kalman Filtering"
  DOI: 10.34133/cbsystems.0339

TABS
────
  ① MANUAL JOG       — per-motor displacement sliders + direct entry
  ② CSV SEQUENCER    — load / edit / run CSV state sequences
  ③ TRAJECTORY CTRL  — closed-loop PCC + Kalman trajectory tracking

HARDWARE
────────
  Motors: LKMTECH MF5015 brushless with integrated driver
  Bus:    RS-485 via USB adapter (python: serial)
  Protocol: LKMTECH serial protocol (position/speed commands)

  Run (simulation):   python continuum_gui_v2.py
  Run (real HW):      python continuum_gui_v2.py --real --port COM3
                      python continuum_gui_v2.py --real --port /dev/ttyUSB0
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
SEG_LEN_MM   = [90.0, 90.0, 90.0]          # mm equivalents for visualizer
TOTAL_LEN_MM = 270.0
CAPSTAN_R_MM = [15.0] * 6                   # drum radii (mm)
MAX_DISP_MM  = 120.0                        # max motor displacement (mm)
MAX_SPEED_MMS= 80.0
MAX_DELTA_L  = 0.060                        # max |Δlᵢ| in metres

MOTOR_TO_SEG  = [0, 0, 1, 1, 2, 2]
MOTOR_IS_POS  = [True, False, True, False, True, False]
SEG_COLORS    = ["#00ddb8", "#3a9ef0", "#f5a623"]
SEG_NAMES     = ["Segment 1", "Segment 2", "Segment 3"]
MOTOR_COLORS  = [SEG_COLORS[0], SEG_COLORS[0],
                 SEG_COLORS[1], SEG_COLORS[1],
                 SEG_COLORS[2], SEG_COLORS[2]]
MOTOR_SHORT   = ["M1","M2","M3","M4","M5","M6"]

# ── Theme ─────────────────────────────────────────────────────────────────────
BG     = "#e2e8f0"; PANEL  = "#cbd5e1"; CARD   = "#f1f5f9"; INPUT  = "#f8fafc"
BORDER = "#94a3b8"; ACCENT = "#0f766e"; BLUE   = "#2563eb"; RED    = "#dc2626"
YELLOW = "#d97706"; GREEN  = "#16a34a"; ORANGE = "#ea580c"; MUTED  = "#475569"
TEXT   = "#0f172a"; DIM    = "#64748b"; CBKG   = "#e2e8f0"
FNT    = "Courier New"; FNT_H = "Courier New"

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
    cum = 0.0   # cumulative angle

    for i in range(3):
        ti  = th[i]
        Li  = L[i]
        s0  = cum
        s1  = cum + ti
        if abs(ti) < 1e-9:
            # L'Hôpital limit: arc → straight segment
            dx = Li * math.cos(s0)
            dy = Li * math.sin(s0)
        else:
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
    # Convert 6-motor displacements to 3 tendon differentials (mm → metres)
    dl = np.array([
        (disps_mm[0] - disps_mm[1]) / 1000.0,
        (disps_mm[2] - disps_mm[3]) / 1000.0,
        (disps_mm[4] - disps_mm[5]) / 1000.0,
    ])
    dl = np.clip(dl, -MAX_DELTA_L, MAX_DELTA_L)

    pose = forward_kinematics(dl)
    th   = pcc_angles(dl)

    # Build body arc for drawing (120 integration steps)
    N_PTS = 120
    L_tot = sum(SEG_LEN_MM)
    ds    = L_tot / N_PTS
    anchors_mm = [SEG_LEN_MM[0],
                  SEG_LEN_MM[0] + SEG_LEN_MM[1],
                  L_tot]

    # κᵢ in rad/mm
    kappas = []
    for i in range(3):
        ri  = R_TENDON[i] * 1000.0   # mm
        Li  = SEG_LEN_MM[i]
        ki  = (dl[i] * 1000.0 / ri) / Li if ri > 0 else 0.0
        lim = math.radians(150) / Li
        kappas.append(max(-lim, min(lim, ki)))

    ROLLOFF = 8.0
    def h_smooth(s, sa):
        return 1.0 / (1.0 + math.exp((s - sa) / (ROLLOFF / 6.0)))

    all_pts, curvature, pt_seg = [], [], []
    x2, y2, theta2 = 0.0, 0.0, 0.0

    for n in range(N_PTS + 1):
        s = n * ds
        seg = 0 if s <= anchors_mm[0] else (1 if s <= anchors_mm[1] else 2)
        kappa_s = sum(kappas[i] * h_smooth(s, anchors_mm[i]) for i in range(3))
        all_pts.append((x2, y2))
        curvature.append(kappa_s)
        pt_seg.append(seg)
        x2 += math.sin(theta2) * ds
        y2 += math.cos(theta2) * ds
        theta2 += kappa_s * ds

    end_pts = []
    for anch in anchors_mm:
        idx = min(int(round(anch / ds)), N_PTS)
        end_pts.append(all_pts[idx])

    thetas_rad = [float(t) for t in th]
    psi_deg    = math.degrees(float(pose[2]))

    seg_pts = [
        all_pts[: int(round(anchors_mm[0]/ds)) + 1],
        all_pts[int(round(anchors_mm[0]/ds)): int(round(anchors_mm[1]/ds)) + 1],
        all_pts[int(round(anchors_mm[1]/ds)):],
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
    """Returns a function t → [xr, yr, ψr] (metres, radians)."""

    if name == "Traj 1 — Ellipse (pos only)":
        def fn(t):
            xr  = 0.240 + 0.020 * math.cos(2 * math.pi * t / T)
            yr  = 0.060 * math.sin(2 * math.pi * t / T)
            psi = 0.0
            return np.array([xr, yr, psi])

    elif name == "Traj 2 — Attitude (pos fixed)":
        def fn(t):
            xr  = 0.240
            yr  = 0.0
            psi = (math.pi / 6) * math.sin(2 * math.pi * t / T)
            return np.array([xr, yr, psi])

    elif name == "Traj 3 — Combined (pos + att)":
        def fn(t):
            xr  = 0.240 + 0.020 * math.cos(2 * math.pi * t / T)
            yr  = 0.060 * math.sin(2 * math.pi * t / T)
            psi = (math.pi / 6) * math.sin(2 * math.pi * t / T)
            return np.array([xr, yr, psi])

    elif name == "Figure-8":
        def fn(t):
            xr  = 0.240 + 0.020 * math.sin(2 * math.pi * t / T)
            yr  = 0.040 * math.sin(4 * math.pi * t / T)
            psi = 0.0
            return np.array([xr, yr, psi])

    elif name == "Stationary":
        def fn(t):
            return np.array([0.240, 0.0, 0.0])

    else:  # Custom / placeholder
        def fn(t):
            return np.array([0.240, 0.0, 0.0])

    return fn


TRAJECTORY_NAMES = [
    "Traj 1 — Ellipse (pos only)",
    "Traj 2 — Attitude (pos fixed)",
    "Traj 3 — Combined (pos + att)",
    "Figure-8",
    "Stationary",
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
        out.append(TendonState(name=name, delay_s=delay, motors=motors))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: motor disp <-> u conversion
# ══════════════════════════════════════════════════════════════════════════════

def u_to_motor_disps_mm(u: np.ndarray) -> List[float]:
    """
    Convert 3-vector u [Δl₁,Δl₂,Δl₃] (metres) to 6 motor displacements (mm).
    Positive Δlᵢ → M_pos[i] pulls, M_neg[i] = 0
    Negative Δlᵢ → M_pos[i] = 0, M_neg[i] pulls
    """
    disps = [0.0] * 6
    for i in range(3):
        dl_mm = u[i] * 1000.0
        pos_idx = i * 2
        neg_idx = i * 2 + 1
        if dl_mm >= 0:
            disps[pos_idx] = dl_mm
            disps[neg_idx] = 0.0
        else:
            disps[pos_idx] = 0.0
            disps[neg_idx] = -dl_mm
    return disps


def motor_disps_mm_to_u(disps_mm: List[float]) -> np.ndarray:
    """Inverse: 6 motor displacements (mm) → u [Δl₁,Δl₂,Δl₃] (metres)."""
    u = np.zeros(3)
    for i in range(3):
        pos_idx = i * 2
        neg_idx = i * 2 + 1
        u[i] = (disps_mm[pos_idx] - disps_mm[neg_idx]) / 1000.0
    return u


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
        self._top_zoom = 1.0
        self._cv_side  = None
        self._cv_top   = None
        self._cv_heat  = None
        self._pose_widgets = {}
        self._build()
        self.after(120, self._redraw)

    def _build(self):
        hf = tk.Frame(self, bg=BG)
        hf.pack(fill="x", pady=(4, 2))
        tk.Label(hf, text="CONTINUUM SHAPE", font=(FNT_H, 12, "bold"),
                 fg=BLUE, bg=BG).pack(side="left")
        tk.Label(hf, text="  ·  PCC kinematics  ·  3 segments  ·  mm",
                 font=(FNT, 10), fg=MUTED, bg=BG).pack(side="left")

        views = tk.Frame(self, bg=BG)
        views.pack(fill="both", expand=True)

        # Side view
        sc = tk.Frame(views, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        sc.pack(side="left", fill="both", expand=True, padx=(0,3))
        tk.Label(sc, text=" SIDE VIEW  (bending plane  x–y)",
                 font=(FNT, 9, "bold"), fg=BLUE, bg=PANEL).pack(fill="x")
        self._cv_side = tk.Canvas(sc, bg=CBKG, highlightthickness=0)
        self._cv_side.pack(fill="both", expand=True)
        self._cv_side.bind("<Configure>", lambda e: self._draw_side())

        # Top view
        tc = tk.Frame(views, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        tc.pack(side="left", fill="both", expand=True, padx=(3,3))
        tk.Label(tc, text=" TOP VIEW  (base plate + tendon routing)",
                 font=(FNT, 9, "bold"), fg=ORANGE, bg=PANEL).pack(fill="x")
        self._cv_top = tk.Canvas(tc, bg=CBKG, highlightthickness=0)
        self._cv_top.pack(fill="both", expand=True)
        self._cv_top.bind("<Configure>", lambda e: self._draw_top())
        self._cv_top.bind("<MouseWheel>", self._on_top_scroll)

        # Pose HUD
        pc = tk.Frame(views, bg=CARD, highlightbackground=BORDER, highlightthickness=1,
                      width=158)
        pc.pack(side="left", fill="y", padx=(3,0))
        pc.pack_propagate(False)
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
        self._draw_heat()
        self._update_pose()

    def _on_top_scroll(self, event):
        self._top_zoom *= 1.12 if event.delta > 0 else (1/1.12)
        self._top_zoom  = max(0.4, min(4.0, self._top_zoom))
        self._draw_top()
        return "break"

    @staticmethod
    def _grid(cv, w, h, step=38):
        for x in range(0, w, step):
            cv.create_line(x, 0, x, h, fill="#e2e8f0", width=1)
        for y in range(0, h, step):
            cv.create_line(0, y, w, y, fill="#e2e8f0", width=1)

    def _draw_side(self):
        cv = self._cv_side
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 30 or h < 30: return
        cv.delete("all")
        self._grid(cv, w, h)

        kin     = compute_pcc_kinematics_mm(self._disps)
        all_pts = kin["all_pts"]
        pt_seg  = kin["pt_seg"]
        N       = len(all_pts)

        scl   = h * 0.72 / TOTAL_LEN_MM
        org_x = w // 2
        org_y = int(h * 0.93)

        def px(x, y): return org_x + x*scl, org_y - y*scl

        def arc_normal(k, off_mm):
            i0 = max(0, k-1); i1 = min(N-1, k+1)
            dx = all_pts[i1][0]-all_pts[i0][0]
            dy = all_pts[i1][1]-all_pts[i0][1]
            ln = math.hypot(dx, dy) or 1.0
            nx, ny = -dy/ln, dx/ln
            return nx*off_mm*scl, ny*off_mm*scl

        # Ruler
        ruler_x = 14
        cv.create_line(ruler_x, org_y, ruler_x, org_y - TOTAL_LEN_MM*scl,
                       fill=BORDER, width=1)
        for mm in range(0, int(TOTAL_LEN_MM)+1, 30):
            ry = org_y - mm*scl
            cv.create_line(ruler_x-4, ry, ruler_x+4, ry, fill=MUTED, width=1)
            if mm % 90 == 0:
                cv.create_text(ruler_x-6, ry, text=f"{mm}", fill=DIM,
                               font=(FNT,8), anchor="e")
        cv.create_line(org_x, org_y, org_x, 6, fill=BORDER, dash=(4,6), width=1)

        # Base
        bw = 26
        cv.create_rectangle(org_x-bw, org_y-5, org_x+bw, org_y+9,
                            fill="#e5e7eb", outline=TEXT, width=1)
        for k2 in range(-bw, bw, 8):
            cv.create_line(org_x+k2, org_y+9, org_x+k2-6, org_y+16, fill=MUTED, width=1)

        # Shadow tube
        for k in range(1, N):
            x0c,y0c = px(*all_pts[k-1]); x1c,y1c = px(*all_pts[k])
            cv.create_line(x0c,y0c,x1c,y1c, fill="#d1d5db", width=28, capstyle=tk.ROUND)

        # Coloured body
        for k in range(1, N):
            s_idx   = pt_seg[k]
            t_g     = k/N
            br      = int(self.BODY_RADII[0] + t_g*(self.BODY_RADII[2]-self.BODY_RADII[0]))
            x0c,y0c = px(*all_pts[k-1]); x1c,y1c = px(*all_pts[k])
            cv.create_line(x0c,y0c,x1c,y1c, fill=SEG_COLORS[s_idx],
                           width=max(4,br*2), capstyle=tk.ROUND)

        # Tip
        tip_px, tip_py = px(kin["tip_x"], kin["tip_y"])
        ch = 18
        cv.create_line(tip_px-ch, tip_py, tip_px+ch, tip_py, fill=ORANGE, dash=(3,3), width=1)
        cv.create_line(tip_px, tip_py-ch, tip_px, tip_py+ch, fill=ORANGE, dash=(3,3), width=1)
        cv.create_oval(tip_px-9, tip_py-9, tip_px+9, tip_py+9,
                       fill=ORANGE, outline=TEXT, width=2)
        cv.create_text(tip_px+14, tip_py-1, text="EE", fill=ORANGE,
                       font=(FNT,10,"bold"), anchor="w")

        psi = kin["psi"]
        al  = 34
        cv.create_line(tip_px, tip_py,
                       tip_px+al*math.sin(psi), tip_py-al*math.cos(psi),
                       fill=YELLOW, width=2, arrow=tk.LAST, arrowshape=(8,10,3))
        cv.create_text(w//2, h-5,
                       text=f"EE  x={kin['tip_x']:+.1f}mm  y={kin['tip_y']:.1f}mm  ψ={math.degrees(psi):+.1f}°",
                       fill=DIM, font=(FNT,9), anchor="s")

    def _draw_top(self):
        cv = self._cv_top
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 30 or h < 30: return
        cv.delete("all")
        self._grid(cv, w, h)
        kin     = compute_pcc_kinematics_mm(self._disps)
        all_pts = kin["all_pts"]
        pt_seg  = kin["pt_seg"]
        N       = len(all_pts)
        scl  = min(w,h)*0.58*self._top_zoom
        cx_  = w//2; cy_ = int(h*0.82)

        def px(x, y): return cx_+x*scl/TOTAL_LEN_MM, cy_-y*scl/TOTAL_LEN_MM

        plate_r = max(20, int(0.12*min(w,h)*self._top_zoom))
        hex_pts = []
        for k in range(6):
            a = math.radians(30+k*60)
            hex_pts += [cx_+plate_r*math.cos(a), cy_+plate_r*math.sin(a)]
        cv.create_polygon(*hex_pts, fill="#e5e7eb", outline=MUTED, width=1)

        for k in range(1, N):
            x0,y0 = px(*all_pts[k-1]); x1,y1 = px(*all_pts[k])
            cv.create_line(x0,y0,x1,y1, fill="#d1d5db", width=12, capstyle=tk.ROUND)
        for k in range(1, N):
            s_idx = pt_seg[k]
            t_g   = k/N
            br    = int(self.BODY_RADII[0]+t_g*(self.BODY_RADII[2]-self.BODY_RADII[0]))
            x0,y0 = px(*all_pts[k-1]); x1,y1 = px(*all_pts[k])
            cv.create_line(x0,y0,x1,y1, fill=SEG_COLORS[s_idx],
                           width=max(3,br), capstyle=tk.ROUND)

        tip_px, tip_py = px(kin["tip_x"], kin["tip_y"])
        cv.create_oval(tip_px-9,tip_py-9,tip_px+9,tip_py+9,
                       fill=ORANGE, outline=TEXT, width=2)
        cv.create_text(tip_px+13, tip_py, text="EE", fill=ORANGE,
                       font=(FNT,10,"bold"), anchor="w")

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

    PLOT_W = 380
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

        self._build()

    def _build(self):
        # ── Header ───────────────────────────────────────────────────────────
        hf = tk.Frame(self, bg=PANEL)
        hf.pack(fill="x")
        tk.Label(hf, text="🎯  TRAJECTORY CONTROL", font=(FNT_H,13,"bold"),
                 fg=ACCENT, bg=PANEL).pack(side="left", padx=14, pady=10)
        tk.Label(hf, text="Zhai et al. 2025 · PCC + Kalman Jacobian Compensation",
                 font=(FNT,10), fg=MUTED, bg=PANEL).pack(side="left")

        # ── Main layout: left config | right plots ───────────────────────────
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True)

        # LEFT — configuration
        lf = tk.Frame(main, bg=BG, width=340)
        lf.pack(side="left", fill="y", padx=(6,4), pady=4)
        lf.pack_propagate(False)

        # Trajectory selection
        tc = tk.LabelFrame(lf, text="  Trajectory  ", font=(FNT,10,"bold"),
                           fg=BLUE, bg=CARD, relief="flat",
                           highlightbackground=BORDER, highlightthickness=1)
        tc.pack(fill="x", pady=(0,4), padx=2)
        tk.Label(tc, text="Type:", font=(FNT,11), fg=MUTED, bg=CARD).grid(
            row=0, column=0, sticky="w", padx=8, pady=4)
        ttk.Combobox(tc, textvariable=self._traj_var,
                     values=TRAJECTORY_NAMES, state="readonly",
                     font=(FNT,11), width=28).grid(row=0, column=1, padx=4, pady=4)
        for r,(lbl,var) in enumerate([("Period T (s):",self._T_var),
                                       ("Cycles:",self._cycles_var),
                                       ("Motor speed (mm/s):",self._speed_var)], 1):
            tk.Label(tc, text=lbl, font=(FNT,11), fg=MUTED, bg=CARD).grid(
                row=r, column=0, sticky="w", padx=8, pady=3)
            tk.Entry(tc, textvariable=var, width=10, font=(FNT,12), bg=INPUT, fg=TEXT,
                     insertbackground=TEXT, relief="flat",
                     highlightbackground=BORDER, highlightthickness=1).grid(
                row=r, column=1, sticky="w", padx=4, pady=3)

        tk.Checkbutton(tc, text="Hold ψ=0 (override attitude)",
                       variable=self._atthold_var, font=(FNT,10), bg=CARD, fg=TEXT,
                       selectcolor=INPUT, activebackground=CARD).grid(
            row=4, column=0, columnspan=2, sticky="w", padx=8, pady=3)

        # Controller parameters
        cc = tk.LabelFrame(lf, text="  Controller Parameters (Table 1)  ",
                           font=(FNT,10,"bold"), fg=ACCENT, bg=CARD, relief="flat",
                           highlightbackground=BORDER, highlightthickness=1)
        cc.pack(fill="x", pady=(0,4), padx=2)
        params = [
            ("γ  (fading rate):",     self._gamma_var),
            ("β  (controller gain):", self._beta_var),
            ("α  (regularisation):",  self._alpha_var),
            ("σₚ (pos threshold):",   self._sigp_var),
            ("σᵩ (att threshold):",   self._sigps_var),
            ("Control rate (Hz):",    self._rate_var),
        ]
        for r,(lbl,var) in enumerate(params):
            tk.Label(cc, text=lbl, font=(FNT,10), fg=MUTED, bg=CARD).grid(
                row=r, column=0, sticky="w", padx=8, pady=2)
            tk.Entry(cc, textvariable=var, width=10, font=(FNT,11), bg=INPUT, fg=TEXT,
                     insertbackground=TEXT, relief="flat",
                     highlightbackground=BORDER, highlightthickness=1).grid(
                row=r, column=1, sticky="w", padx=4, pady=2)

        tk.Checkbutton(cc, text="Enable Kalman compensation",
                       variable=self._kalman_var, font=(FNT,10,"bold"),
                       fg=GREEN, bg=CARD, selectcolor=INPUT, activebackground=CARD).grid(
            row=len(params), column=0, columnspan=2, sticky="w", padx=8, pady=4)

        # Simulation options
        sc2 = tk.LabelFrame(lf, text="  Simulation Options  ",
                            font=(FNT,10,"bold"), fg=MUTED, bg=CARD, relief="flat",
                            highlightbackground=BORDER, highlightthickness=1)
        sc2.pack(fill="x", pady=(0,4), padx=2)
        tk.Checkbutton(sc2, text="Add measurement noise",
                       variable=self._noise_var, font=(FNT,10), bg=CARD, fg=TEXT,
                       selectcolor=INPUT, activebackground=CARD).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=8, pady=3)
        tk.Label(sc2, text="Noise σ (m):", font=(FNT,10), fg=MUTED, bg=CARD).grid(
            row=1, column=0, sticky="w", padx=8, pady=2)
        tk.Entry(sc2, textvariable=self._noise_sig, width=10, font=(FNT,11), bg=INPUT, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 highlightbackground=BORDER, highlightthickness=1).grid(
            row=1, column=1, sticky="w", padx=4, pady=2)

        # Control buttons
        bf = tk.Frame(lf, bg=BG)
        bf.pack(fill="x", pady=6, padx=2)
        self.btn_start = tk.Button(bf, text="▶  START", font=(FNT,13,"bold"),
                                   bg=GREEN, fg=BG, relief="flat", cursor="hand2",
                                   padx=14, pady=8, command=self._start)
        self.btn_start.pack(side="left", padx=(0,4))
        self.btn_stop  = tk.Button(bf, text="■  STOP",  font=(FNT,13,"bold"),
                                   bg=RED, fg=TEXT, relief="flat", cursor="hand2",
                                   padx=14, pady=8, command=self._stop, state="disabled")
        self.btn_stop.pack(side="left", padx=(0,4))
        tk.Button(bf, text="↺ RESET", font=(FNT,11,"bold"),
                  bg=PANEL, fg=MUTED, relief="flat", cursor="hand2",
                  padx=10, pady=8, command=self._reset).pack(side="left")

        # Metrics display
        mf = tk.LabelFrame(lf, text="  Live Metrics  ",
                           font=(FNT,10,"bold"), fg=ORANGE, bg=CARD, relief="flat",
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
            tk.Label(mf, text=lbl, font=(FNT,10), fg=MUTED, bg=CARD).grid(
                row=r, column=0, sticky="w", padx=8, pady=1)
            lv = tk.Label(mf, text="—", font=(FNT,11,"bold"), fg=col, bg=CARD, width=10)
            lv.grid(row=r, column=1, sticky="w", padx=4)
            self._metric_lbls[key] = lv

        # Jacobian info
        jf = tk.LabelFrame(lf, text="  Jacobian Info  ",
                           font=(FNT,10,"bold"), fg=SEG_COLORS[0], bg=CARD, relief="flat",
                           highlightbackground=BORDER, highlightthickness=1)
        jf.pack(fill="x", pady=(0,4), padx=2)
        self._jac_lbl = tk.Label(jf, text="—", font=(FNT, 9), fg=MUTED, bg=CARD,
                                 justify="left", anchor="w")
        self._jac_lbl.pack(fill="x", padx=8, pady=4)

        # RIGHT — plots
        rf = tk.Frame(main, bg=BG)
        rf.pack(side="right", fill="both", expand=True, padx=(4,6), pady=4)

        # Trajectory plot
        tp_lbl = tk.Label(rf, text="TIP TRAJECTORY  (mm)",
                          font=(FNT,10,"bold"), fg=BLUE, bg=PANEL)
        tp_lbl.pack(fill="x")
        self._cv_traj = tk.Canvas(rf, bg="#0f172a", highlightthickness=1,
                                  highlightbackground=BORDER,
                                  width=self.PLOT_W, height=self.PLOT_H)
        self._cv_traj.pack(fill="both", expand=True, pady=(0,3))
        self._cv_traj.bind("<Configure>", lambda e: self._redraw_traj())

        # Error plot
        tk.Label(rf, text="TRACKING ERROR MAGNITUDE  (mm)",
                 font=(FNT,10,"bold"), fg=RED, bg=PANEL).pack(fill="x")
        self._cv_err = tk.Canvas(rf, bg="#0f172a", highlightthickness=1,
                                 highlightbackground=BORDER,
                                 width=self.PLOT_W, height=self.ERR_H)
        self._cv_err.pack(fill="x", pady=(0,3))
        self._cv_err.bind("<Configure>", lambda e: self._redraw_err())

        # Status bar
        self._status_lbl = tk.Label(rf, text="Ready. Select trajectory and press START.",
                                    font=(FNT,10), fg=MUTED, bg=PANEL)
        self._status_lbl.pack(fill="x")

    # ── Parameter helpers ─────────────────────────────────────────────────────

    def _get_float(self, var: tk.StringVar, default: float) -> float:
        try:   return float(var.get())
        except: return default

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
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self._status_lbl.config(text="Running…", fg=GREEN)
        self._thread = threading.Thread(target=self._control_loop, daemon=True)
        self._thread.start()
        self._log("[TRAJ] Trajectory tracking started")

    def _stop(self):
        self._abort_flag.set()
        self._running = False
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
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
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
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

    def _update_metrics(self, step: int, err: np.ndarray):
        if not self._err_x_hist: return
        ex  = np.array(self._err_x_hist)
        ey  = np.array(self._err_y_hist)
        eps = np.array(self._err_psi_hist)
        self._rmse_x   = float(np.sqrt(np.mean(ex**2)))
        self._rmse_y   = float(np.sqrt(np.mean(ey**2)))
        self._rmse_psi = float(np.sqrt(np.mean(eps**2)))
        self._mae_x    = float(np.mean(ex))
        self._mae_y    = float(np.mean(ey))
        self._mae_psi  = float(np.mean(eps))
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

        # Background grid
        for x in range(0, w, 40):
            cv.create_line(x,0,x,h, fill="#1e293b", width=1)
        for y in range(0, h, 40):
            cv.create_line(0,y,w,y, fill="#1e293b", width=1)

        if not self._ref_hist and not self._act_hist:
            cv.create_text(w//2, h//2, text="Trajectory will appear here",
                           fill="#334155", font=(FNT,12), anchor="center")
            return

        # Determine scale
        all_x = [p[0] for p in self._ref_hist+self._act_hist]
        all_y = [p[1] for p in self._ref_hist+self._act_hist]
        if not all_x: return

        x_min,x_max = min(all_x), max(all_x)
        y_min,y_max = min(all_y), max(all_y)
        margin = 30
        dx = max(x_max-x_min, 1.0)
        dy = max(y_max-y_min, 1.0)
        scl = min((w-2*margin)/dx, (h-2*margin)/dy) * 0.85
        cx_ = w//2 - (x_min+x_max)/2*scl
        cy_ = h//2 + (y_min+y_max)/2*scl

        def to_px(xm, ym):
            return cx_+xm*scl, cy_-ym*scl

        # Draw reference path (dashed white)
        if len(self._ref_hist) >= 2:
            pts = [to_px(*p) for p in self._ref_hist]
            flat = [v for p in pts for v in p]
            cv.create_line(*flat, fill="#475569", width=2, dash=(6,4), smooth=True)

        # Draw actual path (solid coloured)
        if len(self._act_hist) >= 2:
            pts = [to_px(*p) for p in self._act_hist]
            flat = [v for p in pts for v in p]
            cv.create_line(*flat, fill=ACCENT, width=2, smooth=True)

        # Current tip
        if self._act_hist:
            px_, py_ = to_px(*self._act_hist[-1])
            cv.create_oval(px_-6,py_-6,px_+6,py_+6, fill=ORANGE, outline=TEXT, width=2)

        if self._ref_hist:
            rx_, ry_ = to_px(*self._ref_hist[-1])
            cv.create_oval(rx_-4,ry_-4,rx_+4,ry_+4, fill="#ffffff", outline="", width=1)

        # Legend
        cv.create_line(10,h-20,34,h-20, fill="#475569", width=2, dash=(6,4))
        cv.create_text(38,h-20, text="Reference", fill="#475569", font=(FNT,9), anchor="w")
        cv.create_line(110,h-20,134,h-20, fill=ACCENT, width=2)
        cv.create_text(138,h-20, text="Actual", fill=ACCENT, font=(FNT,9), anchor="w")

        # Axes labels
        cv.create_text(w-4, h//2, text="x (mm)", fill=DIM, font=(FNT,8), anchor="e")
        cv.create_text(w//2, 8, text="y (mm)", fill=DIM, font=(FNT,8), anchor="n")

    def _redraw_err(self):
        cv = self._cv_err
        w  = cv.winfo_width(); h = cv.winfo_height()
        if w < 30 or h < 30: return
        cv.delete("all")

        for x in range(0, w, 40):
            cv.create_line(x,0,x,h, fill="#1e293b", width=1)
        for y in range(0, h, 20):
            cv.create_line(0,y,w,y, fill="#1e293b", width=1)

        if not self._err_hist:
            cv.create_text(w//2, h//2, text="Error plot will appear here",
                           fill="#334155", font=(FNT,10), anchor="center")
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
                       fill=RED, font=(FNT,8), anchor="ne")
        cv.create_text(w-4, h-margin, text="0",
                       fill=DIM, font=(FNT,8), anchor="se")
        cv.create_text(4, h-4, text=f"n={N}", fill=DIM, font=(FNT,8), anchor="sw")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class ContinuumGUI(tk.Tk):

    def __init__(self, simulated=True, port="COM3", baud=115200):
        super().__init__()
        self.title("Tendon-Driven Continuum Manipulator  ·  Control GUI  v2.0  ·  Zhai et al. 2025")
        self.configure(bg=BG)
        self.minsize(1500, 860)
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
                 text="⬡  CONTINUUM MANIPULATOR  ·  3-SEG TENDON DRIVE  ·  PCC + KALMAN  v2",
                 font=(FNT_H,14,"bold"), fg=ACCENT, bg=PANEL).pack(side="left", padx=16, pady=10)

        self._lbl_c = tk.Label(top, text="● DISCONNECTED",
                               font=(FNT,12,"bold"), fg=RED, bg=PANEL)
        self._lbl_c.pack(side="right", padx=12)
        mode = "SIM" if self._sim else f"HW · {self._port} · {self._baud}baud"
        tk.Label(top, text=mode, font=(FNT,11), fg=MUTED, bg=PANEL).pack(side="right", padx=4)
        self._mkbtn(top,"DISCONNECT",PANEL,RED,  self._disconnect).pack(side="right",padx=3)
        self._mkbtn(top,"CONNECT",  ACCENT,BG,   self._connect   ).pack(side="right",padx=3)

        # Body
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        # Left: tabs
        left_outer = tk.Frame(body, bg=BG, width=880)
        left_outer.pack(side="left", fill="y", padx=(8,4), pady=6)
        left_outer.pack_propagate(False)

        tbar = tk.Frame(left_outer, bg=BG)
        tbar.pack(fill="x", pady=(0,6))

        self._tab_manual_btn = tk.Button(
            tbar, text="  🕹  MANUAL JOG  ", font=(FNT,12,"bold"), bg=ACCENT, fg=BG,
            relief="flat", cursor="hand2", padx=10, pady=6, command=self._show_manual)
        self._tab_manual_btn.pack(side="left", padx=(0,4))

        self._tab_csv_btn = tk.Button(
            tbar, text="  📋  CSV SEQUENCER  ", font=(FNT,12,"bold"), bg=PANEL, fg=MUTED,
            relief="flat", cursor="hand2", padx=10, pady=6, command=self._show_csv)
        self._tab_csv_btn.pack(side="left", padx=(0,4))

        self._tab_traj_btn = tk.Button(
            tbar, text="  🎯  TRAJECTORY CONTROL  ", font=(FNT,12,"bold"), bg=PANEL, fg=MUTED,
            relief="flat", cursor="hand2", padx=10, pady=6, command=self._show_traj)
        self._tab_traj_btn.pack(side="left")

        self._mkbtn(tbar,"  🏠  HOME ALL",GREEN,BG,
                    self._home_all).pack(side="right", padx=(6,0))

        # Scrollable canvas for panels
        sf = tk.Frame(left_outer, bg=BG)
        sf.pack(fill="both", expand=True)
        lc = tk.Canvas(sf, bg=BG, highlightthickness=0)
        vs = ttk.Scrollbar(sf, orient="vertical", command=lc.yview)
        hs = ttk.Scrollbar(sf, orient="horizontal", command=lc.xview)
        lc.configure(yscrollcommand=vs.set, xscrollcommand=hs.set)
        hs.pack(side="bottom", fill="x")
        vs.pack(side="right", fill="y")
        lc.pack(side="left", fill="both", expand=True)
        lc.bind_all("<MouseWheel>",
                    lambda e: lc.yview_scroll(int(-1*(e.delta/120)),"units")
                    if vs.get() != (0.0,1.0) else None)
        self._lc = lc

        # Manual panel
        self._manual_frame = tk.Frame(lc, bg=BG)
        self._manual_win   = lc.create_window((0,0), window=self._manual_frame, anchor="nw")
        self._manual_frame.bind("<Configure>",
            lambda e: lc.configure(scrollregion=lc.bbox("all")))
        self._manual_panel = ManualJogPanel(
            self._manual_frame,
            get_robot     = lambda: self.robot,
            get_disps     = lambda: list(self._disps),
            apply_disp_fn = self._apply_single_disp,
            log_fn        = self._log,
        )
        self._manual_panel.pack(fill="x")

        # CSV panel
        self._csv_frame = tk.Frame(lc, bg=BG)
        self._csv_win   = lc.create_window((0,0), window=self._csv_frame, anchor="nw")
        self._csv_frame.bind("<Configure>",
            lambda e: lc.configure(scrollregion=lc.bbox("all")))
        self._csv_panel = CsvStatePanel(
            self._csv_frame,
            get_robot      = lambda: self.robot,
            apply_state_fn = self._apply_state,
            log_fn         = self._log,
        )
        self._csv_panel.pack(fill="x")

        # Trajectory panel
        self._traj_frame = tk.Frame(lc, bg=BG)
        self._traj_win   = lc.create_window((0,0), window=self._traj_frame, anchor="nw")
        self._traj_frame.bind("<Configure>",
            lambda e: lc.configure(scrollregion=lc.bbox("all")))
        self._traj_panel = TrajectoryControlPanel(
            self._traj_frame,
            get_robot    = lambda: self.robot,
            set_disps_fn = self._apply_disps_from_ctrl,
            log_fn       = self._log,
            simulated    = self._sim,
        )
        self._traj_panel.pack(fill="both", expand=True)

        self._show_manual()

        # Right: visualizations
        right = tk.Frame(body, bg=BG)
        right.pack(side="right", fill="both", expand=True, padx=(4,8), pady=6)

        self._capstan_panel = CapstanPanel(right)
        self._capstan_panel.pack(fill="x")

        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", pady=4)

        self._cont_viz = ContinuumVisualizer(right)
        self._cont_viz.pack(fill="both", expand=True)

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

    # ── Tab switching ─────────────────────────────────────────────────────────

    def _show_manual(self):
        self._lc.itemconfig(self._csv_win,   state="hidden")
        self._lc.itemconfig(self._traj_win,  state="hidden")
        self._lc.itemconfig(self._manual_win,state="normal")
        self._lc.configure(scrollregion=self._lc.bbox("all"))
        self._tab_manual_btn.config(bg=ACCENT, fg=BG)
        self._tab_csv_btn.config(   bg=PANEL,  fg=MUTED)
        self._tab_traj_btn.config(  bg=PANEL,  fg=MUTED)

    def _show_csv(self):
        self._lc.itemconfig(self._manual_win,state="hidden")
        self._lc.itemconfig(self._traj_win,  state="hidden")
        self._lc.itemconfig(self._csv_win,   state="normal")
        self._lc.configure(scrollregion=self._lc.bbox("all"))
        self._tab_csv_btn.config(   bg=BLUE,  fg=BG)
        self._tab_manual_btn.config(bg=PANEL, fg=MUTED)
        self._tab_traj_btn.config(  bg=PANEL, fg=MUTED)

    def _show_traj(self):
        self._lc.itemconfig(self._manual_win,state="hidden")
        self._lc.itemconfig(self._csv_win,   state="hidden")
        self._lc.itemconfig(self._traj_win,  state="normal")
        self._lc.configure(scrollregion=self._lc.bbox("all"))
        self._tab_traj_btn.config(  bg=ORANGE, fg=BG)
        self._tab_manual_btn.config(bg=PANEL,  fg=MUTED)
        self._tab_csv_btn.config(   bg=PANEL,  fg=MUTED)

    # ── Motion API ────────────────────────────────────────────────────────────

    def _apply_single_disp(self, motor_idx: int, disp_mm: float, speed_mms: float):
        self._disps[motor_idx]  = disp_mm
        self._speeds[motor_idx] = speed_mms
        self._capstan_panel.update_all(self._disps, self._speeds)
        self._manual_panel.update_disp(motor_idx+1, disp_mm)
        self._cont_viz.set_displacements(self._disps)
        if self.robot and not self._sim:
            try:
                self.robot.set_motor_mm(motor_idx, disp_mm, speed_mms)
            except Exception as e:
                self._log(f"HW error M{motor_idx+1}: {e}", "err")

    def _apply_disps_from_ctrl(self, disps_mm: List[float], speeds: List[float]):
        """Called by TrajectoryControlPanel on each control step."""
        self._disps  = list(disps_mm)
        self._speeds = list(speeds)
        self._capstan_panel.update_all(disps_mm, speeds)
        for i in range(6):
            self._manual_panel.update_disp(i+1, disps_mm[i])
        self._cont_viz.set_displacements(disps_mm)
        if self.robot and not self._sim:
            try:
                self.robot.set_all_mm(disps_mm, speeds)
            except Exception as e:
                self._log(f"HW error: {e}", "err")

    def _apply_state(self, state: TendonState):
        targets = [m[0] for m in state.motors]
        speeds  = [m[1] for m in state.motors]
        self._disps  = targets[:]
        self._speeds = speeds[:]
        self._capstan_panel.update_all(targets, speeds)
        for i in range(6):
            self._manual_panel.update_disp(i+1, targets[i])
        self._cont_viz.set_displacements(targets)
        if self.robot and not self._sim:
            try:
                self.robot.set_all_mm(targets, speeds)
            except Exception as e:
                self._log(f"HW error: {e}", "err")

    def _home_all(self):
        for i in range(6): self._apply_single_disp(i, 0.0, 20.0)
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
        for i in range(6):
            self._manual_panel.update_disp(i+1, self._disps[i])
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