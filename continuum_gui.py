"""
continuum_gui.py
================
Tendon-Driven Continuum Manipulator — Control GUI
MyActuator RMD-X8-120  ·  6 Capstan Motors  ·  RS-485 / CAN

ARCHITECTURE (from image)
──────────────────────────
6 capstan motors are mounted on a fixed actuation plate.
Each motor winds/unwinds an independent tendon cable that threads through
a continuum (flexible) manipulator body.  The motors do NOT move rigid links.

Motor → Capstan drum → Tendon cable → Continuum body bending/elongation

CONTROL UNITS
─────────────
  Input  : displacement (mm) and speed (mm/s) per tendon
  Internal: converted to angle (°) and angular speed (°/s) via:
            angle_deg     = displacement_mm / CAPSTAN_RADII_MM[i] * (180/π)
            ang_speed_dps = speed_mms       / CAPSTAN_RADII_MM[i] * (180/π)
            Each motor has an independently configurable capstan radius.

TWO CONTROL MODES
─────────────────
  ① CSV STATE MACHINE  — load a CSV, each row = one state (named),
                          12 parameters per row (disp + speed per motor),
                          delay between states, run automatically.

  ② MANUAL JOG         — per-motor displacement + speed entry,
                          + / − continuous jog buttons,
                          step jog, direct SEND.

VISUALIZATION  (unique to this architecture)
─────────────────────────────────────────────
  LEFT  — 6 capstan drum gauges with real-time displacement bars
  RIGHT — Continuum body shape estimated from tendon lengths (2D bending)
          + per-tendon force/tension heatmap
          + top-view and side-view of estimated continuum shape

Run:
    python3.10 continuum_gui.py             ← simulation
    python3.10 continuum_gui.py --real --port /dev/ttyACM0
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import time
import logging
import queue
import csv
import os
import math
import argparse
from dataclasses import dataclass, field
from typing import Optional

# ── Optional robot hardware imports ──────────────────────────────────────────
try:
    from can_interface    import CANBus
    from robot_controller import RobotController, DEFAULT_JOINT_CONFIG
    HAS_ROBOT = True
except ImportError:
    HAS_ROBOT = False

# ══════════════════════════════════════════════════════════════════════════════
# CAPSTAN PHYSICS
# ══════════════════════════════════════════════════════════════════════════════

CAPSTAN_RADII_MM     = [15.0, 15.0, 15.0, 15.0, 15.0, 15.0]  # per-motor drum radii (mm)
CAPSTAN_MAX_DISP_MM  = 120.0         # max tendon pull per motor (mm)
CAPSTAN_MAX_SPEED_MMS = 80.0         # max tendon speed (mm/s)

# Tendon layout on the actuation plate (relative positions for visualization)
# Each tendon connects to a different point around the continuum body cross-section
# Angles in degrees, radius from centre of the plate
TENDON_ANGLES_DEG = [210, 270, 330, 30, 90, 150]  # 6 tendons, evenly spaced + offset
TENDON_COLORS     = ["#00ddb8","#3a9ef0","#f0c030","#e87830","#c070f8","#60d870"]
TENDON_NAMES      = ["T1","T2","T3","T4","T5","T6"]
MOTOR_NAMES       = ["M1 (Bot-Left)","M2 (Bot-Ctr)","M3 (Bot-Right)",
                     "M4 (Top-Right)","M5 (Top-Ctr)","M6 (Top-Left)"]


def mm_to_deg(mm: float, motor_idx: int = 0) -> float:
    """Convert linear tendon displacement (mm) to capstan angle (degrees)."""
    return (mm / CAPSTAN_RADII_MM[motor_idx]) * (180.0 / math.pi)


def deg_to_mm(deg: float, motor_idx: int = 0) -> float:
    """Convert capstan angle (degrees) to tendon displacement (mm)."""
    return deg * CAPSTAN_RADII_MM[motor_idx] * (math.pi / 180.0)


def mms_to_dps(mms: float, motor_idx: int = 0) -> float:
    """Convert tendon speed (mm/s) to angular speed (°/s)."""
    return (mms / CAPSTAN_RADII_MM[motor_idx]) * (180.0 / math.pi)


def estimate_continuum_shape(displacements_mm: list) -> list:
    """
    Estimate the 2D continuum body shape from tendon displacements.

    Uses a simplified piecewise constant curvature (PCC) model:
    - Each pair of opposing tendons produces a bending moment
    - Net bending angle proportional to differential tendon tension
    - Body length proportional to average tendon displacement

    Returns a list of (x, y) points along the body centreline (mm),
    starting from origin (0,0) pointing in +Y direction.
    """
    N_SEGMENTS = 20
    BODY_REST_LENGTH = 200.0   # rest length of continuum body (mm)

    d = displacements_mm

    # Opposing tendon pairs produce bending in 2 planes:
    # Pair 1-4 (T1 vs T4): bending in XZ plane → side bend
    # Pair 2-5 (T2 vs T5): bending in YZ plane → forward/back
    # Pair 3-6 (T3 vs T6): secondary bending
    diff_x  = (d[0] - d[3]) * 0.5   # T1 vs T4
    diff_y  = (d[1] - d[4]) * 0.5   # T2 vs T5
    diff_x2 = (d[2] - d[5]) * 0.25  # T3 vs T6 (smaller contribution)

    bx = (diff_x + diff_x2) / CAPSTAN_MAX_DISP_MM   # normalised bending -1..1
    by = diff_y / CAPSTAN_MAX_DISP_MM

    # Average displacement affects total body length (tendons pulling = shortening)
    avg_pull = sum(max(0, di) for di in d) / len(d)
    body_len = BODY_REST_LENGTH * (1.0 - avg_pull / CAPSTAN_MAX_DISP_MM * 0.3)
    body_len = max(body_len, BODY_REST_LENGTH * 0.5)

    # Total bending angle (radians)
    kappa_x = bx * math.pi * 0.8   # max ±144° bend
    kappa_y = by * math.pi * 0.8

    seg_len = body_len / N_SEGMENTS
    pts_side = [(0.0, 0.0)]   # XZ plane (side view: x=lateral, y=along body)
    pts_top  = [(0.0, 0.0)]   # XY plane (top view:  x=lateral, y=forward)

    theta_s = 0.0   # cumulative angle in side view
    theta_t = 0.0   # cumulative angle in top view

    for s in range(N_SEGMENTS):
        t_norm = s / N_SEGMENTS
        # Smoothly interpolate angle along body
        theta_s += kappa_x / N_SEGMENTS
        theta_t += kappa_y / N_SEGMENTS

        dx_s = seg_len * math.sin(theta_s)
        dy_s = seg_len * math.cos(theta_s)
        dx_t = seg_len * math.sin(theta_t)
        dy_t = seg_len * math.cos(theta_t)

        pts_side.append((pts_side[-1][0] + dx_s, pts_side[-1][1] + dy_s))
        pts_top.append((pts_top[-1][0]  + dx_t, pts_top[-1][1]  + dy_t))

    return pts_side, pts_top, body_len


# ══════════════════════════════════════════════════════════════════════════════
# THEME
# ══════════════════════════════════════════════════════════════════════════════

BG        = "#07090e"
PANEL     = "#0d1420"
CARD      = "#121b28"
INPUT     = "#182030"
BORDER    = "#1c2d44"
ACCENT    = "#00ddb8"
BLUE      = "#3a9ef0"
RED       = "#f04040"
YELLOW    = "#f0c030"
GREEN     = "#38c050"
ORANGE    = "#e87830"
MUTED     = "#48607a"
TEXT      = "#c8daf0"
DIM       = "#6888a0"
CANVAS_BG = "#040710"
FNT       = "Courier New"

# ── Logging ───────────────────────────────────────────────────────────────────
_lq: queue.Queue = queue.Queue()
class _QH(logging.Handler):
    def emit(self, r): _lq.put(self.format(r))


# ══════════════════════════════════════════════════════════════════════════════
# CSV STATE MACHINE DATA
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TendonState:
    """One row of the CSV — one named state."""
    name:    str
    delay_s: float
    # Per motor: (displacement_mm, speed_mms)
    motors:  list = field(default_factory=lambda: [(0.0, 20.0)] * 6)

    def summary(self) -> str:
        parts = [f"M{i+1}: {self.motors[i][0]:+.1f}mm@{self.motors[i][1]:.0f}mm/s"
                 for i in range(6)]
        return f"[{self.name}]  " + "  ".join(parts)


def parse_states_csv(path: str) -> tuple[list, str]:
    """
    Parse the tendon state CSV file.

    Expected columns:
        state_name, delay_s,
        m1_disp_mm, m1_speed_mms,
        m2_disp_mm, m2_speed_mms,
        ...
        m6_disp_mm, m6_speed_mms

    Returns (list_of_states, error_string).
    """
    states = []
    errors = []
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            required = ["state_name","delay_s"]
            for col in required:
                if col not in reader.fieldnames:
                    return [], f"Missing required column: '{col}'"

            for row_num, row in enumerate(reader, start=2):
                name  = row.get("state_name", f"State {row_num}").strip()
                try:
                    delay = float(row.get("delay_s", "1.0") or "1.0")
                    delay = max(0.0, delay)
                except ValueError:
                    delay = 1.0
                    errors.append(f"Row {row_num}: invalid delay_s, using 1.0")

                motors = []
                for m in range(1, 7):
                    d_key = f"m{m}_disp_mm"
                    s_key = f"m{m}_speed_mms"
                    try:
                        disp  = float(row.get(d_key, "0") or "0")
                    except ValueError:
                        disp = 0.0
                        errors.append(f"Row {row_num} M{m}: invalid displacement")
                    try:
                        speed = float(row.get(s_key, "20") or "20")
                        speed = max(0.1, min(speed, CAPSTAN_MAX_SPEED_MMS))
                    except ValueError:
                        speed = 20.0
                        errors.append(f"Row {row_num} M{m}: invalid speed, using 20")
                    # Clamp displacement
                    disp = max(-CAPSTAN_MAX_DISP_MM, min(CAPSTAN_MAX_DISP_MM, disp))
                    motors.append((disp, speed))

                states.append(TendonState(name=name, delay_s=delay, motors=motors))

    except FileNotFoundError:
        return [], f"File not found: {path}"
    except Exception as e:
        return [], f"CSV error: {e}"

    return states, "\n".join(errors)


def create_sample_csv(path: str = "sample_states.csv"):
    """Write a sample tendon state CSV file."""
    header = ["state_name","delay_s"]
    for m in range(1, 7):
        header += [f"m{m}_disp_mm", f"m{m}_speed_mms"]

    rows = [
        # name, delay, m1_d, m1_s, m2_d, m2_s, ...
        ["Home",           1.5,   0, 20,   0, 20,   0, 20,   0, 20,   0, 20,   0, 20],
        ["Bend Forward",   2.0,  30, 30,  30, 30,  -30,30, -30, 30,   0, 20,   0, 20],
        ["Bend Backward",  2.0, -30, 30, -30, 30,  30, 30,  30, 30,   0, 20,   0, 20],
        ["Bend Left",      2.0,  30, 25,  -30,25,   0, 20,   0, 20,  30, 25, -30, 25],
        ["Bend Right",     2.0, -30, 25,  30, 25,   0, 20,   0, 20, -30, 25,  30, 25],
        ["Elongate",       2.0, -20, 20, -20, 20, -20, 20, -20, 20, -20, 20, -20, 20],
        ["Shorten",        2.0,  40, 20,  40, 20,  40, 20,  40, 20,  40, 20,  40, 20],
        ["Spiral CW",      2.5,  20, 15,  10, 15,  -10,15, -20, 15, -10, 15,  10, 15],
        ["Spiral CCW",     2.5, -20, 15, -10, 15,  10, 15,  20, 15,  10, 15, -10, 15],
        ["Home",           1.0,   0, 30,   0, 30,   0, 30,   0, 30,   0, 30,   0, 30],
    ]

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    return path


# ══════════════════════════════════════════════════════════════════════════════
# CAPSTAN DRUM VISUALIZER  (left panel)
# ══════════════════════════════════════════════════════════════════════════════

class CapstanPanel(tk.Frame):
    """
    Shows 6 capstan drum gauges.
    Each drum shows:
      - Circular drum icon with rotation indicator line
      - Displacement bar (filled proportionally to max displacement)
      - Numeric displacement (mm) and speed (mm/s) readout
      - Target vs actual comparison
    """

    DRUM_R = 28   # drum circle radius in pixels

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._disps  = [0.0] * 6    # current displacements (mm)
        self._speeds = [0.0] * 6    # current speeds (mm/s)
        self._targets = [0.0] * 6   # target displacements (mm)
        self._canvases = []
        self._disp_lbls  = []
        self._speed_lbls = []
        self._tgt_lbls   = []
        self._build()

    def _build(self):
        tk.Label(self, text="CAPSTAN DRUMS  —  LIVE TENDON STATE",
                 font=(FNT,11,"bold"), fg=ACCENT, bg=BG).pack(anchor="w", pady=(8,6))
        tk.Label(self,
                 text="Each motor winds/unwinds one tendon cable.  + = pull (shorten)  − = release",
                 font=(FNT,9), fg=MUTED, bg=BG).pack(anchor="w", pady=(0,8))

        grid = tk.Frame(self, bg=BG)
        grid.pack(fill="x")

        for i in range(6):
            col = TENDON_COLORS[i]
            r, c = divmod(i, 3)

            cell = tk.Frame(grid, bg=CARD,
                            highlightbackground=BORDER, highlightthickness=1)
            cell.grid(row=r, column=c, padx=4, pady=4, sticky="nsew")
            grid.columnconfigure(c, weight=1)

            # Header
            hdr = tk.Frame(cell, bg=col, height=3)
            hdr.pack(fill="x")
            tk.Label(cell, text=f"M{i+1}  {MOTOR_NAMES[i]}",
                     font=(FNT,10,"bold"), fg=col, bg=CARD
                     ).pack(anchor="w", padx=8, pady=(6,2))

            # Drum canvas
            cv = tk.Canvas(cell, width=130, height=70, bg=CARD, highlightthickness=0)
            cv.pack(padx=8, pady=2)
            self._canvases.append(cv)

            # Stats row
            sr = tk.Frame(cell, bg=CARD)
            sr.pack(fill="x", padx=8, pady=(0,6))

            dl = tk.Label(sr, text="  0.0 mm", font=(FNT,12,"bold"),
                          fg=col, bg=CARD, anchor="w", width=9)
            dl.pack(side="left")
            self._disp_lbls.append(dl)

            vl = tk.Label(sr, text="0.0 mm/s", font=(FNT,10),
                          fg=MUTED, bg=CARD, anchor="w")
            vl.pack(side="left", padx=(4,0))
            self._speed_lbls.append(vl)

            tl = tk.Label(sr, text="→ 0.0", font=(FNT,9),
                          fg=YELLOW, bg=CARD, anchor="e")
            tl.pack(side="right")
            self._tgt_lbls.append(tl)

        # Draw initial state
        self.after(50, self._draw_all)

    def _draw_all(self):
        for i in range(6):
            self._draw_drum(i)

    def _draw_drum(self, i: int):
        cv  = self._canvases[i]
        col = TENDON_COLORS[i]
        w, h = 130, 70

        cv.delete("all")

        # Background gradient suggestion
        cv.create_rectangle(0, 0, w, h, fill=CARD, outline="")

        # Displacement bar (horizontal fill)
        d     = self._disps[i]
        ratio = (d + CAPSTAN_MAX_DISP_MM) / (2 * CAPSTAN_MAX_DISP_MM)
        ratio = max(0.0, min(1.0, ratio))
        bar_w = int(w * ratio)

        # Bar background
        cv.create_rectangle(0, h-16, w, h, fill=INPUT, outline="")
        # Bar fill (green = positive pull, blue = release)
        bar_col = col if d >= 0 else BLUE
        if bar_w > 0:
            cv.create_rectangle(0, h-16, bar_w, h, fill=bar_col, outline="")
        # Centre marker
        cv.create_line(w//2, h-16, w//2, h, fill=BORDER, width=1, dash=(2,2))
        cv.create_text(4, h-8, text=f"{-CAPSTAN_MAX_DISP_MM:.0f}",
                       fill=MUTED, font=(FNT,6), anchor="w")
        cv.create_text(w-4, h-8, text=f"+{CAPSTAN_MAX_DISP_MM:.0f}",
                       fill=MUTED, font=(FNT,6), anchor="e")
        cv.create_text(w//2, h-8, text="0",
                       fill=MUTED, font=(FNT,6), anchor="center")

        # Drum circle
        dr = self.DRUM_R
        cx, cy = 38, 32
        cv.create_oval(cx-dr, cy-dr, cx+dr, cy+dr,
                       fill="#0f1a28", outline=col, width=2)

        # Rotation indicator line (angle proportional to displacement)
        angle_rad = math.radians(mm_to_deg(d, i) % 360)
        lx = cx + (dr-6) * math.sin(angle_rad)
        ly = cy - (dr-6) * math.cos(angle_rad)
        cv.create_line(cx, cy, lx, ly, fill=col, width=2)
        cv.create_oval(cx-3, cy-3, cx+3, cy+3, fill=col, outline="")

        # Tendon line coming off drum
        tx = cx + dr
        ty = cy
        cv.create_line(tx, ty, tx+18, ty, fill=col, width=2)
        # Arrow indicating direction
        arrow = "→" if d >= 0 else "←"
        cv.create_text(tx+28, ty, text=arrow, fill=col,
                       font=(FNT,9,"bold"), anchor="w")

        # Target marker on displacement bar
        t_ratio = (self._targets[i] + CAPSTAN_MAX_DISP_MM) / (2 * CAPSTAN_MAX_DISP_MM)
        t_ratio = max(0.0, min(1.0, t_ratio))
        tx2 = int(w * t_ratio)
        cv.create_line(tx2, h-18, tx2, h, fill=YELLOW, width=2)

        # Speed indicator — tiny arc
        spd_ratio = abs(self._speeds[i]) / CAPSTAN_MAX_SPEED_MMS
        spd_arc = spd_ratio * 270
        if spd_arc > 0:
            cv.create_arc(cx-dr+4, cy-dr+4, cx+dr-4, cy+dr-4,
                          start=90, extent=-spd_arc,
                          style="arc", outline=YELLOW, width=2)

        # Numeric HUD inside drum area (right of drum)
        cv.create_text(95, 14, text=f"{d:+.1f}mm",
                       fill=col, font=(FNT,8,"bold"), anchor="center")
        spd_col = YELLOW if abs(self._speeds[i]) > 0.1 else MUTED
        cv.create_text(95, 28, text=f"{self._speeds[i]:.1f}mm/s",
                       fill=spd_col, font=(FNT,7), anchor="center")
        err = abs(d - self._targets[i])
        err_col = GREEN if err < 1.0 else (YELLOW if err < 5.0 else RED)
        cv.create_text(95, 42, text=f"err:{err:.1f}mm",
                       fill=err_col, font=(FNT,7), anchor="center")

    def update(self, motor_id: int, disp_mm: float, speed_mms: float,
               target_mm: float = None):
        """Update one drum. motor_id is 1-indexed."""
        i = motor_id - 1
        if not 0 <= i < 6:
            return
        self._disps[i]  = disp_mm
        self._speeds[i] = speed_mms
        if target_mm is not None:
            self._targets[i] = target_mm
        self._disp_lbls[i].config(text=f"{disp_mm:+.1f} mm")
        spd_col = YELLOW if abs(speed_mms) > 0.1 else MUTED
        self._speed_lbls[i].config(text=f"{speed_mms:.1f}mm/s", fg=spd_col)
        self._tgt_lbls[i].config(
            text=f"→ {self._targets[i]:+.1f}")
        self._draw_drum(i)

    def update_all(self, disps: list, speeds: list, targets: list = None):
        for i in range(6):
            tgt = targets[i] if targets else self._targets[i]
            self.update(i+1, disps[i], speeds[i], tgt)


# ══════════════════════════════════════════════════════════════════════════════
# CONTINUUM BODY SHAPE VISUALIZER  (right panel)
# ══════════════════════════════════════════════════════════════════════════════

class ContinuumVisualizer(tk.Frame):
    """
    Shows the estimated shape of the continuum body in two views:
      LEFT  — side view  (lateral bending)
      RIGHT — top view   (forward/back bending + tendon attachment layout)
    Also shows the tendon force heatmap across the 6 tendons.
    """

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._disps = [0.0] * 6
        self._cv_side = None
        self._cv_top  = None
        self._cv_heat = None
        self._top_zoom = 1.0
        self._build()
        self.after(100, self._draw_all)

    def _build(self):
        tk.Label(self, text="CONTINUUM BODY  —  ESTIMATED SHAPE",
                 font=(FNT,11,"bold"), fg=BLUE, bg=BG).pack(anchor="w", pady=(8,4))
        tk.Label(self, text="Shape estimated using piecewise constant curvature model  ·  drag = real shape may differ",
                 font=(FNT,9), fg=MUTED, bg=BG).pack(anchor="w", pady=(0,8))

        views = tk.Frame(self, bg=BG)
        views.pack(fill="both", expand=True)

        # Side view
        sc = tk.Frame(views, bg=CARD,
                      highlightbackground=BORDER, highlightthickness=1)
        sc.pack(side="left", fill="both", expand=True, padx=(0,4))
        tk.Label(sc, text="SIDE VIEW  (lateral bend)",
                 font=(FNT,7,"bold"), fg=BLUE, bg=PANEL
                 ).pack(fill="x", padx=0)
        self._cv_side = tk.Canvas(sc, bg=CANVAS_BG, highlightthickness=0)
        self._cv_side.pack(fill="both", expand=True)
        self._cv_side.bind("<Configure>", lambda e: self._draw_side())

        # Top view + tendon layout
        tc = tk.Frame(views, bg=CARD,
                      highlightbackground=BORDER, highlightthickness=1)
        tc.pack(side="left", fill="both", expand=True, padx=(4,0))
        tk.Label(tc, text="TOP VIEW  (forward bend)  +  tendon layout",
                 font=(FNT,7,"bold"), fg=ORANGE, bg=PANEL
                 ).pack(fill="x", padx=0)
        self._cv_top = tk.Canvas(tc, bg=CANVAS_BG, highlightthickness=0)
        self._cv_top.pack(fill="both", expand=True)
        self._cv_top.bind("<Configure>", lambda e: self._draw_top())
        self._cv_top.bind("<MouseWheel>", self._on_top_scroll)

        # Heatmap strip at bottom
        hc = tk.Frame(self, bg=CARD,
                      highlightbackground=BORDER, highlightthickness=1)
        hc.pack(fill="x", pady=(6,0))
        tk.Label(hc, text="TENDON LOAD HEATMAP",
                 font=(FNT,7,"bold"), fg=MUTED, bg=PANEL
                 ).pack(fill="x")
        self._cv_heat = tk.Canvas(hc, height=48, bg=CANVAS_BG, highlightthickness=0)
        self._cv_heat.pack(fill="x")
        self._cv_heat.bind("<Configure>", lambda e: self._draw_heat())

    def set_displacements(self, disps: list):
        self._disps = list(disps)
        self._draw_all()

    def _on_top_scroll(self, event):
        if event.delta > 0:
            self._top_zoom *= 1.1
        elif event.delta < 0:
            self._top_zoom /= 1.1
        self._draw_top()
        return "break"

    def _draw_all(self):
        self._draw_side()
        self._draw_top()
        self._draw_heat()

    def _draw_side(self):
        cv = self._cv_side
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 20 or h < 20:
            return
        cv.delete("all")
        pts_side, _, _ = estimate_continuum_shape(self._disps)

        # Background grid
        self._grid(cv, w, h)

        # Scale to canvas
        scl   = min(w, h) * 0.65
        cx, cy = w / 2, h * 0.85

        # Draw body
        ppx = [cx + p[0] / 200.0 * scl for p in pts_side]
        ppy = [cy - p[1] / 200.0 * scl for p in pts_side]

        # Body thickness tubes
        n = len(pts_side)
        BODY_R = 8   # body radius in px
        for i in range(1, n):
            # Gradient along body: base → tip
            t = i / n
            r = int(0x12 + t*(0x00 - 0x12))
            g = int(0x1b + t*(0xdd - 0x1b))
            b_c = int(0x28 + t*(0xb8 - 0x28))
            col = f"#{r:02x}{g:02x}{b_c:02x}"
            bw  = max(2, int(BODY_R * (1 - t * 0.4)))
            cv.create_line(ppx[i-1], ppy[i-1], ppx[i], ppy[i],
                           fill=col, width=bw*2, capstyle=tk.ROUND)

        # Centreline
        for i in range(1, n):
            cv.create_line(ppx[i-1], ppy[i-1], ppx[i], ppy[i],
                           fill=ACCENT, width=2, capstyle=tk.ROUND)

        # Base marker
        cv.create_rectangle(ppx[0]-12, ppy[0]-4, ppx[0]+12, ppy[0]+4,
                            fill=MUTED, outline=TEXT, width=1)
        cv.create_text(ppx[0], ppy[0]+14, text="BASE",
                       fill=MUTED, font=(FNT,7))

        # Tip marker
        cv.create_oval(ppx[-1]-8, ppy[-1]-8, ppx[-1]+8, ppy[-1]+8,
                       fill=ACCENT, outline=TEXT, width=2)
        cv.create_text(ppx[-1]+14, ppy[-1], text="TIP",
                       fill=ACCENT, font=(FNT,8,"bold"))

        # Tip coordinates
        tip_x = pts_side[-1][0]
        tip_y = pts_side[-1][1]
        cv.create_text(w//2, h-8,
                       text=f"Tip offset: {tip_x:+.1f}mm lateral  |  "
                            f"body along: {tip_y:.1f}mm",
                       fill=DIM, font=(FNT,7), anchor="s")

    def _draw_top(self):
        cv = self._cv_top
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 20 or h < 20:
            return
        cv.delete("all")
        _, pts_top, body_len = estimate_continuum_shape(self._disps)

        self._grid(cv, w, h)

        scl   = min(w, h) * 0.60 * self._top_zoom
        cx, cy = w / 2, h * 0.85

        # Continuum body (top view)
        n   = len(pts_top)
        ppx = [cx + p[0] / 200.0 * scl for p in pts_top]
        ppy = [cy - p[1] / 200.0 * scl for p in pts_top]

        for i in range(1, n):
            t   = i / n
            bw  = max(2, int(10 * (1 - t * 0.4)))
            col = ORANGE
            cv.create_line(ppx[i-1], ppy[i-1], ppx[i], ppy[i],
                           fill=col, width=bw*2, capstyle=tk.ROUND)
        for i in range(1, n):
            cv.create_line(ppx[i-1], ppy[i-1], ppx[i], ppy[i],
                           fill=YELLOW, width=2, capstyle=tk.ROUND)

        # Tendon attachment points on the base plate
        plate_r  = 40
        tendon_r = 18
        plate_cx, plate_cy = cx, cy

        # Draw base plate (hexagon)
        hex_pts = []
        for k in range(6):
            a = math.radians(30 + k*60)
            hex_pts += [plate_cx + plate_r*math.cos(a),
                        plate_cy + plate_r*math.sin(a)]
        cv.create_polygon(*hex_pts, fill="#0f1a28", outline=MUTED, width=1)

        # Draw tendon attachment points and lines to tip
        for i in range(6):
            ta = math.radians(TENDON_ANGLES_DEG[i])
            tx = plate_cx + tendon_r * math.cos(ta)
            ty = plate_cy + tendon_r * math.sin(ta)
            col = TENDON_COLORS[i]

            # Tendon line to body tip
            tip_x_px = ppx[-1]
            tip_y_px = ppy[-1]

            # Line width proportional to tension (pull = thicker)
            d = max(0, self._disps[i])
            lw = 1 + int(d / CAPSTAN_MAX_DISP_MM * 3)
            cv.create_line(tx, ty, tip_x_px, tip_y_px,
                           fill=col, width=lw, dash=(4, 3))

            r2 = 5
            cv.create_oval(tx-r2, ty-r2, tx+r2, ty+r2,
                           fill=col, outline="")
            cv.create_text(tx + 7*math.cos(ta), ty + 7*math.sin(ta),
                           text=TENDON_NAMES[i],
                           fill=col, font=(FNT,6))

        # Tip
        cv.create_oval(ppx[-1]-8, ppy[-1]-8, ppx[-1]+8, ppy[-1]+8,
                       fill=YELLOW, outline=TEXT, width=2)
        cv.create_text(ppx[-1]+14, ppy[-1], text="TIP",
                       fill=YELLOW, font=(FNT,8,"bold"), anchor="w")

        # Base
        cv.create_text(cx, cy+plate_r+12, text="BASE PLATE",
                       fill=MUTED, font=(FNT,6))

        # Body length indicator
        cv.create_text(w//2, h-8,
                       text=f"Est. body length: {body_len:.0f}mm  |  "
                            f"tip fwd: {pts_top[-1][1]:.1f}mm",
                       fill=DIM, font=(FNT,7), anchor="s")

    def _draw_heat(self):
        cv = self._cv_heat
        w = cv.winfo_width()
        if w < 20:
            return
        cv.delete("all")
        h  = 48
        bw = w // 6

        for i in range(6):
            x0 = i * bw
            x1 = (i+1) * bw
            d  = self._disps[i]
            # Normalise: negative=release(cool), positive=pull(hot)
            norm  = (d + CAPSTAN_MAX_DISP_MM) / (2 * CAPSTAN_MAX_DISP_MM)
            norm  = max(0.0, min(1.0, norm))

            # Color: blue(release) → dark(zero) → red(pull) with tendon accent
            if norm > 0.5:
                t2 = (norm - 0.5) * 2
                r  = int(0x20 + t2*(0xf0 - 0x20))
                g  = int(0x40 * (1-t2))
                b  = int(0x40 * (1-t2))
            else:
                t2 = (0.5 - norm) * 2
                r  = int(0x20 * (1-t2))
                g  = int(0x40 * (1-t2))
                b  = int(0x20 + t2*(0xe0 - 0x20))
            heat_col = f"#{r:02x}{g:02x}{b:02x}"

            cv.create_rectangle(x0, 0, x1, h, fill=heat_col, outline=BORDER)

            # Tendon label
            col = TENDON_COLORS[i]
            cv.create_text((x0+x1)//2, 10, text=TENDON_NAMES[i],
                           fill=col, font=(FNT,7,"bold"))
            cv.create_text((x0+x1)//2, 26, text=f"{d:+.0f}mm",
                           fill=TEXT, font=(FNT,8,"bold"))
            cv.create_text((x0+x1)//2, 40, text=MOTOR_NAMES[i].split()[0],
                           fill=DIM, font=(FNT,6))

    @staticmethod
    def _grid(cv, w, h):
        step_px = 40
        for x in range(0, w, step_px):
            cv.create_line(x, 0, x, h, fill="#0c1520")
        for y in range(0, h, step_px):
            cv.create_line(0, y, w, y, fill="#0c1520")


# ══════════════════════════════════════════════════════════════════════════════
# MANUAL JOG PANEL
# ══════════════════════════════════════════════════════════════════════════════

class ManualJogPanel(tk.Frame):
    """
    Manual per-motor displacement and speed control.
    Each motor row has:
      - + / − continuous jog buttons (hold to jog)
      - Step jog buttons with configurable step (mm)
      - Direct displacement entry + SEND
      - Speed entry (mm/s)
      - Live position bar
    """

    def __init__(self, parent, get_robot, get_disps, apply_disp_fn, log_fn, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._get_robot   = get_robot
        self._get_disps   = get_disps         # () → list[float mm]
        self._apply_disp  = apply_disp_fn     # (motor_idx, disp_mm, speed_mms)
        self._log         = log_fn

        self._disp_vars  = [tk.StringVar(value="0.0") for _ in range(6)]
        self._speed_vars = [tk.StringVar(value="20.0") for _ in range(6)]
        self._step_var   = tk.StringVar(value="5")
        self._jog_jobs   = {}
        self._build()

    def _build(self):
        gbar = tk.Frame(self, bg=PANEL)
        gbar.pack(fill="x")
        tk.Label(gbar, text="MANUAL JOG  —  DISPLACEMENT CONTROL",
                 font=(FNT,11,"bold"), fg=ACCENT, bg=PANEL
                 ).pack(side="left", padx=14, pady=10)
        tk.Label(gbar, text="STEP (mm):", font=(FNT,10), fg=MUTED, bg=PANEL
                 ).pack(side="left", padx=(16,4))
        for s in ["1","2","5","10","20"]:
            tk.Radiobutton(gbar, text=f"{s}mm", variable=self._step_var, value=s,
                           font=(FNT,10,"bold"), bg=PANEL, fg=TEXT,
                           selectcolor=INPUT, activebackground=PANEL,
                           indicatoron=False, relief="flat", cursor="hand2",
                           padx=8, pady=3
                           ).pack(side="left", padx=2)
        for txt, col, fg2, cmd in [
            ("⚠ E-STOP", RED, TEXT,     self._estop),
            ("STOP ALL", YELLOW, BG,   self._stop_all),
            ("ZERO ALL", INPUT, TEXT,   self._zero_all),
        ]:
            tk.Button(gbar, text=txt, font=(FNT,10,"bold"),
                      bg=col, fg=fg2, relief="flat", cursor="hand2",
                      padx=10, pady=4, command=cmd
                      ).pack(side="right", padx=4)

        # Headers
        hdr = tk.Frame(self, bg=PANEL)
        hdr.pack(fill="x")
        for txt, w in [
            ("", 4), ("Motor", 14), ("Displacement bar", 28),
            ("Live mm", 10), ("  −  JOG  +", 15), ("Step", 10),
            ("Target mm", 10), ("Speed mm/s", 12),
        ]:
            tk.Label(hdr, text=txt, font=(FNT,9,"bold"), fg=MUTED, bg=PANEL,
                     width=w, anchor="w").pack(side="left", padx=3, pady=4)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        for i in range(6):
            self._build_row(i)

    def _build_row(self, i: int):
        col = TENDON_COLORS[i]
        bg  = BG if i % 2 == 0 else CARD
        row = tk.Frame(self, bg=bg)
        row.pack(fill="x", pady=2)

        tk.Frame(row, bg=col, width=5).pack(side="left", fill="y")
        tk.Label(row, text=f"M{i+1}\n{TENDON_NAMES[i]}",
                 font=(FNT,8,"bold"), fg=col, bg=bg, width=5
                 ).pack(side="left", padx=(4,2))
        tk.Label(row, text=MOTOR_NAMES[i],
                 font=(FNT,7), fg=MUTED, bg=bg, width=14
                 ).pack(side="left", padx=2)

        # Displacement bar canvas
        c = tk.Canvas(row, width=160, height=28, bg=bg, highlightthickness=0)
        c.pack(side="left", padx=4)
        # Track
        c.create_rectangle(2, 10, 158, 20, fill=INPUT, outline="")
        # Zero line
        c.create_line(80, 6, 80, 24, fill=BORDER, width=1)
        bar_fill = c.create_rectangle(80, 10, 80, 20, fill=col, outline="")
        bar_lbl  = c.create_text(80, 14, text="0.0", fill=TEXT, font=(FNT,7))
        # Store canvas refs for update
        if not hasattr(self, '_bar_canvases'):
            self._bar_canvases = []
            self._bar_fills    = []
            self._bar_lbls     = []
        self._bar_canvases.append(c)
        self._bar_fills.append(bar_fill)
        self._bar_lbls.append(bar_lbl)

        # Live displacement label
        ll = tk.Label(row, text="  0.0", font=(FNT,12,"bold"),
                      fg=col, bg=bg, width=7)
        ll.pack(side="left", padx=4)
        if not hasattr(self, '_live_lbls'):
            self._live_lbls = []
        self._live_lbls.append(ll)

        # Jog buttons
        b_neg = tk.Button(row, text=" − ", font=(FNT,13,"bold"),
                          bg=INPUT, fg=col, relief="flat", cursor="hand2",
                          padx=10, pady=3)
        b_neg.pack(side="left", padx=2)
        b_pos = tk.Button(row, text=" + ", font=(FNT,13,"bold"),
                          bg=INPUT, fg=col, relief="flat", cursor="hand2",
                          padx=10, pady=3)
        b_pos.pack(side="left", padx=2)
        b_neg.bind("<ButtonPress-1>",   lambda e, idx=i: self._jog_start(idx, -1))
        b_neg.bind("<ButtonRelease-1>", lambda e, idx=i: self._jog_stop(idx))
        b_pos.bind("<ButtonPress-1>",   lambda e, idx=i: self._jog_start(idx, +1))
        b_pos.bind("<ButtonRelease-1>", lambda e, idx=i: self._jog_stop(idx))

        # Step buttons
        tk.Button(row, text="−S", font=(FNT,9,"bold"),
                  bg=INPUT, fg=DIM, relief="flat", cursor="hand2",
                  padx=6, pady=2,
                  command=lambda idx=i: self._step_jog(idx, -1)
                  ).pack(side="left", padx=2)
        tk.Button(row, text="+S", font=(FNT,9,"bold"),
                  bg=INPUT, fg=DIM, relief="flat", cursor="hand2",
                  padx=6, pady=2,
                  command=lambda idx=i: self._step_jog(idx, +1)
                  ).pack(side="left", padx=2)

        # Direct entry + SEND
        tk.Entry(row, textvariable=self._disp_vars[i], width=7,
                 font=(FNT,11), bg=INPUT, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 highlightbackground=BORDER, highlightthickness=1
                 ).pack(side="left", padx=(8,2))
        tk.Label(row, text="mm", font=(FNT,9), fg=MUTED, bg=bg
                 ).pack(side="left")
        tk.Button(row, text="SEND", font=(FNT,10,"bold"),
                  bg=col, fg=BG, relief="flat", cursor="hand2",
                  padx=10, pady=3,
                  command=lambda idx=i: self._send(idx)
                  ).pack(side="left", padx=6)

        # Speed entry
        tk.Entry(row, textvariable=self._speed_vars[i], width=6,
                 font=(FNT,11), bg=INPUT, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 highlightbackground=BORDER, highlightthickness=1
                 ).pack(side="left", padx=(6,2))
        tk.Label(row, text="mm/s", font=(FNT,9), fg=MUTED, bg=bg
                 ).pack(side="left")

    # ── Feedback update ───────────────────────────────────────────────────────

    def update_disp(self, motor_id: int, disp_mm: float):
        """Update the visual bar for one motor (1-indexed)."""
        i = motor_id - 1
        if not 0 <= i < 6: return
        self._live_lbls[i].config(text=f"{disp_mm:+.1f}")
        # Update bar
        c    = self._bar_canvases[i]
        fill = self._bar_fills[i]
        lbl  = self._bar_lbls[i]
        ratio = (disp_mm + CAPSTAN_MAX_DISP_MM) / (2 * CAPSTAN_MAX_DISP_MM)
        ratio = max(0.0, min(1.0, ratio))
        bar_x = 2 + int(156 * ratio)
        c.coords(fill, min(80, bar_x), 10, max(80, bar_x), 20)
        c.itemconfig(fill, fill=TENDON_COLORS[i] if disp_mm >= 0 else BLUE)
        c.coords(lbl, 80, 14)
        c.itemconfig(lbl, text=f"{disp_mm:+.1f}")

    # ── Jog actions ───────────────────────────────────────────────────────────

    def _get_speed(self, i: int) -> float:
        try:
            return max(0.1, float(self._speed_vars[i].get()))
        except ValueError:
            return 20.0

    def _jog_start(self, i: int, direction: int):
        self._jog_stop(i)
        self._do_jog(i, direction)

    def _do_jog(self, i: int, direction: int):
        spd  = self._get_speed(i)
        step = spd * 0.08 * direction   # 80ms tick
        disps = self._get_disps()
        new_d = max(-CAPSTAN_MAX_DISP_MM,
                    min(CAPSTAN_MAX_DISP_MM, disps[i] + step))
        self._apply_disp(i, new_d, spd)
        self._jog_jobs[i] = self.after(80, lambda: self._do_jog(i, direction))

    def _jog_stop(self, i: int):
        if i in self._jog_jobs:
            self.after_cancel(self._jog_jobs.pop(i))
        robot = self._get_robot()
        if robot and HAS_ROBOT:
            try:
                from robot_controller import DEFAULT_JOINT_CONFIG
                cfg = DEFAULT_JOINT_CONFIG[i]
                threading.Thread(target=robot.motors[cfg.motor_id].stop,
                                 daemon=True).start()
            except Exception:
                pass

    def _step_jog(self, i: int, direction: int):
        try:
            step_mm = float(self._step_var.get()) * direction
        except ValueError:
            step_mm = 5.0 * direction
        disps = self._get_disps()
        new_d = max(-CAPSTAN_MAX_DISP_MM,
                    min(CAPSTAN_MAX_DISP_MM, disps[i] + step_mm))
        spd   = self._get_speed(i)
        self._apply_disp(i, new_d, spd)
        self._log(f"[JOG] M{i+1} step {step_mm:+.1f}mm → {new_d:.1f}mm")

    def _send(self, i: int):
        try:
            target = float(self._disp_vars[i].get())
        except ValueError:
            messagebox.showerror("Input Error", f"M{i+1}: enter a valid number (mm).")
            return
        target = max(-CAPSTAN_MAX_DISP_MM, min(CAPSTAN_MAX_DISP_MM, target))
        spd = self._get_speed(i)
        self._apply_disp(i, target, spd)
        self._log(f"[JOG] M{i+1} SEND → {target:+.1f}mm @ {spd:.1f}mm/s")

    def _zero_all(self):
        disps = self._get_disps()
        for i in range(6):
            spd = self._get_speed(i)
            self._apply_disp(i, 0.0, spd)
        self._log("[JOG] ZERO ALL motors")

    def _stop_all(self):
        for i in list(self._jog_jobs.keys()):
            self._jog_stop(i)
        robot = self._get_robot()
        if robot and HAS_ROBOT:
            threading.Thread(target=robot.stop_all, daemon=True).start()
        self._log("[JOG] STOP ALL")

    def _estop(self):
        for i in list(self._jog_jobs.keys()):
            self.after_cancel(self._jog_jobs.pop(i))
        robot = self._get_robot()
        if robot and HAS_ROBOT:
            robot.estop()
        self._log("[JOG] ⚠ E-STOP", "err")


# ══════════════════════════════════════════════════════════════════════════════
# CSV STATE MACHINE PANEL
# ══════════════════════════════════════════════════════════════════════════════

class CsvStatePanel(tk.Frame):
    def __init__(self, parent, get_robot, apply_state_fn, log_fn, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._get_robot    = get_robot
        self._apply_state  = apply_state_fn   # (TendonState) → None
        self._log          = log_fn
        self._states: list[TendonState] = []
        self._filepath     = tk.StringVar()
        self._delay_var    = tk.StringVar(value="0")
        self._loop_var     = tk.BooleanVar(value=False)
        self._wait_var     = tk.BooleanVar(value=True)
        self._running      = False
        self._paused       = False
        self._abort_flag   = threading.Event()
        self._current_step = -1
        self._row_frames   = []

        # Stopwatch state
        self._seq_start    = 0.0
        self._step_start   = 0.0
        self._delay_start  = 0.0
        self._phase        = "idle"
        self._sw_job       = None
        self._timing_log   = []

        self._build()

    def _build(self):
        # File loader
        fc = tk.Frame(self, bg=CARD,
                      highlightbackground=BORDER, highlightthickness=1)
        fc.pack(fill="x", pady=(0,6))
        tk.Label(fc, text="CSV STATE FILE",
                 font=(FNT,10,"bold"), fg=ACCENT, bg=CARD
                 ).pack(anchor="w", padx=10, pady=(7,3))
        fr = tk.Frame(fc, bg=CARD)
        fr.pack(fill="x", padx=10, pady=(0,8))
        tk.Entry(fr, textvariable=self._filepath,
                 font=(FNT,11), bg=INPUT, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 highlightbackground=BORDER, highlightthickness=1,
                 width=40).pack(side="left", padx=(0,8))
        for txt, col, cmd in [
            ("BROWSE",        BLUE,   self._browse),
            ("LOAD",          ACCENT, self._load),
            ("SAMPLE CSV",    INPUT,  self._make_sample),
        ]:
            tk.Button(fr, text=txt, font=(FNT,10,"bold"),
                      bg=col, fg=BG if col != INPUT else TEXT,
                      relief="flat", cursor="hand2",
                      padx=8, pady=3, command=cmd).pack(side="left", padx=3)

        # Options
        opt = tk.Frame(self, bg=BG)
        opt.pack(fill="x", pady=4)
        tk.Label(opt, text="OVERRIDE DELAY (s):",
                 font=(FNT,10), fg=MUTED, bg=BG).pack(side="left")
        tk.Entry(opt, textvariable=self._delay_var, width=5,
                 font=(FNT,11), bg=INPUT, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 highlightbackground=BORDER, highlightthickness=1
                 ).pack(side="left", padx=(4,14))
        tk.Label(opt, text="(0 = use CSV delay)",
                 font=(FNT,9), fg=MUTED, bg=BG).pack(side="left", padx=(0,14))
        tk.Checkbutton(opt, text="Wait for motors", variable=self._wait_var,
                       font=(FNT,10), bg=BG, fg=TEXT,
                       selectcolor=INPUT, activebackground=BG
                       ).pack(side="left", padx=(0,14))
        tk.Checkbutton(opt, text="Loop", variable=self._loop_var,
                       font=(FNT,10), bg=BG, fg=TEXT,
                       selectcolor=INPUT, activebackground=BG
                       ).pack(side="left")

        # Execution controls
        ec = tk.Frame(self, bg=BG)
        ec.pack(fill="x", pady=(3,6))
        self.btn_run   = tk.Button(ec, text="▶  RUN",
                                   font=(FNT,12,"bold"), bg=GREEN, fg=BG,
                                   relief="flat", cursor="hand2",
                                   padx=18, pady=6, command=self._run,
                                   state="disabled")
        self.btn_run.pack(side="left", padx=(0,6))
        self.btn_pause = tk.Button(ec, text="⏸  PAUSE",
                                   font=(FNT,12,"bold"), bg=YELLOW, fg=BG,
                                   relief="flat", cursor="hand2",
                                   padx=18, pady=6, command=self._pause,
                                   state="disabled")
        self.btn_pause.pack(side="left", padx=(0,6))
        self.btn_abort = tk.Button(ec, text="■  ABORT",
                                   font=(FNT,12,"bold"), bg=RED, fg=TEXT,
                                   relief="flat", cursor="hand2",
                                   padx=18, pady=6, command=self._abort,
                                   state="disabled")
        self.btn_abort.pack(side="left", padx=(0,14))
        self.lbl_status = tk.Label(ec, text="No states loaded.",
                                   font=(FNT,10), fg=MUTED, bg=BG)
        self.lbl_status.pack(side="left")

        # Progress bar
        self.prog_var = tk.DoubleVar(value=0)
        style = ttk.Style()
        style.configure("Seq.Horizontal.TProgressbar",
                        troughcolor=INPUT, background=GREEN, bordercolor=BORDER)
        ttk.Progressbar(self, variable=self.prog_var, maximum=100,
                        style="Seq.Horizontal.TProgressbar"
                        ).pack(fill="x", pady=(0,6))

        # Stopwatch + state preview side by side
        mid = tk.Frame(self, bg=BG)
        mid.pack(fill="x", pady=(0,4))

        # Stopwatch
        sw = tk.Frame(mid, bg=CARD,
                      highlightbackground=BORDER, highlightthickness=1)
        sw.pack(side="left", fill="both", expand=True, padx=(0,6))
        tk.Label(sw, text="⏱  TIMING", font=(FNT,7,"bold"), fg=MUTED, bg=CARD
                 ).pack(anchor="w", padx=8, pady=(5,2))
        cr = tk.Frame(sw, bg=CARD); cr.pack(padx=8, pady=(0,4))

        sc = tk.Frame(cr, bg=CARD); sc.pack(side="left", padx=(0,14))
        tk.Label(sc, text="TOTAL", font=(FNT,6), fg=MUTED, bg=CARD).pack()
        self.lbl_total = tk.Label(sc, text="00:00.0",
                                  font=(FNT,18,"bold"), fg=ACCENT, bg=CARD)
        self.lbl_total.pack()

        mc = tk.Frame(cr, bg=CARD); mc.pack(side="left", padx=(0,10))
        tk.Label(mc, text="MOTION", font=(FNT,6), fg=MUTED, bg=CARD).pack()
        self.lbl_motion = tk.Label(mc, text="0.00s",
                                   font=(FNT,12,"bold"), fg=BLUE, bg=CARD)
        self.lbl_motion.pack()

        dc = tk.Frame(cr, bg=CARD); dc.pack(side="left", padx=(0,10))
        tk.Label(dc, text="DELAY", font=(FNT,6), fg=MUTED, bg=CARD).pack()
        self.lbl_delay = tk.Label(dc, text="0.00s",
                                  font=(FNT,12,"bold"), fg=YELLOW, bg=CARD)
        self.lbl_delay.pack()

        ph = tk.Frame(cr, bg=CARD); ph.pack(side="left")
        tk.Label(ph, text="PHASE", font=(FNT,6), fg=MUTED, bg=CARD).pack()
        self.lbl_phase = tk.Label(ph, text="IDLE",
                                  font=(FNT,10,"bold"), fg=MUTED, bg=CARD)
        self.lbl_phase.pack()

        tk.Label(sw, text="STEP HISTORY", font=(FNT,6), fg=MUTED, bg=CARD
                 ).pack(anchor="w", padx=8)
        self._timing_lbls = []
        for _ in range(4):
            l = tk.Label(sw, text="", font=(FNT,7), fg=MUTED, bg=CARD, anchor="w")
            l.pack(fill="x", padx=8)
            self._timing_lbls.append(l)
        tk.Label(sw, text="", bg=CARD).pack(pady=2)

        # State preview table (scrollable)
        tbl_outer = tk.Frame(mid, bg=CARD,
                             highlightbackground=BORDER, highlightthickness=1)
        tbl_outer.pack(side="right", fill="both", expand=True)
        tk.Label(tbl_outer, text="STATE PREVIEW",
                 font=(FNT,7,"bold"), fg=MUTED, bg=PANEL
                 ).pack(fill="x", padx=0)

        tbl_cv   = tk.Canvas(tbl_outer, bg=CARD, highlightthickness=0, height=160)
        tbl_sb   = ttk.Scrollbar(tbl_outer, orient="vertical", command=tbl_cv.yview)
        self._tbl_inner = tk.Frame(tbl_cv, bg=CARD)
        self._tbl_inner.bind("<Configure>",
            lambda e: tbl_cv.configure(scrollregion=tbl_cv.bbox("all")))
        tbl_cv.create_window((0,0), window=self._tbl_inner, anchor="nw")
        tbl_cv.configure(yscrollcommand=tbl_sb.set)
        tbl_sb.pack(side="right", fill="y")
        tbl_cv.pack(side="left", fill="both", expand=True)

        self._build_table_header()

    def _build_table_header(self):
        hdr = tk.Frame(self._tbl_inner, bg=PANEL)
        hdr.pack(fill="x")
        cols = [("#",3),("State Name",14),("Delay",6)]
        for i in range(6):
            cols.append((f"M{i+1} mm@mm/s", 14))
        for t, w in cols:
            tk.Label(hdr, text=t, font=(FNT,7,"bold"), fg=MUTED, bg=PANEL,
                     width=w, anchor="w").pack(side="left", padx=2, pady=2)

    def _populate_table(self):
        for w in self._tbl_inner.winfo_children():
            if isinstance(w, tk.Frame) and w.cget("bg") != PANEL:
                w.destroy()
        self._row_frames.clear()
        for idx, state in enumerate(self._states):
            bg = CARD if idx % 2 == 0 else INPUT
            row = tk.Frame(self._tbl_inner, bg=bg)
            row.pack(fill="x")
            self._row_frames.append(row)
            tk.Label(row, text=str(idx+1), font=(FNT,7), fg=MUTED, bg=bg,
                     width=3, anchor="w").pack(side="left", padx=2, pady=1)
            tk.Label(row, text=state.name[:14], font=(FNT,7), fg=TEXT, bg=bg,
                     width=14, anchor="w").pack(side="left", padx=2)
            tk.Label(row, text=f"{state.delay_s:.1f}s", font=(FNT,7), fg=YELLOW, bg=bg,
                     width=6, anchor="w").pack(side="left", padx=2)
            for i in range(6):
                d, s = state.motors[i]
                txt  = f"{d:+.0f}@{s:.0f}"
                col  = TENDON_COLORS[i] if abs(d) > 0.5 else MUTED
                tk.Label(row, text=txt, font=(FNT,7), fg=col, bg=bg,
                         width=14, anchor="w").pack(side="left", padx=2)

    def _highlight_row(self, idx):
        for i, fr in enumerate(self._row_frames):
            bg = GREEN if i == idx else (CARD if i % 2 == 0 else INPUT)
            fr.config(bg=bg)
            for w in fr.winfo_children():
                try: w.config(bg=bg)
                except: pass

    def _clear_highlight(self):
        for i, fr in enumerate(self._row_frames):
            bg = CARD if i % 2 == 0 else INPUT
            fr.config(bg=bg)
            for w in fr.winfo_children():
                try: w.config(bg=bg)
                except: pass

    # ── File actions ──────────────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select State CSV",
            filetypes=[("CSV","*.csv"),("All","*.*")])
        if path:
            self._filepath.set(path)

    def _load(self):
        path = self._filepath.get().strip()
        if not path:
            messagebox.showwarning("No File", "Choose a CSV file first.")
            return
        states, err = parse_states_csv(path)
        if not states:
            messagebox.showerror("Load Failed", err or "No states found.")
            return
        self._states = states
        self._populate_table()
        self.btn_run.config(state="normal")
        self.lbl_status.config(
            text=f"Loaded {len(states)} states from {os.path.basename(path)}",
            fg=GREEN)
        self._log(f"[CSV] Loaded {len(states)} states from {os.path.basename(path)}")
        if err:
            self._log(f"[CSV] Warnings: {err}", "warn")

    def _make_sample(self):
        path = create_sample_csv("sample_states.csv")
        self._filepath.set(os.path.abspath(path))
        self._log(f"[CSV] Sample file created: {path}")
        self._load()

    # ── Execution ─────────────────────────────────────────────────────────────

    def _run(self):
        if self._running or not self._states:
            return
        self._running    = True
        self._paused     = False
        self._abort_flag.clear()
        self._timing_log.clear()
        self.btn_run.config(state="disabled")
        self.btn_pause.config(state="normal", text="⏸  PAUSE")
        self.btn_abort.config(state="normal")
        self.prog_var.set(0)
        self.lbl_total.config(text="00:00.0")
        self.lbl_motion.config(text="0.00s")
        self.lbl_delay.config(text="0.00s")
        for l in self._timing_lbls:
            l.config(text="")
        self._seq_start = time.time()
        self._sw_job    = self.after(100, self._tick_sw)
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        total = len(self._states)
        loop  = self._loop_var.get()
        run_n = 0

        while True:
            run_n += 1
            self.after(0, lambda: self._log(f"[CSV] ── Run #{run_n} ──"))

            for idx, state in enumerate(self._states):
                if self._abort_flag.is_set():
                    self.after(0, self._done, "Aborted.")
                    return
                while self._paused:
                    if self._abort_flag.is_set():
                        self.after(0, self._done, "Aborted.")
                        return
                    time.sleep(0.1)

                self._current_step = idx
                self.after(0, self._highlight_row, idx)
                pct = idx / total * 100
                self.after(0, lambda p=pct: self.prog_var.set(p))
                self.after(0, self.lbl_status.config,
                           {"text": f"State {idx+1}/{total}: {state.name}",
                            "fg": GREEN})

                # Log
                self.after(0, lambda s=state.summary():
                           self._log(f"[CSV] {s}"))

                # Motion phase
                self._step_start = time.time()
                self._set_phase("motion")
                self.after(0, self._apply_state, state)

                # Wait for motors
                if self._wait_var.get():
                    # Simple time estimate based on max displacement needed
                    max_dist = max(abs(m[0]) for m in state.motors)
                    max_spd  = max(m[1] for m in state.motors)
                    est_t    = (max_dist / max_spd + 0.3) if max_spd > 0 else 1.0
                    t_wait = time.time() + est_t
                    while time.time() < t_wait:
                        if self._abort_flag.is_set(): break
                        while self._paused and not self._abort_flag.is_set():
                            time.sleep(0.05)
                        time.sleep(0.05)

                motion_dur = time.time() - self._step_start
                self.after(0, self.lbl_motion.config,
                           {"text": f"{motion_dur:.2f}s", "fg": GREEN})

                # Delay phase
                try:    gui_delay = float(self._delay_var.get())
                except: gui_delay = 0.0
                delay = gui_delay if gui_delay > 0 else state.delay_s

                if delay > 0:
                    self._delay_start = time.time()
                    self._set_phase("delay")
                    t0 = time.time()
                    while time.time() - t0 < delay:
                        if self._abort_flag.is_set():
                            self.after(0, self._done, "Aborted."); return
                        while self._paused and not self._abort_flag.is_set():
                            time.sleep(0.05)
                        time.sleep(0.05)
                    actual_delay = time.time() - self._delay_start
                else:
                    actual_delay = 0.0

                self.after(0, self.lbl_delay.config,
                           {"text": f"{actual_delay:.2f}s", "fg": YELLOW})

                self._timing_log.append((state.name, motion_dur, actual_delay))
                self.after(0, self._refresh_timing)
                self.after(0, lambda n=state.name, md=motion_dur, ad=actual_delay:
                           self._log(f"[CSV] ✓ {n}: motion {md:.2f}s  delay {ad:.2f}s"))

            self.after(0, self.prog_var.set, 100)
            self.after(0, self._clear_highlight)
            self._set_phase("idle")

            if not loop or self._abort_flag.is_set():
                break
            time.sleep(0.2)

        self.after(0, self._done,
                   f"Complete ({run_n} run{'s' if run_n>1 else ''}).")

    def _done(self, msg):
        self._running = False
        self._paused  = False
        if self._sw_job:
            self.after_cancel(self._sw_job)
            self._sw_job = None
        self._set_phase("idle")
        self.btn_run.config(state="normal")
        self.btn_pause.config(state="disabled", text="⏸  PAUSE")
        self.btn_abort.config(state="disabled")
        self.lbl_status.config(text=msg, fg=GREEN if "Abort" not in msg else YELLOW)
        self._log(f"[CSV] {msg}")
        self._clear_highlight()

    def _pause(self):
        if not self._running: return
        self._paused = not self._paused
        if self._paused:
            self.btn_pause.config(text="▶  RESUME", bg=GREEN)
            self.lbl_status.config(text="Paused.", fg=YELLOW)
            self._log("[CSV] Paused.")
        else:
            self.btn_pause.config(text="⏸  PAUSE", bg=YELLOW)
            self._log("[CSV] Resumed.")

    def _abort(self):
        self._abort_flag.set()
        self._paused = False
        self._log("[CSV] Abort requested.")

    def _set_phase(self, phase):
        self._phase = phase
        colors = {"motion": BLUE, "delay": YELLOW, "idle": MUTED}
        labels = {"motion": "▶ MOTION", "delay": "⏳ DELAY", "idle": "◼ IDLE"}
        self.after(0, self.lbl_phase.config,
                   {"text": labels.get(phase, "IDLE"),
                    "fg":   colors.get(phase, MUTED)})

    def _tick_sw(self):
        now = time.time()
        total = now - self._seq_start
        m, s  = divmod(total, 60)
        self.lbl_total.config(text=f"{int(m):02d}:{s:04.1f}")
        if self._phase == "motion":
            self.lbl_motion.config(
                text=f"{now - self._step_start:.2f}s", fg=BLUE)
        elif self._phase == "delay":
            self.lbl_delay.config(
                text=f"{now - self._delay_start:.2f}s", fg=YELLOW)
        self._sw_job = self.after(100, self._tick_sw)

    def _refresh_timing(self):
        recent = self._timing_log[-4:]
        for i, lbl in enumerate(self._timing_lbls):
            if i < len(recent):
                nm, md, dd = recent[i]
                lbl.config(text=f"  {nm[:14]:<14}  m:{md:.2f}s  d:{dd:.2f}s",
                           fg=TEXT)
            else:
                lbl.config(text="")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class ContinuumGUI(tk.Tk):
    def __init__(self, simulated=True, port="COM3", bustype="slcan"):
        super().__init__()
        self.title("⬡  Tendon-Driven Continuum Manipulator  ·  Control Panel")
        self.configure(bg=BG)
        self.minsize(1500, 920)

        self._sim       = simulated
        self._port      = port
        self._bus_t     = bustype
        self.robot      = None
        self.bus        = None
        self._connected = False
        self._mon_job   = None

        # Current tendon displacements (mm) — ground truth for simulation
        self._disps  = [0.0] * 6
        self._speeds = [0.0] * 6

        _h = _QH()
        _h.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(_h)
        logging.getLogger().setLevel(logging.INFO)

        self._build()
        self._poll_log()
        self.protocol("WM_DELETE_WINDOW", self._close)

        # Start monitor immediately
        self._start_monitor()

        if simulated:
            self.after(400, self._connect)

    def _build(self):
        # Top bar
        top = tk.Frame(self, bg=PANEL)
        top.pack(fill="x")
        tk.Label(top,
                 text="⬡  TENDON-DRIVEN CONTINUUM MANIPULATOR  ·  MOTOR CONTROL",
                 font=(FNT,13,"bold"), fg=ACCENT, bg=PANEL
                 ).pack(side="left", padx=16, pady=12)
        self._lbl_c = tk.Label(top, text="● DISCONNECTED",
                               font=(FNT,10,"bold"), fg=RED, bg=PANEL)
        self._lbl_c.pack(side="right", padx=12)
        mode = "SIMULATION" if self._sim else f"REAL · {self._port}"
        tk.Label(top, text=mode, font=(FNT,10), fg=MUTED, bg=PANEL
                 ).pack(side="right", padx=4)
        self._mkbtn(top,"DISCONNECT",PANEL,RED,  self._disconnect).pack(side="right",padx=4)
        self._mkbtn(top,"CONNECT",   ACCENT,BG,  self._connect   ).pack(side="right",padx=4)

        # ── Capstan Radii Configuration ──────────────────────────────────────
        rad_bar = tk.Frame(self, bg=CARD,
                           highlightbackground=BORDER, highlightthickness=1)
        rad_bar.pack(fill="x", padx=8, pady=(4,2))
        tk.Label(rad_bar, text="CAPSTAN RADII (mm):",
                 font=(FNT,10,"bold"), fg=ACCENT, bg=CARD
                 ).pack(side="left", padx=(12,8), pady=8)
        self._radii_vars = []
        for i in range(6):
            col = TENDON_COLORS[i]
            tk.Label(rad_bar, text=f"M{i+1}:",
                     font=(FNT,10,"bold"), fg=col, bg=CARD
                     ).pack(side="left", padx=(8,2))
            rv = tk.StringVar(value=str(CAPSTAN_RADII_MM[i]))
            self._radii_vars.append(rv)
            tk.Entry(rad_bar, textvariable=rv, width=5,
                     font=(FNT,11), bg=INPUT, fg=TEXT,
                     insertbackground=TEXT, relief="flat",
                     highlightbackground=BORDER, highlightthickness=1
                     ).pack(side="left", padx=(0,4))
        tk.Button(rad_bar, text="APPLY RADII", font=(FNT,10,"bold"),
                  bg=ACCENT, fg=BG, relief="flat", cursor="hand2",
                  padx=12, pady=5, command=self._apply_radii
                  ).pack(side="left", padx=(14,0))
        tk.Label(rad_bar, text="(capstan drum radius per motor)",
                 font=(FNT,9), fg=MUTED, bg=CARD
                 ).pack(side="left", padx=(10,0))

        # Body: left = controls, right = visualizations
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        # ── Left: tab bar + manual / CSV panels ──────────────────────────────
        left_outer = tk.Frame(body, bg=BG, width=780)
        left_outer.pack(side="left", fill="y", padx=(8,4), pady=6)
        left_outer.pack_propagate(False)

        # Tab bar
        tbar = tk.Frame(left_outer, bg=BG)
        tbar.pack(fill="x", pady=(0,6))
        self._tab_manual_btn = tk.Button(
            tbar, text="  🕹  MANUAL JOG  ",
            font=(FNT,11,"bold"), bg=ACCENT, fg=BG,
            relief="flat", cursor="hand2", padx=12, pady=6,
            command=self._show_manual)
        self._tab_manual_btn.pack(side="left", padx=(0,4))
        self._tab_csv_btn = tk.Button(
            tbar, text="  📋  CSV STATES  ",
            font=(FNT,11,"bold"), bg=PANEL, fg=MUTED,
            relief="flat", cursor="hand2", padx=12, pady=6,
            command=self._show_csv)
        self._tab_csv_btn.pack(side="left")
        # HOME button
        self._home_btn = tk.Button(
            tbar, text="  🏠  HOME  ",
            font=(FNT,11,"bold"), bg=GREEN, fg=BG,
            relief="flat", cursor="hand2", padx=12, pady=6,
            command=self._home_all)
        self._home_btn.pack(side="right", padx=(6,0))

        # Scrollable container with vertical + horizontal scrollbars
        lhs = ttk.Scrollbar(left_outer, orient="horizontal")
        lhs.pack(side="bottom", fill="x")
        scroll_frame = tk.Frame(left_outer, bg=BG)
        scroll_frame.pack(fill="both", expand=True)
        lc = tk.Canvas(scroll_frame, bg=BG, highlightthickness=0)
        ls = ttk.Scrollbar(scroll_frame, orient="vertical", command=lc.yview)
        lc.configure(yscrollcommand=ls.set, xscrollcommand=lhs.set)
        lhs.configure(command=lc.xview)
        ls.pack(side="right", fill="y")
        lc.pack(side="left", fill="both", expand=True)
        def _mw(e):
            if ls.get() != (0.0, 1.0):
                lc.yview_scroll(int(-1*(e.delta/120)), "units")
        lc.bind_all("<MouseWheel>", _mw)
        self._lc = lc

        # Manual jog panel
        self._manual_frame = tk.Frame(lc, bg=BG)
        self._manual_win   = lc.create_window((0,0), window=self._manual_frame, anchor="nw")
        self._manual_frame.bind("<Configure>",
            lambda e: lc.configure(scrollregion=lc.bbox("all")))

        self._manual_panel = ManualJogPanel(
            self._manual_frame,
            get_robot   = lambda: self.robot,
            get_disps   = lambda: list(self._disps),
            apply_disp_fn = self._apply_single_disp,
            log_fn      = self._log,
        )
        self._manual_panel.pack(fill="x")

        # CSV state panel
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

        self._show_manual()

        # ── Right: visualizations ─────────────────────────────────────────────
        right = tk.Frame(body, bg=BG)
        right.pack(side="right", fill="both", expand=True, padx=(4,8), pady=6)

        # Top half: capstan drums
        self._capstan_panel = CapstanPanel(right)
        self._capstan_panel.pack(fill="x")

        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", pady=4)

        # Bottom half: continuum shape visualizer
        self._cont_viz = ContinuumVisualizer(right)
        self._cont_viz.pack(fill="both", expand=True)

        # Log
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        lh = tk.Frame(self, bg=PANEL); lh.pack(fill="x")
        tk.Label(lh, text="SYSTEM LOG", font=(FNT,8), fg=MUTED,
                 bg=PANEL, pady=3).pack(side="left", padx=12)
        tk.Button(lh, text="CLEAR", font=(FNT,8), bg=PANEL, fg=MUTED,
                  relief="flat", cursor="hand2",
                  command=self._clear_log).pack(side="right", padx=12)
        self._lbox = scrolledtext.ScrolledText(
            self, height=5, font=(FNT,9),
            bg=PANEL, fg=MUTED, relief="flat",
            state="disabled", wrap="word")
        self._lbox.pack(fill="x")
        for t, c in [("ok",GREEN),("warn",YELLOW),("err",RED),("info",DIM)]:
            self._lbox.tag_config(t, foreground=c)

    def _show_manual(self):
        self._lc.itemconfig(self._csv_win,    state="hidden")
        self._lc.itemconfig(self._manual_win, state="normal")
        self._lc.configure(scrollregion=self._lc.bbox("all"))
        self._tab_manual_btn.config(bg=ACCENT, fg=BG)
        self._tab_csv_btn.config(   bg=PANEL,  fg=MUTED)

    def _show_csv(self):
        self._lc.itemconfig(self._manual_win, state="hidden")
        self._lc.itemconfig(self._csv_win,    state="normal")
        self._lc.configure(scrollregion=self._lc.bbox("all"))
        self._tab_csv_btn.config(   bg=BLUE,  fg=BG)
        self._tab_manual_btn.config(bg=PANEL, fg=MUTED)

    @staticmethod
    def _mkbtn(parent, text, bg, fg, cmd):
        return tk.Button(parent, text=text, font=(FNT,11,"bold"),
                         bg=bg, fg=fg, activebackground=bg,
                         relief="flat", cursor="hand2",
                         command=cmd, padx=14, pady=6)

    # ── Motion application ────────────────────────────────────────────────────

    def _apply_single_disp(self, motor_idx: int, disp_mm: float, speed_mms: float):
        """Apply displacement to one motor (0-indexed)."""
        self._disps[motor_idx]  = disp_mm
        self._speeds[motor_idx] = speed_mms

        self._capstan_panel.update(motor_idx + 1, disp_mm, speed_mms,
                                   target_mm=disp_mm)
        self._manual_panel.update_disp(motor_idx + 1, disp_mm)
        self._cont_viz.set_displacements(self._disps)

        robot = self.robot
        if robot and HAS_ROBOT:
            try:
                from robot_controller import DEFAULT_JOINT_CONFIG
                cfg = DEFAULT_JOINT_CONFIG[motor_idx]
                angle_deg = mm_to_deg(disp_mm, motor_idx)
                ang_spd   = mms_to_dps(speed_mms, motor_idx)
                threading.Thread(
                    target=robot.motors[cfg.motor_id].set_position,
                    kwargs={"position_deg": angle_deg,
                            "max_speed_dps": ang_spd,
                            "wait": False},
                    daemon=True).start()
            except Exception as e:
                self._log(f"HW error M{motor_idx+1}: {e}", "err")

    def _apply_state(self, state: TendonState):
        """Apply a full TendonState (all 6 motors simultaneously)."""
        targets = [m[0] for m in state.motors]
        speeds  = [m[1] for m in state.motors]

        robot = self.robot
        for i in range(6):
            self._disps[i]  = targets[i]
            self._speeds[i] = speeds[i]
            if robot and HAS_ROBOT:
                try:
                    from robot_controller import DEFAULT_JOINT_CONFIG
                    cfg       = DEFAULT_JOINT_CONFIG[i]
                    angle_deg = mm_to_deg(targets[i], i)
                    ang_spd   = mms_to_dps(speeds[i], i)
                    threading.Thread(
                        target=robot.motors[cfg.motor_id].set_position,
                        kwargs={"position_deg": angle_deg,
                                "max_speed_dps": ang_spd,
                                "wait": False},
                        daemon=True).start()
                    time.sleep(0.001)
                except Exception as e:
                    self._log(f"HW error M{i+1}: {e}", "err")

        self._capstan_panel.update_all(targets, speeds, targets)
        for i in range(6):
            self._manual_panel.update_disp(i + 1, targets[i])
        self._cont_viz.set_displacements(targets)
        self._log(f"[STATE] Applied: {state.name}")

    def _home_all(self):
        """Send all motors to home position (0 mm displacement)."""
        for i in range(6):
            self._apply_single_disp(i, 0.0, 20.0)
        self._log("[HOME] All motors sent to 0.0 mm (home position)", "ok")

    def _apply_radii(self):
        """Apply user-entered capstan radii values."""
        new_radii = []
        for i in range(6):
            try:
                r = float(self._radii_vars[i].get())
                if r <= 0:
                    messagebox.showwarning("Invalid Radius",
                        f"M{i+1}: radius must be positive.")
                    return
                new_radii.append(r)
            except ValueError:
                messagebox.showwarning("Invalid Radius",
                    f"M{i+1}: enter a valid number.")
                return
        for i in range(6):
            CAPSTAN_RADII_MM[i] = new_radii[i]
        self._log(f"[CONFIG] Capstan radii updated: "
                  + "  ".join(f"M{i+1}={CAPSTAN_RADII_MM[i]:.1f}" for i in range(6)),
                  "ok")

    # ── Connection ────────────────────────────────────────────────────────────

    def _connect(self):
        if not HAS_ROBOT:
            self._lbl_c.config(text="● VISUAL SIM", fg=YELLOW)
            self._log("Robot modules not found — visual simulation only.", "warn")
            return
        threading.Thread(target=self._connect_bg, daemon=True).start()

    def _connect_bg(self):
        try:
            self.bus   = CANBus(simulated=self._sim, channel=self._port,
                                bustype=self._bus_t, num_motors=6)
            self.robot = RobotController(self.bus, DEFAULT_JOINT_CONFIG)
            self.robot.start()
            self._connected = True
            self.after(0, self._conn_ok)
        except Exception as e:
            self.after(0, lambda: self._conn_fail(str(e)))

    def _conn_ok(self):
        self._lbl_c.config(text="● CONNECTED", fg=ACCENT)
        self._log("Connected — all 6 capstan motors enabled.", "ok")

    def _conn_fail(self, err):
        self._lbl_c.config(text="● CONN FAILED", fg=RED)
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
        if self.robot and self._connected and HAS_ROBOT:
            try:
                from robot_controller import DEFAULT_JOINT_CONFIG
                fb = self.robot.get_all_feedback()
                for i, cfg in enumerate(DEFAULT_JOINT_CONFIG):
                    f = fb.get(cfg.motor_id)
                    if f:
                        disp_mm  = deg_to_mm(f.position_deg, i)
                        speed_ms = deg_to_mm(f.velocity_dps, i)   # dps → mm/s
                        self._disps[i]  = disp_mm
                        self._speeds[i] = speed_ms
                        self._capstan_panel.update(i+1, disp_mm, speed_ms)
                        self._manual_panel.update_disp(i+1, disp_mm)
                self._cont_viz.set_displacements(self._disps)
            except Exception:
                pass
        else:
            # Simulation: update displays from current internal state
            self._capstan_panel.update_all(self._disps, self._speeds)
            for i in range(6):
                self._manual_panel.update_disp(i+1, self._disps[i])
            self._cont_viz.set_displacements(self._disps)

        self._mon_job = self.after(300, self._monitor)

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
        description="Tendon-Driven Continuum Manipulator Control GUI")
    ap.add_argument("--real",    action="store_true",
                    help="Connect to real hardware (default: simulation)")
    ap.add_argument("--port",    default="COM3",
                    help="Serial port for USB-CAN adapter")
    ap.add_argument("--bustype", default="slcan",
                    choices=["slcan","pcan","kvaser","socketcan"],
                    help="python-can bus type")
    args = ap.parse_args()
    ContinuumGUI(simulated=not args.real,
                 port=args.port,
                 bustype=args.bustype).mainloop()
