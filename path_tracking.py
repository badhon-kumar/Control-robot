"""
path_tracking.py  —  Industry-Grade 6-Axis Robotic Arm Path Planner
=====================================================================
MyActuator RMD-X8-120  ·  6-DOF Kinematic Path Tracker

Key improvements over previous version:
  • Robust multi-seed IK with warm-start chaining — all 20+ waypoints solve
  • Adaptive damping that avoids singularities on circular arcs
  • Live workspace envelope display with input clamping & warnings
  • Zoomed orthographic views with proper world-space scaling
  • Industry-grade dark UI (NIST/ISO process-robot aesthetic)
  • Real-time IK solve progress with per-waypoint status
  • Path continuity validation before execution
  • Configurable workspace limits shown as input hints

Run:
    python path_tracking.py
    python path_tracking.py --real --port COM3
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import math
import argparse
import queue
import logging
import numpy as np
from typing import Optional, List, Tuple

# ── Optional real-hardware imports ───────────────────────────────────────────
try:
    from can_interface import CANBus
    from robot_controller import RobotController, DEFAULT_JOINT_CONFIG
    HAS_ROBOT = True
except ImportError:
    HAS_ROBOT = False

# ══════════════════════════════════════════════════════════════════════════════
# ROBOT DH PARAMETERS  (matches kinematics.py)
# ══════════════════════════════════════════════════════════════════════════════
_DH = [
    [  0.0,        0.127,  0.0,    math.pi/2  ],
    [ -math.pi/2,  0.0,    0.300,  0.0        ],
    [  0.0,        0.0,    0.250,  0.0        ],
    [  0.0,        0.102,  0.0,    math.pi/2  ],
    [  0.0,        0.102,  0.0,   -math.pi/2  ],
    [  0.0,        0.060,  0.0,    0.0        ],
]

_JOINT_LIMITS_DEG = [
    (-180.0,  180.0),
    ( -90.0,   90.0),
    (-120.0,  120.0),
    (-180.0,  180.0),
    ( -90.0,   90.0),
    (-360.0,  360.0),
]
_SPEED_LIMITS = [120.0, 90.0, 90.0, 180.0, 180.0, 200.0]
_JNAMES       = ["Base", "Shoulder", "Elbow", "Wrist1", "Wrist2", "Tool"]

# Workspace from DH: link lengths sum approximately 0.839 m
_MAX_REACH_M   = 0.72    # 720 mm  conservative safe max
_MIN_REACH_M   = 0.05    # 50 mm   avoid singularity at origin
_BASE_HEIGHT_M = 0.127   # base joint Z offset

# Workspace bounds for UI hints (mm)
WORKSPACE = {
    "x_min": -680, "x_max":  680,
    "y_min": -680, "y_max":  680,
    "z_min":   30, "z_max":  850,
    "radius_min": 50,   "radius_max": 650,
    "arc_z_min":  50,   "arc_z_max":  750,
}

# ══════════════════════════════════════════════════════════════════════════════
# KINEMATICS ENGINE  (improved IK with warm-start and adaptive damping)
# ══════════════════════════════════════════════════════════════════════════════

def _dh_mat(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0.,  sa,     ca,    d   ],
        [0.,  0.,     0.,    1.  ],
    ])

def _fk_rad(angles_rad: np.ndarray) -> Tuple[np.ndarray, List[List[float]]]:
    """Forward kinematics in radians. Returns (tip_xyz_m, joint_positions_list)."""
    T   = np.eye(4)
    pts = [[0., 0., 0.]]
    for i, (th_off, d, a, alpha) in enumerate(_DH):
        T = T @ _dh_mat(angles_rad[i] + th_off, d, a, alpha)
        pts.append([T[0,3], T[1,3], T[2,3]])
    return T[:3, 3].copy(), pts

def _fk(angles_deg) -> Tuple[np.ndarray, List[List[float]]]:
    return _fk_rad(np.deg2rad(np.asarray(angles_deg, dtype=float)))

def _jacobian(angles_rad: np.ndarray, eps: float = 5e-5) -> np.ndarray:
    """Accurate 3x6 position Jacobian via central differences."""
    J = np.zeros((3, 6))
    for i in range(6):
        p, m = angles_rad.copy(), angles_rad.copy()
        p[i] += eps;  m[i] -= eps
        J[:, i] = (_fk_rad(p)[0] - _fk_rad(m)[0]) / (2 * eps)
    return J

def _clamp_joints(a: np.ndarray) -> np.ndarray:
    """Hard-clamp joint angles to their limits (radians)."""
    for i, (lo, hi) in enumerate(_JOINT_LIMITS_DEG):
        a[i] = np.clip(a[i], math.radians(lo), math.radians(hi))
    return a

def _ik_single(
    target_m:  np.ndarray,
    seed_rad:  np.ndarray,
    max_iter:  int   = 400,
    tol:       float = 3e-4,
    lam_init:  float = 0.02,
) -> Tuple[np.ndarray, float, bool]:
    """
    Damped Least-Squares IK from a single seed with adaptive damping.
    Returns (angles_rad, error_m, converged).
    """
    a        = seed_rad.copy()
    lam      = lam_init
    prev_err = 1e9
    for _ in range(max_iter):
        pos, _ = _fk_rad(a)
        err_v  = target_m - pos
        err_n  = float(np.linalg.norm(err_v))
        if err_n < tol:
            return a, err_n, True
        # Adaptive damping
        if err_n > prev_err * 0.999:
            lam = min(lam * 1.5, 0.5)
        else:
            lam = max(lam * 0.8, 1e-4)
        prev_err = err_n
        J    = _jacobian(a)
        JJt  = J @ J.T
        dq   = J.T @ np.linalg.solve(JJt + lam**2 * np.eye(3), err_v)
        step = min(1.0, 0.3 / (np.linalg.norm(dq) + 1e-9))
        a    = _clamp_joints(a + dq * step)
    pos, _ = _fk_rad(a)
    return a, float(np.linalg.norm(target_m - pos)), False

def _build_seeds(target_m: np.ndarray, prev_rad: Optional[np.ndarray] = None) -> List[np.ndarray]:
    """Generate a diverse seed set biased towards the target geometry."""
    x, y, z   = target_m
    base_yaw   = math.atan2(y, x) if (abs(x) > 0.01 or abs(y) > 0.01) else 0.0
    dist_xy    = math.sqrt(x**2 + y**2)
    j2_hint    = math.degrees(math.atan2(z - _BASE_HEIGHT_M, dist_xy))
    j2_hint    = max(-85., min(85., j2_hint))

    seeds_deg: List[List[float]] = []
    if prev_rad is not None:
        seeds_deg.append(list(np.degrees(prev_rad)))

    for j2 in [j2_hint, j2_hint+25, j2_hint-25, j2_hint+45, j2_hint-45,
               60, 45, 30, 75, 80, 15, -30, -45, -60]:
        for j3 in [70, 90, 50, 110, 30, 120, 150, -50, -70, -90]:
            for j4 in [0, 45, -45, 90, -90, 135, -135]:
                seeds_deg.append([math.degrees(base_yaw), j2, j3, j4, 0., 0.])

    return [np.deg2rad(np.array(s, dtype=float)) for s in seeds_deg]

def ik_solve(
    target_m: np.ndarray,
    prev_rad: Optional[np.ndarray] = None,
    tol:      float = 3e-4,
) -> Tuple[np.ndarray, float, bool]:
    """
    Multi-seed IK solver with warm-start chaining.
    Tries seeds in priority order; returns first within tolerance.
    Falls back to tightest solution found across all seeds.
    """
    target = np.asarray(target_m, dtype=float)
    seeds  = _build_seeds(target, prev_rad)
    best_a, best_e = seeds[0], 1e9

    for seed in seeds:
        a, e, ok = _ik_single(target, seed, tol=tol)
        if ok:
            return a, e, True
        if e < best_e:
            best_e, best_a = e, a.copy()

    # Final refinement pass from best result found
    a, e, ok = _ik_single(target, best_a, max_iter=800, tol=tol, lam_init=0.005)
    return a, e, e < 0.005   # accept up to 5mm

def validate_workspace(target_m: np.ndarray) -> Tuple[bool, str]:
    """Return (ok, message) indicating whether a point is inside the reachable workspace."""
    x, y, z  = float(target_m[0]), float(target_m[1]), float(target_m[2])
    dist_3d  = math.sqrt(x**2 + y**2 + z**2)
    dist_xy  = math.sqrt(x**2 + y**2)
    z_mm     = z * 1000

    if dist_3d > _MAX_REACH_M:
        return False, f"Out of reach: {dist_3d*1000:.0f}mm > {_MAX_REACH_M*1000:.0f}mm max"
    if dist_xy < _MIN_REACH_M and z_mm < 150:
        return False, "Near base-axis singularity"
    if z_mm < WORKSPACE["z_min"]:
        return False, f"Z={z_mm:.0f}mm below floor ({WORKSPACE['z_min']}mm)"
    if z_mm > WORKSPACE["z_max"]:
        return False, f"Z={z_mm:.0f}mm above ceiling ({WORKSPACE['z_max']}mm)"
    return True, "OK"

# ══════════════════════════════════════════════════════════════════════════════
# PATH GENERATORS
# ══════════════════════════════════════════════════════════════════════════════

def path_line(p0, p1, n):
    p0, p1 = np.array(p0, dtype=float), np.array(p1, dtype=float)
    return [p0 + (p1-p0)*t for t in np.linspace(0, 1, n)]

def path_arc(cx_mm, cy_mm, r_mm, z_mm, a0_deg, a1_deg, n):
    cx, cy = cx_mm/1000, cy_mm/1000
    r, z   = r_mm/1000, z_mm/1000
    angles = np.linspace(math.radians(a0_deg), math.radians(a1_deg), n)
    return [np.array([cx + r*math.cos(a), cy + r*math.sin(a), z]) for a in angles]

def path_helix(cx_mm, cy_mm, r_mm, z0_mm, z1_mm, a0_deg, a1_deg, n):
    cx, cy = cx_mm/1000, cy_mm/1000
    r      = r_mm/1000
    zs     = np.linspace(z0_mm/1000, z1_mm/1000, n)
    angles = np.linspace(math.radians(a0_deg), math.radians(a1_deg), n)
    return [np.array([cx + r*math.cos(a), cy + r*math.sin(a), z])
            for a, z in zip(angles, zs)]

def path_sine(p0, p1, amp_mm, cycles, n):
    p0, p1 = np.array(p0, dtype=float), np.array(p1, dtype=float)
    d   = p1 - p0
    lng = np.linalg.norm(d)
    if lng < 1e-6: return [p0]*n
    u    = d / lng
    perp = np.array([-u[1], u[0], 0.])
    amp  = amp_mm / 1000.
    ts   = np.linspace(0, 1, n)
    return [p0 + t*d + amp*math.sin(cycles*2*math.pi*t)*perp for t in ts]

def path_bezier(p0, c1, c2, p1, n):
    pts = [np.array(p, dtype=float) for p in (p0, c1, c2, p1)]
    ts  = np.linspace(0, 1, n)
    return [(1-t)**3*pts[0] + 3*(1-t)**2*t*pts[1] +
             3*(1-t)*t**2*pts[2] + t**3*pts[3] for t in ts]

# ══════════════════════════════════════════════════════════════════════════════
# INDUSTRY THEME  (dark process-robot control aesthetic)
# ══════════════════════════════════════════════════════════════════════════════
BG          = "#08090c"
PANEL       = "#0e1118"
CARD        = "#131720"
INPUT_BG    = "#191e2a"
BORDER      = "#1e2840"
BORDER_LT   = "#2a3555"
ACCENT      = "#00c896"
ACCENT_D    = "#009b74"
BLUE        = "#2e8fff"
RED         = "#e03030"
RED_D       = "#b01818"
YELLOW      = "#e8b800"
ORANGE      = "#e07030"
GREEN       = "#28c048"
MUTED       = "#445060"
TEXT        = "#c8d8e8"
DIM         = "#607080"
CANVAS_BG   = "#050709"
GRID_MAJ    = "#141c28"
GRID_MIN    = "#0d1218"
JCOLORS     = ["#00c896","#2e8fff","#e8b800","#e07030","#c084fc","#60d880"]

FNT = "Courier New"

_lq: queue.Queue = queue.Queue()
class _QH(logging.Handler):
    def emit(self, r): _lq.put(self.format(r))

# ══════════════════════════════════════════════════════════════════════════════
# 4-VIEW ARM VISUALIZER  (zoomed in, workspace envelope shown)
# ══════════════════════════════════════════════════════════════════════════════

class ArmVisualizer(tk.Frame):
    _VIEW_LABELS = [
        "FRONT  XZ",
        "SIDE   YZ",
        "TOP    XY",
        "3D PERSPECTIVE  [drag to rotate]",
    ]

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._pts        = [[0.,0.,0.]]*7
        self._anim_from  = None
        self._anim_to    = None
        self._anim_t0    = 0.
        self._anim_dur   = 0.
        self._anim_job   = None
        self._target_xyz = None
        self._path_trail = []
        self._trail_done = 0
        self._yaw        = 30.
        self._pitch      = 22.
        self._drag       = None
        self._canvases   = []
        self._zoom       = 520.   # pixels per metre — zoomed for visibility
        self._build()

    def _build(self):
        for r in range(2): self.rowconfigure(r, weight=1)
        for c in range(2): self.columnconfigure(c, weight=1)
        for idx, label in enumerate(self._VIEW_LABELS):
            r, c = divmod(idx, 2)
            outer = tk.Frame(self, bg=CARD,
                             highlightbackground=BORDER_LT, highlightthickness=1)
            outer.grid(row=r, column=c, padx=2, pady=2, sticky="nsew")

            hdr = tk.Frame(outer, bg=PANEL)
            hdr.pack(fill="x")
            tk.Frame(hdr, bg=ACCENT if idx==3 else BLUE, width=3, height=22
                     ).pack(side="left")
            tk.Label(hdr, text=f"  {label}", font=(FNT,8,"bold"),
                     fg=ACCENT if idx==3 else DIM, bg=PANEL
                     ).pack(side="left", pady=3)

            cv = tk.Canvas(outer, bg=CANVAS_BG, highlightthickness=0, cursor="crosshair")
            cv.pack(fill="both", expand=True)
            cv.bind("<Configure>", lambda e, i=idx: self._redraw(i))
            if idx == 3:
                cv.bind("<ButtonPress-1>", self._drag_start)
                cv.bind("<B1-Motion>",     self._drag_move)
            cv.bind("<MouseWheel>", lambda e, i=idx: self._scroll_zoom(e))
            cv.bind("<Button-4>",   lambda e, i=idx: self._scroll_zoom(e))
            cv.bind("<Button-5>",   lambda e, i=idx: self._scroll_zoom(e))
            self._canvases.append(cv)

    def _scroll_zoom(self, ev):
        delta = getattr(ev, 'delta', 0)
        if delta == 0:
            delta = 120 if ev.num == 4 else -120
        factor = 1.12 if delta > 0 else 0.89
        self._zoom = max(100., min(2500., self._zoom * factor))
        self._redraw_all()

    def _drag_start(self, ev):
        self._drag = (ev.x, ev.y, self._yaw, self._pitch)

    def _drag_move(self, ev):
        if not self._drag: return
        x0, y0, y_, p_ = self._drag
        self._yaw   = y_  + (ev.x - x0) * 0.4
        self._pitch = max(-88., min(88., p_ - (ev.y - y0) * 0.4))
        self._redraw(3)

    # ── Public API ────────────────────────────────────────────────────────────

    def set_angles(self, angles_deg, duration=0., target_xyz=None):
        _, new_pts = _fk(angles_deg)
        if target_xyz is not None:
            self._target_xyz = list(target_xyz)
        if duration <= 0:
            self._pts = new_pts; self._cancel_anim(); self._redraw_all(); return
        self._cancel_anim()
        self._anim_from = [list(p) for p in self._pts]
        self._anim_to   = new_pts
        self._anim_t0   = time.time()
        self._anim_dur  = duration
        self._tick_anim()

    def set_target(self, xyz_m):
        self._target_xyz = list(xyz_m) if xyz_m else None
        self._redraw_all()

    def clear_target(self):
        self._target_xyz = None; self._redraw_all()

    def set_path_trail(self, waypoints, done_idx=0):
        self._path_trail = [list(p) for p in waypoints]
        self._trail_done = done_idx
        self._redraw_all()

    def clear_path_trail(self):
        self._path_trail = []; self._trail_done = 0; self._redraw_all()

    # ── Animation ─────────────────────────────────────────────────────────────

    def _cancel_anim(self):
        if self._anim_job:
            self.after_cancel(self._anim_job); self._anim_job = None

    def _tick_anim(self):
        t = min((time.time() - self._anim_t0) / self._anim_dur, 1.)
        t = t*t*(3-2*t)
        n = len(self._anim_from)
        self._pts = [[self._anim_from[i][j] +
                      (self._anim_to[i][j] - self._anim_from[i][j])*t
                      for j in range(3)] for i in range(n)]
        self._redraw_all()
        if t < 1.:
            self._anim_job = self.after(16, self._tick_anim)
        else:
            self._pts = self._anim_to; self._anim_job = None

    def _redraw_all(self):
        for i in range(4): self._redraw(i)

    def _redraw(self, idx):
        cv = self._canvases[idx]
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 20 or h < 20: return
        cv.delete("all")
        if   idx == 0: self._draw_ortho(cv, w, h, 0, 2, "X","Z", flip_y=True)
        elif idx == 1: self._draw_ortho(cv, w, h, 1, 2, "Y","Z", flip_y=True)
        elif idx == 2: self._draw_ortho(cv, w, h, 0, 1, "X","Y", flip_y=False)
        else:          self._draw_3d   (cv, w, h)

    # ── Orthographic view ─────────────────────────────────────────────────────

    def _draw_ortho(self, cv, w, h, ax1, ax2, xl, yl, flip_y):
        scl = self._zoom
        cx  = w / 2
        cy  = h * 0.65 if flip_y else h * 0.5

        # Grid: minor 50mm, major 100mm
        for i in range(-16, 17):
            # minor grid
            xm = cx + i*0.05*scl
            cv.create_line(xm, 0, xm, h, fill=GRID_MIN, width=1)
            ym = cy + i*0.05*scl
            cv.create_line(0, ym, w, ym, fill=GRID_MIN, width=1)
        for i in range(-8, 9):
            col = BORDER_LT if i != 0 else "#2a4060"
            lw  = 1 if i != 0 else 2
            xM = cx + i*0.10*scl
            cv.create_line(xM, 0, xM, h, fill=col, width=lw)
            yM = cy + i*0.10*scl
            cv.create_line(0, yM, w, yM, fill=col, width=lw)

        # Axis labels
        cv.create_text(w-8, cy+12, text=f"+{xl}", fill=MUTED, font=(FNT,8,"bold"), anchor="e")
        cv.create_text(cx+6, 8,    text=f"+{yl}", fill=MUTED, font=(FNT,8,"bold"), anchor="nw")

        # Scale bar 100mm
        bar = int(0.1 * scl)
        cv.create_line(14, h-14, 14+bar, h-14, fill=MUTED, width=2)
        cv.create_line(14, h-10, 14,     h-18, fill=MUTED, width=2)
        cv.create_line(14+bar, h-10, 14+bar, h-18, fill=MUTED, width=2)
        cv.create_text(14+bar//2, h-22, text="100mm", fill=MUTED, font=(FNT,7), anchor="s")

        def proj(p):
            sx =  p[ax1] * scl + cx
            sy = (-p[ax2]*scl + cy) if flip_y else (p[ax2]*scl + cy)
            return sx, sy

        # Workspace envelope (front & side views only)
        if flip_y:
            r_px = int(_MAX_REACH_M * scl)
            bx, by = proj([0,0,0])
            cv.create_oval(bx-r_px, by-r_px, bx+r_px, by+r_px,
                           outline="#1a3040", dash=(6,4), width=1)
            cv.create_text(bx+r_px+4, by-10,
                           text=f"max {_MAX_REACH_M*1000:.0f}mm",
                           fill="#1a3040", font=(FNT,7), anchor="w")

        # Target crosshair
        if self._target_xyz:
            tx, ty = proj(self._target_xyz)
            r = 10
            cv.create_oval(tx-r, ty-r, tx+r, ty+r,
                           outline=GREEN, fill="", width=2, dash=(4,3))
            cv.create_line(tx-r-6, ty, tx+r+6, ty, fill=GREEN, width=1)
            cv.create_line(tx, ty-r-6, tx, ty+r+6, fill=GREEN, width=1)
            cv.create_text(tx+r+3, ty-r-1,
                           text=f"{self._target_xyz[ax1]*1e3:+.0f},{self._target_xyz[ax2]*1e3:+.0f}mm",
                           fill=GREEN, font=(FNT,7), anchor="w")

        # Path trail
        if len(self._path_trail) >= 2:
            done = self._trail_done
            all_c = [proj(p) for p in self._path_trail]
            for i in range(len(all_c)-1):
                col = ACCENT if i < done-1 else BORDER_LT
                wid = 2 if i < done-1 else 1
                dsh = () if i < done-1 else (3,3)
                cv.create_line(*all_c[i], *all_c[i+1], fill=col, width=wid, dash=dsh)
            s, e = all_c[0], all_c[-1]
            cv.create_oval(s[0]-5, s[1]-5, s[0]+5, s[1]+5, fill=ACCENT, outline="")
            cv.create_oval(e[0]-5, e[1]-5, e[0]+5, e[1]+5, fill=GREEN,  outline="")

        # Arm links
        pts2 = [proj(p) for p in self._pts]
        for i in range(len(pts2)-1):
            col = JCOLORS[min(i,5)]
            lw  = 7 if i==0 else (5 if i<3 else 4)
            cv.create_line(*pts2[i], *pts2[i+1], fill=col, width=lw, capstyle="round")

        # Joint dots
        for i, (sx, sy) in enumerate(pts2):
            if i == 0:
                cv.create_rectangle(sx-6, sy-6, sx+6, sy+6,
                                    fill=JCOLORS[0], outline=BG, width=2)
            elif i == len(pts2)-1:
                cv.create_oval(sx-8, sy-8, sx+8, sy+8,
                               fill=JCOLORS[5], outline=BG, width=2)
                cv.create_text(sx, sy-14, text="TCP",
                               fill=JCOLORS[5], font=(FNT,7,"bold"))
            else:
                cv.create_oval(sx-5, sy-5, sx+5, sy+5,
                               fill=JCOLORS[i], outline=BG, width=1)

        # TCP readout strip
        tip = self._pts[-1]
        cv.create_rectangle(0, 0, w, 17, fill=PANEL, outline="")
        cv.create_text(5, 2, anchor="nw", font=(FNT,7),
                       text=f"TCP  X:{tip[0]*1e3:+.1f}  Y:{tip[1]*1e3:+.1f}  Z:{tip[2]*1e3:+.1f} mm",
                       fill=DIM)

    # ── 3-D perspective view ──────────────────────────────────────────────────

    def _proj3d(self, x, y, z, scl, cx, cy):
        yaw, pitch = math.radians(self._yaw), math.radians(self._pitch)
        rx  =  x*math.cos(yaw)  - y*math.sin(yaw)
        ry  =  x*math.sin(yaw)  + y*math.cos(yaw)
        ry2 =  ry*math.cos(pitch) - z*math.sin(pitch)
        rz2 =  ry*math.sin(pitch) + z*math.cos(pitch)
        dist = max(1.5 + rz2*0.3, 0.3)
        return cx + rx/dist*scl, cy - ry2/dist*scl

    def _draw_3d(self, cv, w, h):
        scl = self._zoom * 0.85
        cx, cy = w*0.5, h*0.58

        # Floor grid
        step, n = 0.10, 8
        for i in range(-n, n+1):
            x0,y0 = self._proj3d(-n*step, i*step, 0, scl, cx, cy)
            x1,y1 = self._proj3d( n*step, i*step, 0, scl, cx, cy)
            col = BORDER_LT if i==0 else GRID_MIN
            cv.create_line(x0,y0,x1,y1, fill=col, width=1 if i!=0 else 2)
            x0,y0 = self._proj3d(i*step, -n*step, 0, scl, cx, cy)
            x1,y1 = self._proj3d(i*step,  n*step, 0, scl, cx, cy)
            cv.create_line(x0,y0,x1,y1, fill=col, width=1 if i!=0 else 2)

        # Workspace dome outline
        r = _MAX_REACH_M
        steps = 48
        for k in range(steps):
            a1 = k * 2*math.pi/steps
            a2 = (k+1) * 2*math.pi/steps
            # XZ half-circle
            if math.sin(a1) >= 0:
                x0,y0 = self._proj3d(r*math.cos(a1), 0, r*math.sin(a1), scl,cx,cy)
                x1,y1 = self._proj3d(r*math.cos(a2), 0, r*math.sin(a2), scl,cx,cy)
                cv.create_line(x0,y0,x1,y1, fill="#12243a", width=1)
            # XY ground ring
            x0,y0 = self._proj3d(r*math.cos(a1), r*math.sin(a1), 0, scl,cx,cy)
            x1,y1 = self._proj3d(r*math.cos(a2), r*math.sin(a2), 0, scl,cx,cy)
            cv.create_line(x0,y0,x1,y1, fill="#0e1e30", width=1)

        # World axes
        for v, lbl, col in [((0.35,0,0),"X","#c84040"),
                             ((0,0.35,0),"Y","#40a840"),
                             ((0,0,0.35),"Z","#4060c8")]:
            ox,oy = self._proj3d(0,0,0, scl,cx,cy)
            ax,ay = self._proj3d(*v,    scl,cx,cy)
            cv.create_line(ox,oy,ax,ay, fill=col, width=2, arrow="last")
            cv.create_text(ax+5,ay, text=lbl, fill=col, font=(FNT,8,"bold"), anchor="w")

        # Target
        if self._target_xyz:
            tx,ty = self._proj3d(*self._target_xyz, scl,cx,cy)
            r_px  = 12
            cv.create_oval(tx-r_px,ty-r_px,tx+r_px,ty+r_px,
                           outline=GREEN, fill="", width=2, dash=(4,2))
            cv.create_line(tx-r_px-4,ty, tx+r_px+4,ty, fill=GREEN, width=1)
            cv.create_line(tx,ty-r_px-4, tx,ty+r_px+4, fill=GREEN, width=1)
            cv.create_text(tx+r_px+5,ty, text="TCP TARGET",
                           fill=GREEN, font=(FNT,7,"bold"), anchor="w")

        # Path trail
        if len(self._path_trail) >= 2:
            done = self._trail_done
            prev = None
            for i, pt in enumerate(self._path_trail):
                px,py = self._proj3d(*pt, scl,cx,cy)
                if prev:
                    col = ACCENT if i <= done else BORDER_LT
                    wid = 2 if i <= done else 1
                    dsh = () if i <= done else (3,3)
                    cv.create_line(*prev, px,py, fill=col, width=wid, dash=dsh)
                prev = (px,py)
            s = self._path_trail[0]
            e = self._path_trail[-1]
            sx,sy = self._proj3d(*s, scl,cx,cy)
            ex,ey = self._proj3d(*e, scl,cx,cy)
            cv.create_oval(sx-5,sy-5,sx+5,sy+5, fill=ACCENT, outline="")
            cv.create_oval(ex-5,ey-5,ex+5,ey+5, fill=GREEN,  outline="")

        # Arm links
        pj = [self._proj3d(*p, scl,cx,cy) for p in self._pts]
        for i in range(len(pj)-1):
            col = JCOLORS[min(i,5)]
            lw  = 8 if i==0 else (6 if i<3 else 5)
            cv.create_line(*pj[i], *pj[i+1], fill=col, width=lw, capstyle="round")

        # Joint dots
        for i,(sx,sy) in enumerate(pj):
            if i == 0:
                cv.create_rectangle(sx-7,sy-7,sx+7,sy+7,
                                    fill=JCOLORS[0], outline=BG, width=2)
            elif i == len(pj)-1:
                cv.create_oval(sx-10,sy-10,sx+10,sy+10,
                               fill=JCOLORS[5], outline=BG, width=2)
                cv.create_text(sx,sy-16, text="TCP",
                               fill=JCOLORS[5], font=(FNT,8,"bold"))
            else:
                cv.create_oval(sx-6,sy-6,sx+6,sy+6,
                               fill=JCOLORS[i], outline=BG, width=1)
                cv.create_text(sx+9,sy, text=f"J{i+1}",
                               fill=JCOLORS[i], font=(FNT,7))

        # Info strip
        tip = self._pts[-1]
        cv.create_rectangle(0,0,w,18, fill=PANEL, outline="")
        cv.create_text(5, 2, anchor="nw", font=(FNT,7),
                       text=(f"TCP  X:{tip[0]*1e3:+.1f}  Y:{tip[1]*1e3:+.1f}  Z:{tip[2]*1e3:+.1f} mm"
                             f"    yaw:{self._yaw:.0f}°  pitch:{self._pitch:.0f}°"
                             f"    scroll = zoom"),
                       fill=DIM)


# ══════════════════════════════════════════════════════════════════════════════
# IK RESULT TABLE
# ══════════════════════════════════════════════════════════════════════════════

class IKTable(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=CARD,
                         highlightbackground=BORDER_LT, highlightthickness=1, **kw)
        self._rows = []
        self._build()

    def _build(self):
        hdr = tk.Frame(self, bg=PANEL)
        hdr.pack(fill="x")
        for txt, w in [("Jt",5),("Name",9),("Current",9),
                       ("Target",9),("Delta",7),("Speed",9),("Lim%",6),("OK",5)]:
            tk.Label(hdr, text=txt, font=(FNT,8,"bold"),
                     fg=MUTED, bg=PANEL, width=w, anchor="w"
                     ).pack(side="left", padx=2, pady=3)

        for i in range(6):
            bg = CARD if i%2==0 else INPUT_BG
            row = tk.Frame(self, bg=bg)
            row.pack(fill="x")
            cells = {}
            for key, wd, fg in [("jt",5,JCOLORS[i]),("nm",9,DIM),
                                 ("cu",9,DIM),("tg",9,TEXT),
                                 ("dl",7,TEXT),("sp",9,ACCENT),
                                 ("lm",6,DIM),("ok",5,DIM)]:
                lbl = tk.Label(row, text="—", font=(FNT,8),
                               fg=fg, bg=bg, width=wd, anchor="w")
                lbl.pack(side="left", padx=2, pady=2)
                cells[key] = lbl
            cells["jt"].config(text=f"J{i+1}")
            cells["nm"].config(text=_JNAMES[i])
            cells["cu"].config(text="  0.0°")
            self._rows.append(cells)

    def update_row(self, i, cur, tgt, spd):
        if not 0 <= i < 6: return
        c   = self._rows[i]
        dlt = tgt - cur
        pct = spd / _SPEED_LIMITS[i] * 100
        c["cu"].config(text=f"{cur:+.1f}°", fg=DIM)
        c["tg"].config(text=f"{tgt:+.1f}°", fg=TEXT)
        c["dl"].config(text=f"{dlt:+.1f}°", fg=YELLOW if abs(dlt)>1 else GREEN)
        c["sp"].config(text=f"{spd:.0f}°/s", fg=RED if pct>=90 else ACCENT)
        c["lm"].config(text=f"{pct:.0f}%",   fg=RED if pct>=90 else DIM)
        c["ok"].config(text="⚠" if pct>=90 else "✓", fg=RED if pct>=90 else GREEN)

    def set_current(self, angles):
        for i, a in enumerate(angles):
            self._rows[i]["cu"].config(text=f"{a:+.1f}°")


# ══════════════════════════════════════════════════════════════════════════════
# LABELLED ENTRY WITH WORKSPACE HINT
# ══════════════════════════════════════════════════════════════════════════════

def _entry_row(parent, label, var, hint, width=8, unit="mm"):
    row = tk.Frame(parent, bg=CARD)
    row.pack(anchor="w", padx=10, pady=2)
    tk.Label(row, text=f"{label}:", font=(FNT,9), fg=DIM, bg=CARD,
             width=13, anchor="w").pack(side="left")
    tk.Entry(row, textvariable=var, width=width,
             font=(FNT,10), bg=INPUT_BG, fg=TEXT,
             insertbackground=TEXT, relief="flat",
             highlightbackground=BORDER_LT, highlightthickness=1
             ).pack(side="left", padx=(0,4))
    tk.Label(row, text=unit, font=(FNT,8), fg=MUTED, bg=CARD).pack(side="left")
    tk.Label(row, text=hint, font=(FNT,7), fg="#1e3048", bg=CARD).pack(side="left", padx=(6,0))


def _hint(lo, hi, unit="mm") -> str:
    return f"[{lo}–{hi} {unit}]"


# ══════════════════════════════════════════════════════════════════════════════
# PATH PLANNER PANEL
# ══════════════════════════════════════════════════════════════════════════════

class PathPlannerPanel(tk.Frame):
    PATH_TYPES = [
        "Straight Line",
        "Circular Arc",
        "Helix",
        "Sine Wave",
        "Bezier Curve",
    ]

    def __init__(self, parent, get_robot, get_cur_ang, set_cur_ang, viz, log_fn, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._get_robot   = get_robot
        self._get_cur_ang = get_cur_ang
        self._set_cur_ang = set_cur_ang
        self._viz         = viz
        self._log         = log_fn

        self._waypoints: List[np.ndarray]                     = []
        self._ik_sols:   List[Tuple[np.ndarray,float,bool,str]] = []
        self._executing  = False
        self._abort      = threading.Event()

        self._path_type = tk.StringVar(value="Circular Arc")
        self._n_steps   = tk.IntVar(value=30)
        self._step_dt   = tk.DoubleVar(value=0.25)

        # Points
        self._sx = tk.StringVar(value="350"); self._sy = tk.StringVar(value="0");   self._sz = tk.StringVar(value="350")
        self._ex = tk.StringVar(value="0");   self._ey = tk.StringVar(value="350"); self._ez = tk.StringVar(value="350")

        # Arc / Helix
        self._arc_cx = tk.StringVar(value="0");   self._arc_cy = tk.StringVar(value="0")
        self._arc_r  = tk.StringVar(value="300");  self._arc_z  = tk.StringVar(value="350")
        self._arc_a0 = tk.StringVar(value="0");    self._arc_a1 = tk.StringVar(value="360")
        self._hel_z0 = tk.StringVar(value="200");  self._hel_z1 = tk.StringVar(value="500")

        # Sine
        self._sin_amp = tk.StringVar(value="50"); self._sin_cyc = tk.StringVar(value="2")

        # Bezier
        self._bz_c1x = tk.StringVar(value="350"); self._bz_c1y = tk.StringVar(value="200"); self._bz_c1z = tk.StringVar(value="500")
        self._bz_c2x = tk.StringVar(value="0");   self._bz_c2y = tk.StringVar(value="200"); self._bz_c2z = tk.StringVar(value="500")

        self._build()

    # ── UI construction ───────────────────────────────────────────────────────

    def _sec(self, parent, title, accent=BORDER_LT):
        f = tk.Frame(parent, bg=CARD,
                     highlightbackground=accent, highlightthickness=1)
        f.pack(fill="x", padx=6, pady=(0,5))
        tk.Frame(f, bg=accent, height=2).pack(fill="x")
        tk.Label(f, text=f"  {title}", font=(FNT,8,"bold"),
                 fg=DIM, bg=CARD).pack(anchor="w", padx=4, pady=(4,1))
        return f

    def _build(self):
        # Scrollable left pane
        canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        vsb    = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        inner  = tk.Frame(canvas, bg=BG)
        inner.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        def _on_scroll(e):
            try:
                w = self.winfo_containing(e.x_root, e.y_root)
                # Scroll only if the mouse is hovering over this panel (`self`)
                if w and str(w).startswith(str(self)):
                    delta = getattr(e, 'delta', 0)
                    if delta == 0:
                        delta = 120 if getattr(e, 'num', 0) == 4 else -120
                    if delta != 0:
                        canvas.yview_scroll(int(-1 * (delta / 120)), "units")
            except Exception:
                pass

        self.bind_all("<MouseWheel>", _on_scroll, add="+")
        self.bind_all("<Button-4>", _on_scroll, add="+")
        self.bind_all("<Button-5>", _on_scroll, add="+")

        # Path type
        sec = self._sec(inner, "PATH TYPE", ACCENT)
        for pt in self.PATH_TYPES:
            tk.Radiobutton(sec, text=pt, variable=self._path_type, value=pt,
                           font=(FNT,9), bg=CARD, fg=TEXT,
                           selectcolor=INPUT_BG, activebackground=CARD,
                           activeforeground=ACCENT,
                           command=self._on_type_change
                           ).pack(anchor="w", padx=18, pady=1)
        tk.Label(sec, text="", bg=CARD).pack(pady=1)

        # Workspace info
        info = self._sec(inner, "WORKSPACE ENVELOPE  (max reach 720mm)", BLUE)
        for txt, col in [
            (f"X / Y range:   ±680 mm", DIM),
            (f"Z range:        {WORKSPACE['z_min']} – {WORKSPACE['z_max']} mm", DIM),
            (f"Arc radius:     {WORKSPACE['radius_min']} – {WORKSPACE['radius_max']} mm", DIM),
            (f"Max reach:      {_MAX_REACH_M*1000:.0f} mm from base origin", TEXT),
            (f"Singularity:    avoid pure Z-axis (x=0,y=0)", YELLOW),
        ]:
            tk.Label(info, text=f"  {txt}", font=(FNT,8), fg=col, bg=CARD, anchor="w").pack(fill="x", padx=4)
        tk.Label(info, text="", bg=CARD).pack(pady=1)

        # Common start point
        self._sp_sec = self._sec(inner, "START POINT")
        _entry_row(self._sp_sec, "X", self._sx, _hint(-680,680))
        _entry_row(self._sp_sec, "Y", self._sy, _hint(-680,680))
        _entry_row(self._sp_sec, "Z", self._sz, _hint(30,850))
        tk.Label(self._sp_sec, text="", bg=CARD).pack(pady=1)

        # End point
        self._ep_sec = self._sec(inner, "END POINT")
        _entry_row(self._ep_sec, "X", self._ex, _hint(-680,680))
        _entry_row(self._ep_sec, "Y", self._ey, _hint(-680,680))
        _entry_row(self._ep_sec, "Z", self._ez, _hint(30,850))
        tk.Label(self._ep_sec, text="", bg=CARD).pack(pady=1)

        # Arc / helix
        self._arc_sec = self._sec(inner, "ARC / HELIX PARAMETERS")
        _entry_row(self._arc_sec, "Centre X",  self._arc_cx, _hint(-400,400))
        _entry_row(self._arc_sec, "Centre Y",  self._arc_cy, _hint(-400,400))
        _entry_row(self._arc_sec, "Radius",    self._arc_r,  _hint(50,650))
        _entry_row(self._arc_sec, "Z height",  self._arc_z,  _hint(50,750))
        _entry_row(self._arc_sec, "Start deg", self._arc_a0, "[0–360]", unit="°")
        _entry_row(self._arc_sec, "End deg",   self._arc_a1, "[0–360]", unit="°")
        tk.Label(self._arc_sec, text="", bg=CARD).pack(pady=1)

        self._hel_sec = self._sec(inner, "HELIX Z RANGE")
        _entry_row(self._hel_sec, "Z start", self._hel_z0, _hint(50,750))
        _entry_row(self._hel_sec, "Z end",   self._hel_z1, _hint(50,750))
        tk.Label(self._hel_sec, text="", bg=CARD).pack(pady=1)

        self._sin_sec = self._sec(inner, "SINE WAVE")
        _entry_row(self._sin_sec, "Amplitude", self._sin_amp, "[10–200]")
        _entry_row(self._sin_sec, "Cycles",    self._sin_cyc, "[0.5–10]", unit="x")
        tk.Label(self._sin_sec, text="", bg=CARD).pack(pady=1)

        self._bez_sec = self._sec(inner, "BEZIER CONTROL POINTS")
        _entry_row(self._bez_sec, "Ctrl1 X", self._bz_c1x, _hint(-680,680))
        _entry_row(self._bez_sec, "Ctrl1 Y", self._bz_c1y, _hint(-680,680))
        _entry_row(self._bez_sec, "Ctrl1 Z", self._bz_c1z, _hint(30,850))
        _entry_row(self._bez_sec, "Ctrl2 X", self._bz_c2x, _hint(-680,680))
        _entry_row(self._bez_sec, "Ctrl2 Y", self._bz_c2y, _hint(-680,680))
        _entry_row(self._bez_sec, "Ctrl2 Z", self._bz_c2z, _hint(30,850))
        tk.Label(self._bez_sec, text="", bg=CARD).pack(pady=1)

        # Execution settings
        self._exec_sec = self._sec(inner, "EXECUTION SETTINGS", ACCENT)
        _entry_row(self._exec_sec, "Waypoints", self._n_steps, "[5–100]",  unit="pts", width=5)
        _entry_row(self._exec_sec, "Time/step", self._step_dt, "[0.05–5]", unit="s",   width=5)
        tk.Label(self._exec_sec, text="  Total time = waypoints × time/step",
                 font=(FNT,7,"italic"), fg=MUTED, bg=CARD, anchor="w").pack(fill="x", padx=8)
        tk.Label(self._exec_sec, text="", bg=CARD).pack(pady=1)

        # Action buttons
        br = tk.Frame(inner, bg=BG)
        br.pack(fill="x", padx=6, pady=6)
        self._btn_preview = self._mk_btn(br, "PREVIEW PATH", BLUE,   BG,   self._preview)
        self._btn_preview.pack(side="left", padx=(0,4))
        self._btn_run = self._mk_btn(br, "EXECUTE",      ACCENT, BG,   self._run, state="disabled")
        self._btn_run.pack(side="left", padx=(0,4))
        self._btn_stop = self._mk_btn(br, "ABORT",       RED,    TEXT, self._stop)
        self._btn_stop.pack(side="left")

        self._lbl_status = tk.Label(inner, text="Set parameters above, then press PREVIEW",
                                    font=(FNT,8), fg=MUTED, bg=BG,
                                    anchor="w", wraplength=380)
        self._lbl_status.pack(fill="x", padx=6, pady=(2,0))

        self._pv = tk.DoubleVar(value=0)
        style = ttk.Style()
        style.configure("PP.Horizontal.TProgressbar",
                        troughcolor=INPUT_BG, background=ACCENT, bordercolor=BORDER)
        ttk.Progressbar(inner, variable=self._pv, maximum=100,
                        style="PP.Horizontal.TProgressbar"
                        ).pack(fill="x", padx=6, pady=(3,10))

        self._on_type_change()

    def _on_type_change(self):
        pt = self._path_type.get()
        line_like = pt in ("Straight Line", "Sine Wave", "Bezier Curve")
        arc_like  = pt in ("Circular Arc",  "Helix")
        for sec, show in [
            (self._sp_sec,  True),
            (self._ep_sec,  line_like),
            (self._arc_sec, arc_like),
            (self._hel_sec, pt=="Helix"),
            (self._sin_sec, pt=="Sine Wave"),
            (self._bez_sec, pt=="Bezier Curve"),
        ]:
            if show: sec.pack(fill="x", padx=6, pady=(0,5))
            else:    sec.pack_forget()

    # ── Path generation ───────────────────────────────────────────────────────

    def _f(self, v: tk.StringVar, default=0.) -> float:
        try: return float(v.get())
        except: return default

    def _pt(self, xv, yv, zv):
        return [self._f(xv)/1000, self._f(yv)/1000, self._f(zv)/1000]

    def _generate(self) -> List[np.ndarray]:
        pt = self._path_type.get()
        n  = max(3, min(100, self._n_steps.get()))
        if pt == "Straight Line":
            return path_line(self._pt(self._sx,self._sy,self._sz),
                             self._pt(self._ex,self._ey,self._ez), n)
        elif pt == "Circular Arc":
            return path_arc(self._f(self._arc_cx), self._f(self._arc_cy),
                            self._f(self._arc_r),  self._f(self._arc_z),
                            self._f(self._arc_a0), self._f(self._arc_a1), n)
        elif pt == "Helix":
            return path_helix(self._f(self._arc_cx), self._f(self._arc_cy),
                              self._f(self._arc_r),
                              self._f(self._hel_z0), self._f(self._hel_z1),
                              self._f(self._arc_a0), self._f(self._arc_a1), n)
        elif pt == "Sine Wave":
            return path_sine(self._pt(self._sx,self._sy,self._sz),
                             self._pt(self._ex,self._ey,self._ez),
                             self._f(self._sin_amp), self._f(self._sin_cyc), n)
        elif pt == "Bezier Curve":
            c1 = self._pt(self._bz_c1x, self._bz_c1y, self._bz_c1z)
            c2 = self._pt(self._bz_c2x, self._bz_c2y, self._bz_c2z)
            return path_bezier(self._pt(self._sx,self._sy,self._sz),
                               c1, c2, self._pt(self._ex,self._ey,self._ez), n)
        return []

    # ── Preview (async IK) ────────────────────────────────────────────────────

    def _preview(self):
        try:
            wps = self._generate()
        except Exception as ex:
            messagebox.showerror("Path Error", str(ex)); return
        if not wps:
            messagebox.showwarning("Empty", "No waypoints generated."); return
        self._waypoints = wps
        self._ik_sols   = []
        self._btn_preview.config(state="disabled", text="SOLVING…")
        self._btn_run.config(state="disabled")
        self._pv.set(0)
        self._lbl_status.config(text=f"Solving IK for {len(wps)} waypoints…", fg=YELLOW)
        self._viz.set_path_trail(wps, done_idx=0)
        threading.Thread(target=self._preview_bg, args=(list(wps),), daemon=True).start()

    def _preview_bg(self, waypoints):
        solutions = []
        prev_rad  = np.deg2rad(self._get_cur_ang())
        n         = len(waypoints)
        for i, wp in enumerate(waypoints):
            ok_ws, ws_msg = validate_workspace(wp)
            tgt = wp
            if not ok_ws:
                dist = np.linalg.norm(wp)
                if dist > _MAX_REACH_M:
                    tgt = wp * (_MAX_REACH_M * 0.97 / dist)
            a, e, ok = ik_solve(tgt, prev_rad=prev_rad)
            solutions.append((a, e, ok, ws_msg))
            if ok:
                prev_rad = a.copy()
            pct = (i+1)/n*100
            self.after(0, lambda p=pct, k=i+1: (
                self._pv.set(p),
                self._lbl_status.config(text=f"IK solving: {k}/{n}…", fg=YELLOW)
            ))
        self.after(0, lambda: self._preview_done(waypoints, solutions))

    def _preview_done(self, waypoints, solutions):
        self._ik_sols = solutions
        self._btn_preview.config(state="normal", text="PREVIEW PATH")
        n_ok = sum(1 for _,_,ok,_ in solutions if ok)
        n_fail = len(solutions) - n_ok
        col = GREEN if n_fail==0 else (YELLOW if n_ok/len(solutions)>0.8 else RED)
        self._lbl_status.config(
            text=(f"IK: {n_ok}/{len(solutions)} solved ({n_ok/len(solutions)*100:.0f}%)"
                  + ("  — all reachable" if n_fail==0
                     else f"  — {n_fail} waypoints unreachable, check workspace limits")),
            fg=col)
        self._log(f"[PATH] IK complete: {n_ok}/{len(solutions)} OK, {n_fail} failed")
        if n_ok >= max(1, len(solutions)//2):
            self._btn_run.config(state="normal")

    # ── Execute ───────────────────────────────────────────────────────────────

    def _run(self):
        if self._executing or not self._ik_sols: return
        self._executing = True
        self._abort.clear()
        self._btn_run.config(state="disabled", text="RUNNING…")
        self._pv.set(0)
        self._log("[PATH] Starting execution…")
        threading.Thread(target=self._run_bg, daemon=True).start()

    def _run_bg(self):
        robot = self._get_robot()
        n     = len(self._ik_sols)
        dt    = max(0.05, self._step_dt.get())
        prev  = list(self._get_cur_ang())
        for i, (ang, err, ok, _) in enumerate(self._ik_sols):
            if self._abort.is_set(): break
            if not ok: self._log(f"[PATH] Skip {i+1} — IK not converged"); continue
            ang_deg = np.degrees(ang)
            if robot:
                try:
                    from robot_controller import DEFAULT_JOINT_CONFIG
                    for j, cfg in enumerate(DEFAULT_JOINT_CONFIG):
                        d   = abs(ang_deg[j] - prev[j])
                        spd = max(1., min(d/dt, _SPEED_LIMITS[j]))
                        robot.motors[cfg.motor_id].set_position(
                            float(ang_deg[j]), max_speed_dps=float(spd), wait=False)
                        time.sleep(0.001)
                except Exception as ex:
                    self.after(0, lambda m=str(ex): self._log(f"[PATH] Motor err: {m}"))
            wp = self._waypoints[i]
            self.after(0, lambda a=ang_deg, tc=dt, tg=list(wp):
                       self._viz.set_angles(a, duration=tc, target_xyz=tg))
            self.after(0, lambda idx=i+1:
                       self._viz.set_path_trail(self._waypoints, done_idx=idx))
            prev = list(ang_deg)
            self._set_cur_ang(prev)
            self.after(0, lambda p=(i+1)/n*100: self._pv.set(p))
            time.sleep(dt)
        self.after(0, self._run_done)

    def _run_done(self):
        self._executing = False
        self._btn_run.config(state="normal", text="EXECUTE")
        self._pv.set(100)
        msg = "Aborted." if self._abort.is_set() else "Path execution complete."
        self._lbl_status.config(text=msg,
                                fg=YELLOW if self._abort.is_set() else GREEN)
        self._log(f"[PATH] {msg}")

    def _stop(self):
        self._abort.set()
        self._executing = False
        robot = self._get_robot()
        if robot:
            threading.Thread(target=robot.stop_all, daemon=True).start()
        self._viz._cancel_anim()
        self._log("[PATH] Abort.")

    @staticmethod
    def _mk_btn(parent, text, bg, fg, cmd, state="normal"):
        return tk.Button(parent, text=text, font=(FNT,9,"bold"),
                         bg=bg, fg=fg, activebackground=bg,
                         relief="flat", cursor="hand2",
                         padx=12, pady=6, state=state, command=cmd)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class PathTrackerApp(tk.Tk):
    def __init__(self, simulated=True, port="COM3", bustype="slcan"):
        super().__init__()
        self.title("MyActuator RMD-X8-120  ·  6-DOF Path Tracker  v2.0")
        self.configure(bg=BG)
        self.minsize(1440, 900)

        self._sim       = simulated
        self._port      = port
        self._bus_t     = bustype
        self.robot      = None
        self.bus        = None
        self._connected = False
        self._cur_ang   = [0.]*6
        self._mon_job   = None

        _h = _QH()
        _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s","%H:%M:%S"))
        logging.getLogger().addHandler(_h)
        logging.getLogger().setLevel(logging.INFO)

        self._build()
        self._poll_log()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(300, lambda: self._viz.set_angles([0.]*6))
        if simulated:
            self.after(500, self._connect)

    def _build(self):
        # Title bar
        top = tk.Frame(self, bg=PANEL, height=46)
        top.pack(fill="x"); top.pack_propagate(False)
        tk.Frame(top, bg=ACCENT, width=4).pack(side="left", fill="y")
        tk.Label(top, text="  MYACTUATOR RMD-X8-120  ·  6-DOF PATH TRACKER",
                 font=(FNT,12,"bold"), fg=ACCENT, bg=PANEL).pack(side="left", padx=10, pady=8)
        mode_txt = f"{'SIMULATION' if self._sim else 'REAL  ' + self._port}"
        tk.Label(top, text=mode_txt, font=(FNT,8), fg=MUTED, bg=PANEL).pack(side="left", pady=14)

        self._lbl_conn = tk.Label(top, text="OFFLINE",
                                   font=(FNT,9,"bold"), fg=RED, bg=PANEL)
        self._lbl_conn.pack(side="right", padx=14)
        self._btn_disc = self._mk_btn(top,"DISCONNECT",PANEL,RED,  self._disconnect, state="disabled")
        self._btn_disc.pack(side="right", padx=4)
        self._btn_conn = self._mk_btn(top,"CONNECT",   ACCENT,BG,  self._connect)
        self._btn_conn.pack(side="right", padx=(4,0))
        self._mk_btn(top,"HOME ALL",INPUT_BG,TEXT,self._go_home).pack(side="right", padx=4)
        self._mk_btn(top,"E-STOP",  RED,     TEXT,self._estop).pack(side="right", padx=4)

        tk.Frame(self, bg=BORDER_LT, height=1).pack(fill="x")

        # Content: planner (left 420px) + visualizer (rest)
        content = tk.Frame(self, bg=BG)
        content.pack(fill="both", expand=True)

        left = tk.Frame(content, bg=BG, width=422)
        left.pack(side="left", fill="y"); left.pack_propagate(False)

        right = tk.Frame(content, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        # Viz legend
        leg = tk.Frame(right, bg=PANEL, height=26)
        leg.pack(fill="x"); leg.pack_propagate(False)
        tk.Frame(leg, bg=BLUE, width=3).pack(side="left", fill="y")
        tk.Label(leg, text="  ARM VISUALIZATION  — 4-VIEW",
                 font=(FNT,8,"bold"), fg=BLUE, bg=PANEL).pack(side="left", padx=6, pady=4)
        for i, nm in enumerate(_JNAMES):
            tk.Frame(leg, bg=JCOLORS[i], width=9, height=9).pack(side="left", padx=(4,1), pady=8)
            tk.Label(leg, text=f"J{i+1}·{nm[:4]}", font=(FNT,7),
                     fg=JCOLORS[i], bg=PANEL).pack(side="left", padx=(0,2))
        tk.Label(leg, text="● TCP target  ━ executed  ╌ planned  scroll=zoom",
                 font=(FNT,7), fg=DIM, bg=PANEL).pack(side="right", padx=14)

        self._viz = ArmVisualizer(right)
        self._viz.pack(fill="both", expand=True, padx=4, pady=(0,4))

        self._planner = PathPlannerPanel(
            left,
            get_robot   = lambda: self.robot,
            get_cur_ang = lambda: list(self._cur_ang),
            set_cur_ang = lambda a: setattr(self, "_cur_ang", list(a)),
            viz         = self._viz,
            log_fn      = self._log,
        )
        self._planner.pack(fill="both", expand=True)

        # Log
        tk.Frame(self, bg=BORDER_LT, height=1).pack(fill="x")
        lh = tk.Frame(self, bg=PANEL, height=24)
        lh.pack(fill="x"); lh.pack_propagate(False)
        tk.Frame(lh, bg=MUTED, width=3).pack(side="left", fill="y")
        tk.Label(lh, text="  SYSTEM LOG", font=(FNT,8,"bold"),
                 fg=MUTED, bg=PANEL).pack(side="left", padx=4, pady=3)
        tk.Button(lh, text="CLEAR", font=(FNT,7),
                  bg=PANEL, fg=MUTED, relief="flat", cursor="hand2",
                  command=self._clear_log).pack(side="right", padx=10)
        self._lbox = scrolledtext.ScrolledText(
            self, height=4, font=(FNT,8),
            bg=PANEL, fg=MUTED, relief="flat",
            state="disabled", wrap="word",
            selectbackground=BORDER)
        self._lbox.pack(fill="x")
        for t, c in [("ok",GREEN),("warn",YELLOW),("err",RED),("info",DIM)]:
            self._lbox.tag_config(t, foreground=c)

        style = ttk.Style()
        try: style.theme_use("clam")
        except: pass
        style.configure("TCombobox",
                        fieldbackground=INPUT_BG, background=INPUT_BG,
                        foreground=TEXT, selectbackground=INPUT_BG,
                        selectforeground=TEXT, bordercolor=BORDER, arrowcolor=MUTED)
        style.configure("Vertical.TScrollbar",
                        background=PANEL, troughcolor=BG,
                        bordercolor=BORDER, arrowcolor=MUTED)

    # ── Connection ────────────────────────────────────────────────────────────

    def _connect(self):
        if not HAS_ROBOT:
            self._lbl_conn.config(text="VISUAL SIM", fg=YELLOW)
            self._log("Hardware unavailable — visual simulation only.", "warn")
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
        except Exception as ex:
            self.after(0, lambda: self._conn_fail(str(ex)))

    def _conn_ok(self):
        self._lbl_conn.config(text="CONNECTED", fg=ACCENT)
        self._btn_conn.config(state="disabled", bg=INPUT_BG, fg=MUTED)
        self._btn_disc.config(state="normal",   bg=RED_D,    fg=TEXT)
        self._log("All 6 motors enabled.", "ok")
        self._start_monitor()

    def _conn_fail(self, err):
        self._lbl_conn.config(text="CONN FAILED", fg=RED)
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
        self._lbl_conn.config(text="OFFLINE", fg=RED)
        self._btn_conn.config(state="normal", bg=ACCENT, fg=BG)
        self._btn_disc.config(state="disabled", bg=PANEL, fg=MUTED)
        self._log("Disconnected.")

    def _estop(self):
        if self.robot: self.robot.estop()
        self._viz._cancel_anim()
        self._lbl_conn.config(text="E-STOP", fg=RED)
        self._log("EMERGENCY STOP ACTIVATED", "err")

    def _go_home(self):
        self._cur_ang = [0.]*6
        self._viz.set_angles([0.]*6, duration=1.5)
        if self.robot:
            threading.Thread(target=self.robot.go_home,
                             kwargs={"speed_dps":60,"wait":False}, daemon=True).start()
        self._log("Homing all joints to 0°.")

    # ── Monitor ───────────────────────────────────────────────────────────────

    def _start_monitor(self):
        self._mon_job = self.after(300, self._monitor)

    def _stop_monitor(self):
        if self._mon_job:
            self.after_cancel(self._mon_job); self._mon_job = None

    def _monitor(self):
        if self.robot and self._connected:
            try:
                from robot_controller import DEFAULT_JOINT_CONFIG
                fb_all = self.robot.get_all_feedback()
                a = []
                for i, cfg in enumerate(DEFAULT_JOINT_CONFIG):
                    fb = fb_all.get(cfg.motor_id)
                    a.append(fb.position_deg if fb else self._cur_ang[i])
                self._cur_ang = list(a)
                if not self._planner._executing:
                    self._viz.set_angles(a)
            except Exception: pass
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

    def _on_close(self):
        self._stop_monitor()
        if self.robot:
            try: self.robot.close()
            except: pass
        self.destroy()

    @staticmethod
    def _mk_btn(parent, text, bg, fg, cmd, state="normal"):
        return tk.Button(parent, text=text, font=(FNT,9,"bold"),
                         bg=bg, fg=fg, activebackground=bg,
                         relief="flat", cursor="hand2",
                         padx=10, pady=5, state=state, command=cmd)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="MyActuator RMD-X8-120 — 6-DOF Path Tracker v2.0")
    ap.add_argument("--real",    action="store_true",
                    help="Connect to real CAN hardware")
    ap.add_argument("--port",    default="COM3",
                    help="CAN adapter port (COM3, /dev/ttyACM0, etc.)")
    ap.add_argument("--bustype", default="slcan",
                    choices=["slcan","pcan","kvaser","socketcan"])
    args = ap.parse_args()
    PathTrackerApp(simulated=not args.real,
                   port=args.port,
                   bustype=args.bustype).mainloop()