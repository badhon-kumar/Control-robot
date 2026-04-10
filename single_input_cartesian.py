"""
single_input_cartesian_gui.py
==============================
Enter a target X/Y/Z position → IK solves the 6 joint angles →
the arm animates smoothly to the new pose in a live 3-D canvas view.

Views rendered simultaneously:
  • Front  (XZ plane)
  • Side   (YZ plane)
  • Top    (XY plane)
  • 3-D    (perspective — drag to rotate)

Run:
    python single_input_cartesian_gui.py              ← simulation / visual only
    python single_input_cartesian_gui.py --real --port COM3
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
from typing import Optional

# ── Optional real-hardware imports ────────────────────────────────────────────
try:
    from can_interface import CANBus
    from robot_controller import RobotController, DEFAULT_JOINT_CONFIG
    HAS_ROBOT = True
except ImportError:
    HAS_ROBOT = False

# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE KINEMATICS (PURE RADIAN + MULTI-SEED)
# ══════════════════════════════════════════════════════════════════════════════
# The columns are: [theta_offset, d (Z-offset), a (X-length), alpha (twist)]
_DH_PARAMS = [
    [  0.0,        0.100,  0.0,    math.pi/2  ],  # Joint 1 — Base
    [ -math.pi/2,  0.0,    0.060,  0.0        ],  # Joint 2 — Shoulder
    [  0.0,        0.0,    0.100,  0.0        ],  # Joint 3 — Elbow
    [  0.0,        0.100,  0.0,    math.pi/2  ],  # Joint 4 — Wrist1
    [  0.0,        0.100,  0.0,   -math.pi/2  ],  # Joint 5 — Wrist2
    [  0.0,        0.100,  0.0,    0.0        ],  # Joint 6 — Tool
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


def _dh(theta, d, a, alpha):
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct,  -st*ca,  st*sa,  a*ct],
        [st,   ct*ca, -ct*sa,  a*st],
        [0.0,  sa,     ca,     d   ],
        [0.0,  0.0,    0.0,    1.0 ],
    ])

def _fk_rad(angles_rad):
    """FK operating entirely in radians. Returns (tip_xyz_m, joint_pts_list)."""
    T   = np.eye(4)
    pts = [[0.0, 0.0, 0.0]]
    for i, (th_off, d, a, alpha) in enumerate(_DH_PARAMS):
        T = T @ _dh(angles_rad[i] + th_off, d, a, alpha)
        pts.append([T[0,3], T[1,3], T[2,3]])
    return T[:3, 3].copy(), pts

def _fk(angles_deg):
    """Forward kinematics from degrees. Returns (tip_xyz_m, joint_pts_list)."""
    return _fk_rad(np.deg2rad(angles_deg))

def _jacobian_rad(angles_rad, eps=1e-4):
    """3×6 position Jacobian, angles in radians, result in m/rad."""
    base = _fk_rad(angles_rad)[0]
    J = np.zeros((3, 6))
    for i in range(6):
        p = np.array(angles_rad, dtype=float)
        p[i] += eps
        J[:, i] = (_fk_rad(p)[0] - base) / eps
    return J

def _ik(target_m, start_deg=None, max_iter=300, tol=5e-4):
    """
    Damped least-squares IK with multi-seed restart.
    Works entirely in radians to avoid unit-mixing bugs.
    Returns (angles_deg, error_m, success).
    """
    tgt = np.array(target_m, dtype=float)
    lim_rad = [(math.radians(lo), math.radians(hi)) for lo, hi in _JOINT_LIMITS_DEG]

    # Geometry hint: rotate base to face the target horizontally
    x, y, _ = tgt
    base_yaw = math.degrees(math.atan2(y, x)) if (abs(x) > 0.01 or abs(y) > 0.01) else 0.0

    # Rich seed set — varying shoulder, elbow, and wrist-pitch.
    seeds_deg = []
    if start_deg is not None:
        seeds_deg.append(list(start_deg))
    for j2 in [60, 45, 75, 30, 80]:
        for j3 in [80, 60, 100, 40, 110]:
            for j4 in [0, 45, -45, 90, -90, 135, -135]:
                seeds_deg.append([base_yaw, j2, j3, j4, 0.0, 0.0])
    
    # Negative-elbow configs (arm bent the other way)
    for j2 in [-45, -60, -30]:
        for j3 in [-80, -60, -100]:
            for j4 in [0, 90, -90]:
                seeds_deg.append([base_yaw, j2, j3, j4, 0.0, 0.0])

    best_angles_deg = np.zeros(6)
    best_err = 1e9
    lam = 0.05

    for sd in seeds_deg:
        a = np.deg2rad(np.array(sd, dtype=float))
        for _ in range(max_iter):
            pos, _ = _fk_rad(a)
            err    = tgt - pos
            n      = float(np.linalg.norm(err))
            if n < tol:
                return np.degrees(a), n, True
            J   = _jacobian_rad(a)
            JJt = J @ J.T
            dq  = J.T @ np.linalg.solve(JJt + lam**2 * np.eye(3), err)
            a   = a + dq * 0.5
            for i in range(6):
                lo, hi = lim_rad[i]
                a[i] = max(lo, min(hi, a[i]))
        
        pos, _ = _fk_rad(a)
        e = float(np.linalg.norm(tgt - pos))
        if e < best_err:
            best_err = e
            best_angles_deg = np.degrees(a).copy()

    return best_angles_deg, best_err, best_err < 0.003


# ══════════════════════════════════════════════════════════════════════════════
# THEME
# ══════════════════════════════════════════════════════════════════════════════
BG        = "#0a0e14"
PANEL     = "#111720"
CARD      = "#161d27"
INPUT     = "#1c2535"
BORDER    = "#243044"
ACCENT    = "#00e5c0"
BLUE      = "#3d9df5"
RED       = "#f54d4d"
YELLOW    = "#f5c542"
GREEN     = "#40c057"
ORANGE    = "#f07830"
MUTED     = "#5a6a80"
TEXT      = "#d8e4f0"
DIM       = "#8898aa"
CANVAS_BG = "#070c12"
FNT       = "Courier New"
JCOLORS   = ["#00e5c0","#3d9df5","#f5c542","#f07830","#c084fc","#74d97a"]

# ══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════════════════
_lq: queue.Queue = queue.Queue()

class _QH(logging.Handler):
    def emit(self, r): _lq.put(self.format(r))


# ══════════════════════════════════════════════════════════════════════════════
# 3-D ARM VISUALIZER
# ══════════════════════════════════════════════════════════════════════════════

class ArmVisualizer(tk.Frame):

    _VIEW_LABELS = [
        "FRONT  (X – Z)",
        "SIDE   (Y – Z)",
        "TOP    (X – Y)",
        "3-D  PERSPECTIVE  (drag to rotate)",
    ]

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._pts        = [[0.0, 0.0, 0.0]] * 7
        self._anim_from  = None
        self._anim_to    = None
        self._anim_t0    = 0.0
        self._anim_dur   = 0.0
        self._anim_job   = None
        self._target_xyz = None      # green crosshair in world space (metres)
        self._yaw        = 35.0
        self._pitch      = 25.0
        self._drag       = None
        self._canvases   = []
        self._build()

    def _build(self):
        for r in range(2): self.rowconfigure(r, weight=1)
        for c in range(2): self.columnconfigure(c, weight=1)

        for idx, label in enumerate(self._VIEW_LABELS):
            r, c = divmod(idx, 2)
            outer = tk.Frame(self, bg=CARD,
                             highlightbackground=BORDER, highlightthickness=1)
            outer.grid(row=r, column=c, padx=3, pady=3, sticky="nsew")

            hbar = tk.Frame(outer, bg=PANEL)
            hbar.pack(fill="x")
            clr = ACCENT if idx == 3 else DIM
            tk.Label(hbar, text=label, font=(FNT, 8, "bold"),
                     fg=clr, bg=PANEL).pack(side="left", padx=8, pady=3)

            cv = tk.Canvas(outer, bg=CANVAS_BG, highlightthickness=0)
            cv.pack(fill="both", expand=True)
            cv.bind("<Configure>", lambda e, i=idx: self._redraw(i))

            if idx == 3:
                cv.bind("<ButtonPress-1>",  self._drag_start)
                cv.bind("<B1-Motion>",       self._drag_move)

            self._canvases.append(cv)

    def _drag_start(self, ev):
        self._drag = (ev.x, ev.y, self._yaw, self._pitch)

    def _drag_move(self, ev):
        if not self._drag: return
        x0, y0, yaw0, pit0 = self._drag
        self._yaw   = yaw0   + (ev.x - x0) * 0.45
        self._pitch = max(-89.0, min(89.0, pit0 - (ev.y - y0) * 0.45))
        self._redraw(3)

    def set_angles(self, angles_deg, duration=0.0, target_xyz=None):
        _, new_pts = _fk(angles_deg)
        if target_xyz is not None:
            self._target_xyz = list(target_xyz)
        if duration <= 0:
            self._pts = new_pts
            self._cancel_anim()
            self._redraw_all()
            return
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
        self._target_xyz = None
        self._redraw_all()

    def _cancel_anim(self):
        if self._anim_job:
            self.after_cancel(self._anim_job)
            self._anim_job = None

    def _tick_anim(self):
        elapsed = time.time() - self._anim_t0
        t = min(elapsed / self._anim_dur, 1.0)
        t = t * t * (3 - 2 * t)            # smooth step
        self._pts = [
            [self._anim_from[i][j] +
             (self._anim_to[i][j] - self._anim_from[i][j]) * t
             for j in range(3)]
            for i in range(len(self._anim_from))
        ]
        self._redraw_all()
        if t < 1.0:
            self._anim_job = self.after(16, self._tick_anim)
        else:
            self._pts = self._anim_to
            self._anim_job = None

    def _redraw_all(self):
        for i in range(4): self._redraw(i)

    def _redraw(self, idx):
        cv = self._canvases[idx]
        w = cv.winfo_width()
        h = cv.winfo_height()
        if w < 20 or h < 20: return
        cv.delete("all")
        if   idx == 0: self._draw_ortho(cv, w, h, 0, 2, "X", "Z", True)
        elif idx == 1: self._draw_ortho(cv, w, h, 1, 2, "Y", "Z", True)
        elif idx == 2: self._draw_ortho(cv, w, h, 0, 1, "X", "Y", False)
        else:          self._draw_3d(cv, w, h)

    def _draw_ortho(self, cv, w, h, ax1, ax2, xl, yl, flip_y):
        mg  = 32
        scl = min(w - 2*mg, h - 2*mg) / 0.90
        cx, cy = w / 2, h / 2

        step = 0.10
        n    = 5
        for i in range(-n, n+1):
            col = "#1e2d40" if i != 0 else "#253548"
            cv.create_line(cx + i*step*scl, 0, cx + i*step*scl, h, fill=col)
            cv.create_line(0, cy + i*step*scl, w, cy + i*step*scl, fill=col)

        cv.create_text(w-6, cy, text=f"+{xl}", fill=MUTED, font=(FNT, 8), anchor="e")
        cv.create_text(cx,  6,  text=f"+{yl}", fill=MUTED, font=(FNT, 8), anchor="n")

        def proj(p):
            sx = p[ax1] * scl + cx
            sy = ((-p[ax2]) if flip_y else p[ax2]) * scl + cy
            return sx, sy

        if self._target_xyz:
            tx, ty = proj(self._target_xyz)
            r = 8
            cv.create_oval(tx-r, ty-r, tx+r, ty+r,
                           outline=GREEN, fill="", width=2, dash=(4, 3))
            cv.create_line(tx-r-4, ty, tx+r+4, ty, fill=GREEN, width=1)
            cv.create_line(tx, ty-r-4, tx, ty+r+4, fill=GREEN, width=1)

        pts2 = [proj(p) for p in self._pts]
        for i in range(len(pts2)-1):
            col = JCOLORS[min(i, 5)]
            lw  = 5 if i == 0 else (4 if i < 4 else 3)
            cv.create_line(*pts2[i], *pts2[i+1],
                           fill=col, width=lw, capstyle="round")

        for i, (sx, sy) in enumerate(pts2):
            if i == 0:
                s = 5
                cv.create_rectangle(sx-s, sy-s, sx+s, sy+s,
                                    fill=JCOLORS[0], outline=CANVAS_BG)
            elif i == len(pts2)-1:
                r = 8
                cv.create_oval(sx-r, sy-r, sx+r, sy+r,
                               fill=JCOLORS[5], outline=CANVAS_BG, width=2)
            else:
                r = 4
                cv.create_oval(sx-r, sy-r, sx+r, sy+r,
                               fill=JCOLORS[i], outline=CANVAS_BG)

        tip = self._pts[-1]
        cv.create_text(6, h-6,
                       text=f"X:{tip[0]*1000:+.0f}  Y:{tip[1]*1000:+.0f}  Z:{tip[2]*1000:+.0f} mm",
                       fill=DIM, font=(FNT, 7), anchor="sw")

    def _proj3d(self, x, y, z, scl, cx, cy):
        yaw   = math.radians(self._yaw)
        pitch = math.radians(self._pitch)
        rx =  x * math.cos(yaw) - y * math.sin(yaw)
        ry =  x * math.sin(yaw) + y * math.cos(yaw)
        ry2 =  ry * math.cos(pitch) - z * math.sin(pitch)
        rz2 =  ry * math.sin(pitch) + z * math.cos(pitch)
        dist = max(1.8 + rz2 * 0.35, 0.3)
        return cx + rx/dist*scl, cy - ry2/dist*scl

    def _draw_3d(self, cv, w, h):
        mg  = 36
        scl = min(w-2*mg, h-2*mg) / 0.55
        cx, cy = w*0.5, h*0.52

        step, n = 0.10, 4
        for i in range(-n, n+1):
            x0, y0 = self._proj3d(-n*step, i*step, 0, scl, cx, cy)
            x1, y1 = self._proj3d( n*step, i*step, 0, scl, cx, cy)
            cv.create_line(x0, y0, x1, y1, fill=BORDER)
            x0, y0 = self._proj3d(i*step, -n*step, 0, scl, cx, cy)
            x1, y1 = self._proj3d(i*step,  n*step, 0, scl, cx, cy)
            cv.create_line(x0, y0, x1, y1, fill=BORDER)

        for v, lbl, col in [((0.3,0,0),"X","#c04040"),
                             ((0,0.3,0),"Y","#40c040"),
                             ((0,0,0.3),"Z","#4040ff")]:
            ox, oy = self._proj3d(0,0,0, scl, cx, cy)
            ax, ay = self._proj3d(*v,    scl, cx, cy)
            cv.create_line(ox,oy,ax,ay, fill=col, width=2, arrow="last")
            cv.create_text(ax+4, ay, text=lbl, fill=col, font=(FNT,8,"bold"))

        if self._target_xyz:
            tx, ty = self._proj3d(*self._target_xyz, scl, cx, cy)
            r = 10
            cv.create_oval(tx-r, ty-r, tx+r, ty+r,
                           outline=GREEN, fill="", width=2, dash=(4,2))
            cv.create_text(tx+r+4, ty, text="TARGET",
                           fill=GREEN, font=(FNT,7), anchor="w")

        proj = [self._proj3d(p[0],p[1],p[2], scl, cx, cy) for p in self._pts]
        for i in range(len(proj)-1):
            col = JCOLORS[min(i,5)]
            lw  = 5 if i==0 else (4 if i<4 else 3)
            cv.create_line(*proj[i], *proj[i+1],
                           fill=col,      width=lw,   capstyle="round")

        for i, (sx, sy) in enumerate(proj):
            if i == 0:
                r = 7
                cv.create_oval(sx-r,sy-r,sx+r,sy+r,
                               fill=JCOLORS[0], outline=CANVAS_BG, width=2)
            elif i == len(proj)-1:
                r = 9
                cv.create_oval(sx-r,sy-r,sx+r,sy+r,
                               fill=JCOLORS[5], outline=CANVAS_BG, width=2)
                cv.create_text(sx, sy-r-5, text="TIP",
                               fill=JCOLORS[5], font=(FNT,7,"bold"))
            else:
                r = 5
                cv.create_oval(sx-r,sy-r,sx+r,sy+r,
                               fill=JCOLORS[i], outline=CANVAS_BG)
                cv.create_text(sx+r+3, sy, text=f"M{i+1}",
                               fill=JCOLORS[i], font=(FNT,7))

        tip = self._pts[-1]
        cv.create_text(6, h-6,
                       text=f"TIP  X:{tip[0]*1000:+.0f}  Y:{tip[1]*1000:+.0f}  Z:{tip[2]*1000:+.0f} mm",
                       fill=DIM, font=(FNT, 7), anchor="sw")
        cv.create_text(w-6, 6,
                       text=f"yaw {self._yaw:.0f}°  pitch {self._pitch:.0f}°",
                       fill=MUTED, font=(FNT, 7), anchor="ne")


# ══════════════════════════════════════════════════════════════════════════════
# IK RESULT TABLE
# ══════════════════════════════════════════════════════════════════════════════

class IKTable(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=CARD,
                         highlightbackground=BORDER, highlightthickness=1, **kw)
        self._rows = []
        self._build()

    def _build(self):
        hdr = tk.Frame(self, bg=PANEL)
        hdr.pack(fill="x")
        for txt, w in [("Motor",9),("Name",10),("Current°",10),
                       ("Target°",10),("Δ°",8),("Speed°/s",11),("Status",8)]:
            tk.Label(hdr, text=txt, font=(FNT,8,"bold"),
                     fg=MUTED, bg=PANEL, width=w, anchor="w"
                     ).pack(side="left", padx=3, pady=3)

        for i in range(6):
            bg  = CARD if i%2==0 else INPUT
            row = tk.Frame(self, bg=bg)
            row.pack(fill="x")
            cells = {}
            defs  = [("motor",9,JCOLORS[i]),("name",10,DIM),
                     ("cur",10,DIM),("tgt",10,TEXT),
                     ("dlt",8,TEXT),("spd",11,ACCENT),("sts",8,DIM)]
            for key, w, fg in defs:
                lbl = tk.Label(row, text="—", font=(FNT,9),
                               fg=fg, bg=bg, width=w, anchor="w")
                lbl.pack(side="left", padx=3, pady=2)
                cells[key] = lbl
            cells["motor"].config(text=f"Motor {i+1}")
            cells["name"].config(text=_JNAMES[i])
            cells["cur"].config(text="  0.0°")
            self._rows.append(cells)

    def update_row(self, i, cur_deg, tgt_deg, spd):
        if not 0 <= i < 6: return
        c    = self._rows[i]
        dlt  = tgt_deg - cur_deg
        clmp = spd >= _SPEED_LIMITS[i] * 0.95
        c["cur"].config(text=f"{cur_deg:+.1f}°",  fg=DIM)
        c["tgt"].config(text=f"{tgt_deg:+.1f}°",  fg=TEXT)
        c["dlt"].config(text=f"{dlt:+.1f}°",
                        fg=YELLOW if abs(dlt)>0.5 else GREEN)
        c["spd"].config(text=f"{spd:.1f}°/s",
                        fg=RED if clmp else ACCENT)
        c["sts"].config(text="⚠CLAMP" if clmp else "✓ OK",
                        fg=RED if clmp else GREEN)

    def set_current(self, angles):
        for i, a in enumerate(angles):
            self._rows[i]["cur"].config(text=f"{a:+.1f}°")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class CartesianGUI(tk.Tk):
    def __init__(self, simulated=True, port="COM3", bustype="slcan"):
        super().__init__()
        self.title("6-Axis Robot  ·  Cartesian Input + Arm Visualizer")
        self.configure(bg=BG)
        self.minsize(1300, 880)

        self._sim       = simulated
        self._port      = port
        self._bus_t     = bustype
        self.robot      = None
        self.bus        = None
        self._connected = False
        self._cur_ang   = [0.0] * 6
        self._ik_ang    = None
        self._ik_err    = None
        self._executing = False
        self._mon_job   = None

        _h = _QH()
        _h.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(_h)
        logging.getLogger().setLevel(logging.INFO)

        self._build()
        self._poll_log()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(200, lambda: self._viz.set_angles([0.0]*6))

        if simulated:
            self.after(400, self._connect)

    def _build(self):
        top = tk.Frame(self, bg=PANEL, height=52)
        top.pack(fill="x"); top.pack_propagate(False)
        tk.Label(top, text="⬡  6-AXIS CARTESIAN CONTROL  +  ARM VISUALIZER",
                 font=(FNT,13,"bold"), fg=ACCENT, bg=PANEL
                 ).pack(side="left", padx=18, pady=12)
        self._lbl_c = tk.Label(top, text="● DISCONNECTED",
                                font=(FNT,9,"bold"), fg=RED, bg=PANEL)
        self._lbl_c.pack(side="right", padx=16)
        mode = "SIMULATION" if self._sim else f"REAL · {self._port}"
        tk.Label(top, text=mode, font=(FNT,8), fg=MUTED, bg=PANEL
                 ).pack(side="right", padx=4)
        self._btn(top,"DISCONNECT",PANEL,RED,  self._disconnect).pack(side="right",padx=4)
        self._btn(top,"CONNECT",   ACCENT,BG,  self._connect).pack(side="right",padx=4)

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg=BG, width=430)
        left.pack(side="left", fill="y", padx=(10,5), pady=8)
        left.pack_propagate(False)
        self._build_controls(left)

        right = tk.Frame(body, bg=BG)
        right.pack(side="right", fill="both", expand=True, padx=(5,10), pady=8)
        self._build_viz_panel(right)

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        lh = tk.Frame(self, bg=PANEL); lh.pack(fill="x")
        tk.Label(lh, text="SYSTEM LOG", font=(FNT,8),
                 fg=MUTED, bg=PANEL, pady=3).pack(side="left", padx=12)
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

    def _build_controls(self, parent):
        card = tk.Frame(parent, bg=CARD,
                        highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill="x", pady=(0,8))

        tk.Label(card, text="TARGET POSITION  (mm from base origin)",
                 font=(FNT,9,"bold"), fg=ACCENT, bg=CARD
                 ).pack(anchor="w", padx=12, pady=(10,6))

        xyz_f = tk.Frame(card, bg=CARD); xyz_f.pack(pady=(0,8))
        
        # Reachable Defaults!
        self._xv = tk.StringVar(value="300.0")
        self._yv = tk.StringVar(value="0.0")
        self._zv = tk.StringVar(value="400.0")

        for ax, var, col in [("X",self._xv,ACCENT),
                              ("Y",self._yv,BLUE),
                              ("Z",self._zv,YELLOW)]:
            cf = tk.Frame(xyz_f, bg=CARD); cf.pack(side="left", padx=14)
            tk.Label(cf, text=ax, font=(FNT,20,"bold"),
                     fg=col, bg=CARD).pack()
            e = tk.Entry(cf, textvariable=var, width=8,
                         font=(FNT,11), bg=INPUT, fg=TEXT,
                         insertbackground=TEXT, relief="flat",
                         highlightbackground=BORDER, highlightthickness=1)
            e.pack()
            e.bind("<Return>", lambda _: self._solve())
            tk.Label(cf, text="mm", font=(FNT,8), fg=MUTED, bg=CARD).pack()

        tr = tk.Frame(card, bg=CARD); tr.pack(padx=12, pady=(0,10))
        
        # Adding constraints label
        tk.Label(card, text="Max reach is a 460mm sphere centered at X=0, Y=0, Z=100mm",
                 font=(FNT,8,"italic"), fg=YELLOW, bg=CARD
                 ).pack(anchor="w", padx=12, pady=(0,10))

        tk.Label(tr, text="Motion time:", font=(FNT,9), fg=MUTED, bg=CARD
                 ).pack(side="left")
        self._tv = tk.StringVar(value="3.0")
        tk.Entry(tr, textvariable=self._tv, width=6,
                 font=(FNT,10), bg=INPUT, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 highlightbackground=BORDER, highlightthickness=1
                 ).pack(side="left", padx=6)
        tk.Label(tr, text="seconds", font=(FNT,9), fg=MUTED, bg=CARD
                 ).pack(side="left")

        br = tk.Frame(parent, bg=BG); br.pack(fill="x", pady=(0,6))
        self._bsolve = self._btn(br,"🔍  SOLVE IK",BLUE, BG,   self._solve)
        self._bsolve.pack(side="left", padx=(0,5))
        self._bexec  = self._btn(br,"▶  EXECUTE",  ACCENT,BG,  self._execute,
                                 state="disabled")
        self._bexec.pack(side="left", padx=(0,5))
        self._btn(br,"■  STOP",  RED,   TEXT, self._stop).pack(side="left",padx=(0,5))
        self._btn(br,"HOME",     PANEL, MUTED,self._home).pack(side="left")

        self._lbl_ik = tk.Label(parent, text="Enter X/Y/Z and press SOLVE IK  ·  then EXECUTE",
                                font=(FNT,9), fg=MUTED, bg=BG, anchor="w", wraplength=410)
        self._lbl_ik.pack(fill="x", pady=(0,3))

        self._pv = tk.DoubleVar(value=0)
        style = ttk.Style()
        style.configure("C.Horizontal.TProgressbar",
                        troughcolor=INPUT, background=ACCENT, bordercolor=BORDER)
        ttk.Progressbar(parent, variable=self._pv, maximum=100,
                        style="C.Horizontal.TProgressbar"
                        ).pack(fill="x", pady=(0,8))

        tk.Label(parent, text="IK SOLUTION  —  per motor",
                 font=(FNT,8,"bold"), fg=MUTED, bg=BG
                 ).pack(anchor="w", pady=(0,3))
        self._tbl = IKTable(parent)
        self._tbl.pack(fill="x")

        # Reachable Presets!
        pc = tk.Frame(parent, bg=CARD,
                      highlightbackground=BORDER, highlightthickness=1)
        pc.pack(fill="x", pady=(10,0))
        tk.Label(pc, text="QUICK PRESETS",
                 font=(FNT,8,"bold"), fg=MUTED, bg=CARD
                 ).pack(anchor="w", padx=10, pady=(7,4))
        pg = tk.Frame(pc, bg=CARD); pg.pack(padx=10, pady=(0,8))
        presets = [
            ("Reach Fwd",  "300", "  0", "300"),
            ("Reach Left", "  0", "300", "300"),
            ("Reach Right","  0","-300", "300"),
            ("High Fwd",   "200", "  0", "400"),
            ("Low Fwd",    "400", "  0", "150"),
            ("Diagonal",   "200", "200", "300"),
        ]
        for idx, (nm, x, y, z) in enumerate(presets):
            def _cb(nx=x,ny=y,nz=z):
                self._xv.set(nx.strip())
                self._yv.set(ny.strip())
                self._zv.set(nz.strip())
                self._solve()
            tk.Button(pg, text=nm, font=(FNT,8),
                      bg=INPUT, fg=TEXT, activebackground=BORDER,
                      relief="flat", cursor="hand2",
                      padx=8, pady=4, command=_cb
                      ).grid(row=idx//3, column=idx%3, padx=3, pady=2, sticky="ew")
        for c in range(3): pg.columnconfigure(c, weight=1)

        # ── CSV Sequence Panel ──
        cc = tk.Frame(parent, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        cc.pack(fill="x", pady=(10,0))
        tk.Label(cc, text="CARTESIAN CSV SEQUENCE", font=(FNT,8,"bold"), fg=MUTED, bg=CARD).pack(anchor="w", padx=10, pady=(7,4))

        cf = tk.Frame(cc, bg=CARD); cf.pack(fill="x", padx=10, pady=(0,8))
        self._csv_path = tk.StringVar()
        tk.Entry(cf, textvariable=self._csv_path, font=(FNT,9), bg=INPUT, fg=TEXT, insertbackground=TEXT, relief="flat", highlightbackground=BORDER, highlightthickness=1).pack(side="left", fill="x", expand=True, padx=(0,5))
        self._btn(cf,"BROWSE", BLUE, BG, self._browse_csv).pack(side="left", padx=2)
        self._bcsv_run = self._btn(cf,"RUN", GREEN, BG, self._run_csv, state="disabled")
        self._bcsv_run.pack(side="left", padx=2)

    def _build_viz_panel(self, parent):
        tk.Label(parent, text="ARM VISUALIZATION  —  live kinematic display",
                 font=(FNT,9,"bold"), fg=DIM, bg=BG
                 ).pack(anchor="w", pady=(0,4))

        leg = tk.Frame(parent, bg=PANEL); leg.pack(fill="x", pady=(0,5))
        for i, nm in enumerate(_JNAMES):
            f = tk.Frame(leg, bg=PANEL); f.pack(side="left", padx=8, pady=4)
            tk.Frame(f, bg=JCOLORS[i], width=12, height=12
                     ).pack(side="left", padx=(0,3))
            tk.Label(f, text=f"M{i+1} {nm}",
                     font=(FNT,8), fg=JCOLORS[i], bg=PANEL).pack(side="left")
        tf = tk.Frame(leg, bg=PANEL); tf.pack(side="right", padx=10, pady=4)
        tk.Label(tf, text="⊕ TARGET", font=(FNT,8), fg=GREEN, bg=PANEL).pack()

        self._viz = ArmVisualizer(parent)
        self._viz.pack(fill="both", expand=True)

    @staticmethod
    def _btn(parent, text, bg, fg, cmd, state="normal"):
        return tk.Button(parent, text=text, font=(FNT,9,"bold"),
                         bg=bg, fg=fg, activebackground=bg,
                         relief="flat", cursor="hand2",
                         command=cmd, padx=10, pady=5, state=state)

    def _solve(self):
        try:
            x = float(self._xv.get()) / 1000.0
            y = float(self._yv.get()) / 1000.0
            z = float(self._zv.get()) / 1000.0
        except ValueError:
            messagebox.showerror("Input Error", "X, Y, Z must be numbers.")
            return

        tgt = [x, y, z]
        self._viz.set_target(tgt)
        self._bsolve.config(state="disabled", text="SOLVING…")
        self._lbl_ik.config(text="Running IK solver…", fg=YELLOW)
        self._log(f"IK solve → X={x*1e3:+.1f}  Y={y*1e3:+.1f}  Z={z*1e3:+.1f} mm")
        threading.Thread(target=self._solve_bg, args=(tgt,), daemon=True).start()

    def _solve_bg(self, tgt):
        angles, err, ok = _ik(tgt, start_deg=self._cur_ang)
        self.after(0, lambda: self._solve_done(tgt, angles, err, ok))

    def _solve_done(self, tgt, angles, err, ok):
        self._bsolve.config(state="normal", text="🔍  SOLVE IK")
        if not ok:
            self._lbl_ik.config(
                text=f"⚠  IK failed — target may be out of reach  (error: {err*1000:.1f} mm)",
                fg=RED)
            self._log(f"IK failed — err={err*1000:.1f}mm", "err")
            self._bexec.config(state="disabled")
            return

        self._ik_ang = list(angles)
        self._ik_err = err

        try:
            t = max(0.1, float(self._tv.get()))
        except ValueError:
            t = 3.0

        speeds = []
        for i in range(6):
            d   = abs(angles[i] - self._cur_ang[i])
            spd = max(1.0, min(d/t if t>0 else _SPEED_LIMITS[i],
                               _SPEED_LIMITS[i]))
            speeds.append(spd)
            self._tbl.update_row(i, self._cur_ang[i], angles[i], spd)

        self._viz.set_angles(angles, duration=1.0, target_xyz=tgt)

        col = GREEN if err < 0.001 else (YELLOW if err < 0.005 else ORANGE)
        self._lbl_ik.config(
            text=f"✓ IK solved  —  error: {err*1000:.2f} mm  |  "
                 f"time: {t:.1f}s  |  max speed: {max(speeds):.0f}°/s",
            fg=col)
        self._log("IK OK — " +
                  "  ".join(f"M{i+1}:{angles[i]:+.1f}°" for i in range(6)))
        self._bexec.config(state="normal")

    def _execute(self):
        if self._ik_ang is None or self._executing: return

        robot = self.robot
        try:
            t = max(0.1, float(self._tv.get()))
        except ValueError:
            t = 3.0

        if not robot:
            self._executing = True
            self._bexec.config(state="disabled", text="EXECUTING…")
            self._pv.set(0)
            target = list(self._ik_ang)
            self._viz.set_angles(target, duration=t)
            self._log(f"[SIM] Animating {t:.1f}s")
            t0 = time.time()
            def tick():
                el = time.time()-t0
                self._pv.set(min(el/t*100, 100))
                if el < t: self.after(80, tick)
            tick()
            def done():
                time.sleep(t+0.25)
                self.after(0, self._exec_done, target)
            threading.Thread(target=done, daemon=True).start()
            return

        self._executing = True
        self._bexec.config(state="disabled", text="EXECUTING…")
        self._pv.set(0)
        threading.Thread(target=self._exec_bg,
                         args=(robot, list(self._ik_ang), t),
                         daemon=True).start()

    def _exec_bg(self, robot, angles, t):
        from robot_controller import DEFAULT_JOINT_CONFIG
        for i, cfg in enumerate(DEFAULT_JOINT_CONFIG):
            d   = abs(angles[i] - self._cur_ang[i])
            spd = max(1.0, min(d/t, _SPEED_LIMITS[i]))
            robot.motors[cfg.motor_id].set_position(
                float(angles[i]), max_speed_dps=float(spd), wait=False)
            time.sleep(0.001)
        self.after(0, lambda: self._viz.set_angles(angles, duration=t))
        t0 = time.time()
        while True:
            el = time.time()-t0
            self.after(0, lambda p=min(el/t*100,100): self._pv.set(p))
            if el >= t+0.5: break
            time.sleep(0.05)
        self.after(0, self._exec_done, angles)

    def _exec_done(self, final):
        self._cur_ang = list(final)
        self._tbl.set_current(final)
        self._executing = False
        self._pv.set(100)
        self._bexec.config(state="normal", text="▶  EXECUTE")
        self._lbl_ik.config(text="✓ Motion complete.", fg=GREEN)
        self._log("Motion complete.")

    def _stop(self):
        robot = self.robot
        if robot:
            threading.Thread(target=robot.stop_all, daemon=True).start()
        self._executing = False
        self._pv.set(0)
        self._viz._cancel_anim()
        self._bexec.config(
            state="normal" if self._ik_ang else "disabled",
            text="▶  EXECUTE")
        self._log("STOP.")

    def _home(self):
        self._xv.set("300"); self._yv.set("0"); self._zv.set("300")
        self._ik_ang  = [0.0]*6
        self._cur_ang = [0.0]*6
        self._viz.set_angles([0.0]*6, duration=1.5)
        self._viz.clear_target()
        self._tbl.set_current([0.0]*6)
        for i in range(6): self._tbl.update_row(i,0,0,0)
        robot = self.robot
        if robot:
            threading.Thread(target=robot.go_home,
                             kwargs={"speed_dps":60,"wait":False},
                             daemon=True).start()
        self._lbl_ik.config(text="Homing…", fg=DIM)
        self._log("Homing.")

    def _browse_csv(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(filetypes=[("CSV","*.csv")])
        if path:
            self._csv_path.set(path)
            self._bcsv_run.config(state="normal")

    def _run_csv(self):
        path = self._csv_path.get()
        import csv
        try:
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                rows = list(reader)
        except Exception as e:
            messagebox.showerror("CSV Error", str(e))
            return
        
        self._executing = True
        self._bcsv_run.config(state="disabled", text="RUNNING...")
        self._bexec.config(state="disabled")
        threading.Thread(target=self._run_csv_bg, args=(rows,), daemon=True).start()

    def _run_csv_bg(self, rows):
        import math
        self._log(f"Started CSV Sequence. Steps: {len(rows)}", "ok")
        for i, r in enumerate(rows):
            if not self._executing: break
            step = r.get("step", f"Step {i+1}")
            try:
                delay = float(r.get("delay_s", 0))
                x = float(r.get("x_mm", 0)) / 1000.0
                y = float(r.get("y_mm", 0)) / 1000.0
                z = float(r.get("z_mm", 0)) / 1000.0
                speed = float(r.get("speed_mms", 50))
            except Exception as e:
                self._log(f"[{step}] Parse err: {e}", "err")
                continue

            tgt = [x, y, z]
            self._log(f"[{step}] X:{x*1000:.0f} Y:{y*1000:.0f} Z:{z*1000:.0f} @{speed}mm/s")
            angles, err, ok = _ik(tgt, start_deg=self._cur_ang)
            if not ok:
                self._log(f"[{step}] IK Failed! err={err*1000:.1f}mm", "err")
                break
                
            _, pts = _fk(self._cur_ang)
            curr_xyz = pts[-1]
            dist_mm = math.dist(curr_xyz, tgt) * 1000.0
            t = dist_mm / speed if speed > 0 else 2.0
            if t < 0.1: t = 0.5
            
            robot = self.robot
            if robot:
                from robot_controller import DEFAULT_JOINT_CONFIG
                for j, cfg in enumerate(DEFAULT_JOINT_CONFIG):
                    d = abs(angles[j] - self._cur_ang[j])
                    spd = max(1.0, min(d/t, _SPEED_LIMITS[j]))
                    robot.motors[cfg.motor_id].set_position(
                        float(angles[j]), max_speed_dps=float(spd), wait=False)
                    time.sleep(0.001)

            self.after(0, lambda a=angles, tc=t, tg=tgt: self._viz.set_angles(a, duration=tc, target_xyz=tg))
            
            el = 0.0
            while el < t + 0.1:
                if not self._executing: break
                time.sleep(0.1)
                el += 0.1
                
            self._cur_ang = list(angles)
            self.after(0, lambda a=angles: self._tbl.set_current(a))
            
            if delay > 0 and self._executing:
                self._log(f"[{step}] Delay {delay}s")
                el = 0.0
                while el < delay:
                    if not self._executing: break
                    time.sleep(0.1)
                    el += 0.1

        self._log("CSV Sequence Complete.", "ok")
        self._executing = False
        self.after(0, lambda: self._bcsv_run.config(state="normal", text="RUN"))
        self.after(0, lambda: self._bexec.config(state="normal" if self._ik_ang else "disabled"))

    # ── Connection ─────────────────────────────────────────────────────────────

    def _connect(self):
        if not HAS_ROBOT:
            self._lbl_c.config(text="● VISUAL SIM", fg=YELLOW)
            self._log("Robot modules not available — visual simulation only.", "warn")
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
        self._log("Connected — all 6 motors enabled.", "ok")
        self._start_monitor()

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

    # ── Monitor (Thread-Safe Fix) ──────────────────────────────────────────────

    def _start_monitor(self):
        self._mon_job = self.after(300, self._monitor)

    def _stop_monitor(self):
        if self._mon_job:
            self.after_cancel(self._mon_job)
            self._mon_job = None

    def _monitor(self):
        if self.robot and self._connected:
            try:
                from robot_controller import DEFAULT_JOINT_CONFIG
                # Fix: Use get_all_feedback instead of blocking get_position calls
                fb_all = self.robot.get_all_feedback()
                a = []
                for i, cfg in enumerate(DEFAULT_JOINT_CONFIG):
                    fb = fb_all.get(cfg.motor_id)
                    a.append(fb.position_deg if fb else self._cur_ang[i])
                
                self._cur_ang = list(a)
                if not self._executing:
                    self._viz.set_angles(a)
                self._tbl.set_current(a)
            except Exception:
                pass
        self._mon_job = self.after(300, self._monitor)

    # ── Log ────────────────────────────────────────────────────────────────────

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

    # ── Close ──────────────────────────────────────────────────────────────────

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
    ap = argparse.ArgumentParser(description="6-Axis Cartesian + Visualizer")
    ap.add_argument("--real",    action="store_true")
    ap.add_argument("--port",    default="COM3")
    ap.add_argument("--bustype", default="slcan")
    args = ap.parse_args()
    CartesianGUI(simulated=not args.real,
                 port=args.port, bustype=args.bustype).mainloop()