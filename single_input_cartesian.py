"""
single_input_cartesian.py
=========================
Cartesian Motion Control  +  Live 3D Robot Arm Visualizer

Enter one X/Y/Z target position → system solves IK → shows how every joint
adjusts in real time through two live orthographic projection views:

    FRONT VIEW  (XZ plane — side profile)
    TOP VIEW    (XY plane — bird's eye)

The arm animates live during motion using the FK joint_positions list.
A ghost outline shows the TARGET pose while the real arm moves toward it.

Run:
    python3.10 single_input_cartesian.py             ← simulation
    python3.10 single_input_cartesian.py --real --port /dev/ttyACM0

Requires: numpy   (pip install numpy)
No matplotlib needed — pure tkinter canvas drawing.
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import logging
import queue
import argparse
import math
import numpy as np
from typing import Optional

from can_interface import CANBus
from robot_controller import RobotController, DEFAULT_JOINT_CONFIG
from kinematics import (
    forward_kinematics, plan_cartesian_motion,
    CartesianMotionPlan, JOINT_SPEED_LIMITS_DPS, DH_PARAMS
)

# ── Logging ───────────────────────────────────────────────────────────────────
log_queue: queue.Queue = queue.Queue()

class QueueHandler(logging.Handler):
    def emit(self, record):
        log_queue.put(self.format(record))

# ── Theme ─────────────────────────────────────────────────────────────────────
BG     = "#0d1117"
PANEL  = "#161b22"
CARD   = "#1c2128"
INPUT  = "#21262d"
BORDER = "#30363d"
ACCENT = "#00d4aa"
BLUE   = "#4dabf7"
RED    = "#ff4444"
YELLOW = "#e3b341"
GREEN  = "#3fb950"
ORANGE = "#f0883e"
MUTED  = "#7d8590"
TEXT   = "#e6edf3"
JCOLORS = ["#00d4aa","#4dabf7","#e3b341","#ff7eb6","#a5d6ff","#c3e88d"]
JNAMES  = ["Base","Shoulder","Elbow","Wrist1","Wrist2","Tool"]

# ── Helpers ───────────────────────────────────────────────────────────────────
def lbl(parent, text, size=8, color=MUTED, bold=False, bg=CARD, **kw):
    return tk.Label(parent, text=text,
                    font=("Courier New", size, "bold" if bold else "normal"),
                    fg=color, bg=bg, **kw)

def ent(parent, var, width=9, **kw):
    return tk.Entry(parent, textvariable=var, width=width,
                    font=("Courier New", 10), bg=INPUT, fg=TEXT,
                    insertbackground=TEXT, relief="flat",
                    highlightbackground=BORDER, highlightthickness=1, **kw)

def btn(parent, text, color=ACCENT, fg=BG, cmd=None, **kw):
    return tk.Button(parent, text=text,
                     font=("Courier New", 9, "bold"),
                     bg=color, fg=fg, activebackground=color,
                     relief="flat", cursor="hand2",
                     command=cmd, padx=10, pady=5, **kw)


# ══════════════════════════════════════════════════════════════════════════════
# ARM VISUALIZER  — dual orthographic canvas views
# ══════════════════════════════════════════════════════════════════════════════

class ArmVisualizer(tk.Frame):
    """
    Draws two live orthographic projections of the 6-DOF robot arm:
      LEFT  — Front view  : X (horizontal) vs Z (vertical)
      RIGHT — Top view    : X (horizontal) vs Y (vertical, going away)

    Anatomy of one frame:
      - Solid colored lines  = real current arm pose (from live FK)
      - Dashed grey lines    = target ghost pose (after IK, before motion starts)
      - Colored filled dots  = each joint
      - Red cross / target   = commanded end-effector target
      - Coordinate grid      = faint background grid with axis labels
      - Reachability sphere  = dashed circle showing approximate workspace limit

    The visualizer updates at ~30 Hz via a tkinter after() loop.
    """

    # World scale: metres → canvas pixels
    # Adjust SCALE to zoom in/out (pixels per metre)
    SCALE = 280.0

    # Canvas dimensions
    W = 340
    H = 340

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)

        # State
        self._current_joints: list = [[0,0,0]] * 7   # base + 6 joints
        self._target_joints:  list = None             # ghost pose (None = hidden)
        self._target_xyz:     Optional[np.ndarray] = None
        self._animating:      bool = False

        # Approximate max reach (sum of link lengths from DH)
        self._max_reach = sum(abs(p[2]) + abs(p[1]) for p in DH_PARAMS) * 0.9

        self._build()
        self._redraw()

    def _build(self):
        # Title
        lbl(self, "LIVE ARM VISUALIZATION",
            size=9, color=ACCENT, bold=True, bg=BG).pack(anchor="w", pady=(6,2))
        lbl(self, "Solid = current pose  ·  Dashed = target ghost  ·  ✕ = target tip",
            size=7, color=MUTED, bg=BG).pack(anchor="w", pady=(0,6))

        views_row = tk.Frame(self, bg=BG)
        views_row.pack(fill="x")

        # Front view canvas
        front_col = tk.Frame(views_row, bg=BG)
        front_col.pack(side="left", padx=(0,8))
        lbl(front_col, "FRONT VIEW  (X–Z)", size=8,
            color=BLUE, bold=True, bg=BG).pack()
        lbl(front_col, "X →  Z ↑", size=7, color=MUTED, bg=BG).pack()
        self.canvas_front = tk.Canvas(front_col,
                                      width=self.W, height=self.H,
                                      bg=CARD, highlightthickness=1,
                                      highlightbackground=BORDER)
        self.canvas_front.pack()

        # Top view canvas
        top_col = tk.Frame(views_row, bg=BG)
        top_col.pack(side="left")
        lbl(top_col, "TOP VIEW  (X–Y)", size=8,
            color=ORANGE, bold=True, bg=BG).pack()
        lbl(top_col, "X →  Y ↑ (into screen)", size=7, color=MUTED, bg=BG).pack()
        self.canvas_top = tk.Canvas(top_col,
                                    width=self.W, height=self.H,
                                    bg=CARD, highlightthickness=1,
                                    highlightbackground=BORDER)
        self.canvas_top.pack()

        # Joint angle readout strip
        ang_row = tk.Frame(self, bg=BG)
        ang_row.pack(fill="x", pady=(8,0))
        lbl(ang_row, "JOINT ANGLES:", size=7, color=MUTED,
            bold=True, bg=BG).pack(side="left", padx=(0,8))
        self._angle_lbls = []
        for i in range(6):
            f = tk.Frame(ang_row, bg=BG)
            f.pack(side="left", padx=4)
            lbl(f, f"J{i+1}", size=7, color=JCOLORS[i], bold=True, bg=BG).pack()
            al = lbl(f, "0.0°", size=8, color=TEXT, bg=BG)
            al.pack()
            self._angle_lbls.append(al)

        # End-effector XYZ readout
        xyz_row = tk.Frame(self, bg=BG)
        xyz_row.pack(fill="x", pady=(4,0))
        lbl(xyz_row, "END-EFFECTOR:", size=7, color=MUTED,
            bold=True, bg=BG).pack(side="left", padx=(0,8))
        self._ee_lbl = lbl(xyz_row, "X: ---  Y: ---  Z: ---  mm",
                           size=8, color=ACCENT, bg=BG)
        self._ee_lbl.pack(side="left")

        # Target XYZ readout
        tgt_row = tk.Frame(self, bg=BG)
        tgt_row.pack(fill="x", pady=(2,6))
        lbl(tgt_row, "TARGET:      ", size=7, color=MUTED,
            bold=True, bg=BG).pack(side="left", padx=(0,8))
        self._tgt_lbl = lbl(tgt_row, "not set",
                            size=8, color=YELLOW, bg=BG)
        self._tgt_lbl.pack(side="left")

    # ── Coordinate transforms ─────────────────────────────────────────────────

    def _origin(self):
        """Canvas centre point (pixels)."""
        return self.W // 2, self.H // 2 + 40   # shift down a bit for vertical arms

    def _to_front(self, xyz):
        """World [x,y,z] metres → canvas (px, py) for front view (X rightward, Z upward)."""
        ox, oy = self._origin()
        px = ox + xyz[0] * self.SCALE
        py = oy - xyz[2] * self.SCALE   # Z is up, canvas Y is down
        return px, py

    def _to_top(self, xyz):
        """World [x,y,z] metres → canvas (px, py) for top view (X rightward, Y upward)."""
        ox, oy = self._origin()
        px = ox + xyz[0] * self.SCALE
        py = oy - xyz[1] * self.SCALE   # Y into screen, canvas Y is down
        return px, py

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def _draw_grid(self, canvas, proj_fn, axis_h, axis_v):
        """Draw background grid with axis labels."""
        ox, oy = self._origin()
        step_m  = 0.1   # grid every 100mm
        step_px = step_m * self.SCALE

        # Grid lines
        grid_range = range(-4, 5)
        for gi in grid_range:
            x0  = ox + gi * step_px
            canvas.create_line(x0, 0, x0, self.H,
                               fill=INPUT, width=1)
        for gi in grid_range:
            y0 = oy - gi * step_px
            canvas.create_line(0, y0, self.W, y0,
                               fill=INPUT, width=1)

        # Axis lines (bright)
        canvas.create_line(0, oy, self.W, oy, fill=BORDER, width=1)
        canvas.create_line(ox, 0, ox, self.H, fill=BORDER, width=1)

        # Axis labels
        canvas.create_text(self.W-10, oy-6,
                           text=f"{axis_h}+",
                           fill=MUTED, font=("Courier New",7))
        canvas.create_text(ox+12, 8,
                           text=f"{axis_v}+",
                           fill=MUTED, font=("Courier New",7))

        # Workspace radius circle
        r_px = self._max_reach * self.SCALE
        canvas.create_oval(ox-r_px, oy-r_px, ox+r_px, oy+r_px,
                           outline=BORDER, width=1, dash=(4,6))
        canvas.create_text(ox + r_px*0.72, oy - r_px*0.72,
                           text="workspace", fill=BORDER,
                           font=("Courier New",6))

        # Scale bar
        bar_px = step_px
        canvas.create_line(10, self.H-14, 10+bar_px, self.H-14,
                           fill=MUTED, width=2)
        canvas.create_text(10 + bar_px//2, self.H-6,
                           text="100mm", fill=MUTED,
                           font=("Courier New",6))

    def _draw_arm(self, canvas, joints, proj_fn, alpha=1.0,
                  dashed=False, label_joints=True):
        """
        Draw the arm as connected segments on a canvas.
        joints  : list of [x,y,z] world coords (7 points: base + 6 joints)
        proj_fn : projection function (front or top)
        alpha   : unused (tkinter can't do opacity) — dashed instead
        dashed  : True for ghost target pose
        """
        n     = len(joints)
        dash  = (6, 4) if dashed else ()
        width = 1 if dashed else 3

        pts = [proj_fn(j) for j in joints]

        # Draw links
        for i in range(n - 1):
            x0, y0 = pts[i]
            x1, y1 = pts[i+1]
            color = JCOLORS[i] if not dashed else MUTED
            canvas.create_line(x0, y0, x1, y1,
                               fill=color, width=width, dash=dash,
                               capstyle=tk.ROUND, joinstyle=tk.ROUND)

        # Draw joint dots
        for i in range(n):
            x, y = pts[i]
            if dashed:
                r = 3
                canvas.create_oval(x-r, y-r, x+r, y+r,
                                   outline=MUTED, fill="", dash=(3,3))
            else:
                r = 5 if i == 0 else (7 if i == n-1 else 4)
                color = JCOLORS[min(i, 5)]
                canvas.create_oval(x-r, y-r, x+r, y+r,
                                   fill=color, outline=BG, width=1)
                if label_joints and i > 0:
                    canvas.create_text(x+10, y-8,
                                       text=f"J{i}",
                                       fill=color,
                                       font=("Courier New", 6))

        # Base marker (cross)
        bx, by = pts[0]
        canvas.create_line(bx-8, by, bx+8, by, fill=MUTED, width=2)
        canvas.create_line(bx, by-8, bx, by+8, fill=MUTED, width=2)

        # End-effector highlight
        ex, ey = pts[-1]
        r = 8
        canvas.create_oval(ex-r, ey-r, ex+r, ey+r,
                           outline=JCOLORS[5] if not dashed else MUTED,
                           fill="", width=2)

    def _draw_target_cross(self, canvas, xyz, proj_fn):
        """Draw a red ✕ at the target XYZ position."""
        tx, ty = proj_fn(xyz)
        s = 10
        canvas.create_line(tx-s, ty-s, tx+s, ty+s,
                           fill=RED, width=2)
        canvas.create_line(tx+s, ty-s, tx-s, ty+s,
                           fill=RED, width=2)
        canvas.create_oval(tx-s, ty-s, tx+s, ty+s,
                           outline=RED, width=1, dash=(3,3))
        canvas.create_text(tx+14, ty-12, text="TARGET",
                           fill=RED, font=("Courier New", 6, "bold"))

    # ── Main redraw ───────────────────────────────────────────────────────────

    def _redraw(self):
        """Clear and redraw both canvases. Called every 33ms (≈30 Hz)."""
        self._draw_canvas(self.canvas_front, self._to_front, "X", "Z")
        self._draw_canvas(self.canvas_top,   self._to_top,   "X", "Y")
        self.after(33, self._redraw)

    def _draw_canvas(self, canvas, proj_fn, axis_h, axis_v):
        canvas.delete("all")

        # Grid
        self._draw_grid(canvas, proj_fn, axis_h, axis_v)

        # Ghost target arm
        if self._target_joints is not None:
            self._draw_arm(canvas, self._target_joints, proj_fn,
                           dashed=True, label_joints=False)

        # Target cross
        if self._target_xyz is not None:
            self._draw_target_cross(canvas, self._target_xyz / 1000.0, proj_fn)

        # Real current arm
        self._draw_arm(canvas, self._current_joints, proj_fn,
                       dashed=False, label_joints=True)

    # ── Public API (called from outside) ─────────────────────────────────────

    def set_joint_angles(self, angles_deg: list | np.ndarray):
        """
        Update visualizer with new joint angles.
        Recomputes FK and redraws both views.
        Called from the monitor loop every ~300ms.
        """
        fk = forward_kinematics(angles_deg)
        self._current_joints = fk.joint_positions   # list of [x,y,z]

        # Update angle strip
        for i, a in enumerate(angles_deg):
            self._angle_lbls[i].config(text=f"{float(a):+.1f}°")

        # Update EE readout
        p = fk.position_mm()
        self._ee_lbl.config(text=f"X:{p[0]:+.0f}  Y:{p[1]:+.0f}  Z:{p[2]:+.0f} mm")

    def set_target(self, target_mm: np.ndarray, target_angles_deg: np.ndarray):
        """
        Set the ghost target pose.
        target_mm         : [x,y,z] in mm (for the red cross)
        target_angles_deg : IK solution angles (for the ghost arm)
        """
        self._target_xyz   = target_mm.copy()
        fk = forward_kinematics(target_angles_deg)
        self._target_joints = fk.joint_positions
        self._tgt_lbl.config(
            text=f"X:{target_mm[0]:+.0f}  Y:{target_mm[1]:+.0f}  Z:{target_mm[2]:+.0f} mm",
            fg=YELLOW)

    def clear_target(self):
        self._target_joints = None
        self._target_xyz    = None
        self._tgt_lbl.config(text="not set", fg=MUTED)


# ══════════════════════════════════════════════════════════════════════════════
# INPUT PANEL  — single X/Y/Z + Time input
# ══════════════════════════════════════════════════════════════════════════════

class InputPanel(tk.Frame):
    """
    Left-side control panel:
    - Single X / Y / Z target input (mm)
    - Time input (seconds)
    - Mode: Relative or Absolute
    - Compute Plan → Execute → Stop
    - IK result table
    - Speed bars
    """

    def __init__(self, parent, get_robot, get_angles, log_fn,
                 on_plan_cb=None, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._get_robot  = get_robot
        self._get_angles = get_angles
        self._log        = log_fn
        self._on_plan_cb = on_plan_cb   # called with (plan) when IK solved
        self._plan: Optional[CartesianMotionPlan] = None
        self._executing  = False
        self._build()

    def _build(self):
        # Mode
        mc = tk.Frame(self, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        mc.pack(fill="x", pady=(0,8))
        lbl(mc, "MOTION MODE", size=7, bg=CARD).pack(anchor="w", padx=12, pady=(8,4))
        self.mode_var = tk.StringVar(value="Relative")
        mr = tk.Frame(mc, bg=CARD)
        mr.pack(padx=12, pady=(0,8))
        for m, desc in [("Relative","Δ from current  (move by)"),
                        ("Absolute","exact position from base")]:
            tk.Radiobutton(mr, text=f"  {m}  —  {desc}",
                           variable=self.mode_var, value=m,
                           font=("Courier New",8), bg=CARD, fg=TEXT,
                           selectcolor=INPUT, activebackground=CARD,
                           activeforeground=TEXT,
                           command=self._on_mode).pack(anchor="w")

        # Target input card
        ic = tk.Frame(self, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        ic.pack(fill="x", pady=(0,8))
        self.inp_hdr = lbl(ic, "TARGET DISPLACEMENT  (mm)", size=7, bg=CARD)
        self.inp_hdr.pack(anchor="w", padx=12, pady=(8,4))

        # Big X Y Z entries
        xyz_f = tk.Frame(ic, bg=CARD)
        xyz_f.pack(pady=6)
        self.x_var = tk.StringVar(value="0.0")
        self.y_var = tk.StringVar(value="0.0")
        self.z_var = tk.StringVar(value="0.0")
        for axis, var, color in [("X", self.x_var, ACCENT),
                                  ("Y", self.y_var, BLUE),
                                  ("Z", self.z_var, YELLOW)]:
            f = tk.Frame(xyz_f, bg=CARD)
            f.pack(side="left", padx=14)
            lbl(f, axis, size=20, color=color, bold=True, bg=CARD).pack()
            ent(f, var, width=9).pack()
            lbl(f, "mm", size=7, bg=CARD).pack()

        # Separator + Time
        tk.Frame(ic, bg=BORDER, height=1).pack(fill="x", pady=6)
        tr = tk.Frame(ic, bg=CARD)
        tr.pack(padx=12, pady=(0,10))
        lbl(tr, "T", size=18, color="#ff7eb6", bold=True, bg=CARD).pack(side="left", padx=(0,8))
        self.t_var = tk.StringVar(value="3.0")
        ent(tr, self.t_var, width=7).pack(side="left")
        lbl(tr, "  seconds\n  (speeds auto-computed per joint)",
            size=7, color=MUTED, bg=CARD).pack(side="left", padx=8)

        # Action buttons
        br = tk.Frame(self, bg=BG)
        br.pack(fill="x", pady=6)
        self.btn_compute = btn(br, "COMPUTE + PREVIEW", BLUE, fg=BG,
                               cmd=self._compute)
        self.btn_compute.pack(side="left", padx=(0,4))
        self.btn_exec = btn(br, "▶  EXECUTE", ACCENT, fg=BG,
                            cmd=self._execute, state="disabled")
        self.btn_exec.pack(side="left", padx=(0,4))
        btn(br, "■  STOP", RED, fg=TEXT,
            cmd=self._stop).pack(side="left", padx=(0,4))
        btn(br, "HOME", PANEL, fg=MUTED,
            cmd=self._home).pack(side="left")

        # Status + progress
        self.lbl_status = lbl(self, "Enter X Y Z and click COMPUTE.",
                              size=8, color=MUTED, bg=BG)
        self.lbl_status.pack(anchor="w", pady=(4,2))
        self.prog_var = tk.DoubleVar(value=0)
        style = ttk.Style()
        style.configure("IP.Horizontal.TProgressbar",
                        troughcolor=INPUT, background=ACCENT,
                        bordercolor=BORDER)
        ttk.Progressbar(self, variable=self.prog_var, maximum=100,
                        style="IP.Horizontal.TProgressbar").pack(fill="x")

        # IK result table
        tbl = tk.Frame(self, bg=CARD, highlightbackground=BORDER,
                       highlightthickness=1)
        tbl.pack(fill="x", pady=(8,6))
        lbl(tbl, "IK SOLUTION  —  per joint", size=7, bg=CARD).pack(
            anchor="w", padx=12, pady=(8,4))

        hdr = tk.Frame(tbl, bg=PANEL)
        hdr.pack(fill="x", padx=8)
        for txt, w in [("Jt",3),("Name",9),("Current",9),
                       ("Target",9),("Δ°",7),("Speed",10),("",6)]:
            lbl(hdr, txt, size=7, bold=True, bg=PANEL,
                width=w).pack(side="left", padx=2, pady=3)

        self.tbl_rows = []
        for i in range(6):
            r = tk.Frame(tbl, bg=CARD)
            r.pack(fill="x", padx=8, pady=1)
            cells = {}
            cells["j"]   = lbl(r, f"J{i+1}", size=8, color=JCOLORS[i], bold=True, bg=CARD, width=3)
            cells["nm"]  = lbl(r, JNAMES[i], size=8, color=MUTED, bg=CARD, width=9)
            cells["cur"] = lbl(r, "---", size=8, color=MUTED, bg=CARD, width=9)
            cells["tgt"] = lbl(r, "---", size=8, color=TEXT, bg=CARD, width=9)
            cells["d"]   = lbl(r, "---", size=8, bg=CARD, width=7)
            cells["spd"] = lbl(r, "---", size=8, color=ACCENT, bg=CARD, width=10)
            cells["ok"]  = lbl(r, "",   size=7, color=GREEN, bg=CARD, width=6)
            for c in cells.values():
                c.pack(side="left", padx=2)
            self.tbl_rows.append(cells)

        self.lbl_ik = lbl(tbl, "", size=8, color=MUTED, bg=CARD)
        self.lbl_ik.pack(anchor="w", padx=12, pady=(4,8))

        # Speed bars
        bars = tk.Frame(self, bg=CARD, highlightbackground=BORDER,
                        highlightthickness=1)
        bars.pack(fill="x")
        lbl(bars, "JOINT SPEED LOAD", size=7, bg=CARD).pack(
            anchor="w", padx=12, pady=(8,4))
        self._bar_fills = []
        self._bar_lbls  = []
        for i in range(6):
            br2 = tk.Frame(bars, bg=CARD)
            br2.pack(fill="x", padx=12, pady=1)
            lbl(br2, f"J{i+1}", size=8, color=JCOLORS[i], bold=True,
                bg=CARD, width=3).pack(side="left")
            bg_bar = tk.Frame(br2, bg=INPUT, height=12, width=200)
            bg_bar.pack(side="left", padx=4)
            bg_bar.pack_propagate(False)
            fill = tk.Frame(bg_bar, bg=JCOLORS[i], height=12, width=0)
            fill.place(x=0, y=0, relheight=1)
            self._bar_fills.append((bg_bar, fill))
            bl = lbl(br2, "---", size=8, color=TEXT, bg=CARD)
            bl.pack(side="left", padx=4)
            self._bar_lbls.append(bl)
        lbl(bars, "", bg=CARD).pack(pady=4)

        # Mini log
        log_hdr = tk.Frame(self, bg=PANEL)
        log_hdr.pack(fill="x", pady=(8,0))
        lbl(log_hdr, "LOG", size=7, color=MUTED, bg=PANEL).pack(
            side="left", padx=8, pady=3)
        tk.Button(log_hdr, text="CLR", font=("Courier New",7),
                  bg=PANEL, fg=MUTED, relief="flat", cursor="hand2",
                  command=lambda: (self.log_box.config(state="normal"),
                                   self.log_box.delete("1.0","end"),
                                   self.log_box.config(state="disabled"))
                  ).pack(side="right", padx=6)
        self.log_box = scrolledtext.ScrolledText(
            self, height=6, font=("Courier New",8),
            bg=PANEL, fg=MUTED, relief="flat",
            state="disabled", wrap="word")
        self.log_box.pack(fill="x")
        self.log_box.tag_config("ok",   foreground=GREEN)
        self.log_box.tag_config("warn", foreground=YELLOW)
        self.log_box.tag_config("err",  foreground=RED)
        self.log_box.tag_config("info", foreground=MUTED)

    def _on_mode(self):
        if self.mode_var.get() == "Absolute":
            self.inp_hdr.config(text="ABSOLUTE TARGET  (mm from base)")
        else:
            self.inp_hdr.config(text="TARGET DISPLACEMENT  (mm)")

    def _parse(self):
        try:
            x = float(self.x_var.get())
            y = float(self.y_var.get())
            z = float(self.z_var.get())
            t = float(self.t_var.get())
        except ValueError:
            messagebox.showerror("Input Error", "X, Y, Z, T must all be numbers.")
            return None
        if t <= 0:
            messagebox.showerror("Input Error", "Time T must be > 0.")
            return None
        return x, y, z, t

    def _compute(self):
        v = self._parse()
        if v is None:
            return
        x, y, z, t = v
        cur = self._get_angles()
        self.btn_compute.config(state="disabled", text="COMPUTING IK...")
        self._log_m(f"IK → ({x:+.0f},{y:+.0f},{z:+.0f}) mm  T={t}s")
        threading.Thread(target=self._compute_worker,
                         args=(x,y,z,t,cur), daemon=True).start()

    def _compute_worker(self, x, y, z, t, cur):
        mode = self.mode_var.get().lower()
        if mode == "absolute":
            plan = plan_cartesian_motion([], t, cur, "absolute", [x,y,z])
        else:
            plan = plan_cartesian_motion([x,y,z], t, cur, "relative")
        self.after(0, lambda: self._on_plan(plan))

    def _on_plan(self, plan: CartesianMotionPlan):
        self.btn_compute.config(state="normal", text="COMPUTE + PREVIEW")
        self._plan = plan

        if not plan.success:
            messagebox.showerror("IK Failed", plan.message)
            self.lbl_status.config(text=f"Failed: {plan.message}", fg=RED)
            self._log_m(f"IK FAILED: {plan.message}", "err")
            return

        self.btn_exec.config(state="normal")

        # Update table
        for i, r in enumerate(self.tbl_rows):
            spd     = plan.speeds_dps[i]
            clamped = spd >= JOINT_SPEED_LIMITS_DPS[i] * 0.99
            r["cur"].config(text=f"{plan.start_angles_deg[i]:+.1f}°", fg=MUTED)
            r["tgt"].config(text=f"{plan.target_angles_deg[i]:+.1f}°", fg=TEXT)
            r["d"].config(text=f"{plan.angle_changes_deg[i]:.1f}°", fg=TEXT)
            r["spd"].config(text=f"{spd:.1f}°/s",
                            fg=RED if clamped else ACCENT)
            r["ok"].config(text="⚠CLMP" if clamped else "✓ OK",
                           fg=RED if clamped else GREEN)

        # Speed bars
        for i in range(6):
            ratio = min(plan.speeds_dps[i] / JOINT_SPEED_LIMITS_DPS[i], 1.0)
            _, fill = self._bar_fills[i]
            fill.place(width=int(200*ratio))
            fill.config(bg=RED if ratio>0.9 else (YELLOW if ratio>0.7 else JCOLORS[i]))
            self._bar_lbls[i].config(text=f"{plan.speeds_dps[i]:.1f}°/s")

        ik_c = GREEN if plan.ik_error_mm < 1 else (YELLOW if plan.ik_error_mm < 5 else RED)
        self.lbl_ik.config(
            text=f"IK err:{plan.ik_error_mm:.2f}mm  max:{plan.max_speed_dps:.1f}°/s  t:{plan.time_s:.2f}s",
            fg=ik_c)
        self.lbl_status.config(text=f"Plan ready — {plan.time_s:.1f}s", fg=GREEN)
        self._log_m(f"Plan OK — IK err {plan.ik_error_mm:.2f}mm", "ok")

        # Notify visualizer
        if self._on_plan_cb:
            self._on_plan_cb(plan)

    def _execute(self):
        robot = self._get_robot()
        if not robot:
            messagebox.showwarning("Not Connected", "Connect first.")
            return
        if not self._plan or self._executing:
            return
        self._executing = True
        self.btn_exec.config(state="disabled", text="EXECUTING...")
        self.prog_var.set(0)
        self._log_m(f"▶ Executing {self._plan.time_s:.1f}s motion...", "ok")
        threading.Thread(target=self._exec_worker,
                         args=(robot, self._plan), daemon=True).start()

    def _exec_worker(self, robot, plan):
        for i, cfg in enumerate(DEFAULT_JOINT_CONFIG):
            robot.motors[cfg.motor_id].set_position(
                float(plan.target_angles_deg[i]),
                max_speed_dps=float(plan.speeds_dps[i]),
                wait=False)
            time.sleep(0.001)

        t0 = time.time()
        while True:
            elapsed = time.time() - t0
            pct = min(elapsed / plan.time_s * 100, 100)
            self.after(0, lambda p=pct: self.prog_var.set(p))
            if elapsed >= plan.time_s + 0.5:
                break
            time.sleep(0.05)
        self.after(0, self._exec_done)

    def _exec_done(self):
        self._executing = False
        self.prog_var.set(100)
        self.btn_exec.config(state="normal", text="▶  EXECUTE")
        self.lbl_status.config(text="Motion complete.", fg=GREEN)
        self._log_m("Motion complete.", "ok")

    def _stop(self):
        robot = self._get_robot()
        if robot:
            threading.Thread(target=robot.stop_all, daemon=True).start()
        self._executing = False
        self.prog_var.set(0)
        self.btn_exec.config(
            state="normal" if self._plan else "disabled", text="▶  EXECUTE")
        self._log_m("Stopped.", "warn")

    def _home(self):
        robot = self._get_robot()
        if robot:
            threading.Thread(target=robot.go_home,
                             kwargs={"speed_dps":60,"wait":False},
                             daemon=True).start()
            self._log_m("Homing.")

    def _log_m(self, msg, tag="info"):
        ts = time.strftime("%H:%M:%S")
        self.log_box.config(state="normal")
        self.log_box.insert("end", f"{ts}  {msg}\n", tag)
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def poll_log(self):
        while not log_queue.empty():
            msg = log_queue.get_nowait()
            tag = "warn" if "WARNING" in msg else ("err" if "ERROR" in msg else "info")
            self._log_m(msg, tag)
        self.after(150, self.poll_log)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class CartesianVisApp(tk.Tk):
    def __init__(self, simulated=True, port="COM3", bustype="slcan"):
        super().__init__()
        self.title("RMD-X8-120  ·  Cartesian Control  +  Live Arm Visualizer")
        self.configure(bg=BG)
        self.minsize(1100, 820)

        self.robot: Optional[RobotController] = None
        self.bus:   Optional[CANBus]          = None
        self._sim   = simulated
        self._port  = port
        self._bus_t = bustype
        self._monitor_job = None

        handler = QueueHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        if simulated:
            self.after(400, self._connect)

    def _build(self):
        # ── Top bar ───────────────────────────────────────────────────────────
        top = tk.Frame(self, bg=PANEL, height=50)
        top.pack(fill="x")
        top.pack_propagate(False)
        lbl(top, "⬡  RMD-X8-120  CARTESIAN  +  VISUALIZER",
            size=12, color=ACCENT, bold=True, bg=PANEL
            ).pack(side="left", padx=16, pady=12)

        self.lbl_conn = lbl(top, "● DISCONNECTED", size=9,
                            color=RED, bold=True, bg=PANEL)
        self.lbl_conn.pack(side="right", padx=12)

        mode_txt = "SIMULATION" if self._sim else f"REAL · {self._port}"
        lbl(top, mode_txt, size=8, color=MUTED, bg=PANEL).pack(side="right", padx=4)
        btn(top, "DISCONNECT", PANEL, fg=RED,
            cmd=self._disconnect).pack(side="right", padx=4)
        btn(top, "CONNECT", ACCENT, fg=BG,
            cmd=self._connect).pack(side="right", padx=4)

        # ── Main layout: left=input panel, right=visualizer ───────────────────
        main = tk.Frame(self, bg=BG)
        main.pack(fill="both", expand=True, padx=12, pady=8)

        # Scrollable left panel
        left_outer = tk.Frame(main, bg=BG, width=420)
        left_outer.pack(side="left", fill="y", padx=(0,10))
        left_outer.pack_propagate(False)

        left_canvas = tk.Canvas(left_outer, bg=BG, highlightthickness=0, width=410)
        left_scroll  = ttk.Scrollbar(left_outer, orient="vertical",
                                     command=left_canvas.yview)
        left_inner   = tk.Frame(left_canvas, bg=BG)
        left_inner.bind("<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        left_canvas.create_window((0,0), window=left_inner, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_scroll.pack(side="right", fill="y")
        left_canvas.pack(side="left", fill="both", expand=True)

        # Bind mousewheel on left panel
        def _on_wheel(e):
            left_canvas.yview_scroll(int(-1*(e.delta/120)), "units")
        left_canvas.bind_all("<MouseWheel>", _on_wheel)

        self.input_panel = InputPanel(
            left_inner,
            get_robot=lambda: self.robot,
            get_angles=self._get_angles,
            log_fn=lambda m, t="info": None,
            on_plan_cb=self._on_plan_ready,
        )
        self.input_panel.pack(fill="x", padx=4)
        self.input_panel.poll_log()

        # Right: visualizer
        right = tk.Frame(main, bg=BG)
        right.pack(side="right", fill="both", expand=True)

        self.visualizer = ArmVisualizer(right)
        self.visualizer.pack(fill="both", expand=True)

        # ── Coordinate reference card ─────────────────────────────────────────
        ref = tk.Frame(right, bg=CARD,
                       highlightbackground=BORDER, highlightthickness=1)
        ref.pack(fill="x", pady=(6,0))
        lbl(ref, "COORDINATE SYSTEM REFERENCE", size=7,
            color=MUTED, bg=CARD).pack(anchor="w", padx=12, pady=(6,2))
        ref_row = tk.Frame(ref, bg=CARD)
        ref_row.pack(fill="x", padx=12, pady=(0,8))
        for axis, color, desc in [
            ("X →", ACCENT, "Forward from base"),
            ("Y →", BLUE,   "Left from base"),
            ("Z ↑", YELLOW, "Up (vertical)"),
        ]:
            f = tk.Frame(ref_row, bg=CARD)
            f.pack(side="left", padx=16)
            lbl(f, axis, size=10, color=color, bold=True, bg=CARD).pack()
            lbl(f, desc, size=7, color=MUTED, bg=CARD).pack()

    # ── Connection ────────────────────────────────────────────────────────────

    def _connect(self):
        try:
            self.bus = CANBus(simulated=self._sim, channel=self._port,
                              bustype=self._bus_t, num_motors=6)
            self.robot = RobotController(self.bus, DEFAULT_JOINT_CONFIG)
            self.robot.start()
            self.lbl_conn.config(text="● CONNECTED", fg=ACCENT)
            self._start_monitor()
        except Exception as e:
            messagebox.showerror("Connection Failed", str(e))

    def _disconnect(self):
        self._stop_monitor()
        if self.robot:
            self.robot.close()
            self.robot = None
        self.bus = None
        self.lbl_conn.config(text="● DISCONNECTED", fg=RED)

    # ── Monitor loop ──────────────────────────────────────────────────────────

    def _get_angles(self):
        if self.robot:
            return np.array([
                self.robot.motors[cfg.motor_id].get_position()
                for cfg in DEFAULT_JOINT_CONFIG], dtype=float)
        return np.zeros(6)

    def _start_monitor(self):
        self._monitor_job = self.after(100, self._monitor_tick)

    def _stop_monitor(self):
        if self._monitor_job:
            self.after_cancel(self._monitor_job)
            self._monitor_job = None

    def _monitor_tick(self):
        """Updates visualizer with live joint angles at ~10 Hz."""
        angles = self._get_angles()
        self.visualizer.set_joint_angles(angles)
        self._monitor_job = self.after(100, self._monitor_tick)

    # ── Plan callback — show ghost target on visualizer ───────────────────────

    def _on_plan_ready(self, plan: CartesianMotionPlan):
        """Called by InputPanel when IK completes. Shows ghost arm."""
        self.visualizer.set_target(
            plan.target_position_mm,
            plan.target_angles_deg)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def _on_close(self):
        self._stop_monitor()
        if self.robot:
            try:
                self.robot.close()
            except:
                pass
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cartesian Control + Live Arm Visualizer")
    parser.add_argument("--real",    action="store_true")
    parser.add_argument("--port",    default="COM3")
    parser.add_argument("--bustype", default="slcan")
    args = parser.parse_args()

    app = CartesianVisApp(
        simulated=not args.real,
        port=args.port,
        bustype=args.bustype,
    )
    app.mainloop()