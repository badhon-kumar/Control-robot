"""
cartesian_gui.py
================
Cartesian Motion Control Panel for RMD-X8-120 6-Axis Robot Arm

This is a standalone window that plugs into your existing system.
It adds X/Y/Z displacement + time control on top of the joint angle system.

Run:
    python3.10 cartesian_gui.py           ← standalone (simulation mode)
    python3.10 cartesian_gui.py --real    ← with real hardware

Or launch from within gui.py by importing CartesianPanel.

Requires: numpy (pip install numpy)
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import logging
import queue
import argparse
import numpy as np
from typing import Optional

from can_interface import CANBus
from robot_controller import RobotController, DEFAULT_JOINT_CONFIG
from kinematics import (
    forward_kinematics, plan_cartesian_motion,
    CartesianMotionPlan, JOINT_SPEED_LIMITS_DPS
)

# ── Logging → queue ───────────────────────────────────────────────────────────
log_queue: queue.Queue = queue.Queue()

class QueueHandler(logging.Handler):
    def emit(self, record):
        log_queue.put(self.format(record))

# ── Theme ─────────────────────────────────────────────────────────────────────
BG       = "#0d1117"
PANEL    = "#161b22"
CARD     = "#1c2128"
INPUT    = "#21262d"
BORDER   = "#30363d"
ACCENT   = "#00d4aa"
ACCENT2  = "#4dabf7"
RED      = "#ff4444"
YELLOW   = "#e3b341"
GREEN    = "#3fb950"
MUTED    = "#7d8590"
TEXT     = "#e6edf3"
JCOLORS  = ["#00d4aa","#4dabf7","#e3b341","#ff7eb6","#a5d6ff","#c3e88d"]


# ══════════════════════════════════════════════════════════════════════════════
# HELPER WIDGETS
# ══════════════════════════════════════════════════════════════════════════════

def label(parent, text, size=8, color=MUTED, bold=False, **kw):
    weight = "bold" if bold else "normal"
    return tk.Label(parent, text=text, font=("Courier New", size, weight),
                    fg=color, bg=kw.pop("bg", CARD), **kw)

def entry(parent, var, width=10, **kw):
    return tk.Entry(parent, textvariable=var, width=width,
                    font=("Courier New", 10), bg=INPUT, fg=TEXT,
                    insertbackground=TEXT, relief="flat",
                    highlightbackground=BORDER, highlightthickness=1, **kw)

def btn(parent, text, color, fg=BG, cmd=None, **kw):
    return tk.Button(parent, text=text, font=("Courier New", 9, "bold"),
                     bg=color, fg=fg, activebackground=color,
                     relief="flat", cursor="hand2",
                     command=cmd, padx=10, pady=5, **kw)


# ══════════════════════════════════════════════════════════════════════════════
# JOINT SPEED BAR  — visual per-joint speed indicator
# ══════════════════════════════════════════════════════════════════════════════

class JointSpeedBar(tk.Frame):
    """Shows 6 horizontal bars representing each joint's computed speed."""

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=CARD, **kw)
        self.bars   = []
        self.labels = []
        self.speed_labels = []

        label(self, "COMPUTED JOINT SPEEDS", size=7, color=MUTED,
              bg=CARD).pack(anchor="w", padx=12, pady=(10,4))

        for i in range(6):
            row = tk.Frame(self, bg=CARD)
            row.pack(fill="x", padx=12, pady=1)

            label(row, f"J{i+1}", size=8, color=JCOLORS[i],
                  bold=True, bg=CARD, width=4).pack(side="left")

            # Bar background
            bar_bg = tk.Frame(row, bg=INPUT, height=14, width=260)
            bar_bg.pack(side="left", padx=4)
            bar_bg.pack_propagate(False)

            # Bar fill
            bar_fill = tk.Frame(bar_bg, bg=JCOLORS[i], height=14, width=0)
            bar_fill.place(x=0, y=0, relheight=1)
            self.bars.append((bar_bg, bar_fill))

            # Speed text
            spd_lbl = label(row, "0.0°/s", size=8, color=TEXT, bg=CARD)
            spd_lbl.pack(side="left", padx=6)
            self.speed_labels.append(spd_lbl)

            # Limit warning
            warn = label(row, "", size=7, color=RED, bg=CARD)
            warn.pack(side="left")
            self.labels.append(warn)

    def update(self, speeds_dps: np.ndarray):
        """Update bars with computed speeds. speeds_dps: array of 6 floats."""
        for i in range(6):
            spd   = speeds_dps[i]
            limit = JOINT_SPEED_LIMITS_DPS[i]
            ratio = min(spd / limit, 1.0)

            bar_bg, bar_fill = self.bars[i]
            bar_width = int(bar_bg.winfo_width() * ratio) if bar_bg.winfo_width() > 1 else int(260 * ratio)
            bar_fill.place(x=0, y=0, width=bar_width, relheight=1)

            color = RED if ratio > 0.9 else (YELLOW if ratio > 0.7 else JCOLORS[i])
            bar_fill.config(bg=color)

            self.speed_labels[i].config(text=f"{spd:6.1f}°/s")
            self.labels[i].config(text="⚠ AT LIMIT" if ratio >= 1.0 else "")

    def clear(self):
        for i in range(6):
            _, bar_fill = self.bars[i]
            bar_fill.place(x=0, y=0, width=0, relheight=1)
            self.speed_labels[i].config(text="—")
            self.labels[i].config(text="")


# ══════════════════════════════════════════════════════════════════════════════
# FK LIVE DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

class FKDisplay(tk.Frame):
    """Shows live end-effector X, Y, Z position computed from current joint angles."""

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=CARD, highlightbackground=BORDER,
                         highlightthickness=1, **kw)

        label(self, "LIVE END-EFFECTOR POSITION (FK)", size=7,
              color=MUTED, bg=CARD).pack(anchor="w", padx=12, pady=(10,4))

        coords = tk.Frame(self, bg=CARD)
        coords.pack(padx=12, pady=(0,10))

        self.x_lbl = self._coord(coords, "X", ACCENT)
        self.y_lbl = self._coord(coords, "Y", ACCENT2)
        self.z_lbl = self._coord(coords, "Z", YELLOW)

        label(self, "mm from base origin", size=7, color=MUTED,
              bg=CARD).pack(pady=(0,8))

    def _coord(self, parent, axis, color):
        f = tk.Frame(parent, bg=CARD)
        f.pack(side="left", padx=16)
        label(f, axis, size=9, color=color, bold=True, bg=CARD).pack()
        lbl = tk.Label(f, text="---", font=("Courier New", 18, "bold"),
                       fg=color, bg=CARD)
        lbl.pack()
        label(f, "mm", size=7, color=MUTED, bg=CARD).pack()
        return lbl

    def update(self, x_mm, y_mm, z_mm):
        self.x_lbl.config(text=f"{x_mm:+.1f}")
        self.y_lbl.config(text=f"{y_mm:+.1f}")
        self.z_lbl.config(text=f"{z_mm:+.1f}")

    def clear(self):
        for lbl in (self.x_lbl, self.y_lbl, self.z_lbl):
            lbl.config(text="---")


# ══════════════════════════════════════════════════════════════════════════════
# MOTION PLAN RESULT TABLE
# ══════════════════════════════════════════════════════════════════════════════

class PlanResultTable(tk.Frame):
    """Displays the computed motion plan — angles and speeds per joint."""

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=CARD, highlightbackground=BORDER,
                         highlightthickness=1, **kw)

        label(self, "MOTION PLAN PREVIEW", size=7, color=MUTED,
              bg=CARD).pack(anchor="w", padx=12, pady=(10,4))

        # Header
        hdr = tk.Frame(self, bg=PANEL)
        hdr.pack(fill="x", padx=12)
        for col, w in [("Joint",50),("Current",90),("Target",90),("Δ Angle",80),("Speed",90),("Status",80)]:
            label(hdr, col, size=7, color=MUTED, bold=True, bg=PANEL,
                  width=w//7).pack(side="left", padx=2, pady=3)

        # Rows
        self.rows = []
        for i in range(6):
            row = tk.Frame(self, bg=CARD)
            row.pack(fill="x", padx=12, pady=1)
            cells = {}
            cells["joint"]   = label(row, f"J{i+1}", size=8, color=JCOLORS[i], bold=True, bg=CARD, width=4)
            cells["current"] = label(row, "---",     size=8, color=MUTED,       bg=CARD, width=9)
            cells["target"]  = label(row, "---",     size=8, color=TEXT,        bg=CARD, width=9)
            cells["delta"]   = label(row, "---",     size=8, color=TEXT,        bg=CARD, width=8)
            cells["speed"]   = label(row, "---",     size=8, color=ACCENT,      bg=CARD, width=9)
            cells["status"]  = label(row, "",        size=7, color=GREEN,       bg=CARD, width=8)
            for c in cells.values():
                c.pack(side="left", padx=2)
            self.rows.append(cells)

        self.lbl_summary = label(self, "", size=8, color=MUTED, bg=CARD)
        self.lbl_summary.pack(pady=(4,8), padx=12, anchor="w")

    def update(self, plan: CartesianMotionPlan):
        for i in range(6):
            r = self.rows[i]
            limit = JOINT_SPEED_LIMITS_DPS[i]
            spd   = plan.speeds_dps[i]
            at_limit = spd >= limit * 0.99

            r["current"].config(text=f"{plan.start_angles_deg[i]:+.1f}°")
            r["target"].config(text=f"{plan.target_angles_deg[i]:+.1f}°")
            r["delta"].config(text=f"{plan.angle_changes_deg[i]:.1f}°")
            r["speed"].config(text=f"{spd:.1f}°/s",
                              fg=RED if at_limit else ACCENT)
            r["status"].config(text="⚠CLAMPED" if at_limit else "✓ OK",
                               fg=RED if at_limit else GREEN)

        ik_color = GREEN if plan.ik_error_mm < 1.0 else (YELLOW if plan.ik_error_mm < 5.0 else RED)
        self.lbl_summary.config(
            text=f"IK accuracy: {plan.ik_error_mm:.2f}mm  |  "
                 f"Fastest joint: {plan.max_speed_dps:.1f}°/s  |  "
                 f"Duration: {plan.time_s:.2f}s",
            fg=ik_color,
        )

    def clear(self):
        for r in self.rows:
            for key in ("current","target","delta","speed"):
                r[key].config(text="---", fg=MUTED)
            r["status"].config(text="")
        self.lbl_summary.config(text="")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CARTESIAN CONTROL PANEL
# ══════════════════════════════════════════════════════════════════════════════

class CartesianPanel(tk.Frame):
    """
    The main Cartesian motion panel. Can be embedded inside gui.py
    or used standalone (see CartesianWindow below).
    """

    def __init__(self, parent, robot: RobotController = None, **kw):
        super().__init__(parent, bg=BG, **kw)
        self.robot        = robot
        self._plan:  Optional[CartesianMotionPlan] = None
        self._executing   = False
        self._monitor_job = None

        self._build()
        if robot:
            self._start_fk_monitor()

    def set_robot(self, robot: RobotController):
        """Called when connection is established (for embedded use)."""
        self._stop_fk_monitor()
        self.robot = robot
        self._start_fk_monitor()

    # ── UI Build ──────────────────────────────────────────────────────────────

    def _build(self):

        # ── Title ─────────────────────────────────────────────────────────────
        title = tk.Frame(self, bg=PANEL, height=44)
        title.pack(fill="x")
        title.pack_propagate(False)
        label(title, "⊕  CARTESIAN MOTION CONTROL", size=13, color=ACCENT,
              bold=True, bg=PANEL).pack(side="left", padx=16, pady=8)
        label(title, "Time-Parameterized  ·  Auto Speed  ·  IK Solver",
              size=8, color=MUTED, bg=PANEL).pack(side="left", pady=14)

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # ── Main content area ─────────────────────────────────────────────────
        content = tk.Frame(self, bg=BG)
        content.pack(fill="both", expand=True, padx=16, pady=12)

        left  = tk.Frame(content, bg=BG)
        right = tk.Frame(content, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(0,8))
        right.pack(side="right", fill="both", expand=True, padx=(8,0))

        # ── LEFT: FK display + Input ───────────────────────────────────────────
        self.fk_display = FKDisplay(left)
        self.fk_display.pack(fill="x", pady=(0,10))

        # Input card
        input_card = tk.Frame(left, bg=CARD, highlightbackground=BORDER,
                              highlightthickness=1)
        input_card.pack(fill="x", pady=(0,10))

        label(input_card, "MOTION INPUT", size=7, color=MUTED,
              bg=CARD).pack(anchor="w", padx=12, pady=(10,6))

        # Mode selector
        mode_row = tk.Frame(input_card, bg=CARD)
        mode_row.pack(fill="x", padx=12, pady=(0,8))
        label(mode_row, "MODE:", size=8, color=MUTED, bg=CARD).pack(side="left")
        self.mode_var = tk.StringVar(value="Relative")
        for m in ("Relative", "Absolute"):
            tk.Radiobutton(mode_row, text=m, variable=self.mode_var, value=m,
                           font=("Courier New", 9), bg=CARD, fg=TEXT,
                           selectcolor=INPUT, activebackground=CARD,
                           activeforeground=TEXT,
                           command=self._on_mode_change).pack(side="left", padx=8)

        # Coordinate inputs — X Y Z
        coords_frame = tk.Frame(input_card, bg=CARD)
        coords_frame.pack(fill="x", padx=12, pady=4)

        self.x_var = tk.StringVar(value="0.0")
        self.y_var = tk.StringVar(value="0.0")
        self.z_var = tk.StringVar(value="0.0")
        self.t_var = tk.StringVar(value="3.0")

        # Header labels
        self.coord_header = label(coords_frame,
            "DISPLACEMENT  (mm from current position)",
            size=7, color=MUTED, bg=CARD)
        self.coord_header.grid(row=0, column=0, columnspan=6, sticky="w", pady=(0,4))

        axes = [
            ("X", self.x_var, ACCENT),
            ("Y", self.y_var, ACCENT2),
            ("Z", self.z_var, YELLOW),
        ]
        self.axis_labels = []
        self.axis_entries = []
        for col, (axis, var, color) in enumerate(axes):
            lbl = label(coords_frame, axis, size=11, color=color, bold=True, bg=CARD)
            lbl.grid(row=1, column=col*2, padx=(8,2), pady=4)
            self.axis_labels.append(lbl)

            e = entry(coords_frame, var, width=9)
            e.grid(row=1, column=col*2+1, padx=(0,12))
            self.axis_entries.append(e)

        label(coords_frame, "mm", size=8, color=MUTED,
              bg=CARD).grid(row=1, column=6, sticky="w")

        # Time input (independent variable)
        time_frame = tk.Frame(input_card, bg=CARD)
        time_frame.pack(fill="x", padx=12, pady=(4,12))

        tk.Frame(time_frame, bg=BORDER, height=1).pack(fill="x", pady=6)

        time_row = tk.Frame(time_frame, bg=CARD)
        time_row.pack(fill="x")
        label(time_row, "T", size=11, color="#ff7eb6", bold=True,
              bg=CARD).pack(side="left", padx=(8,4))
        entry(time_row, self.t_var, width=9).pack(side="left")
        label(time_row, "seconds  (motion duration — independent variable)",
              size=8, color=MUTED, bg=CARD).pack(side="left", padx=8)

        label(input_card,
              "Speed is auto-computed per joint so all axes finish simultaneously.",
              size=7, color=MUTED, bg=CARD).pack(anchor="w", padx=12, pady=(0,10))

        # ── Action buttons ────────────────────────────────────────────────────
        act = tk.Frame(left, bg=BG)
        act.pack(fill="x", pady=4)

        self.btn_compute = btn(act, "  COMPUTE PLAN  ", ACCENT2, fg=BG,
                               cmd=self._on_compute)
        self.btn_compute.pack(side="left", padx=(0,6))

        self.btn_execute = btn(act, "  ▶  EXECUTE  ", ACCENT, fg=BG,
                               cmd=self._on_execute, state="disabled")
        self.btn_execute.pack(side="left", padx=(0,6))

        self.btn_stop = btn(act, "  ■  STOP  ", RED, fg=TEXT,
                            cmd=self._on_stop)
        self.btn_stop.pack(side="left", padx=(0,6))

        self.btn_home = btn(act, "HOME", PANEL, fg=MUTED,
                            cmd=self._on_home)
        self.btn_home.pack(side="left")

        # Status bar
        self.lbl_exec_status = label(left, "", size=9, color=MUTED, bg=BG)
        self.lbl_exec_status.pack(anchor="w", pady=4)

        # Progress bar
        self.progress_var = tk.DoubleVar(value=0)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Cart.Horizontal.TProgressbar",
                        troughcolor=INPUT, background=ACCENT,
                        bordercolor=BORDER, lightcolor=ACCENT, darkcolor=ACCENT)
        self.progress = ttk.Progressbar(left, variable=self.progress_var,
                                        maximum=100, length=400,
                                        style="Cart.Horizontal.TProgressbar")
        self.progress.pack(fill="x", pady=2)

        # ── RIGHT: Plan table + speed bars + log ──────────────────────────────
        self.plan_table = PlanResultTable(right)
        self.plan_table.pack(fill="x", pady=(0,10))

        self.speed_bars = JointSpeedBar(right)
        self.speed_bars.pack(fill="x", pady=(0,10))

        # Mini log
        log_card = tk.Frame(right, bg=CARD, highlightbackground=BORDER,
                            highlightthickness=1)
        log_card.pack(fill="both", expand=True)

        lh = tk.Frame(log_card, bg=PANEL)
        lh.pack(fill="x")
        label(lh, "SOLVER LOG", size=7, color=MUTED, bg=PANEL).pack(side="left",
                                                                      padx=8, pady=3)
        tk.Button(lh, text="CLR", font=("Courier New",7), bg=PANEL, fg=MUTED,
                  relief="flat", cursor="hand2",
                  command=lambda: (self.log_box.config(state="normal"),
                                   self.log_box.delete("1.0","end"),
                                   self.log_box.config(state="disabled"))
                  ).pack(side="right", padx=6)

        self.log_box = scrolledtext.ScrolledText(
            log_card, height=10, font=("Courier New", 8),
            bg=CARD, fg=MUTED, relief="flat", state="disabled", wrap="word",
        )
        self.log_box.pack(fill="both", expand=True, padx=4, pady=4)
        self.log_box.tag_config("ok",   foreground=GREEN)
        self.log_box.tag_config("warn", foreground=YELLOW)
        self.log_box.tag_config("err",  foreground=RED)
        self.log_box.tag_config("info", foreground=MUTED)

        self._poll_log()

    # ── Mode toggle ───────────────────────────────────────────────────────────

    def _on_mode_change(self):
        if self.mode_var.get() == "Absolute":
            self.coord_header.config(text="ABSOLUTE TARGET POSITION  (mm from base)")
        else:
            self.coord_header.config(text="DISPLACEMENT  (mm from current position)")
        self._plan = None
        self.btn_execute.config(state="disabled")
        self.plan_table.clear()
        self.speed_bars.clear()

    # ── Get current joint angles ──────────────────────────────────────────────

    def _get_current_angles(self) -> np.ndarray:
        """Read current angles from robot or return zeros in simulation."""
        if self.robot:
            return np.array([
                self.robot.motors[cfg.motor_id].get_position()
                for cfg in DEFAULT_JOINT_CONFIG
            ], dtype=float)
        return np.zeros(6)

    # ── Parse inputs ─────────────────────────────────────────────────────────

    def _parse_inputs(self):
        try:
            x = float(self.x_var.get())
            y = float(self.y_var.get())
            z = float(self.z_var.get())
            t = float(self.t_var.get())
        except ValueError:
            messagebox.showerror("Input Error",
                                 "X, Y, Z, and T must all be numbers.\n"
                                 "Example: X=50, Y=0, Z=-30, T=3.0")
            return None
        if t <= 0:
            messagebox.showerror("Input Error", "Time T must be greater than 0.")
            return None
        return x, y, z, t

    # ── Compute plan ─────────────────────────────────────────────────────────

    def _on_compute(self):
        vals = self._parse_inputs()
        if vals is None:
            return
        x, y, z, t = vals
        current = self._get_current_angles()

        self._log(f"Computing IK for {'Δ' if self.mode_var.get()=='Relative' else '→'}"
                  f"({x:+.1f}, {y:+.1f}, {z:+.1f}) mm  T={t}s ...", "info")

        self.btn_compute.config(state="disabled", text="COMPUTING...")
        threading.Thread(target=self._compute_thread,
                         args=(x, y, z, t, current), daemon=True).start()

    def _compute_thread(self, x, y, z, t, current):
        mode = self.mode_var.get().lower()
        if mode == "absolute":
            plan = plan_cartesian_motion(
                displacement_mm=[0,0,0], time_s=t,
                current_angles_deg=current,
                mode="absolute", absolute_target_mm=[x, y, z],
            )
        else:
            plan = plan_cartesian_motion(
                displacement_mm=[x, y, z], time_s=t,
                current_angles_deg=current,
                mode="relative",
            )
        self.after(0, lambda: self._on_plan_ready(plan))

    def _on_plan_ready(self, plan: CartesianMotionPlan):
        self.btn_compute.config(state="normal", text="  COMPUTE PLAN  ")
        self._plan = plan

        if plan.success:
            self.plan_table.update(plan)
            self.speed_bars.update(plan.speeds_dps)
            self.btn_execute.config(state="normal")
            color = GREEN if plan.ik_error_mm < 1.0 else YELLOW
            self._log(f"✓ Plan ready — IK error: {plan.ik_error_mm:.2f}mm  "
                      f"Max speed: {plan.max_speed_dps:.1f}°/s", "ok")
            if "CLAMPED" in plan.message:
                self._log(f"⚠ {plan.message}", "warn")
            self.lbl_exec_status.config(
                text=f"Plan ready: {plan.time_s:.1f}s  |  "
                     f"IK accuracy: {plan.ik_error_mm:.2f}mm  |  "
                     f"Max joint speed: {plan.max_speed_dps:.1f}°/s",
                fg=color,
            )
        else:
            self.plan_table.clear()
            self.speed_bars.clear()
            self.btn_execute.config(state="disabled")
            self._log(f"✗ {plan.message}", "err")
            self.lbl_exec_status.config(text=f"Plan failed: {plan.message}", fg=RED)
            messagebox.showerror("IK Failed", plan.message)

    # ── Execute plan ──────────────────────────────────────────────────────────

    def _on_execute(self):
        if self._plan is None or not self._plan.success:
            return
        if self.robot is None:
            messagebox.showwarning("Not Connected",
                                   "No robot connected.\n"
                                   "Connect first, then compute and execute.")
            return
        if self._executing:
            return

        self._executing = True
        self.btn_execute.config(state="disabled", text="EXECUTING...")
        self.progress_var.set(0)

        self._log(f"▶ Executing — {self._plan.time_s:.1f}s motion...", "ok")
        threading.Thread(target=self._execute_thread,
                         args=(self._plan,), daemon=True).start()

    def _execute_thread(self, plan: CartesianMotionPlan):
        """Send motion commands and track progress."""
        # Build position + per-joint-speed dict
        positions = {
            cfg.motor_id: float(plan.target_angles_deg[i])
            for i, cfg in enumerate(DEFAULT_JOINT_CONFIG)
        }

        # Send all joints simultaneously with their computed speeds
        for i, cfg in enumerate(DEFAULT_JOINT_CONFIG):
            self.robot.motors[cfg.motor_id].set_position(
                position_deg=float(plan.target_angles_deg[i]),
                max_speed_dps=float(plan.speeds_dps[i]),
                wait=False,
            )
            time.sleep(0.001)

        # Progress tracking — update bar over the planned duration
        start_time = time.time()
        duration   = plan.time_s

        while True:
            elapsed = time.time() - start_time
            progress = min(elapsed / duration * 100, 100)
            self.after(0, lambda p=progress: self.progress_var.set(p))

            if elapsed >= duration + 0.5:   # Extra 0.5s settling time
                break
            time.sleep(0.05)

        # Final FK read
        final_angles = self._get_current_angles()
        fk = forward_kinematics(final_angles)
        p  = fk.position_mm()

        self.after(0, lambda: self._on_execute_done(p))

    def _on_execute_done(self, final_pos_mm):
        self._executing = False
        self.progress_var.set(100)
        self.btn_execute.config(state="normal", text="  ▶  EXECUTE  ")
        target = self._plan.target_position_mm
        err = np.linalg.norm(final_pos_mm - target)
        self._log(
            f"✓ Motion complete.  Final pos: "
            f"X={final_pos_mm[0]:+.1f}  Y={final_pos_mm[1]:+.1f}  Z={final_pos_mm[2]:+.1f} mm  "
            f"(error from target: {err:.1f}mm)", "ok"
        )
        self.lbl_exec_status.config(
            text=f"Done. Final position error from target: {err:.1f}mm",
            fg=GREEN if err < 5 else YELLOW,
        )

    # ── Stop / Home ───────────────────────────────────────────────────────────

    def _on_stop(self):
        if self.robot:
            threading.Thread(target=self.robot.stop_all, daemon=True).start()
        self._executing = False
        self.progress_var.set(0)
        self.btn_execute.config(state="normal", text="  ▶  EXECUTE  ")
        self._log("■ Stopped.", "warn")

    def _on_home(self):
        if self.robot:
            threading.Thread(
                target=self.robot.go_home, kwargs={"speed_dps": 60.0, "wait": False},
                daemon=True
            ).start()
            self._log("→ Homing all joints.", "info")

    # ── Live FK monitor ───────────────────────────────────────────────────────

    def _start_fk_monitor(self):
        self._monitor_job = self.after(300, self._fk_tick)

    def _stop_fk_monitor(self):
        if self._monitor_job:
            self.after_cancel(self._monitor_job)
            self._monitor_job = None

    def _fk_tick(self):
        if self.robot:
            angles = self._get_current_angles()
            fk = forward_kinematics(angles)
            p  = fk.position_mm()
            self.fk_display.update(p[0], p[1], p[2])
        self._monitor_job = self.after(300, self._fk_tick)

    # ── Log ───────────────────────────────────────────────────────────────────

    def _log(self, msg, tag="info"):
        ts = time.strftime("%H:%M:%S")
        self.log_box.config(state="normal")
        self.log_box.insert("end", f"{ts}  {msg}\n", tag)
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def _poll_log(self):
        while not log_queue.empty():
            msg = log_queue.get_nowait()
            tag = "warn" if "WARNING" in msg else ("err" if "ERROR" in msg else "info")
            self._log(msg, tag)
        self.after(150, self._poll_log)

    def destroy(self):
        self._stop_fk_monitor()
        super().destroy()


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE WINDOW WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class CartesianWindow(tk.Tk):
    """
    Full standalone window — use this when running cartesian_gui.py directly.
    Includes its own connect panel so you don't need gui.py.
    """

    def __init__(self, simulated=True, port="COM3", bustype="slcan"):
        super().__init__()
        self.title("RMD-X8-120  ·  Cartesian Motion Control")
        self.configure(bg=BG)
        self.minsize(1100, 780)

        self.robot = None
        self.bus   = None
        self._sim  = simulated
        self._port = port
        self._bus_t = bustype

        # Logging
        handler = QueueHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

        self._build_connect_bar()
        self.panel = CartesianPanel(self)
        self.panel.pack(fill="both", expand=True)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Auto-connect in simulation mode
        if simulated:
            self.after(300, self._connect)

    def _build_connect_bar(self):
        bar = tk.Frame(self, bg=PANEL, height=40)
        bar.pack(fill="x")
        bar.pack_propagate(False)

        self.lbl_conn = label(bar, "● DISCONNECTED", size=9, color=RED,
                              bold=True, bg=PANEL)
        self.lbl_conn.pack(side="right", padx=16, pady=8)

        mode_txt = "SIMULATION" if self._sim else f"REAL  [{self._port}]"
        label(bar, f"MODE: {mode_txt}", size=8, color=MUTED,
              bg=PANEL).pack(side="left", padx=16, pady=10)

        btn(bar, "CONNECT", ACCENT, fg=BG,
            cmd=self._connect).pack(side="left", padx=4)
        btn(bar, "DISCONNECT", PANEL, fg=MUTED,
            cmd=self._disconnect).pack(side="left", padx=4)

    def _connect(self):
        try:
            self.bus = CANBus(simulated=self._sim, channel=self._port,
                              bustype=self._bus_t, num_motors=6)
            self.robot = RobotController(self.bus, DEFAULT_JOINT_CONFIG)
            self.robot.start()
            self.panel.set_robot(self.robot)
            self.lbl_conn.config(text="● CONNECTED", fg=ACCENT)
        except Exception as e:
            messagebox.showerror("Connection Failed", str(e))

    def _disconnect(self):
        if self.robot:
            self.robot.close()
            self.robot = None
        self.bus = None
        self.panel.robot = None
        self.lbl_conn.config(text="● DISCONNECTED", fg=RED)

    def _on_close(self):
        if self.robot:
            try: self.robot.close()
            except: pass
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cartesian Motion Control GUI")
    parser.add_argument("--real",    action="store_true")
    parser.add_argument("--port",    default="COM3")
    parser.add_argument("--bustype", default="slcan")
    args = parser.parse_args()

    app = CartesianWindow(
        simulated=not args.real,
        port=args.port,
        bustype=args.bustype,
    )
    app.mainloop()
