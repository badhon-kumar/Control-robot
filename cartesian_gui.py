"""
cartesian_gui.py  (v2 — Multi-Mode Control Panel)
==================================================
Three fully independent control modes, switchable anytime:

  TAB 1 — CARTESIAN   : Set X/Y/Z displacement or absolute target + Time
                         Speeds auto-computed per joint via IK
  TAB 2 — JOINT ANGLE : Set each motor's angle individually + per-motor speed
  TAB 3 — VELOCITY    : Set each motor's continuous spin velocity independently

Run:
    python3.10 cartesian_gui.py              ← simulation
    python3.10 cartesian_gui.py --real --port /dev/ttyACM0
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
BG      = "#0d1117"
PANEL   = "#161b22"
CARD    = "#1c2128"
INPUT   = "#21262d"
BORDER  = "#30363d"
ACCENT  = "#00d4aa"
BLUE    = "#4dabf7"
RED     = "#ff4444"
YELLOW  = "#e3b341"
ORANGE  = "#f0883e"
GREEN   = "#3fb950"
MUTED   = "#7d8590"
TEXT    = "#e6edf3"
JCOLORS = ["#00d4aa","#4dabf7","#e3b341","#ff7eb6","#a5d6ff","#c3e88d"]
JOINT_NAMES = ["Base", "Shoulder", "Elbow", "Wrist1", "Wrist2", "Tool"]


# ══════════════════════════════════════════════════════════════════════════════
# SHARED HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def mk_label(parent, text, size=8, color=MUTED, bold=False, bg=CARD, **kw):
    return tk.Label(parent, text=text,
                    font=("Courier New", size, "bold" if bold else "normal"),
                    fg=color, bg=bg, **kw)

def mk_entry(parent, var, width=9, **kw):
    return tk.Entry(parent, textvariable=var, width=width,
                    font=("Courier New", 10), bg=INPUT, fg=TEXT,
                    insertbackground=TEXT, relief="flat",
                    highlightbackground=BORDER, highlightthickness=1, **kw)

def mk_btn(parent, text, color=ACCENT, fg=BG, cmd=None, width=None, **kw):
    b = tk.Button(parent, text=text,
                  font=("Courier New", 9, "bold"),
                  bg=color, fg=fg, activebackground=color,
                  relief="flat", cursor="hand2",
                  command=cmd, padx=10, pady=5, **kw)
    if width:
        b.config(width=width)
    return b

def sep(parent, bg=BORDER, h=1, pady=4):
    tk.Frame(parent, bg=bg, height=h).pack(fill="x", pady=pady)


# ══════════════════════════════════════════════════════════════════════════════
# FK LIVE DISPLAY  (shared across all tabs)
# ══════════════════════════════════════════════════════════════════════════════

class FKDisplay(tk.Frame):
    def __init__(self, parent, **kw):
        super().__init__(parent, bg=CARD,
                         highlightbackground=BORDER, highlightthickness=1, **kw)
        mk_label(self, "LIVE END-EFFECTOR POSITION  (Forward Kinematics)",
                 size=7, color=MUTED, bg=CARD).pack(anchor="w", padx=12, pady=(8,4))
        row = tk.Frame(self, bg=CARD)
        row.pack(padx=12, pady=(0,8))
        self.x_lbl = self._axis(row, "X", ACCENT)
        self.y_lbl = self._axis(row, "Y", BLUE)
        self.z_lbl = self._axis(row, "Z", YELLOW)
        mk_label(self, "mm from base origin  ·  updates every 300 ms",
                 size=6, color=MUTED, bg=CARD).pack(pady=(0,6))

    def _axis(self, parent, name, color):
        f = tk.Frame(parent, bg=CARD)
        f.pack(side="left", padx=20)
        mk_label(f, name, size=10, color=color, bold=True, bg=CARD).pack()
        lbl = tk.Label(f, text="---", font=("Courier New", 20, "bold"),
                       fg=color, bg=CARD, width=7)
        lbl.pack()
        mk_label(f, "mm", size=7, color=MUTED, bg=CARD).pack()
        return lbl

    def update(self, x, y, z):
        self.x_lbl.config(text=f"{x:+.1f}")
        self.y_lbl.config(text=f"{y:+.1f}")
        self.z_lbl.config(text=f"{z:+.1f}")

    def reset(self):
        for l in (self.x_lbl, self.y_lbl, self.z_lbl):
            l.config(text="---")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CARTESIAN CONTROL
# ══════════════════════════════════════════════════════════════════════════════

class CartesianTab(tk.Frame):
    def __init__(self, parent, get_robot, get_angles, log_fn, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._get_robot  = get_robot
        self._get_angles = get_angles
        self._log        = log_fn
        self._plan: Optional[CartesianMotionPlan] = None
        self._executing  = False
        self._build()

    def _build(self):
        left = tk.Frame(self, bg=BG)
        left.pack(side="left", fill="both", expand=True, padx=(12,6), pady=10)
        right = tk.Frame(self, bg=BG)
        right.pack(side="right", fill="both", expand=True, padx=(6,12), pady=10)

        # Mode selector
        mc = tk.Frame(left, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        mc.pack(fill="x", pady=(0,8))
        mk_label(mc, "DISPLACEMENT MODE", size=7, bg=CARD).pack(anchor="w", padx=12, pady=(8,4))
        self.mode_var = tk.StringVar(value="Relative")
        for m, desc in [("Relative", "Move by ΔX ΔY ΔZ from current position"),
                        ("Absolute", "Go to exact X Y Z coordinate from base")]:
            tk.Radiobutton(mc, text=f"  {m}  —  {desc}",
                           variable=self.mode_var, value=m,
                           font=("Courier New", 8), bg=CARD, fg=TEXT,
                           selectcolor=INPUT, activebackground=CARD,
                           activeforeground=TEXT,
                           command=self._on_mode_change).pack(anchor="w", padx=12)
        mk_label(mc, "", size=4, bg=CARD).pack()

        # X Y Z inputs
        ic = tk.Frame(left, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        ic.pack(fill="x", pady=(0,8))
        self.inp_title = mk_label(ic, "TARGET DISPLACEMENT  (mm)", size=7, bg=CARD)
        self.inp_title.pack(anchor="w", padx=12, pady=(8,6))

        xyz = tk.Frame(ic, bg=CARD)
        xyz.pack(pady=4)
        self.x_var = tk.StringVar(value="0.0")
        self.y_var = tk.StringVar(value="0.0")
        self.z_var = tk.StringVar(value="0.0")
        for axis, var, color in [("X", self.x_var, ACCENT),
                                  ("Y", self.y_var, BLUE),
                                  ("Z", self.z_var, YELLOW)]:
            f = tk.Frame(xyz, bg=CARD)
            f.pack(side="left", padx=14)
            mk_label(f, axis, size=16, color=color, bold=True, bg=CARD).pack()
            mk_entry(f, var, width=10).pack()
            mk_label(f, "mm", size=7, bg=CARD).pack()

        sep(ic, pady=4)

        # Time input
        tr = tk.Frame(ic, bg=CARD)
        tr.pack(padx=12, pady=(0,10))
        mk_label(tr, "T", size=16, color="#ff7eb6", bold=True, bg=CARD).pack(side="left", padx=(0,8))
        self.t_var = tk.StringVar(value="3.0")
        mk_entry(tr, self.t_var, width=8).pack(side="left")
        mk_label(tr,
                 "  seconds  ←  independent variable\n"
                 "  Joint speeds auto-computed so all\n"
                 "  axes finish at exactly this time.",
                 size=7, color=MUTED, bg=CARD).pack(side="left", padx=10)

        # Buttons
        br = tk.Frame(left, bg=BG)
        br.pack(fill="x", pady=4)
        self.btn_compute = mk_btn(br, "COMPUTE PLAN", BLUE, fg=BG, cmd=self._on_compute)
        self.btn_compute.pack(side="left", padx=(0,6))
        self.btn_exec = mk_btn(br, "▶  EXECUTE", ACCENT, fg=BG,
                               cmd=self._on_execute, state="disabled")
        self.btn_exec.pack(side="left", padx=(0,6))
        mk_btn(br, "■  STOP", RED, fg=TEXT, cmd=self._on_stop).pack(side="left", padx=(0,6))
        mk_btn(br, "HOME ALL", PANEL, fg=MUTED, cmd=self._on_home).pack(side="left")

        self.lbl_status = mk_label(left, "No plan computed yet.", size=8, color=MUTED, bg=BG)
        self.lbl_status.pack(anchor="w", pady=(6,2))
        self.progress_var = tk.DoubleVar(value=0)
        style = ttk.Style()
        style.configure("C.Horizontal.TProgressbar",
                        troughcolor=INPUT, background=ACCENT, bordercolor=BORDER)
        ttk.Progressbar(left, variable=self.progress_var, maximum=100,
                        style="C.Horizontal.TProgressbar").pack(fill="x")

        # Right: plan table
        tbl = tk.Frame(right, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        tbl.pack(fill="x", pady=(0,8))
        mk_label(tbl, "MOTION PLAN  —  per joint", size=7, bg=CARD).pack(
            anchor="w", padx=12, pady=(8,4))
        hdr = tk.Frame(tbl, bg=PANEL)
        hdr.pack(fill="x", padx=8)
        for txt, w in [("Jt",3),("Name",9),("Current",9),
                       ("Target",9),("Δ°",7),("Speed °/s",10),("",6)]:
            mk_label(hdr, txt, size=7, bold=True, bg=PANEL, width=w).pack(
                side="left", padx=2, pady=3)

        self.tbl_rows = []
        for i in range(6):
            r = tk.Frame(tbl, bg=CARD)
            r.pack(fill="x", padx=8, pady=1)
            cells = {}
            cells["j"]   = mk_label(r, f"J{i+1}", size=8, color=JCOLORS[i], bold=True, bg=CARD, width=3)
            cells["nm"]  = mk_label(r, JOINT_NAMES[i], size=8, color=MUTED, bg=CARD, width=9)
            cells["cur"] = mk_label(r, "---", size=8, color=MUTED, bg=CARD, width=9)
            cells["tgt"] = mk_label(r, "---", size=8, color=TEXT, bg=CARD, width=9)
            cells["d"]   = mk_label(r, "---", size=8, bg=CARD, width=7)
            cells["spd"] = mk_label(r, "---", size=8, color=ACCENT, bg=CARD, width=10)
            cells["ok"]  = mk_label(r, "",   size=7, color=GREEN, bg=CARD, width=6)
            for c in cells.values():
                c.pack(side="left", padx=2)
            self.tbl_rows.append(cells)

        self.lbl_ik = mk_label(tbl, "", size=8, color=MUTED, bg=CARD)
        self.lbl_ik.pack(anchor="w", padx=12, pady=(4,8))

        # Speed bars
        bars = tk.Frame(right, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        bars.pack(fill="x")
        mk_label(bars, "JOINT SPEED LOAD", size=7, bg=CARD).pack(
            anchor="w", padx=12, pady=(8,4))
        self._bar_fills = []
        self._bar_lbls  = []
        for i in range(6):
            br2 = tk.Frame(bars, bg=CARD)
            br2.pack(fill="x", padx=12, pady=1)
            mk_label(br2, f"J{i+1}", size=8, color=JCOLORS[i], bold=True,
                     bg=CARD, width=3).pack(side="left")
            bg_bar = tk.Frame(br2, bg=INPUT, height=12, width=220)
            bg_bar.pack(side="left", padx=4)
            bg_bar.pack_propagate(False)
            fill = tk.Frame(bg_bar, bg=JCOLORS[i], height=12, width=0)
            fill.place(x=0, y=0, relheight=1)
            self._bar_fills.append((bg_bar, fill))
            lbl = mk_label(br2, "---", size=8, color=TEXT, bg=CARD)
            lbl.pack(side="left", padx=4)
            self._bar_lbls.append(lbl)
        mk_label(bars, "", bg=CARD).pack(pady=4)

    def _on_mode_change(self):
        if self.mode_var.get() == "Absolute":
            self.inp_title.config(text="ABSOLUTE TARGET  (mm from base origin)")
        else:
            self.inp_title.config(text="TARGET DISPLACEMENT  (mm)")
        self._clear_plan()

    def _clear_plan(self):
        self._plan = None
        self.btn_exec.config(state="disabled")
        for r in self.tbl_rows:
            for k in ("cur","tgt","d","spd"):
                r[k].config(text="---", fg=MUTED)
            r["ok"].config(text="")
        self.lbl_ik.config(text="")
        for i in range(6):
            _, fill = self._bar_fills[i]
            fill.place(width=0)
            self._bar_lbls[i].config(text="---")

    def _on_compute(self):
        try:
            x, y, z = float(self.x_var.get()), float(self.y_var.get()), float(self.z_var.get())
            t = float(self.t_var.get())
        except ValueError:
            messagebox.showerror("Input Error", "X, Y, Z and T must be numbers.")
            return
        if t <= 0:
            messagebox.showerror("Input Error", "Time T must be > 0.")
            return
        cur = self._get_angles()
        self.btn_compute.config(state="disabled", text="COMPUTING IK...")
        self._log(f"[CARTESIAN] IK → ({x:+.1f},{y:+.1f},{z:+.1f}) mm  T={t}s")
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
        self.btn_compute.config(state="normal", text="COMPUTE PLAN")
        self._plan = plan
        if not plan.success:
            messagebox.showerror("IK Failed", plan.message)
            self.lbl_status.config(text=f"Failed: {plan.message}", fg=RED)
            return
        self.btn_exec.config(state="normal")
        for i, r in enumerate(self.tbl_rows):
            spd     = plan.speeds_dps[i]
            clamped = spd >= JOINT_SPEED_LIMITS_DPS[i] * 0.99
            r["cur"].config(text=f"{plan.start_angles_deg[i]:+.1f}°", fg=MUTED)
            r["tgt"].config(text=f"{plan.target_angles_deg[i]:+.1f}°", fg=TEXT)
            r["d"].config(text=f"{plan.angle_changes_deg[i]:.1f}°", fg=TEXT)
            r["spd"].config(text=f"{spd:.1f}°/s", fg=RED if clamped else ACCENT)
            r["ok"].config(text="⚠CLAMP" if clamped else "✓ OK",
                           fg=RED if clamped else GREEN)
        for i in range(6):
            ratio = min(plan.speeds_dps[i] / JOINT_SPEED_LIMITS_DPS[i], 1.0)
            _, fill = self._bar_fills[i]
            fill.place(width=int(220*ratio))
            fill.config(bg=RED if ratio>0.9 else (YELLOW if ratio>0.7 else JCOLORS[i]))
            self._bar_lbls[i].config(text=f"{plan.speeds_dps[i]:.1f}°/s")
        ik_c = GREEN if plan.ik_error_mm < 1 else (YELLOW if plan.ik_error_mm < 5 else RED)
        self.lbl_ik.config(
            text=f"IK accuracy: {plan.ik_error_mm:.2f}mm  |  "
                 f"Max joint speed: {plan.max_speed_dps:.1f}°/s  |  {plan.time_s:.2f}s",
            fg=ik_c)
        self.lbl_status.config(
            text=f"Plan ready — {plan.time_s:.1f}s  |  IK err: {plan.ik_error_mm:.2f}mm",
            fg=GREEN)
        self._log(f"[CARTESIAN] Plan OK — IK err {plan.ik_error_mm:.2f}mm")

    def _on_execute(self):
        robot = self._get_robot()
        if not robot:
            messagebox.showwarning("Not Connected", "Connect first.")
            return
        if not self._plan or self._executing:
            return
        self._executing = True
        self.btn_exec.config(state="disabled", text="EXECUTING...")
        self.progress_var.set(0)
        threading.Thread(target=self._exec_worker,
                         args=(robot, self._plan), daemon=True).start()

    def _exec_worker(self, robot, plan):
        for i, cfg in enumerate(DEFAULT_JOINT_CONFIG):
            robot.motors[cfg.motor_id].set_position(
                float(plan.target_angles_deg[i]),
                max_speed_dps=float(plan.speeds_dps[i]), wait=False)
            time.sleep(0.001)
        t0 = time.time()
        while True:
            elapsed = time.time() - t0
            pct = min(elapsed / plan.time_s * 100, 100)
            self.after(0, lambda p=pct: self.progress_var.set(p))
            if elapsed >= plan.time_s + 0.5:
                break
            time.sleep(0.05)
        self.after(0, self._exec_done)

    def _exec_done(self):
        self._executing = False
        self.progress_var.set(100)
        self.btn_exec.config(state="normal", text="▶  EXECUTE")
        self.lbl_status.config(text="Motion complete.", fg=GREEN)
        self._log("[CARTESIAN] Motion complete.")

    def _on_stop(self):
        robot = self._get_robot()
        if robot:
            threading.Thread(target=robot.stop_all, daemon=True).start()
        self._executing = False
        self.progress_var.set(0)
        self.btn_exec.config(
            state="normal" if self._plan else "disabled", text="▶  EXECUTE")
        self._log("[CARTESIAN] Stopped.")

    def _on_home(self):
        robot = self._get_robot()
        if robot:
            threading.Thread(target=robot.go_home,
                             kwargs={"speed_dps":60.0,"wait":False},
                             daemon=True).start()
            self._log("[CARTESIAN] Homing.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — JOINT ANGLE CONTROL
# ══════════════════════════════════════════════════════════════════════════════

class JointAngleTab(tk.Frame):
    def __init__(self, parent, get_robot, get_angles, log_fn, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._get_robot  = get_robot
        self._get_angles = get_angles
        self._log        = log_fn
        self.angle_vars  = [tk.StringVar(value="0.0") for _ in range(6)]
        self.speed_vars  = [tk.StringVar(value=str(int(JOINT_SPEED_LIMITS_DPS[i]*0.5)))
                            for i in range(6)]
        self.pos_labels  = []
        self.vel_labels  = []
        self.cur_labels  = []
        self.temp_labels = []
        self._sliders    = []
        self._build()

    def _build(self):
        top = tk.Frame(self, bg=PANEL)
        top.pack(fill="x")
        mk_label(top, "INDIVIDUAL JOINT ANGLE CONTROL", size=10,
                 color=ACCENT, bold=True, bg=PANEL).pack(side="left", padx=16, pady=8)
        mk_label(top, "Set angle + speed per motor — independently or all at once",
                 size=8, color=MUTED, bg=PANEL).pack(side="left", pady=12)

        # Global buttons
        gbl = tk.Frame(self, bg=BG)
        gbl.pack(fill="x", padx=16, pady=(8,4))
        mk_label(gbl, "GLOBAL:", size=8, color=MUTED, bg=BG).pack(side="left", padx=(0,8))
        mk_btn(gbl, "▶  MOVE ALL", ACCENT, fg=BG,
               cmd=self._move_all).pack(side="left", padx=4)
        mk_btn(gbl, "■  STOP ALL", RED, fg=TEXT,
               cmd=self._stop_all).pack(side="left", padx=4)
        mk_btn(gbl, "HOME (0° all)", PANEL, fg=MUTED,
               cmd=self._home_all).pack(side="left", padx=4)
        mk_btn(gbl, "SYNC FROM ROBOT", PANEL, fg=MUTED,
               cmd=self._sync).pack(side="left", padx=4)

        sep(self, pady=2)

        # Column headers
        hdr = tk.Frame(self, bg=PANEL)
        hdr.pack(fill="x", padx=16)
        for txt, w in [("Jt",4),("Name",9),("Pos",10),("Vel",10),
                       ("Current",9),("Temp",8),
                       ("   →  Target °",14),("@ Speed °/s",12),("",18)]:
            mk_label(hdr, txt, size=7, bold=True, bg=PANEL,
                     width=w).pack(side="left", padx=2, pady=4)

        sep(self, pady=1)

        # Joint rows
        for i, cfg in enumerate(DEFAULT_JOINT_CONFIG):
            self._build_row(i, cfg)

        sep(self, pady=6)

        # Presets
        pre = tk.Frame(self, bg=BG)
        pre.pack(fill="x", padx=16, pady=(0,10))
        mk_label(pre, "PRESETS:", size=8, color=MUTED, bg=BG).pack(side="left", padx=(0,8))
        for name, angles in [
                ("All Zero",    [0,0,0,0,0,0]),
                ("Arm Up",      [0,-45,90,0,45,0]),
                ("Arm Forward", [0,30,-60,0,30,0]),
                ("Spread",      [0,45,-90,45,-45,0]),
        ]:
            mk_btn(pre, name, CARD, fg=TEXT,
                   cmd=lambda a=angles: self._apply_preset(a)
                   ).pack(side="left", padx=3)

    def _build_row(self, i, cfg):
        color = JCOLORS[i]
        bg    = BG if i % 2 == 0 else CARD
        row   = tk.Frame(self, bg=bg)
        row.pack(fill="x", padx=16, pady=2)

        # Joint ID
        mk_label(row, f"J{cfg.motor_id}", size=11, color=color,
                 bold=True, bg=bg, width=4).pack(side="left", padx=4)
        # Name
        mk_label(row, JOINT_NAMES[i], size=9, color=MUTED,
                 bg=bg, width=9).pack(side="left", padx=4)

        # Live readouts
        pl = mk_label(row, "---°", size=9, color=TEXT, bg=bg, width=10)
        pl.pack(side="left", padx=3)
        self.pos_labels.append(pl)

        vl = mk_label(row, "---°/s", size=9, color=MUTED, bg=bg, width=10)
        vl.pack(side="left", padx=3)
        self.vel_labels.append(vl)

        cl = mk_label(row, "---A", size=9, color=MUTED, bg=bg, width=9)
        cl.pack(side="left", padx=3)
        self.cur_labels.append(cl)

        tl = mk_label(row, "---°C", size=9, color=MUTED, bg=bg, width=8)
        tl.pack(side="left", padx=3)
        self.temp_labels.append(tl)

        # Divider
        tk.Frame(row, bg=BORDER, width=2).pack(side="left", fill="y", padx=6)

        # Target angle entry
        mk_label(row, "→", size=11, color=color, bg=bg).pack(side="left")
        mk_entry(row, self.angle_vars[i], width=8).pack(side="left", padx=(4,2))
        mk_label(row, "°", size=9, color=MUTED, bg=bg).pack(side="left", padx=(0,6))

        # Speed entry
        mk_label(row, "@", size=9, color=MUTED, bg=bg).pack(side="left")
        mk_entry(row, self.speed_vars[i], width=7).pack(side="left", padx=(4,2))
        mk_label(row, "°/s", size=9, color=MUTED, bg=bg).pack(side="left", padx=(0,8))

        # Move / Stop buttons
        mk_btn(row, "MOVE", color, fg=BG,
               cmd=lambda idx=i, c=cfg: self._move_single(idx, c)
               ).pack(side="left", padx=3)
        mk_btn(row, "STOP", CARD, fg=RED,
               cmd=lambda c=cfg: self._stop_single(c)
               ).pack(side="left", padx=3)

        # Limits hint
        mk_label(row, f"[{cfg.min_deg:.0f}°…{cfg.max_deg:.0f}°]",
                 size=7, color=MUTED, bg=bg).pack(side="left", padx=6)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _parse(self, i):
        try:
            ang = float(self.angle_vars[i].get())
            spd = float(self.speed_vars[i].get())
        except ValueError:
            messagebox.showerror("Input Error",
                                 f"J{i+1}: angle and speed must be numbers.")
            return None
        cfg = DEFAULT_JOINT_CONFIG[i]
        if not cfg.min_deg <= ang <= cfg.max_deg:
            messagebox.showerror("Limit Exceeded",
                                 f"J{i+1} ({JOINT_NAMES[i]}): {ang:.1f}° is outside "
                                 f"[{cfg.min_deg}°, {cfg.max_deg}°].")
            return None
        spd = max(1.0, min(spd, JOINT_SPEED_LIMITS_DPS[i]))
        return ang, spd

    def _move_single(self, i, cfg):
        robot = self._get_robot()
        if not robot:
            messagebox.showwarning("Not Connected", "Connect first.")
            return
        v = self._parse(i)
        if v is None:
            return
        ang, spd = v
        self._log(f"[JOINT] J{cfg.motor_id} ({JOINT_NAMES[i]}) → {ang:+.1f}° @ {spd:.0f}°/s")
        threading.Thread(
            target=robot.motors[cfg.motor_id].set_position,
            kwargs={"position_deg": ang, "max_speed_dps": spd, "wait": False},
            daemon=True).start()

    def _stop_single(self, cfg):
        robot = self._get_robot()
        if robot:
            threading.Thread(target=robot.motors[cfg.motor_id].stop,
                             daemon=True).start()
            self._log(f"[JOINT] J{cfg.motor_id} STOP")

    def _move_all(self):
        robot = self._get_robot()
        if not robot:
            messagebox.showwarning("Not Connected", "Connect first.")
            return
        parsed = []
        for i in range(6):
            v = self._parse(i)
            if v is None:
                return
            parsed.append(v)
        self._log("[JOINT] Moving all joints simultaneously...")
        def worker():
            for i, cfg in enumerate(DEFAULT_JOINT_CONFIG):
                ang, spd = parsed[i]
                robot.motors[cfg.motor_id].set_position(
                    ang, max_speed_dps=spd, wait=False)
                time.sleep(0.001)
        threading.Thread(target=worker, daemon=True).start()

    def _stop_all(self):
        robot = self._get_robot()
        if robot:
            threading.Thread(target=robot.stop_all, daemon=True).start()
            self._log("[JOINT] All motors STOP.")

    def _home_all(self):
        robot = self._get_robot()
        if robot:
            for v in self.angle_vars:
                v.set("0.0")
            threading.Thread(target=robot.go_home,
                             kwargs={"speed_dps":60.0,"wait":False},
                             daemon=True).start()
            self._log("[JOINT] Homing all to 0°.")

    def _sync(self):
        robot = self._get_robot()
        if not robot:
            messagebox.showwarning("Not Connected", "Connect first.")
            return
        for i, cfg in enumerate(DEFAULT_JOINT_CONFIG):
            pos = robot.motors[cfg.motor_id].get_position()
            self.angle_vars[i].set(f"{pos:.1f}")
        self._log("[JOINT] Synced targets from live motor positions.")

    def _apply_preset(self, angles):
        for i, a in enumerate(angles):
            self.angle_vars[i].set(str(float(a)))

    def update_feedback(self, motor_id, pos, vel, cur, temp):
        i = motor_id - 1
        if 0 <= i < 6:
            self.pos_labels[i].config(text=f"{pos:+.1f}°")
            self.vel_labels[i].config(text=f"{vel:+.1f}°/s",
                                      fg=ACCENT if abs(vel)>1 else MUTED)
            self.cur_labels[i].config(text=f"{cur:.2f}A",
                                      fg=YELLOW if cur>15 else MUTED)
            tc = RED if temp>75 else (YELLOW if temp>60 else MUTED)
            self.temp_labels[i].config(text=f"{temp:.1f}°C", fg=tc)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VELOCITY CONTROL
# ══════════════════════════════════════════════════════════════════════════════

class VelocityTab(tk.Frame):
    def __init__(self, parent, get_robot, log_fn, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._get_robot    = get_robot
        self._log          = log_fn
        self.vel_vars      = [tk.StringVar(value="0.0") for _ in range(6)]
        self._sliders      = []
        self._status_lbls  = []
        self._build()

    def _build(self):
        top = tk.Frame(self, bg=PANEL)
        top.pack(fill="x")
        mk_label(top, "VELOCITY CONTROL", size=10, color=ORANGE,
                 bold=True, bg=PANEL).pack(side="left", padx=16, pady=8)
        mk_label(top,
                 "Continuous spin per motor  ·  Positive = CCW  ·  Negative = CW  ·  0 = Stop",
                 size=8, color=MUTED, bg=PANEL).pack(side="left", pady=12)

        gbl = tk.Frame(self, bg=BG)
        gbl.pack(fill="x", padx=16, pady=(8,4))
        mk_label(gbl, "GLOBAL:", size=8, color=MUTED, bg=BG).pack(side="left", padx=(0,8))
        mk_btn(gbl, "▶  START ALL", ORANGE, fg=BG,
               cmd=self._start_all).pack(side="left", padx=4)
        mk_btn(gbl, "■  STOP ALL", RED, fg=TEXT,
               cmd=self._stop_all).pack(side="left", padx=4)
        mk_btn(gbl, "ZERO ALL SLIDERS", PANEL, fg=MUTED,
               cmd=self._zero_all).pack(side="left", padx=4)

        sep(self, pady=2)

        hdr = tk.Frame(self, bg=PANEL)
        hdr.pack(fill="x", padx=16)
        for txt, w in [("Jt",4),("Name",9),
                       ("          Velocity Slider  (drag or type)",35),
                       ("Entry",10),("",16),("Status",12)]:
            mk_label(hdr, txt, size=7, bold=True, bg=PANEL,
                     width=w).pack(side="left", padx=2, pady=4)

        sep(self, pady=1)

        for i, cfg in enumerate(DEFAULT_JOINT_CONFIG):
            self._build_row(i, cfg)

        sep(self, pady=8)

        note = tk.Frame(self, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        note.pack(fill="x", padx=16, pady=(0,10))
        mk_label(note,
                 "ℹ  Velocity mode spins the motor continuously at the given speed.\n"
                 "   The motor does NOT hold a fixed position — it keeps rotating.\n"
                 "   Switch to JOINT ANGLE tab for precise position control.",
                 size=8, color=MUTED, bg=CARD).pack(padx=12, pady=8, anchor="w")

    def _build_row(self, i, cfg):
        color = JCOLORS[i]
        bg    = BG if i % 2 == 0 else CARD
        row   = tk.Frame(self, bg=bg)
        row.pack(fill="x", padx=16, pady=3)

        mk_label(row, f"J{cfg.motor_id}", size=11, color=color,
                 bold=True, bg=bg, width=4).pack(side="left", padx=4)
        mk_label(row, JOINT_NAMES[i], size=9, color=MUTED,
                 bg=bg, width=9).pack(side="left", padx=4)

        limit = JOINT_SPEED_LIMITS_DPS[i]
        sl = tk.Scale(row, from_=-limit, to=limit,
                      orient="horizontal", variable=self.vel_vars[i],
                      resolution=1, showvalue=False,
                      bg=bg, fg=TEXT, troughcolor=INPUT,
                      highlightthickness=0, activebackground=color,
                      bd=0, relief="flat", length=300,
                      command=lambda v, idx=i: self.vel_vars[idx].set(f"{float(v):.1f}"))
        sl.pack(side="left", padx=4)
        self._sliders.append(sl)

        mk_entry(row, self.vel_vars[i], width=8).pack(side="left", padx=(4,2))
        mk_label(row, "°/s", size=9, color=MUTED, bg=bg).pack(side="left", padx=(0,8))

        mk_btn(row, "SEND", color, fg=BG,
               cmd=lambda idx=i, c=cfg: self._send(idx, c)
               ).pack(side="left", padx=3)
        mk_btn(row, "STOP", CARD, fg=RED,
               cmd=lambda idx=i, c=cfg: self._stop(idx, c)
               ).pack(side="left", padx=3)

        st = mk_label(row, "IDLE", size=8, color=MUTED, bg=bg)
        st.pack(side="left", padx=8)
        self._status_lbls.append(st)

        mk_label(row, f"max ±{limit:.0f}°/s", size=7, color=MUTED,
                 bg=bg).pack(side="left", padx=4)

    def _send(self, i, cfg):
        robot = self._get_robot()
        if not robot:
            messagebox.showwarning("Not Connected", "Connect first.")
            return
        try:
            vel = float(self.vel_vars[i].get())
        except ValueError:
            messagebox.showerror("Input Error", f"J{i+1}: velocity must be a number.")
            return
        limit = JOINT_SPEED_LIMITS_DPS[i]
        vel   = max(-limit, min(limit, vel))
        self.vel_vars[i].set(f"{vel:.1f}")
        self._sliders[i].set(vel)
        threading.Thread(target=robot.motors[cfg.motor_id].set_velocity,
                         args=(vel,), daemon=True).start()
        color  = ORANGE if abs(vel) > 0 else MUTED
        status = f"▶ {vel:+.1f}°/s" if abs(vel) > 0 else "IDLE"
        self._status_lbls[i].config(text=status, fg=color)
        self._log(f"[VEL] J{cfg.motor_id} ({JOINT_NAMES[i]}) → {vel:+.1f}°/s")

    def _stop(self, i, cfg):
        robot = self._get_robot()
        if robot:
            threading.Thread(target=robot.motors[cfg.motor_id].set_velocity,
                             args=(0.0,), daemon=True).start()
        self.vel_vars[i].set("0.0")
        self._sliders[i].set(0)
        self._status_lbls[i].config(text="IDLE", fg=MUTED)
        self._log(f"[VEL] J{cfg.motor_id} stopped.")

    def _start_all(self):
        robot = self._get_robot()
        if not robot:
            messagebox.showwarning("Not Connected", "Connect first.")
            return
        for i, cfg in enumerate(DEFAULT_JOINT_CONFIG):
            self._send(i, cfg)

    def _stop_all(self):
        for i, cfg in enumerate(DEFAULT_JOINT_CONFIG):
            self._stop(i, cfg)

    def _zero_all(self):
        for i in range(6):
            self.vel_vars[i].set("0.0")
            self._sliders[i].set(0)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

class ControlApp(tk.Tk):
    def __init__(self, simulated=True, port="COM3", bustype="slcan"):
        super().__init__()
        self.title("RMD-X8-120  ·  Multi-Mode Robot Control Panel")
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
        self._poll_log()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        if simulated:
            self.after(400, self._connect)

    def _build(self):
        # Top bar
        top = tk.Frame(self, bg=PANEL, height=50)
        top.pack(fill="x")
        top.pack_propagate(False)
        mk_label(top, "⬡  RMD-X8-120  MULTI-MODE CONTROL PANEL",
                 size=12, color=ACCENT, bold=True, bg=PANEL).pack(side="left", padx=16, pady=12)
        self.lbl_conn = mk_label(top, "● DISCONNECTED", size=9,
                                 color=RED, bold=True, bg=PANEL)
        self.lbl_conn.pack(side="right", padx=12)
        mode_txt = "SIMULATION" if self._sim else f"REAL · {self._port}"
        mk_label(top, mode_txt, size=8, color=MUTED, bg=PANEL).pack(side="right", padx=4)
        mk_btn(top, "DISCONNECT", PANEL, fg=RED, cmd=self._disconnect).pack(side="right", padx=4)
        mk_btn(top, "CONNECT", ACCENT, fg=BG, cmd=self._connect).pack(side="right", padx=4)

        # FK display
        self.fk_display = FKDisplay(self)
        self.fk_display.pack(fill="x", padx=12, pady=(8,4))

        # Tab switcher bar
        tab_bar = tk.Frame(self, bg=BG)
        tab_bar.pack(fill="x", padx=12, pady=(4,0))

        self._tab_btns   = {}
        self._tab_frames = {}

        container = tk.Frame(self, bg=BG)
        container.pack(fill="both", expand=True, padx=12, pady=(0,4))

        tabs = [
            ("joint",     "⟳  JOINT ANGLE",  ACCENT, "Per-motor angle + speed",
             lambda: JointAngleTab(container,
                                   get_robot=lambda: self.robot,
                                   get_angles=self._get_angles,
                                   log_fn=self._log_msg)),
            ("cartesian", "⊕  CARTESIAN",    BLUE,   "Displacement / Absolute  +  Time",
             lambda: CartesianTab(container,
                                  get_robot=lambda: self.robot,
                                  get_angles=self._get_angles,
                                  log_fn=self._log_msg)),
            ("velocity",  "⚡  VELOCITY",    ORANGE, "Continuous spin per motor",
             lambda: VelocityTab(container,
                                 get_robot=lambda: self.robot,
                                 log_fn=self._log_msg)),
        ]

        tab_colors = {}
        for key, title, color, hint, factory in tabs:
            frame = factory()
            frame.pack(fill="both", expand=True)
            frame.pack_forget()
            self._tab_frames[key] = frame
            tab_colors[key] = color

            tb = tk.Button(tab_bar, text=f"  {title}  ",
                           font=("Courier New", 9, "bold"),
                           bg=PANEL, fg=MUTED, relief="flat",
                           cursor="hand2", pady=6,
                           command=lambda k=key: self._switch_tab(k))
            tb.pack(side="left", padx=2)
            self._tab_btns[key] = tb

            mk_label(tab_bar, hint, size=7, color=MUTED,
                     bg=BG).pack(side="left", padx=(0,14))

        self._tab_colors = tab_colors
        self._switch_tab("joint")

        # Log panel
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        lh = tk.Frame(self, bg=PANEL)
        lh.pack(fill="x")
        mk_label(lh, "SYSTEM LOG", size=7, color=MUTED,
                 bg=PANEL).pack(side="left", padx=12, pady=3)
        tk.Button(lh, text="CLEAR", font=("Courier New",7), bg=PANEL,
                  fg=MUTED, relief="flat", cursor="hand2",
                  command=self._clear_log).pack(side="right", padx=12)
        self.log_box = scrolledtext.ScrolledText(
            self, height=6, font=("Courier New", 8),
            bg=PANEL, fg=MUTED, relief="flat",
            state="disabled", wrap="word")
        self.log_box.pack(fill="x")
        self.log_box.tag_config("ok",   foreground=GREEN)
        self.log_box.tag_config("warn", foreground=YELLOW)
        self.log_box.tag_config("err",  foreground=RED)
        self.log_box.tag_config("info", foreground=MUTED)

    def _switch_tab(self, key):
        for k, f in self._tab_frames.items():
            f.pack_forget()
        for k, b in self._tab_btns.items():
            b.config(bg=self._tab_colors[k] if k==key else PANEL,
                     fg=BG if k==key else MUTED)
        self._tab_frames[key].pack(fill="both", expand=True)

    def _connect(self):
        try:
            self.bus = CANBus(simulated=self._sim, channel=self._port,
                              bustype=self._bus_t, num_motors=6)
            self.robot = RobotController(self.bus, DEFAULT_JOINT_CONFIG)
            self.robot.start()
            self.lbl_conn.config(text="● CONNECTED", fg=ACCENT)
            self._log_msg("Connected. All 6 motors enabled.", "ok")
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
        self.fk_display.reset()
        self._log_msg("Disconnected.")

    def _get_angles(self):
        if self.robot:
            return np.array([
                self.robot.motors[cfg.motor_id].get_position()
                for cfg in DEFAULT_JOINT_CONFIG], dtype=float)
        return np.zeros(6)

    def _start_monitor(self):
        self._monitor_job = self.after(300, self._monitor_tick)

    def _stop_monitor(self):
        if self._monitor_job:
            self.after_cancel(self._monitor_job)
            self._monitor_job = None

    def _monitor_tick(self):
        if self.robot:
            angles = self._get_angles()
            fk = forward_kinematics(angles)
            p  = fk.position_mm()
            self.fk_display.update(p[0], p[1], p[2])
            fb_all = self.robot.get_all_feedback()
            jt = self._tab_frames["joint"]
            for motor_id, fb in fb_all.items():
                jt.update_feedback(motor_id, fb.position_deg, fb.velocity_dps,
                                   fb.current_a, fb.temperature_c)
        self._monitor_job = self.after(300, self._monitor_tick)

    def _log_msg(self, msg, tag="info"):
        ts = time.strftime("%H:%M:%S")
        self.log_box.config(state="normal")
        self.log_box.insert("end", f"{ts}  {msg}\n", tag)
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def _poll_log(self):
        while not log_queue.empty():
            msg = log_queue.get_nowait()
            tag = "warn" if "WARNING" in msg else ("err" if "ERROR" in msg else "info")
            self._log_msg(msg, tag)
        self.after(150, self._poll_log)

    def _clear_log(self):
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--real",    action="store_true")
    parser.add_argument("--port",    default="COM3")
    parser.add_argument("--bustype", default="slcan")
    args = parser.parse_args()

    app = ControlApp(simulated=not args.real,
                     port=args.port, bustype=args.bustype)
    app.mainloop()
