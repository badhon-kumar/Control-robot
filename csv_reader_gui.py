"""
gui.py  (v2 — Manual + CSV Sequence Control)
=============================================
Two control modes selectable at any time:

  TAB 1 — MANUAL CONTROL : 6 joint cards with sliders, speed, move/stop per joint
  TAB 2 — CSV SEQUENCE   : Load a CSV file → preview all steps → execute with
                           configurable inter-step delay, pause/resume/abort

CSV FORMAT (see sample_sequence.csv):
  step, delay_s, j1_angle, j1_speed, j2_angle, j2_speed, ... j6_angle, j6_speed
  - Angles in degrees. Empty cell = skip that motor (it won't move).
  - delay_s = wait time AFTER this step completes before the next step begins.
  - step column is just a label (can be any text).

Run:
    python3.10 gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import time
import logging
import queue
import csv
import os
from dataclasses import dataclass, field
from typing import Optional

from can_interface import CANBus
from robot_controller import RobotController, DEFAULT_JOINT_CONFIG

# ── Logging → queue ───────────────────────────────────────────────────────────
log_queue = queue.Queue()

class QueueHandler(logging.Handler):
    def emit(self, record):
        log_queue.put(self.format(record))

# ── Theme ─────────────────────────────────────────────────────────────────────
BG        = "#0d1117"
PANEL     = "#161b22"
CARD      = "#1c2128"
INPUT     = "#21262d"
BORDER    = "#30363d"
ACCENT    = "#00d4aa"
ACCENT_D  = "#00a87f"
BLUE      = "#4dabf7"
RED       = "#ff4444"
RED_D     = "#cc2222"
YELLOW    = "#e3b341"
GREEN     = "#3fb950"
ORANGE    = "#f0883e"
MUTED     = "#7d8590"
TEXT      = "#e6edf3"
JCOLORS   = ["#00d4aa","#4dabf7","#e3b341","#ff7eb6","#a5d6ff","#c3e88d"]
JNAMES    = ["Base","Shoulder","Elbow","Wrist1","Wrist2","Tool"]


# ══════════════════════════════════════════════════════════════════════════════
# CSV DATA STRUCTURES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SequenceStep:
    """One row from the CSV file."""
    step_label: str                          # e.g. "Step 1" or any text label
    delay_s:    float                        # wait after step completes (seconds)
    # Per-joint: (target_angle_deg, speed_dps) or None if cell was empty
    joints: list = field(default_factory=lambda: [None]*6)

    def motors_to_move(self):
        """Returns list of (motor_id, angle, speed) for non-empty joints."""
        result = []
        for i, j in enumerate(self.joints):
            if j is not None:
                ang, spd = j
                result.append((i+1, ang, spd))
        return result


def parse_csv(filepath: str) -> tuple[list[SequenceStep], str]:
    """
    Parse a sequence CSV file.

    Expected columns:
        step, delay_s,
        j1_angle, j1_speed,
        j2_angle, j2_speed,
        j3_angle, j3_speed,
        j4_angle, j4_speed,
        j5_angle, j5_speed,
        j6_angle, j6_speed

    Returns (list_of_steps, error_message).
    error_message is "" if parsing succeeded.
    """
    steps = []
    errors = []

    try:
        with open(filepath, newline="") as f:
            reader = csv.DictReader(f)

            # Validate header
            required = ["step", "delay_s"]
            for col in required:
                if col not in reader.fieldnames:
                    return [], f"Missing required column: '{col}'"

            for row_num, row in enumerate(reader, start=2):
                label = row.get("step", f"Step {row_num}").strip()

                # Parse delay
                try:
                    delay = float(row.get("delay_s", "0") or "0")
                    delay = max(0.0, delay)
                except ValueError:
                    errors.append(f"Row {row_num}: invalid delay_s '{row.get('delay_s')}'")
                    delay = 0.0

                # Parse per-joint values
                joints = []
                for i in range(1, 7):
                    ang_key = f"j{i}_angle"
                    spd_key = f"j{i}_speed"
                    ang_str = row.get(ang_key, "").strip()
                    spd_str = row.get(spd_key, "").strip()

                    if ang_str == "":
                        joints.append(None)   # skip this motor
                        continue

                    try:
                        ang = float(ang_str)
                    except ValueError:
                        errors.append(f"Row {row_num} J{i}: invalid angle '{ang_str}'")
                        joints.append(None)
                        continue

                    try:
                        spd = float(spd_str) if spd_str else DEFAULT_JOINT_CONFIG[i-1].max_speed_dps
                        spd = max(1.0, spd)
                    except ValueError:
                        errors.append(f"Row {row_num} J{i}: invalid speed '{spd_str}', using default")
                        spd = DEFAULT_JOINT_CONFIG[i-1].max_speed_dps

                    # Clamp angle to joint limits
                    cfg = DEFAULT_JOINT_CONFIG[i-1]
                    if not cfg.min_deg <= ang <= cfg.max_deg:
                        errors.append(
                            f"Row {row_num} J{i}: angle {ang}° outside "
                            f"[{cfg.min_deg}°, {cfg.max_deg}°] — clamped")
                        ang = max(cfg.min_deg, min(cfg.max_deg, ang))

                    joints.append((ang, spd))

                steps.append(SequenceStep(label, delay, joints))

    except FileNotFoundError:
        return [], f"File not found: {filepath}"
    except Exception as e:
        return [], f"CSV parse error: {e}"

    error_str = "\n".join(errors) if errors else ""
    return steps, error_str


# ══════════════════════════════════════════════════════════════════════════════
# JOINT CARD  (used in Manual tab)
# ══════════════════════════════════════════════════════════════════════════════

class JointCard(tk.Frame):
    def __init__(self, parent, cfg, color, **kw):
        super().__init__(parent, bg=CARD,
                         highlightbackground=BORDER, highlightthickness=1, **kw)
        self.cfg   = cfg
        self.color = color
        self.on_move_cb = None
        self.on_stop_cb = None
        self._build()

    def _build(self):
        tk.Frame(self, bg=self.color, height=4).pack(fill="x")

        tr = tk.Frame(self, bg=CARD)
        tr.pack(fill="x", padx=12, pady=(10,4))
        tk.Label(tr, text=f"J{self.cfg.motor_id}", font=("Courier New",18,"bold"),
                 fg=self.color, bg=CARD).pack(side="left")
        tk.Label(tr, text=self.cfg.name.upper(), font=("Courier New",9),
                 fg=MUTED, bg=CARD).pack(side="left", padx=(8,0), pady=(6,0))

        fb = tk.Frame(self, bg=CARD)
        fb.pack(fill="x", padx=12, pady=2)
        self.lbl_pos  = self._stat(fb, "POS",  "0.00°")
        self.lbl_vel  = self._stat(fb, "VEL",  "0.0°/s")
        self.lbl_cur  = self._stat(fb, "CUR",  "0.00A")
        self.lbl_temp = self._stat(fb, "TEMP", "35.0°C")

        sf = tk.Frame(self, bg=CARD)
        sf.pack(fill="x", padx=12, pady=(8,2))
        tk.Label(sf, text="TARGET POSITION", font=("Courier New",7),
                 fg=MUTED, bg=CARD).pack(anchor="w")

        self.slider_var = tk.DoubleVar(value=0.0)
        tk.Scale(sf, from_=self.cfg.min_deg, to=self.cfg.max_deg,
                 orient="horizontal", variable=self.slider_var,
                 resolution=0.5, showvalue=False,
                 bg=CARD, fg=TEXT, troughcolor=INPUT,
                 highlightthickness=0, activebackground=self.color,
                 bd=0, relief="flat").pack(fill="x")

        vr = tk.Frame(sf, bg=CARD)
        vr.pack(fill="x")
        tk.Label(vr, text=f"{self.cfg.min_deg:.0f}°",
                 font=("Courier New",7), fg=MUTED, bg=CARD).pack(side="left")
        self.lbl_val = tk.Label(vr, text="0.0°",
                 font=("Courier New",9,"bold"), fg=self.color, bg=CARD)
        self.lbl_val.pack(side="left", expand=True)
        tk.Label(vr, text=f"{self.cfg.max_deg:.0f}°",
                 font=("Courier New",7), fg=MUTED, bg=CARD).pack(side="right")
        self.slider_var.trace_add("write",
            lambda *_: self.lbl_val.config(text=f"{self.slider_var.get():.1f}°"))

        sr = tk.Frame(self, bg=CARD)
        sr.pack(fill="x", padx=12, pady=2)
        tk.Label(sr, text="SPEED  °/s", font=("Courier New",7),
                 fg=MUTED, bg=CARD).pack(side="left")
        self.speed_var = tk.StringVar(value=str(int(self.cfg.max_speed_dps)))
        tk.Entry(sr, textvariable=self.speed_var, width=6,
                 font=("Courier New",9), bg=INPUT, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 highlightbackground=BORDER, highlightthickness=1
                 ).pack(side="right")

        br = tk.Frame(self, bg=CARD)
        br.pack(fill="x", padx=12, pady=(6,12))
        tk.Button(br, text="MOVE", font=("Courier New",9,"bold"),
                  bg=self.color, fg=BG, activebackground=ACCENT_D,
                  relief="flat", cursor="hand2",
                  command=self._on_move).pack(side="left", fill="x",
                                              expand=True, padx=(0,4))
        tk.Button(br, text="STOP", font=("Courier New",9,"bold"),
                  bg=INPUT, fg=RED, activebackground=RED_D,
                  relief="flat", cursor="hand2",
                  command=self._on_stop).pack(side="right", fill="x",
                                              expand=True, padx=(4,0))

    def _stat(self, parent, tag, val):
        f = tk.Frame(parent, bg=CARD)
        f.pack(side="left", expand=True)
        tk.Label(f, text=tag, font=("Courier New",6), fg=MUTED, bg=CARD).pack()
        lbl = tk.Label(f, text=val, font=("Courier New",9,"bold"),
                       fg=TEXT, bg=CARD)
        lbl.pack()
        return lbl

    def _on_move(self):
        if self.on_move_cb:
            try:   spd = float(self.speed_var.get())
            except: spd = self.cfg.max_speed_dps
            self.on_move_cb(self.cfg.motor_id, self.slider_var.get(), spd)

    def _on_stop(self):
        if self.on_stop_cb:
            self.on_stop_cb(self.cfg.motor_id)

    def update_feedback(self, pos, vel, cur, temp):
        self.lbl_pos.config(text=f"{pos:+.1f}°")
        self.lbl_vel.config(text=f"{vel:+.1f}°/s")
        self.lbl_cur.config(text=f"{cur:.2f}A")
        tc = RED if temp > 75 else (YELLOW if temp > 60 else TEXT)
        self.lbl_temp.config(text=f"{temp:.1f}°C", fg=tc)


# ══════════════════════════════════════════════════════════════════════════════
# MANUAL CONTROL TAB
# ══════════════════════════════════════════════════════════════════════════════

class ManualTab(tk.Frame):
    def __init__(self, parent, get_robot, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._get_robot = get_robot
        self.cards: dict[int, JointCard] = {}
        self._build()

    def _build(self):
        # Global control buttons
        ctrl = tk.Frame(self, bg=BG, pady=8)
        ctrl.pack(fill="x", padx=16)
        tk.Label(ctrl, text="GLOBAL:", font=("Courier New",7),
                 fg=MUTED, bg=BG).pack(side="left", padx=(0,10))
        for txt, color, fg, cmd in [
            ("HOME ALL",     ACCENT,  BG,   self._home),
            ("MOVE ALL",     BLUE,    BG,   self._move_all),
            ("STOP ALL",     YELLOW,  BG,   self._stop_all),
            ("⚠ E-STOP",     RED,     TEXT, self._estop),
            ("RESET E-STOP", INPUT,   TEXT, self._reset),
        ]:
            tk.Button(ctrl, text=txt, font=("Courier New",9,"bold"),
                      bg=color, fg=fg, relief="flat", cursor="hand2",
                      padx=10, pady=4, command=cmd).pack(side="left", padx=4)

        # 6 joint cards in 2×3 grid
        grid = tk.Frame(self, bg=BG)
        grid.pack(fill="both", expand=True, padx=16, pady=(4,8))
        for i, cfg in enumerate(DEFAULT_JOINT_CONFIG):
            r, c = divmod(i, 3)
            card = JointCard(grid, cfg, JCOLORS[i])
            card.grid(row=r, column=c, padx=5, pady=5, sticky="nsew")
            card.on_move_cb = self._move_joint
            card.on_stop_cb = self._stop_joint
            self.cards[cfg.motor_id] = card
            grid.columnconfigure(c, weight=1)
        for r in range(2):
            grid.rowconfigure(r, weight=1)

    def _req(self):
        robot = self._get_robot()
        if not robot:
            messagebox.showwarning("Not Connected", "Connect first.")
        return robot

    def _move_joint(self, mid, pos, spd):
        robot = self._req()
        if robot:
            threading.Thread(target=robot.move_joint,
                             args=(mid, pos, spd, False), daemon=True).start()

    def _stop_joint(self, mid):
        robot = self._req()
        if robot:
            threading.Thread(target=robot.motors[mid].stop, daemon=True).start()

    def _home(self):
        robot = self._req()
        if robot:
            threading.Thread(target=robot.go_home,
                             kwargs={"speed_dps":60,"wait":False},
                             daemon=True).start()

    def _move_all(self):
        robot = self._req()
        if robot:
            pos = {mid: card.slider_var.get() for mid, card in self.cards.items()}
            threading.Thread(target=robot.move_all_joints,
                             kwargs={"positions":pos,"speed_dps":90,"wait":False},
                             daemon=True).start()

    def _stop_all(self):
        robot = self._req()
        if robot:
            threading.Thread(target=robot.stop_all, daemon=True).start()

    def _estop(self):
        robot = self._req()
        if robot:
            robot.estop()

    def _reset(self):
        robot = self._req()
        if robot:
            robot.reset()

    def update_feedback(self, motor_id, pos, vel, cur, temp):
        card = self.cards.get(motor_id)
        if card:
            card.update_feedback(pos, vel, cur, temp)


# ══════════════════════════════════════════════════════════════════════════════
# CSV SEQUENCE TAB
# ══════════════════════════════════════════════════════════════════════════════

class CsvTab(tk.Frame):
    def __init__(self, parent, get_robot, log_fn, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._get_robot  = get_robot
        self._log        = log_fn
        self._steps: list[SequenceStep] = []
        self._filepath   = tk.StringVar(value="")
        self._delay_var  = tk.StringVar(value="1.0")
        self._wait_var   = tk.BooleanVar(value=True)
        self._loop_var   = tk.BooleanVar(value=False)
        self._running    = False
        self._paused     = False
        self._abort_flag = threading.Event()
        self._current_step = -1
        self._build()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build(self):
        # ── Top: file loader ──────────────────────────────────────────────────
        file_card = tk.Frame(self, bg=CARD,
                             highlightbackground=BORDER, highlightthickness=1)
        file_card.pack(fill="x", padx=16, pady=(12,6))

        tk.Label(file_card, text="CSV SEQUENCE FILE",
                 font=("Courier New",7), fg=MUTED, bg=CARD).pack(
                     anchor="w", padx=12, pady=(8,4))

        fr = tk.Frame(file_card, bg=CARD)
        fr.pack(fill="x", padx=12, pady=(0,8))

        self.path_entry = tk.Entry(fr, textvariable=self._filepath,
                                   font=("Courier New",9), bg=INPUT, fg=TEXT,
                                   insertbackground=TEXT, relief="flat",
                                   highlightbackground=BORDER, highlightthickness=1,
                                   width=55)
        self.path_entry.pack(side="left", padx=(0,6))

        tk.Button(fr, text="BROWSE…", font=("Courier New",9,"bold"),
                  bg=BLUE, fg=BG, relief="flat", cursor="hand2",
                  command=self._browse).pack(side="left", padx=(0,6))

        tk.Button(fr, text="LOAD & PREVIEW", font=("Courier New",9,"bold"),
                  bg=ACCENT, fg=BG, relief="flat", cursor="hand2",
                  command=self._load).pack(side="left", padx=(0,6))

        tk.Button(fr, text="SAMPLE CSV", font=("Courier New",9),
                  bg=INPUT, fg=MUTED, relief="flat", cursor="hand2",
                  command=self._create_sample).pack(side="left")

        # ── Options row ───────────────────────────────────────────────────────
        opt = tk.Frame(self, bg=BG)
        opt.pack(fill="x", padx=16, pady=4)

        tk.Label(opt, text="INTER-STEP DELAY  (s):",
                 font=("Courier New",8), fg=MUTED, bg=BG).pack(side="left")
        tk.Entry(opt, textvariable=self._delay_var, width=6,
                 font=("Courier New",10), bg=INPUT, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 highlightbackground=BORDER, highlightthickness=1
                 ).pack(side="left", padx=(4,20))

        tk.Label(opt, text="(overrides per-step delay from CSV — set 0 to use CSV values)",
                 font=("Courier New",7), fg=MUTED, bg=BG).pack(side="left", padx=(0,20))

        tk.Checkbutton(opt, text="Wait for motion to finish before next step",
                       variable=self._wait_var,
                       font=("Courier New",8), bg=BG, fg=TEXT,
                       selectcolor=INPUT, activebackground=BG,
                       activeforeground=TEXT).pack(side="left", padx=(0,12))

        tk.Checkbutton(opt, text="Loop sequence",
                       variable=self._loop_var,
                       font=("Courier New",8), bg=BG, fg=TEXT,
                       selectcolor=INPUT, activebackground=BG,
                       activeforeground=TEXT).pack(side="left")

        # ── Execution controls ────────────────────────────────────────────────
        ec = tk.Frame(self, bg=BG)
        ec.pack(fill="x", padx=16, pady=(4,6))

        self.btn_run = tk.Button(ec, text="▶  RUN SEQUENCE",
                                 font=("Courier New",10,"bold"),
                                 bg=GREEN, fg=BG, relief="flat",
                                 cursor="hand2", padx=14, pady=6,
                                 command=self._run, state="disabled")
        self.btn_run.pack(side="left", padx=(0,6))

        self.btn_pause = tk.Button(ec, text="⏸  PAUSE",
                                   font=("Courier New",10,"bold"),
                                   bg=YELLOW, fg=BG, relief="flat",
                                   cursor="hand2", padx=14, pady=6,
                                   command=self._pause, state="disabled")
        self.btn_pause.pack(side="left", padx=(0,6))

        self.btn_abort = tk.Button(ec, text="■  ABORT",
                                   font=("Courier New",10,"bold"),
                                   bg=RED, fg=TEXT, relief="flat",
                                   cursor="hand2", padx=14, pady=6,
                                   command=self._abort, state="disabled")
        self.btn_abort.pack(side="left", padx=(0,20))

        self.lbl_exec = tk.Label(ec, text="No sequence loaded.",
                                 font=("Courier New",9), fg=MUTED, bg=BG)
        self.lbl_exec.pack(side="left")

        # ── Progress bar ──────────────────────────────────────────────────────
        self.progress_var = tk.DoubleVar(value=0)
        style = ttk.Style()
        style.configure("Seq.Horizontal.TProgressbar",
                        troughcolor=INPUT, background=GREEN,
                        bordercolor=BORDER)
        ttk.Progressbar(self, variable=self.progress_var, maximum=100,
                        style="Seq.Horizontal.TProgressbar",
                        length=400).pack(fill="x", padx=16, pady=(0,6))

        # ── Step preview table ────────────────────────────────────────────────
        tbl_frame = tk.Frame(self, bg=BG)
        tbl_frame.pack(fill="both", expand=True, padx=16, pady=(0,8))

        tk.Label(tbl_frame, text="SEQUENCE PREVIEW",
                 font=("Courier New",7), fg=MUTED, bg=BG).pack(anchor="w", pady=(0,4))

        # Scrollable canvas for the table
        canvas = tk.Canvas(tbl_frame, bg=CARD, highlightthickness=0)
        scrollbar = ttk.Scrollbar(tbl_frame, orient="vertical",
                                  command=canvas.yview)
        self.tbl_inner = tk.Frame(canvas, bg=CARD)
        self.tbl_inner.bind("<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0,0), window=self.tbl_inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Table header
        self._build_table_header()

        # Row highlight tracker
        self._row_frames: list[tk.Frame] = []

    def _build_table_header(self):
        hdr = tk.Frame(self.tbl_inner, bg=PANEL)
        hdr.pack(fill="x")
        cols = [("#", 3), ("Label", 14), ("Delay", 6)]
        for i in range(6):
            cols.append((f"J{i+1} {JNAMES[i][:4]}", 13))
        for txt, w in cols:
            tk.Label(hdr, text=txt, font=("Courier New",7,"bold"),
                     fg=MUTED, bg=PANEL, width=w, anchor="w"
                     ).pack(side="left", padx=2, pady=3)

    def _populate_table(self):
        # Clear existing rows
        for w in self.tbl_inner.winfo_children():
            if isinstance(w, tk.Frame) and w.cget("bg") != PANEL:
                w.destroy()
        self._row_frames.clear()

        for idx, step in enumerate(self._steps):
            bg = CARD if idx % 2 == 0 else INPUT
            row = tk.Frame(self.tbl_inner, bg=bg)
            row.pack(fill="x")
            self._row_frames.append(row)

            # Step number
            tk.Label(row, text=str(idx+1), font=("Courier New",8),
                     fg=MUTED, bg=bg, width=3, anchor="w").pack(side="left", padx=2, pady=2)
            # Label
            tk.Label(row, text=step.step_label[:14], font=("Courier New",8),
                     fg=TEXT, bg=bg, width=14, anchor="w").pack(side="left", padx=2)
            # Delay
            tk.Label(row, text=f"{step.delay_s:.1f}s", font=("Courier New",8),
                     fg=YELLOW, bg=bg, width=6, anchor="w").pack(side="left", padx=2)

            # Per-joint cells
            for i in range(6):
                j = step.joints[i]
                if j is None:
                    txt  = "—"
                    fgc  = MUTED
                else:
                    ang, spd = j
                    txt  = f"{ang:+.0f}° @ {spd:.0f}"
                    fgc  = JCOLORS[i]
                tk.Label(row, text=txt, font=("Courier New",8),
                         fg=fgc, bg=bg, width=13, anchor="w").pack(side="left", padx=2)

    def _highlight_row(self, idx: int):
        """Highlight the currently executing row in green."""
        for i, frame in enumerate(self._row_frames):
            bg = GREEN if i == idx else (CARD if i%2==0 else INPUT)
            frame.config(bg=bg)
            for w in frame.winfo_children():
                try:
                    w.config(bg=bg)
                except:
                    pass

    def _clear_highlight(self):
        for i, frame in enumerate(self._row_frames):
            bg = CARD if i%2==0 else INPUT
            frame.config(bg=bg)
            for w in frame.winfo_children():
                try:
                    w.config(bg=bg)
                except:
                    pass

    # ── File loading ──────────────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select Sequence CSV",
            filetypes=[("CSV files","*.csv"), ("All files","*.*")])
        if path:
            self._filepath.set(path)

    def _load(self):
        path = self._filepath.get().strip()
        if not path:
            messagebox.showwarning("No File", "Choose a CSV file first.")
            return

        steps, err = parse_csv(path)

        if not steps:
            messagebox.showerror("Load Failed", err or "No steps found in file.")
            self._log(f"[CSV] Load failed: {err}", "err")
            return

        self._steps = steps
        self._populate_table()
        self.btn_run.config(state="normal")

        warn_msg = ""
        if err:
            warn_msg = f"\nWarnings:\n{err}"
            messagebox.showwarning("Loaded with warnings",
                                   f"Loaded {len(steps)} steps.{warn_msg}")
        else:
            self.lbl_exec.config(
                text=f"Loaded {len(steps)} steps from {os.path.basename(path)}",
                fg=GREEN)

        self._log(f"[CSV] Loaded {len(steps)} steps from {os.path.basename(path)}")
        if err:
            self._log(f"[CSV] Warnings: {err}", "warn")

    def _create_sample(self):
        """Write a sample CSV to the current directory and load it."""
        sample_path = "sample_sequence.csv"
        lines = [
            "step,delay_s,j1_angle,j1_speed,j2_angle,j2_speed,j3_angle,j3_speed,"
            "j4_angle,j4_speed,j5_angle,j5_speed,j6_angle,j6_speed",
            "Home,1.0,0,60,0,60,0,60,0,60,0,60,0,60",
            "Reach Forward,2.0,,, 30,90,-60,90,,,,,, ",
            "Rotate Base,2.0,45,120,,,,,,,,,,",
            "Wrist Move,1.5,,,,,,,,-20,180, 15,180,,",
            "Tool Spin,2.0,,,,,,,,,,,, 90,200",
            "Return Home,2.0,0,60,0,60,0,60,0,60,0,60,0,60",
        ]
        with open(sample_path, "w") as f:
            f.write("\n".join(lines))
        self._filepath.set(os.path.abspath(sample_path))
        self._log(f"[CSV] Sample file created: {sample_path}")
        self._load()

    # ── Sequence execution ────────────────────────────────────────────────────

    def _run(self):
        robot = self._get_robot()
        if not robot:
            messagebox.showwarning("Not Connected", "Connect first.")
            return
        if not self._steps:
            messagebox.showwarning("No Sequence", "Load a CSV file first.")
            return
        if self._running:
            return

        self._running    = True
        self._paused     = False
        self._abort_flag.clear()
        self.btn_run.config(state="disabled")
        self.btn_pause.config(state="normal", text="⏸  PAUSE")
        self.btn_abort.config(state="normal")
        self.progress_var.set(0)

        threading.Thread(target=self._sequence_worker,
                         args=(robot,), daemon=True).start()

    def _sequence_worker(self, robot):
        """Background thread: executes the full sequence."""
        total = len(self._steps)
        loop  = self._loop_var.get()

        run_count = 0
        while True:
            run_count += 1
            self.after(0, lambda: self._log(
                f"[CSV] ── Starting sequence (run #{run_count}) ──"))

            for idx, step in enumerate(self._steps):
                # Abort check
                if self._abort_flag.is_set():
                    self.after(0, self._on_sequence_done, "Aborted.")
                    return

                # Pause check — block here until unpaused
                while self._paused:
                    if self._abort_flag.is_set():
                        self.after(0, self._on_sequence_done, "Aborted.")
                        return
                    time.sleep(0.1)

                # Highlight row and update status
                self.after(0, self._highlight_row, idx)
                pct = (idx / total) * 100
                self.after(0, lambda p=pct: self.progress_var.set(p))
                self.after(0, self.lbl_exec.config,
                           {"text": f"Step {idx+1}/{total}: {step.step_label}",
                            "fg": GREEN})

                # Log the step
                motors = step.motors_to_move()
                if motors:
                    parts = ", ".join(f"J{m}→{a:+.0f}°@{s:.0f}°/s"
                                      for m, a, s in motors)
                    self.after(0, lambda msg=f"[CSV] Step {idx+1} — {step.step_label}: {parts}":
                               self._log(msg))
                else:
                    self.after(0, lambda msg=f"[CSV] Step {idx+1} — {step.step_label}: (delay only)":
                               self._log(msg))

                # Send motion commands (non-blocking)
                for motor_id, angle, speed in motors:
                    robot.motors[motor_id].set_position(
                        position_deg=angle,
                        max_speed_dps=speed,
                        wait=False)
                    time.sleep(0.001)

                # Wait for motion to finish if requested
                if self._wait_var.get() and motors:
                    # Poll until all commanded motors reach their targets
                    deadline = time.time() + 20.0  # 20s max per step
                    tol = 2.0  # degrees
                    while time.time() < deadline:
                        if self._abort_flag.is_set():
                            break
                        while self._paused and not self._abort_flag.is_set():
                            time.sleep(0.1)
                        arrived = all(
                            abs(robot.motors[mid].get_position() - ang) < tol
                            for mid, ang, _ in motors
                        )
                        if arrived:
                            break
                        time.sleep(0.1)

                # Inter-step delay
                try:
                    gui_delay = float(self._delay_var.get())
                except ValueError:
                    gui_delay = 0.0

                delay = gui_delay if gui_delay > 0 else step.delay_s
                if delay > 0:
                    t0 = time.time()
                    while time.time() - t0 < delay:
                        if self._abort_flag.is_set():
                            self.after(0, self._on_sequence_done, "Aborted.")
                            return
                        while self._paused and not self._abort_flag.is_set():
                            time.sleep(0.05)
                        time.sleep(0.05)

            # Sequence complete — loop or finish
            self.after(0, self.progress_var.set, 100)
            self.after(0, self._clear_highlight)

            if not loop or self._abort_flag.is_set():
                break

            self.after(0, lambda: self._log("[CSV] Looping sequence..."))
            time.sleep(0.2)

        self.after(0, self._on_sequence_done,
                   f"Sequence complete ({run_count} run{'s' if run_count>1 else ''}).")

    def _on_sequence_done(self, msg):
        self._running = False
        self._paused  = False
        self.btn_run.config(state="normal")
        self.btn_pause.config(state="disabled", text="⏸  PAUSE")
        self.btn_abort.config(state="disabled")
        self.lbl_exec.config(text=msg, fg=GREEN if "Abort" not in msg else YELLOW)
        self._log(f"[CSV] {msg}")
        self._clear_highlight()

    def _pause(self):
        if not self._running:
            return
        self._paused = not self._paused
        if self._paused:
            self.btn_pause.config(text="▶  RESUME", bg=GREEN)
            self.lbl_exec.config(text="Paused — click RESUME to continue.", fg=YELLOW)
            self._log("[CSV] Paused.")
        else:
            self.btn_pause.config(text="⏸  PAUSE", bg=YELLOW)
            self._log("[CSV] Resumed.")

    def _abort(self):
        self._abort_flag.set()
        self._paused = False
        robot = self._get_robot()
        if robot:
            robot.stop_all()
        self._log("[CSV] Abort requested — stopping motors.")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class RobotGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("RMD-X8-120  ·  6-Axis Control Panel")
        self.configure(bg=BG)
        self.resizable(True, True)
        self.minsize(1060, 760)

        self.robot: Optional[RobotController] = None
        self.bus:   Optional[CANBus]          = None
        self._connected   = False
        self._monitor_job = None

        handler = QueueHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

        self._build_ui()
        self._poll_log()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top bar
        top = tk.Frame(self, bg=PANEL, height=56)
        top.pack(fill="x")
        top.pack_propagate(False)
        tk.Label(top, text="⬡  RMD-X8-120",
                 font=("Courier New",16,"bold"), fg=ACCENT, bg=PANEL
                 ).pack(side="left", padx=20, pady=10)
        tk.Label(top, text="6-AXIS MOTOR CONTROL PANEL",
                 font=("Courier New",8), fg=MUTED, bg=PANEL
                 ).pack(side="left", pady=18)
        self.lbl_status = tk.Label(top, text="● DISCONNECTED",
                                   font=("Courier New",9,"bold"),
                                   fg=RED, bg=PANEL)
        self.lbl_status.pack(side="right", padx=20)

        # Connection panel
        conn = tk.Frame(self, bg=PANEL, pady=8)
        conn.pack(fill="x")
        inner = tk.Frame(conn, bg=PANEL)
        inner.pack(padx=20)

        tk.Label(inner, text="MODE", font=("Courier New",7),
                 fg=MUTED, bg=PANEL).grid(row=0, column=0, sticky="w")
        self.mode_var = tk.StringVar(value="Simulation")
        mode_cb = ttk.Combobox(inner, textvariable=self.mode_var,
                               values=["Simulation","Real Hardware"],
                               state="readonly", width=14,
                               font=("Courier New",9))
        mode_cb.grid(row=1, column=0, padx=(0,12))
        mode_cb.bind("<<ComboboxSelected>>", self._on_mode_change)

        tk.Label(inner, text="PORT", font=("Courier New",7),
                 fg=MUTED, bg=PANEL).grid(row=0, column=1, sticky="w")
        self.port_var = tk.StringVar(value="COM3")
        self.port_entry = tk.Entry(inner, textvariable=self.port_var, width=14,
                                   font=("Courier New",9), bg=INPUT, fg=TEXT,
                                   insertbackground=TEXT, relief="flat",
                                   highlightbackground=BORDER, highlightthickness=1,
                                   state="disabled")
        self.port_entry.grid(row=1, column=1, padx=(0,12))

        tk.Label(inner, text="BUS TYPE", font=("Courier New",7),
                 fg=MUTED, bg=PANEL).grid(row=0, column=2, sticky="w")
        self.bustype_var = tk.StringVar(value="slcan")
        self.bustype_cb = ttk.Combobox(inner, textvariable=self.bustype_var,
                                       values=["slcan","pcan","kvaser","socketcan"],
                                       state="disabled", width=10,
                                       font=("Courier New",9))
        self.bustype_cb.grid(row=1, column=2, padx=(0,12))

        tk.Label(inner, text="BITRATE", font=("Courier New",7),
                 fg=MUTED, bg=PANEL).grid(row=0, column=3, sticky="w")
        self.bitrate_var = tk.StringVar(value="1000000")
        self.bitrate_cb = ttk.Combobox(inner, textvariable=self.bitrate_var,
                                       values=["1000000","500000","250000"],
                                       state="disabled", width=10,
                                       font=("Courier New",9))
        self.bitrate_cb.grid(row=1, column=3, padx=(0,20))

        self.btn_connect = tk.Button(inner, text="CONNECT", width=12,
                                     font=("Courier New",10,"bold"),
                                     bg=ACCENT, fg=BG, relief="flat",
                                     cursor="hand2", command=self._on_connect)
        self.btn_connect.grid(row=1, column=4, padx=(0,8))

        self.btn_disconnect = tk.Button(inner, text="DISCONNECT", width=12,
                                        font=("Courier New",10,"bold"),
                                        bg=INPUT, fg=MUTED, relief="flat",
                                        cursor="hand2", state="disabled",
                                        command=self._on_disconnect)
        self.btn_disconnect.grid(row=1, column=5)

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # ── Mode selector tabs ────────────────────────────────────────────────
        tab_bar = tk.Frame(self, bg=BG, pady=6)
        tab_bar.pack(fill="x", padx=16)

        self._tab_container = tk.Frame(self, bg=BG)
        self._tab_container.pack(fill="both", expand=True)

        self._manual_tab = ManualTab(self._tab_container,
                                     get_robot=lambda: self.robot)
        self._csv_tab    = CsvTab(self._tab_container,
                                  get_robot=lambda: self.robot,
                                  log_fn=self._log_csv)

        self._tab_btns = {}
        for key, label, color, frame in [
            ("manual", "🕹  MANUAL CONTROL",  ACCENT,  self._manual_tab),
            ("csv",    "📋  CSV SEQUENCE",    ORANGE,  self._csv_tab),
        ]:
            b = tk.Button(tab_bar, text=f"  {label}  ",
                          font=("Courier New",10,"bold"),
                          bg=PANEL, fg=MUTED, relief="flat",
                          cursor="hand2", pady=6,
                          command=lambda k=key: self._switch_tab(k))
            b.pack(side="left", padx=4)
            self._tab_btns[key] = b

        # Hint label
        self.lbl_tab_hint = tk.Label(tab_bar, text="",
                                     font=("Courier New",8), fg=MUTED, bg=BG)
        self.lbl_tab_hint.pack(side="left", padx=12)

        self._switch_tab("manual")

        # ── Log panel ─────────────────────────────────────────────────────────
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        lh = tk.Frame(self, bg=PANEL)
        lh.pack(fill="x")
        tk.Label(lh, text="SYSTEM LOG", font=("Courier New",7),
                 fg=MUTED, bg=PANEL, pady=4).pack(side="left", padx=12)
        tk.Button(lh, text="CLEAR", font=("Courier New",7),
                  bg=PANEL, fg=MUTED, relief="flat", cursor="hand2",
                  command=self._clear_log).pack(side="right", padx=12)

        self.log_box = scrolledtext.ScrolledText(
            self, height=6, font=("Courier New",8),
            bg=PANEL, fg=MUTED, insertbackground=TEXT,
            relief="flat", state="disabled", wrap="word",
            selectbackground=BORDER)
        self.log_box.pack(fill="x")
        self.log_box.tag_config("INFO",    foreground=MUTED)
        self.log_box.tag_config("WARNING", foreground=YELLOW)
        self.log_box.tag_config("ERROR",   foreground=RED)
        self.log_box.tag_config("ok",      foreground=GREEN)
        self.log_box.tag_config("warn",    foreground=YELLOW)
        self.log_box.tag_config("err",     foreground=RED)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox",
                        fieldbackground=INPUT, background=INPUT,
                        foreground=TEXT, selectbackground=INPUT,
                        selectforeground=TEXT, bordercolor=BORDER,
                        arrowcolor=MUTED)

    # ── Tab switching ─────────────────────────────────────────────────────────

    def _switch_tab(self, key):
        self._manual_tab.pack_forget()
        self._csv_tab.pack_forget()

        tab_colors = {"manual": ACCENT, "csv": ORANGE}
        hints = {
            "manual": "Use sliders and buttons to move motors individually or all at once.",
            "csv":    "Load a CSV file with a sequence of motor commands and run them automatically.",
        }
        for k, b in self._tab_btns.items():
            b.config(bg=tab_colors[k] if k==key else PANEL,
                     fg=BG if k==key else MUTED)

        if key == "manual":
            self._manual_tab.pack(fill="both", expand=True)
        else:
            self._csv_tab.pack(fill="both", expand=True)

        self.lbl_tab_hint.config(text=hints.get(key,""))

    # ── Connection ────────────────────────────────────────────────────────────

    def _on_mode_change(self, *_):
        real = self.mode_var.get() == "Real Hardware"
        s = "normal" if real else "disabled"
        self.port_entry.config(state=s)
        self.bustype_cb.config(state="readonly" if real else "disabled")
        self.bitrate_cb.config(state="readonly" if real else "disabled")

    def _on_connect(self):
        if self._connected:
            return
        self._log_msg("Connecting...")
        threading.Thread(target=self._connect_thread, daemon=True).start()

    def _connect_thread(self):
        try:
            sim = self.mode_var.get() == "Simulation"
            self.bus = CANBus(simulated=sim, channel=self.port_var.get(),
                              bustype=self.bustype_var.get(),
                              bitrate=int(self.bitrate_var.get()),
                              num_motors=6)
            self.robot = RobotController(self.bus, DEFAULT_JOINT_CONFIG)
            self.robot.start()
            self._connected = True
            self.after(0, self._on_connected)
        except Exception as e:
            self.after(0, lambda: self._on_failed(str(e)))

    def _on_connected(self):
        self.lbl_status.config(text="● CONNECTED", fg=ACCENT)
        self.btn_connect.config(state="disabled", bg=INPUT, fg=MUTED)
        self.btn_disconnect.config(state="normal", bg=RED_D, fg=TEXT)
        self._log_msg("✓ Connected. All 6 motors enabled.", "ok")
        self._start_monitor()

    def _on_failed(self, err):
        self._log_msg(f"✗ {err}", "err")
        messagebox.showerror("Connection Failed", err)

    def _on_disconnect(self):
        self._stop_monitor()
        if self.robot:
            self.robot.close()
            self.robot = None
        self.bus = None
        self._connected = False
        self.lbl_status.config(text="● DISCONNECTED", fg=RED)
        self.btn_connect.config(state="normal", bg=ACCENT, fg=BG)
        self.btn_disconnect.config(state="disabled", bg=INPUT, fg=MUTED)
        self._log_msg("Disconnected.")

    # ── Live feedback monitor ─────────────────────────────────────────────────

    def _start_monitor(self):
        self._monitor_job = self.after(200, self._monitor_tick)

    def _stop_monitor(self):
        if self._monitor_job:
            self.after_cancel(self._monitor_job)
            self._monitor_job = None

    def _monitor_tick(self):
        if self.robot and self._connected:
            fb_all = self.robot.get_all_feedback()
            for motor_id, fb in fb_all.items():
                self._manual_tab.update_feedback(
                    motor_id, fb.position_deg, fb.velocity_dps,
                    fb.current_a, fb.temperature_c)
            self._monitor_job = self.after(200, self._monitor_tick)

    # ── Log ───────────────────────────────────────────────────────────────────

    def _log_msg(self, msg: str, tag: str = "INFO"):
        self.log_box.config(state="normal")
        ts = time.strftime("%H:%M:%S")
        self.log_box.insert("end", f"{ts} {msg}\n", tag)
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def _log_csv(self, msg: str, tag: str = "INFO"):
        self._log_msg(msg, tag)

    def _poll_log(self):
        while not log_queue.empty():
            msg = log_queue.get_nowait()
            level = "INFO"
            for l in ("WARNING","ERROR","CRITICAL"):
                if f"[{l}]" in msg:
                    level = l
                    break
            self._log_msg(msg, level)
        self.after(100, self._poll_log)

    def _clear_log(self):
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")

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
    app = RobotGUI()
    app.mainloop()