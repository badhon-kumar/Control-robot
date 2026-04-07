"""
gui.py
======
Graphical User Interface for MyActuator RMD-X8-120 Motor Control System
6-Axis Robotic Arm Control Panel

Run with:
    python3.10 gui.py

No extra installation needed — uses tkinter which is built into Python.
Connects to your existing can_interface.py, motor_driver.py, robot_controller.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
import logging
import queue
from typing import Optional

# ── Import your existing modules ──────────────────────────────────────────────
from can_interface import CANBus
from robot_controller import RobotController, DEFAULT_JOINT_CONFIG

# ── Redirect logging to the GUI log panel ────────────────────────────────────
log_queue = queue.Queue()

class QueueHandler(logging.Handler):
    def emit(self, record):
        log_queue.put(self.format(record))

# ══════════════════════════════════════════════════════════════════════════════
# THEME & STYLE CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

BG_DARK       = "#0d1117"
BG_PANEL      = "#161b22"
BG_CARD       = "#1c2128"
BG_INPUT      = "#21262d"
ACCENT        = "#00d4aa"        # teal-green
ACCENT_DIM    = "#00a87f"
RED           = "#ff4444"
RED_DIM       = "#cc2222"
YELLOW        = "#e3b341"
TEXT_PRIMARY  = "#e6edf3"
TEXT_MUTED    = "#7d8590"
TEXT_ACCENT   = "#00d4aa"
BORDER        = "#30363d"
JOINT_COLORS  = ["#00d4aa","#4dabf7","#e3b341","#ff7eb6","#a5d6ff","#c3e88d"]


# ══════════════════════════════════════════════════════════════════════════════
# JOINT CARD WIDGET  — one per motor
# ══════════════════════════════════════════════════════════════════════════════

class JointCard(tk.Frame):
    def __init__(self, parent, cfg, color, **kwargs):
        super().__init__(parent, bg=BG_CARD, highlightbackground=BORDER,
                         highlightthickness=1, **kwargs)
        self.cfg   = cfg
        self.color = color
        self._build()

    def _build(self):
        # ── Header bar ───────────────────────────────────────────────────────
        header = tk.Frame(self, bg=self.color, height=4)
        header.pack(fill="x")

        title_row = tk.Frame(self, bg=BG_CARD)
        title_row.pack(fill="x", padx=12, pady=(10, 4))

        tk.Label(title_row, text=f"J{self.cfg.motor_id}", font=("Courier New", 18, "bold"),
                 fg=self.color, bg=BG_CARD).pack(side="left")
        tk.Label(title_row, text=self.cfg.name.upper(), font=("Courier New", 9),
                 fg=TEXT_MUTED, bg=BG_CARD).pack(side="left", padx=(8, 0), pady=(6, 0))

        # ── Live feedback display ─────────────────────────────────────────────
        fb_frame = tk.Frame(self, bg=BG_CARD)
        fb_frame.pack(fill="x", padx=12, pady=2)

        self.lbl_pos  = self._stat_label(fb_frame, "POS",  "0.00°")
        self.lbl_vel  = self._stat_label(fb_frame, "VEL",  "0.0°/s")
        self.lbl_cur  = self._stat_label(fb_frame, "CUR",  "0.00A")
        self.lbl_temp = self._stat_label(fb_frame, "TEMP", "35.0°C")

        # ── Position slider ───────────────────────────────────────────────────
        slider_frame = tk.Frame(self, bg=BG_CARD)
        slider_frame.pack(fill="x", padx=12, pady=(8, 2))

        tk.Label(slider_frame, text="TARGET POSITION", font=("Courier New", 7),
                 fg=TEXT_MUTED, bg=BG_CARD).pack(anchor="w")

        self.slider_var = tk.DoubleVar(value=0.0)
        self.slider = tk.Scale(
            slider_frame,
            from_=self.cfg.min_deg, to=self.cfg.max_deg,
            orient="horizontal", variable=self.slider_var,
            resolution=0.5, showvalue=False,
            bg=BG_CARD, fg=TEXT_PRIMARY, troughcolor=BG_INPUT,
            highlightthickness=0, activebackground=self.color,
            bd=0, relief="flat",
        )
        self.slider.pack(fill="x")

        val_row = tk.Frame(slider_frame, bg=BG_CARD)
        val_row.pack(fill="x")
        tk.Label(val_row, text=f"{self.cfg.min_deg:.0f}°",
                 font=("Courier New", 7), fg=TEXT_MUTED, bg=BG_CARD).pack(side="left")
        self.lbl_slider_val = tk.Label(val_row, text="0.0°",
                 font=("Courier New", 9, "bold"), fg=self.color, bg=BG_CARD)
        self.lbl_slider_val.pack(side="left", expand=True)
        tk.Label(val_row, text=f"{self.cfg.max_deg:.0f}°",
                 font=("Courier New", 7), fg=TEXT_MUTED, bg=BG_CARD).pack(side="right")

        self.slider_var.trace_add("write", self._on_slider)

        # ── Speed entry ───────────────────────────────────────────────────────
        speed_row = tk.Frame(self, bg=BG_CARD)
        speed_row.pack(fill="x", padx=12, pady=2)
        tk.Label(speed_row, text="SPEED  °/s", font=("Courier New", 7),
                 fg=TEXT_MUTED, bg=BG_CARD).pack(side="left")
        self.speed_var = tk.StringVar(value=str(int(self.cfg.max_speed_dps)))
        speed_entry = tk.Entry(speed_row, textvariable=self.speed_var, width=6,
                               font=("Courier New", 9), bg=BG_INPUT, fg=TEXT_PRIMARY,
                               insertbackground=TEXT_PRIMARY, relief="flat",
                               highlightbackground=BORDER, highlightthickness=1)
        speed_entry.pack(side="right")

        # ── Move / Stop buttons ───────────────────────────────────────────────
        btn_row = tk.Frame(self, bg=BG_CARD)
        btn_row.pack(fill="x", padx=12, pady=(6, 12))

        self.btn_move = tk.Button(
            btn_row, text="MOVE", font=("Courier New", 9, "bold"),
            bg=self.color, fg=BG_DARK, activebackground=ACCENT_DIM,
            activeforeground=BG_DARK, relief="flat", cursor="hand2",
            command=self._on_move
        )
        self.btn_move.pack(side="left", fill="x", expand=True, padx=(0, 4))

        self.btn_stop = tk.Button(
            btn_row, text="STOP", font=("Courier New", 9, "bold"),
            bg=BG_INPUT, fg=RED, activebackground=RED_DIM,
            activeforeground=TEXT_PRIMARY, relief="flat", cursor="hand2",
            command=self._on_stop
        )
        self.btn_stop.pack(side="right", fill="x", expand=True, padx=(4, 0))

        # Callbacks (set by main app after robot is ready)
        self.on_move_cb = None
        self.on_stop_cb = None

    def _stat_label(self, parent, tag, value):
        f = tk.Frame(parent, bg=BG_CARD)
        f.pack(side="left", expand=True)
        tk.Label(f, text=tag, font=("Courier New", 6), fg=TEXT_MUTED,
                 bg=BG_CARD).pack()
        lbl = tk.Label(f, text=value, font=("Courier New", 9, "bold"),
                       fg=TEXT_PRIMARY, bg=BG_CARD)
        lbl.pack()
        return lbl

    def _on_slider(self, *_):
        self.lbl_slider_val.config(text=f"{self.slider_var.get():.1f}°")

    def _on_move(self):
        if self.on_move_cb:
            try:
                speed = float(self.speed_var.get())
            except ValueError:
                speed = self.cfg.max_speed_dps
            self.on_move_cb(self.cfg.motor_id, self.slider_var.get(), speed)

    def _on_stop(self):
        if self.on_stop_cb:
            self.on_stop_cb(self.cfg.motor_id)

    def update_feedback(self, pos, vel, cur, temp):
        """Called from the monitoring thread via after()."""
        self.lbl_pos.config(text=f"{pos:+.1f}°")
        self.lbl_vel.config(text=f"{vel:+.1f}°/s")
        self.lbl_cur.config(text=f"{cur:.2f}A")
        color = YELLOW if temp > 60 else (RED if temp > 75 else TEXT_PRIMARY)
        self.lbl_temp.config(text=f"{temp:.1f}°C", fg=color)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class RobotGUI(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("RMD-X8-120  ·  6-Axis Control Panel")
        self.configure(bg=BG_DARK)
        self.resizable(True, True)
        self.minsize(1050, 700)

        # State
        self.robot:    Optional[RobotController] = None
        self.bus:      Optional[CANBus]          = None
        self._connected   = False
        self._monitor_job = None

        # Setup logging → queue
        handler = QueueHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                                               datefmt="%H:%M:%S"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

        self._build_ui()
        self._poll_log()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Top bar ───────────────────────────────────────────────────────────
        topbar = tk.Frame(self, bg=BG_PANEL, height=56)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)

        tk.Label(topbar, text="⬡  RMD-X8-120", font=("Courier New", 16, "bold"),
                 fg=ACCENT, bg=BG_PANEL).pack(side="left", padx=20, pady=10)
        tk.Label(topbar, text="6-AXIS MOTOR CONTROL PANEL",
                 font=("Courier New", 8), fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left", pady=18)

        # Status indicator (right side of topbar)
        self.lbl_status = tk.Label(topbar, text="● DISCONNECTED",
                                   font=("Courier New", 9, "bold"),
                                   fg=RED, bg=BG_PANEL)
        self.lbl_status.pack(side="right", padx=20)

        # ── Connection panel ──────────────────────────────────────────────────
        conn = tk.Frame(self, bg=BG_PANEL, pady=8)
        conn.pack(fill="x", padx=0)

        inner = tk.Frame(conn, bg=BG_PANEL)
        inner.pack(padx=20)

        # Mode
        tk.Label(inner, text="MODE", font=("Courier New", 7), fg=TEXT_MUTED,
                 bg=BG_PANEL).grid(row=0, column=0, sticky="w", padx=(0,4))
        self.mode_var = tk.StringVar(value="Simulation")
        mode_menu = ttk.Combobox(inner, textvariable=self.mode_var,
                                 values=["Simulation", "Real Hardware"],
                                 state="readonly", width=14,
                                 font=("Courier New", 9))
        mode_menu.grid(row=1, column=0, padx=(0, 12))
        mode_menu.bind("<<ComboboxSelected>>", self._on_mode_change)

        # Port
        tk.Label(inner, text="PORT", font=("Courier New", 7), fg=TEXT_MUTED,
                 bg=BG_PANEL).grid(row=0, column=1, sticky="w", padx=(0,4))
        self.port_var = tk.StringVar(value="COM3")
        self.port_entry = tk.Entry(inner, textvariable=self.port_var, width=14,
                                   font=("Courier New", 9), bg=BG_INPUT, fg=TEXT_PRIMARY,
                                   insertbackground=TEXT_PRIMARY, relief="flat",
                                   highlightbackground=BORDER, highlightthickness=1,
                                   state="disabled")
        self.port_entry.grid(row=1, column=1, padx=(0, 12))

        # Bus type
        tk.Label(inner, text="BUS TYPE", font=("Courier New", 7), fg=TEXT_MUTED,
                 bg=BG_PANEL).grid(row=0, column=2, sticky="w", padx=(0,4))
        self.bustype_var = tk.StringVar(value="slcan")
        self.bustype_menu = ttk.Combobox(inner, textvariable=self.bustype_var,
                                         values=["slcan","pcan","kvaser","socketcan"],
                                         state="disabled", width=10,
                                         font=("Courier New", 9))
        self.bustype_menu.grid(row=1, column=2, padx=(0, 12))

        # Bitrate
        tk.Label(inner, text="BITRATE", font=("Courier New", 7), fg=TEXT_MUTED,
                 bg=BG_PANEL).grid(row=0, column=3, sticky="w", padx=(0,4))
        self.bitrate_var = tk.StringVar(value="1000000")
        self.bitrate_menu = ttk.Combobox(inner, textvariable=self.bitrate_var,
                                         values=["1000000","500000","250000"],
                                         state="disabled", width=10,
                                         font=("Courier New", 9))
        self.bitrate_menu.grid(row=1, column=3, padx=(0, 20))

        # Connect button
        self.btn_connect = tk.Button(inner, text="CONNECT", width=12,
                                     font=("Courier New", 10, "bold"),
                                     bg=ACCENT, fg=BG_DARK,
                                     activebackground=ACCENT_DIM, relief="flat",
                                     cursor="hand2", command=self._on_connect)
        self.btn_connect.grid(row=1, column=4, padx=(0, 8))

        self.btn_disconnect = tk.Button(inner, text="DISCONNECT", width=12,
                                        font=("Courier New", 10, "bold"),
                                        bg=BG_INPUT, fg=TEXT_MUTED,
                                        relief="flat", cursor="hand2",
                                        state="disabled",
                                        command=self._on_disconnect)
        self.btn_disconnect.grid(row=1, column=5)

        # Separator
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # ── Global controls ───────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=BG_DARK, pady=10)
        ctrl.pack(fill="x", padx=20)

        tk.Label(ctrl, text="GLOBAL CONTROLS", font=("Courier New", 7),
                 fg=TEXT_MUTED, bg=BG_DARK).pack(side="left", padx=(0,12))

        btns = [
            ("HOME ALL",       ACCENT,    self._on_home),
            ("MOVE ALL",       "#4dabf7", self._on_move_all),
            ("STOP ALL",       YELLOW,    self._on_stop_all),
            ("⚠ E-STOP",       RED,       self._on_estop),
            ("RESET E-STOP",   BG_INPUT,  self._on_reset),
            ("RUN TEST SUITE", BG_INPUT,  self._on_run_tests),
        ]
        for label, color, cmd in btns:
            fg = BG_DARK if color not in (BG_INPUT,) else TEXT_PRIMARY
            if label == "⚠ E-STOP":
                fg = TEXT_PRIMARY
            tk.Button(ctrl, text=label, font=("Courier New", 9, "bold"),
                      bg=color, fg=fg, activebackground=color,
                      relief="flat", cursor="hand2", padx=10, pady=4,
                      command=cmd).pack(side="left", padx=4)

        # ── Joint cards grid ──────────────────────────────────────────────────
        cards_frame = tk.Frame(self, bg=BG_DARK)
        cards_frame.pack(fill="both", expand=True, padx=20, pady=(0, 8))

        self.cards: dict[int, JointCard] = {}
        for i, cfg in enumerate(DEFAULT_JOINT_CONFIG):
            row, col = divmod(i, 3)
            card = JointCard(cards_frame, cfg, JOINT_COLORS[i])
            card.grid(row=row, column=col, padx=6, pady=6, sticky="nsew")
            card.on_move_cb = self._move_joint
            card.on_stop_cb = self._stop_joint
            self.cards[cfg.motor_id] = card
            cards_frame.columnconfigure(col, weight=1)
        for r in range(2):
            cards_frame.rowconfigure(r, weight=1)

        # ── Log panel ─────────────────────────────────────────────────────────
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        log_header = tk.Frame(self, bg=BG_PANEL)
        log_header.pack(fill="x")
        tk.Label(log_header, text="SYSTEM LOG", font=("Courier New", 7),
                 fg=TEXT_MUTED, bg=BG_PANEL, pady=4).pack(side="left", padx=12)
        tk.Button(log_header, text="CLEAR", font=("Courier New", 7),
                  bg=BG_PANEL, fg=TEXT_MUTED, relief="flat", cursor="hand2",
                  command=self._clear_log).pack(side="right", padx=12)

        self.log_box = scrolledtext.ScrolledText(
            self, height=7, font=("Courier New", 8),
            bg=BG_PANEL, fg=TEXT_MUTED, insertbackground=TEXT_PRIMARY,
            relief="flat", state="disabled", wrap="word",
            selectbackground=BORDER,
        )
        self.log_box.pack(fill="x", padx=0, pady=0)
        self.log_box.tag_config("INFO",     foreground=TEXT_MUTED)
        self.log_box.tag_config("WARNING",  foreground=YELLOW)
        self.log_box.tag_config("ERROR",    foreground=RED)
        self.log_box.tag_config("CRITICAL", foreground=RED)

        # Apply ttk style for comboboxes
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=BG_INPUT, background=BG_INPUT,
                        foreground=TEXT_PRIMARY, selectbackground=BG_INPUT,
                        selectforeground=TEXT_PRIMARY, bordercolor=BORDER,
                        arrowcolor=TEXT_MUTED)

    # ── Mode toggle ───────────────────────────────────────────────────────────

    def _on_mode_change(self, *_):
        real = self.mode_var.get() == "Real Hardware"
        state = "normal" if real else "disabled"
        self.port_entry.config(state=state)
        self.bustype_menu.config(state="readonly" if real else "disabled")
        self.bitrate_menu.config(state="readonly" if real else "disabled")

    # ── Connect / Disconnect ──────────────────────────────────────────────────

    def _on_connect(self):
        if self._connected:
            return
        self._log("Connecting...")
        threading.Thread(target=self._connect_thread, daemon=True).start()

    def _connect_thread(self):
        try:
            simulated = self.mode_var.get() == "Simulation"
            self.bus = CANBus(
                simulated=simulated,
                channel=self.port_var.get(),
                bustype=self.bustype_var.get(),
                bitrate=int(self.bitrate_var.get()),
                num_motors=6,
            )
            self.robot = RobotController(self.bus, DEFAULT_JOINT_CONFIG)
            self.robot.start()
            self._connected = True
            self.after(0, self._on_connected)
        except Exception as e:
            self.after(0, lambda: self._on_connect_failed(str(e)))

    def _on_connected(self):
        self.lbl_status.config(text="● CONNECTED", fg=ACCENT)
        self.btn_connect.config(state="disabled", bg=BG_INPUT, fg=TEXT_MUTED)
        self.btn_disconnect.config(state="normal", bg=RED_DIM, fg=TEXT_PRIMARY)
        self._log("✓ Connected. All motors enabled.")
        self._start_monitor()

    def _on_connect_failed(self, err):
        self._log(f"✗ Connection failed: {err}", level="ERROR")
        messagebox.showerror("Connection Failed", err)

    def _on_disconnect(self):
        self._stop_monitor()
        if self.robot:
            self.robot.close()
            self.robot = None
        self.bus = None
        self._connected = False
        self.lbl_status.config(text="● DISCONNECTED", fg=RED)
        self.btn_connect.config(state="normal", bg=ACCENT, fg=BG_DARK)
        self.btn_disconnect.config(state="disabled", bg=BG_INPUT, fg=TEXT_MUTED)
        self._log("Disconnected.")

    # ── Motor commands (run in background threads) ────────────────────────────

    def _require_connected(self) -> bool:
        if not self._connected or not self.robot:
            messagebox.showwarning("Not Connected", "Please connect first.")
            return False
        return True

    def _move_joint(self, motor_id: int, position: float, speed: float):
        if not self._require_connected(): return
        threading.Thread(
            target=self.robot.move_joint,
            args=(motor_id, position, speed, False),
            daemon=True
        ).start()

    def _stop_joint(self, motor_id: int):
        if not self._require_connected(): return
        threading.Thread(
            target=lambda: self.robot.motors[motor_id].stop(),
            daemon=True
        ).start()

    def _on_home(self):
        if not self._require_connected(): return
        threading.Thread(
            target=self.robot.go_home, kwargs={"speed_dps": 60.0, "wait": False},
            daemon=True
        ).start()
        self._log("→ Homing all joints to 0°")

    def _on_move_all(self):
        """Move all joints to the positions currently set on their sliders."""
        if not self._require_connected(): return
        positions = {}
        for motor_id, card in self.cards.items():
            positions[motor_id] = card.slider_var.get()
        threading.Thread(
            target=self.robot.move_all_joints,
            kwargs={"positions": positions, "speed_dps": 90.0, "wait": False},
            daemon=True
        ).start()
        self._log(f"→ Moving all joints to slider positions")

    def _on_stop_all(self):
        if not self._require_connected(): return
        threading.Thread(target=self.robot.stop_all, daemon=True).start()
        self._log("→ Stop all motors")

    def _on_estop(self):
        if not self._require_connected(): return
        self.robot.estop()
        self.lbl_status.config(text="⚠ E-STOP ACTIVE", fg=RED)
        self._log("⚠ EMERGENCY STOP ACTIVATED", level="WARNING")

    def _on_reset(self):
        if not self._require_connected(): return
        self.robot.reset()
        self.lbl_status.config(text="● CONNECTED", fg=ACCENT)
        self._log("✓ E-Stop cleared. Motors re-enabled.")

    def _on_run_tests(self):
        if not self._require_connected(): return
        import subprocess, sys
        subprocess.Popen([sys.executable, "main.py", "--tests", "0,4,2,3,5"])
        self._log("→ Launched test suite in separate terminal window")

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
            for motor_id, card in self.cards.items():
                fb = fb_all.get(motor_id)
                if fb:
                    card.update_feedback(fb.position_deg, fb.velocity_dps,
                                         fb.current_a, fb.temperature_c)
            self._monitor_job = self.after(200, self._monitor_tick)

    # ── Log panel ─────────────────────────────────────────────────────────────

    def _poll_log(self):
        """Drain the log queue and write to the log panel."""
        while not log_queue.empty():
            msg = log_queue.get_nowait()
            level = "INFO"
            for l in ("WARNING", "ERROR", "CRITICAL"):
                if f"[{l}]" in msg:
                    level = l
                    break
            self.log_box.config(state="normal")
            self.log_box.insert("end", msg + "\n", level)
            self.log_box.see("end")
            self.log_box.config(state="disabled")
        self.after(100, self._poll_log)

    def _log(self, msg: str, level: str = "INFO"):
        self.log_box.config(state="normal")
        ts = time.strftime("%H:%M:%S")
        self.log_box.insert("end", f"{ts} [{level}] {msg}\n", level)
        self.log_box.see("end")
        self.log_box.config(state="disabled")

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
            except Exception:
                pass
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = RobotGUI()
    app.mainloop()
