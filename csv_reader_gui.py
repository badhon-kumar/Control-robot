"""
gui.py
======
Graphical User Interface for MyActuator RMD-X8-120 Motor Control System
6-Axis Robotic Arm Control Panel

Run with:
    python gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading
import time
import logging
import queue
import csv
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
        header = tk.Frame(self, bg=self.color, height=4)
        header.pack(fill="x")

        title_row = tk.Frame(self, bg=BG_CARD)
        title_row.pack(fill="x", padx=12, pady=(10, 4))

        tk.Label(title_row, text=f"J{self.cfg.motor_id}", font=("Courier New", 18, "bold"),
                 fg=self.color, bg=BG_CARD).pack(side="left")
        tk.Label(title_row, text=self.cfg.name.upper(), font=("Courier New", 9),
                 fg=TEXT_MUTED, bg=BG_CARD).pack(side="left", padx=(8, 0), pady=(6, 0))

        fb_frame = tk.Frame(self, bg=BG_CARD)
        fb_frame.pack(fill="x", padx=12, pady=2)

        self.lbl_pos  = self._stat_label(fb_frame, "POS",  "0.00°")
        self.lbl_vel  = self._stat_label(fb_frame, "VEL",  "0.0°/s")
        self.lbl_cur  = self._stat_label(fb_frame, "CUR",  "0.00A")
        self.lbl_temp = self._stat_label(fb_frame, "TEMP", "35.0°C")

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

        self.on_move_cb = None
        self.on_stop_cb = None

    def _stat_label(self, parent, tag, value):
        f = tk.Frame(parent, bg=BG_CARD)
        f.pack(side="left", expand=True)
        tk.Label(f, text=tag, font=("Courier New", 6), fg=TEXT_MUTED, bg=BG_CARD).pack()
        lbl = tk.Label(f, text=value, font=("Courier New", 9, "bold"), fg=TEXT_PRIMARY, bg=BG_CARD)
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
        self.minsize(1050, 800)

        self.robot:    Optional[RobotController] = None
        self.bus:      Optional[CANBus]          = None
        self._connected   = False
        
        # Hardware polling state
        self._monitor_running = False
        self._latest_feedback = {}
        self._ui_update_job = None
        
        # CSV Sequence state
        self.sequence_data = [] 
        self._seq_running = False
        self._seq_start_time = 0.0
        self._current_step = 1
        self._step_elapsed = 0.0
        self._step_target = 0.0

        handler = QueueHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))
        logging.getLogger().addHandler(handler)
        logging.getLogger().setLevel(logging.INFO)

        self._build_ui()
        self._poll_log()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # ── Top bar ───────────────────────────────────────────────────────────
        topbar = tk.Frame(self, bg=BG_PANEL, height=56)
        topbar.pack(fill="x")
        topbar.pack_propagate(False)

        tk.Label(topbar, text="⬡  RMD-X8-120", font=("Courier New", 16, "bold"), fg=ACCENT, bg=BG_PANEL).pack(side="left", padx=20, pady=10)
        tk.Label(topbar, text="6-AXIS MOTOR CONTROL PANEL", font=("Courier New", 8), fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left", pady=18)

        self.lbl_status = tk.Label(topbar, text="● DISCONNECTED", font=("Courier New", 9, "bold"), fg=RED, bg=BG_PANEL)
        self.lbl_status.pack(side="right", padx=20)

        # ── Connection panel ──────────────────────────────────────────────────
        conn = tk.Frame(self, bg=BG_PANEL, pady=8)
        conn.pack(fill="x", padx=0)

        inner = tk.Frame(conn, bg=BG_PANEL)
        inner.pack(padx=20)

        tk.Label(inner, text="MODE", font=("Courier New", 7), fg=TEXT_MUTED, bg=BG_PANEL).grid(row=0, column=0, sticky="w", padx=(0,4))
        self.mode_var = tk.StringVar(value="Simulation")
        mode_menu = ttk.Combobox(inner, textvariable=self.mode_var, values=["Simulation", "Real Hardware"], state="readonly", width=14, font=("Courier New", 9))
        mode_menu.grid(row=1, column=0, padx=(0, 12))
        mode_menu.bind("<<ComboboxSelected>>", self._on_mode_change)

        tk.Label(inner, text="PORT", font=("Courier New", 7), fg=TEXT_MUTED, bg=BG_PANEL).grid(row=0, column=1, sticky="w", padx=(0,4))
        self.port_var = tk.StringVar(value="COM3")
        self.port_entry = tk.Entry(inner, textvariable=self.port_var, width=14, font=("Courier New", 9), bg=BG_INPUT, fg=TEXT_PRIMARY, insertbackground=TEXT_PRIMARY, relief="flat", highlightbackground=BORDER, highlightthickness=1, state="disabled")
        self.port_entry.grid(row=1, column=1, padx=(0, 12))

        tk.Label(inner, text="BUS TYPE", font=("Courier New", 7), fg=TEXT_MUTED, bg=BG_PANEL).grid(row=0, column=2, sticky="w", padx=(0,4))
        self.bustype_var = tk.StringVar(value="slcan")
        self.bustype_menu = ttk.Combobox(inner, textvariable=self.bustype_var, values=["slcan","pcan","kvaser","socketcan"], state="disabled", width=10, font=("Courier New", 9))
        self.bustype_menu.grid(row=1, column=2, padx=(0, 12))

        tk.Label(inner, text="BITRATE", font=("Courier New", 7), fg=TEXT_MUTED, bg=BG_PANEL).grid(row=0, column=3, sticky="w", padx=(0,4))
        self.bitrate_var = tk.StringVar(value="1000000")
        self.bitrate_menu = ttk.Combobox(inner, textvariable=self.bitrate_var, values=["1000000","500000","250000"], state="disabled", width=10, font=("Courier New", 9))
        self.bitrate_menu.grid(row=1, column=3, padx=(0, 20))

        self.btn_connect = tk.Button(inner, text="CONNECT", width=12, font=("Courier New", 10, "bold"), bg=ACCENT, fg=BG_DARK, activebackground=ACCENT_DIM, relief="flat", cursor="hand2", command=self._on_connect)
        self.btn_connect.grid(row=1, column=4, padx=(0, 8))

        self.btn_disconnect = tk.Button(inner, text="DISCONNECT", width=12, font=("Courier New", 10, "bold"), bg=BG_INPUT, fg=TEXT_MUTED, relief="flat", cursor="hand2", state="disabled", command=self._on_disconnect)
        self.btn_disconnect.grid(row=1, column=5)

        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # ── Global controls ───────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=BG_DARK, pady=10)
        ctrl.pack(fill="x", padx=20)

        tk.Label(ctrl, text="GLOBAL CONTROLS", font=("Courier New", 7), fg=TEXT_MUTED, bg=BG_DARK).pack(side="left", padx=(0,12))

        btns = [
            ("HOME ALL",       ACCENT,    self._on_home),
            ("MOVE ALL",       "#4dabf7", self._on_move_all),
            ("STOP ALL",       YELLOW,    self._on_stop_all),
            ("⚠ E-STOP",       RED,       self._on_estop),
            ("RESET E-STOP",   BG_INPUT,  self._on_reset),
        ]
        for label, color, cmd in btns:
            fg = BG_DARK if color not in (BG_INPUT,) else TEXT_PRIMARY
            if label == "⚠ E-STOP": fg = TEXT_PRIMARY
            tk.Button(ctrl, text=label, font=("Courier New", 9, "bold"), bg=color, fg=fg, activebackground=color, relief="flat", cursor="hand2", padx=10, pady=4, command=cmd).pack(side="left", padx=4)

        # ── Sequence Runner ───────────────────────────────────────────────────
        seq_frame = tk.Frame(self, bg=BG_PANEL, pady=10)
        seq_frame.pack(fill="x", padx=20, pady=(0, 8))

        tk.Label(seq_frame, text="CSV AUTOMATION", font=("Courier New", 7), fg=TEXT_MUTED, bg=BG_PANEL).pack(side="left", padx=(12, 12))

        self.btn_load_csv = tk.Button(seq_frame, text="LOAD CSV", font=("Courier New", 9, "bold"), bg=BG_INPUT, fg=TEXT_PRIMARY, relief="flat", cursor="hand2", command=self._load_csv)
        self.btn_load_csv.pack(side="left", padx=4)

        self.lbl_csv_file = tk.Label(seq_frame, text="No sequence loaded", font=("Courier New", 9), fg=YELLOW, bg=BG_PANEL)
        self.lbl_csv_file.pack(side="left", padx=12)

        self.btn_play_csv = tk.Button(seq_frame, text="▶ PLAY SEQUENCE", font=("Courier New", 9, "bold"), bg=ACCENT, fg=BG_DARK, activebackground=ACCENT_DIM, relief="flat", cursor="hand2", state="disabled", command=self._play_csv)
        self.btn_play_csv.pack(side="right", padx=12)

        # ── LIVE STOPWATCH DISPLAY ──
        self.lbl_seq_timer = tk.Label(seq_frame, text="", font=("Courier New", 10, "bold"), fg=YELLOW, bg=BG_PANEL)
        self.lbl_seq_timer.pack(side="right", padx=20)

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
        tk.Label(log_header, text="SYSTEM LOG", font=("Courier New", 7), fg=TEXT_MUTED, bg=BG_PANEL, pady=4).pack(side="left", padx=12)
        tk.Button(log_header, text="CLEAR", font=("Courier New", 7), bg=BG_PANEL, fg=TEXT_MUTED, relief="flat", cursor="hand2", command=self._clear_log).pack(side="right", padx=12)

        self.log_box = scrolledtext.ScrolledText(self, height=7, font=("Courier New", 8), bg=BG_PANEL, fg=TEXT_MUTED, insertbackground=TEXT_PRIMARY, relief="flat", state="disabled", wrap="word", selectbackground=BORDER)
        self.log_box.pack(fill="x", padx=0, pady=0)
        self.log_box.tag_config("INFO", foreground=TEXT_MUTED)
        self.log_box.tag_config("WARNING", foreground=YELLOW)
        self.log_box.tag_config("ERROR", foreground=RED)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TCombobox", fieldbackground=BG_INPUT, background=BG_INPUT, foreground=TEXT_PRIMARY, selectbackground=BG_INPUT, selectforeground=TEXT_PRIMARY, bordercolor=BORDER, arrowcolor=TEXT_MUTED)

    # ── Logic ───────────────────────────────────────────────────────────────

    def _on_mode_change(self, *_):
        real = self.mode_var.get() == "Real Hardware"
        state = "normal" if real else "disabled"
        self.port_entry.config(state=state)
        self.bustype_menu.config(state="readonly" if real else "disabled")
        self.bitrate_menu.config(state="readonly" if real else "disabled")

    def _on_connect(self):
        if self._connected: return
        self._log("Connecting...")
        threading.Thread(target=self._connect_thread, daemon=True).start()

    def _connect_thread(self):
        try:
            simulated = self.mode_var.get() == "Simulation"
            self.bus = CANBus(simulated=simulated, channel=self.port_var.get(), bustype=self.bustype_var.get(), bitrate=int(self.bitrate_var.get()), num_motors=6)
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

    def _require_connected(self) -> bool:
        if not self._connected or not self.robot:
            messagebox.showwarning("Not Connected", "Please connect first.")
            return False
        return True

    def _move_joint(self, motor_id: int, position: float, speed: float):
        if not self._require_connected(): return
        threading.Thread(target=self.robot.move_joint, args=(motor_id, position, speed, False), daemon=True).start()

    def _stop_joint(self, motor_id: int):
        if not self._require_connected(): return
        threading.Thread(target=lambda: self.robot.motors[motor_id].stop(), daemon=True).start()

    def _on_home(self):
        if not self._require_connected(): return
        threading.Thread(target=self.robot.go_home, kwargs={"speed_dps": 60.0, "wait": False}, daemon=True).start()
        self._log("→ Homing all joints to 0°")

    def _on_move_all(self):
        if not self._require_connected(): return
        positions = {motor_id: card.slider_var.get() for motor_id, card in self.cards.items()}
        threading.Thread(target=self.robot.move_all_joints, kwargs={"positions": positions, "speed_dps": 90.0, "wait": False}, daemon=True).start()
        self._log("→ Moving all joints to slider positions")

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

    # ── CSV SEQUENCE LOGIC ────────────────────────────────────────────────────
    
    def _load_csv(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not filepath:
            return
        
        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                self.sequence_data = list(reader)
            
            filename = filepath.split('/')[-1]
            self.lbl_csv_file.config(text=f"Loaded: {filename}", fg=ACCENT)
            self.btn_play_csv.config(state="normal")
            self._log(f"Loaded CSV sequence with {len(self.sequence_data)} steps.")
        except Exception as e:
            messagebox.showerror("CSV Error", f"Failed to read CSV:\n{e}")
            self._log(f"Failed to load CSV: {e}", level="ERROR")

    def _play_csv(self):
        if not self._require_connected(): return
        if not self.sequence_data: return
        
        self.btn_play_csv.config(state="disabled", text="EXECUTING...")
        
        # Reset and start the UI Timer
        self._seq_running = True
        self._seq_start_time = time.time()
        self._current_step = 1
        self._step_elapsed = 0.0
        self._step_target = 0.0
        self._update_seq_timer()
        
        # Launch sequence thread
        threading.Thread(target=self._sequence_thread, daemon=True).start()

    def _update_seq_timer(self):
        """Recursively updates the stopwatch label on the screen."""
        if self._seq_running:
            total_elapsed = time.time() - self._seq_start_time
            step = getattr(self, '_current_step', 1)
            step_el = getattr(self, '_step_elapsed', 0.0)
            step_tgt = getattr(self, '_step_target', 0.0)
            
            # Format:  Step 1 Delay: 1.5s / 5.0s  |  Total: 12.3s
            if step_tgt > 0:
                timer_text = f"Step {step} Delay: {step_el:.1f}s / {step_tgt:.1f}s  |  Total: {total_elapsed:.1f}s"
            else:
                timer_text = f"Step {step} Moving...  |  Total: {total_elapsed:.1f}s"
                
            self.lbl_seq_timer.config(text=timer_text, fg=ACCENT)
            self.after(100, self._update_seq_timer)

    def _sequence_thread(self):
        self._log("▶ Starting CSV Sequence...")
        
        try:
            for i, row in enumerate(self.sequence_data):
                if not self._connected or getattr(self.robot, '_estop', False):
                    self._log("Sequence aborted due to E-Stop or disconnect.", level="WARNING")
                    break
                    
                self._current_step = i + 1
                self._step_elapsed = 0.0
                self._step_target = 0.0
                    
                positions = {}
                for j in range(1, 7):
                    val = row.get(f"J{j}", "").strip()
                    if val:
                        positions[j] = float(val)
                
                speed = float(row.get("Speed", 60.0).strip() or 60.0)
                delay = float(row.get("Delay", 1.0).strip() or 1.0)
                
                if positions:
                    self._log(f"Step {i+1}: Moving joints {list(positions.keys())} at {speed}°/s")
                    
                    # Ghost-in-the-machine animation
                    for j_id, tgt in positions.items():
                        if j_id in self.cards:
                            self.after(0, lambda j=j_id, t=tgt: self.cards[j].slider_var.set(t))
                    
                    self.robot.move_all_joints(positions=positions, speed_dps=speed, wait=True)
                
                if delay > 0:
                    self._log(f"Step {i+1}: Waiting {delay}s...")
                    self._step_target = delay
                    step_start = time.time()
                    
                    # Safety Interrupt Loop: Break delay into tiny pieces to check E-Stop
                    while time.time() - step_start < delay:
                        if not self._connected or getattr(self.robot, '_estop', False):
                            break
                        self._step_elapsed = time.time() - step_start
                        time.sleep(0.05)
                
            if not getattr(self.robot, '_estop', False):
                self._log("✓ Sequence complete.")
                
        except Exception as e:
            self._log(f"Sequence failed: {e}", level="ERROR")
            
        # Clean up UI after sequence finishes or aborts
        self._seq_running = False
        self.after(0, lambda: self.btn_play_csv.config(state="normal", text="▶ PLAY SEQUENCE"))
        self.after(0, lambda: self.lbl_seq_timer.config(text="Sequence Finished", fg=YELLOW))

    # ── BACKGROUND MONITOR (Decoupled & Non-Blocking) ─────────────────────────

    def _start_monitor(self):
        self._monitor_running = True
        self._latest_feedback = {}
        threading.Thread(target=self._hardware_poll_loop, daemon=True).start()
        self._ui_update_job = self.after(100, self._ui_update_tick)

    def _hardware_poll_loop(self):
        while self._monitor_running:
            if self.robot and self._connected:
                try:
                    self._latest_feedback = self.robot.get_all_feedback()
                except Exception:
                    pass
            time.sleep(0.05)

    def _stop_monitor(self):
        self._monitor_running = False
        if getattr(self, '_ui_update_job', None):
            self.after_cancel(self._ui_update_job)
            self._ui_update_job = None

    def _ui_update_tick(self):
        if self.robot and self._connected and self._latest_feedback:
            for motor_id, card in self.cards.items():
                fb = self._latest_feedback.get(motor_id)
                if fb:
                    card.update_feedback(fb.position_deg, fb.velocity_dps, fb.current_a, fb.temperature_c)
        self._ui_update_job = self.after(100, self._ui_update_tick)

    # ── LOGGING ROUTINES ──────────────────────────────────────────────────────

    def _poll_log(self):
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

    def _on_close(self):
        self._stop_monitor()
        if self.robot:
            try: self.robot.close()
            except Exception: pass
        self.destroy()

if __name__ == "__main__":
    app = RobotGUI()
    app.mainloop()