"""
continuum_gui.py  —  v2.0
═══════════════════════════════════════════════════════════════════════════════
Tendon-Driven Continuum Manipulator  ·  Control GUI
3-Segment  ·  6 Tendons  ·  CAN Bus  ·  Piecewise Constant Curvature (PCC)

TENDON TOPOLOGY  (from Zhai et al. 2025, Fig. 2 & Fig. 3)
──────────────────────────────────────────────────────────
The manipulator has 3 segments.  Each segment is actuated by ONE antagonistic
tendon pair:
  Segment 1  →  M1 (T1+) / M2 (T1−)   — bending angle θ₁
  Segment 2  →  M3 (T2+) / M4 (T2−)   — bending angle θ₂
  Segment 3  →  M5 (T3+) / M6 (T3−)   — bending angle θ₃

Tendon routing (Fig. 2C): the driving tendons for segment i attach to the
end-disk of segment i and slide freely through all preceding disks.
Therefore M1/M2 control ONLY segment 1 bending, M3/M4 ONLY segment 2, etc.
The actuation input for each segment is the tendon length differential:
    Δlᵢ = lᵢ,1 − lᵢ,2   (pull M_pos / release M_neg → positive bend)

PCC KINEMATICS  (Eqs. 1–5, Zhai et al. 2025)
──────────────────────────────────────────────
    θᵢ = Δlᵢ / rᵢ     (bending angle per segment, rᵢ = tendon offset radius)
    x  = Σ (Lᵢ/θᵢ) · [sin(Σθ) − sin(Σθ−θᵢ)]
    y  = Σ (Lᵢ/θᵢ) · [cos(Σθ−θᵢ) − cos(Σθ)]
    ψ  = θ₁ + θ₂ + θ₃

CONTROL
────────
  ① MANUAL JOG   — per-motor displacement sliders + direct entry
  ② CSV STATES   — load / edit / run CSV state sequences in the built-in editor

Run:
    python continuum_gui.py             ← simulation
    python continuum_gui.py --real --port COM3
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import threading, time, logging, queue, csv, os, math, argparse, io
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

# ── Optional robot hardware imports ──────────────────────────────────────────
try:
    from can_interface    import CANBus
    from robot_controller import RobotController, DEFAULT_JOINT_CONFIG
    HAS_ROBOT = True
except ImportError:
    HAS_ROBOT = False

# ══════════════════════════════════════════════════════════════════════════════
# ROBOT PHYSICAL PARAMETERS  (Table 1, Zhai et al. 2025)
# ══════════════════════════════════════════════════════════════════════════════

SEG_LENGTHS_MM   = [90.0, 90.0, 90.0]       # L₁, L₂, L₃  (mm)
TENDON_OFFSETS_MM = [5.0, 3.5, 2.0]          # r₁, r₂, r₃  (mm)
CAPSTAN_RADII_MM  = [15.0] * 6               # per-motor drum radius (mm)
MAX_DISP_MM       = 120.0                    # max tendon displacement (mm)
MAX_SPEED_MMS     = 80.0                     # max tendon speed (mm/s)
TOTAL_LENGTH_MM   = sum(SEG_LENGTHS_MM)      # 270 mm

# Motor → Segment mapping (0-indexed)
#   M0, M1  →  Segment 0 (θ₁)
#   M2, M3  →  Segment 1 (θ₂)
#   M4, M5  →  Segment 2 (θ₃)
MOTOR_TO_SEG = [0, 0, 1, 1, 2, 2]
MOTOR_IS_POS = [True, False, True, False, True, False]  # positive / negative of pair

SEG_COLORS  = ["#00ddb8", "#3a9ef0", "#f5a623"]   # teal, blue, orange
SEG_NAMES   = ["Segment 1", "Segment 2", "Segment 3"]
MOTOR_COLORS = [
    SEG_COLORS[0], SEG_COLORS[0],
    SEG_COLORS[1], SEG_COLORS[1],
    SEG_COLORS[2], SEG_COLORS[2],
]
MOTOR_NAMES  = [
    "M1  S1+ (pull)", "M2  S1− (release)",
    "M3  S2+ (pull)", "M4  S2− (release)",
    "M5  S3+ (pull)", "M6  S3− (release)",
]
MOTOR_SHORT  = ["M1 S1+", "M2 S1-", "M3 S2+", "M4 S2-", "M5 S3+", "M6 S3-"]


def mm_to_deg(mm: float, motor_idx: int = 0) -> float:
    return (mm / CAPSTAN_RADII_MM[motor_idx]) * (180.0 / math.pi)

def deg_to_mm(deg: float, motor_idx: int = 0) -> float:
    return deg * CAPSTAN_RADII_MM[motor_idx] * (math.pi / 180.0)

def mms_to_dps(mms: float, motor_idx: int = 0) -> float:
    return (mms / CAPSTAN_RADII_MM[motor_idx]) * (180.0 / math.pi)


def compute_pcc_kinematics(disps_mm: List[float]) -> dict:
    """
    Distributed-curvature soft-robot forward kinematics.

    Physical model
    ──────────────
    The manipulator is a cantilever flexible beam (elastic backbone + spacer
    disks + rotating pins).  Six tendons run inside the structure:

      T1/T2  (M1/M2)  anchor at s = L1              (end of zone 1)
      T3/T4  (M3/M4)  anchor at s = L1+L2            (end of zone 2)
      T5/T6  (M5/M6)  anchor at s = L1+L2+L3 = L_tot (tip)

    Each tendon pair exerts a moment M_i = F_i · r_i over its entire running
    length from the base to its anchor point.  The net curvature κ(s) at arc
    position s is the superposition of all moments whose tendon is still running
    at s, divided by the beam's bending stiffness EI (absorbed into the gain).

        κ(s) = Σ_i  [ κ_i  ·  H(s_anchor_i − s) ]

    where κ_i = Δl_i / r_i  is the curvature contribution from tendon pair i
    and H is the Heaviside step (tendon i only runs from 0 to s_anchor_i).

    Because each tendon ALSO runs through the preceding zones it bends those
    zones too — exactly the soft-robot behaviour described.  A tendon anchored
    at the tip (M5/M6) contributes curvature along the full length; one anchored
    at end-disk 1 contributes only in zone 1.

    Integration is done numerically with N_PTS steps using the Frenet–Serret
    tangent update:
        θ(s+ds) = θ(s) + κ(s)·ds
        x(s+ds) = x(s) + sin(θ(s))·ds
        y(s+ds) = y(s) + cos(θ(s))·ds

    A smooth exponential roll-off near each anchor point prevents any curvature
    discontinuity (the real beam's stiffness smears the load transition).

    Returns
    ───────
      all_pts    : (N_PTS+1) world-frame (x,y) points for the full body
      pt_seg     : segment zone index (0/1/2) for each point
      curvature  : κ(s) array (rad/mm) at each point
      end_pts    : [(x,y)] at each tendon anchor (end of zones 1, 2, 3)
      thetas     : [θ_zone1, θ_zone2, θ_zone3] — net angle change per zone
      tip_x,y    : end-effector world position (mm)
      psi        : total tip angle (rad)
    """
    N_PTS    = 120          # integration steps — more = smoother curve
    L_tot    = TOTAL_LENGTH_MM
    ds       = L_tot / N_PTS

    # Tendon anchor positions along arc (mm from base)
    anchors  = [
        SEG_LENGTHS_MM[0],
        SEG_LENGTHS_MM[0] + SEG_LENGTHS_MM[1],
        L_tot,
    ]

    # Per-pair curvature contributions κ_i = Δl_i / r_i  (rad/mm)
    # Clamped to ±(π / L_i) so no single zone exceeds 180° of bending
    deltas = [
        disps_mm[0] - disps_mm[1],
        disps_mm[2] - disps_mm[3],
        disps_mm[4] - disps_mm[5],
    ]
    kappas = []
    for i in range(3):
        r  = TENDON_OFFSETS_MM[i]
        k  = (deltas[i] / r) / SEG_LENGTHS_MM[i] if r > 0 else 0.0
        # cap so worst-case single segment bends ≤ 150°
        lim = math.radians(150) / SEG_LENGTHS_MM[i]
        kappas.append(max(-lim, min(lim, k)))

    # Smooth Heaviside: H_smooth(s, s_anchor) ≈ 1 for s < s_anchor, rolls off
    # over ~10 mm so the transition is physically smooth (beam stiffness effect)
    ROLLOFF = 8.0   # mm over which curvature fades at anchor point
    def h_smooth(s, s_anchor):
        return 1.0 / (1.0 + math.exp((s - s_anchor) / (ROLLOFF / 6.0)))

    # Integrate Frenet–Serret along arc
    all_pts   = []
    curvature = []
    pt_seg    = []

    x, y, theta = 0.0, 0.0, 0.0   # base: upward = +y, lateral = +x, angle=0

    for n in range(N_PTS + 1):
        s = n * ds

        # Zone label for colouring
        if   s <= anchors[0]: seg = 0
        elif s <= anchors[1]: seg = 1
        else:                 seg = 2

        # Distributed curvature: sum of all tendon contributions active at s
        kappa_s = sum(kappas[i] * h_smooth(s, anchors[i]) for i in range(3))

        all_pts.append((x, y))
        curvature.append(kappa_s)
        pt_seg.append(seg)

        # Advance geometry
        x     += math.sin(theta) * ds
        y     += math.cos(theta) * ds
        theta += kappa_s * ds

    # Snap end-disk positions to the arc points nearest each anchor
    end_pts = []
    for anch in anchors:
        idx = min(int(round(anch / ds)), N_PTS)
        end_pts.append(all_pts[idx])

    tip_x, tip_y = all_pts[-1]
    psi = sum(kappas[i] * anchors[i] * 0.5 for i in range(3))   # approx attitude

    # Zone-level effective angles for HUD display
    def zone_angle(s0, s1):
        n0 = int(round(s0 / ds)); n1 = int(round(s1 / ds))
        return sum(curvature[k] * ds for k in range(n0, min(n1, N_PTS)))

    thetas = [
        zone_angle(0,           anchors[0]),
        zone_angle(anchors[0],  anchors[1]),
        zone_angle(anchors[1],  anchors[2]),
    ]
    psi = sum(thetas)

    return {
        "all_pts":   all_pts,
        "pt_seg":    pt_seg,
        "curvature": curvature,
        "end_pts":   end_pts,
        "thetas":    thetas,
        "tip_x":     tip_x,
        "tip_y":     tip_y,
        "psi":       psi,
        # legacy keys so HUD / top-view code that references seg_points still works
        "seg_points": [
            all_pts[: int(round(anchors[0]/ds)) + 1],
            all_pts[int(round(anchors[0]/ds)): int(round(anchors[1]/ds)) + 1],
            all_pts[int(round(anchors[1]/ds)):],
        ],
    }


# ══════════════════════════════════════════════════════════════════════════════
# THEME — industrial dark with precise accent colours
# ══════════════════════════════════════════════════════════════════════════════

BG       = "#e2e8f0"
PANEL    = "#cbd5e1"
CARD     = "#f1f5f9"
INPUT    = "#f8fafc"
BORDER   = "#94a3b8"
ACCENT   = "#0f766e"
BLUE     = "#2563eb"
RED      = "#dc2626"
YELLOW   = "#d97706"
GREEN    = "#16a34a"
ORANGE   = "#ea580c"
MUTED    = "#475569"
TEXT     = "#0f172a"
DIM      = "#64748b"
CBKG     = "#e2e8f0"
FNT      = "Courier New"
FNT_H    = "Courier New"   # heading font

_lq: queue.Queue = queue.Queue()
class _QH(logging.Handler):
    def emit(self, r): _lq.put(self.format(r))


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TendonState:
    name:    str
    delay_s: float
    motors:  list = field(default_factory=lambda: [(0.0, 20.0)] * 6)

    def summary(self) -> str:
        parts = [f"M{i+1}:{self.motors[i][0]:+.1f}@{self.motors[i][1]:.0f}"
                 for i in range(6)]
        return f"[{self.name}]  " + "  ".join(parts)


def parse_states_csv(path: str) -> Tuple[List[TendonState], str]:
    states, errors = [], []
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if "state_name" not in (reader.fieldnames or []):
                return [], "Missing required column: 'state_name'"
            for row_num, row in enumerate(reader, start=2):
                name = row.get("state_name", f"State {row_num}").strip()
                try:   delay = max(0.0, float(row.get("delay_s", "1.0") or "1.0"))
                except ValueError:
                    delay = 1.0
                    errors.append(f"Row {row_num}: bad delay_s → 1.0")
                motors = []
                for m in range(1, 7):
                    try:    disp = float(row.get(f"m{m}_disp_mm", "0") or "0")
                    except: disp = 0.0
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


def states_to_csv_text(states: List[TendonState]) -> str:
    """Serialise state list to CSV string (for built-in editor)."""
    header = ["state_name", "delay_s"]
    for m in range(1, 7):
        header += [f"m{m}_disp_mm", f"m{m}_speed_mms"]
    rows = [",".join(header)]
    for s in states:
        row = [s.name, f"{s.delay_s:.2f}"]
        for d, spd in s.motors:
            row += [f"{d:.2f}", f"{spd:.1f}"]
        rows.append(",".join(row))
    return "\n".join(rows)


def make_sample_states() -> List[TendonState]:
    """Return default demo states matching the correct tendon topology."""
    raw = [
        #  name              delay  M1    M2    M3    M4    M5    M6    (disp, speed)
        ("Home",             1.5,   (0,20),(0,20),(0,20),(0,20),(0,20),(0,20)),
        ("S1 Bend +30mm",    2.0,  (30,25),(0,25),(0,20),(0,20),(0,20),(0,20)),
        ("S1 Bend −30mm",    2.0,   (0,25),(30,25),(0,20),(0,20),(0,20),(0,20)),
        ("S2 Bend +30mm",    2.0,   (0,20),(0,20),(30,25),(0,25),(0,20),(0,20)),
        ("S2 Bend −30mm",    2.0,   (0,20),(0,20),(0,25),(30,25),(0,20),(0,20)),
        ("S3 Bend +30mm",    2.0,   (0,20),(0,20),(0,20),(0,20),(30,25),(0,25)),
        ("S3 Bend −30mm",    2.0,   (0,20),(0,20),(0,20),(0,20),(0,25),(30,25)),
        ("Forward Arc",      2.5,  (20,20),(0,20),(20,20),(0,20),(20,20),(0,20)),
        ("Reverse Arc",      2.5,   (0,20),(20,20),(0,20),(20,20),(0,20),(20,20)),
        ("Home",             1.0,   (0,30),(0,30),(0,30),(0,30),(0,30),(0,30)),
    ]
    out = []
    for r in raw:
        name, delay = r[0], r[1]
        motors = [(d, s) for d, s in r[2:]]
        out.append(TendonState(name=name, delay_s=delay, motors=motors))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# CAPSTAN PANEL  (6 motor drums)
# ══════════════════════════════════════════════════════════════════════════════

class CapstanPanel(tk.Frame):
    DRUM_R = 26

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._disps  = [0.0] * 6
        self._speeds = [0.0] * 6
        self._canvases, self._disp_lbls, self._speed_lbls, self._tgt_lbls = [], [], [], []
        self._build()

    def _build(self):
        hf = tk.Frame(self, bg=BG)
        hf.pack(fill="x", pady=(8, 4))
        tk.Label(hf, text="CAPSTAN DRUMS", font=(FNT_H, 12, "bold"),
                 fg=ACCENT, bg=BG).pack(side="left")
        tk.Label(hf, text="  ·  live tendon displacement per motor",
                 font=(FNT, 10), fg=MUTED, bg=BG).pack(side="left")

        # Segment group labels above
        grp = tk.Frame(self, bg=BG)
        grp.pack(fill="x")
        for i, (sc, sn) in enumerate(zip(SEG_COLORS, SEG_NAMES)):
            g = tk.Frame(grp, bg=BG)
            g.pack(side="left", expand=True, fill="x")
            tk.Frame(g, bg=sc, height=2).pack(fill="x")
            tk.Label(g, text=f"── {sn} ──", font=(FNT, 10, "bold"),
                     fg=sc, bg=BG).pack(pady=2)

        grid = tk.Frame(self, bg=BG)
        grid.pack(fill="x")

        for i in range(6):
            seg = MOTOR_TO_SEG[i]
            col = MOTOR_COLORS[i]
            sc  = SEG_COLORS[seg]
            c   = i % 2   # column within segment group
            g   = i // 2  # segment group

            cell = tk.Frame(grid, bg=CARD,
                            highlightbackground=BORDER, highlightthickness=1)
            cell.grid(row=0, column=i, padx=3, pady=2, sticky="nsew")
            grid.columnconfigure(i, weight=1)

            tk.Frame(cell, bg=sc, height=3).pack(fill="x")
            role = "PULL +" if MOTOR_IS_POS[i] else "REL −"
            tk.Label(cell, text=f"M{i+1}", font=(FNT, 14, "bold"),
                     fg=col, bg=CARD).pack(pady=(5, 0))
            tk.Label(cell, text=role, font=(FNT, 9), fg=DIM, bg=CARD).pack()

            cv = tk.Canvas(cell, width=100, height=66, bg=CARD, highlightthickness=0)
            cv.pack(padx=4, pady=3)
            self._canvases.append(cv)

            dl = tk.Label(cell, text="  0.0", font=(FNT, 13, "bold"),
                          fg=col, bg=CARD, anchor="center")
            dl.pack()
            self._disp_lbls.append(dl)

            vl = tk.Label(cell, text="0 mm/s", font=(FNT, 9),
                          fg=MUTED, bg=CARD)
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

        # Displacement bar (bottom strip)
        d     = self._disps[i]
        ratio = (d + MAX_DISP_MM) / (2 * MAX_DISP_MM)
        ratio = max(0.0, min(1.0, ratio))
        bar_w = int(w * ratio)
        cv.create_rectangle(0, h - 14, w, h, fill=INPUT, outline="")
        bar_col = col if d >= 0 else "#d1d5db"
        if bar_w > 0:
            cv.create_rectangle(0, h - 14, bar_w, h, fill=bar_col, outline="")
        cv.create_line(w // 2, h - 14, w // 2, h, fill=BORDER, dash=(2, 2))
        cv.create_text(w // 2, h - 6, text=f"{d:+.1f}mm",
                       fill=TEXT, font=(FNT, 9), anchor="center")

        # Drum circle
        dr = self.DRUM_R
        cx_, cy_ = w // 2, 28
        cv.create_oval(cx_ - dr, cy_ - dr, cx_ + dr, cy_ + dr,
                       fill="#f3f4f6", outline=col, width=2)

        # Rotation indicator
        ang  = math.radians(d / CAPSTAN_RADII_MM[i] * (180 / math.pi) % 360)
        ix   = cx_ + int(dr * 0.7 * math.cos(ang - math.pi / 2))
        iy   = cy_ + int(dr * 0.7 * math.sin(ang - math.pi / 2))
        cv.create_line(cx_, cy_, ix, iy, fill=col, width=2, capstyle=tk.ROUND)
        cv.create_oval(cx_ - 3, cy_ - 3, cx_ + 3, cy_ + 3, fill=col, outline="")

        # Speed arc ring
        if abs(self._speeds[i]) > 0.1:
            frac = min(1.0, abs(self._speeds[i]) / MAX_SPEED_MMS)
            cv.create_arc(cx_ - dr + 3, cy_ - dr + 3,
                          cx_ + dr - 3, cy_ + dr - 3,
                          start=90, extent=-frac * 270,
                          outline=YELLOW, style="arc", width=2)

    def update(self, idx1: int, disp: float, speed: float, target_mm: float = None):
        i = idx1 - 1
        self._disps[i]  = disp
        self._speeds[i] = speed
        self._disp_lbls[i].config(text=f"{disp:+.1f}mm")
        self._speed_lbls[i].config(text=f"{speed:.0f}mm/s")
        self._draw(i)

    def update_all(self, disps, speeds, targets=None):
        for i in range(6):
            self._disps[i]  = disps[i]
            self._speeds[i] = speeds[i]
            self._disp_lbls[i].config(text=f"{disps[i]:+.1f}mm")
            self._speed_lbls[i].config(text=f"{speeds[i]:.0f}mm/s")
        self._draw_all()


# ══════════════════════════════════════════════════════════════════════════════
# CONTINUUM SHAPE VISUALIZER  — 3-panel: SIDE VIEW + TOP VIEW + HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

class ContinuumVisualizer(tk.Frame):
    """
    Full 3-panel visualizer matching the original GUI richness but driven
    by the correct 3-segment PCC kinematics from Zhai et al. 2025.

    Panel layout
    ────────────
      LEFT   — SIDE VIEW  : correct per-segment coloured arcs with gradient
                            tube body, disk markers, tip crosshair & attitude
                            arrow, axis rulers, coordinate readout
      CENTRE — TOP VIEW   : forward/back projection + base-plate hexagon +
                            per-motor tendon lines from attachment point to
                            current tip (line width ∝ tension), zoom scroll
      RIGHT  — POSE HUD   : live x, y, ψ, θ₁, θ₂, θ₃ with colour coding

      BOTTOM — HEATMAP    : per-motor tension bar (blue=release → red=pull)
                            plus segment Δl and θ readout
    """

    BODY_RADII = [11, 8, 6]   # tube radius in px per segment (tapers toward tip)

    # Tendon termination: which segment end-disk each motor's tendon anchors to
    # M1/M2 → seg 0 end-disk, M3/M4 → seg 1 end-disk, M5/M6 → seg 2 end-disk (tip)
    TENDON_TERM = [0, 0, 1, 1, 2, 2]

    # Tendon attachment layout on the base cross-section.
    # Motors are in 3 antagonistic pairs; each pair straddles the backbone.
    # We give them angular positions around the cross-section centre so the
    # top-view tendon lines look realistic.
    TENDON_ANGLES_DEG = [150, 330,   # M1/M2  — Segment 1 pair  (±y axis side)
                          90, 270,   # M3/M4  — Segment 2 pair  (±x axis side)
                          210, 30]   # M5/M6  — Segment 3 pair  (diagonal)

    def __init__(self, parent, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._disps   = [0.0] * 6
        self._top_zoom = 1.0
        self._cv_side = None
        self._cv_top  = None
        self._cv_heat = None
        self._pose_widgets = {}
        self._build()
        self.after(120, self._redraw)

    # ── Layout construction ───────────────────────────────────────────────────

    def _build(self):
        # Header
        hf = tk.Frame(self, bg=BG)
        hf.pack(fill="x", pady=(4, 2))
        tk.Label(hf, text="CONTINUUM SHAPE", font=(FNT_H, 12, "bold"),
                 fg=BLUE, bg=BG).pack(side="left")
        tk.Label(hf, text="  ·  piecewise constant curvature  ·  3 segments  ·  mm",
                 font=(FNT, 10), fg=MUTED, bg=BG).pack(side="left")
        tk.Label(hf, text="scroll top-view to zoom",
                 font=(FNT, 9), fg=DIM, bg=BG).pack(side="right")

        # Three canvases row
        views = tk.Frame(self, bg=BG)
        views.pack(fill="both", expand=True)

        # ── SIDE VIEW ─────────────────────────────────────────────────────────
        sc = tk.Frame(views, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        sc.pack(side="left", fill="both", expand=True, padx=(0, 3))
        tk.Label(sc, text=" SIDE VIEW  (bending plane  x–y)",
                 font=(FNT, 9, "bold"), fg=BLUE, bg=PANEL).pack(fill="x")
        self._cv_side = tk.Canvas(sc, bg=CBKG, highlightthickness=0)
        self._cv_side.pack(fill="both", expand=True)
        self._cv_side.bind("<Configure>", lambda e: self._draw_side())

        # ── TOP VIEW ──────────────────────────────────────────────────────────
        tc = tk.Frame(views, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        tc.pack(side="left", fill="both", expand=True, padx=(3, 3))
        tk.Label(tc, text=" TOP VIEW  (base plate + tendon routing)",
                 font=(FNT, 9, "bold"), fg=ORANGE, bg=PANEL).pack(fill="x")
        self._cv_top = tk.Canvas(tc, bg=CBKG, highlightthickness=0)
        self._cv_top.pack(fill="both", expand=True)
        self._cv_top.bind("<Configure>", lambda e: self._draw_top())
        self._cv_top.bind("<MouseWheel>", self._on_top_scroll)

        # ── POSE HUD ──────────────────────────────────────────────────────────
        pc = tk.Frame(views, bg=CARD, highlightbackground=BORDER, highlightthickness=1,
                      width=158)
        pc.pack(side="left", fill="y", padx=(3, 0))
        pc.pack_propagate(False)
        tk.Label(pc, text=" END-EFFECTOR", font=(FNT, 9, "bold"),
                 fg=ORANGE, bg=PANEL).pack(fill="x")

        pose_entries = [
            ("tip_x", "X",   "mm",  ACCENT),
            ("tip_y", "Y",   "mm",  ACCENT),
            ("psi",   "ψ",   "deg", YELLOW),
            ("th1",   "θ₁",  "deg", SEG_COLORS[0]),
            ("th2",   "θ₂",  "deg", SEG_COLORS[1]),
            ("th3",   "θ₃",  "deg", SEG_COLORS[2]),
            ("len",   "L",   "mm",  MUTED),
        ]
        for key, label, unit, col in pose_entries:
            pf = tk.Frame(pc, bg=CARD)
            pf.pack(fill="x", padx=6, pady=3)
            tk.Label(pf, text=label, font=(FNT, 10), fg=MUTED,
                     bg=CARD, anchor="w", width=3).pack(side="left")
            lv = tk.Label(pf, text="---", font=(FNT, 13, "bold"),
                          fg=col, bg=CARD, anchor="e", width=7)
            lv.pack(side="right")
            tk.Label(pf, text=unit, font=(FNT, 9), fg=DIM, bg=CARD).pack(side="right")
            self._pose_widgets[key] = lv
            tk.Frame(pc, bg=BORDER, height=1).pack(fill="x", padx=4)

        # ── HEATMAP (per-motor, all 6 bars) ───────────────────────────────────
        hc = tk.Frame(self, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        hc.pack(fill="x", pady=(3, 0))
        tk.Label(hc, text=" MOTOR LOAD HEATMAP  (blue=release  ·  red=pull  ·  width ∝ |displacement|)",
                 font=(FNT, 9, "bold"), fg=MUTED, bg=PANEL).pack(fill="x")
        self._cv_heat = tk.Canvas(hc, height=52, bg=CBKG, highlightthickness=0)
        self._cv_heat.pack(fill="x")
        self._cv_heat.bind("<Configure>", lambda e: self._draw_heat())

    # ── Public API ────────────────────────────────────────────────────────────

    def set_displacements(self, disps):
        self._disps = list(disps)
        self._redraw()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _redraw(self):
        self._draw_side()
        self._draw_top()
        self._draw_heat()
        self._update_pose()

    def _on_top_scroll(self, event):
        self._top_zoom *= 1.12 if event.delta > 0 else (1 / 1.12)
        self._top_zoom  = max(0.4, min(4.0, self._top_zoom))
        self._draw_top()
        return "break"

    @staticmethod
    def _grid(cv, w, h, step=38):
        for x in range(0, w, step):
            cv.create_line(x, 0, x, h, fill="#e2e8f0", width=1)
        for y in range(0, h, step):
            cv.create_line(0, y, w, y, fill="#e2e8f0", width=1)

    # ── SIDE VIEW ─────────────────────────────────────────────────────────────

    def _draw_side(self):
        cv = self._cv_side
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 30 or h < 30:
            return
        cv.delete("all")
        self._grid(cv, w, h)

        kin    = compute_pcc_kinematics(self._disps)
        all_pts = kin["all_pts"]
        pt_seg  = kin["pt_seg"]
        N       = len(all_pts)

        # Scale: fit total rest-length in 72% of canvas height
        scl   = h * 0.72 / TOTAL_LENGTH_MM
        org_x = w // 2
        org_y = int(h * 0.93)

        def px(x, y):
            return org_x + x * scl, org_y - y * scl

        # Arc-normal helper: lateral offset in mm → canvas offset
        def arc_normal(k, off_mm):
            i0 = max(0, k - 1)
            i1 = min(N - 1, k + 1)
            dx = all_pts[i1][0] - all_pts[i0][0]
            dy = all_pts[i1][1] - all_pts[i0][1]
            ln = math.hypot(dx, dy) or 1.0
            nx, ny = -dy / ln, dx / ln
            return nx * off_mm * scl, ny * off_mm * scl

        # ── Ruler ─────────────────────────────────────────────────────────────
        ruler_x = 14
        cv.create_line(ruler_x, org_y, ruler_x, org_y - TOTAL_LENGTH_MM * scl,
                       fill=BORDER, width=1)
        for mm in range(0, int(TOTAL_LENGTH_MM) + 1, 30):
            ry = org_y - mm * scl
            cv.create_line(ruler_x - 4, ry, ruler_x + 4, ry, fill=MUTED, width=1)
            if mm % 90 == 0:
                cv.create_text(ruler_x - 6, ry, text=f"{mm}", fill=DIM,
                               font=(FNT, 8), anchor="e")
        cv.create_text(ruler_x, org_y - TOTAL_LENGTH_MM * scl - 8,
                       text="mm", fill=DIM, font=(FNT, 8), anchor="s")
        cv.create_line(org_x, org_y, org_x, 6, fill=BORDER, dash=(4, 6), width=1)

        # ── Base platform ─────────────────────────────────────────────────────
        bw = 26
        cv.create_rectangle(org_x - bw, org_y - 5, org_x + bw, org_y + 9,
                            fill="#e5e7eb", outline=TEXT, width=1)
        for k in range(-bw, bw, 8):
            cv.create_line(org_x + k, org_y + 9, org_x + k - 6, org_y + 16,
                           fill=MUTED, width=1)
        cv.create_text(org_x, org_y + 18, text="BASE",
                       fill=MUTED, font=(FNT, 9), anchor="n")

        # ── LAYER 1: outer shadow tube ─────────────────────────────────────────
        for k in range(1, N):
            x0c, y0c = px(*all_pts[k - 1])
            x1c, y1c = px(*all_pts[k])
            cv.create_line(x0c, y0c, x1c, y1c,
                           fill="#d1d5db", width=28,
                           capstyle=tk.ROUND)

        # ── LAYER 2: coloured flexible body, tapers toward tip ─────────────────
        # Tube width interpolates smoothly from BODY_RADII[0] at base to [2] at tip
        for k in range(1, N):
            s_idx   = pt_seg[k]
            seg_col = SEG_COLORS[s_idx]
            t_global = k / N   # 0=base 1=tip
            br = int(self.BODY_RADII[0] + t_global *
                     (self.BODY_RADII[2] - self.BODY_RADII[0]))
            x0c, y0c = px(*all_pts[k - 1])
            x1c, y1c = px(*all_pts[k])
            cv.create_line(x0c, y0c, x1c, y1c,
                           fill=seg_col, width=max(4, br * 2),
                           capstyle=tk.ROUND)

        # ── LAYER 3: spacer disks — uniform along whole arc ────────────────────
        # The disks are identical throughout; segment boundary disks are slightly
        # highlighted but drawn the same way — same pin joint, no hard pivot.
        TOTAL_DISKS = 24
        anchor_positions = set()
        for ep in kin["end_pts"][:-1]:   # only internal end-disks, not tip
            # find closest point index
            best = min(range(N), key=lambda k: math.hypot(
                all_pts[k][0] - ep[0], all_pts[k][1] - ep[1]))
            anchor_positions.add(best)

        disk_col_fill = "#f3f4f6"
        disk_col_edge = "#9ca3af"

        for di in range(TOTAL_DISKS + 1):
            t     = di / TOTAL_DISKS
            k     = min(int(t * (N - 1)), N - 1)
            s_idx = pt_seg[k]
            seg_col = SEG_COLORS[s_idx]
            t_global = k / N
            br = int(self.BODY_RADII[0] + t_global *
                     (self.BODY_RADII[2] - self.BODY_RADII[0]))
            hw = max(4, br + 5)   # disk half-width in canvas px

            cx_d, cy_d = px(*all_pts[k])

            # Tangent direction at this disk
            i0 = max(0, k - 1); i1 = min(N - 1, k + 1)
            dtx = all_pts[i1][0] - all_pts[i0][0]
            dty = all_pts[i1][1] - all_pts[i0][1]
            dn  = math.hypot(dtx, dty) or 1
            # perpendicular unit vector in canvas coords
            px_ = (-dty / dn) * hw
            py_ = ( dtx / dn) * hw
            # along-arc unit vector (for disk thickness)
            ax_ = (dtx / dn) * 2
            ay_ = (dty / dn) * 2

            is_anchor = k in anchor_positions
            fill_c = seg_col   if is_anchor else disk_col_fill
            edge_c = seg_col   if is_anchor else disk_col_edge
            lw_d   = 2         if is_anchor else 1

            pts4 = [
                cx_d + px_ - ax_, cy_d + py_ - ay_,
                cx_d + px_ + ax_, cy_d + py_ + ay_,
                cx_d - px_ + ax_, cy_d - py_ + ay_,
                cx_d - px_ - ax_, cy_d - py_ - ay_,
            ]
            cv.create_polygon(*pts4, fill=fill_c, outline=edge_c, width=lw_d)

            # Centre rotating pin dot
            pr = 2 if is_anchor else 1
            cv.create_oval(cx_d - pr, cy_d - pr, cx_d + pr, cy_d + pr,
                           fill=seg_col if is_anchor else "#6b7280", outline="")

        # ── LAYER 4: tendons running INSIDE the body ───────────────────────────
        # Physical offsets: outermost (M1/M2) → middle (M3/M4) → inner (M5/M6)
        # Expressed as mm lateral offset from centreline.
        # Each tendon runs from base to its anchor; inside the tube width.
        tendon_offsets_mm = [
            TENDON_OFFSETS_MM[0],  -TENDON_OFFSETS_MM[0],   # M1/M2
            TENDON_OFFSETS_MM[1],  -TENDON_OFFSETS_MM[1],   # M3/M4
            TENDON_OFFSETS_MM[2],  -TENDON_OFFSETS_MM[2],   # M5/M6
        ]
        tendon_terminations = self.TENDON_TERM

        for ti in range(6):
            col      = MOTOR_COLORS[ti]
            seg_stop = tendon_terminations[ti]
            off_mm   = tendon_offsets_mm[ti]
            d_val    = self._disps[ti]
            is_pull  = d_val > 0.5
            lw_t     = 2 if is_pull else 1
            dash_t   = () if is_pull else (5, 3)

            # Find stop index: last point in zone seg_stop
            stop_k = N - 1
            for k in range(N - 1, -1, -1):
                if pt_seg[k] == seg_stop:
                    stop_k = k
                    break

            prev_cx, prev_cy = None, None
            for k in range(stop_k + 1):
                dnx, dny = arc_normal(k, off_mm)
                cx_t = px(*all_pts[k])[0] + dnx
                cy_t = px(*all_pts[k])[1] + dny
                if prev_cx is not None:
                    cv.create_line(prev_cx, prev_cy, cx_t, cy_t,
                                   fill=col, width=lw_t, dash=dash_t,
                                   capstyle=tk.ROUND)
                prev_cx, prev_cy = cx_t, cy_t

            # Anchor dot at termination
            if prev_cx is not None:
                cv.create_oval(prev_cx - 3, prev_cy - 3,
                               prev_cx + 3, prev_cy + 3,
                               fill=col, outline=TEXT, width=1)

        # ── LAYER 5: end-disk labels ───────────────────────────────────────────
        for s_idx, ep in enumerate(kin["end_pts"]):
            seg_col = SEG_COLORS[s_idx]
            epx, epy = px(*ep)
            t_g = (s_idx + 1) / 3
            br  = int(self.BODY_RADII[0] + t_g *
                      (self.BODY_RADII[2] - self.BODY_RADII[0]))
            cv.create_text(epx + br + 10, epy,
                           text=f"End Disk {s_idx+1}  ·  T{s_idx*2+1}/T{s_idx*2+2} anchor",
                           fill=seg_col, font=(FNT, 9), anchor="w")

        # ── LAYER 6: curvature annotation per zone ────────────────────────────
        zone_starts = [0, int(round(SEG_LENGTHS_MM[0] / TOTAL_LENGTH_MM * (N-1))),
                          int(round((SEG_LENGTHS_MM[0]+SEG_LENGTHS_MM[1]) / TOTAL_LENGTH_MM * (N-1)))]
        for s_idx in range(3):
            th  = kin["thetas"][s_idx]
            if abs(math.degrees(th)) < 1.5:
                continue
            k0  = zone_starts[s_idx]
            spx, spy = px(*all_pts[k0])
            arc_r = 16
            cv.create_arc(spx - arc_r, spy - arc_r, spx + arc_r, spy + arc_r,
                          start=80, extent=-math.degrees(th) * 0.5,
                          style="arc", outline=SEG_COLORS[s_idx], width=1)
            cv.create_text(spx - arc_r - 4, spy,
                           text=f"θ{s_idx+1}={math.degrees(th):.0f}°",
                           fill=SEG_COLORS[s_idx], font=(FNT, 8), anchor="e")

        # ── LAYER 7: end-effector ─────────────────────────────────────────────
        tip_px, tip_py = px(kin["tip_x"], kin["tip_y"])
        ch = 18
        cv.create_line(tip_px - ch, tip_py, tip_px + ch, tip_py,
                       fill=ORANGE, dash=(3, 3), width=1)
        cv.create_line(tip_px, tip_py - ch, tip_px, tip_py + ch,
                       fill=ORANGE, dash=(3, 3), width=1)
        cv.create_oval(tip_px - 9, tip_py - 9, tip_px + 9, tip_py + 9,
                       fill=ORANGE, outline=TEXT, width=2)
        cv.create_text(tip_px + 14, tip_py - 1,
                       text="EE", fill=ORANGE, font=(FNT, 10, "bold"), anchor="w")

        psi = kin["psi"]
        al  = 34
        ax_ = tip_px + al * math.sin(psi)
        ay_ = tip_py - al * math.cos(psi)
        cv.create_line(tip_px, tip_py, ax_, ay_,
                       fill=YELLOW, width=2, arrow=tk.LAST, arrowshape=(8, 10, 3))
        cv.create_text(ax_ + 5, ay_,
                       text=f"ψ={math.degrees(psi):.1f}°",
                       fill=YELLOW, font=(FNT, 9), anchor="w")

        # Legend
        legend_y = h - 18
        for ti in range(6):
            lx  = org_x - 96 + ti * 33
            col = MOTOR_COLORS[ti]
            cv.create_line(lx, legend_y, lx + 22, legend_y, fill=col, width=2)
            cv.create_text(lx + 11, legend_y - 7, text=f"M{ti+1}",
                           fill=col, font=(FNT, 8), anchor="s")

        cv.create_text(w // 2, h - 5,
                       text=f"EE  x={kin['tip_x']:+.1f}mm  y={kin['tip_y']:.1f}mm  "
                            f"ψ={math.degrees(psi):+.1f}°",
                       fill=DIM, font=(FNT, 9), anchor="s")


    # ── TOP VIEW ──────────────────────────────────────────────────────────────

    def _draw_top(self):
        cv = self._cv_top
        w, h = cv.winfo_width(), cv.winfo_height()
        if w < 30 or h < 30:
            return
        cv.delete("all")
        self._grid(cv, w, h)

        kin     = compute_pcc_kinematics(self._disps)
        all_pts = kin["all_pts"]
        pt_seg  = kin["pt_seg"]
        N       = len(all_pts)

        scl  = min(w, h) * 0.58 * self._top_zoom
        cx_  = w // 2
        cy_  = int(h * 0.82)

        def px(x, y):
            return cx_ + x * scl / TOTAL_LENGTH_MM, cy_ - y * scl / TOTAL_LENGTH_MM

        # ── Base plate ────────────────────────────────────────────────────────
        plate_r = max(20, int(0.12 * min(w, h) * self._top_zoom))
        hex_pts = []
        for k in range(6):
            a = math.radians(30 + k * 60)
            hex_pts += [cx_ + plate_r * math.cos(a), cy_ + plate_r * math.sin(a)]
        cv.create_polygon(*hex_pts, fill="#e5e7eb", outline=MUTED, width=1)
        cv.create_line(cx_ - plate_r, cy_, cx_ + plate_r, cy_,
                       fill=BORDER, dash=(2, 4), width=1)
        cv.create_line(cx_, cy_ - plate_r, cx_, cy_ + plate_r,
                       fill=BORDER, dash=(2, 4), width=1)
        cv.create_text(cx_, cy_ + plate_r + 10, text="BASE PLATE",
                       fill=MUTED, font=(FNT, 8), anchor="n")

        # ── Continuous body arc (shadow + coloured, no segment gaps) ──────────
        for k in range(1, N):
            x0, y0 = px(*all_pts[k - 1])
            x1, y1 = px(*all_pts[k])
            cv.create_line(x0, y0, x1, y1, fill="#d1d5db",
                           width=12, capstyle=tk.ROUND)

        for k in range(1, N):
            s_idx   = pt_seg[k]
            t_g     = k / N
            br      = int(self.BODY_RADII[0] + t_g *
                          (self.BODY_RADII[2] - self.BODY_RADII[0]))
            x0, y0  = px(*all_pts[k - 1])
            x1, y1  = px(*all_pts[k])
            cv.create_line(x0, y0, x1, y1,
                           fill=SEG_COLORS[s_idx], width=max(3, br),
                           capstyle=tk.ROUND)

        # ── End-disk markers (tendon anchors — same joint, just labelled) ─────
        seg_end_px = []
        for ep in kin["end_pts"]:
            seg_end_px.append(px(*ep))

        for s_idx, (epx_, epy_) in enumerate(seg_end_px):
            seg_col = SEG_COLORS[s_idx]
            dr_ = 6
            cv.create_oval(epx_ - dr_, epy_ - dr_, epx_ + dr_, epy_ + dr_,
                           fill=CARD, outline=seg_col, width=2)
            cv.create_text(epx_, epy_ - dr_ - 5, text=f"D{s_idx+1}",
                           fill=seg_col, font=(FNT, 8), anchor="s")

        # ── Tendons: from base plate → through body → anchor at end-disk ──────
        tendon_r = max(12, int(plate_r * 0.55))

        for i in range(6):
            ta       = math.radians(self.TENDON_ANGLES_DEG[i])
            tx       = cx_ + tendon_r * math.cos(ta)
            ty       = cy_ + tendon_r * math.sin(ta)
            col      = MOTOR_COLORS[i]
            seg_stop = MOTOR_TO_SEG[i]
            d        = max(0.0, self._disps[i])
            lw       = 1 + int(d / MAX_DISP_MM * 4)

            # Tendon passes through body to its anchor — draw segment by segment
            # with faded colour in pass-through zones, full colour in own zone
            for si in range(seg_stop + 1):
                term_px_, term_py_ = seg_end_px[si]
                src_x, src_y = (tx, ty) if si == 0 else seg_end_px[si - 1]
                alpha    = 1.0 if si == seg_stop else 0.3
                rc = int(int(col[1:3], 16) * alpha)
                gc = int(int(col[3:5], 16) * alpha)
                bc = int(int(col[5:7], 16) * alpha)
                fade_col = f"#{max(rc,0):02x}{max(gc,0):02x}{max(bc,0):02x}"
                dash_    = () if (si == seg_stop and d > 1) else (4, 3)
                cv.create_line(src_x, src_y, term_px_, term_py_,
                               fill=fade_col,
                               width=lw if si == seg_stop else 1,
                               dash=dash_)

            # Base attachment dot
            cv.create_oval(tx - 5, ty - 5, tx + 5, ty + 5,
                           fill=col, outline="")
            # Anchor dot on end-disk
            term_px_, term_py_ = seg_end_px[seg_stop]
            cv.create_oval(term_px_ - 4, term_py_ - 4,
                           term_px_ + 4, term_py_ + 4,
                           fill=col, outline=TEXT, width=1)
            # Label
            lx_ = cx_ + (tendon_r + 13) * math.cos(ta)
            ly_ = cy_ + (tendon_r + 13) * math.sin(ta)
            cv.create_text(lx_, ly_, text=f"M{i+1}",
                           fill=col, font=(FNT, 8, "bold"))

        # ── Tip ───────────────────────────────────────────────────────────────
        tip_px, tip_py = px(kin["tip_x"], kin["tip_y"])
        cv.create_oval(tip_px - 9, tip_py - 9, tip_px + 9, tip_py + 9,
                       fill=ORANGE, outline=TEXT, width=2)
        cv.create_text(tip_px + 13, tip_py,
                       text="EE", fill=ORANGE, font=(FNT, 10, "bold"), anchor="w")
        cv.create_line(tip_px - 14, tip_py, tip_px + 14, tip_py,
                       fill=ORANGE, dash=(3, 3))
        cv.create_line(tip_px, tip_py - 14, tip_px, tip_py + 14,
                       fill=ORANGE, dash=(3, 3))

        psi = kin["psi"]
        al  = 28
        cv.create_line(tip_px, tip_py,
                       tip_px + al * math.sin(psi),
                       tip_py - al * math.cos(psi),
                       fill=YELLOW, width=2, arrow=tk.LAST)

        cv.create_text(w // 2, h - 5,
                       text=f"tip  x={kin['tip_x']:+.1f}  y={kin['tip_y']:.1f}mm  "
                            f"zoom×{self._top_zoom:.1f}",
                       fill=DIM, font=(FNT, 9), anchor="s")

    # ── HEATMAP  (per-motor, 6 bars) ─────────────────────────────────────────

    def _draw_heat(self):
        cv = self._cv_heat
        w  = cv.winfo_width()
        if w < 30:
            return
        cv.delete("all")
        h  = 52
        bw = w // 6

        for i in range(6):
            x0  = i * bw
            x1  = (i + 1) * bw
            d   = self._disps[i]
            col = MOTOR_COLORS[i]

            # Normalise 0-centre: −MAX … 0 … +MAX → 0 … 0.5 … 1
            norm = (d + MAX_DISP_MM) / (2 * MAX_DISP_MM)
            norm = max(0.0, min(1.0, norm))

            # Background heat colour: blue (release) → dark (zero) → red (pull)
            if norm > 0.5:
                t2 = (norm - 0.5) * 2
                rr = int(0x18 + t2 * (0xe8 - 0x18))
                gg = int(0x38 * (1 - t2))
                bb = int(0x38 * (1 - t2))
            else:
                t2 = (0.5 - norm) * 2
                rr = int(0x18 * (1 - t2))
                gg = int(0x38 * (1 - t2))
                bb = int(0x18 + t2 * (0xd8 - 0x18))
            heat_col = f"#{rr:02x}{gg:02x}{bb:02x}"

            cv.create_rectangle(x0, 0, x1, h, fill=heat_col, outline=BORDER)

            # Segment indicator stripe at top
            seg = MOTOR_TO_SEG[i]
            cv.create_rectangle(x0, 0, x1, 3, fill=SEG_COLORS[seg], outline="")

            # Motor label
            cv.create_text((x0 + x1) // 2, 11,
                           text=f"M{i+1}", fill=col, font=(FNT, 9, "bold"))

            # Displacement value
            cv.create_text((x0 + x1) // 2, 27,
                           text=f"{d:+.1f}mm", fill=TEXT, font=(FNT, 11, "bold"))

            # Role label
            role = "PULL" if MOTOR_IS_POS[i] else "REL"
            cv.create_text((x0 + x1) // 2, 43,
                           text=role, fill=DIM, font=(FNT, 8))

    # ── POSE HUD update ───────────────────────────────────────────────────────

    def _update_pose(self):
        kin = compute_pcc_kinematics(self._disps)
        self._pose_widgets["tip_x"].config(text=f"{kin['tip_x']:+.2f}")
        self._pose_widgets["tip_y"].config(text=f"{kin['tip_y']:.2f}")
        self._pose_widgets["psi"].config(text=f"{math.degrees(kin['psi']):+.1f}")
        for j, key in enumerate(["th1", "th2", "th3"]):
            self._pose_widgets[key].config(
                text=f"{math.degrees(kin['thetas'][j]):+.1f}")
        self._pose_widgets["len"].config(text=f"{TOTAL_LENGTH_MM:.0f}")


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
        self._live_lbls = []
        self._build()

    def _build(self):
        # Top toolbar
        tb = tk.Frame(self, bg=PANEL)
        tb.pack(fill="x")
        tk.Label(tb, text="MANUAL JOG", font=(FNT_H, 12, "bold"),
                 fg=ACCENT, bg=PANEL).pack(side="left", padx=14, pady=10)

        tk.Label(tb, text="STEP:", font=(FNT, 11), fg=MUTED, bg=PANEL
                 ).pack(side="left", padx=(14, 4))
        for s in ["0.1", "0.5", "1", "2", "5"]:
            tk.Radiobutton(tb, text=f"{s}mm", variable=self._step_var, value=s,
                           font=(FNT, 11, "bold"), bg=PANEL, fg=TEXT,
                           selectcolor=INPUT, activebackground=PANEL,
                           indicatoron=False, relief="flat", cursor="hand2",
                           padx=7, pady=3).pack(side="left", padx=2)

        for txt, c, fg2, cmd in [
            ("⚠ E-STOP", RED, TEXT, self._estop),
            ("STOP ALL", YELLOW, BG, self._stop_all),
            ("ZERO ALL", MUTED, TEXT, self._zero_all),
        ]:
            tk.Button(tb, text=txt, font=(FNT, 11, "bold"),
                      bg=c, fg=fg2, relief="flat", cursor="hand2",
                      padx=8, pady=4, command=cmd).pack(side="right", padx=3)

        # Column headers
        hdr = tk.Frame(self, bg=PANEL)
        hdr.pack(fill="x")
        specs = [("", 4), ("Motor / Role", 16), ("Displacement bar", 24),
                 ("Live mm", 9), ("  Jog", 10), ("Step", 9),
                 ("Target mm", 10), ("Speed mm/s", 11)]
        for t, w in specs:
            tk.Label(hdr, text=t, font=(FNT, 10, "bold"), fg=MUTED, bg=PANEL,
                     width=w, anchor="w").pack(side="left", padx=2, pady=3)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # Per-segment sub-headers + motor rows
        for seg in range(3):
            sc_ = SEG_COLORS[seg]
            sf  = tk.Frame(self, bg=BG)
            sf.pack(fill="x")
            tk.Frame(sf, bg=sc_, width=4).pack(side="left", fill="y")
            tk.Label(sf, text=f"  ─── {SEG_NAMES[seg]} (θ{seg+1})  ───",
                     font=(FNT, 10, "bold"), fg=sc_, bg=BG).pack(
                     side="left", padx=6, pady=2)
            for j in range(2):
                i = seg * 2 + j
                self._build_row(i)

    def _build_row(self, i: int):
        col = MOTOR_COLORS[i]
        bg  = BG if i % 2 == 0 else CARD
        row = tk.Frame(self, bg=bg)
        row.pack(fill="x", pady=1)

        # Colour stripe
        tk.Frame(row, bg=col, width=4).pack(side="left", fill="y")
        tk.Label(row, text=f"M{i+1}\n{'PUL' if MOTOR_IS_POS[i] else 'REL'}",
                 font=(FNT, 9, "bold"), fg=col, bg=bg, width=5).pack(side="left", padx=3)
        tk.Label(row, text=MOTOR_NAMES[i], font=(FNT, 9), fg=MUTED,
                 bg=bg, width=16).pack(side="left", padx=2)

        # Displacement bar
        c = tk.Canvas(row, width=155, height=26, bg=bg, highlightthickness=0)
        c.pack(side="left", padx=4)
        c.create_rectangle(2, 9, 153, 19, fill=INPUT, outline="")
        c.create_line(77, 5, 77, 23, fill=BORDER, width=1)
        bar_fill = c.create_rectangle(77, 9, 77, 19, fill=col, outline="")
        bar_lbl  = c.create_text(77, 14, text="0.0", fill=TEXT, font=(FNT, 9))
        self._bar_canvases.append(c)
        self._bar_fills.append(bar_fill)
        self._bar_lbls.append(bar_lbl)

        # Live label
        ll = tk.Label(row, text="  0.0", font=(FNT, 13, "bold"),
                      fg=col, bg=bg, width=7)
        ll.pack(side="left", padx=4)
        self._live_lbls.append(ll)

        # Jog buttons (hold to jog)
        b_neg = tk.Button(row, text=" − ", font=(FNT, 14, "bold"),
                          bg=INPUT, fg=col, relief="flat", cursor="hand2",
                          padx=8, pady=2)
        b_pos = tk.Button(row, text=" + ", font=(FNT, 14, "bold"),
                          bg=INPUT, fg=col, relief="flat", cursor="hand2",
                          padx=8, pady=2)
        b_neg.pack(side="left", padx=1)
        b_pos.pack(side="left", padx=1)
        b_neg.bind("<ButtonPress-1>",   lambda e, idx=i: self._jog_start(idx, -1))
        b_neg.bind("<ButtonRelease-1>", lambda e, idx=i: self._jog_stop(idx))
        b_pos.bind("<ButtonPress-1>",   lambda e, idx=i: self._jog_start(idx, +1))
        b_pos.bind("<ButtonRelease-1>", lambda e, idx=i: self._jog_stop(idx))

        # Step buttons
        tk.Button(row, text="−S", font=(FNT, 10, "bold"),
                  bg=INPUT, fg=DIM, relief="flat", cursor="hand2",
                  padx=5, pady=1,
                  command=lambda idx=i: self._step_jog(idx, -1)
                  ).pack(side="left", padx=1)
        tk.Button(row, text="+S", font=(FNT, 10, "bold"),
                  bg=INPUT, fg=DIM, relief="flat", cursor="hand2",
                  padx=5, pady=1,
                  command=lambda idx=i: self._step_jog(idx, +1)
                  ).pack(side="left", padx=1)

        # Target entry
        e = tk.Entry(row, textvariable=self._disp_vars[i], width=8,
                     font=(FNT, 12), bg=INPUT, fg=TEXT,
                     insertbackground=TEXT, relief="flat",
                     highlightbackground=BORDER, highlightthickness=1)
        e.pack(side="left", padx=5)
        e.bind("<Return>",   lambda ev, idx=i: self._send(idx))
        e.bind("<KP_Enter>", lambda ev, idx=i: self._send(idx))
        tk.Button(row, text="SEND", font=(FNT, 10, "bold"),
                  bg=ACCENT, fg=BG, relief="flat", cursor="hand2",
                  padx=6, pady=2,
                  command=lambda idx=i: self._send(idx)).pack(side="left", padx=2)

        # Speed entry
        tk.Entry(row, textvariable=self._speed_vars[i], width=7,
                 font=(FNT, 12), bg=INPUT, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 highlightbackground=BORDER, highlightthickness=1
                 ).pack(side="left", padx=4)

    # ── Actions ───────────────────────────────────────────────────────────────

    def _get_speed(self, i):
        try:   return max(0.1, min(float(self._speed_vars[i].get()), MAX_SPEED_MMS))
        except: return 20.0

    def _get_step(self):
        try:   return float(self._step_var.get())
        except: return 5.0

    def _send(self, i):
        try:
            d = float(self._disp_vars[i].get())
            d = max(-MAX_DISP_MM, min(MAX_DISP_MM, d))
        except ValueError:
            self._log(f"M{i+1}: invalid displacement value", "warn"); return
        self._apply_disp(i, d, self._get_speed(i))
        self._log(f"[JOG] M{i+1} → {d:+.2f}mm @ {self._get_speed(i):.0f}mm/s")

    def _jog_start(self, i, sign):
        self._jog_stop(i)
        def tick():
            disps = self._get_disps()
            nd = max(-MAX_DISP_MM, min(MAX_DISP_MM,
                                       disps[i] + sign * self._get_step() * 0.25))
            self._apply_disp(i, nd, self._get_speed(i))
            self._disp_vars[i].set(f"{nd:.2f}")
            self._jog_jobs[i] = self.after(100, tick)
        tick()

    def _jog_stop(self, i):
        if i in self._jog_jobs:
            self.after_cancel(self._jog_jobs[i])
            del self._jog_jobs[i]

    def _step_jog(self, i, sign):
        disps = self._get_disps()
        nd = max(-MAX_DISP_MM, min(MAX_DISP_MM, disps[i] + sign * self._get_step()))
        self._apply_disp(i, nd, self._get_speed(i))
        self._disp_vars[i].set(f"{nd:.2f}")

    def _estop(self):
        for i in range(6): self._jog_stop(i)
        self._log("[E-STOP] Emergency stop — all jog cancelled", "err")
        self._zero_all()

    def _stop_all(self):
        for i in range(6): self._jog_stop(i)
        self._log("[STOP] All jog stopped", "warn")

    def _zero_all(self):
        for i in range(6):
            self._apply_disp(i, 0.0, 20.0)
            self._disp_vars[i].set("0.0")
        self._log("[ZERO] All motors zeroed")

    def update_disp(self, idx1: int, disp: float):
        i = idx1 - 1
        # Bar
        c  = self._bar_canvases[i]
        bw = 155
        ratio = (disp + MAX_DISP_MM) / (2 * MAX_DISP_MM)
        ratio = max(0.0, min(1.0, ratio))
        fill_x = 2 + int((bw - 4) * ratio)
        centre = 77
        col = MOTOR_COLORS[i]
        c.coords(self._bar_fills[i],
                 min(centre, fill_x), 9, max(centre, fill_x), 19)
        c.itemconfig(self._bar_fills[i],
                     fill=col if disp >= 0 else "#d1d5db")
        c.itemconfig(self._bar_lbls[i], text=f"{disp:+.1f}")
        self._live_lbls[i].config(text=f"{disp:+.1f}")
        self._disp_vars[i].set(f"{disp:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# BUILT-IN CSV EDITOR + SEQUENCER
# ══════════════════════════════════════════════════════════════════════════════

class CsvStatePanel(tk.Frame):
    """
    Full CSV state sequencer with built-in editor.
    Three sub-panels:
      ① File toolbar (browse / load / save / generate sample)
      ② Built-in CSV editor (editable text widget with syntax colour)
      ③ State preview table + playback controls + timing readout
    """

    def __init__(self, parent, get_robot, apply_state_fn, log_fn, **kw):
        super().__init__(parent, bg=BG, **kw)
        self._get_robot    = get_robot
        self._apply_state  = apply_state_fn
        self._log          = log_fn

        self._states: List[TendonState] = []
        self._filepath  = tk.StringVar()
        self._loop_var  = tk.BooleanVar(value=False)
        self._delay_var = tk.StringVar(value="0")
        self._wait_var  = tk.BooleanVar(value=True)

        self._running      = False
        self._paused       = False
        self._abort_flag   = threading.Event()
        self._current_step = -1
        self._row_frames   = []
        self._seq_start = self._step_start = self._delay_start = 0.0
        self._phase     = "idle"
        self._sw_job    = None
        self._timing_log: List[Tuple] = []

        self._build()

    def _build(self):
        # ── ① File toolbar ────────────────────────────────────────────────────
        fc = tk.Frame(self, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        fc.pack(fill="x", pady=(0, 4))

        tk.Label(fc, text="CSV STATE FILE", font=(FNT_H, 12, "bold"),
                 fg=ACCENT, bg=CARD).pack(side="left", padx=10, pady=8)

        tk.Entry(fc, textvariable=self._filepath,
                 font=(FNT, 12), bg=INPUT, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 highlightbackground=BORDER, highlightthickness=1,
                 width=36).pack(side="left", padx=(0, 6))

        for txt, col, fg2, cmd in [
            ("BROWSE",       BLUE,   BG, self._browse),
            ("LOAD FILE",    ACCENT, BG, self._load_file),
            ("SAMPLE DATA",  INPUT,  TEXT, self._load_sample),
        ]:
            tk.Button(fc, text=txt, font=(FNT, 11, "bold"),
                      bg=col, fg=fg2, relief="flat", cursor="hand2",
                      padx=8, pady=4, command=cmd).pack(side="left", padx=3)



        # ── ③ Playback controls ───────────────────────────────────────────────
        pc = tk.Frame(self, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        pc.pack(fill="x", pady=(0, 4))

        ctrl = tk.Frame(pc, bg=CARD)
        ctrl.pack(fill="x", padx=8, pady=8)

        self.btn_run   = tk.Button(ctrl, text="▶  RUN",
                                   font=(FNT, 13, "bold"), bg=GREEN, fg=BG,
                                   relief="flat", cursor="hand2",
                                   padx=16, pady=6, command=self._run,
                                   state="disabled")
        self.btn_run.pack(side="left", padx=(0, 4))

        self.btn_step  = tk.Button(ctrl, text="⏭  STEP",
                                   font=(FNT, 13, "bold"), bg=BLUE, fg=BG,
                                   relief="flat", cursor="hand2",
                                   padx=14, pady=6, command=self._step_once,
                                   state="disabled")
        self.btn_step.pack(side="left", padx=(0, 4))

        self.btn_pause = tk.Button(ctrl, text="⏸  PAUSE",
                                   font=(FNT, 13, "bold"), bg=YELLOW, fg=BG,
                                   relief="flat", cursor="hand2",
                                   padx=14, pady=6, command=self._pause,
                                   state="disabled")
        self.btn_pause.pack(side="left", padx=(0, 4))

        self.btn_abort = tk.Button(ctrl, text="■  ABORT",
                                   font=(FNT, 13, "bold"), bg=RED, fg=TEXT,
                                   relief="flat", cursor="hand2",
                                   padx=14, pady=6, command=self._abort,
                                   state="disabled")
        self.btn_abort.pack(side="left", padx=(0, 14))

        # Options inline
        tk.Label(ctrl, text="DELAY OVR:", font=(FNT, 11), fg=MUTED, bg=CARD
                 ).pack(side="left")
        tk.Entry(ctrl, textvariable=self._delay_var, width=5,
                 font=(FNT, 12), bg=INPUT, fg=TEXT,
                 insertbackground=TEXT, relief="flat",
                 highlightbackground=BORDER, highlightthickness=1
                 ).pack(side="left", padx=(3, 10))
        tk.Checkbutton(ctrl, text="Wait motors", variable=self._wait_var,
                       font=(FNT, 11), bg=CARD, fg=TEXT,
                       selectcolor=INPUT, activebackground=CARD
                       ).pack(side="left", padx=(0, 10))
        tk.Checkbutton(ctrl, text="Loop", variable=self._loop_var,
                       font=(FNT, 11), bg=CARD, fg=TEXT,
                       selectcolor=INPUT, activebackground=CARD
                       ).pack(side="left")

        self.lbl_status = tk.Label(ctrl, text="No states loaded.",
                                   font=(FNT, 11), fg=MUTED, bg=CARD)
        self.lbl_status.pack(side="right", padx=10)

        # Progress bar
        style = ttk.Style()
        style.configure("Seq.Horizontal.TProgressbar",
                        troughcolor=INPUT, background=GREEN)
        self.prog_var = tk.DoubleVar(value=0)
        ttk.Progressbar(pc, variable=self.prog_var, maximum=100,
                        style="Seq.Horizontal.TProgressbar"
                        ).pack(fill="x", padx=8, pady=(0, 6))

        # ── Timing + state table side by side ─────────────────────────────────
        bot = tk.Frame(self, bg=BG)
        bot.pack(fill="x", pady=(0, 4))

        # Timing panel
        tw = tk.Frame(bot, bg=CARD, highlightbackground=BORDER, highlightthickness=1,
                      width=200)
        tw.pack(side="left", fill="y", padx=(0, 4))
        tw.pack_propagate(False)
        tk.Label(tw, text=" ⏱ TIMING", font=(FNT, 9, "bold"),
                 fg=MUTED, bg=PANEL).pack(fill="x", pady=2)

        clocks = tk.Frame(tw, bg=CARD)
        clocks.pack(padx=8, pady=4)

        def _clock_col(parent, label, init, col, w=9):
            f = tk.Frame(parent, bg=CARD)
            f.pack(side="left", padx=4)
            tk.Label(f, text=label, font=(FNT, 8), fg=MUTED, bg=CARD).pack()
            lbl = tk.Label(f, text=init, font=(FNT, w, "bold"), fg=col, bg=CARD)
            lbl.pack()
            return lbl

        self.lbl_total  = _clock_col(clocks, "TOTAL",  "00:00.0", ACCENT, 14)
        self.lbl_motion = _clock_col(clocks, "MOTION", "0.00s",   BLUE,   9)
        self.lbl_delay  = _clock_col(clocks, "DELAY",  "0.00s",   YELLOW, 9)

        ph = tk.Frame(tw, bg=CARD)
        ph.pack(pady=2)
        tk.Label(ph, text="PHASE", font=(FNT, 8), fg=MUTED, bg=CARD).pack()
        self.lbl_phase = tk.Label(ph, text="IDLE",
                                  font=(FNT, 11, "bold"), fg=MUTED, bg=CARD)
        self.lbl_phase.pack()

        tk.Label(tw, text="STEP HISTORY", font=(FNT, 8), fg=MUTED, bg=CARD
                 ).pack(anchor="w", padx=8)
        self._timing_lbls = []
        for _ in range(5):
            l = tk.Label(tw, text="", font=(FNT, 9), fg=DIM, bg=CARD, anchor="w")
            l.pack(fill="x", padx=8)
            self._timing_lbls.append(l)

        # State table
        tbl_outer = tk.Frame(bot, bg=CARD,
                             highlightbackground=BORDER, highlightthickness=1)
        tbl_outer.pack(side="left", fill="both", expand=True)
        tk.Label(tbl_outer, text=" STATE TABLE  (click row to jump)",
                 font=(FNT, 9, "bold"), fg=MUTED, bg=PANEL).pack(fill="x")

        tbl_cv = tk.Canvas(tbl_outer, bg=CARD, highlightthickness=0, height=200)
        tbl_sb = ttk.Scrollbar(tbl_outer, orient="vertical", command=tbl_cv.yview)
        self._tbl_inner = tk.Frame(tbl_cv, bg=CARD)
        self._tbl_inner.bind("<Configure>",
            lambda e: tbl_cv.configure(scrollregion=tbl_cv.bbox("all")))
        tbl_cv.create_window((0, 0), window=self._tbl_inner, anchor="nw")
        tbl_cv.configure(yscrollcommand=tbl_sb.set)
        tbl_sb.pack(side="right", fill="y")
        tbl_cv.pack(side="left", fill="both", expand=True)
        self._build_table_header()

    # ── File actions ──────────────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askopenfilename(
            title="Select State CSV",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if path:
            self._filepath.set(path)

    def _load_file(self):
        path = self._filepath.get().strip()
        if not path:
            messagebox.showwarning("No File", "Enter or browse a CSV path.")
            return
        try:
            states, err = parse_states_csv(path)
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            return

        if not states:
            messagebox.showerror("Parse Error", err or f"No states found in {path}.")
            return

        self._states = states
        self._populate_table()
        self._enable_run()
        self.lbl_status.config(
            text=f"{len(states)} states loaded", fg=GREEN)
        self._log(f"[CSV] Loaded file: {os.path.basename(path)}")
        if err:
            self._log(f"[CSV] Warnings: {err}", "warn")

    def _load_sample(self):
        states = make_sample_states()
        self._states = states
        self._populate_table()
        self._enable_run()
        self.lbl_status.config(text=f"Sample data loaded ({len(states)} states)", fg=GREEN)
        self._log("[CSV] Sample states loaded")

    # ── State table ───────────────────────────────────────────────────────────

    def _build_table_header(self):
        hdr = tk.Frame(self._tbl_inner, bg=PANEL)
        hdr.pack(fill="x")
        cols = [("#", 3), ("State Name", 14), ("Delay", 6)]
        for i in range(6):
            cols.append((MOTOR_SHORT[i], 13))
        for t, w in cols:
            tk.Label(hdr, text=t, font=(FNT, 9, "bold"), fg=MUTED, bg=PANEL,
                     width=w, anchor="w").pack(side="left", padx=2, pady=2)

    def _populate_table(self):
        for w in self._tbl_inner.winfo_children():
            if isinstance(w, tk.Frame) and w != self._tbl_inner.winfo_children()[0]:
                w.destroy()
        self._row_frames.clear()
        for idx, state in enumerate(self._states):
            bg = CARD if idx % 2 == 0 else INPUT
            row = tk.Frame(self._tbl_inner, bg=bg, cursor="hand2")
            row.pack(fill="x")
            row.bind("<Button-1>", lambda e, i=idx: self._jump_to(i))
            self._row_frames.append(row)

            tk.Label(row, text=str(idx + 1), font=(FNT, 9), fg=MUTED, bg=bg,
                     width=3, anchor="w").pack(side="left", padx=2, pady=2)
            tk.Label(row, text=state.name[:14], font=(FNT, 9), fg=TEXT, bg=bg,
                     width=14, anchor="w").pack(side="left", padx=2)
            tk.Label(row, text=f"{state.delay_s:.1f}s", font=(FNT, 9),
                     fg=YELLOW, bg=bg, width=6, anchor="w").pack(side="left", padx=2)
            for i in range(6):
                d, s = state.motors[i]
                col  = MOTOR_COLORS[i] if abs(d) > 0.5 else MUTED
                tk.Label(row, text=f"{d:+.0f}@{s:.0f}", font=(FNT, 9),
                         fg=col, bg=bg, width=13, anchor="w").pack(side="left", padx=2)

    def _jump_to(self, idx: int):
        """Immediately apply a single state from the table."""
        if not self._states or self._running:
            return
        state = self._states[idx]
        self._apply_state(state)
        self._highlight_row(idx)
        self._log(f"[JUMP] Applied state {idx+1}: {state.name}")

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

    def _enable_run(self):
        self.btn_run.config(state="normal")
        self.btn_step.config(state="normal")

    # ── Playback ──────────────────────────────────────────────────────────────

    def _run(self):
        if self._running or not self._states: return
        self._running = True
        self._paused  = False
        self._abort_flag.clear()
        self._timing_log.clear()
        self.btn_run.config(state="disabled")
        self.btn_step.config(state="disabled")
        self.btn_pause.config(state="normal", text="⏸  PAUSE")
        self.btn_abort.config(state="normal")
        self.prog_var.set(0)
        self._seq_start = time.time()
        self._sw_job    = self.after(100, self._tick_sw)
        threading.Thread(target=self._worker, daemon=True).start()

    def _step_once(self):
        """Execute the next state once (manual step mode)."""
        if self._running or not self._states: return
        idx = (self._current_step + 1) % len(self._states)
        self._current_step = idx
        state = self._states[idx]
        self._apply_state(state)
        self._highlight_row(idx)
        self.lbl_status.config(
            text=f"Step {idx+1}/{len(self._states)}: {state.name}", fg=BLUE)
        self._log(f"[STEP] {state.summary()}")

    def _pause(self):
        if self._paused:
            self._paused = False
            self.btn_pause.config(text="⏸  PAUSE", bg=YELLOW)
            self._set_phase("motion")
        else:
            self._paused = True
            self.btn_pause.config(text="▶  RESUME", bg=GREEN)
            self._set_phase("paused")

    def _abort(self):
        self._abort_flag.set()
        self._paused = False

    def _worker(self):
        total = len(self._states)
        run_n = 0
        while True:
            run_n += 1
            self.after(0, lambda: self._log(f"[SEQ] ── Run #{run_n} start ──"))
            for idx, state in enumerate(self._states):
                if self._abort_flag.is_set():
                    self.after(0, self._done, "Aborted.")
                    return
                while self._paused:
                    if self._abort_flag.is_set():
                        self.after(0, self._done, "Aborted."); return
                    time.sleep(0.05)

                self._current_step = idx
                self.after(0, self._highlight_row, idx)
                self.after(0, self.prog_var.set, idx / total * 100)
                self.after(0, self.lbl_status.config,
                           {"text": f"State {idx+1}/{total}: {state.name}", "fg": GREEN})
                self.after(0, lambda s=state.summary(): self._log(f"[SEQ] {s}"))

                t_step = time.time()
                self._set_phase("motion")
                self.after(0, self._apply_state, state)

                if self._wait_var.get():
                    max_d = max(abs(m[0]) for m in state.motors)
                    max_s = max(m[1] for m in state.motors)
                    est   = (max_d / max_s + 0.3) if max_s > 0 else 1.0
                    t_end = time.time() + est
                    while time.time() < t_end:
                        if self._abort_flag.is_set(): break
                        while self._paused and not self._abort_flag.is_set():
                            time.sleep(0.05)
                        time.sleep(0.04)

                mot_dur = time.time() - t_step
                self.after(0, self.lbl_motion.config,
                           {"text": f"{mot_dur:.2f}s", "fg": GREEN})

                try:    gui_delay = float(self._delay_var.get())
                except: gui_delay = 0.0
                delay = gui_delay if gui_delay > 0 else state.delay_s

                if delay > 0:
                    self._set_phase("delay")
                    t0 = time.time()
                    while time.time() - t0 < delay:
                        if self._abort_flag.is_set():
                            self.after(0, self._done, "Aborted."); return
                        while self._paused and not self._abort_flag.is_set():
                            time.sleep(0.05)
                        time.sleep(0.04)
                    act_delay = time.time() - t0
                else:
                    act_delay = 0.0

                self.after(0, self.lbl_delay.config,
                           {"text": f"{act_delay:.2f}s", "fg": YELLOW})
                self._timing_log.append((state.name, mot_dur, act_delay))
                self.after(0, self._refresh_timing)
                self.after(0, lambda n=state.name, md=mot_dur, ad=act_delay:
                           self._log(f"[SEQ] ✓ {n}: motion {md:.2f}s  delay {ad:.2f}s"))

            self.after(0, self.prog_var.set, 100)
            self.after(0, self._clear_highlight)
            self._set_phase("idle")
            if not self._loop_var.get() or self._abort_flag.is_set():
                break
            time.sleep(0.1)
        self.after(0, self._done, f"Complete ({run_n} run{'s' if run_n > 1 else ''}).")

    def _done(self, msg):
        self._running = False
        self._paused  = False
        if self._sw_job:
            self.after_cancel(self._sw_job)
            self._sw_job = None
        self._set_phase("idle")
        self.btn_run.config(state="normal")
        self.btn_step.config(state="normal")
        self.btn_pause.config(state="disabled", text="⏸  PAUSE", bg=YELLOW)
        self.btn_abort.config(state="disabled")
        self.lbl_status.config(text=msg, fg=DIM)
        self._log(f"[SEQ] {msg}")

    def _set_phase(self, phase):
        self._phase = phase
        cols = {"idle": MUTED, "motion": GREEN, "delay": YELLOW,
                "paused": ORANGE, "": MUTED}
        self.after(0, self.lbl_phase.config,
                   {"text": phase.upper(), "fg": cols.get(phase, MUTED)})

    def _tick_sw(self):
        if not self._running: return
        elapsed = time.time() - self._seq_start
        m = int(elapsed // 60)
        s = elapsed % 60
        self.lbl_total.config(text=f"{m:02d}:{s:04.1f}")
        self._sw_job = self.after(100, self._tick_sw)

    def _refresh_timing(self):
        last5 = self._timing_log[-5:]
        for j, l in enumerate(self._timing_lbls):
            if j < len(last5):
                n, md, ad = last5[-(j + 1)]
                l.config(text=f"  {n[:10]:<10}  {md:.1f}s / {ad:.1f}s")
            else:
                l.config(text="")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION WINDOW
# ══════════════════════════════════════════════════════════════════════════════

class ContinuumGUI(tk.Tk):

    def __init__(self, simulated=True, port="COM3", bustype="slcan"):
        super().__init__()
        self.title("Tendon-Driven Continuum Manipulator  ·  Control GUI  v3.0")
        self.configure(bg=BG)
        self.minsize(1440, 820)

        # Try to set icon / DPI awareness
        try:
            self.state("zoomed")
        except Exception:
            self.geometry("1600x900")

        self._sim   = simulated
        self._port  = port
        self._bus_t = bustype
        self._disps  = [0.0] * 6
        self._speeds = [0.0] * 6
        self._connected = False
        self._mon_job   = None
        self.robot = None
        self.bus   = None

        self._build()
        self._start_monitor()
        self._poll_log()

        if simulated:
            self.after(400, self._connect)

    def _build(self):
        # ── Top bar ───────────────────────────────────────────────────────────
        top = tk.Frame(self, bg=PANEL)
        top.pack(fill="x")

        tk.Label(top,
                 text="⬡  CONTINUUM MANIPULATOR  ·  3-SEG TENDON DRIVE  ·  PCC CONTROL  v3",
                 font=(FNT_H, 14, "bold"), fg=ACCENT, bg=PANEL
                 ).pack(side="left", padx=16, pady=10)

        self._lbl_c = tk.Label(top, text="● DISCONNECTED",
                               font=(FNT, 12, "bold"), fg=RED, bg=PANEL)
        self._lbl_c.pack(side="right", padx=12)
        mode = "SIM" if self._sim else f"HW · {self._port}"
        tk.Label(top, text=mode, font=(FNT, 11), fg=MUTED, bg=PANEL
                 ).pack(side="right", padx=4)
        self._mkbtn(top, "DISCONNECT", PANEL, RED,   self._disconnect).pack(side="right", padx=3)
        self._mkbtn(top, "CONNECT",    ACCENT, BG,   self._connect   ).pack(side="right", padx=3)

        # ── Capstan radii config bar ───────────────────────────────────────────
        rb = tk.Frame(self, bg=CARD, highlightbackground=BORDER, highlightthickness=1)
        rb.pack(fill="x", padx=8, pady=(4, 2))
        tk.Label(rb, text="CAPSTAN RADII (mm):", font=(FNT, 11, "bold"),
                 fg=ACCENT, bg=CARD).pack(side="left", padx=(12, 6), pady=6)
        self._radii_vars = []
        for i in range(6):
            col = MOTOR_COLORS[i]
            tk.Label(rb, text=f"M{i+1}:", font=(FNT, 11, "bold"),
                     fg=col, bg=CARD).pack(side="left", padx=(6, 2))
            rv = tk.StringVar(value=str(CAPSTAN_RADII_MM[i]))
            self._radii_vars.append(rv)
            tk.Entry(rb, textvariable=rv, width=5,
                     font=(FNT, 12), bg=INPUT, fg=TEXT,
                     insertbackground=TEXT, relief="flat",
                     highlightbackground=BORDER, highlightthickness=1
                     ).pack(side="left", padx=(0, 3))
        tk.Button(rb, text="APPLY", font=(FNT, 11, "bold"),
                  bg=ACCENT, fg=BG, relief="flat", cursor="hand2",
                  padx=10, pady=4, command=self._apply_radii
                  ).pack(side="left", padx=(10, 0))

        # ── Body ──────────────────────────────────────────────────────────────
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        # Left: tab panels
        left_outer = tk.Frame(body, bg=BG, width=820)
        left_outer.pack(side="left", fill="y", padx=(8, 4), pady=6)
        left_outer.pack_propagate(False)

        # Tab buttons
        tbar = tk.Frame(left_outer, bg=BG)
        tbar.pack(fill="x", pady=(0, 6))

        self._tab_manual_btn = tk.Button(
            tbar, text="  🕹  MANUAL JOG  ",
            font=(FNT, 12, "bold"), bg=ACCENT, fg=BG,
            relief="flat", cursor="hand2", padx=10, pady=6,
            command=self._show_manual)
        self._tab_manual_btn.pack(side="left", padx=(0, 4))

        self._tab_csv_btn = tk.Button(
            tbar, text="  📋  CSV SEQUENCER  ",
            font=(FNT, 12, "bold"), bg=PANEL, fg=MUTED,
            relief="flat", cursor="hand2", padx=10, pady=6,
            command=self._show_csv)
        self._tab_csv_btn.pack(side="left")

        self._mkbtn(tbar, "  🏠  HOME ALL", GREEN, BG,
                    self._home_all).pack(side="right", padx=(6, 0))

        # Scrollable left area
        hscroll = ttk.Scrollbar(left_outer, orient="horizontal")
        hscroll.pack(side="bottom", fill="x")
        sf = tk.Frame(left_outer, bg=BG)
        sf.pack(fill="both", expand=True)
        lc = tk.Canvas(sf, bg=BG, highlightthickness=0)
        vs = ttk.Scrollbar(sf, orient="vertical", command=lc.yview)
        lc.configure(yscrollcommand=vs.set, xscrollcommand=hscroll.set)
        hscroll.configure(command=lc.xview)
        vs.pack(side="right", fill="y")
        lc.pack(side="left", fill="both", expand=True)
        lc.bind_all("<MouseWheel>",
                    lambda e: lc.yview_scroll(int(-1 * (e.delta / 120)), "units")
                    if vs.get() != (0.0, 1.0) else None)
        self._lc = lc

        # Manual panel
        self._manual_frame = tk.Frame(lc, bg=BG)
        self._manual_win   = lc.create_window((0, 0), window=self._manual_frame, anchor="nw")
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
        self._csv_win   = lc.create_window((0, 0), window=self._csv_frame, anchor="nw")
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

        # Right: visualizations
        right = tk.Frame(body, bg=BG)
        right.pack(side="right", fill="both", expand=True, padx=(4, 8), pady=6)

        self._capstan_panel = CapstanPanel(right)
        self._capstan_panel.pack(fill="x")

        tk.Frame(right, bg=BORDER, height=1).pack(fill="x", pady=4)

        self._cont_viz = ContinuumVisualizer(right)
        self._cont_viz.pack(fill="both", expand=True)

        # ── Log ───────────────────────────────────────────────────────────────
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        lh = tk.Frame(self, bg=PANEL)
        lh.pack(fill="x")
        tk.Label(lh, text="SYSTEM LOG", font=(FNT, 10), fg=MUTED,
                 bg=PANEL, pady=3).pack(side="left", padx=12)
        tk.Button(lh, text="CLEAR", font=(FNT, 10), bg=PANEL, fg=MUTED,
                  relief="flat", cursor="hand2",
                  command=self._clear_log).pack(side="right", padx=12)
        self._lbox = scrolledtext.ScrolledText(
            self, height=5, font=(FNT, 11),
            bg=PANEL, fg=MUTED, relief="flat",
            state="disabled", wrap="word")
        self._lbox.pack(fill="x")
        for t, c in [("ok", GREEN), ("warn", YELLOW), ("err", RED), ("info", DIM)]:
            self._lbox.tag_config(t, foreground=c)

    # ── Tab switching ─────────────────────────────────────────────────────────

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

    # ── Motion ────────────────────────────────────────────────────────────────

    def _apply_single_disp(self, motor_idx: int, disp_mm: float, speed_mms: float):
        self._disps[motor_idx]  = disp_mm
        self._speeds[motor_idx] = speed_mms
        self._capstan_panel.update(motor_idx + 1, disp_mm, speed_mms)
        self._manual_panel.update_disp(motor_idx + 1, disp_mm)
        self._cont_viz.set_displacements(self._disps)

        if self.robot and HAS_ROBOT:
            try:
                from robot_controller import DEFAULT_JOINT_CONFIG
                cfg = DEFAULT_JOINT_CONFIG[motor_idx]
                threading.Thread(
                    target=self.robot.motors[cfg.motor_id].set_position,
                    kwargs={"position_deg": mm_to_deg(disp_mm, motor_idx),
                            "max_speed_dps": mms_to_dps(speed_mms, motor_idx),
                            "wait": False},
                    daemon=True).start()
            except Exception as e:
                self._log(f"HW error M{motor_idx+1}: {e}", "err")

    def _apply_state(self, state: TendonState):
        targets = [m[0] for m in state.motors]
        speeds  = [m[1] for m in state.motors]
        for i in range(6):
            self._disps[i]  = targets[i]
            self._speeds[i] = speeds[i]
            if self.robot and HAS_ROBOT:
                try:
                    from robot_controller import DEFAULT_JOINT_CONFIG
                    cfg = DEFAULT_JOINT_CONFIG[i]
                    threading.Thread(
                        target=self.robot.motors[cfg.motor_id].set_position,
                        kwargs={"position_deg": mm_to_deg(targets[i], i),
                                "max_speed_dps": mms_to_dps(speeds[i], i),
                                "wait": False},
                        daemon=True).start()
                    time.sleep(0.001)
                except Exception as e:
                    self._log(f"HW error M{i+1}: {e}", "err")
        self._capstan_panel.update_all(targets, speeds)
        for i in range(6):
            self._manual_panel.update_disp(i + 1, targets[i])
        self._cont_viz.set_displacements(targets)
        self._log(f"[STATE] {state.name}")

    def _home_all(self):
        for i in range(6):
            self._apply_single_disp(i, 0.0, 20.0)
        self._log("[HOME] All motors → 0.0 mm", "ok")

    def _apply_radii(self):
        for i in range(6):
            try:
                r = float(self._radii_vars[i].get())
                if r <= 0: raise ValueError
                CAPSTAN_RADII_MM[i] = r
            except ValueError:
                messagebox.showwarning("Invalid Radius", f"M{i+1}: must be positive number.")
                return
        self._log("[CONFIG] Radii → " + "  ".join(
            f"M{i+1}={CAPSTAN_RADII_MM[i]:.1f}" for i in range(6)), "ok")

    # ── Connection ────────────────────────────────────────────────────────────

    def _connect(self):
        if not HAS_ROBOT:
            self._lbl_c.config(text="● SIMULATION", fg=YELLOW)
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
        self._log("Connected — 6 capstan motors active.", "ok")

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
            self.after_cancel(self._mon_job)
            self._mon_job = None

    def _monitor(self):
        if self.robot and self._connected and HAS_ROBOT:
            try:
                from robot_controller import DEFAULT_JOINT_CONFIG
                fb = self.robot.get_all_feedback()
                for i, cfg in enumerate(DEFAULT_JOINT_CONFIG):
                    f = fb.get(cfg.motor_id)
                    if f:
                        d = deg_to_mm(f.position_deg, i)
                        v = deg_to_mm(abs(f.velocity_dps), i)
                        self._disps[i]  = d
                        self._speeds[i] = v
                        self._capstan_panel.update(i + 1, d, v)
                        self._manual_panel.update_disp(i + 1, d)
                self._cont_viz.set_displacements(self._disps)
            except Exception:
                pass
        else:
            self._capstan_panel.update_all(self._disps, self._speeds)
            for i in range(6):
                self._manual_panel.update_disp(i + 1, self._disps[i])
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
        self._lbox.delete("1.0", "end")
        self._lbox.config(state="disabled")

    @staticmethod
    def _mkbtn(parent, text, bg, fg, cmd):
        return tk.Button(parent, text=text, font=(FNT, 12, "bold"),
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
    ap = argparse.ArgumentParser(description="Tendon-Driven Continuum Manipulator GUI v2")
    ap.add_argument("--real",    action="store_true",
                    help="Connect to real hardware")
    ap.add_argument("--port",    default="COM3",
                    help="CAN adapter port (e.g. COM3, /dev/ttyACM0)")
    ap.add_argument("--bustype", default="slcan",
                    choices=["slcan", "pcan", "kvaser", "socketcan"],
                    help="python-can bus type")
    args = ap.parse_args()
    ContinuumGUI(simulated=not args.real,
                 port=args.port,
                 bustype=args.bustype).mainloop()