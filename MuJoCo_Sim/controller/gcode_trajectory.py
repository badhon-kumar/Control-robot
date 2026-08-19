"""
gcode_trajectory.py
═══════════════════════════════════════════════════════════════════════════════
Extended G-code path parser for the tendon-driven continuum manipulator.

`continuum_ellipse.py` imports this module if it is present and falls back to a
smaller built-in parser otherwise. The GUI reports which parser is active in the
G-code status line, so a failed import here is visible rather than silent.

Interface consumed by the GUI (must stay stable):
    GCodeProgram.points        list of GCodePoint(t_s, pose)
    GCodeProgram.duration_s    float
    GCodeProgram.pose_at(t)    -> np.array([x_m, y_m, psi_rad])
    GCodeProgram.xy_points_mm  -> [(x_mm, y_mm), ...]
    load_gcode_file(path)      -> GCodeProgram

Coordinate meaning
──────────────────
    X, Y    end-effector position in the bending plane   (mm by default)
    A, PSI  end-effector attitude psi                     (degrees)
    F       feed rate                                     (mm/min by default)

Supported commands
──────────────────
    G0  / G00       rapid linear move
    G1  / G01       linear move
    G2  / G02       clockwise arc          — I/J centre offsets, or R radius
    G3  / G03       counter-clockwise arc  — I/J centre offsets, or R radius
    G4  P..         dwell (seconds; values > 100 are read as milliseconds)
    G5              cubic Bezier      I J P Q X Y   (I/J omitted = tangent continue)
    G5.1            quadratic Bezier  I J X Y       (I/J omitted = tangent continue)
    G6              ellipse           X Y I J [P Q R]      (non-standard extension)
    G17             XY plane select (accepted, no effect — the robot is planar)
    G20 / G21       inches / millimetres
    G61             exact stop: disable corner blending
    G64 [P..]       blended path: round corners by up to P mm
    G90 / G91       absolute / incremental coordinates
    M-codes         ignored

Letter conflicts — the rules are fixed and unambiguous
──────────────────────────────────────────────────────
    A, PSI  always attitude, in every command.
    P, Q    dwell time in G4; Bezier control offsets in G5;
            start/end angle in G6. P is attitude ONLY when no G-word on the
            line claims it (kept for backwards compatibility; prefer A).
    I, J    arc centre offsets in G2/G3; Bezier control offsets in G5/G5.1;
            ellipse semi-radii in G6.
    R       arc radius in G2/G3; ellipse rotation in G6; blend tolerance is P
            in G64.

Curves are flattened to polylines at CHORD_TOL_MM so that the GUI's linear
interpolation between stored points cannot degrade the shape.
"""

from __future__ import annotations

import math
import re
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ── Tuning constants ─────────────────────────────────────────────────────────

CHORD_TOL_MM = 0.02          # max deviation of a flattened chord from the true curve
DEFAULT_FEED_MM_MIN = 600.0
RAPID_FEED_MM_MIN = 1200.0
DEFAULT_BLEND_TOL_MM = 0.10  # G64 with no P
MIN_BLEND_ANGLE_DEG = 2.0    # direction changes below this are not worth blending
ARC_RADIUS_ABS_TOL_MM = 0.01 # I/J endpoint consistency check
ARC_RADIUS_REL_TOL = 0.001
PSI_TO_MM = 20.0             # attitude-only moves: rad -> pseudo-distance for timing
MAX_SUBDIV_DEPTH = 12
MAX_CURVE_POINTS = 4000
HOME_POSE_MM = (270.0, 0.0, 0.0)   # straight arm: FK(u=0) = (0.270 m, 0, 0)

_WORD_RE = re.compile(r"([A-Za-z]+)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")
_PAREN_RE = re.compile(r"\([^)]*\)")


# ── Public data types ────────────────────────────────────────────────────────

@dataclass
class GCodePoint:
    t_s: float
    pose: np.ndarray             # [x_m, y_m, psi_rad]


@dataclass
class GCodeProgram:
    points: List[GCodePoint]
    source: str = ""
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        self._times = [p.t_s for p in self.points]

    @property
    def duration_s(self) -> float:
        return float(self.points[-1].t_s) if self.points else 0.0

    def pose_at(self, t_s: float) -> np.ndarray:
        pts = self.points
        if not pts:
            raise ValueError("G-code program has no trajectory points")
        if t_s <= pts[0].t_s:
            return pts[0].pose.copy()
        if t_s >= pts[-1].t_s:
            return pts[-1].pose.copy()

        i = bisect_right(self._times, t_s) - 1
        i = max(0, min(i, len(pts) - 2))
        a, b = pts[i], pts[i + 1]
        span = b.t_s - a.t_s
        if span <= 1e-12:
            return b.pose.copy()

        s = (t_s - a.t_s) / span
        pose = a.pose + s * (b.pose - a.pose)
        pose[2] = a.pose[2] + s * _wrap(b.pose[2] - a.pose[2])
        return pose

    def xy_points_mm(self) -> List[Tuple[float, float]]:
        out: List[Tuple[float, float]] = []
        for point in self.points:
            xy = (float(point.pose[0] * 1000.0), float(point.pose[1] * 1000.0))
            if not out or abs(out[-1][0] - xy[0]) > 1e-9 or abs(out[-1][1] - xy[1]) > 1e-9:
                out.append(xy)
        return out


# ── Internal vertex (millimetres, radians) ───────────────────────────────────

@dataclass
class _V:
    x: float
    y: float
    psi: float
    feed: float          # mm/min used to REACH this vertex
    corner: bool         # True at a junction between two moves
    dwell: float = 0.0   # seconds to hold after arriving
    blend: float = 0.0   # G64 tolerance in force when this vertex was emitted


class GCodeError(ValueError):
    """Raised with a line number for anything the parser cannot honour."""


# ── Small helpers ────────────────────────────────────────────────────────────

def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _strip_comment(line: str) -> str:
    return _PAREN_RE.sub("", line).split(";", 1)[0].strip()


def _parse_words(line: str) -> Tuple[Dict[str, float], List[float]]:
    """
    Split a line into non-G words plus an ordered list of G values.

    G codes are kept separate because several may share a line — `G17 G21 G90`
    is ordinary CAM output, and collapsing them into one dict key would silently
    discard all but the last.
    """
    words: Dict[str, float] = {}
    g_codes: List[float] = []
    for key, value in _WORD_RE.findall(line):
        key = key.upper()
        if key == "G":
            g_codes.append(float(value))
        else:
            words[key] = float(value)
    return words, g_codes


def _seg_count_for_radius(radius_mm: float, sweep_rad: float) -> int:
    """Chord count so the sagitta of every chord stays under CHORD_TOL_MM."""
    sweep = abs(sweep_rad)
    if radius_mm <= CHORD_TOL_MM or sweep <= 1e-12:
        return 8
    ratio = 1.0 - CHORD_TOL_MM / radius_mm
    ratio = max(-1.0, min(1.0, ratio))
    step = 2.0 * math.acos(ratio)
    if step <= 1e-9:
        return MAX_CURVE_POINTS
    return int(max(4, min(MAX_CURVE_POINTS, math.ceil(sweep / step))))


def _point_line_distance(p, a, b) -> float:
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    den = math.hypot(dx, dy)
    if den < 1e-12:
        return math.hypot(px - ax, py - ay)
    return abs(dy * (px - ax) - dx * (py - ay)) / den


# ── Curve flattening ─────────────────────────────────────────────────────────

def _flatten_cubic(p0, p1, p2, p3, depth: int = 0) -> List[Tuple[float, float]]:
    """Adaptive de Casteljau subdivision. Returns points AFTER p0, including p3."""
    if depth >= MAX_SUBDIV_DEPTH:
        return [p3]
    d = max(_point_line_distance(p1, p0, p3), _point_line_distance(p2, p0, p3))
    if d <= CHORD_TOL_MM:
        return [p3]

    def mid(a, b):
        return ((a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5)

    p01, p12, p23 = mid(p0, p1), mid(p1, p2), mid(p2, p3)
    p012, p123 = mid(p01, p12), mid(p12, p23)
    mp = mid(p012, p123)
    return (_flatten_cubic(p0, p01, p012, mp, depth + 1) +
            _flatten_cubic(mp, p123, p23, p3, depth + 1))


def _flatten_quadratic(p0, p1, p2) -> List[Tuple[float, float]]:
    """Quadratic Bezier raised to a cubic, then flattened."""
    c1 = (p0[0] + 2.0 / 3.0 * (p1[0] - p0[0]), p0[1] + 2.0 / 3.0 * (p1[1] - p0[1]))
    c2 = (p2[0] + 2.0 / 3.0 * (p1[0] - p2[0]), p2[1] + 2.0 / 3.0 * (p1[1] - p2[1]))
    return _flatten_cubic(p0, c1, c2, p2)


def _flatten_arc(cx, cy, radius, ang0, sweep) -> List[Tuple[float, float]]:
    n = _seg_count_for_radius(radius, sweep)
    return [(cx + radius * math.cos(ang0 + sweep * k / n),
             cy + radius * math.sin(ang0 + sweep * k / n))
            for k in range(1, n + 1)]


def _flatten_ellipse(cx, cy, ra, rb, rot, ang0, ang1) -> List[Tuple[float, float]]:
    """
    Sample uniformly in the ellipse parameter, with a step chosen so the chord
    sagitta stays under CHORD_TOL_MM everywhere.

    For (a·cos t, b·sin t) let s(t) = sqrt(a²sin²t + b²cos²t). A step dt spans a
    chord of s·dt over a local curvature radius of s³/(ab), so the sagitta is
    c²/(8R) = ab·dt²/(8s). That is worst where s is smallest, giving a bound of
    max(a,b)·dt²/8 and hence dt = sqrt(8·tol/max(a,b)).

    Sizing this off the tightest curvature radius instead — as one would for a
    circular arc — under-samples by the axis ratio, because arc length per unit
    parameter is not constant on an ellipse.
    """
    sweep = ang1 - ang0
    big = max(abs(ra), abs(rb))
    if big <= 1e-9:
        n = 8
    else:
        dt = math.sqrt(8.0 * CHORD_TOL_MM / big)
        n = int(max(4, min(MAX_CURVE_POINTS, math.ceil(abs(sweep) / dt))))
    cr, sr = math.cos(rot), math.sin(rot)
    out = []
    for k in range(1, n + 1):
        a = ang0 + sweep * k / n
        ex, ey = ra * math.cos(a), rb * math.sin(a)
        out.append((cx + ex * cr - ey * sr, cy + ex * sr + ey * cr))
    return out


# ── Arc geometry ─────────────────────────────────────────────────────────────

def _arc_from_ij(start, target, words, scale, clockwise, line_no) -> Tuple[float, float, float, float, float]:
    cx = start[0] + words["I"] * scale
    cy = start[1] + words["J"] * scale
    r0 = math.hypot(start[0] - cx, start[1] - cy)
    if r0 < 1e-9:
        raise GCodeError(f"line {line_no}: arc radius is zero")

    r1 = math.hypot(target[0] - cx, target[1] - cy)
    tol = max(ARC_RADIUS_ABS_TOL_MM, ARC_RADIUS_REL_TOL * r0)
    if abs(r1 - r0) > tol:
        raise GCodeError(
            f"line {line_no}: inconsistent arc — start radius {r0:.4f} mm but "
            f"end radius {r1:.4f} mm (tolerance {tol:.4f} mm). Check I/J, or use the R form."
        )

    ang0 = math.atan2(start[1] - cy, start[0] - cx)
    ang1 = math.atan2(target[1] - cy, target[0] - cx)
    sweep = _sweep_between(ang0, ang1, clockwise)

    # Coincident start/end means a full circle, not a zero move.
    if math.hypot(target[0] - start[0], target[1] - start[1]) < 1e-9:
        sweep = -2.0 * math.pi if clockwise else 2.0 * math.pi
    return cx, cy, r0, ang0, sweep


def _arc_from_r(start, target, radius, clockwise, line_no) -> Tuple[float, float, float, float, float]:
    """
    Radius form. Sign convention follows the NIST/LinuxCNC rule:
        R > 0 -> the minor arc (sweep <= 180 deg)
        R < 0 -> the major arc (sweep >  180 deg)
    """
    dx, dy = target[0] - start[0], target[1] - start[1]
    chord = math.hypot(dx, dy)
    if chord < 1e-9:
        raise GCodeError(
            f"line {line_no}: R-form arc needs distinct start and end points "
            f"(a full circle must use the I/J form)"
        )
    r = abs(radius)
    if chord > 2.0 * r + 1e-6:
        raise GCodeError(
            f"line {line_no}: radius {r:.4f} mm is too small to span a "
            f"{chord:.4f} mm chord (needs at least {chord / 2.0:.4f} mm)"
        )

    h = math.sqrt(max(0.0, r * r - (chord * 0.5) ** 2))
    mx, my = start[0] + dx * 0.5, start[1] + dy * 0.5
    nx, ny = -dy / chord, dx / chord          # unit normal, 90 deg CCW of travel

    # Minor arc: centre sits left of travel for CCW, right for CW.
    sign = 1.0 if not clockwise else -1.0
    if radius < 0.0:
        sign = -sign

    cx, cy = mx + sign * h * nx, my + sign * h * ny
    ang0 = math.atan2(start[1] - cy, start[0] - cx)
    ang1 = math.atan2(target[1] - cy, target[0] - cx)
    return cx, cy, r, ang0, _sweep_between(ang0, ang1, clockwise)


def _sweep_between(ang0: float, ang1: float, clockwise: bool) -> float:
    sweep = ang1 - ang0
    if clockwise:
        while sweep >= 0.0:
            sweep -= 2.0 * math.pi
    else:
        while sweep <= 0.0:
            sweep += 2.0 * math.pi
    return sweep


# ── Corner blending (G64) ────────────────────────────────────────────────────

def _cumulative_lengths(verts: Sequence[_V]) -> List[float]:
    s = [0.0]
    for i in range(1, len(verts)):
        s.append(s[-1] + math.hypot(verts[i].x - verts[i - 1].x,
                                    verts[i].y - verts[i - 1].y))
    return s


def _sample_at_s(verts: Sequence[_V], s_arr: Sequence[float], s_target: float) -> _V:
    i = bisect_right(s_arr, s_target) - 1
    i = max(0, min(i, len(verts) - 2))
    span = s_arr[i + 1] - s_arr[i]
    f = 0.0 if span <= 1e-12 else (s_target - s_arr[i]) / span
    a, b = verts[i], verts[i + 1]
    return _V(a.x + f * (b.x - a.x),
              a.y + f * (b.y - a.y),
              a.psi + f * _wrap(b.psi - a.psi),
              b.feed, False)


def _blend_corners(verts: List[_V]) -> List[_V]:
    """
    Replace sharp junctions with quadratic Bezier fillets, each deviating from
    its original corner by at most that corner's own G64 tolerance.

    For a quadratic Bezier with control point C and endpoints backed off by L
    along each leg, the peak deviation is L*sin(turn/2)/2, so L = 2*tol/sin(turn/2).
    """
    if len(verts) < 3 or all(v.blend <= 0.0 for v in verts):
        return verts

    s_arr = _cumulative_lengths(verts)
    min_angle = math.radians(MIN_BLEND_ANGLE_DEG)

    # Collect blendable corners with their back-off distance.
    corners: List[Tuple[int, float]] = []
    for i in range(1, len(verts) - 1):
        if not verts[i].corner:
            continue
        if verts[i].dwell > 0.0:
            continue        # a G4 stops the machine here — rounding it would be wrong
        tol_mm = verts[i].blend
        if tol_mm <= 0.0:
            continue        # blending switched off (G61) when this corner was emitted
        ax, ay = verts[i].x - verts[i - 1].x, verts[i].y - verts[i - 1].y
        bx, by = verts[i + 1].x - verts[i].x, verts[i + 1].y - verts[i].y
        la, lb = math.hypot(ax, ay), math.hypot(bx, by)
        if la < 1e-9 or lb < 1e-9:
            continue
        cosang = max(-1.0, min(1.0, (ax * bx + ay * by) / (la * lb)))
        turn = math.acos(cosang)
        if turn < min_angle or abs(math.pi - turn) < 1e-9:
            continue
        half = math.sin(turn * 0.5)
        if half < 1e-9:
            continue
        corners.append((i, 2.0 * tol_mm / half))

    if not corners:
        return verts

    # Clamp back-off so neighbouring fillets and the path ends never overlap.
    limits: List[float] = []
    for k, (idx, want) in enumerate(corners):
        prev_s = s_arr[corners[k - 1][0]] if k > 0 else s_arr[0]
        next_s = s_arr[corners[k + 1][0]] if k + 1 < len(corners) else s_arr[-1]
        room = min(s_arr[idx] - prev_s, next_s - s_arr[idx]) * 0.5
        limits.append(max(0.0, min(want, room)))

    out: List[_V] = []
    i = 0
    for (idx, _), L in zip(corners, limits):
        if L <= 1e-9:
            continue
        s_in, s_out = s_arr[idx] - L, s_arr[idx] + L
        while i < len(verts) and s_arr[i] < s_in - 1e-12:
            out.append(verts[i])
            i += 1

        p_in = _sample_at_s(verts, s_arr, s_in)
        p_out = _sample_at_s(verts, s_arr, s_out)
        c = verts[idx]
        feed = min(p_in.feed, p_out.feed, c.feed)

        fillet = _flatten_quadratic((p_in.x, p_in.y), (c.x, c.y), (p_out.x, p_out.y))
        out.append(_V(p_in.x, p_in.y, p_in.psi, feed, False))
        n = len(fillet)
        for k, (fx, fy) in enumerate(fillet, start=1):
            psi = p_in.psi + _wrap(p_out.psi - p_in.psi) * (k / n)
            out.append(_V(fx, fy, psi, feed, False))

        while i < len(verts) and s_arr[i] <= s_out + 1e-12:
            i += 1

    out.extend(verts[i:])
    return out


# ── Timing ───────────────────────────────────────────────────────────────────

def _assign_times(verts: Sequence[_V]) -> List[GCodePoint]:
    points: List[GCodePoint] = []
    t = 0.0
    first = verts[0]
    points.append(GCodePoint(0.0, np.array([first.x / 1000.0, first.y / 1000.0, first.psi])))
    if first.dwell > 0.0:
        t += first.dwell
        points.append(GCodePoint(t, points[-1].pose.copy()))

    for i in range(1, len(verts)):
        a, b = verts[i - 1], verts[i]
        dist = math.hypot(b.x - a.x, b.y - a.y)
        turn = abs(_wrap(b.psi - a.psi)) * PSI_TO_MM
        path = max(dist, turn, 1e-6)
        feed_mm_s = max(1e-6, b.feed / 60.0)
        t += max(path / feed_mm_s, 1e-6)
        points.append(GCodePoint(t, np.array([b.x / 1000.0, b.y / 1000.0, b.psi])))
        if b.dwell > 0.0:
            t += b.dwell
            points.append(GCodePoint(t, points[-1].pose.copy()))
    return points


# ── Parser ───────────────────────────────────────────────────────────────────

def load_gcode_file(path: str) -> GCodeProgram:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return parse_gcode(handle.read(), source=path)


def parse_gcode(text: str, source: str = "") -> GCodeProgram:
    scale = 1.0                      # unit word -> mm
    absolute = True
    feed = DEFAULT_FEED_MM_MIN
    motion = "G1"
    blend_tol = 0.0                  # G61 (exact stop) until G64 says otherwise
    warnings: List[str] = []

    pos = [HOME_POSE_MM[0], HOME_POSE_MM[1], HOME_POSE_MM[2]]
    verts: List[_V] = [_V(pos[0], pos[1], pos[2], feed, False)]
    started = False                  # has any real motion been emitted yet
    last_ctrl: Optional[Tuple[float, float]] = None   # for G5/G5.1 tangent continuation

    def emit_line(target, mv_feed):
        verts.append(_V(target[0], target[1], target[2], mv_feed, True, blend=blend_tol))

    def emit_curve(xy_pts, start_xy, psi0, psi1, mv_feed):
        """Attach psi along the flattened curve by cumulative chord length."""
        if not xy_pts:
            return
        lens, total, prev = [], 0.0, start_xy
        for p in xy_pts:
            total += math.hypot(p[0] - prev[0], p[1] - prev[1])
            lens.append(total)
            prev = p
        dpsi = _wrap(psi1 - psi0)
        last = len(xy_pts) - 1
        for k, (px, py) in enumerate(xy_pts):
            f = 1.0 if total <= 1e-12 else lens[k] / total
            verts.append(_V(px, py, psi0 + dpsi * f, mv_feed,
                            k == last, blend=blend_tol))

    def same(a: float, b: float) -> bool:
        return abs(a - b) < 1e-6

    for line_no, raw in enumerate(text.splitlines(), start=1):
        words, g_codes = _parse_words(_strip_comment(raw))
        if not words and not g_codes:
            continue
        if "M" in words and not g_codes and not any(
                k in words for k in ("X", "Y", "A", "PSI", "I", "J", "R")):
            continue                                        # bare M-code

        # Units first: a G64 or a coordinate on the same line must scale correctly.
        for g in g_codes:
            if same(g, 20):
                scale = 25.4
            elif same(g, 21):
                scale = 1.0

        dwell_s: Optional[float] = None
        for g in g_codes:
            if any(same(g, c) for c in (0, 1, 2, 3, 5, 5.1, 6)):
                motion = f"G{g:g}"
            elif same(g, 4):
                d = float(words.get("P", 0.0))
                if d > 100.0:
                    d /= 1000.0                              # milliseconds
                dwell_s = d
            elif same(g, 17) or same(g, 20) or same(g, 21):
                pass                                         # planar robot; units done above
            elif same(g, 18) or same(g, 19):
                raise GCodeError(
                    f"line {line_no}: G{g:g} selects a non-XY plane, but this "
                    f"manipulator bends only in the XY plane"
                )
            elif same(g, 61):
                blend_tol = 0.0
            elif same(g, 64):
                # Consume P so it cannot also be read as an attitude word below.
                blend_tol = float(words.pop("P")) * scale if "P" in words else DEFAULT_BLEND_TOL_MM
                if blend_tol < 0.0:
                    raise GCodeError(f"line {line_no}: G64 P must not be negative")
            elif same(g, 90):
                absolute = True
            elif same(g, 91):
                absolute = False
            else:
                raise GCodeError(f"line {line_no}: unsupported command G{g:g}")

        if dwell_s is not None:
            if dwell_s > 0.0:
                verts[-1].dwell += dwell_s
            continue                                         # G4 consumes the line

        if "F" in words:
            feed = max(1e-6, float(words["F"]) * scale)

        if not any(k in words for k in ("X", "Y", "A", "PSI", "P", "Q", "I", "J", "R")):
            continue

        # ── Resolve the commanded end pose ───────────────────────────────────
        target = list(pos)
        if "X" in words:
            v = words["X"] * scale
            target[0] = v if absolute else target[0] + v
        if "Y" in words:
            v = words["Y"] * scale
            target[1] = v if absolute else target[1] + v

        # A and PSI are always attitude. P is attitude only when the active
        # motion mode does not already use P for geometry.
        psi_key = None
        if "PSI" in words:
            psi_key = "PSI"
        elif "A" in words:
            psi_key = "A"
        elif "P" in words and motion not in ("G5", "G6"):
            psi_key = "P"
        if psi_key is not None:
            v = math.radians(float(words[psi_key]))
            target[2] = v if absolute else target[2] + v

        start_xy = (pos[0], pos[1])
        psi0 = pos[2]

        # A leading G0 positions the start of the path instead of tracing to it.
        if motion == "G0" and not started:
            pos = target
            verts[0] = _V(pos[0], pos[1], pos[2], feed, False)
            continue

        mv_feed = RAPID_FEED_MM_MIN if motion == "G0" else feed

        # ── Dispatch ─────────────────────────────────────────────────────────
        if motion in ("G2", "G3"):
            cw = (motion == "G2")
            if "I" in words or "J" in words:
                if "I" not in words or "J" not in words:
                    raise GCodeError(f"line {line_no}: arc needs both I and J (or use R)")
                cx, cy, r, a0, sweep = _arc_from_ij(start_xy, target, words, scale, cw, line_no)
            elif "R" in words:
                cx, cy, r, a0, sweep = _arc_from_r(
                    start_xy, target, words["R"] * scale, cw, line_no)
            else:
                raise GCodeError(f"line {line_no}: arc needs I/J centre offsets or an R radius")
            emit_curve(_flatten_arc(cx, cy, r, a0, sweep), start_xy, psi0, target[2], mv_feed)
            last_ctrl = None

        elif motion == "G5":
            if "X" not in words and "Y" not in words:
                raise GCodeError(f"line {line_no}: G5 needs an end point (X and/or Y)")
            if "I" in words or "J" in words:
                c1 = (start_xy[0] + words.get("I", 0.0) * scale,
                      start_xy[1] + words.get("J", 0.0) * scale)
            elif last_ctrl is not None:
                c1 = (2.0 * start_xy[0] - last_ctrl[0], 2.0 * start_xy[1] - last_ctrl[1])
            else:
                raise GCodeError(
                    f"line {line_no}: the first G5 of a run needs I and J "
                    f"(only a following G5 may omit them to continue tangentially)"
                )
            if "P" not in words or "Q" not in words:
                raise GCodeError(f"line {line_no}: G5 needs P and Q (offsets from the end point)")
            c2 = (target[0] + words["P"] * scale, target[1] + words["Q"] * scale)
            emit_curve(_flatten_cubic(start_xy, c1, c2, (target[0], target[1])),
                       start_xy, psi0, target[2], mv_feed)
            last_ctrl = c2

        elif motion == "G5.1":
            if "X" not in words and "Y" not in words:
                raise GCodeError(f"line {line_no}: G5.1 needs an end point (X and/or Y)")
            if "I" in words or "J" in words:
                c1 = (start_xy[0] + words.get("I", 0.0) * scale,
                      start_xy[1] + words.get("J", 0.0) * scale)
            elif last_ctrl is not None:
                c1 = (2.0 * start_xy[0] - last_ctrl[0], 2.0 * start_xy[1] - last_ctrl[1])
            else:
                raise GCodeError(f"line {line_no}: the first G5.1 of a run needs I and J")
            emit_curve(_flatten_quadratic(start_xy, c1, (target[0], target[1])),
                       start_xy, psi0, target[2], mv_feed)
            last_ctrl = c1

        elif motion == "G6":
            if "I" not in words or "J" not in words:
                raise GCodeError(
                    f"line {line_no}: G6 needs I and J (the X and Y semi-radii of the ellipse)"
                )
            ra, rb = words["I"] * scale, words["J"] * scale
            if abs(ra) < 1e-9 or abs(rb) < 1e-9:
                raise GCodeError(f"line {line_no}: G6 semi-radii must be non-zero")
            cx = target[0] if "X" in words else pos[0]
            cy = target[1] if "Y" in words else pos[1]
            a0 = math.radians(float(words.get("P", 0.0)))
            a1 = math.radians(float(words.get("Q", 360.0)))
            rot = math.radians(float(words.get("R", 0.0)))
            if abs(a1 - a0) < 1e-9:
                raise GCodeError(f"line {line_no}: G6 start and end angles are identical")

            cr, sr = math.cos(rot), math.sin(rot)
            ex, ey = ra * math.cos(a0), rb * math.sin(a0)
            entry = (cx + ex * cr - ey * sr, cy + ex * sr + ey * cr)
            if math.hypot(entry[0] - start_xy[0], entry[1] - start_xy[1]) > 1e-6:
                verts.append(_V(entry[0], entry[1], psi0, mv_feed, True, blend=blend_tol))
                warnings.append(
                    f"line {line_no}: added a lead-in move to the ellipse start "
                    f"({entry[0]:.2f}, {entry[1]:.2f}) mm"
                )
            pts = _flatten_ellipse(cx, cy, ra, rb, rot, a0, a1)
            emit_curve(pts, entry, psi0, target[2], mv_feed)
            target[0], target[1] = pts[-1]
            last_ctrl = None

        else:                                   # G0 / G1
            emit_line(target, mv_feed)
            last_ctrl = None

        pos = target
        started = True

    if len(verts) < 2:
        raise GCodeError("no supported motion commands found")

    verts = _blend_corners(verts)
    return GCodeProgram(points=_assign_times(verts), source=source, warnings=warnings)


__all__ = ["GCodePoint", "GCodeProgram", "GCodeError", "load_gcode_file", "parse_gcode"]
