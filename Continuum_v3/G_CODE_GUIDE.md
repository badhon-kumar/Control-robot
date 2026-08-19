# G-code Guide for the Continuum Robot

The robot treats G-code as a reference trajectory for the tip pose:

```text
[x, y, psi]
```

`x` and `y` are position in the robot's bending plane, `psi` is end-effector attitude.

Parsing is done by [`gcode_trajectory.py`](gcode_trajectory.py). If that file is missing, the
GUI falls back to a smaller parser built into `continuum_ellipse.py` that supports only
`G0 G1 G2 G3 G4 G20 G21 G90 G91`. The G-code status line tells you which one is active
(`helper parser` vs `built-in parser`).

## Coordinate meaning

| Word | Meaning | Unit |
| :--- | :--- | :--- |
| `X` | desired end-effector x position | mm by default |
| `Y` | desired end-effector y position | mm by default |
| `A` | desired attitude `psi` | degrees |
| `F` | feed rate | mm/min by default |

```gcode
G1 X240 Y20 A5 F600
```

Move the tip to `x = 240 mm`, `y = 20 mm`, attitude `psi = 5 deg`, at `600 mm/min`.

## Supported commands

| Command | Meaning |
| :--- | :--- |
| `G0` | rapid linear move |
| `G1` | linear move |
| `G2` | clockwise arc — `I/J` centre offsets **or** `R` radius |
| `G3` | counter-clockwise arc — `I/J` centre offsets **or** `R` radius |
| `G4 P1` | dwell 1 second |
| `G5` | cubic Bezier curve |
| `G5.1` | quadratic Bezier curve |
| `G6` | ellipse *(non-standard extension)* |
| `G17` | XY plane — accepted, no effect (the robot is planar) |
| `G20` / `G21` | inches / millimetres |
| `G61` | exact stop: sharp corners |
| `G64 P..` | blended path: round corners by up to `P` mm |
| `G90` / `G91` | absolute / incremental |
| `M...` | ignored |

`G18` and `G19` are rejected: they select a non-XY plane, which this manipulator cannot bend in.

Several G words may share a line, so `G17 G21 G90` works.

---

## Curves

### Arcs — `G2` / `G3`

Centre-offset form. `I` and `J` give the centre relative to the current point:

```gcode
G3 X220 Y20 I-20 J0
```

Radius form — what most CAD/CAM software emits:

```gcode
G3 X220 Y20 R20
```

- `R > 0` takes the short way round (sweep ≤ 180°)
- `R < 0` takes the long way round (sweep > 180°)
- A **full circle** must use `I/J`. With `R` the start and end coincide, so the radius is ambiguous.

The endpoint is checked against the centre. If `I/J` place the end at a different radius than the
start, the file is rejected with both radii reported, rather than silently moving to the wrong place.

See [sample_arc_r.gcode](sample_arc_r.gcode) and [sample_circle_path.gcode](sample_circle_path.gcode).

### Cubic Bezier — `G5`

```gcode
G5 I<dx> J<dy> P<dx> Q<dy> X<end> Y<end>
```

- `I/J` — first control point, as an offset from the **current** point
- `P/Q` — second control point, as an offset from the **end** point

A following `G5` may omit `I/J`. The control point is then mirrored from the previous curve, so the
join is tangent-continuous with no kink:

```gcode
G5 I15 J15 P-15 Q-15 X245 Y0     ; first curve needs I and J
G5 P15 Q-15 X245 Y50             ; continues smoothly out of the one above
```

The first `G5` of a run must supply `I/J` — there is nothing to continue from.

### Quadratic Bezier — `G5.1`

```gcode
G5.1 I<dx> J<dy> X<end> Y<end>
```

One control point instead of two. Same tangent-continuation rule.

See [sample_spline_g5.gcode](sample_spline_g5.gcode).

### Ellipse — `G6`

Not standard G-code — an extension for this robot, since ellipses are the paper's reference shape.

```gcode
G6 X<cx> Y<cy> I<rx> J<ry> [P<start_deg>] [Q<end_deg>] [R<rot_deg>]
```

- `X/Y` — centre
- `I/J` — semi-radius along x and along y
- `P/Q` — start and end angle in degrees (default `0` to `360`, a full ellipse).
  `Q < P` sweeps clockwise.
- `R` — rotation of the whole ellipse in degrees

The paper's Traj. 1 in one line:

```gcode
G0 X260 Y0
G6 X240 Y0 I20 J60      ; xr = 240 + 20cos, yr = 60sin
```

If the current position is not on the ellipse, a straight lead-in move is inserted automatically and
a warning is recorded. Move to the start with `G0` first to avoid it.

See [sample_ellipse_g6.gcode](sample_ellipse_g6.gcode).

---

## Corner rounding — `G64` / `G61`

```gcode
G64 P1.0    ; corners may be cut by up to 1.0 mm
G61         ; exact corners (default)
```

Without `G64`, every corner is a sharp reversal of direction. `G64` replaces each corner with a
rounded fillet that never strays more than `P` mm from the original corner.

Both are **modal** — they apply to every corner from that line onward until changed, so put `G64`
in the header if you want the whole path blended.

A corner with a `G4` dwell on it is never rounded: the machine is meant to stop there.

See [sample_blend_g64.gcode](sample_blend_g64.gcode).

---

## Letter conflicts

Some letters mean different things in different commands. The rules are fixed:

| Letter | In `G4` | In `G5` | In `G6` | In `G64` | Elsewhere |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `A`, `PSI` | attitude | attitude | attitude | attitude | attitude |
| `P` | dwell seconds | control point dx | start angle | blend tolerance | attitude |
| `Q` | — | control point dy | end angle | — | — |
| `I`, `J` | — | control point | semi-radii | — | arc centre |
| `R` | — | — | rotation | — | arc radius |

**Use `A` for attitude.** `P` still works for backwards compatibility, but only where nothing else
claims it.

---

## Accuracy

Curves are converted to short straight chords that stay within **0.02 mm** of the true curve, so the
controller's linear interpolation between stored points cannot visibly degrade the shape.

That makes `G6` far more accurate than hand-typing an ellipse as `G1` segments:

| | points | deviation from a true ellipse |
| :--- | ---: | ---: |
| 20 hand-typed `G1` chords | 21 | 0.34 mm |
| `G6` | 72 | < 0.02 mm |

## Speed

The controller assumes quasi-static motion, so **slow paths track much better than fast ones**.

The paper's baseline is a 267 mm ellipse in 40 s, which is about **400 mm/min**. Roughly doubling
that tripled the tracking error in the paper's own tests.

- `F400` — matches the validated baseline. Recommended.
- `F600` — 1.5× faster; noticeably worse tracking.
- `F3000` — far outside anything that has been validated.

There is currently no warning if a file is too fast, and no check that the path stays inside the
robot's 270 mm reach. Both are listed in [UPGRADES.md](UPGRADES.md).

## Notes

- Comments start with `;`, or are wrapped in parentheses.
- If attitude is omitted on a move, the previous attitude is kept.
- A leading `G0` sets the path's starting point instead of tracing a move to it.
- `G4 P` values above 100 are read as milliseconds.

## Sample files

| File | Shows |
| :--- | :--- |
| [sample_square_path.gcode](sample_square_path.gcode) | basic `G1` moves |
| [sample_circle_path.gcode](sample_circle_path.gcode) | full circle via `I/J` arcs |
| [sample_arc_r.gcode](sample_arc_r.gcode) | `R`-form arcs, both directions |
| [sample_ellipse_path.gcode](sample_ellipse_path.gcode) | ellipse the old way, as `G1` chords |
| [sample_ellipse_g6.gcode](sample_ellipse_g6.gcode) | the paper's Traj. 1 as one `G6` line |
| [sample_spline_g5.gcode](sample_spline_g5.gcode) | `G5` splines with tangent continuation |
| [sample_blend_g64.gcode](sample_blend_g64.gcode) | `G64` corner rounding |
