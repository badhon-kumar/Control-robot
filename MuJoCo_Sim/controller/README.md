# `controller/` — vendored copies, do not edit

These three files are **verbatim copies** taken from `Continuum_v3/`. They are
here so that `MuJoCo_Sim/` runs standalone, with no sibling folder required.

| File | What the simulation uses from it |
| :--- | :--- |
| `continuum_ellipse.py` | `KalmanJacobianController`, `forward_kinematics`, `pcc_angles`, `make_trajectory`, and the `L_SEG` / `R_TENDON` constants |
| `gcode_trajectory.py` | `load_gcode_file`, `GCodeProgram` |
| `pose_feedback.py` | `UdpPoseFeedbackReceiver` (used only by the Phase 4 UDP test) |

## Why these are copies and not a rewrite

The entire premise of this project is that the MuJoCo plant is **independent of
the controller**: the controller must be *your actual code*, unmodified, so that
Phase 5 measures the real method rather than a reimplementation of it. Extracting
or tidying these files would break that guarantee. They are copied byte-for-byte.

## The risk this creates, and how it is handled

Vendoring means there are now two copies of the controller, which can drift apart
— and silent drift would invalidate every result without any error appearing.

`MANIFEST.sha256` records the SHA-256 of each file as vendored.
`checks/geometry.py` verifies it on every run:

- it always checks the vendored files still match the manifest, so local edits
  here are caught immediately;
- if a sibling `Continuum_v3/` is also present, it additionally compares against
  that upstream copy and **warns loudly** if they differ.

On a machine that only has `MuJoCo_Sim/`, the upstream comparison is skipped and
the manifest check still applies.

## Re-syncing after changing the original

If you edit `Continuum_v3/continuum_ellipse.py`, the vendored copy is stale.
Refresh it and regenerate the manifest:

```powershell
python run.py sync
```

That copies the current `Continuum_v3/` files in and rewrites `MANIFEST.sha256`.
Re-run `checks/geometry.py` afterwards to confirm the geometry constants still agree.

## Note on dependencies

`continuum_ellipse.py` imports `tkinter` at module level (it is a GUI program),
so a Python build without tkinter cannot import it even headlessly. `tkinter`
ships with standard CPython on Windows and macOS; on minimal Linux images it may
need installing separately. `pyserial` is imported inside a `try/except` and is
genuinely optional.
