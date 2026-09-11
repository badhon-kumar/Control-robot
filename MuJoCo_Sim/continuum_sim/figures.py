"""
Publication-quality figures, in the style of Zhai et al. 2025.

This module PLOTS. It does not run anything. The simulation produces the data -
see tools/simulate.py - and hands it here, so a saved figure is always a picture
of a run that actually happened rather than a fresh run made at save time.

A "run" is one completed pass, as a dict:

    {"label": "Proposed", "t": (N,), "ref": (N,3), "pose": (N,3), "err": (N,3)}

with position in mm and attitude in degrees. save() takes the experiment and the
list of runs recorded for it, and writes the whole folder.

Output goes to outputs/<experiment key>/ and OVERWRITES what was there, so the
folder always holds the most recent run of that experiment and nothing else.

Author: Badhon Kumar
"""

import json

import numpy as np

import matplotlib
matplotlib.use("Agg")           # no GUI: must work over SSH and beside a viewer
import matplotlib.pyplot as plt

from . import params as P

# ── House style ──────────────────────────────────────────────────────────────
# Applied per figure, so importing this module cannot change anyone else's plots.
STYLE = {
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.8",
    "lines.linewidth": 1.4,
}

C_REF = "#000000"                                       # reference: dashed black
SERIES = ("#1B6CA8", "#D95F02", "#1B9E77", "#7570B3")   # measured series

LABEL = ("x (mm)", "y (mm)", r"$\psi$ ($^\circ$)")
ERRLAB = ("x error (mm)", "y error (mm)", r"$\psi$ error ($^\circ$)")

SCALE = np.array([1000.0, 1000.0, 180.0 / np.pi])       # SI -> mm / degrees


def make_run(label, t, ref, pose, meta=None):
    """Package one completed pass. Inputs in SI; stored in mm and degrees."""
    t = np.asarray(t, float)
    ref = np.asarray(ref, float)
    pose = np.asarray(pose, float)
    err = pose - ref
    err[:, 2] = (err[:, 2] + np.pi) % (2.0 * np.pi) - np.pi
    return {"label": label, "t": t, "ref": ref * SCALE, "pose": pose * SCALE,
            "err": err * SCALE, "n": len(t), "meta": meta or {}}


def stats(run, warmup=0):
    """Per-axis RMSE, MAE and MaxAE, excluding an initial transient."""
    e = run["err"][warmup:]
    if len(e) == 0:
        return {k: np.zeros(3) for k in ("rmse", "mae", "maxae")}
    return {"rmse": np.sqrt(np.mean(e ** 2, axis=0)),
            "mae": np.mean(np.abs(e), axis=0),
            "maxae": np.max(np.abs(e), axis=0)}


# ══════════════════════════════════════════════════════════════════════════════
# Figures
# ══════════════════════════════════════════════════════════════════════════════

def _finish(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"))
    plt.close(fig)


def _shade(ax, run, warmup):
    """Grey the initial approach, which the statistics exclude."""
    if warmup:
        ax.axvspan(run["t"][0], run["t"][min(warmup, run["n"] - 1)],
                   color="0.6", alpha=0.13, lw=0, zorder=0)


def fig_time(runs, stem, title=None, warmup=0, ylims=()):
    """Position and attitude against time (Fig. 6C, Fig. 7B)."""
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, 1, figsize=(6.0, 6.0), sharex=True)
        for i, ax in enumerate(axes):
            ax.plot(runs[0]["t"], runs[0]["ref"][:, i], "--",
                    color=C_REF, lw=1.2, label="Reference")
            for k, r in enumerate(runs):
                ax.plot(r["t"], r["pose"][:, i], "-",
                        color=SERIES[k % len(SERIES)], label=r["label"])
            ax.set_ylabel(LABEL[i])
            if i < len(ylims) and ylims[i]:
                ax.set_ylim(*ylims[i])
            ax.margins(x=0)
            _shade(ax, runs[0], warmup)
        axes[-1].set_xlabel("Time (s)")
        axes[0].legend(loc="upper right", ncol=len(runs) + 1)
        if title:
            fig.suptitle(title, y=0.995)
        fig.tight_layout()
        _finish(fig, stem)


def fig_error(runs, stem, channels=(0, 1, 2), title=None, warmup=0, ylim=()):
    """
    Tracking error, measurement minus reference (Fig. 6D, Fig. 7C).

    `channels` picks which of x, y, psi to show. Fig. 6D is position only and
    Fig. 7C is attitude only; showing all three regardless would bury whichever
    one the experiment was actually about.
    """
    channels = tuple(channels)
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(len(channels), 1, sharex=True, squeeze=False,
                                 figsize=(6.0, 2.0 * len(channels) + 0.8))
        axes = axes[:, 0]
        for ax, i in zip(axes, channels):
            ax.axhline(0.0, color=C_REF, lw=0.8, ls="--", alpha=0.5)
            for k, r in enumerate(runs):
                ax.plot(r["t"], r["err"][:, i], "-",
                        color=SERIES[k % len(SERIES)], label=r["label"])
            ax.set_ylabel(ERRLAB[i])
            if ylim:
                ax.set_ylim(*ylim)
            ax.margins(x=0)
            _shade(ax, runs[0], warmup)
        axes[-1].set_xlabel("Time (s)")
        axes[0].legend(loc="upper right", ncol=len(runs))
        if title:
            fig.suptitle(title, y=0.995)
        fig.tight_layout()
        _finish(fig, stem)


def fig_path(runs, stem, title=None, warmup=0):
    """The trajectory in the x-y plane (Fig. 6B)."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(4.2, 4.2))
        ax.plot(runs[0]["ref"][:, 0], runs[0]["ref"][:, 1], "--",
                color=C_REF, lw=1.2, label="Reference")
        for k, r in enumerate(runs):
            c = SERIES[k % len(SERIES)]
            if warmup:
                # The arm starts straight at x = 270 mm, so the opening steps
                # are an approach onto the path, not tracking of it. Faded, and
                # they are the same steps the statistics exclude.
                ax.plot(r["pose"][:warmup + 1, 0], r["pose"][:warmup + 1, 1],
                        "-", color=c, lw=1.0, alpha=0.30)
            ax.plot(r["pose"][warmup:, 0], r["pose"][warmup:, 1], "-",
                    color=c, label=r["label"])
        if warmup:
            ax.plot([], [], "-", color="0.5", lw=1.0, alpha=0.5,
                    label="(faded: approach)")
        ax.set_xlabel(LABEL[0])
        ax.set_ylabel(LABEL[1])
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend(loc="best")
        if title:
            ax.set_title(title)
        fig.tight_layout()
        _finish(fig, stem)


def fig_grid(runs, stem, title=None, mark_s=None):
    """
    One column per run, rows x, y and psi (Fig. 12 D to F).

    Each column keeps its own y scale: a shared one would flatten the lightest
    payload into a straight line. `mark_s` draws the instant the weight is hung,
    because a deviation with no visible cause reads as instability rather than
    as a response to something.
    """
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(3, len(runs), sharex=True, squeeze=False,
                                 figsize=(4.0 * len(runs), 6.4))
        for col, r in enumerate(runs):
            if isinstance(mark_s, (list, tuple)):
                mark = mark_s[col] if col < len(mark_s) else None
            else:
                mark = mark_s
            for row in range(3):
                ax = axes[row][col]
                ax.plot(r["t"], r["ref"][:, row], "--",
                        color=C_REF, lw=1.2, label="Desired")
                ax.plot(r["t"], r["pose"][:, row], "-",
                        color=SERIES[0], label="Measured")
                if mark is not None:
                    ax.axvline(mark, color=SERIES[1], lw=1.0, ls=":",
                               label="payload hung")
                ax.margins(x=0)
                if col == 0:
                    ax.set_ylabel(LABEL[row])
                if row == 0:
                    ax.set_title(r["label"])
                if row == 2:
                    ax.set_xlabel("Time (s)")
        axes[0][0].legend(loc="best", fontsize=7)
        if title:
            fig.suptitle(title, y=0.995)
        fig.tight_layout()
        _finish(fig, stem)


# ══════════════════════════════════════════════════════════════════════════════
# Saving
# ══════════════════════════════════════════════════════════════════════════════

def _results_text(exp, runs, warmup):
    L = [f"{exp.figure} - {exp.title}", ""]
    if exp.kind == "disturbance":
        L.append("Maximum absolute error after the payload is hung")
        L.append("")
        L.append(f"{'Payload':28s}{'t / s':>9s}{'x / mm':>12s}{'y / mm':>12s}{'psi / deg':>12s}")
        L.append("-" * 73)
        for r in runs:
            drop_s = r["meta"].get("payload_drop_s", exp.drop_s)
            after = r["err"][r["t"] >= drop_s]
            m = np.max(np.abs(after), axis=0) if len(after) else np.zeros(3)
            L.append(f"{r['label']:28s}{drop_s:9.2f}{m[0]:12.1f}{m[1]:12.1f}{m[2]:12.1f}")
    else:
        L.append(f"{runs[0]['n']} control steps, first {warmup} excluded as the "
                 f"initial approach")
        realism = runs[0]["meta"].get("realism")
        if realism:
            L.append(f"Realism preset: {realism} "
                     f"({runs[0]['meta'].get('realism_details', 'details unavailable')})")
        L.append("")
        L.append(f"{'':14s}{'x / mm':>16s}{'y / mm':>16s}{'psi / deg':>16s}")
        L.append(f"{'':14s}{'RMSE / MAE':>16s}{'RMSE / MAE':>16s}{'RMSE / MAE':>16s}")
        L.append("-" * 62)
        for r in runs:
            s = stats(r, warmup)
            cells = "".join(f"{s['rmse'][i]:7.2f} /{s['mae'][i]:6.2f}  "
                            for i in range(3))
            L.append(f"{r['label']:14s}{cells}")
    L += ["", "Error is measurement minus reference, as in the paper.", ""]
    L.append("NOTE: simulation numbers. The plant is the MuJoCo model, so what")
    L.append("is compensated is MuJoCo-vs-PCC mismatch, not hardware error.")
    L.append("They are not directly comparable with the paper's tables.")
    if tuple(exp.gravity) != (0.0, 0.0, 0.0):
        L.append("")
        L.append(f"Gravity {tuple(exp.gravity)}: the bending plane is taken to "
                 f"be VERTICAL for")
        L.append("this experiment, so that a hung weight acts in plane at all.")
        L.append("See continuum_sim/experiments.py.")
    return "\n".join(L)


def _slug(text):
    return "".join(c if c.isalnum() else "_" for c in text).strip("_").lower()


def save(exp, runs, outdir, warmup=0, quiet=False):
    """
    Write the whole folder for one experiment: figures, data, results, settings.

    The folder is cleared first, so it always reflects the most recent run and
    cannot end up holding a mix of this run and the last one.
    """
    if not runs:
        raise ValueError("nothing to save: no completed run")

    outdir.mkdir(parents=True, exist_ok=True)
    for old in outdir.iterdir():
        if old.is_file():
            old.unlink()

    title = f"{exp.figure} - {exp.title}"

    if exp.kind == "disturbance":
        marks = [r["meta"].get("payload_drop_s", exp.drop_s) for r in runs]
        fig_grid(runs, outdir / "payload_response", title, mark_s=marks)
    else:
        fig_time(runs, outdir / "time_position_attitude", title, warmup,
                 getattr(exp, "time_ylims", ()))
        fig_error(runs, outdir / "tracking_error", exp.error_channels,
                  title, warmup, getattr(exp, "error_ylim", ()))
        if exp.path_figure:
            fig_path(runs, outdir / "path_xy", title, warmup)

    hdr = ("t_s,ref_x_mm,ref_y_mm,ref_psi_deg,pose_x_mm,pose_y_mm,pose_psi_deg,"
           "err_x_mm,err_y_mm,err_psi_deg")
    for r in runs:
        rows = np.column_stack([r["t"], r["ref"], r["pose"], r["err"]])
        np.savetxt(outdir / f"data_{_slug(r['label'])}.csv", rows,
                   delimiter=",", header=hdr, comments="", fmt="%.5f")

    text = _results_text(exp, runs, warmup)
    (outdir / "results.txt").write_text(text + "\n", encoding="utf-8")

    info = {
        "experiment": exp.key,
        "figure": exp.figure,
        "title": exp.title,
        "kind": exp.kind,
        "control_dt_s": exp.ctrl_dt,
        "duration_s": exp.duration_s,
        "gravity": list(exp.gravity),
        "warmup_steps_excluded": warmup,
        "runs": [{"label": r["label"], "steps": r["n"], **r["meta"]}
                 for r in runs],
        "results_units": {"x": "mm", "y": "mm", "psi": "deg"},
        "plant_params": {"L_SEG": list(P.L_SEG), "R_TENDON": list(P.R_TENDON),
                         "n_links": P.N_LINKS},
    }
    (outdir / "run.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    if not quiet:
        print()
        print(text)
        print(f"\n  saved to {outdir}")
        for p in sorted(outdir.iterdir()):
            print(f"    {p.name}")
    return outdir
