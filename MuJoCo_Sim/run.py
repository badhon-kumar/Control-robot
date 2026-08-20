"""
Single entry point for everything in MuJoCo_Sim.

    python run.py check all           run every verification script
    python run.py check geometry      one of: install geometry tendons
                                              calibration closed-loop
    python run.py inspect             structural report + manual-drive viewer
    python run.py live                live closed-loop tracking viewer
    python run.py fig6                reproduce Fig. 6C and 6D from the paper
    python run.py build               regenerate models/continuum_planar.xml
    python run.py sync                refresh controller/ from a Continuum_v3/

Anything after the command name is passed through untouched, so the tools keep
their own flags:

    python run.py live --gcode triangle.gcode --speed 8
    python run.py fig6 --steps 400 --paper-limits
    python run.py sync --check

Every target is also runnable on its own (`python checks/geometry.py`,
`python tools/run_live.py`) - this dispatcher exists so the common commands are
short and discoverable, not because the scripts depend on it.

Author: Badhon Kumar
"""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Verification scripts, in the order they should be run: each assumes the
# previous one passed, so a failure early is the one worth reading.
CHECKS = {
    "install":     "toolchain: mujoco, rendering, a tendon smoke test",
    "geometry":    "arm structure, vendored-controller integrity, PCC error",
    "tendons":     "tendon routing and actuation vs the controller's model",
    "calibration": "plant-vs-PCC model error budget, stiffness fitting",
    "closed-loop": "the controller drives the plant, ellipse tracking",
}

TOOLS = {
    "inspect": ("tools/inspect_model.py",  "structural report + manual-drive viewer"),
    "live":    ("tools/run_live.py",       "live closed-loop tracking viewer"),
    "fig6":    ("tools/make_fig6.py",      "reproduce Fig. 6C and 6D"),
    "sync":    ("tools/sync_controller.py", "refresh controller/ from Continuum_v3/"),
}


def usage(code=0):
    print(__doc__.strip())
    print("\nChecks:")
    for name, what in CHECKS.items():
        print(f"  {name:<13} {what}")
    print("\nTools:")
    for name, (_, what) in TOOLS.items():
        print(f"  {name:<13} {what}")
    print()
    return code


def run_script(relpath, argv):
    """Execute a script as __main__ with argv, in this interpreter."""
    sys.argv = [str(ROOT / relpath)] + list(argv)
    runpy.run_path(str(ROOT / relpath), run_name="__main__")


def run_check(name):
    """Run one check. Returns its exit code (checks call sys.exit on failure)."""
    try:
        run_script(f"checks/{name.replace('-', '_')}.py", [])
    except SystemExit as e:
        return int(e.code or 0)
    return 0


def main(argv):
    if not argv or argv[0] in ("-h", "--help", "help"):
        return usage()

    cmd, rest = argv[0], argv[1:]

    if cmd == "check":
        if not rest or rest[0] == "all":
            failed = [n for n in CHECKS if run_check(n) != 0]
            print("=" * 70)
            if failed:
                print(f"FAILED: {', '.join(failed)}")
                return 1
            print(f"All {len(CHECKS)} checks passed.")
            return 0
        if rest[0] not in CHECKS:
            print(f"Unknown check '{rest[0]}'. Choose from: "
                  f"{', '.join(CHECKS)}, or 'all'.")
            return 2
        return run_check(rest[0])

    if cmd == "build":
        runpy.run_module("continuum_sim.build_model", run_name="__main__")
        return 0

    if cmd in TOOLS:
        run_script(TOOLS[cmd][0], rest)
        return 0

    print(f"Unknown command '{cmd}'.\n")
    return usage(2)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
