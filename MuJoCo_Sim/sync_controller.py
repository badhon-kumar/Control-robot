"""
Re-sync the vendored controller in controller/ from a sibling Continuum_v3/.

MuJoCo_Sim runs standalone by carrying its own copy of the controller (see
controller/README.md). That copy goes stale the moment the original is edited,
and stale means the simulation is quietly testing an OLD controller. This script
refreshes it and rewrites MANIFEST.sha256.

    python MuJoCo_Sim/sync_controller.py            # copy if different
    python MuJoCo_Sim/sync_controller.py --check    # report only, change nothing

Run check_phase1.py afterwards: it verifies the manifest and re-checks that the
geometry constants still agree with sim/params.py.

Author: Badhon Kumar
"""

import argparse
import hashlib
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(HERE, "controller")
UPSTREAM = os.path.join(os.path.dirname(HERE), "Continuum_v3")

FILES = ["continuum_ellipse.py", "gcode_trajectory.py", "pose_feedback.py"]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def write_manifest():
    lines = []
    for n in sorted(FILES):
        p = os.path.join(VENDOR, n)
        if os.path.isfile(p):
            lines.append(f"{sha(p)}  {n}")
    with open(os.path.join(VENDOR, "MANIFEST.sha256"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="report differences without copying anything")
    ap.add_argument("--from", dest="src", default=UPSTREAM,
                    help="source folder (default ../Continuum_v3)")
    a = ap.parse_args()

    if not os.path.isdir(a.src):
        print(f"No source folder at {a.src}")
        print("MuJoCo_Sim runs standalone from controller/, so this is only an "
              "error if you meant to re-sync.")
        return 1

    changed, missing = [], []
    for n in FILES:
        s, d = os.path.join(a.src, n), os.path.join(VENDOR, n)
        if not os.path.isfile(s):
            missing.append(n)
        elif not os.path.isfile(d) or sha(s) != sha(d):
            changed.append(n)

    for n in missing:
        print(f"  MISSING upstream: {n}")

    if not changed:
        print("  vendored controller is already up to date")
        return 1 if missing else 0

    for n in changed:
        print(f"  {'would update' if a.check else 'updated'}: {n}")
        if not a.check:
            shutil.copy2(os.path.join(a.src, n), os.path.join(VENDOR, n))

    if a.check:
        print("\n  --check: nothing written. Re-run without it to apply.")
        return 1

    write_manifest()
    print("\n  MANIFEST.sha256 rewritten.")
    print("  Now run:  python MuJoCo_Sim/check_phase1.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
