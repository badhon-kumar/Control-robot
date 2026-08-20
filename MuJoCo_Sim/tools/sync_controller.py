"""
Re-sync the vendored controller in controller/ from a sibling Continuum_v3/.

MuJoCo_Sim runs standalone by carrying its own copy of the controller (see
controller/README.md). That copy goes stale the moment the original is edited,
and stale means the simulation is quietly testing an OLD controller. This script
refreshes it and rewrites MANIFEST.sha256.

    python run.py sync                    # copy if different
    python run.py sync --check            # report only, change nothing

Run `python run.py check geometry` afterwards: it verifies the manifest and
re-checks that the geometry constants still agree with continuum_sim/params.py.

The SHA-256 and manifest logic lives in continuum_sim.vendor, shared with the
geometry check, so the two can never disagree about what "up to date" means.

Author: Badhon Kumar
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import shutil

from continuum_sim import paths, vendor
from continuum_sim.vendor import FILES, sha256 as sha, write_manifest

VENDOR = paths.CONTROLLER
UPSTREAM = paths.UPSTREAM


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
        s, d = os.path.join(a.src, n), os.path.join(str(VENDOR), n)
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
            shutil.copy2(os.path.join(a.src, n), os.path.join(str(VENDOR), n))

    if a.check:
        print("\n  --check: nothing written. Re-run without it to apply.")
        return 1

    write_manifest()
    print("\n  MANIFEST.sha256 rewritten.")
    print("  Now run:  python run.py check geometry")
    return 0


if __name__ == "__main__":
    sys.exit(main())
