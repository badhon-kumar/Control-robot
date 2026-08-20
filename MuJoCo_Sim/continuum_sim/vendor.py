"""
The single point of contact with the vendored controller in controller/.

controller/ holds byte-identical copies of three files from Continuum_v3/. They
are copies and not rewrites on purpose: the premise of this project is that the
plant does not share the controller's model, so the controller under test must be
the real code, unmodified. See controller/README.md.

Importing this module puts controller/ on sys.path and nothing else - it pulls in
no heavy dependency of its own, so it is cheap for callers that only want the
integrity check. Callers that want the controller then import it plainly:

    from continuum_sim import vendor        # noqa: F401  (puts controller/ on path)
    from continuum_ellipse import make_trajectory

The import stays explicit and visible at the call site, rather than being
re-exported through here, because which controller symbols the simulation touches
is part of what the project is documenting.

This module also owns the SHA-256 integrity logic, which was previously written
out twice - once in the phase-1 check and once in sync_controller.py.

Author: Badhon Kumar
"""

import hashlib
import sys

from . import paths

# The three vendored files, in the order MANIFEST.sha256 lists them.
FILES = ("continuum_ellipse.py", "gcode_trajectory.py", "pose_feedback.py")

if str(paths.CONTROLLER) not in sys.path:
    sys.path.insert(0, str(paths.CONTROLLER))


def sha256(path):
    """Hash a file's exact bytes."""
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def read_manifest():
    """{filename: expected_sha} from MANIFEST.sha256, or {} if absent."""
    if not paths.MANIFEST.is_file():
        return {}
    out = {}
    for line in paths.MANIFEST.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            want, name = line.split("  ", 1)
            out[name] = want
    return out


def write_manifest():
    """Rewrite MANIFEST.sha256 from what is currently in controller/."""
    lines = [f"{sha256(paths.CONTROLLER / n)}  {n}"
             for n in sorted(FILES) if (paths.CONTROLLER / n).is_file()]
    paths.MANIFEST.write_text("\n".join(lines) + "\n",
                              encoding="utf-8", newline="\n")


def check_integrity():
    """
    Verify controller/ against MANIFEST.sha256.

    Returns (ok, problems). ok is False only for real corruption - a missing
    manifest yields (True, ["..."]) with an explanatory note, because a manifest
    that was never written is not evidence the files are wrong.
    """
    manifest = read_manifest()
    if not manifest:
        return True, ["MANIFEST.sha256 missing - cannot verify vendored files"]

    problems = []
    for name, want in manifest.items():
        p = paths.CONTROLLER / name
        if not p.is_file():
            problems.append(f"{name} missing")
        elif sha256(p) != want:
            problems.append(f"{name} modified")
    return not problems, problems


def check_drift():
    """
    Compare controller/ against the sibling Continuum_v3/, if one is present.

    Returns (status, names) where status is:
      "standalone" - no upstream folder; normal for a portable install
      "clean"      - copies are identical to upstream
      "drift"      - names lists the files that differ, so the simulation is
                     running an OLD controller
    """
    if not paths.UPSTREAM.is_dir():
        return "standalone", []
    drift = [n for n in FILES
             if (paths.UPSTREAM / n).is_file()
             and sha256(paths.UPSTREAM / n) != sha256(paths.CONTROLLER / n)]
    return ("drift" if drift else "clean"), drift
