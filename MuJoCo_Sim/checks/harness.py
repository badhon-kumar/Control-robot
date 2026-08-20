"""
Shared pass/fail reporting for the verification scripts in checks/.

These four helpers were previously copy-pasted into all four phase scripts, so a
change to the output format meant four edits and the failure-tracking list was
per-file. Report collects failures instead, and check_* scripts end with
report.finish(name), which prints the summary and sets the exit code.

Author: Badhon Kumar
"""

import sys


def ok(msg):   print(f"  [ OK ] {msg}")
def fail(msg): print(f"  [FAIL] {msg}")
def warn(msg): print(f"  [WARN] {msg}")


class Report:
    """Accumulates failures across a check script and renders the verdict."""

    def __init__(self):
        self.failures = []

    def expect(self, cond, good, bad):
        """Report `good` if cond holds, otherwise record `bad` as a failure."""
        if cond:
            ok(good)
        else:
            fail(bad)
            self.failures.append(bad)
        return bool(cond)

    ok = staticmethod(ok)
    fail = staticmethod(fail)
    warn = staticmethod(warn)

    def finish(self, name, exit_on_failure=True):
        """Print the summary line. Exits non-zero if anything failed."""
        print("\n" + "-" * 70)
        if self.failures:
            print(f"{name} FAILED - {len(self.failures)} check(s):")
            for f in self.failures:
                print(f"  - {f}")
            if exit_on_failure:
                sys.exit(1)
            return False
        print(f"{name} complete - all checks passed.\n")
        return True
