#!/usr/bin/env python3
"""Every required status check must go red when the license audit fails.

License-first ordering makes each gate wait for the audit's verdict
(``license-first-await.yml``) before running. The trap is that "don't run the
gate" expresses itself as *skipped*, and a workflow run whose jobs all skip
concludes **success** -- so a failed audit would leave the required check green
having run nothing at all. Re-running the audit alone until it passes would
then produce an all-green PR whose gates never executed.

Each required gate therefore needs both halves:

  * an ``if:`` arm that still runs the reporting job on a negative verdict, and
  * a first step that fails when the verdict was negative.

Either half alone is silently a no-op, so this checks for both.
"""

from __future__ import annotations

import pathlib
import sys

import yaml

# The contexts required by the branch ruleset on `dev`, minus the audit itself.
# A gate added to the ruleset must be added here too.
REQUIRED_GATES = (
    "backend-required",
    "container-build-check",
    "coverage-required",
    "e2e-required",
    "frontend-required",
    "security-required",
)

NEGATIVE_ARM = "needs.await_license.outputs.license_passed != 'true'"
ASSERTION = "License audit did not pass"


def check_job(job: dict) -> list[str]:
    """Return the reasons this reporting job would not go red on a failed audit."""
    problems = []
    if "await_license" not in (job.get("needs") or []):
        problems.append("does not list await_license in `needs`, so it cannot see the verdict")
    if NEGATIVE_ARM not in " ".join(str(job.get("if", "")).split()):
        problems.append(f"`if:` has no arm on `{NEGATIVE_ARM}`, so it skips (and reads green)")
    steps = job.get("steps") or []
    if not any(ASSERTION in str(step.get("run", "")) for step in steps):
        problems.append(f"no step fails with {ASSERTION!r}, so running it still passes")
    elif ASSERTION not in str(steps[0].get("run", "")):
        problems.append("the license assertion is not the first step, so earlier steps run first")
    return problems


def _self_test() -> None:
    """The detector is worthless if it cannot reject a gate that skips to green."""
    bad = {"needs": ["changes"], "if": "always()", "steps": [{"run": "echo hi"}]}
    assert len(check_job(bad)) == 3, check_job(bad)
    good = {
        "needs": ["changes", "await_license"],
        "if": f"always() && ({NEGATIVE_ARM})",
        "steps": [{"run": f"echo '{ASSERTION}'; exit 1"}],
    }
    assert check_job(good) == [], check_job(good)


def main() -> int:
    _self_test()
    workflows = pathlib.Path(__file__).resolve().parents[2] / ".github" / "workflows"
    failed = False
    for gate in REQUIRED_GATES:
        path = workflows / f"{gate}.yml"
        if not path.exists():
            print(f"FAIL {gate}: {path} is missing")
            failed = True
            continue
        jobs = yaml.safe_load(path.read_text()).get("jobs") or {}
        if gate not in jobs:
            print(f"FAIL {gate}: no job named {gate!r}; the required check would never report")
            failed = True
            continue
        problems = check_job(jobs[gate])
        for problem in problems:
            print(f"FAIL {gate}: {problem}")
        failed = failed or bool(problems)
        if not problems:
            print(f"ok   {gate}")
    if failed:
        print("\nA required gate would report green when the license audit fails.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
