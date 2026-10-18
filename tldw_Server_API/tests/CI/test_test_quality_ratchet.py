"""The test-quality triage ratchet must stay green (no new or worse offenses).

Runs ``Helper_Scripts/ci/test_quality_triage.py --enforce`` in a subprocess
from the repo root so the committed baseline gates all future test-quality
drift. To clear a failure, fix the flagged test and shrink the baseline with
``--write-baseline``; never raise an entry to make the gate pass.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
RATCHET = REPO_ROOT / "Helper_Scripts" / "ci" / "test_quality_triage.py"
BASELINE = REPO_ROOT / "Helper_Scripts" / "ci" / "test_quality_baseline.txt"


def test_test_quality_triage_enforces_the_committed_baseline() -> None:
    assert BASELINE.is_file(), "missing test-quality baseline; regenerate with --write-baseline"
    result = subprocess.run(
        [sys.executable, str(RATCHET), "--enforce"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert result.returncode == 0, (
        "test-quality ratchet failed - new or worse offenses vs the baseline "
        "(fix the flagged test and regenerate the baseline, never raise a count):\n"
        f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
    )
