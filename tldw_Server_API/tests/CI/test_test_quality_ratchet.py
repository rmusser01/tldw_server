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

import pytest
from Helper_Scripts.ci.test_quality_triage import analyze_file

REPO_ROOT = Path(__file__).resolve().parents[3]
RATCHET = REPO_ROOT / "Helper_Scripts" / "ci" / "test_quality_triage.py"
BASELINE = REPO_ROOT / "Helper_Scripts" / "ci" / "test_quality_baseline.txt"


@pytest.mark.parametrize(
    ("imports", "base"),
    [
        ("from hypothesis.stateful import RuleBasedStateMachine", "RuleBasedStateMachine"),
        ("from hypothesis.stateful import RuleBasedStateMachine as StateMachine", "StateMachine"),
        ("import hypothesis.stateful as stateful", "stateful.RuleBasedStateMachine"),
        ("import hypothesis", "hypothesis.stateful.RuleBasedStateMachine"),
    ],
)
def test_hypothesis_state_machine_invariants_are_not_fake_assertions(tmp_path, imports, base) -> None:
    source = tmp_path / "test_state_machine.py"
    source.write_text(
        f"{imports}\n\nclass WorkflowMachine({base}):\n"
        "    def check_store(self):\n"
        "        assert self.store.read() == self.expected\n",
        encoding="utf-8",
    )

    assert analyze_file(source, tmp_path) is None


@pytest.mark.parametrize(
    ("imports", "base"),
    [
        ("", ""),
        ("", "(RuleBasedStateMachine)"),
        ("from application import RuleBasedStateMachine", "(RuleBasedStateMachine)"),
    ],
)
def test_non_test_class_assertions_remain_flagged_without_hypothesis_import(tmp_path, imports, base) -> None:
    source = tmp_path / "test_fake.py"
    source.write_text(
        f"{imports}\n\nclass FakeWorkflow{base}:\n"
        "    def run(self):\n"
        "        assert self.store.read() == self.expected\n",
        encoding="utf-8",
    )

    report = analyze_file(source, tmp_path)
    assert report is not None
    assert report.flags["tautology_suspect"] == 1


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
