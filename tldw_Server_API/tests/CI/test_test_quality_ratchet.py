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
        ("from hypothesis import stateful", "stateful.RuleBasedStateMachine"),
        ("from hypothesis import stateful as states", "states.RuleBasedStateMachine"),
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
        ("from application import stateful", "(stateful.RuleBasedStateMachine)"),
        ("from .hypothesis import stateful", "(stateful.RuleBasedStateMachine)"),
        ("from .hypothesis.stateful import RuleBasedStateMachine", "(RuleBasedStateMachine)"),
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


@pytest.mark.parametrize("intermediate", [False, True])
def test_pytest_contract_mixin_assertions_are_not_fake_assertions(tmp_path, intermediate) -> None:
    source = tmp_path / "test_storage_contract.py"
    parent = "Intermediate" if intermediate else "StorageContract"
    bridge = "class Intermediate(StorageContract):\n    pass\n" if intermediate else ""
    source.write_text(
        "class StorageContract:\n"
        "    def test_persisted_value(self, database):\n"
        "        assert database.read() == {'tokens': 47}\n"
        f"{bridge}class TestSQLite({parent}):\n    pass\n",
        encoding="utf-8",
    )

    assert analyze_file(source, tmp_path) is None


@pytest.mark.parametrize(
    ("body", "flag"),
    [
        ("        assert response.status_code == 200\n", "status_only"),
        (
            "        calls = []\n"
            "        def fake():\n            calls.append('called')\n"
            "        fake()\n        assert calls == ['called']\n",
            "tautology_suspect",
        ),
    ],
)
def test_pytest_contract_mixin_tests_still_receive_quality_checks(tmp_path, body, flag) -> None:
    source = tmp_path / "test_storage_contract.py"
    source.write_text(
        "class StorageContract:\n"
        "    def test_inherited_behavior(self):\n"
        f"{body}class TestSQLite(StorageContract):\n    pass\n",
        encoding="utf-8",
    )

    report = analyze_file(source, tmp_path)
    assert report is not None
    assert report.flags.get(flag) == 1


def test_uncollected_class_with_test_named_method_remains_a_fake(tmp_path) -> None:
    source = tmp_path / "test_fake_contract.py"
    source.write_text(
        "class FakeContract:\n" "    def test_value(self):\n        assert self.value == 'only-a-fake'\n",
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
