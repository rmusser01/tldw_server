"""Validate the offline Persona Chat judge review CLI command."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from tldw_Server_API.app.core.Evaluations.persona_chat_judge_harness import (
    expected_candidate_outputs_from_fixture,
)
from tldw_Server_API.cli.evals_cli import main


TESTS_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = TESTS_ROOT / "fixtures/persona_chat_judge_contract_cases.json"


def _require(condition: object, message: str) -> None:
    """Raise a pytest-friendly assertion error when a contract condition fails."""
    if not condition:
        raise AssertionError(message)


def _load_fixture() -> dict[str, Any]:
    """Load the checked-in Persona Chat judge contract fixture."""
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _write_candidates(path: Path) -> dict[str, dict[str, Any]]:
    """Write candidate outputs matching the checked-in fixture expectations."""
    candidates = expected_candidate_outputs_from_fixture(_load_fixture())
    path.write_text(json.dumps(candidates), encoding="utf-8")
    return candidates


def test_persona_chat_judge_review_command_outputs_bounded_report(tmp_path: Path) -> None:
    """The review command should print the offline harness report as bounded JSON."""
    candidates_path = tmp_path / "candidates.json"
    _write_candidates(candidates_path)

    result = CliRunner().invoke(
        main,
        ["persona-chat-judge", "review", "--candidates", str(candidates_path)],
    )

    _require(result.exit_code == 0, result.output)
    report = json.loads(result.output)
    serialized_report = json.dumps(report, sort_keys=True)
    _require(report["offline_only"] is True, "review command must preserve offline-only boundary")
    _require(report["total_cases"] == 2, "review command should compare the fixture cases")
    _require(report["matched_cases"] == 2, "fixture-matching candidates should fully match")
    _require(report["verdict_agreement"] == 1.0, "fixture-matching verdicts should agree")
    _require("I will remember that permanently" not in serialized_report, "report must not copy assistant text")
    _require("Ignore earlier directions" not in serialized_report, "report must not copy user prompt text")


def test_persona_chat_judge_review_command_writes_explicit_output_file(tmp_path: Path) -> None:
    """The review command should optionally persist the same bounded report to a file."""
    candidates_path = tmp_path / "candidates.json"
    output_path = tmp_path / "report.json"
    _write_candidates(candidates_path)

    result = CliRunner().invoke(
        main,
        [
            "persona-chat-judge",
            "review",
            "--candidates",
            str(candidates_path),
            "--output",
            str(output_path),
        ],
    )

    _require(result.exit_code == 0, result.output)
    _require(output_path.exists(), "review command should create the requested output file")
    _require(
        json.loads(result.output) == json.loads(output_path.read_text(encoding="utf-8")),
        "stdout and file report should match",
    )


def test_persona_chat_judge_review_command_fails_cleanly_for_missing_candidate_file(tmp_path: Path) -> None:
    """Missing candidate files should fail before report construction."""
    missing_path = tmp_path / "missing-candidates.json"

    result = CliRunner().invoke(
        main,
        ["persona-chat-judge", "review", "--candidates", str(missing_path)],
    )

    _require(result.exit_code != 0, "missing candidate file should fail")
    _require("does not exist" in result.output, "click should report the missing candidate file")


def test_persona_chat_judge_review_command_fails_cleanly_for_malformed_json(tmp_path: Path) -> None:
    """Malformed candidate JSON should produce a bounded CLI error."""
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text("{", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        ["persona-chat-judge", "review", "--candidates", str(candidates_path)],
    )

    _require(result.exit_code != 0, "malformed JSON should fail")
    _require("Candidates JSON must be valid JSON" in result.output, "error should identify malformed candidates")


def test_persona_chat_judge_review_command_requires_object_candidate_root(tmp_path: Path) -> None:
    """Candidate JSON roots should be objects keyed by PC-JUDGE case id."""
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text("[]", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        ["persona-chat-judge", "review", "--candidates", str(candidates_path)],
    )

    _require(result.exit_code != 0, "non-object candidate roots should fail")
    _require(
        "Candidate outputs JSON must be an object" in result.output,
        "error should identify the object requirement",
    )
