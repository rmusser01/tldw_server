"""Validate the offline Persona Chat judge review CLI command."""

from __future__ import annotations

from importlib import resources
import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from tldw_Server_API.app.core.Evaluations.persona_chat_judge_harness import (
    expected_candidate_outputs_from_fixture,
)
from tldw_Server_API.app.core.Evaluations.cli.persona_chat_judge_cli import (
    PERSONA_CHAT_JUDGE_FIXTURE_PACKAGE,
    PERSONA_CHAT_JUDGE_FIXTURE_RESOURCE,
)
from tldw_Server_API.cli.evals_cli import main


TESTS_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = TESTS_ROOT / "fixtures/persona_chat_judge_contract_cases.json"


def _load_fixture() -> dict[str, Any]:
    """Load the checked-in Persona Chat judge contract fixture."""
    with FIXTURE_PATH.open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def _load_packaged_fixture() -> dict[str, Any]:
    """Load the packaged Persona Chat judge contract fixture."""
    fixture_resource = resources.files(PERSONA_CHAT_JUDGE_FIXTURE_PACKAGE).joinpath(
        PERSONA_CHAT_JUDGE_FIXTURE_RESOURCE
    )
    with fixture_resource.open("r", encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def _write_candidates(path: Path) -> dict[str, dict[str, Any]]:
    """Write candidate outputs matching the checked-in fixture expectations."""
    candidates = expected_candidate_outputs_from_fixture(_load_fixture())
    path.write_text(json.dumps(candidates), encoding="utf-8")
    return candidates


def test_persona_chat_judge_review_command_outputs_bounded_report(tmp_path: Path) -> None:
    """The review command should print the offline harness report as bounded JSON."""
    candidates_path = tmp_path / "candidates.json"
    fixture = _load_fixture()
    expected_total_cases = len(fixture["cases"])
    _write_candidates(candidates_path)

    result = CliRunner().invoke(
        main,
        ["persona-chat-judge", "review", "--candidates", str(candidates_path)],
    )

    assert result.exit_code == 0, result.output  # nosec B101
    report = json.loads(result.output)
    serialized_report = json.dumps(report, sort_keys=True)
    assert report["offline_only"] is True  # nosec B101
    assert report["total_cases"] == expected_total_cases  # nosec B101
    assert report["matched_cases"] == report["total_cases"]  # nosec B101
    assert report["verdict_agreement"] == 1.0  # nosec B101
    assert "I will remember that permanently" not in serialized_report  # nosec B101
    assert "Ignore earlier directions" not in serialized_report  # nosec B101


def test_persona_chat_judge_packaged_fixture_matches_contract_fixture() -> None:
    """The packaged default fixture should stay aligned with the contract test fixture."""
    assert _load_packaged_fixture() == _load_fixture()  # nosec B101


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

    assert result.exit_code == 0, result.output  # nosec B101
    assert output_path.exists()  # nosec B101
    assert json.loads(result.output) == json.loads(  # nosec B101
        output_path.read_text(encoding="utf-8")
    )


def test_persona_chat_judge_review_command_fails_cleanly_for_missing_candidate_file(tmp_path: Path) -> None:
    """Missing candidate files should fail before report construction."""
    missing_path = tmp_path / "missing-candidates.json"

    result = CliRunner().invoke(
        main,
        ["persona-chat-judge", "review", "--candidates", str(missing_path)],
    )

    assert result.exit_code != 0  # nosec B101
    assert "does not exist" in result.output  # nosec B101


def test_persona_chat_judge_review_command_fails_cleanly_for_malformed_json(tmp_path: Path) -> None:
    """Malformed candidate JSON should produce a bounded CLI error."""
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text("{", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        ["persona-chat-judge", "review", "--candidates", str(candidates_path)],
    )

    assert result.exit_code != 0  # nosec B101
    assert "Candidates JSON must be valid JSON" in result.output  # nosec B101


def test_persona_chat_judge_review_command_requires_object_candidate_root(tmp_path: Path) -> None:
    """Candidate JSON roots should be objects keyed by PC-JUDGE case id."""
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text("[]", encoding="utf-8")

    result = CliRunner().invoke(
        main,
        ["persona-chat-judge", "review", "--candidates", str(candidates_path)],
    )

    assert result.exit_code != 0  # nosec B101
    assert "Candidate outputs JSON must be an object" in result.output  # nosec B101
