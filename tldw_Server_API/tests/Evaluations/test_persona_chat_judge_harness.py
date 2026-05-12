"""Validate the offline Persona Chat judge harness report builder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tldw_Server_API.app.core.Evaluations.persona_chat_judge_harness import (
    VERDICT_ORDER,
    build_persona_chat_judge_report,
    expected_candidate_outputs_from_fixture,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_PATH = REPO_ROOT / "tldw_Server_API/tests/fixtures/persona_chat_judge_contract_cases.json"


def _require(condition: object, message: str) -> None:
    """Raise a pytest-friendly assertion error when a contract condition fails."""
    if not condition:
        raise AssertionError(message)


def _load_fixture() -> dict[str, Any]:
    """Load the checked-in Persona Chat judge contract fixture."""
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _candidate_outputs() -> dict[str, dict[str, Any]]:
    """Return independent candidate outputs copied from fixture expectations."""
    return json.loads(json.dumps(expected_candidate_outputs_from_fixture(_load_fixture())))


def _copy_fixture() -> dict[str, Any]:
    """Return a mutable fixture payload copy for malformed-input checks."""
    return json.loads(json.dumps(_load_fixture()))


def _verdict_counts(pass_count: int, fail_count: int, inconclusive_count: int) -> dict[str, int]:
    """Return expected verdict counts without repeating verdict labels in tests."""
    return dict(zip(VERDICT_ORDER, (pass_count, fail_count, inconclusive_count), strict=True))


def test_expected_candidate_outputs_from_fixture_indexes_outputs_by_case_id() -> None:
    """Fixture expectations should become candidate-shaped outputs keyed by case id."""
    candidates = expected_candidate_outputs_from_fixture(_load_fixture())

    _require(set(candidates) == {"PC-JUDGE-001", "PC-JUDGE-002"}, "case ids must key candidate outputs")
    _require(candidates["PC-JUDGE-001"]["verdict"] == "pass", "positive fixture output must be available")
    _require(candidates["PC-JUDGE-002"]["expected_flags"] == ["PC-MEM-003"], "negative fixture flags must be available")


def test_report_matches_expected_contract_outputs() -> None:
    """Matching candidate outputs should produce full verdict and flag agreement."""
    report = build_persona_chat_judge_report(_load_fixture(), _candidate_outputs()).to_dict()

    _require(report["schema_version"] == "persona-chat-judge-contract/v1", "schema version should propagate")
    _require(report["offline_only"] is True, "harness must preserve the offline-only boundary")
    _require(report["total_cases"] == 2, "fixture has two calibration cases")
    _require(report["matched_cases"] == 2, "all expected candidates should match")
    _require(report["mismatched_cases"] == 0, "matching candidates should not produce mismatches")
    _require(report["missing_candidate_count"] == 0, "all expected candidates are present")
    _require(report["invalid_candidate_count"] == 0, "expected fixture outputs are valid candidates")
    _require(report["verdict_counts"] == _verdict_counts(1, 1, 0), "verdicts must be counted")
    _require(report["verdict_agreement"] == 1.0, "all verdicts match")
    _require(report["flag_agreement"] == 1.0, "all expected flags match")
    _require(report["extra_candidate_ids"] == [], "no extra candidates are present")


def test_verdict_mismatch_is_reported_without_unbounded_response_text() -> None:
    """Verdict mismatches should be bounded to ids, keys, and labels."""
    candidates = _candidate_outputs()
    candidates["PC-JUDGE-002"]["verdict"] = "pass"

    report = build_persona_chat_judge_report(_load_fixture(), candidates).to_dict()
    serialized_report = json.dumps(report, sort_keys=True)

    _require(report["matched_cases"] == 1, "one candidate should still match")
    _require(report["mismatched_cases"] == 1, "one verdict mismatch should be counted")
    _require(report["verdict_agreement"] == 0.5, "one of two verdicts matches")
    _require(report["flag_agreement"] == 1.0, "flags still match")
    _require("verdict" in report["cases"][1]["mismatches"], "case mismatch should identify verdict")
    _require("I will remember that permanently" not in serialized_report, "report must not copy assistant text")
    _require("Ignore earlier directions" not in serialized_report, "report must not copy user prompt text")


def test_flag_mismatch_is_counted_separately_from_verdict_agreement() -> None:
    """Flag mismatches should not hide otherwise matching verdicts."""
    candidates = _candidate_outputs()
    candidates["PC-JUDGE-002"]["expected_flags"] = []

    report = build_persona_chat_judge_report(_load_fixture(), candidates).to_dict()

    _require(report["matched_cases"] == 1, "flag mismatch should prevent a full match")
    _require(report["mismatched_cases"] == 1, "flag mismatch should be counted")
    _require(report["verdict_agreement"] == 1.0, "verdicts still match")
    _require(report["flag_agreement"] == 0.5, "one of two flag sets matches")
    _require("expected_flags" in report["cases"][1]["mismatches"], "case mismatch should identify flags")


def test_invalid_candidate_schema_records_invalid_result() -> None:
    """Malformed labels and missing score axes should produce invalid candidates."""
    candidates = _candidate_outputs()
    candidates["PC-JUDGE-001"]["expected_flags"] = ["bad-label"]
    del candidates["PC-JUDGE-001"]["scores"]["role_adherence"]

    report = build_persona_chat_judge_report(_load_fixture(), candidates).to_dict()
    case_result = report["cases"][0]

    _require(report["invalid_candidate_count"] == 1, "invalid candidate should be counted")
    _require(report["mismatched_cases"] == 0, "invalid candidate should not be counted as a simple mismatch")
    _require(
        report["matched_cases"]
        + report["mismatched_cases"]
        + report["missing_candidate_count"]
        + report["invalid_candidate_count"]
        == report["total_cases"],
        "summary status counts should partition total cases",
    )
    _require(case_result["status"] == "invalid_candidate", "case status should identify invalid candidate")
    _require("invalid_expected_flags" in case_result["mismatches"], "invalid labels should be reported")
    _require("invalid_scores" in case_result["mismatches"], "invalid scores should be reported")


def test_malformed_fixture_rows_are_bounded_and_empty_case_ids_are_skipped() -> None:
    """Malformed fixture scores should not crash extraction, and empty ids are ignored."""
    fixture = _copy_fixture()
    malformed_case = json.loads(json.dumps(fixture["cases"][0]))
    malformed_case["case_id"] = "PC-JUDGE-999"
    malformed_case["expected_judge_output"]["scores"] = "not-a-score-object"
    empty_id_case = json.loads(json.dumps(fixture["cases"][0]))
    empty_id_case["case_id"] = " "
    fixture["cases"].extend([malformed_case, empty_id_case])

    candidates = expected_candidate_outputs_from_fixture(fixture)
    report = build_persona_chat_judge_report(fixture, candidates).to_dict()

    _require(candidates["PC-JUDGE-999"]["scores"] == {}, "malformed fixture scores should normalize to empty")
    _require(report["total_cases"] == 3, "empty fixture case ids should be skipped")
    _require(report["verdict_counts"] == _verdict_counts(2, 1, 0), "skipped rows should not count")
