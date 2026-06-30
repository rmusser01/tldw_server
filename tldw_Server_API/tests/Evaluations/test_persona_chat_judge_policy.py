"""Persona Chat judge calibration policy tests."""

from __future__ import annotations

from importlib import resources
import json
from typing import Any

from tldw_Server_API.app.core.Evaluations.cli.persona_chat_judge_cli import (
    PERSONA_CHAT_JUDGE_FIXTURE_PACKAGE,
    PERSONA_CHAT_JUDGE_FIXTURE_RESOURCE,
)
from tldw_Server_API.app.core.Evaluations.persona_chat_judge_harness import (
    VERDICT_ORDER,
    build_persona_chat_judge_report,
    expected_candidate_outputs_from_fixture,
)
from tldw_Server_API.app.core.Evaluations.persona_chat_judge_policy import (
    evaluate_persona_chat_judge_report_policy,
)


def _load_fixture() -> dict[str, Any]:
    """Load the packaged Persona Chat judge fixture."""
    fixture_resource = resources.files(PERSONA_CHAT_JUDGE_FIXTURE_PACKAGE).joinpath(
        PERSONA_CHAT_JUDGE_FIXTURE_RESOURCE
    )
    with fixture_resource.open("r", encoding="utf-8") as fixture_file:
        payload = json.load(fixture_file)
    assert isinstance(payload, dict)  # nosec B101
    return payload


def _candidate_outputs() -> dict[str, Any]:
    """Return independent candidate outputs copied from fixture expectations."""
    return json.loads(
        json.dumps(expected_candidate_outputs_from_fixture(_load_fixture()))
    )


def test_clean_fixture_report_remains_advisory_until_sample_threshold() -> None:
    """The tiny synthetic fixture can be reviewed but is not production calibrated."""
    report = build_persona_chat_judge_report(_load_fixture(), _candidate_outputs())

    policy = evaluate_persona_chat_judge_report_policy(report)

    assert policy.status == "advisory"  # nosec B101
    assert policy.production_calibrated is False  # nosec B101
    assert policy.runtime_gating_allowed is False  # nosec B101
    assert "sample_too_small" in policy.reason_keys  # nosec B101
    assert policy.case_issues == ()  # nosec B101


def test_invalid_missing_and_extra_candidates_are_blocked_with_bounded_issues() -> None:
    """Structural report failures should block trust without echoing raw trace text."""
    candidates = _candidate_outputs()
    candidates["PC-JUDGE-001"] = "not-a-candidate-object"
    del candidates["PC-JUDGE-002"]
    candidates["PC-JUDGE-999"] = {"verdict": "pass"}
    report = build_persona_chat_judge_report(_load_fixture(), candidates)

    policy = evaluate_persona_chat_judge_report_policy(report)
    policy_payload = policy.to_dict()
    serialized = json.dumps(policy_payload, sort_keys=True)

    assert policy.status == "blocked"  # nosec B101
    assert policy.production_calibrated is False  # nosec B101
    assert "invalid_candidates" in policy.reason_keys  # nosec B101
    assert "missing_candidates" in policy.reason_keys  # nosec B101
    assert "extra_candidates" in policy.reason_keys  # nosec B101
    assert {
        (issue.case_id, issue.source_case_id, issue.reason_keys)
        for issue in policy.case_issues
    } == {
        ("PC-JUDGE-001", "PC-CASE-008", ("invalid_candidate",)),
        ("PC-JUDGE-002", "PC-CASE-015", ("missing_candidate",)),
    }  # nosec B101
    assert "I will remember that permanently" not in serialized  # nosec B101
    assert "Ignore earlier directions" not in serialized  # nosec B101


def test_low_agreement_blocks_report_even_when_candidate_schema_is_valid() -> None:
    """A schema-valid verdict disagreement should be a blocked calibration result."""
    candidates = _candidate_outputs()
    candidates["PC-JUDGE-002"]["verdict"] = "pass"
    report = build_persona_chat_judge_report(_load_fixture(), candidates)

    policy = evaluate_persona_chat_judge_report_policy(
        report,
        min_cases_per_verdict=1,
        min_verdict_agreement=0.75,
    )

    assert policy.status == "blocked"  # nosec B101
    assert "verdict_agreement_below_threshold" in policy.reason_keys  # nosec B101
    assert "sample_too_small" not in policy.reason_keys  # nosec B101
    assert policy.case_issues[0].case_id == "PC-JUDGE-002"  # nosec B101
    assert policy.case_issues[0].reason_keys == ("verdict",)  # nosec B101


def test_policy_accepts_dict_report_input_and_serializes_stable_shape() -> None:
    """The CLI-facing report dict should classify the same as the dataclass report."""
    report = build_persona_chat_judge_report(
        _load_fixture(),
        _candidate_outputs(),
    ).to_dict()

    policy = evaluate_persona_chat_judge_report_policy(report)
    payload = policy.to_dict()

    assert set(payload) == {  # nosec B101
        "status",
        "production_calibrated",
        "runtime_gating_allowed",
        "reason_keys",
        "case_issues",
    }
    assert (
        json.loads(json.dumps(payload, sort_keys=True)) == payload
    )  # nosec B101
    assert policy.status == "advisory"  # nosec B101


def test_policy_blocks_dict_report_with_malformed_case_rows() -> None:
    """Malformed case rows should fail closed instead of being skipped."""
    report = build_persona_chat_judge_report(
        _load_fixture(),
        _candidate_outputs(),
    ).to_dict()
    report["cases"] = ["not-a-dict"]

    policy = evaluate_persona_chat_judge_report_policy(report)

    assert policy.status == "blocked"  # nosec B101
    assert policy.production_calibrated is False  # nosec B101
    assert policy.reason_keys == ("invalid_report",)  # nosec B101
    assert policy.case_issues == ()  # nosec B101


def test_policy_blocks_dict_report_with_unbounded_case_fields() -> None:
    """Unbounded ids and mismatch text should not enter policy output."""
    report = build_persona_chat_judge_report(
        _load_fixture(),
        _candidate_outputs(),
    ).to_dict()
    report["cases"][0]["status"] = "mismatched"
    report["cases"][0]["mismatches"] = ["assistant_text: raw response"]

    policy = evaluate_persona_chat_judge_report_policy(report)
    serialized = json.dumps(policy.to_dict(), sort_keys=True)

    assert policy.status == "blocked"  # nosec B101
    assert policy.reason_keys == ("invalid_report",)  # nosec B101
    assert "assistant_text: raw response" not in serialized  # nosec B101


def test_policy_needs_dimension_counts_for_production_calibration() -> None:
    """Aggregate pass/fail counts alone should not claim dimension calibration."""
    report = build_persona_chat_judge_report(
        _load_fixture(),
        _candidate_outputs(),
    ).to_dict()
    report["total_cases"] = 40
    report["matched_cases"] = 40
    report["verdict_counts"] = dict(zip(VERDICT_ORDER, (20, 20, 0), strict=True))
    report["cases"] = [
        {
            "case_id": f"PC-JUDGE-{case_number:03d}",
            "source_case_id": f"PC-CASE-{case_number:03d}",
            "status": "matched",
            "mismatches": [],
            "verdict_match": True,
            "flag_match": True,
            "score_schema_valid": True,
        }
        for case_number in range(1, 41)
    ]

    policy = evaluate_persona_chat_judge_report_policy(report)

    assert policy.status == "advisory"  # nosec B101
    assert policy.production_calibrated is False  # nosec B101
    assert "sample_too_small" not in policy.reason_keys  # nosec B101
    assert "dimension_sample_counts_unavailable" in policy.reason_keys  # nosec B101


def test_malformed_report_is_blocked_as_invalid_report() -> None:
    """Malformed report dictionaries should fail closed."""
    policy = evaluate_persona_chat_judge_report_policy(
        {
            "total_cases": "two",
            "verdict_agreement": "high",
            "flag_agreement": 1.0,
            "cases": "not-a-list",
        }
    )

    assert policy.status == "blocked"  # nosec B101
    assert policy.production_calibrated is False  # nosec B101
    assert policy.reason_keys == ("invalid_report",)  # nosec B101
    assert policy.case_issues == ()  # nosec B101
