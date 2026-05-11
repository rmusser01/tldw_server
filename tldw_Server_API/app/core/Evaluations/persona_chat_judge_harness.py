"""Offline Persona Chat judge contract replay harness.

The harness compares already-produced candidate judge outputs with the V1
Persona Chat judge contract fixture. It performs deterministic schema checks
and bounded mismatch reporting only; it does not call LLM providers, persist
evaluation runs, enqueue Jobs, or gate runtime Persona Chat responses.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any, Literal


ALLOWED_VERDICTS = frozenset({"pass", "fail", "inconclusive"})
REQUIRED_SCORE_NAMES = frozenset(
    {
        "role_adherence",
        "boundary_behavior",
        "memory_semantics",
        "exemplar_use",
        "grounding_separation",
    }
)
LABEL_RE = re.compile(r"^PC-[A-Z]+-\d{3}$")
CaseStatus = Literal["matched", "mismatched", "missing_candidate", "invalid_candidate"]


@dataclass(frozen=True)
class PersonaChatJudgeCaseResult:
    """Bounded comparison result for one Persona Chat judge fixture case."""

    case_id: str
    source_case_id: str
    status: CaseStatus
    verdict_match: bool
    flag_match: bool
    score_schema_valid: bool
    mismatches: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable case result without prompt or response text."""
        return {
            "case_id": self.case_id,
            "source_case_id": self.source_case_id,
            "status": self.status,
            "verdict_match": self.verdict_match,
            "flag_match": self.flag_match,
            "score_schema_valid": self.score_schema_valid,
            "mismatches": list(self.mismatches),
        }


@dataclass(frozen=True)
class PersonaChatJudgeHarnessReport:
    """Aggregated offline Persona Chat judge calibration report."""

    schema_version: str
    offline_only: bool
    total_cases: int
    matched_cases: int
    mismatched_cases: int
    missing_candidate_count: int
    invalid_candidate_count: int
    extra_candidate_ids: tuple[str, ...]
    verdict_agreement: float
    flag_agreement: float
    cases: tuple[PersonaChatJudgeCaseResult, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report with bounded mismatch details."""
        return {
            "schema_version": self.schema_version,
            "offline_only": self.offline_only,
            "total_cases": self.total_cases,
            "matched_cases": self.matched_cases,
            "mismatched_cases": self.mismatched_cases,
            "missing_candidate_count": self.missing_candidate_count,
            "invalid_candidate_count": self.invalid_candidate_count,
            "extra_candidate_ids": list(self.extra_candidate_ids),
            "verdict_agreement": self.verdict_agreement,
            "flag_agreement": self.flag_agreement,
            "cases": [case.to_dict() for case in self.cases],
        }


def expected_candidate_outputs_from_fixture(fixture_payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Extract candidate-shaped expected judge outputs keyed by fixture case id."""
    candidates: dict[str, dict[str, Any]] = {}
    for case in _case_rows(fixture_payload):
        case_id = _string_or_empty(case.get("case_id"))
        output = case.get("expected_judge_output")
        if case_id and isinstance(output, Mapping):
            candidates[case_id] = {
                "verdict": output.get("verdict"),
                "scores": dict(output.get("scores") or {}),
                "expected_flags": list(output.get("expected_flags") or []),
                "rationale": output.get("rationale"),
                "evidence": list(output.get("evidence") or []),
            }
    return candidates


def build_persona_chat_judge_report(
    fixture_payload: Mapping[str, Any],
    candidate_outputs_by_case_id: Mapping[str, Mapping[str, Any]],
) -> PersonaChatJudgeHarnessReport:
    """Compare candidate judge outputs with fixture expectations."""
    case_results: list[PersonaChatJudgeCaseResult] = []
    fixture_case_ids: set[str] = set()
    for case in _case_rows(fixture_payload):
        case_id = _string_or_empty(case.get("case_id"))
        source_case_id = _string_or_empty(case.get("source_case_id"))
        fixture_case_ids.add(case_id)
        expected_output = _mapping_or_empty(case.get("expected_judge_output"))
        candidate_output = candidate_outputs_by_case_id.get(case_id)
        case_results.append(
            _compare_case(
                case_id=case_id,
                source_case_id=source_case_id,
                expected_output=expected_output,
                candidate_output=candidate_output,
            )
        )

    total_cases = len(case_results)
    matched_cases = sum(1 for result in case_results if result.status == "matched")
    missing_candidate_count = sum(1 for result in case_results if result.status == "missing_candidate")
    invalid_candidate_count = sum(1 for result in case_results if result.status == "invalid_candidate")
    verdict_matches = sum(1 for result in case_results if result.verdict_match)
    flag_matches = sum(1 for result in case_results if result.flag_match)
    extra_candidate_ids = tuple(
        sorted(str(case_id) for case_id in candidate_outputs_by_case_id if str(case_id) not in fixture_case_ids)
    )

    return PersonaChatJudgeHarnessReport(
        schema_version=_string_or_empty(fixture_payload.get("schema_version")),
        offline_only=bool(fixture_payload.get("offline_only")),
        total_cases=total_cases,
        matched_cases=matched_cases,
        mismatched_cases=total_cases - matched_cases - missing_candidate_count,
        missing_candidate_count=missing_candidate_count,
        invalid_candidate_count=invalid_candidate_count,
        extra_candidate_ids=extra_candidate_ids,
        verdict_agreement=_ratio(verdict_matches, total_cases),
        flag_agreement=_ratio(flag_matches, total_cases),
        cases=tuple(case_results),
    )


def _compare_case(
    *,
    case_id: str,
    source_case_id: str,
    expected_output: Mapping[str, Any],
    candidate_output: Mapping[str, Any] | None,
) -> PersonaChatJudgeCaseResult:
    """Compare one candidate output with one expected fixture output."""
    if candidate_output is None:
        return PersonaChatJudgeCaseResult(
            case_id=case_id,
            source_case_id=source_case_id,
            status="missing_candidate",
            verdict_match=False,
            flag_match=False,
            score_schema_valid=False,
            mismatches=("missing_candidate",),
        )

    mismatches: list[str] = []
    candidate_validation_errors = _candidate_validation_errors(candidate_output)
    if candidate_validation_errors:
        mismatches.extend(candidate_validation_errors)

    verdict_match = candidate_output.get("verdict") == expected_output.get("verdict")
    if not verdict_match:
        mismatches.append("verdict")

    expected_flags = _normalized_flags(expected_output.get("expected_flags"))
    candidate_flags = _normalized_flags(candidate_output.get("expected_flags"))
    flag_match = candidate_flags == expected_flags
    if not flag_match:
        mismatches.append("expected_flags")

    score_schema_valid = "invalid_scores" not in candidate_validation_errors
    status: CaseStatus
    if candidate_validation_errors:
        status = "invalid_candidate"
    elif mismatches:
        status = "mismatched"
    else:
        status = "matched"

    return PersonaChatJudgeCaseResult(
        case_id=case_id,
        source_case_id=source_case_id,
        status=status,
        verdict_match=verdict_match,
        flag_match=flag_match,
        score_schema_valid=score_schema_valid,
        mismatches=tuple(dict.fromkeys(mismatches)),
    )


def _candidate_validation_errors(candidate_output: Mapping[str, Any]) -> list[str]:
    """Return bounded validation error keys for a candidate judge output."""
    errors: list[str] = []
    if candidate_output.get("verdict") not in ALLOWED_VERDICTS:
        errors.append("invalid_verdict")

    raw_flags = candidate_output.get("expected_flags")
    if not isinstance(raw_flags, list):
        errors.append("invalid_expected_flags")
    elif (
        any(not isinstance(label, str) or not LABEL_RE.fullmatch(label) for label in raw_flags)
        or len(set(raw_flags)) != len(raw_flags)
    ):
        errors.append("invalid_expected_flags")

    if _score_validation_errors(candidate_output.get("scores")):
        errors.append("invalid_scores")
    return errors


def _score_validation_errors(scores: Any) -> list[str]:
    """Return validation errors for the required judge score envelope."""
    if not isinstance(scores, Mapping):
        return ["scores_not_object"]
    if set(scores.keys()) != REQUIRED_SCORE_NAMES:
        return ["score_keys"]
    invalid_values = [
        score_name
        for score_name, score in scores.items()
        if not (
            score is None
            or (
                isinstance(score, (int, float))
                and not isinstance(score, bool)
                and 0.0 <= score <= 1.0
            )
        )
    ]
    return ["score_values"] if invalid_values else []


def _case_rows(fixture_payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return fixture case rows as mapping objects only."""
    cases = fixture_payload.get("cases")
    if not isinstance(cases, list):
        return []
    return [case for case in cases if isinstance(case, Mapping)]


def _mapping_or_empty(value: Any) -> Mapping[str, Any]:
    """Return mapping values unchanged and normalize other values to an empty mapping."""
    return value if isinstance(value, Mapping) else {}


def _string_or_empty(value: Any) -> str:
    """Return stripped string values and normalize non-strings to an empty string."""
    return value.strip() if isinstance(value, str) else ""


def _normalized_flags(value: Any) -> tuple[str, ...]:
    """Return sorted unique string flags for comparison."""
    if not isinstance(value, list):
        return ()
    return tuple(sorted({flag for flag in value if isinstance(flag, str)}))


def _ratio(numerator: int, denominator: int) -> float:
    """Return a stable rounded ratio for report summaries."""
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)
