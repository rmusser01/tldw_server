"""Review-only policy for Persona Chat judge calibration reports.

This module classifies bounded reports produced by the offline Persona Chat
judge harness. It is intentionally a policy layer over already-produced
reports: it does not call model providers, persist results, enqueue Jobs,
expose API state, or gate runtime Persona Chat responses.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any, Literal

from tldw_Server_API.app.core.Evaluations.persona_chat_judge_harness import (
    PersonaChatJudgeHarnessReport,
)


PolicyStatus = Literal["advisory", "blocked"]
_VERDICT_CLASSES_FOR_SAMPLE_SIZE = ("pass", "fail")
_CASE_ID_RE = re.compile(r"^PC-JUDGE-\d{3}$")
_SOURCE_CASE_ID_RE = re.compile(r"^PC-CASE-\d{3}$")
_ALLOWED_CASE_STATUSES = frozenset(
    {"matched", "mismatched", "missing_candidate", "invalid_candidate"}
)
_ALLOWED_CASE_REASON_KEYS = frozenset(
    {
        "expected_flags",
        "invalid_candidate",
        "invalid_expected_flags",
        "invalid_scores",
        "invalid_verdict",
        "missing_candidate",
        "verdict",
    }
)


@dataclass(frozen=True)
class PersonaChatJudgePolicyIssue:
    """Trace-safe policy issue for one judge report case."""

    case_id: str
    source_case_id: str
    reason_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable issue without raw prompt or response text."""
        return {
            "case_id": self.case_id,
            "source_case_id": self.source_case_id,
            "reason_keys": list(self.reason_keys),
        }


@dataclass(frozen=True)
class PersonaChatJudgePolicyResult:
    """Review-only policy result for a Persona Chat judge report."""

    status: PolicyStatus
    production_calibrated: bool
    runtime_gating_allowed: bool
    reason_keys: tuple[str, ...]
    case_issues: tuple[PersonaChatJudgePolicyIssue, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable policy result."""
        return {
            "status": self.status,
            "production_calibrated": self.production_calibrated,
            "runtime_gating_allowed": self.runtime_gating_allowed,
            "reason_keys": list(self.reason_keys),
            "case_issues": [issue.to_dict() for issue in self.case_issues],
        }


def evaluate_persona_chat_judge_report_policy(
    report: PersonaChatJudgeHarnessReport | Mapping[str, Any],
    *,
    min_cases_per_verdict: int = 20,
    min_verdict_agreement: float = 1.0,
    min_flag_agreement: float = 1.0,
) -> PersonaChatJudgePolicyResult:
    """Classify a Persona Chat judge harness report for review-only use."""
    report_payload = _report_to_mapping(report)
    if report_payload is None:
        return _invalid_report()

    invalid_report = False
    total_cases = _non_negative_int(report_payload.get("total_cases"))
    matched_cases = _non_negative_int(report_payload.get("matched_cases"))
    mismatched_cases = _non_negative_int(report_payload.get("mismatched_cases"))
    invalid_candidate_count = _non_negative_int(
        report_payload.get("invalid_candidate_count")
    )
    missing_candidate_count = _non_negative_int(
        report_payload.get("missing_candidate_count")
    )
    verdict_agreement = _bounded_float(report_payload.get("verdict_agreement"))
    flag_agreement = _bounded_float(report_payload.get("flag_agreement"))
    extra_candidate_ids = report_payload.get("extra_candidate_ids")
    verdict_counts = report_payload.get("verdict_counts")
    dimension_verdict_counts = report_payload.get("dimension_verdict_counts")
    raw_cases = report_payload.get("cases")

    if (
        total_cases is None
        or matched_cases is None
        or mismatched_cases is None
        or invalid_candidate_count is None
        or missing_candidate_count is None
        or verdict_agreement is None
        or flag_agreement is None
        or not isinstance(extra_candidate_ids, list)
        or not isinstance(verdict_counts, Mapping)
        or not isinstance(raw_cases, list)
    ):
        invalid_report = True

    if invalid_report:
        return _invalid_report()

    case_rows = _valid_case_rows(raw_cases=raw_cases, total_cases=total_cases)
    if case_rows is None:
        return _invalid_report()
    status_counts = _case_status_counts(case_rows)
    if (
        status_counts["matched"] != matched_cases
        or status_counts["mismatched"] != mismatched_cases
        or status_counts["missing_candidate"] != missing_candidate_count
        or status_counts["invalid_candidate"] != invalid_candidate_count
    ):
        return _invalid_report()

    reason_keys: list[str] = []
    blocked_reason_keys: set[str] = set()

    if invalid_candidate_count > 0:
        reason_keys.append("invalid_candidates")
        blocked_reason_keys.add("invalid_candidates")
    if missing_candidate_count > 0:
        reason_keys.append("missing_candidates")
        blocked_reason_keys.add("missing_candidates")
    if extra_candidate_ids:
        reason_keys.append("extra_candidates")
        blocked_reason_keys.add("extra_candidates")
    if verdict_agreement < min_verdict_agreement:
        reason_keys.append("verdict_agreement_below_threshold")
        blocked_reason_keys.add("verdict_agreement_below_threshold")
    if flag_agreement < min_flag_agreement:
        reason_keys.append("flag_agreement_below_threshold")
        blocked_reason_keys.add("flag_agreement_below_threshold")

    sample_too_small = _sample_too_small(
        verdict_counts=verdict_counts,
        min_cases_per_verdict=min_cases_per_verdict,
    )
    dimension_sample_reason = _dimension_sample_reason(
        dimension_verdict_counts=dimension_verdict_counts,
        min_cases_per_verdict=min_cases_per_verdict,
    )
    if sample_too_small:
        reason_keys.append("sample_too_small")
    if dimension_sample_reason is not None:
        reason_keys.append(dimension_sample_reason)

    status: PolicyStatus = "blocked" if blocked_reason_keys else "advisory"
    production_calibrated = (
        status == "advisory"
        and not sample_too_small
        and dimension_sample_reason is None
        and total_cases > 0
    )
    return PersonaChatJudgePolicyResult(
        status=status,
        production_calibrated=production_calibrated,
        runtime_gating_allowed=False,
        reason_keys=tuple(dict.fromkeys(reason_keys)),
        case_issues=_case_issues(case_rows),
    )


def _report_to_mapping(
    report: PersonaChatJudgeHarnessReport | Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Normalize supported report inputs to a mapping."""
    if isinstance(report, PersonaChatJudgeHarnessReport):
        return report.to_dict()
    if isinstance(report, Mapping):
        return report
    return None


def _invalid_report() -> PersonaChatJudgePolicyResult:
    """Return the fail-closed policy result for malformed report input."""
    return PersonaChatJudgePolicyResult(
        status="blocked",
        production_calibrated=False,
        runtime_gating_allowed=False,
        reason_keys=("invalid_report",),
        case_issues=(),
    )


def _non_negative_int(value: Any) -> int | None:
    """Return non-negative integers while rejecting bools and malformed values."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _bounded_float(value: Any) -> float | None:
    """Return a finite agreement value between zero and one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric_value = float(value)
    if 0.0 <= numeric_value <= 1.0:
        return numeric_value
    return None


def _sample_too_small(
    *,
    verdict_counts: Mapping[str, Any],
    min_cases_per_verdict: int,
) -> bool:
    """Return whether pass/fail verdict classes meet the calibration threshold."""
    if min_cases_per_verdict <= 0:
        return False
    for verdict_class in _VERDICT_CLASSES_FOR_SAMPLE_SIZE:
        verdict_count = verdict_counts.get(verdict_class)
        if _non_negative_int(verdict_count) is None:
            return True
        if verdict_count < min_cases_per_verdict:
            return True
    return False


def _dimension_sample_reason(
    *,
    dimension_verdict_counts: Any,
    min_cases_per_verdict: int,
) -> str | None:
    """Return why dimension sample counts are not production-ready."""
    if min_cases_per_verdict <= 0:
        return None
    if (
        not isinstance(dimension_verdict_counts, Mapping)
        or not dimension_verdict_counts
    ):
        return "dimension_sample_counts_unavailable"
    for verdict_counts in dimension_verdict_counts.values():
        if not isinstance(verdict_counts, Mapping):
            return "dimension_sample_too_small"
        if _sample_too_small(
            verdict_counts=verdict_counts,
            min_cases_per_verdict=min_cases_per_verdict,
        ):
            return "dimension_sample_too_small"
    return None


def _valid_case_rows(
    *,
    raw_cases: list[Any],
    total_cases: int,
) -> list[Mapping[str, Any]] | None:
    """Validate report case rows before policy classification."""
    if len(raw_cases) != total_cases:
        return None
    case_rows: list[Mapping[str, Any]] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            return None
        case_id = _safe_case_id(raw_case.get("case_id"))
        source_case_id = _safe_source_case_id(raw_case.get("source_case_id"))
        status = _case_status(raw_case.get("status"))
        reason_keys = _reason_keys(raw_case.get("mismatches"))
        if not case_id or not source_case_id or status is None:
            return None
        if reason_keys is None:
            return None
        if status == "matched" and reason_keys:
            return None
        if status != "matched" and not reason_keys:
            return None
        case_rows.append(raw_case)
    return case_rows


def _case_status_counts(raw_cases: list[Mapping[str, Any]]) -> dict[str, int]:
    """Count validated case rows by harness status."""
    status_counts = {status: 0 for status in _ALLOWED_CASE_STATUSES}
    for raw_case in raw_cases:
        status = _case_status(raw_case.get("status"))
        if status is not None:
            status_counts[status] += 1
    return status_counts


def _case_issues(
    raw_cases: list[Mapping[str, Any]],
) -> tuple[PersonaChatJudgePolicyIssue, ...]:
    """Extract bounded issue summaries from harness case rows."""
    issues: list[PersonaChatJudgePolicyIssue] = []
    for raw_case in raw_cases:
        status = _case_status(raw_case.get("status"))
        reason_keys = _reason_keys(raw_case.get("mismatches"))
        if status == "matched":
            continue
        if status is None or reason_keys is None:
            continue
        issues.append(
            PersonaChatJudgePolicyIssue(
                case_id=_safe_case_id(raw_case.get("case_id")),
                source_case_id=_safe_source_case_id(raw_case.get("source_case_id")),
                reason_keys=reason_keys,
            )
        )
    return tuple(issues)


def _reason_keys(value: Any) -> tuple[str, ...] | None:
    """Return bounded string reason keys from a harness mismatch list."""
    if not isinstance(value, list):
        return None
    keys: list[str] = []
    for item in value:
        key = _safe_reason_key(item)
        if key is None:
            return None
        keys.append(key)
    return tuple(dict.fromkeys(keys))


def _safe_reason_key(value: Any) -> str | None:
    """Return allowlisted case mismatch keys only."""
    if not isinstance(value, str):
        return None
    key = value.strip()
    if key in _ALLOWED_CASE_REASON_KEYS:
        return key
    return None


def _case_status(value: Any) -> str | None:
    """Return allowlisted case status values only."""
    status = _string_or_empty(value)
    if status in _ALLOWED_CASE_STATUSES:
        return status
    return None


def _safe_case_id(value: Any) -> str:
    """Return bounded Persona Chat judge case ids only."""
    case_id = _string_or_empty(value)
    if _CASE_ID_RE.fullmatch(case_id):
        return case_id
    return ""


def _safe_source_case_id(value: Any) -> str:
    """Return bounded source fixture case ids only."""
    source_case_id = _string_or_empty(value)
    if _SOURCE_CASE_ID_RE.fullmatch(source_case_id):
        return source_case_id
    return ""


def _string_or_empty(value: Any) -> str:
    """Return stripped string values and normalize non-strings to an empty string."""
    return value.strip() if isinstance(value, str) else ""


__all__ = [
    "PersonaChatJudgePolicyIssue",
    "PersonaChatJudgePolicyResult",
    "PolicyStatus",
    "evaluate_persona_chat_judge_report_policy",
]
