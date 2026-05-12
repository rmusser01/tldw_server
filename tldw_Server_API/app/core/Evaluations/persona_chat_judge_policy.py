"""Review-only policy for Persona Chat judge calibration reports.

This module classifies bounded reports produced by the offline Persona Chat
judge harness. It is intentionally a policy layer over already-produced
reports: it does not call model providers, persist results, enqueue Jobs,
expose API state, or gate runtime Persona Chat responses.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from tldw_Server_API.app.core.Evaluations.persona_chat_judge_harness import (
    PersonaChatJudgeHarnessReport,
)


PolicyStatus = Literal["advisory", "blocked"]
_VERDICT_CLASSES_FOR_SAMPLE_SIZE = ("pass", "fail")


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
    invalid_candidate_count = _non_negative_int(report_payload.get("invalid_candidate_count"))
    missing_candidate_count = _non_negative_int(report_payload.get("missing_candidate_count"))
    verdict_agreement = _bounded_float(report_payload.get("verdict_agreement"))
    flag_agreement = _bounded_float(report_payload.get("flag_agreement"))
    extra_candidate_ids = report_payload.get("extra_candidate_ids")
    verdict_counts = report_payload.get("verdict_counts")
    raw_cases = report_payload.get("cases")

    if (
        total_cases is None
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
    if sample_too_small:
        reason_keys.append("sample_too_small")

    status: PolicyStatus = "blocked" if blocked_reason_keys else "advisory"
    production_calibrated = status == "advisory" and not sample_too_small and total_cases > 0
    return PersonaChatJudgePolicyResult(
        status=status,
        production_calibrated=production_calibrated,
        runtime_gating_allowed=False,
        reason_keys=tuple(dict.fromkeys(reason_keys)),
        case_issues=_case_issues(raw_cases),
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


def _case_issues(raw_cases: list[Any]) -> tuple[PersonaChatJudgePolicyIssue, ...]:
    """Extract bounded issue summaries from harness case rows."""
    issues: list[PersonaChatJudgePolicyIssue] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            continue
        status = _string_or_empty(raw_case.get("status"))
        raw_mismatches = raw_case.get("mismatches")
        reason_keys = _reason_keys(raw_mismatches)
        if status == "matched" and not reason_keys:
            continue
        if not reason_keys and status:
            reason_keys = (status,)
        issues.append(
            PersonaChatJudgePolicyIssue(
                case_id=_string_or_empty(raw_case.get("case_id")),
                source_case_id=_string_or_empty(raw_case.get("source_case_id")),
                reason_keys=reason_keys,
            )
        )
    return tuple(issues)


def _reason_keys(value: Any) -> tuple[str, ...]:
    """Return bounded string reason keys from a harness mismatch list."""
    if not isinstance(value, list):
        return ()
    return tuple(
        dict.fromkeys(str(item) for item in value if isinstance(item, str) and item.strip())
    )


def _string_or_empty(value: Any) -> str:
    """Normalize optional identifier values to stripped strings."""
    return "" if value is None else str(value).strip()


__all__ = [
    "PersonaChatJudgePolicyIssue",
    "PersonaChatJudgePolicyResult",
    "PolicyStatus",
    "evaluate_persona_chat_judge_report_policy",
]
