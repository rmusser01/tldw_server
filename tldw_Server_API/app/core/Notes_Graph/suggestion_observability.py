"""Privacy-safe local observability for Notes graph suggestions."""

from __future__ import annotations

import math
import re
from enum import Enum

from loguru import logger

from tldw_Server_API.app.core.Metrics import get_metrics_registry

_SAFE_VALUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_MAX_COUNT = 1_000_000


class SuggestionEventName(str, Enum):
    RUN_ADMITTED = "run_admitted"
    SHORTLIST_COMPLETED = "shortlist_completed"
    PROVIDER_STARTED = "provider_started"
    PROVIDER_COMPLETED = "provider_completed"
    VALIDATION_REJECTED = "validation_rejected"
    STAGED = "staged"
    PUBLISHED = "published"
    CANCELLED = "cancelled"
    FAILED = "failed"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    STALE = "stale"
    RECONCILED = "reconciled"


class SuggestionErrorCode(str, Enum):
    CAPABILITIES_CHANGED = "notes_graph_capabilities_changed_before_provider"
    FINGERPRINT_STALE = "notes_graph_fingerprint_stale"
    FTS_NOT_READY = "notes_graph_fts_not_ready"
    GENERATION_CANCELLED = "notes_graph_generation_cancelled"
    JOB_CONTRACT_INVALID = "notes_graph_job_contract_invalid"
    JOB_MISSING = "notes_graph_job_missing"
    JOB_RESULT_CONTRACT_INVALID = "notes_graph_job_result_contract_invalid"
    PROVIDER_RETRY_POLICY_UNSUPPORTED = "notes_graph_provider_retry_policy_unsupported"
    PROVIDER_UNAVAILABLE = "notes_graph_provider_unavailable"
    PUBLICATION_RECEIPT_MISMATCH = "notes_graph_publication_receipt_mismatch"
    PUBLICATION_RECEIPT_MISSING = "notes_graph_publication_receipt_missing"
    PUBLICATION_STATE_MISSING = "notes_graph_publication_state_missing"
    RUN_CONFLICT = "notes_graph_run_conflict"
    SOURCE_TOO_LARGE = "notes_graph_source_too_large"
    SUGGESTION_NO_VALID_ITEMS = "notes_graph_suggestion_no_valid_items"
    SUGGESTION_SUPPRESSION_LIMIT = "notes_graph_suggestion_suppression_limit"


class DecisionOutcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    STALE = "stale"


class ReconciliationOutcome(str, Enum):
    COMPLETED = "completed"
    RELEASED = "released"
    FAILED = "failed"


def _safe_value(value: str, field: str) -> str:
    if not isinstance(value, str) or _SAFE_VALUE.fullmatch(value) is None:
        raise ValueError(f"unsafe suggestion observability {field}")
    return value


def _count(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_COUNT:
        raise ValueError(f"invalid suggestion observability {field}")
    return value


def _duration(value: float, field: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 86_400:
        raise ValueError(f"invalid suggestion observability {field}")
    return number


def _error_code(value: SuggestionErrorCode) -> str:
    if not isinstance(value, SuggestionErrorCode):
        raise ValueError("unknown suggestion error code")
    return value.value


def _write_event(payload: dict[str, object]) -> None:
    logger.bind(**payload).info("Notes graph suggestion lifecycle event")


def record_event(
    event: SuggestionEventName,
    *,
    run_id: str,
    job_id: str | None = None,
    suggestion_id: str | None = None,
    count: int | None = None,
    duration_seconds: float | None = None,
    error_code: SuggestionErrorCode | None = None,
) -> None:
    """Record one closed event containing only safe identifiers and scalars."""

    if not isinstance(event, SuggestionEventName):
        raise ValueError("unknown suggestion event")
    payload: dict[str, object] = {
        "event": event.value,
        "run_id": _safe_value(run_id, "run_id"),
    }
    for field, value in (("job_id", job_id), ("suggestion_id", suggestion_id)):
        if value is not None:
            payload[field] = _safe_value(value, field)
    if count is not None:
        payload["count"] = _count(count, "count")
    if duration_seconds is not None:
        payload["duration_seconds"] = _duration(duration_seconds, "duration")
    if error_code is not None:
        payload["error_code"] = _error_code(error_code)
    _write_event(payload)


def _observe(name: str, value: float) -> None:
    get_metrics_registry().observe(name, value, labels={})


def _increment(name: str, value: int, labels: dict[str, str] | None = None) -> None:
    get_metrics_registry().increment(name, value=value, labels=labels or {})


def record_queue_latency(seconds: float) -> None:
    _observe("notes_graph_suggestion_queue_latency_seconds", _duration(seconds, "latency"))


def record_run_duration(seconds: float) -> None:
    _observe("notes_graph_suggestion_run_duration_seconds", _duration(seconds, "duration"))


def record_candidate_counts(*, candidates: int, evidence: int) -> None:
    _observe("notes_graph_suggestion_candidate_count", _count(candidates, "candidates"))
    _observe("notes_graph_suggestion_evidence_count", _count(evidence, "evidence"))


def record_provider_usage(*, input_tokens: int, output_tokens: int) -> None:
    _observe("notes_graph_suggestion_provider_input_tokens", _count(input_tokens, "input_tokens"))
    _observe("notes_graph_suggestion_provider_output_tokens", _count(output_tokens, "output_tokens"))


def record_validation_counts(*, validated: int, dropped: int) -> None:
    _observe("notes_graph_suggestion_validated_count", _count(validated, "validated"))
    _observe("notes_graph_suggestion_dropped_count", _count(dropped, "dropped"))


def record_run_error(error_code: SuggestionErrorCode) -> None:
    _increment(
        "notes_graph_suggestion_run_errors_total",
        1,
        {"error_code": _error_code(error_code)},
    )


def record_decision_outcome(outcome: DecisionOutcome) -> None:
    if not isinstance(outcome, DecisionOutcome):
        raise ValueError("unknown suggestion decision outcome")
    _increment("notes_graph_suggestion_decisions_total", 1, {"outcome": outcome.value})


def record_acceptance_reconciliation(outcome: ReconciliationOutcome) -> None:
    if not isinstance(outcome, ReconciliationOutcome):
        raise ValueError("unknown suggestion reconciliation outcome")
    _increment(
        "notes_graph_suggestion_acceptance_reconciliation_total",
        1,
        {"outcome": outcome.value},
    )


__all__ = [
    "DecisionOutcome",
    "ReconciliationOutcome",
    "SuggestionErrorCode",
    "SuggestionEventName",
    "record_acceptance_reconciliation",
    "record_candidate_counts",
    "record_decision_outcome",
    "record_event",
    "record_provider_usage",
    "record_queue_latency",
    "record_run_duration",
    "record_run_error",
    "record_validation_counts",
]
