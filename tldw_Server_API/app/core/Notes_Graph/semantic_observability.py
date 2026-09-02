"""Low-cardinality observations for Notes semantic indexing."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from types import MappingProxyType
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.DB_Management.chacha.note_semantic_models import (
    SemanticHealthSnapshot,
)
from tldw_Server_API.app.core.Metrics import (
    MetricDefinition,
    MetricType,
    get_metrics_registry,
    increment_counter,
    observe_histogram,
    set_gauge,
)


class SemanticObservationError(ValueError):
    """An observation attempted to use a non-allowlisted field or value."""


_OPERATIONS = frozenset({"initial_build", "incremental_update", "tombstone", "cleanup", "activation"})
_STATUSES = frozenset({"success", "degraded", "failed", "cancelled", "denied", "stale"})
_BACKENDS = frozenset({"chromadb", "pgvector", "unavailable"})
_ERROR_CODES = frozenset(
    {
        "none",
        "note_excluded",
        "note_failed",
        "provider_failure",
        "configuration_drift",
        "vector_failure",
        "permission_denied",
        "cleanup_failed",
        "backend_unavailable",
        "cleanup_unconfirmed",
        "dataset_limit_exceeded",
        "fence_lost",
        "timeout",
    }
)
_AUDIT_EVENTS = frozenset(
    {
        "cancel",
        "consent_renewal",
        "delete_request",
        "disable",
        "dsr_cleanup",
        "enable",
        "generation_publication",
        "incremental_publication",
        "manual_conversion",
        "rebuild",
        "retry",
        "tombstone_publication",
        "cleanup_completion",
    }
)
_AUDIT_REASONS = _ERROR_CODES | frozenset(
    {
        "cancelled",
        "capability_denied",
        "configuration_denied",
        "kill_switch",
        "unavailable",
    }
)
_COUNT_FIELDS = frozenset({"indexed", "excluded", "failed", "dirty", "pending", "chunks"})
_BUILD_OPERATIONS = frozenset({"build", "rebuild", "incremental_update", "maintain", "retry_failed", "delete"})
_QUERY_STATUSES = frozenset({"success", "degraded", "failed", "denied", "stale"})
_QUERY_STAGES = frozenset({"candidate", "filtered", "admitted"})
_TRUNCATIONS = frozenset(
    {
        "max_degree",
        "max_edges",
        "max_nodes",
        "semantic_candidates",
        "semantic_edges",
        "semantic_evidence_bytes",
        "semantic_nodes",
    }
)
_DENIAL_REASONS = frozenset({"capability", "configuration", "kill_switch", "permission"})
_FAILURE_COMPONENTS = frozenset({"provider", "vector"})
_FAILURE_CATEGORIES = frozenset(
    {"configuration", "execution", "fence", "invalid_response", "timeout", "unavailable", "unknown"}
)
_METRICS_LOCK = Lock()
_METRICS_REGISTERED = False

_METRIC_DEFINITIONS = (
    MetricDefinition(
        name="notes_semantic_build_duration_seconds",
        type=MetricType.HISTOGRAM,
        description="Notes semantic build and update duration",
        unit="s",
        labels=["operation", "status", "backend"],
        buckets=[0.01, 0.1, 0.5, 1, 5, 15, 60, 300],
    ),
    MetricDefinition(
        name="notes_semantic_builds_total",
        type=MetricType.COUNTER,
        description="Notes semantic build and update terminal outcomes",
        labels=["operation", "status", "backend"],
    ),
    MetricDefinition(
        name="notes_semantic_note_count",
        type=MetricType.GAUGE,
        description="Notes semantic note counts by bounded state",
        labels=["state", "backend"],
    ),
    MetricDefinition(
        name="notes_semantic_coverage_ratio",
        type=MetricType.GAUGE,
        description="Notes semantic indexed-note coverage ratio",
        labels=["backend"],
    ),
    MetricDefinition(
        name="notes_semantic_stale_generations",
        type=MetricType.GAUGE,
        description="Notes semantic stale generation count",
        labels=["backend"],
    ),
    MetricDefinition(
        name="notes_semantic_query_duration_seconds",
        type=MetricType.HISTOGRAM,
        description="Notes semantic vector query duration",
        unit="s",
        labels=["status", "backend"],
        buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5],
    ),
    MetricDefinition(
        name="notes_semantic_query_stage_total",
        type=MetricType.COUNTER,
        description="Notes semantic query candidates by bounded stage",
        labels=["stage", "status", "backend"],
    ),
    MetricDefinition(
        name="notes_semantic_truncations_total",
        type=MetricType.COUNTER,
        description="Notes semantic query truncations",
        labels=["reason", "backend"],
    ),
    MetricDefinition(
        name="notes_semantic_failures_total",
        type=MetricType.COUNTER,
        description="Notes semantic provider and vector failures",
        labels=["component", "category", "backend"],
    ),
    MetricDefinition(
        name="notes_semantic_cleanup_backlog",
        type=MetricType.GAUGE,
        description="Notes semantic cleanup backlog",
        labels=["backend"],
    ),
    MetricDefinition(
        name="notes_semantic_cleanup_retries_total",
        type=MetricType.GAUGE,
        description="Current Notes semantic cleanup retry total",
        labels=["status", "backend"],
    ),
    MetricDefinition(
        name="notes_semantic_cleanup_oldest_age_seconds",
        type=MetricType.GAUGE,
        description="Age of the oldest pending Notes semantic cleanup item",
        unit="s",
        labels=["backend"],
    ),
    MetricDefinition(
        name="notes_semantic_denials_total",
        type=MetricType.COUNTER,
        description="Notes semantic permission, capability, configuration, and kill-switch denials",
        labels=["reason"],
    ),
    MetricDefinition(
        name="notes_semantic_cancellations_total",
        type=MetricType.COUNTER,
        description="Notes semantic cancellations",
        labels=["operation"],
    ),
    MetricDefinition(
        name="notes_semantic_dsr_total",
        type=MetricType.COUNTER,
        description="Notes semantic data-subject erasure outcomes",
        labels=["status", "backend"],
    ),
)


@dataclass(frozen=True, slots=True)
class SemanticMetricEvent:
    name: str
    value: int | float
    labels: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class SemanticAuditEvent:
    event: str
    fields: Mapping[str, str | int]


@dataclass(slots=True)
class _SemanticHealthTotal:
    indexed_notes: int = 0
    excluded_notes: int = 0
    failed_notes: int = 0
    dirty_notes: int = 0
    pending_notes: int = 0
    stale_generations: int = 0
    cleanup_backlog: int = 0
    cleanup_retries: int = 0
    oldest_cleanup_created_at: datetime | None = None


def _member(value: object, allowed: frozenset[str], field: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise SemanticObservationError(f"notes_semantic_observation_{field}_invalid")
    return value


def _count(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SemanticObservationError(f"notes_semantic_observation_{field}_invalid")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise SemanticObservationError(f"notes_semantic_observation_{field}_invalid")
    return float(value)


def _backend(value: str) -> str:
    return _member(value, _BACKENDS, "backend")


def _ensure_metrics_registered() -> None:
    global _METRICS_REGISTERED
    if _METRICS_REGISTERED:
        return
    with _METRICS_LOCK:
        if _METRICS_REGISTERED:
            return
        registry = get_metrics_registry()
        for definition in _METRIC_DEFINITIONS:
            registry.register_metric(definition)
        _METRICS_REGISTERED = True


def _metric_call(function: Any, *args: object, **kwargs: object) -> None:
    try:
        _ensure_metrics_registered()
        function(*args, **kwargs)
    except Exception:  # noqa: BLE001 - semantic behavior never depends on telemetry
        logger.debug("Notes semantic metric emission unavailable")


def build_semantic_metric_event(
    *,
    operation: str,
    status: str,
    value: int | float,
    backend: str = "unavailable",
    error_code: str = "none",
) -> SemanticMetricEvent:
    """Build one bounded metric sample without accepting arbitrary labels."""

    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or not math.isfinite(value):
        raise SemanticObservationError("notes_semantic_observation_value_invalid")
    labels = MappingProxyType(
        {
            "operation": _member(operation, _OPERATIONS, "operation"),
            "status": _member(status, _STATUSES, "status"),
            "backend": _member(backend, _BACKENDS, "backend"),
            "error_code": _member(error_code, _ERROR_CODES, "error_code"),
        }
    )
    return SemanticMetricEvent("notes_semantic_operations_total", value, labels)


def build_semantic_audit_event(
    *,
    event: str,
    status: str,
    reason: str = "none",
    counts: Mapping[str, int] | None = None,
    **extra: object,
) -> SemanticAuditEvent:
    """Build one identifier-free audit event from a fixed field vocabulary."""

    if extra:
        raise SemanticObservationError("notes_semantic_observation_audit_field_invalid")
    fields: dict[str, str | int] = {
        "status": _member(status, _STATUSES, "status"),
        "reason": _member(reason, _AUDIT_REASONS, "reason"),
    }
    for key, value in (counts or {}).items():
        if key not in _COUNT_FIELDS:
            raise SemanticObservationError("notes_semantic_observation_audit_field_invalid")
        fields[key] = _count(value, key)
    return SemanticAuditEvent(
        _member(event, _AUDIT_EVENTS, "event"),
        MappingProxyType(fields),
    )


def record_semantic_build_metrics(
    *,
    operation: str,
    status: str,
    backend: str,
    duration_seconds: int | float,
    counts: Mapping[str, int],
) -> None:
    """Record one terminal build/update result without identifier labels."""

    operation = _member(operation, _BUILD_OPERATIONS, "operation")
    status = _member(status, _STATUSES, "status")
    backend = _backend(backend)
    duration = _number(duration_seconds, "duration")
    for key, value in counts.items():
        if key not in _COUNT_FIELDS:
            raise SemanticObservationError("notes_semantic_observation_count_invalid")
        _count(value, key)
    labels = {"operation": operation, "status": status, "backend": backend}
    _metric_call(
        observe_histogram,
        "notes_semantic_build_duration_seconds",
        duration,
        labels,
    )
    _metric_call(increment_counter, "notes_semantic_builds_total", 1, labels)


def record_semantic_health_metrics(
    *,
    backend: str,
    counts: Mapping[str, int],
    stale_generations: int = 0,
) -> None:
    """Record current coverage and state counts without run identifiers."""

    backend = _backend(backend)
    bounded_counts = {key: _count(value, key) for key, value in counts.items() if key in _COUNT_FIELDS}
    if len(bounded_counts) != len(counts):
        raise SemanticObservationError("notes_semantic_observation_count_invalid")
    for state in ("indexed", "excluded", "failed", "dirty", "pending"):
        _metric_call(
            set_gauge,
            "notes_semantic_note_count",
            bounded_counts.get(state, 0),
            {"state": state, "backend": backend},
        )
    denominator = sum(bounded_counts.get(state, 0) for state in ("indexed", "excluded", "failed", "pending"))
    coverage = bounded_counts.get("indexed", 0) / denominator if denominator else 1.0
    _metric_call(
        set_gauge,
        "notes_semantic_coverage_ratio",
        coverage,
        {"backend": backend},
    )
    _metric_call(
        set_gauge,
        "notes_semantic_stale_generations",
        _count(stale_generations, "stale_generations"),
        {"backend": backend},
    )


def record_semantic_query_metrics(
    *,
    status: str,
    backend: str,
    duration_seconds: int | float,
    candidate_count: int,
    filtered_count: int,
    admitted_count: int,
    truncations: tuple[str, ...] = (),
) -> None:
    """Record bounded vector query latency, stages, and truncation reasons."""

    status = _member(status, _QUERY_STATUSES, "status")
    backend = _backend(backend)
    labels = {"status": status, "backend": backend}
    _metric_call(
        observe_histogram,
        "notes_semantic_query_duration_seconds",
        _number(duration_seconds, "duration"),
        labels,
    )
    for stage, count in (
        ("candidate", candidate_count),
        ("filtered", filtered_count),
        ("admitted", admitted_count),
    ):
        _member(stage, _QUERY_STAGES, "stage")
        _metric_call(
            increment_counter,
            "notes_semantic_query_stage_total",
            _count(count, stage),
            {"stage": stage, **labels},
        )
    for reason in tuple(dict.fromkeys(truncations)):
        _metric_call(
            increment_counter,
            "notes_semantic_truncations_total",
            1,
            {"reason": _member(reason, _TRUNCATIONS, "truncation"), "backend": backend},
        )


def record_semantic_cleanup_metrics(
    *,
    status: str,
    backend: str,
    backlog: int,
    retries: int,
    oldest_age_seconds: int | float,
) -> None:
    """Record bounded cleanup backlog, retry, and age observations."""

    status = _member(status, _STATUSES, "status")
    backend = _backend(backend)
    _metric_call(
        set_gauge,
        "notes_semantic_cleanup_backlog",
        _count(backlog, "backlog"),
        {"backend": backend},
    )
    retry_count = _count(retries, "retries")
    for other_status in ("success", "degraded", "failed"):
        if other_status == status:
            continue
        _metric_call(
            set_gauge,
            "notes_semantic_cleanup_retries_total",
            0,
            {"status": other_status, "backend": backend},
        )
    _metric_call(
        set_gauge,
        "notes_semantic_cleanup_retries_total",
        retry_count,
        {"status": status, "backend": backend},
    )
    _metric_call(
        set_gauge,
        "notes_semantic_cleanup_oldest_age_seconds",
        _number(oldest_age_seconds, "oldest_age"),
        {"backend": backend},
    )


def _cleanup_timestamp(value: object) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def aggregate_semantic_health_snapshots(
    snapshots: Sequence[object],
) -> tuple[SemanticHealthSnapshot, ...]:
    """Aggregate persisted dataset snapshots into one row per bounded backend."""

    totals = {backend: _SemanticHealthTotal() for backend in sorted(_BACKENDS)}
    for snapshot in snapshots:
        backend = _backend(str(getattr(snapshot, "backend", "unavailable")))
        total = totals[backend]
        total.indexed_notes += _count(getattr(snapshot, "indexed_notes", 0), "indexed_notes")
        total.excluded_notes += _count(getattr(snapshot, "excluded_notes", 0), "excluded_notes")
        total.failed_notes += _count(getattr(snapshot, "failed_notes", 0), "failed_notes")
        total.dirty_notes += _count(getattr(snapshot, "dirty_notes", 0), "dirty_notes")
        total.pending_notes += _count(getattr(snapshot, "pending_notes", 0), "pending_notes")
        total.stale_generations += _count(
            getattr(snapshot, "stale_generations", 0),
            "stale_generations",
        )
        total.cleanup_backlog += _count(getattr(snapshot, "cleanup_backlog", 0), "cleanup_backlog")
        total.cleanup_retries += _count(getattr(snapshot, "cleanup_retries", 0), "cleanup_retries")
        oldest = _cleanup_timestamp(getattr(snapshot, "oldest_cleanup_created_at", None))
        current_oldest = total.oldest_cleanup_created_at
        if oldest is not None and (current_oldest is None or oldest < current_oldest):
            total.oldest_cleanup_created_at = oldest
    return tuple(
        SemanticHealthSnapshot(
            backend=backend,
            indexed_notes=total.indexed_notes,
            excluded_notes=total.excluded_notes,
            failed_notes=total.failed_notes,
            dirty_notes=total.dirty_notes,
            pending_notes=total.pending_notes,
            stale_generations=total.stale_generations,
            cleanup_backlog=total.cleanup_backlog,
            cleanup_retries=total.cleanup_retries,
            oldest_cleanup_created_at=total.oldest_cleanup_created_at,
        )
        for backend, total in totals.items()
    )


def record_semantic_aggregate_metrics(
    *,
    snapshots: Sequence[object],
    now: datetime,
) -> None:
    """Publish one complete backend-health snapshot after a bounded store sweep."""

    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise SemanticObservationError("notes_semantic_observation_timestamp_invalid")
    observed_at = now.astimezone(timezone.utc)
    for snapshot in aggregate_semantic_health_snapshots(snapshots):
        record_semantic_health_metrics(
            backend=snapshot.backend,
            counts={
                "indexed": snapshot.indexed_notes,
                "excluded": snapshot.excluded_notes,
                "failed": snapshot.failed_notes,
                "dirty": snapshot.dirty_notes,
                "pending": snapshot.pending_notes,
            },
            stale_generations=snapshot.stale_generations,
        )
        oldest = _cleanup_timestamp(snapshot.oldest_cleanup_created_at)
        age = 0.0 if oldest is None else max(0.0, (observed_at - oldest).total_seconds())
        record_semantic_cleanup_metrics(
            status="failed" if snapshot.cleanup_retries else "success",
            backend=snapshot.backend,
            backlog=snapshot.cleanup_backlog,
            retries=snapshot.cleanup_retries,
            oldest_age_seconds=age,
        )


def record_semantic_denial(reason: str) -> None:
    reason = _member(reason, _DENIAL_REASONS, "denial")
    _metric_call(
        increment_counter,
        "notes_semantic_denials_total",
        1,
        {"reason": reason},
    )


def record_semantic_cancellation(operation: str) -> None:
    operation = _member(operation, _BUILD_OPERATIONS, "operation")
    _metric_call(
        increment_counter,
        "notes_semantic_cancellations_total",
        1,
        {"operation": operation},
    )


def record_semantic_failure(*, component: str, category: str, backend: str) -> None:
    _metric_call(
        increment_counter,
        "notes_semantic_failures_total",
        1,
        {
            "component": _member(component, _FAILURE_COMPONENTS, "component"),
            "category": _member(category, _FAILURE_CATEGORIES, "category"),
            "backend": _backend(backend),
        },
    )


def record_semantic_dsr_metrics(*, status: str, backend: str) -> None:
    _metric_call(
        increment_counter,
        "notes_semantic_dsr_total",
        1,
        {
            "status": _member(status, _STATUSES, "status"),
            "backend": _backend(backend),
        },
    )


async def emit_semantic_audit_event(
    *,
    owner_user_id: str,
    dataset_id: str,
    event: str,
    status: str,
    reason: str = "none",
    generation_id: str | None = None,
    run_id: str | None = None,
    source_note_id: str | None = None,
    target_note_id: str | None = None,
    counts: Mapping[str, int] | None = None,
    audit_service: Any | None = None,
) -> None:
    """Write one content-free semantic event through the durable audit service."""

    observation = build_semantic_audit_event(
        event=event,
        status=status,
        reason=reason,
        counts=counts,
    )
    if not isinstance(owner_user_id, str) or not owner_user_id:
        raise SemanticObservationError("notes_semantic_observation_owner_invalid")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise SemanticObservationError("notes_semantic_observation_dataset_invalid")
    if audit_service is None:
        from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import (
            get_or_create_audit_service_for_user_id_optional,
        )

        audit_service = await get_or_create_audit_service_for_user_id_optional(owner_user_id)
    from tldw_Server_API.app.core.Audit.unified_audit_service import (
        AuditContext,
        AuditEventCategory,
        AuditEventType,
    )

    metadata: dict[str, str | int] = {
        "semantic_event": observation.event,
        **dict(observation.fields),
    }
    for key, value in (
        ("generation_id", generation_id),
        ("run_id", run_id),
        ("target_note_id", target_note_id),
    ):
        if isinstance(value, str) and value:
            metadata[key] = value
    delete_events = {"cleanup_completion", "delete_request", "disable", "dsr_cleanup"}
    event_type = AuditEventType.DATA_DELETE if observation.event in delete_events else AuditEventType.DATA_UPDATE
    await audit_service.log_event(
        event_type=event_type,
        category=AuditEventCategory.DATA_MODIFICATION,
        context=AuditContext(user_id=owner_user_id),
        resource_type=(
            "notes_semantic_relationship" if observation.event == "manual_conversion" else "notes_semantic_index"
        ),
        resource_id=source_note_id or dataset_id,
        action=f"notes_semantic.{observation.event}",
        result=status,
        metadata=metadata,
    )
    await audit_service.flush(raise_on_failure=True)


__all__ = [
    "SemanticAuditEvent",
    "SemanticMetricEvent",
    "SemanticObservationError",
    "aggregate_semantic_health_snapshots",
    "build_semantic_audit_event",
    "build_semantic_metric_event",
    "emit_semantic_audit_event",
    "record_semantic_build_metrics",
    "record_semantic_cancellation",
    "record_semantic_cleanup_metrics",
    "record_semantic_denial",
    "record_semantic_dsr_metrics",
    "record_semantic_failure",
    "record_semantic_aggregate_metrics",
    "record_semantic_health_metrics",
    "record_semantic_query_metrics",
]
