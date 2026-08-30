"""Low-cardinality observations for Notes semantic indexing."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


class SemanticObservationError(ValueError):
    """An observation attempted to use a non-allowlisted field or value."""


_OPERATIONS = frozenset(
    {"initial_build", "incremental_update", "tombstone", "cleanup", "activation"}
)
_STATUSES = frozenset({"success", "degraded", "failed", "denied", "stale"})
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
    }
)
_AUDIT_EVENTS = frozenset(
    {
        "generation_publication",
        "incremental_publication",
        "tombstone_publication",
        "cleanup_completion",
    }
)
_AUDIT_REASONS = _ERROR_CODES
_COUNT_FIELDS = frozenset({"indexed", "excluded", "failed", "dirty", "pending", "chunks"})


@dataclass(frozen=True, slots=True)
class SemanticMetricEvent:
    name: str
    value: int | float
    labels: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class SemanticAuditEvent:
    event: str
    fields: Mapping[str, str | int]


def _member(value: object, allowed: frozenset[str], field: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise SemanticObservationError(f"notes_semantic_observation_{field}_invalid")
    return value


def _count(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SemanticObservationError(f"notes_semantic_observation_{field}_invalid")
    return value


def build_semantic_metric_event(
    *,
    operation: str,
    status: str,
    value: int | float,
    backend: str = "unavailable",
    error_code: str = "none",
) -> SemanticMetricEvent:
    """Build one bounded metric sample without accepting arbitrary labels."""

    if (
        isinstance(value, bool)
        or type(value) is not int
        or value < 0
        or not math.isfinite(value)
    ):
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


__all__ = [
    "SemanticAuditEvent",
    "SemanticMetricEvent",
    "SemanticObservationError",
    "build_semantic_audit_event",
    "build_semantic_metric_event",
]
