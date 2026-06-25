from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class OperationOutcome(str, Enum):
    APPLIED = "applied"
    NO_TRANSITION = "no_transition"
    ADMISSION_REJECTED = "admission_rejected"
    BACKEND_CONFLICT = "backend_conflict"
    BACKEND_SCHEMA_ERROR = "backend_schema_error"
    BACKEND_ERROR = "backend_error"


class NoTransitionReason(str, Enum):
    MISSING = "missing"
    WRONG_STATUS = "wrong_status"
    STALE_LEASE = "stale_lease"
    ALREADY_TERMINAL = "already_terminal"
    IDEMPOTENT_EXISTING = "idempotent_existing"
    RLS_FILTERED = "rls_filtered"


class AdmissionRejectionReason(str, Enum):
    QUEUE_PAUSED = "queue_paused"
    QUEUE_DRAINING = "queue_draining"
    QUOTA_EXCEEDED = "quota_exceeded"
    FAIR_SHARE_LIMIT = "fair_share_limit"
    DEPENDENCY_BLOCKED = "dependency_blocked"
    POLICY_REJECTED = "policy_rejected"


@dataclass(frozen=True)
class CreateJobCommand:
    domain: str
    queue: str
    job_type: str
    payload: dict[str, Any] | None
    owner_user_id: str | None
    idempotency_key: str | None = None
    priority: int = 100
    max_retries: int = 3
    available_at: datetime | None = None
    project_id: str | None = None
    batch_group: str | None = None
    request_id: str | None = None
    trace_id: str | None = None


@dataclass(frozen=True)
class AdmissionResult:
    outcome: OperationOutcome
    row: dict[str, Any] | None = None
    inserted: bool = False
    no_transition_reason: NoTransitionReason | None = None
    admission_rejection_reason: AdmissionRejectionReason | None = None
    durable_events: Sequence[dict[str, Any]] = field(default_factory=tuple)
    message: str | None = None

    @classmethod
    def existing(cls, *, row: dict[str, Any]) -> "AdmissionResult":
        return cls(
            outcome=OperationOutcome.NO_TRANSITION,
            row=row,
            inserted=False,
            no_transition_reason=NoTransitionReason.IDEMPOTENT_EXISTING,
        )

    @classmethod
    def rejected(cls, reason: AdmissionRejectionReason, *, message: str | None = None) -> "AdmissionResult":
        return cls(
            outcome=OperationOutcome.ADMISSION_REJECTED,
            admission_rejection_reason=reason,
            message=message,
        )


def _inserted_admission_result(
    cls,
    *,
    row: dict[str, Any],
    durable_events: Sequence[dict[str, Any]] = (),
) -> AdmissionResult:
    return cls(
        outcome=OperationOutcome.APPLIED,
        row=row,
        inserted=True,
        durable_events=durable_events,
    )


AdmissionResult.inserted = classmethod(_inserted_admission_result)  # type: ignore[method-assign]


@dataclass(frozen=True)
class LifecycleResult:
    outcome: OperationOutcome
    transition_applied: bool
    row: dict[str, Any] | None = None
    no_transition_reason: NoTransitionReason | None = None
    durable_events: Sequence[dict[str, Any]] = field(default_factory=tuple)
    message: str | None = None

    @classmethod
    def applied(
        cls,
        *,
        row: dict[str, Any],
        durable_events: Sequence[dict[str, Any]] = (),
    ) -> "LifecycleResult":
        return cls(
            outcome=OperationOutcome.APPLIED,
            transition_applied=True,
            row=row,
            durable_events=durable_events,
        )

    @classmethod
    def no_transition(
        cls,
        reason: NoTransitionReason,
        *,
        row: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> "LifecycleResult":
        return cls(
            outcome=OperationOutcome.NO_TRANSITION,
            transition_applied=False,
            row=row,
            no_transition_reason=reason,
            message=message,
        )


__all__ = [
    "AdmissionRejectionReason",
    "AdmissionResult",
    "CreateJobCommand",
    "LifecycleResult",
    "NoTransitionReason",
    "OperationOutcome",
]
