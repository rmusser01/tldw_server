"""Typed operation contracts for future Jobs backend extraction."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class OperationOutcome(str, Enum):
    """High-level result categories returned by backend operations."""

    APPLIED = "applied"
    NO_TRANSITION = "no_transition"
    ADMISSION_REJECTED = "admission_rejected"
    BACKEND_CONFLICT = "backend_conflict"
    BACKEND_SCHEMA_ERROR = "backend_schema_error"
    BACKEND_ERROR = "backend_error"


class NoTransitionReason(str, Enum):
    """Reasons a lifecycle operation made no durable state transition."""

    MISSING = "missing"
    WRONG_STATUS = "wrong_status"
    STALE_LEASE = "stale_lease"
    ALREADY_TERMINAL = "already_terminal"
    IDEMPOTENT_EXISTING = "idempotent_existing"
    RLS_FILTERED = "rls_filtered"
    NO_ELIGIBLE_JOB = "no_eligible_job"


class AdmissionRejectionReason(str, Enum):
    """Reasons a create/admission request can be rejected before insertion."""

    QUEUE_PAUSED = "queue_paused"
    QUEUE_DRAINING = "queue_draining"
    QUOTA_EXCEEDED = "quota_exceeded"
    FAIR_SHARE_LIMIT = "fair_share_limit"
    DEPENDENCY_BLOCKED = "dependency_blocked"
    POLICY_REJECTED = "policy_rejected"


@dataclass(frozen=True)
class CreateJobCommand:
    """Backend-neutral command payload for creating a Jobs row."""

    domain: str
    queue: str
    job_type: str
    payload: dict[str, Any] | None
    owner_user_id: str | None
    idempotency_key: str | None = None
    priority: int = 100
    max_retries: int = 3
    available_at: datetime | None = None
    project_id: int | str | None = None
    batch_group: str | None = None
    request_id: str | None = None
    trace_id: str | None = None


@dataclass(frozen=True)
class AcquireJobCommand:
    """Backend-neutral command payload for acquiring one eligible Jobs row."""

    domain: str
    queue: str
    lease_seconds: int
    worker_id: str
    lease_id: str
    owner_user_id: str | None = None
    job_type: str | None = None
    max_inflight_quota: int = 0
    priority_direction: str = "ASC"
    tie_break: str | None = None
    single_update: bool = False

    def __post_init__(self) -> None:
        if self.priority_direction not in {"ASC", "DESC"}:
            raise ValueError("priority_direction must be ASC or DESC")
        if self.tie_break not in {None, "fifo", "lifo"}:
            raise ValueError("tie_break must be fifo, lifo, or None")
        if self.lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")


@dataclass(frozen=True)
class RenewLeaseCommand:
    """Backend-neutral command payload for renewing one job lease."""

    job_id: int
    seconds: int
    enforce: bool
    worker_id: str | None = None
    lease_id: str | None = None
    progress_percent: float | None = None
    progress_message: str | None = None

    def __post_init__(self) -> None:
        """Reject non-positive lease extension durations."""

        if self.seconds < 1:
            raise ValueError("seconds must be positive")


@dataclass(frozen=True)
class ReleaseJobCommand:
    """Backend-neutral command payload for releasing one processing job."""

    job_id: int
    enforce: bool
    worker_id: str | None = None
    lease_id: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class AdmissionResult:
    """Result facts produced by a create/admission operation."""

    outcome: OperationOutcome
    row: dict[str, Any] | None = None
    was_inserted: bool = False
    no_transition_reason: NoTransitionReason | None = None
    admission_rejection_reason: AdmissionRejectionReason | None = None
    durable_events: Sequence[dict[str, Any]] = field(default_factory=tuple)
    message: str | None = None

    def __post_init__(self) -> None:
        """Validate admission invariants and freeze mutable facts."""

        if self.outcome is OperationOutcome.APPLIED and not self.was_inserted:
            raise ValueError("applied admission results must mark was_inserted")
        if self.outcome is OperationOutcome.APPLIED and self.row is None:
            raise ValueError("applied admission results require a row")
        if self.outcome is OperationOutcome.NO_TRANSITION and self.no_transition_reason is None:
            raise ValueError("no-transition admission results require a reason")
        if self.outcome is OperationOutcome.ADMISSION_REJECTED and self.admission_rejection_reason is None:
            raise ValueError("rejected admission results require a rejection reason")
        if self.outcome is not OperationOutcome.APPLIED and self.was_inserted:
            raise ValueError("only applied admission results may mark was_inserted")
        if self.outcome is not OperationOutcome.NO_TRANSITION and self.no_transition_reason is not None:
            raise ValueError("only no-transition admission results may include a no-transition reason")
        if self.outcome is not OperationOutcome.ADMISSION_REJECTED and self.admission_rejection_reason is not None:
            raise ValueError("only rejected admission results may include a rejection reason")
        if (
            self.no_transition_reason is NoTransitionReason.IDEMPOTENT_EXISTING
            and self.row is None
        ):
            raise ValueError("idempotent-existing admission results require a row")
        can_include_durable_events = self.outcome is OperationOutcome.APPLIED or (
            self.outcome is OperationOutcome.NO_TRANSITION
            and self.no_transition_reason is NoTransitionReason.IDEMPOTENT_EXISTING
        )
        if not can_include_durable_events and self.durable_events:
            raise ValueError("only applied or idempotent-existing admission results may include durable events")
        object.__setattr__(self, "row", copy.deepcopy(self.row) if self.row is not None else None)
        object.__setattr__(self, "durable_events", tuple(copy.deepcopy(event) for event in self.durable_events))

    @property
    def inserted(self) -> bool:
        """Return whether the admission inserted a new row."""

        return self.was_inserted

    @classmethod
    def applied(cls, *, row: dict[str, Any], durable_events: Sequence[dict[str, Any]] = ()) -> AdmissionResult:
        """Build an applied admission result for a newly inserted row."""

        return cls(
            outcome=OperationOutcome.APPLIED,
            row=row,
            was_inserted=True,
            durable_events=durable_events,
        )

    @classmethod
    def existing(
        cls,
        *,
        row: dict[str, Any],
        durable_events: Sequence[dict[str, Any]] = (),
    ) -> AdmissionResult:
        """Build a no-transition result for an idempotent existing row."""

        return cls(
            outcome=OperationOutcome.NO_TRANSITION,
            row=row,
            was_inserted=False,
            no_transition_reason=NoTransitionReason.IDEMPOTENT_EXISTING,
            durable_events=durable_events,
        )

    @classmethod
    def rejected(cls, reason: AdmissionRejectionReason, *, message: str | None = None) -> AdmissionResult:
        """Build an admission-rejected result with an explicit reason."""

        return cls(
            outcome=OperationOutcome.ADMISSION_REJECTED,
            admission_rejection_reason=reason,
            message=message,
        )


@dataclass(frozen=True)
class LifecycleResult:
    """Result facts produced by a Jobs lifecycle state transition."""

    outcome: OperationOutcome
    transition_applied: bool
    row: dict[str, Any] | None = None
    no_transition_reason: NoTransitionReason | None = None
    durable_events: Sequence[dict[str, Any]] = field(default_factory=tuple)
    message: str | None = None

    def __post_init__(self) -> None:
        """Validate lifecycle invariants and freeze mutable facts."""

        if self.outcome is OperationOutcome.APPLIED and not self.transition_applied:
            raise ValueError("applied lifecycle results must mark transition_applied")
        if self.outcome is OperationOutcome.APPLIED and self.row is None:
            raise ValueError("applied lifecycle results require a row")
        if self.outcome is OperationOutcome.NO_TRANSITION and self.no_transition_reason is None:
            raise ValueError("no-transition lifecycle results require a reason")
        if self.outcome is not OperationOutcome.APPLIED and self.transition_applied:
            raise ValueError("only applied lifecycle results may mark transition_applied")
        if self.outcome is not OperationOutcome.NO_TRANSITION and self.no_transition_reason is not None:
            raise ValueError("only no-transition lifecycle results may include a no-transition reason")
        if self.outcome is not OperationOutcome.APPLIED and self.durable_events:
            raise ValueError("only applied lifecycle results may include durable events")
        object.__setattr__(self, "row", copy.deepcopy(self.row) if self.row is not None else None)
        object.__setattr__(self, "durable_events", tuple(copy.deepcopy(event) for event in self.durable_events))

    @classmethod
    def applied(
        cls,
        *,
        row: dict[str, Any],
        durable_events: Sequence[dict[str, Any]] = (),
    ) -> LifecycleResult:
        """Build an applied lifecycle result for a successful transition."""

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
    ) -> LifecycleResult:
        """Build a no-transition lifecycle result with a reason."""

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
    "AcquireJobCommand",
    "CreateJobCommand",
    "LifecycleResult",
    "NoTransitionReason",
    "OperationOutcome",
    "ReleaseJobCommand",
    "RenewLeaseCommand",
]
