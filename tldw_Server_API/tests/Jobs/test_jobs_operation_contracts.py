"""Contract tests for Jobs operation result dataclasses."""

from __future__ import annotations

import ast
import io
import tokenize
from dataclasses import FrozenInstanceError, fields
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from tldw_Server_API.app.core.Jobs.operations import contracts
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    AcquireJobCommand,
    AdminWebhookDispositionMarker,
    AdmissionRejectionReason,
    AdmissionResult,
    ApplyPreparedDispositionCommand,
    BatchRenewLeaseItem,
    BatchRenewLeasesCommand,
    BatchRenewLeasesResult,
    CreateJobCommand,
    EnsureLeaseHorizonCommand,
    ExpiredLeasePolicy,
    FindJobByIdentityCommand,
    JobIdentityLookupResult,
    JobIdentityLookupState,
    LeaseHorizonResult,
    LifecycleResult,
    NoTransitionReason,
    OperationOutcome,
    PreparedDispositionKind,
    PreparedDispositionOrigin,
    PreparedDispositionResult,
    PreparedJobDisposition,
    ReleaseJobCommand,
    RenewLeaseCommand,
    admin_webhook_disposition_marker_matches,
    prepared_disposition_fingerprint,
    project_admin_webhook_disposition_marker,
)

pytestmark = pytest.mark.unit


class _IntSubclass(int):
    pass


def _uuid() -> str:
    return str(uuid4())


def _token(character: str = "a") -> str:
    return character * 64


def _aware(seconds: int = 0) -> datetime:
    return datetime(2026, 8, 28, tzinfo=timezone.utc) + timedelta(seconds=seconds)


def test_prepared_disposition_factories_build_the_closed_protocol() -> None:
    delivery_id = _uuid()
    attempt_id = _uuid()
    retry_at = _aware(60)
    stale_at = _aware(120)

    complete = PreparedJobDisposition.complete(
        token=_token("a"), delivery_id=delivery_id, attempt_id=attempt_id
    )
    retry = PreparedJobDisposition.retry(
        token=_token("b"),
        delivery_id=delivery_id,
        attempt_id=attempt_id,
        delay_seconds=60,
        not_before_at=retry_at,
        reason_code="receiver_503",
    )
    fail = PreparedJobDisposition.fail(
        token=_token("c"),
        delivery_id=delivery_id,
        attempt_id=attempt_id,
        reason_code="receiver_400",
    )
    cancel = PreparedJobDisposition.cancel(
        token=_token("d"),
        delivery_id=delivery_id,
        reason_code="registration_disabled",
    )
    infrastructure = PreparedJobDisposition.infrastructure_defer(
        token=_token("e"),
        delivery_id=delivery_id,
        reason_code="authnz_unavailable",
    )
    recovery = PreparedJobDisposition.recovery_defer_until(
        token=_token("f"),
        delivery_id=delivery_id,
        not_before_at=stale_at,
        reason_code="attempt_not_stale",
    )

    assert complete.kind is PreparedDispositionKind.COMPLETE
    assert retry.kind is PreparedDispositionKind.RETRY
    assert retry.origin is PreparedDispositionOrigin.AUTHNZ
    assert retry.delay_seconds == 60
    assert retry.not_before_at == retry_at
    assert fail.kind is PreparedDispositionKind.FAIL
    assert cancel.kind is PreparedDispositionKind.CANCEL
    assert infrastructure == PreparedJobDisposition(
        token=_token("e"),
        kind=PreparedDispositionKind.DEFER,
        origin=PreparedDispositionOrigin.INFRASTRUCTURE,
        delivery_id=delivery_id,
        reason_code="authnz_unavailable",
    )
    assert infrastructure.not_before_at is None
    assert recovery.origin is PreparedDispositionOrigin.RECOVERY
    assert recovery.not_before_at == stale_at


def test_no_attempt_fail_is_limited_to_an_explicit_authnz_terminal_fact() -> None:
    disposition = PreparedJobDisposition.fail(
        token=_token("9"),
        delivery_id=_uuid(),
        attempt_id=None,
        reason_code="delivery_expired",
    )

    assert disposition.kind is PreparedDispositionKind.FAIL
    assert disposition.origin is PreparedDispositionOrigin.AUTHNZ
    assert disposition.attempt_id is None

    with pytest.raises(ValueError, match="no-attempt fail reason"):
        PreparedJobDisposition.fail(
            token=_token("8"),
            delivery_id=_uuid(),
            attempt_id=None,
            reason_code="http_client_error",
        )


def test_admin_webhook_marker_projection_is_strict_bounded_and_fingerprint_exact() -> None:
    delivery_id = _uuid()
    attempt_id = _uuid()
    disposition = PreparedJobDisposition.complete(
        token=_token("7"),
        delivery_id=delivery_id,
        attempt_id=attempt_id,
    )
    marker = {
        "schema_version": 1,
        "token": disposition.token,
        "kind": disposition.kind.value,
        "origin": disposition.origin.value,
        "delivery_id": delivery_id,
        "attempt_id": attempt_id,
        "applied_at": _aware().isoformat(),
    }
    row = {
        "uuid": _uuid(),
        "domain": "admin_webhooks",
        "queue": "delivery",
        "job_type": "admin_webhook_delivery",
        "payload": {"delivery_id": delivery_id},
        "idempotency_key": f"admin-webhook-delivery:{delivery_id}",
        "owner_user_id": None,
        "project_id": None,
        "batch_group": None,
        "priority": 5,
        "max_retries": 3,
        "status": "completed",
        "available_at": None,
        "result": marker,
        "prepared_disposition_fingerprint": prepared_disposition_fingerprint(
            disposition
        ),
        "no_attempt_recovery_fingerprint": None,
        "expired_lease_policy": "requeue_no_attempt",
        "quarantine_threshold": 5,
    }

    projected = project_admin_webhook_disposition_marker(
        row,
        expected_payload={"delivery_id": delivery_id},
    )

    assert isinstance(projected, AdminWebhookDispositionMarker)
    assert projected.attempt_id == attempt_id
    assert admin_webhook_disposition_marker_matches(projected, disposition)
    forged = PreparedJobDisposition.complete(
        token=disposition.token,
        delivery_id=delivery_id,
        attempt_id=_uuid(),
    )
    assert not admin_webhook_disposition_marker_matches(projected, forged)
    assert project_admin_webhook_disposition_marker(
        {**row, "result": {**marker, "reason_code": "not_public"}},
        expected_payload={"delivery_id": delivery_id},
    ) is None
    assert project_admin_webhook_disposition_marker(
        {**row, "result": "{" + "x" * 4096 + "}"},
        expected_payload={"delivery_id": delivery_id},
    ) is None


@pytest.mark.parametrize("token", ["a" * 63, "A" * 64, "g" * 64, "a" * 65])
def test_prepared_disposition_rejects_malformed_tokens(token: str) -> None:
    with pytest.raises(ValueError, match="token"):
        PreparedJobDisposition.complete(
            token=token, delivery_id=_uuid(), attempt_id=_uuid()
        )


@pytest.mark.parametrize("field_name", ["delivery_id", "attempt_id"])
def test_prepared_disposition_rejects_noncanonical_uuid4_ids(field_name: str) -> None:
    values = {
        "token": _token(),
        "delivery_id": _uuid(),
        "attempt_id": _uuid(),
    }
    values[field_name] = "not-a-canonical-uuid"
    with pytest.raises(ValueError, match=field_name):
        PreparedJobDisposition.complete(**values)


@pytest.mark.parametrize("delay", [0, -1, 1801, True])
def test_retry_rejects_unbounded_delay(delay: int) -> None:
    with pytest.raises(ValueError, match="delay_seconds"):
        PreparedJobDisposition.retry(
            token=_token(),
            delivery_id=_uuid(),
            attempt_id=_uuid(),
            delay_seconds=delay,
            not_before_at=_aware(),
            reason_code="receiver_503",
        )


def test_prepared_disposition_rejects_illegal_origin_and_field_combinations() -> None:
    common = {"token": _token(), "delivery_id": _uuid()}
    with pytest.raises(ValueError):
        PreparedJobDisposition(
            **common,
            kind=PreparedDispositionKind.COMPLETE,
            origin=PreparedDispositionOrigin.INFRASTRUCTURE,
            attempt_id=_uuid(),
        )
    with pytest.raises(TypeError):
        PreparedJobDisposition.infrastructure_defer(
            **common,
            reason_code="authnz_unavailable",
            not_before_at=_aware(),  # type: ignore[call-arg]
        )
    with pytest.raises(ValueError, match="timezone-aware"):
        PreparedJobDisposition.recovery_defer_until(
            **common,
            reason_code="attempt_not_stale",
            not_before_at=datetime(2026, 8, 28),
        )
    with pytest.raises(ValueError, match="reason_code"):
        PreparedJobDisposition.fail(
            **common,
            attempt_id=_uuid(),
            reason_code="arbitrary exception text with spaces and secrets",
        )


def test_prepared_commands_and_results_copy_mutable_payloads() -> None:
    payload = {"delivery_id": _uuid(), "nested": {"value": 1}}
    disposition = PreparedJobDisposition.cancel(
        token=_token(),
        delivery_id=payload["delivery_id"],
        reason_code="registration_disabled",
    )
    command = ApplyPreparedDispositionCommand(
        job_id=7,
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
        expected_payload=payload,
        disposition=disposition,
    )
    metadata = {"schema_version": 1, "token": _token(), "nested": {"value": 1}}
    result = PreparedDispositionResult.applied(
        state="cancelled", metadata=metadata, already_applied=False
    )
    payload["nested"]["value"] = 2
    metadata["nested"]["value"] = 2

    assert command.expected_payload["nested"] == {"value": 1}
    assert result.metadata["nested"] == {"value": 1}
    assert "message" not in {item.name for item in fields(PreparedDispositionResult)}
    with pytest.raises(FrozenInstanceError):
        command.job_id = 8


def test_lease_horizon_and_identity_contracts_are_closed_and_frozen() -> None:
    payload = {"delivery_id": _uuid()}
    horizon = EnsureLeaseHorizonCommand(
        job_id=1,
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
        expected_payload=payload,
        worker_id="worker-1",
        lease_id="lease-1",
        minimum_seconds=60,
    )
    lookup = FindJobByIdentityCommand(
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
        idempotency_key=f"admin-webhook-delivery:{payload['delivery_id']}",
        expected_payload=payload,
    )
    lookup_result = JobIdentityLookupResult.found(
        JobIdentityLookupState.ACTIVE, {"id": 1, "payload": payload}
    )
    lease_result = LeaseHorizonResult.applied(
        leased_until=_aware(60),
        guaranteed_seconds=60,
    )
    payload["delivery_id"] = _uuid()

    assert horizon.expected_payload != payload
    assert lookup.expected_payload != payload
    assert lookup_result.row["payload"] != payload
    assert lease_result.ensured is True
    assert lease_result.guaranteed_seconds == 60
    assert LeaseHorizonResult.no_transition(
        NoTransitionReason.STALE_LEASE
    ).guaranteed_seconds is None
    with pytest.raises(ValueError, match="minimum_seconds"):
        EnsureLeaseHorizonCommand(
            **{**horizon.__dict__, "minimum_seconds": 0}
        )


@pytest.mark.parametrize(
    "guaranteed_seconds",
    [None, 0, -1, True, 1.5, "60", _IntSubclass(60)],
    ids=["missing", "zero", "negative", "bool", "float", "string", "subclass"],
)
def test_applied_lease_horizon_requires_positive_exact_int_guarantee(
    guaranteed_seconds,
) -> None:
    with pytest.raises(ValueError, match="guaranteed_seconds"):
        LeaseHorizonResult(
            outcome=OperationOutcome.APPLIED,
            ensured=True,
            leased_until=_aware(60),
            guaranteed_seconds=guaranteed_seconds,
        )


def test_non_applied_lease_horizon_cannot_expose_guarantee() -> None:
    with pytest.raises(ValueError, match="guaranteed_seconds"):
        LeaseHorizonResult(
            outcome=OperationOutcome.BACKEND_CONFLICT,
            ensured=False,
            guaranteed_seconds=30,
        )


@pytest.mark.parametrize(
    "result_kwargs",
    [
        {
            "outcome": OperationOutcome.APPLIED,
            "ensured": True,
            "leased_until": _aware(60),
            "no_transition_reason": None,
            "guaranteed_seconds": 60,
        },
        {
            "outcome": OperationOutcome.NO_TRANSITION,
            "ensured": False,
            "leased_until": None,
            "no_transition_reason": NoTransitionReason.MISSING,
            "guaranteed_seconds": None,
        },
        {
            "outcome": OperationOutcome.NO_TRANSITION,
            "ensured": False,
            "leased_until": _aware(30),
            "no_transition_reason": NoTransitionReason.STALE_LEASE,
            "guaranteed_seconds": None,
        },
        {
            "outcome": OperationOutcome.BACKEND_CONFLICT,
            "ensured": False,
            "leased_until": None,
            "no_transition_reason": None,
            "guaranteed_seconds": None,
        },
        {
            "outcome": OperationOutcome.BACKEND_CONFLICT,
            "ensured": False,
            "leased_until": _aware(30),
            "no_transition_reason": None,
            "guaranteed_seconds": None,
        },
    ],
    ids=[
        "applied",
        "no-transition-without-deadline",
        "no-transition-with-deadline",
        "conflict-without-deadline",
        "conflict-with-deadline",
    ],
)
def test_lease_horizon_result_accepts_the_complete_valid_state_matrix(
    result_kwargs,
) -> None:
    result = LeaseHorizonResult(**result_kwargs)

    assert result.outcome is result_kwargs["outcome"]
    assert result.ensured is result_kwargs["ensured"]


@pytest.mark.parametrize(
    "result_kwargs",
    [
        {
            "outcome": OperationOutcome.BACKEND_ERROR,
            "ensured": False,
        },
        {
            "outcome": OperationOutcome.ADMISSION_REJECTED,
            "ensured": False,
        },
        {
            "outcome": "applied",
            "ensured": True,
            "leased_until": _aware(60),
            "guaranteed_seconds": 60,
        },
        {
            "outcome": OperationOutcome.APPLIED,
            "ensured": 1,
            "leased_until": _aware(60),
            "guaranteed_seconds": 60,
        },
        {
            "outcome": OperationOutcome.APPLIED,
            "ensured": False,
            "leased_until": _aware(60),
            "guaranteed_seconds": 60,
        },
        {
            "outcome": OperationOutcome.APPLIED,
            "ensured": True,
            "leased_until": None,
            "guaranteed_seconds": 60,
        },
        {
            "outcome": OperationOutcome.APPLIED,
            "ensured": True,
            "leased_until": datetime(2026, 8, 28),
            "guaranteed_seconds": 60,
        },
        {
            "outcome": OperationOutcome.APPLIED,
            "ensured": True,
            "leased_until": _aware(60),
            "no_transition_reason": NoTransitionReason.STALE_LEASE,
            "guaranteed_seconds": 60,
        },
        {
            "outcome": OperationOutcome.NO_TRANSITION,
            "ensured": 0,
            "no_transition_reason": NoTransitionReason.MISSING,
        },
        {
            "outcome": OperationOutcome.NO_TRANSITION,
            "ensured": True,
            "no_transition_reason": NoTransitionReason.MISSING,
        },
        {
            "outcome": OperationOutcome.NO_TRANSITION,
            "ensured": False,
            "no_transition_reason": None,
        },
        {
            "outcome": OperationOutcome.NO_TRANSITION,
            "ensured": False,
            "no_transition_reason": "missing",
        },
        {
            "outcome": OperationOutcome.NO_TRANSITION,
            "ensured": False,
            "leased_until": datetime(2026, 8, 28),
            "no_transition_reason": NoTransitionReason.MISSING,
        },
        {
            "outcome": OperationOutcome.NO_TRANSITION,
            "ensured": False,
            "no_transition_reason": NoTransitionReason.MISSING,
            "guaranteed_seconds": 30,
        },
        {
            "outcome": OperationOutcome.BACKEND_CONFLICT,
            "ensured": 0,
        },
        {
            "outcome": OperationOutcome.BACKEND_CONFLICT,
            "ensured": True,
        },
        {
            "outcome": OperationOutcome.BACKEND_CONFLICT,
            "ensured": False,
            "no_transition_reason": NoTransitionReason.MISSING,
        },
        {
            "outcome": OperationOutcome.BACKEND_CONFLICT,
            "ensured": False,
            "leased_until": datetime(2026, 8, 28),
        },
    ],
    ids=[
        "backend-error-outcome",
        "admission-rejected-outcome",
        "string-outcome",
        "applied-non-bool-ensured",
        "applied-not-ensured",
        "applied-missing-deadline",
        "applied-naive-deadline",
        "applied-with-reason",
        "no-transition-non-bool-ensured",
        "no-transition-ensured",
        "no-transition-missing-reason",
        "no-transition-string-reason",
        "no-transition-naive-deadline",
        "no-transition-with-guarantee",
        "conflict-non-bool-ensured",
        "conflict-ensured",
        "conflict-with-reason",
        "conflict-naive-deadline",
    ],
)
def test_lease_horizon_result_rejects_every_other_state_shape(
    result_kwargs,
) -> None:
    with pytest.raises(ValueError):
        LeaseHorizonResult(**result_kwargs)


def test_create_job_execution_controls_are_default_compatible_and_validated() -> None:
    default = CreateJobCommand(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={},
        owner_user_id=None,
    )
    canonical = CreateJobCommand(
        domain="admin_webhooks",
        queue="delivery",
        job_type="admin_webhook_delivery",
        payload={"delivery_id": _uuid()},
        owner_user_id=None,
        expired_lease_policy=ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT,
        quarantine_threshold=5,
    )

    assert default.expired_lease_policy is ExpiredLeasePolicy.CONSUME_RETRY
    assert default.quarantine_threshold is None
    assert canonical.expired_lease_policy is ExpiredLeasePolicy.REQUEUE_NO_ATTEMPT
    assert canonical.quarantine_threshold == 5
    with pytest.raises(ValueError, match="expired_lease_policy"):
        CreateJobCommand(
            domain="x", queue="default", job_type="x", payload={}, owner_user_id=None,
            expired_lease_policy="unknown",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="quarantine_threshold"):
        CreateJobCommand(
            domain="x", queue="default", job_type="x", payload={}, owner_user_id=None,
            quarantine_threshold=0,
        )


def test_prepared_operation_contracts_are_exported() -> None:
    assert {
        "ApplyPreparedDispositionCommand",
        "EnsureLeaseHorizonCommand",
        "ExpiredLeasePolicy",
        "FindJobByIdentityCommand",
        "JobIdentityLookupResult",
        "JobIdentityLookupState",
        "LeaseHorizonResult",
        "PreparedDispositionKind",
        "PreparedDispositionOrigin",
        "PreparedDispositionResult",
        "PreparedJobDisposition",
    }.issubset(contracts.__all__)


def test_create_job_command_carries_public_job_facts() -> None:
    """Verify create commands carry public job facts without transformation."""

    command = CreateJobCommand(
        domain="chatbooks",
        queue="default",
        job_type="export",
        payload={"a": 1},
        owner_user_id="user-1",
        idempotency_key="same",
        priority=5,
        max_retries=2,
        request_id="request-1",
        trace_id="trace-1",
    )

    assert command.domain == "chatbooks"
    assert command.queue == "default"
    assert command.payload == {"a": 1}
    assert command.request_id == "request-1"
    assert command.trace_id == "trace-1"


def test_admission_result_distinguishes_inserted_and_existing_rows() -> None:
    """Verify admission results distinguish inserted, existing, and rejected states."""

    inserted = AdmissionResult.applied(
        row={"id": 1, "status": "queued"},
        durable_events=({"event_type": "job.created"},),
    )
    existing = AdmissionResult.existing(row={"id": 1, "status": "queued"})
    rejected = AdmissionResult.rejected(AdmissionRejectionReason.QUEUE_PAUSED)

    assert inserted.outcome is OperationOutcome.APPLIED
    assert inserted.inserted is True
    assert existing.outcome is OperationOutcome.NO_TRANSITION
    assert existing.inserted is False
    assert existing.no_transition_reason is NoTransitionReason.IDEMPOTENT_EXISTING
    assert rejected.outcome is OperationOutcome.ADMISSION_REJECTED
    assert rejected.admission_rejection_reason is AdmissionRejectionReason.QUEUE_PAUSED


def test_admission_existing_can_report_idempotent_durable_event() -> None:
    """Verify idempotent create replays can report the persisted replay event."""

    result = AdmissionResult.existing(
        row={"id": 1, "status": "queued"},
        durable_events=({"event_type": "job.created", "attrs": {"idempotent": True}},),
    )

    assert result.outcome is OperationOutcome.NO_TRANSITION
    assert result.no_transition_reason is NoTransitionReason.IDEMPOTENT_EXISTING
    assert result.durable_events == ({"event_type": "job.created", "attrs": {"idempotent": True}},)


def test_admission_result_rejects_inconsistent_states() -> None:
    """Verify invalid admission result state combinations are rejected."""

    with pytest.raises(ValueError, match="applied admission"):
        AdmissionResult(outcome=OperationOutcome.APPLIED, was_inserted=False)

    with pytest.raises(ValueError, match="applied admission"):
        AdmissionResult(outcome=OperationOutcome.APPLIED, was_inserted=True)

    with pytest.raises(ValueError, match="no-transition admission"):
        AdmissionResult(outcome=OperationOutcome.NO_TRANSITION)

    with pytest.raises(ValueError, match="rejected admission"):
        AdmissionResult(outcome=OperationOutcome.ADMISSION_REJECTED)

    with pytest.raises(ValueError, match="only applied admission"):
        AdmissionResult(
            outcome=OperationOutcome.NO_TRANSITION,
            was_inserted=True,
            no_transition_reason=NoTransitionReason.IDEMPOTENT_EXISTING,
        )

    with pytest.raises(ValueError, match="durable events"):
        AdmissionResult(
            outcome=OperationOutcome.NO_TRANSITION,
            no_transition_reason=NoTransitionReason.MISSING,
            durable_events=({"event_type": "job.noop"},),
        )


def test_admission_result_copies_mutable_facts() -> None:
    """Verify admission results deep-copy nested mutable facts."""

    row = {"id": 1, "status": "queued", "payload": {"topic": "original"}}
    event = {"event_type": "job.created", "attrs": {"attempt": 1}}
    events = [event]

    result = AdmissionResult.applied(row=row, durable_events=events)
    row["status"] = "failed"
    row["payload"]["topic"] = "mutated"
    event["event_type"] = "job.failed"
    event["attrs"]["attempt"] = 2
    events.append({"event_type": "job.completed"})

    assert result.row == {"id": 1, "status": "queued", "payload": {"topic": "original"}}
    assert result.durable_events == ({"event_type": "job.created", "attrs": {"attempt": 1}},)


def test_lifecycle_result_names_no_transition_reason() -> None:
    """Verify lifecycle no-transition results retain their reason."""

    result = LifecycleResult.no_transition(
        NoTransitionReason.STALE_LEASE,
        row={"id": 1, "status": "processing"},
    )

    assert result.outcome is OperationOutcome.NO_TRANSITION
    assert result.no_transition_reason is NoTransitionReason.STALE_LEASE
    assert result.transition_applied is False


def test_acquire_job_command_preserves_all_public_job_facts() -> None:
    """Verify acquisition commands preserve every caller-provided field."""

    command = AcquireJobCommand(
        domain="chatbooks",
        queue="priority",
        lease_seconds=45,
        worker_id="worker-1",
        lease_id="lease-1",
        owner_user_id="user-1",
        job_type="export",
        max_inflight_quota=3,
        priority_direction="DESC",
        tie_break="lifo",
        single_update=True,
    )

    assert command.domain == "chatbooks"
    assert command.queue == "priority"
    assert command.lease_seconds == 45
    assert command.worker_id == "worker-1"
    assert command.lease_id == "lease-1"
    assert command.owner_user_id == "user-1"
    assert command.job_type == "export"
    assert command.max_inflight_quota == 3
    assert command.priority_direction == "DESC"
    assert command.tie_break == "lifo"
    assert command.single_update is True


def test_acquire_job_command_is_frozen() -> None:
    """Verify acquisition command fields cannot be reassigned."""

    command = AcquireJobCommand(
        domain="chatbooks",
        queue="default",
        lease_seconds=30,
        worker_id="worker-1",
        lease_id="lease-1",
    )

    with pytest.raises(FrozenInstanceError):
        command.queue = "other"


def test_acquire_job_command_defaults_to_backend_ordering() -> None:
    """Verify omitted tie-breaking delegates ordering to the backend default."""

    command = AcquireJobCommand(
        domain="chatbooks",
        queue="default",
        lease_seconds=30,
        worker_id="worker-1",
        lease_id="lease-1",
    )

    assert command.tie_break is None


def test_acquire_job_command_rejects_invalid_ordering_values() -> None:
    """Verify acquisition ordering controls accept only documented values."""

    with pytest.raises(ValueError, match="priority_direction must be ASC or DESC"):
        AcquireJobCommand(
            domain="chatbooks",
            queue="default",
            lease_seconds=30,
            worker_id="worker-1",
            lease_id="lease-1",
            priority_direction="invalid",
        )

    with pytest.raises(ValueError, match="tie_break must be fifo, lifo, or None"):
        AcquireJobCommand(
            domain="chatbooks",
            queue="default",
            lease_seconds=30,
            worker_id="worker-1",
            lease_id="lease-1",
            tie_break="invalid",
        )


def test_acquire_job_command_rejects_non_positive_lease_duration() -> None:
    """Verify acquisition leases must have a positive duration."""

    for lease_seconds in (0, -1):
        with pytest.raises(ValueError, match="lease_seconds must be positive"):
            AcquireJobCommand(
                domain="chatbooks",
                queue="default",
                lease_seconds=lease_seconds,
                worker_id="worker-1",
                lease_id="lease-1",
            )


def test_renew_lease_command_preserves_exact_public_facts() -> None:
    """Verify renewal commands retain every normalized facade input."""

    command = RenewLeaseCommand(
        job_id=42,
        seconds=30,
        enforce=True,
        worker_id="worker-1",
        lease_id="lease-1",
        progress_percent=62.5,
        progress_message="indexing",
    )

    assert [item.name for item in fields(command)] == [
        "job_id",
        "seconds",
        "enforce",
        "worker_id",
        "lease_id",
        "progress_percent",
        "progress_message",
    ]
    assert command.job_id == 42
    assert command.seconds == 30
    assert command.enforce is True
    assert command.worker_id == "worker-1"
    assert command.lease_id == "lease-1"
    assert command.progress_percent == 62.5
    assert command.progress_message == "indexing"


def test_renew_lease_command_requires_explicit_enforcement_policy() -> None:
    """Verify renewal callers cannot omit the resolved enforcement policy."""

    with pytest.raises(TypeError, match="enforce"):
        RenewLeaseCommand(job_id=42, seconds=30)  # type: ignore[call-arg]


def test_renew_lease_command_rejects_non_positive_duration() -> None:
    """Verify backend renewal commands always carry a positive duration."""

    for seconds in (0, -1):
        with pytest.raises(ValueError, match="seconds must be positive"):
            RenewLeaseCommand(job_id=42, seconds=seconds, enforce=False)


def test_renew_lease_command_is_frozen() -> None:
    """Verify renewal command fields cannot be reassigned."""

    command = RenewLeaseCommand(job_id=42, seconds=30, enforce=False)

    with pytest.raises(FrozenInstanceError):
        command.seconds = 60


def test_release_job_command_preserves_exact_public_facts() -> None:
    """Verify release commands retain every normalized facade input."""

    command = ReleaseJobCommand(
        job_id=42,
        enforce=True,
        worker_id="worker-1",
        lease_id="lease-1",
        reason="yield",
    )

    assert [item.name for item in fields(command)] == [
        "job_id",
        "enforce",
        "worker_id",
        "lease_id",
        "reason",
    ]
    assert command.job_id == 42
    assert command.enforce is True
    assert command.worker_id == "worker-1"
    assert command.lease_id == "lease-1"
    assert command.reason == "yield"


def test_release_job_command_requires_explicit_enforcement_policy() -> None:
    """Verify release callers cannot omit the resolved enforcement policy."""

    with pytest.raises(TypeError, match="enforce"):
        ReleaseJobCommand(job_id=42)  # type: ignore[call-arg]


def test_release_job_command_is_frozen() -> None:
    """Verify release command fields cannot be reassigned."""

    command = ReleaseJobCommand(job_id=42, enforce=False)

    with pytest.raises(FrozenInstanceError):
        command.reason = "other"


def test_batch_renew_command_snapshots_items_and_is_frozen() -> None:
    source = [BatchRenewLeaseItem(job_id=1, seconds=30)]
    command = BatchRenewLeasesCommand(items=source, enforce=True)  # type: ignore[arg-type]
    source.append(BatchRenewLeaseItem(job_id=2, seconds=45))

    assert command.items == (BatchRenewLeaseItem(job_id=1, seconds=30),)
    with pytest.raises(FrozenInstanceError):
        command.enforce = False


def test_batch_renew_item_is_frozen_and_preserves_order() -> None:
    first = BatchRenewLeaseItem(job_id=1, seconds=30)
    second = BatchRenewLeaseItem(job_id=2, seconds=45)
    command = BatchRenewLeasesCommand(items=(first, second), enforce=False)

    assert first.worker_id is None
    assert command.items == (first, second)
    with pytest.raises(FrozenInstanceError):
        first.seconds = 60


@pytest.mark.parametrize("seconds", [0, -1])
def test_batch_renew_item_rejects_non_positive_normalized_duration(seconds: int) -> None:
    with pytest.raises(ValueError, match="seconds must be positive"):
        BatchRenewLeaseItem(job_id=1, seconds=seconds)


@pytest.mark.parametrize(
    ("requested", "applied"),
    [(-1, 0), (0, -1), (1, 2)],
)
def test_batch_renew_result_rejects_invalid_counts(requested: int, applied: int) -> None:
    with pytest.raises(ValueError):
        BatchRenewLeasesResult(requested_count=requested, applied_count=applied)


def test_batch_renew_result_accepts_zero_and_complete_counts() -> None:
    assert BatchRenewLeasesResult(3, 0).applied_count == 0
    assert BatchRenewLeasesResult(3, 3).applied_count == 3


def test_batch_renew_contracts_are_exported() -> None:
    assert {
        "BatchRenewLeaseItem",
        "BatchRenewLeasesCommand",
        "BatchRenewLeasesResult",
    }.issubset(contracts.__all__)


def test_lifecycle_result_supports_no_eligible_job_reason() -> None:
    """Verify acquisition can report that no eligible job was available."""

    result = LifecycleResult.no_transition(NoTransitionReason.NO_ELIGIBLE_JOB)

    assert result.outcome is OperationOutcome.NO_TRANSITION
    assert result.no_transition_reason is NoTransitionReason.NO_ELIGIBLE_JOB


def test_lifecycle_result_rejects_inconsistent_states() -> None:
    """Verify invalid lifecycle result state combinations are rejected."""

    with pytest.raises(ValueError, match="applied lifecycle"):
        LifecycleResult(outcome=OperationOutcome.APPLIED, transition_applied=False)

    with pytest.raises(ValueError, match="applied lifecycle"):
        LifecycleResult(outcome=OperationOutcome.APPLIED, transition_applied=True)

    with pytest.raises(ValueError, match="no-transition lifecycle"):
        LifecycleResult(outcome=OperationOutcome.NO_TRANSITION, transition_applied=False)

    with pytest.raises(ValueError, match="only applied lifecycle"):
        LifecycleResult(
            outcome=OperationOutcome.NO_TRANSITION,
            transition_applied=True,
            no_transition_reason=NoTransitionReason.STALE_LEASE,
        )

    with pytest.raises(ValueError, match="only applied lifecycle"):
        LifecycleResult(
            outcome=OperationOutcome.NO_TRANSITION,
            transition_applied=False,
            no_transition_reason=NoTransitionReason.STALE_LEASE,
            durable_events=({"event_type": "job.noop"},),
        )


def test_lifecycle_result_copies_mutable_facts() -> None:
    """Verify lifecycle results deep-copy nested mutable facts."""

    row = {"id": 1, "status": "processing", "result": {"ok": True}}
    event = {"event_type": "job.completed", "attrs": {"attempt": 1}}
    events = [event]

    result = LifecycleResult.applied(row=row, durable_events=events)
    row["status"] = "failed"
    row["result"]["ok"] = False
    event["event_type"] = "job.failed"
    event["attrs"]["attempt"] = 2
    events.append({"event_type": "job.cancelled"})

    assert result.row == {"id": 1, "status": "processing", "result": {"ok": True}}
    assert result.durable_events == ({"event_type": "job.completed", "attrs": {"attempt": 1}},)


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_acquire_sql_does_not_interpolate_query_fragments(backend: str) -> None:
    """Verify acquisition SQL uses fixed fragments for ordering and candidate queries."""

    path = (
        Path(__file__).resolve().parents[3]
        / "tldw_Server_API/app/core/Jobs/operations"
        / backend
        / "lifecycle.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    formatted: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            expressions = [value.value for value in node.values if isinstance(value, ast.FormattedValue)]
            lease_parameter_only = expressions and all(
                isinstance(expression, ast.Attribute)
                and isinstance(expression.value, ast.Name)
                and expression.value.id == "command"
                and expression.attr == "lease_seconds"
                for expression in expressions
            )
            if not lease_parameter_only:
                formatted.append((node.lineno, "f-string"))
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"format", "format_map"}
        ):
            formatted.append((node.lineno, node.func.attr))
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod):
            formatted.append((node.lineno, "percent-format"))

    assert formatted == []


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_acquire_sql_has_no_bandit_query_suppressions(backend: str) -> None:
    """Verify acquisition SQL does not rely on Bandit query suppressions."""

    path = (
        Path(__file__).resolve().parents[3]
        / "tldw_Server_API/app/core/Jobs/operations"
        / backend
        / "lifecycle.py"
    )

    source = path.read_text(encoding="utf-8")
    comments = [
        token.string
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.COMMENT and "nosec" in token.string.casefold()
    ]

    assert comments == []


def test_operation_contracts_do_not_import_job_manager() -> None:
    """Verify operation contracts do not depend on the JobManager facade."""

    operations_dir = Path(__file__).resolve().parents[3] / "tldw_Server_API/app/core/Jobs/operations"
    assert operations_dir.exists()
    forbidden_refs: list[str] = []
    scanned_files = 0

    for path in operations_dir.rglob("*.py"):
        scanned_files += 1
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[-1] == "manager" or alias.name.endswith(".Jobs.manager"):
                        forbidden_refs.append(f"{path}:{alias.name}")
                    if alias.asname in {"manager", "JobManager"}:
                        forbidden_refs.append(f"{path}:{alias.asname}")
            if isinstance(node, ast.ImportFrom):
                if node.module and (node.module.split(".")[-1] == "manager" or node.module.endswith(".Jobs.manager")):
                    forbidden_refs.append(f"{path}:{node.module}")
                for alias in node.names:
                    if alias.name in {"manager", "JobManager"}:
                        forbidden_refs.append(f"{path}:{alias.name}")
                    if alias.asname in {"manager", "JobManager"}:
                        forbidden_refs.append(f"{path}:{alias.asname}")
            if isinstance(node, ast.Name):
                if node.id == "JobManager":
                    forbidden_refs.append(f"{path}:{node.id}")
            if isinstance(node, ast.Attribute):
                if node.attr in {"manager", "JobManager"}:
                    forbidden_refs.append(f"{path}:{node.attr}")

    assert scanned_files > 0
    assert forbidden_refs == []
