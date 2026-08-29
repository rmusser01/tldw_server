from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio

from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    DeliveryKind,
    DeliveryReasonCode,
    DeliveryState,
)
from tldw_Server_API.app.core.Admin_Webhooks.reconciler import (
    AdminWebhookReconciler,
    EnqueueFailureKind,
    JobsDeliveryAdmission,
    JobsDeliveryConflictError,
    JobsDeliveryQueue,
    JobsDeliveryRecord,
)
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
)
from tldw_Server_API.app.core.Jobs.operations.contracts import (
    AdmissionRejectionReason,
    AdmissionResult,
    JobIdentityLookupResult,
    JobIdentityLookupState,
    NoTransitionReason,
    OperationOutcome,
    PreparedDispositionResult,
)
from tldw_Server_API.tests.Admin_Webhooks.test_event_expansion import (
    NOW,
    canonical_uuid4,
    event_insert,
    opaque_token,
    seed_registration,
)

pytestmark = pytest.mark.unit


def _canonical_jobs_row(
    delivery_id: str,
    *,
    jobs_job_id: int = 17,
    status: str = "queued",
) -> dict[str, object]:
    return {
        "id": jobs_job_id,
        "uuid": canonical_uuid4(f"jobs-{jobs_job_id}"),
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
        "available_at": None,
        "status": status,
        "result": None,
        "prepared_disposition_fingerprint": None,
        "no_attempt_recovery_fingerprint": None,
        "expired_lease_policy": "requeue_no_attempt",
        "quarantine_threshold": 5,
    }


class StubJobManager:
    def __init__(self, delivery_id: str) -> None:
        self.row = _canonical_jobs_row(delivery_id)
        self.admission = AdmissionResult.applied(row=self.row)
        self.lookup = JobIdentityLookupResult.found(
            JobIdentityLookupState.ACTIVE,
            self.row,
        )
        self.prepared_result = PreparedDispositionResult.applied(
            state="cancelled",
            metadata={"kind": "cancel"},
            already_applied=False,
        )
        self.admit_kwargs: dict[str, object] | None = None
        self.lookup_command = None
        self.get_job_id: int | None = None
        self.disposition_command = None

    def admit_job(self, **kwargs):
        self.admit_kwargs = kwargs
        return self.admission

    def find_job_by_identity(self, command):
        self.lookup_command = command
        return self.lookup

    def get_job(self, job_id: int):
        self.get_job_id = job_id
        return self.row

    def apply_prepared_disposition(self, command):
        self.disposition_command = command
        return self.prepared_result


def test_jobs_delivery_queue_uses_only_fixed_canonical_admission_facts() -> None:
    delivery_id = canonical_uuid4("adapter-admission")
    manager = StubJobManager(delivery_id)
    queue = JobsDeliveryQueue(manager)

    admission = queue.admit_delivery_job(
        delivery_id,
        NOW + timedelta(hours=72),
    )

    assert admission.outcome is OperationOutcome.APPLIED
    assert admission.record == JobsDeliveryRecord(
        jobs_job_id="17",
        delivery_id=delivery_id,
        status="queued",
        archived=False,
    )
    assert manager.admit_kwargs == {
        "domain": "admin_webhooks",
        "queue": "delivery",
        "job_type": "admin_webhook_delivery",
        "payload": {"delivery_id": delivery_id},
        "owner_user_id": None,
        "project_id": None,
        "batch_group": None,
        "priority": 5,
        "max_retries": 3,
        "available_at": None,
        "idempotency_key": f"admin-webhook-delivery:{delivery_id}",
        "request_id": None,
        "trace_id": None,
        "expired_lease_policy": "requeue_no_attempt",
        "quarantine_threshold": 5,
    }


def test_jobs_delivery_queue_turns_idempotent_row_mismatch_into_typed_conflict() -> None:
    delivery_id = canonical_uuid4("adapter-mismatch")
    manager = StubJobManager(delivery_id)
    mismatched = {**manager.row, "max_retries": 2}
    manager.admission = AdmissionResult.existing(row=mismatched)
    queue = JobsDeliveryQueue(manager)

    admission = queue.admit_delivery_job(
        delivery_id,
        NOW + timedelta(hours=72),
    )

    assert admission.outcome is OperationOutcome.BACKEND_CONFLICT
    assert admission.record is None
    assert admission.admission_rejection_reason is None


def test_jobs_delivery_queue_lookup_and_known_id_reads_fail_closed() -> None:
    delivery_id = canonical_uuid4("adapter-lookup")
    manager = StubJobManager(delivery_id)
    queue = JobsDeliveryQueue(manager)

    found = queue.find_delivery_job_by_identity(delivery_id)
    known = queue.get_delivery_job("17")

    assert found == known == JobsDeliveryRecord("17", delivery_id, "queued", False)
    assert manager.lookup_command.domain == "admin_webhooks"
    assert manager.lookup_command.queue == "delivery"
    assert manager.lookup_command.job_type == "admin_webhook_delivery"
    assert manager.lookup_command.expected_payload == {"delivery_id": delivery_id}
    assert (
        manager.lookup_command.idempotency_key
        == f"admin-webhook-delivery:{delivery_id}"
    )
    assert manager.get_job_id == 17

    manager.lookup = JobIdentityLookupResult.conflict()
    with pytest.raises(JobsDeliveryConflictError):
        queue.find_delivery_job_by_identity(delivery_id)

    manager.lookup = JobIdentityLookupResult.missing()
    assert queue.find_delivery_job_by_identity(delivery_id) is None

    manager.row = {**manager.row, "payload": {"delivery_id": canonical_uuid4("other")}}
    with pytest.raises(JobsDeliveryConflictError):
        queue.get_delivery_job("17")

    other_delivery_id = canonical_uuid4("known-other")
    manager.row = _canonical_jobs_row(other_delivery_id)
    assert queue.get_delivery_job("17").delivery_id == other_delivery_id


def test_jobs_delivery_queue_applies_only_tokenized_unleased_cancel() -> None:
    delivery_id = canonical_uuid4("adapter-cancel")
    manager = StubJobManager(delivery_id)
    queue = JobsDeliveryQueue(manager)
    token = opaque_token("adapter-cancel-token")

    result = queue.apply_queued_cancel(
        "17",
        delivery_id,
        token,
        DeliveryReasonCode.CANCELED_DISABLED,
    )

    assert result is manager.prepared_result
    command = manager.disposition_command
    assert command.job_id == 17
    assert command.domain == "admin_webhooks"
    assert command.queue == "delivery"
    assert command.job_type == "admin_webhook_delivery"
    assert command.expected_payload == {"delivery_id": delivery_id}
    assert command.worker_id is None
    assert command.lease_id is None
    assert command.disposition.token == token
    assert command.disposition.delivery_id == delivery_id
    assert command.disposition.reason_code == "canceled_disabled"


@dataclass
class SQLiteAuthFixture:
    repository: AdminWebhookRepository
    pool: DatabasePool


@pytest_asyncio.fixture
async def auth_fixture(tmp_path: Path) -> SQLiteAuthFixture:
    pool = DatabasePool(
        Settings(
            AUTH_MODE="single_user",
            DATABASE_URL=f"sqlite:///{tmp_path / 'enqueue-auth.db'}",
        )
    )
    await pool.initialize()
    fixture = SQLiteAuthFixture(AdminWebhookRepository(pool), pool)
    try:
        yield fixture
    finally:
        await pool.close()


class MutableClock:
    def __init__(self, now: datetime = NOW) -> None:
        self.current = now

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: int) -> None:
        self.current += timedelta(seconds=seconds)


class TokenSource:
    def __init__(self) -> None:
        self.index = 0

    def __call__(self) -> str:
        token = hashlib.sha256(f"enqueue-token-{self.index}".encode()).hexdigest()
        self.index += 1
        return token


class StubDeliveryQueue:
    def __init__(self) -> None:
        self.admission: JobsDeliveryAdmission | BaseException | None = None
        self.identity: JobsDeliveryRecord | None | BaseException = None
        self.records: dict[str, JobsDeliveryRecord] = {}
        self.admit_calls: list[tuple[str, datetime]] = []
        self.find_calls: list[str] = []
        self.get_calls: list[str] = []
        self.cancel_calls: list[tuple[str, str, str, DeliveryReasonCode]] = []
        self.cancel_result = PreparedDispositionResult.applied(
            state="cancelled",
            metadata={"kind": "cancel"},
            already_applied=False,
        )

    def admit_delivery_job(
        self,
        delivery_id: str,
        expires_at: datetime,
    ) -> JobsDeliveryAdmission:
        self.admit_calls.append((delivery_id, expires_at))
        if isinstance(self.admission, BaseException):
            raise self.admission
        if self.admission is not None:
            return self.admission
        record = JobsDeliveryRecord(
            jobs_job_id=str(len(self.records) + 1),
            delivery_id=delivery_id,
            status="queued",
            archived=False,
        )
        self.records[record.jobs_job_id] = record
        return JobsDeliveryAdmission(
            outcome=OperationOutcome.APPLIED,
            record=record,
        )

    def find_delivery_job_by_identity(
        self,
        delivery_id: str,
    ) -> JobsDeliveryRecord | None:
        self.find_calls.append(delivery_id)
        if isinstance(self.identity, BaseException):
            raise self.identity
        return self.identity

    def get_delivery_job(self, jobs_job_id: str) -> JobsDeliveryRecord | None:
        self.get_calls.append(jobs_job_id)
        return self.records.get(jobs_job_id)

    def apply_queued_cancel(
        self,
        jobs_job_id: str,
        delivery_id: str,
        disposition_token: str,
        reason_code: DeliveryReasonCode,
    ) -> PreparedDispositionResult:
        self.cancel_calls.append(
            (jobs_job_id, delivery_id, disposition_token, reason_code)
        )
        return self.cancel_result


async def _seed_automatic_delivery(
    repository: AdminWebhookRepository,
    label: str,
    *,
    expires_at: datetime = NOW + timedelta(hours=72),
) -> tuple[int, str]:
    event_type = f"enqueue.{label}"
    webhook_id = await seed_registration(repository, event_types=(event_type,))
    delivery_id = canonical_uuid4(f"enqueue-{label}-delivery")
    created_at = expires_at - timedelta(hours=72)
    async with repository.transaction() as tx:
        captured = await tx.capture_event_and_expand(
            event_insert(
                event_id=canonical_uuid4(f"enqueue-{label}-event"),
                source_identity=f"enqueue-{label}-command",
                event_type=event_type,
                created_at=created_at,
            ),
            lambda: delivery_id,
            expires_at,
        )
    assert len(captured.deliveries) == 1
    return webhook_id, delivery_id


def _reconciler(
    fixture: SQLiteAuthFixture,
    queue: StubDeliveryQueue,
    clock: MutableClock,
    *,
    observer=lambda _failure: None,
) -> AdminWebhookReconciler:
    return AdminWebhookReconciler(
        repository=fixture.repository,
        queue=queue,
        token_factory=TokenSource(),
        clock=clock,
        claim_ttl_seconds=60,
        failure_observer=observer,
    )


async def test_typed_rejection_releases_only_owned_claim_and_retries_later(
    auth_fixture: SQLiteAuthFixture,
) -> None:
    webhook_id, delivery_id = await _seed_automatic_delivery(
        auth_fixture.repository,
        "typed-rejection",
    )
    queue = StubDeliveryQueue()
    queue.admission = JobsDeliveryAdmission(
        outcome=OperationOutcome.ADMISSION_REJECTED,
        admission_rejection_reason=AdmissionRejectionReason.POLICY_REJECTED,
    )
    clock = MutableClock()
    observed: list[EnqueueFailureKind] = []
    reconciler = _reconciler(auth_fixture, queue, clock, observer=observed.append)

    assert await reconciler.reconcile_enqueue_once() == 1
    first = await auth_fixture.repository.get_delivery_bundle(delivery_id)
    assert first is not None
    assert first.delivery.delivery.state is DeliveryState.PENDING
    assert first.delivery.enqueue_claim_token is None
    assert first.delivery.delivery.reason_code is None
    assert await auth_fixture.repository.list_delivery_attempts(webhook_id, delivery_id) == ()
    assert observed == [EnqueueFailureKind.ADMISSION_REJECTED]

    queue.admission = None
    assert await reconciler.reconcile_enqueue_once() == 1
    queued = await auth_fixture.repository.get_delivery_bundle(delivery_id)
    assert queued is not None
    assert queued.delivery.delivery.state is DeliveryState.QUEUED
    assert queued.delivery.jobs_job_id == "1"


async def test_ambiguous_admission_failure_retains_claim_until_expiry_takeover(
    auth_fixture: SQLiteAuthFixture,
) -> None:
    _, delivery_id = await _seed_automatic_delivery(
        auth_fixture.repository,
        "ambiguous-admission",
    )
    queue = StubDeliveryQueue()
    queue.admission = sqlite3.OperationalError("database unavailable")
    clock = MutableClock()
    observed: list[EnqueueFailureKind] = []
    reconciler = _reconciler(auth_fixture, queue, clock, observer=observed.append)

    assert await reconciler.reconcile_enqueue_once() == 1
    retained = await auth_fixture.repository.get_delivery_bundle(delivery_id)
    assert retained is not None
    assert retained.delivery.delivery.state is DeliveryState.ENQUEUE_CLAIMED
    assert retained.delivery.enqueue_claim_token is not None
    assert observed == [EnqueueFailureKind.BACKEND_UNAVAILABLE]

    clock.advance(61)
    queue.admission = None
    assert await reconciler.reconcile_enqueue_once() == 1
    queued = await auth_fixture.repository.get_delivery_bundle(delivery_id)
    assert queued is not None
    assert queued.delivery.delivery.state is DeliveryState.QUEUED
    assert queued.delivery.enqueue_claim_token is None
    assert len(queue.admit_calls) == 2


async def test_ambiguous_terminal_lookup_retains_expired_claim_for_recovery(
    auth_fixture: SQLiteAuthFixture,
) -> None:
    _, delivery_id = await _seed_automatic_delivery(
        auth_fixture.repository,
        "ambiguous-terminal-lookup",
        expires_at=NOW,
    )
    queue = StubDeliveryQueue()
    queue.identity = sqlite3.OperationalError("database unavailable")
    clock = MutableClock()
    reconciler = _reconciler(auth_fixture, queue, clock)

    assert await reconciler.reconcile_enqueue_once() == 1
    retained = await auth_fixture.repository.get_delivery_bundle(delivery_id)
    assert retained is not None
    assert retained.delivery.delivery.state is DeliveryState.ENQUEUE_CLAIMED
    assert retained.delivery.enqueue_claim_token is not None
    assert queue.admit_calls == []

    clock.advance(61)
    queue.identity = None
    assert await reconciler.reconcile_enqueue_once() == 1
    retired = await auth_fixture.repository.get_delivery_bundle(delivery_id)
    assert retired is not None
    assert retired.delivery.delivery.state is DeliveryState.DEAD
    assert retired.delivery.delivery.reason_code is DeliveryReasonCode.DELIVERY_EXPIRED
    assert retired.delivery.enqueue_claim_token is None
    assert queue.admit_calls == []


@pytest.mark.parametrize(
    "outcome",
    (OperationOutcome.BACKEND_CONFLICT, OperationOutcome.BACKEND_SCHEMA_ERROR),
)
async def test_permanent_jobs_conflict_terminalizes_owned_nonterminal_claim(
    auth_fixture: SQLiteAuthFixture,
    outcome: OperationOutcome,
) -> None:
    webhook_id, delivery_id = await _seed_automatic_delivery(
        auth_fixture.repository,
        f"permanent-{outcome.value}",
    )
    queue = StubDeliveryQueue()
    queue.admission = JobsDeliveryAdmission(outcome=outcome)
    observed: list[EnqueueFailureKind] = []
    reconciler = _reconciler(
        auth_fixture,
        queue,
        MutableClock(),
        observer=observed.append,
    )

    assert await reconciler.reconcile_enqueue_once() == 1

    terminal = await auth_fixture.repository.get_delivery_bundle(delivery_id)
    assert terminal is not None
    assert terminal.delivery.delivery.state is DeliveryState.DEAD
    assert (
        terminal.delivery.delivery.reason_code
        is DeliveryReasonCode.JOBS_IDENTITY_CONFLICT
    )
    assert terminal.delivery.enqueue_claim_token is None
    assert await auth_fixture.repository.list_delivery_attempts(webhook_id, delivery_id) == ()
    assert observed == [EnqueueFailureKind.IDENTITY_CONFLICT]


async def test_repeated_rejection_is_bounded_by_delivery_expiry(
    auth_fixture: SQLiteAuthFixture,
) -> None:
    _, delivery_id = await _seed_automatic_delivery(
        auth_fixture.repository,
        "rejection-expiry",
        expires_at=NOW + timedelta(seconds=30),
    )
    queue = StubDeliveryQueue()
    queue.admission = JobsDeliveryAdmission(
        outcome=OperationOutcome.ADMISSION_REJECTED,
        admission_rejection_reason=AdmissionRejectionReason.POLICY_REJECTED,
    )
    clock = MutableClock()
    reconciler = _reconciler(auth_fixture, queue, clock)

    assert await reconciler.reconcile_enqueue_once() == 1
    clock.advance(30)
    assert await reconciler.reconcile_enqueue_once() == 1

    expired = await auth_fixture.repository.get_delivery_bundle(delivery_id)
    assert expired is not None
    assert expired.delivery.delivery.state is DeliveryState.DEAD
    assert expired.delivery.delivery.reason_code is DeliveryReasonCode.DELIVERY_EXPIRED
    assert len(queue.admit_calls) == 1


async def test_failure_observer_cannot_change_ambiguous_claim_retention(
    auth_fixture: SQLiteAuthFixture,
) -> None:
    _, delivery_id = await _seed_automatic_delivery(
        auth_fixture.repository,
        "observer-failure",
    )
    queue = StubDeliveryQueue()
    queue.admission = RuntimeError("sensitive backend detail")

    def fail_observer(_failure: EnqueueFailureKind) -> None:
        raise RuntimeError("observer failed")

    reconciler = _reconciler(
        auth_fixture,
        queue,
        MutableClock(),
        observer=fail_observer,
    )

    assert await reconciler.reconcile_enqueue_once() == 1
    retained = await auth_fixture.repository.get_delivery_bundle(delivery_id)
    assert retained is not None
    assert retained.delivery.delivery.state is DeliveryState.ENQUEUE_CLAIMED
    assert retained.delivery.enqueue_claim_token is not None


async def test_enqueue_iteration_claims_at_most_one_hundred_and_skips_test_work(
    auth_fixture: SQLiteAuthFixture,
) -> None:
    event_type = "enqueue.batch"
    webhook_ids = [
        await seed_registration(auth_fixture.repository, event_types=(event_type,))
        for _ in range(101)
    ]
    generated: list[str] = []

    def delivery_factory() -> str:
        delivery_id = canonical_uuid4(f"enqueue-batch-{len(generated)}")
        generated.append(delivery_id)
        return delivery_id

    event_id = canonical_uuid4("enqueue-batch-event")
    async with auth_fixture.repository.transaction() as tx:
        captured = await tx.capture_event_and_expand(
            event_insert(
                event_id=event_id,
                source_identity="enqueue-batch-command",
                event_type=event_type,
            ),
            delivery_factory,
            NOW + timedelta(hours=72),
        )
        test_delivery = await tx.insert_delivery(
            canonical_uuid4("enqueue-test-delivery"),
            event_id=event_id,
            webhook_id=webhook_ids[0],
            kind=DeliveryKind.TEST,
            expires_at=NOW + timedelta(minutes=2),
            now=NOW,
        )
    assert len(captured.deliveries) == 101

    queue = StubDeliveryQueue()
    reconciler = _reconciler(auth_fixture, queue, MutableClock())

    assert await reconciler.reconcile_enqueue_once() == 100
    assert len(queue.admit_calls) == 100
    assert await reconciler.reconcile_enqueue_once() == 1
    assert len(queue.admit_calls) == 101
    assert await reconciler.reconcile_enqueue_once() == 0

    stored_test = await auth_fixture.repository.get_delivery_bundle(
        test_delivery.delivery.id
    )
    assert stored_test is not None
    assert stored_test.delivery.delivery.state is DeliveryState.PENDING
    assert stored_test.delivery.jobs_job_id is None


def test_jobs_delivery_admission_rejects_incoherent_typed_shapes() -> None:
    record = JobsDeliveryRecord(
        "1",
        canonical_uuid4("admission-shape"),
        "queued",
        False,
    )
    with pytest.raises(ValueError, match="admission shape"):
        JobsDeliveryAdmission(
            outcome=OperationOutcome.ADMISSION_REJECTED,
            record=record,
            admission_rejection_reason=AdmissionRejectionReason.POLICY_REJECTED,
        )
    with pytest.raises(ValueError, match="admission shape"):
        JobsDeliveryAdmission(
            outcome=OperationOutcome.APPLIED,
            record=None,
        )
    with pytest.raises(ValueError, match="admission shape"):
        JobsDeliveryAdmission(
            outcome=OperationOutcome.NO_TRANSITION,
            record=record,
            no_transition_reason=NoTransitionReason.WRONG_STATUS,
        )
    with pytest.raises(ValueError, match="admission shape"):
        JobsDeliveryAdmission(
            outcome=OperationOutcome.APPLIED,
            record=record,
            no_transition_reason=NoTransitionReason.IDEMPOTENT_EXISTING,
        )
