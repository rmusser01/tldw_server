from __future__ import annotations

import base64
import hashlib
from dataclasses import fields
from datetime import datetime, timedelta, timezone
from typing import Protocol

import pytest

from tldw_Server_API.app.core.Admin_Webhooks.crypto import (
    EVENT_BODY_MAX_BYTES,
    ProtectedValue,
    WebhookKeyError,
    WebhookKeyErrorCode,
    WebhookKeyRing,
)
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    AttemptState,
    DeliveryKind,
    DeliveryReasonCode,
    DeliveryRuntimeComponent,
    DeliveryRuntimeReasonCode,
    DeliveryState,
    EventSourceKind,
    JobsDispositionKind,
)
from tldw_Server_API.app.core.AuthNZ.exceptions import TransactionError
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
    AttemptCompletion,
    AttemptReservation,
    DeliveryBundle,
    EnqueueClaim,
    EventCaptureResult,
    EventInsert,
    PendingJobsDisposition,
    RegistrationInsert,
    RegistrationTarget,
    RetentionBatchResult,
    RuntimeHeartbeatWrite,
    StoredWebhookDelivery,
    StoredWebhookEvent,
    WebhookRepositoryError,
    _attempt_from_row,
    _heartbeat_from_row,
    _stored_delivery_from_row,
    _stored_event_from_row,
)

NOW = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)
KEY_ID = "key-2026-08"
DISPOSITION_TOKEN = "a" * 64


def canonical_uuid4(label: str) -> str:
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-4{digest[13:16]}-8{digest[17:20]}-{digest[20:32]}"


def opaque_token(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class DeliveryRepositoryFixture(Protocol):
    repository: AdminWebhookRepository
    integrity_error: type[BaseException]

    async def execute(self, query: str, *params: object) -> None: ...

    async def fetchval(self, query: str, *params: object) -> object: ...

    async def fetchrow(self, query: str, *params: object) -> object: ...


def key_ring() -> WebhookKeyRing:
    encoded = base64.b64encode(b"k" * 32).decode("ascii")
    return WebhookKeyRing({KEY_ID: encoded}, primary_id=KEY_ID)


def event_insert(
    event_id: str = canonical_uuid4("event-1"),
    *,
    source_kind: EventSourceKind = EventSourceKind.COMMAND,
    source_identity: str = "command-1",
    event_type: str = "user.created",
    body: bytes = b'{"id":1}',
    created_at: datetime = NOW,
) -> EventInsert:
    ring = key_ring()
    return EventInsert(
        id=event_id,
        event_type=event_type,
        api_version="2026-07-01",
        source_kind=source_kind,
        aggregate_type="user" if source_kind is EventSourceKind.AGGREGATE else None,
        aggregate_id=source_identity if source_kind is EventSourceKind.AGGREGATE else None,
        aggregate_version="7" if source_kind is EventSourceKind.AGGREGATE else None,
        source_command_id=(
            source_identity if source_kind is EventSourceKind.COMMAND else None
        ),
        source_component="authnz",
        source_request_id="request-1",
        body=ring.encrypt_event_body(
            event_id=event_id,
            api_version="2026-07-01",
            body=body,
        ),
        body_size_bytes=len(body),
        created_at=created_at,
    )


async def seed_registration(
    repository: AdminWebhookRepository,
    *,
    event_types: tuple[str, ...] = ("user.created",),
    active: bool = True,
    secret_rotation_required: bool = False,
    deleted: bool = False,
    now: datetime = NOW - timedelta(hours=1),
) -> int:
    async with repository.transaction() as tx:
        webhook_id = await tx.allocate_registration_id()
        protected = ProtectedValue(
            ciphertext_json='{"ciphertext":"opaque"}',
            key_id=KEY_ID,
        )
        created = await tx.insert_registration(
            RegistrationInsert(
                id=webhook_id,
                description=f"registration-{webhook_id}",
                target=RegistrationTarget(
                    protected=protected,
                    hostname="hooks.example.com",
                    display="https://hooks.example.com",
                ),
                event_types=event_types,
                active=active,
                timeout_seconds=10,
                secret=protected,
                secret_rotation_required=secret_rotation_required,
                actor_user_id=7,
                now=now,
            )
        )
        if deleted:
            await tx.soft_delete_registration(
                webhook_id,
                expected_revision=created.revision,
                actor_user_id=7,
                at=now + timedelta(minutes=1),
            )
    return webhook_id


def assert_metadata_is_sanitized(value: object) -> None:
    names = {field.name for field in fields(value)}
    assert not names & {
        "body",
        "body_ciphertext_json",
        "target",
        "target_url",
        "secret",
        "secret_ciphertext_json",
    }


@pytest.mark.unit
def test_event_insert_normalizes_time_and_validates_source_shape() -> None:
    event = event_insert(created_at=NOW.astimezone(timezone(timedelta(hours=-7))))
    assert event.created_at == NOW
    assert event.created_at.tzinfo is timezone.utc

    with pytest.raises(ValueError, match="source identity"):
        EventInsert(
            **{
                **event.__dict__,
                "aggregate_type": "user",
                "aggregate_id": "1",
                "aggregate_version": "1",
            }
        )

    with pytest.raises(ValueError, match="event ID"):
        EventInsert(**{**event.__dict__, "id": "EVENT-1"})
    with pytest.raises(ValueError, match="event ID"):
        EventInsert(**{**event.__dict__, "id": canonical_uuid4("event-1").upper()})
    with pytest.raises(ValueError, match="event ID"):
        EventInsert(
            **{
                **event.__dict__,
                "id": "00000000-0000-1000-8000-000000000001",
            }
        )


@pytest.mark.unit
def test_repository_record_contracts_are_closed_and_validate_invariants() -> None:
    event = event_insert()
    assert isinstance(event, EventInsert)
    assert all(
        record.__dataclass_params__.frozen
        for record in (
            EventInsert,
            StoredWebhookEvent,
            EventCaptureResult,
            StoredWebhookDelivery,
            DeliveryBundle,
            EnqueueClaim,
            AttemptReservation,
            AttemptCompletion,
            PendingJobsDisposition,
            RuntimeHeartbeatWrite,
            RetentionBatchResult,
        )
    )

    with pytest.raises(ValueError, match="disposition token"):
        PendingJobsDisposition(
            delivery_id=canonical_uuid4("delivery-1"),
            jobs_job_id="job-1",
            attempt_id=canonical_uuid4("attempt-1"),
            kind=JobsDispositionKind.RETRY,
            delay_seconds=30,
            token="A" * 64,
            not_before_at=NOW,
        )
    with pytest.raises(ValueError, match="retry disposition"):
        PendingJobsDisposition(
            delivery_id=canonical_uuid4("delivery-1"),
            jobs_job_id="job-1",
            attempt_id=canonical_uuid4("attempt-1"),
            kind=JobsDispositionKind.RETRY,
            delay_seconds=None,
            token=DISPOSITION_TOKEN,
            not_before_at=NOW,
        )
    with pytest.raises(ValueError, match="ready heartbeat"):
        RuntimeHeartbeatWrite(
            component=DeliveryRuntimeComponent.WORKER,
            instance_id=canonical_uuid4("worker-1"),
            ready=True,
            reason_code=DeliveryRuntimeReasonCode.JOBS_UNAVAILABLE,
            heartbeat_at=NOW,
            last_success_at=None,
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kind", "delay_seconds", "not_before_at"),
    (
        (JobsDispositionKind.RETRY, 30, NOW),
        (JobsDispositionKind.DEFER, None, NOW),
        (JobsDispositionKind.COMPLETE, None, None),
        (JobsDispositionKind.FAIL, None, None),
        (JobsDispositionKind.CANCEL, None, None),
    ),
)
def test_pending_disposition_accepts_only_canonical_scheduling_shape(
    kind: JobsDispositionKind,
    delay_seconds: int | None,
    not_before_at: datetime | None,
) -> None:
    pending = PendingJobsDisposition(
        delivery_id=canonical_uuid4("disposition-delivery"),
        jobs_job_id="job-1",
        attempt_id=canonical_uuid4("disposition-attempt"),
        kind=kind,
        delay_seconds=delay_seconds,
        token=DISPOSITION_TOKEN,
        not_before_at=not_before_at,
    )
    assert pending.not_before_at == not_before_at


@pytest.mark.unit
@pytest.mark.parametrize("kind", tuple(JobsDispositionKind))
def test_pending_disposition_rejects_every_wrong_not_before_permutation(
    kind: JobsDispositionKind,
) -> None:
    required = kind in {JobsDispositionKind.RETRY, JobsDispositionKind.DEFER}
    with pytest.raises(ValueError, match="not-before"):
        PendingJobsDisposition(
            delivery_id=canonical_uuid4("bad-schedule-delivery"),
            jobs_job_id="job-1",
            attempt_id=None,
            kind=kind,
            delay_seconds=30 if kind is JobsDispositionKind.RETRY else None,
            token=DISPOSITION_TOKEN,
            not_before_at=None if required else NOW,
        )


@pytest.mark.unit
@pytest.mark.parametrize("kind", tuple(JobsDispositionKind))
def test_pending_disposition_rejects_every_wrong_delay_permutation(
    kind: JobsDispositionKind,
) -> None:
    with pytest.raises(ValueError, match="delay"):
        PendingJobsDisposition(
            delivery_id=canonical_uuid4("bad-delay-delivery"),
            jobs_job_id="job-1",
            attempt_id=None,
            kind=kind,
            delay_seconds=None if kind is JobsDispositionKind.RETRY else 30,
            token=DISPOSITION_TOKEN,
            not_before_at=(
                NOW
                if kind in {JobsDispositionKind.RETRY, JobsDispositionKind.DEFER}
                else None
            ),
        )


@pytest.mark.unit
def test_event_body_boundary_and_cross_event_identity_are_enforced() -> None:
    ring = key_ring()
    accepted = ring.encrypt_event_body(
        event_id="event-max",
        api_version="2026-07-01",
        body=b"x" * EVENT_BODY_MAX_BYTES,
    )
    assert len(
        ring.decrypt_event_body(
            event_id="event-max",
            api_version="2026-07-01",
            protected=accepted,
        )
    ) == EVENT_BODY_MAX_BYTES

    with pytest.raises(WebhookKeyError) as oversized:
        ring.encrypt_event_body(
            event_id="event-too-large",
            api_version="2026-07-01",
            body=b"x" * (EVENT_BODY_MAX_BYTES + 1),
        )
    assert oversized.value.code is WebhookKeyErrorCode.EVENT_BODY_TOO_LARGE

    with pytest.raises(WebhookKeyError) as substitution:
        ring.decrypt_event_body(
            event_id="event-other",
            api_version="2026-07-01",
            protected=accepted,
        )
    assert substitution.value.code is WebhookKeyErrorCode.CONTEXT_MISMATCH


@pytest.mark.unit
def test_attempt_completion_rejects_nonterminal_and_incoherent_retry() -> None:
    with pytest.raises(ValueError, match="completion state"):
        AttemptCompletion(
            attempt_state=AttemptState.PROCESSING,
            delivery_state=DeliveryState.PROCESSING,
            disposition=None,
            status_code=None,
            latency_ms=None,
            reason_code=None,
            requested_retry_delay_seconds=None,
            finished_at=NOW,
            completed_after_config_change=False,
        )
    with pytest.raises(ValueError, match="retry delay"):
        AttemptCompletion(
            attempt_state=AttemptState.RETRYABLE,
            delivery_state=DeliveryState.RETRY_WAIT,
            disposition=JobsDispositionKind.RETRY,
            status_code=503,
            latency_ms=1,
            reason_code=None,
            requested_retry_delay_seconds=None,
            finished_at=NOW,
            completed_after_config_change=False,
        )


def persisted_delivery_row() -> dict[str, object]:
    return {
        "id": canonical_uuid4("delivery-1"),
        "event_id": canonical_uuid4("event-1"),
        "webhook_id": 1,
        "kind": "automatic",
        "delivery_config_version": 1,
        "secret_version": 1,
        "jobs_job_id": None,
        "enqueue_claim_token": None,
        "enqueue_claim_expires_at": None,
        "state": "invented-state",
        "attempt_count": 0,
        "current_attempt_id": None,
        "status_code": None,
        "latency_ms": None,
        "reason_code": None,
        "pending_jobs_disposition": None,
        "pending_jobs_disposition_delay_seconds": None,
        "pending_jobs_disposition_token": None,
        "pending_jobs_disposition_not_before_at": None,
        "jobs_disposition_applied": False,
        "completed_after_config_change": False,
        "terminal_at": None,
        "expires_at": NOW + timedelta(hours=72),
        "redelivery_of_id": None,
        "created_at": NOW,
        "updated_at": NOW,
    }


@pytest.mark.unit
def test_malformed_persisted_delivery_enum_fails_closed() -> None:
    row = persisted_delivery_row()
    with pytest.raises(ValueError, match="enum"):
        _stored_delivery_from_row(row)


@pytest.mark.unit
def test_malformed_persisted_coordinates_fail_closed() -> None:
    delivery = persisted_delivery_row()
    delivery["state"] = DeliveryState.PENDING.value
    delivery["id"] = "not-a-uuid"
    with pytest.raises(ValueError, match="delivery ID"):
        _stored_delivery_from_row(delivery)

    delivery = persisted_delivery_row()
    delivery["state"] = DeliveryState.ENQUEUE_CLAIMED.value
    delivery["enqueue_claim_token"] = "A" * 64
    delivery["enqueue_claim_expires_at"] = NOW + timedelta(minutes=1)
    with pytest.raises(ValueError, match="enqueue claim token"):
        _stored_delivery_from_row(delivery)

    delivery = persisted_delivery_row()
    delivery["state"] = DeliveryState.PENDING.value
    delivery["redelivery_of_id"] = canonical_uuid4("redelivery").upper()
    with pytest.raises(ValueError, match="persisted redelivery ID"):
        _stored_delivery_from_row(delivery)

    delivery = persisted_delivery_row()
    delivery.update(
        {
            "state": DeliveryState.RETRY_WAIT.value,
            "jobs_job_id": "job-1",
            "attempt_count": 1,
            "current_attempt_id": canonical_uuid4("pending-attempt"),
            "pending_jobs_disposition": JobsDispositionKind.RETRY.value,
            "pending_jobs_disposition_delay_seconds": 30,
            "pending_jobs_disposition_token": "A" * 64,
            "pending_jobs_disposition_not_before_at": NOW + timedelta(minutes=1),
        }
    )
    with pytest.raises(ValueError, match="pending disposition shape"):
        _stored_delivery_from_row(delivery)

    with pytest.raises(ValueError, match="event ID"):
        _stored_event_from_row(
            {
                "id": "not-a-uuid",
                "event_type": "user.created",
                "api_version": "2026-07-01",
                "source_kind": EventSourceKind.COMMAND.value,
                "aggregate_type": None,
                "aggregate_id": None,
                "aggregate_version": None,
                "source_command_id": "command-1",
                "source_component": "authnz",
                "source_request_id": None,
                "body_ciphertext_json": '{"ciphertext":"opaque"}',
                "body_key_id": KEY_ID,
                "body_size_bytes": 1,
                "created_at": NOW,
            }
        )

    with pytest.raises(ValueError, match="attempt ID"):
        _attempt_from_row(
            {
                "id": "not-a-uuid",
                "delivery_id": canonical_uuid4("attempt-delivery"),
                "attempt_number": 1,
                "jobs_job_id": None,
                "jobs_lease_id": None,
                "test_attempt_token": opaque_token("attempt-token"),
                "request_timeout_seconds": 10,
                "started_at": NOW,
                "finished_at": None,
                "state": AttemptState.PROCESSING.value,
                "status_code": None,
                "latency_ms": None,
                "reason_code": None,
                "requested_retry_delay_seconds": None,
                "jobs_disposition_applied": False,
                "created_at": NOW,
            }
        )

    with pytest.raises(ValueError, match="runtime instance ID"):
        _heartbeat_from_row(
            {
                "component": DeliveryRuntimeComponent.WORKER.value,
                "instance_id": canonical_uuid4("runtime-instance").upper(),
                "ready": True,
                "reason_code": None,
                "heartbeat_at": NOW,
                "last_success_at": NOW,
                "created_at": NOW,
                "updated_at": NOW,
            }
        )


async def exercise_capture_and_history(
    fixture: DeliveryRepositoryFixture,
) -> None:
    repository = fixture.repository
    matching = [await seed_registration(repository) for _ in range(25)]
    await seed_registration(repository, active=False)
    await seed_registration(repository, event_types=("user.deleted",))
    await seed_registration(repository, secret_rotation_required=True)
    await seed_registration(repository, deleted=True)

    generated: list[str] = []

    def delivery_id_factory() -> str:
        value = canonical_uuid4(f"delivery-{len(generated):03d}")
        generated.append(value)
        return value

    event = event_insert()
    expires_at = event.created_at + timedelta(hours=72)
    async with repository.transaction() as tx:
        counts = {"match": 0, "batch": 0}
        original_fetch = tx._fetch
        original_executemany = tx._executemany

        async def counted_fetch(
            query: str, params: tuple[object, ...] = ()
        ) -> list[dict[str, object]]:
            if "admin_webhook_match_subscriptions" in query:
                counts["match"] += 1
            return await original_fetch(query, params)

        async def counted_executemany(
            query: str, rows: tuple[tuple[object, ...], ...]
        ) -> int:
            if "admin_webhook_delivery_fanout" in query:
                counts["batch"] += 1
            return await original_executemany(query, rows)

        tx._fetch = counted_fetch  # type: ignore[method-assign]
        tx._executemany = counted_executemany  # type: ignore[method-assign]
        captured = await tx.capture_event_and_expand(
            event,
            delivery_id_factory,
            expires_at,
        )

    assert captured.inserted is True
    assert captured.event.id == event.id
    assert len(captured.deliveries) == 25
    assert counts == {"match": 1, "batch": 1}
    assert generated == [canonical_uuid4(f"delivery-{index:03d}") for index in range(25)]
    assert {item.delivery.webhook_id for item in captured.deliveries} == set(matching)
    assert all(item.delivery.expires_at == expires_at for item in captured.deliveries)
    assert all(item.delivery.delivery_config_version == 1 for item in captured.deliveries)
    assert all(item.delivery.secret_version == 1 for item in captured.deliveries)

    replay_factory_calls = 0

    def replay_factory() -> str:
        nonlocal replay_factory_calls
        replay_factory_calls += 1
        return "must-not-be-used"

    replay_event = event_insert(event_id=canonical_uuid4("different-id"))
    async with repository.transaction() as tx:
        replay = await tx.capture_event_and_expand(
            replay_event,
            replay_factory,
            expires_at,
        )
    assert replay.inserted is False
    assert replay.event.id == event.id
    assert [item.delivery.id for item in replay.deliveries] == [
        item.delivery.id for item in captured.deliveries
    ]
    assert replay_factory_calls == 0

    aggregate = event_insert(
        event_id=canonical_uuid4("aggregate-event"),
        source_kind=EventSourceKind.AGGREGATE,
        source_identity="user-7",
    )
    aggregate_ids: list[str] = []

    def aggregate_factory() -> str:
        value = canonical_uuid4(f"aggregate-delivery-{len(aggregate_ids):03d}")
        aggregate_ids.append(value)
        return value

    async with repository.transaction() as tx:
        aggregate_capture = await tx.capture_event_and_expand(
            aggregate,
            aggregate_factory,
            expires_at,
        )
    aggregate_calls = 0

    def aggregate_replay_factory() -> str:
        nonlocal aggregate_calls
        aggregate_calls += 1
        return "must-not-be-used"

    async with repository.transaction() as tx:
        aggregate_replay = await tx.capture_event_and_expand(
            event_insert(
                event_id=canonical_uuid4("other-aggregate-id"),
                source_kind=EventSourceKind.AGGREGATE,
                source_identity="user-7",
            ),
            aggregate_replay_factory,
            expires_at,
        )
    assert aggregate_capture.inserted is True
    assert aggregate_replay.inserted is False
    assert aggregate_replay.event.id == aggregate.id
    assert aggregate_calls == 0

    webhook_id = matching[0]
    async with repository.transaction() as tx:
        manual = await tx.insert_delivery(
            canonical_uuid4("manual-z"),
            event_id=event.id,
            webhook_id=webhook_id,
            kind=DeliveryKind.MANUAL,
            expires_at=expires_at,
            now=NOW + timedelta(minutes=1),
            redelivery_of_id=captured.deliveries[0].delivery.id,
        )
        test = await tx.insert_delivery(
            canonical_uuid4("test-a"),
            event_id=event.id,
            webhook_id=webhook_id,
            kind=DeliveryKind.TEST,
            expires_at=expires_at,
            now=NOW + timedelta(minutes=1),
        )
    assert manual.delivery.kind is DeliveryKind.MANUAL
    assert test.delivery.kind is DeliveryKind.TEST

    page = await repository.list_delivery_history(webhook_id, limit=20, offset=0)
    assert [item.id for item in page.items[:2]] == sorted(
        [canonical_uuid4("test-a"), canonical_uuid4("manual-z")],
        reverse=True,
    )
    assert page.total == 4
    assert_metadata_is_sanitized(page.items[0])
    with pytest.raises(ValueError, match="limit"):
        await repository.list_delivery_history(webhook_id, limit=0, offset=0)
    with pytest.raises(ValueError, match="offset"):
        await repository.list_delivery_history(webhook_id, limit=10, offset=-1)

    other_webhook = matching[1]
    assert (
        await repository.get_delivery_for_registration(other_webhook, manual.delivery.id)
        is None
    )
    scoped = await repository.get_delivery_for_registration(webhook_id, manual.delivery.id)
    assert scoped is not None and scoped.id == manual.delivery.id

    with pytest.raises(
        WebhookRepositoryError,
        match="admin_webhook_coordinate_invalid",
    ):
        async with repository.transaction() as tx:
            await tx.capture_event_and_expand(
                event_insert(
                    event_id=canonical_uuid4("bad-factory-event"),
                    source_identity="bad-factory-command",
                ),
                lambda: "not-a-uuid",
                expires_at,
            )


async def _captured_delivery(
    repository: AdminWebhookRepository,
    *,
    event_id: str,
    command_id: str,
    isolated: bool = False,
) -> tuple[int, str]:
    event_type = f"contract.{event_id}" if isolated else "user.created"
    webhook_id = await seed_registration(repository, event_types=(event_type,))
    async with repository.transaction() as tx:
        result = await tx.capture_event_and_expand(
            event_insert(
                event_id=canonical_uuid4(event_id),
                source_identity=command_id,
                event_type=event_type,
            ),
            lambda: canonical_uuid4(f"delivery-{event_id}"),
            NOW + timedelta(hours=72),
        )
    delivery = next(
        item for item in result.deliveries if item.delivery.webhook_id == webhook_id
    )
    return webhook_id, delivery.delivery.id


async def exercise_delivery_state_machine(
    fixture: DeliveryRepositoryFixture,
) -> None:
    repository = fixture.repository
    webhook_id, delivery_id = await _captured_delivery(
        repository,
        event_id="state-event",
        command_id="state-command",
    )

    async with repository.transaction() as tx:
        with pytest.raises(ValueError, match="enqueue claim token"):
            await tx.claim_pending_delivery(
                "A" * 64,
                NOW + timedelta(minutes=1),
                NOW,
            )
        claim = await tx.claim_pending_delivery(
            opaque_token("claim-1"),
            NOW + timedelta(minutes=1),
            NOW,
        )
        assert claim is not None and claim.delivery.delivery.id == delivery_id
        assert await tx.attach_jobs_job(
            delivery_id, opaque_token("stale"), "job-1", NOW
        ) is None
        queued = await tx.attach_jobs_job(
            delivery_id, opaque_token("claim-1"), "job-1", NOW
        )
        assert queued is not None
        assert not await tx.release_expired_enqueue_claim(
            delivery_id, opaque_token("claim-1"), NOW
        )

    await fixture.execute(
        """
        UPDATE admin_webhook_migration_state
        SET first_canonical_activity_at = NULL,
            first_canonical_activity_kind = NULL
        WHERE singleton_id = 1
        """
    )

    for number in range(1, 5):
        started_at = NOW + timedelta(minutes=number)
        lease_id = f"lease-{number}"
        attempt_id = canonical_uuid4(f"attempt-{number}")
        async with repository.transaction() as tx:
            stale = await tx.reserve_jobs_attempt(
                delivery_id,
                "other-job",
                lease_id,
                canonical_uuid4(f"stale-{number}"),
                10,
                started_at,
                started_at + timedelta(seconds=10),
            )
            assert stale is None
            reservation = await tx.reserve_jobs_attempt(
                delivery_id,
                "job-1",
                lease_id,
                attempt_id,
                number,
                started_at,
                started_at + timedelta(seconds=number),
            )
        assert reservation is not None and reservation.reserved is True
        assert reservation.attempt is not None
        assert reservation.attempt.attempt_number == number
        assert reservation.attempt.request_timeout_seconds == number
        if number == 1:
            migration = await repository.get_migration_state()
            assert migration.first_canonical_activity_kind == "delivery_attempt"

        completion = AttemptCompletion(
            attempt_state=(
                AttemptState.FAILED if number == 4 else AttemptState.RETRYABLE
            ),
            delivery_state=(
                DeliveryState.DEAD if number == 4 else DeliveryState.RETRY_WAIT
            ),
            disposition=(
                JobsDispositionKind.FAIL if number == 4 else JobsDispositionKind.RETRY
            ),
            status_code=503,
            latency_ms=number * 10,
            reason_code=(
                DeliveryReasonCode.ATTEMPT_BUDGET_EXHAUSTED
                if number == 4
                else None
            ),
            requested_retry_delay_seconds=None if number == 4 else 30,
            finished_at=started_at + timedelta(seconds=1),
            completed_after_config_change=False,
        )
        disposition_token = f"{number:x}" * 64
        async with repository.transaction() as tx:
            assert (
                await tx.finish_attempt_and_prepare_disposition(
                    "stale-lease",
                    completion,
                    disposition_token,
                    (
                        completion.finished_at
                        if completion.disposition is JobsDispositionKind.RETRY
                        else None
                    ),
                )
                is None
            )
            pending = await tx.finish_attempt_and_prepare_disposition(
                lease_id,
                completion,
                disposition_token,
                (
                    completion.finished_at
                    if completion.disposition is JobsDispositionKind.RETRY
                    else None
                ),
            )
            assert pending is not None
            assert not await tx.acknowledge_jobs_disposition(
                delivery_id,
                "f" * 64,
                "failed" if number == 4 else "queued",
            )
            assert await tx.acknowledge_jobs_disposition(
                delivery_id,
                disposition_token,
                "failed" if number == 4 else "queued",
            )
        assert (
            await fixture.fetchval(
                """
                SELECT jobs_disposition_applied
                FROM admin_webhook_delivery_attempts
                WHERE id = ? AND delivery_id = ?
                """,
                attempt_id,
                delivery_id,
            )
            in (1, True)
        )

    attempts = await repository.list_delivery_attempts(webhook_id, delivery_id)
    assert [item.attempt_number for item in attempts] == [1, 2, 3, 4]
    assert all(item.request_timeout_seconds in range(1, 5) for item in attempts)
    assert all(item.finished_at is not None for item in attempts)
    assert_metadata_is_sanitized(attempts[0])

    before_fifth = await repository.get_delivery_for_registration(webhook_id, delivery_id)
    before_fifth_attempts = await repository.list_delivery_attempts(
        webhook_id, delivery_id
    )
    async with repository.transaction() as tx:
        fifth = await tx.reserve_jobs_attempt(
            delivery_id,
            "job-1",
            "lease-5",
            canonical_uuid4("attempt-5"),
            5,
            NOW + timedelta(minutes=5),
            NOW + timedelta(minutes=5, seconds=5),
        )
        assert fifth is not None and fifth.reserved is False
        assert fifth.reason_code is DeliveryReasonCode.ATTEMPT_BUDGET_EXHAUSTED
        assert not await tx.expire_delivery(delivery_id, DeliveryState.DEAD, NOW)
    assert await repository.get_delivery_for_registration(
        webhook_id, delivery_id
    ) == before_fifth
    assert await repository.list_delivery_attempts(
        webhook_id, delivery_id
    ) == before_fifth_attempts

    async with repository.transaction() as tx:
        test_delivery = await tx.insert_delivery(
            canonical_uuid4("test-reservation"),
            event_id=canonical_uuid4("state-event"),
            webhook_id=webhook_id,
            kind=DeliveryKind.TEST,
            expires_at=NOW + timedelta(hours=72),
            now=NOW + timedelta(minutes=6),
        )
        with pytest.raises(ValueError, match="test attempt token"):
            await tx.reserve_test_attempt(
                "A" * 64,
                test_delivery.delivery.id,
                canonical_uuid4("invalid-token-attempt"),
                7,
                NOW + timedelta(minutes=7),
            )
        test_reservation = await tx.reserve_test_attempt(
            opaque_token("test-token"),
            test_delivery.delivery.id,
            canonical_uuid4("test-attempt"),
            7,
            NOW + timedelta(minutes=7),
        )
        assert test_reservation is not None and test_reservation.reserved
        assert test_reservation.attempt is not None
        assert test_reservation.attempt.request_timeout_seconds == 7
        assert (
            await tx.finish_attempt_and_prepare_disposition(
                opaque_token("test-token"),
                AttemptCompletion(
                    attempt_state=AttemptState.SUCCEEDED,
                    delivery_state=DeliveryState.SUCCEEDED,
                    disposition=None,
                    status_code=204,
                    latency_ms=4,
                    reason_code=None,
                    requested_retry_delay_seconds=None,
                    finished_at=NOW + timedelta(minutes=7, seconds=1),
                    completed_after_config_change=False,
                ),
                None,
                None,
            )
            is None
        )
    completed_test = await repository.get_delivery_for_registration(
        webhook_id, test_delivery.delivery.id
    )
    assert completed_test is not None
    assert completed_test.state is DeliveryState.SUCCEEDED
    test_attempts = await repository.list_delivery_attempts(
        webhook_id, test_delivery.delivery.id
    )
    assert len(test_attempts) == 1
    assert test_attempts[0].state is AttemptState.SUCCEEDED
    assert (
        await fixture.fetchval(
            """
            SELECT pending_jobs_disposition
            FROM admin_webhook_deliveries
            WHERE id = ?
            """,
            test_delivery.delivery.id,
        )
        is None
    )


async def exercise_recovery_runtime_and_retention(
    fixture: DeliveryRepositoryFixture,
) -> None:
    repository = fixture.repository
    webhook_id, delivery_id = await _captured_delivery(
        repository,
        event_id="recovery-event",
        command_id="recovery-command",
    )
    await fixture.execute(
        """
        UPDATE admin_webhook_migration_state
        SET first_canonical_activity_at = NULL,
            first_canonical_activity_kind = NULL
        WHERE singleton_id = 1
        """
    )
    async with repository.transaction() as tx:
        claim = await tx.claim_pending_delivery(
            opaque_token("expiring-claim"),
            NOW - timedelta(seconds=1),
            NOW - timedelta(minutes=1),
        )
        assert claim is not None
        assert not await tx.release_expired_enqueue_claim(
            delivery_id, opaque_token("wrong-token"), NOW
        )
        assert await tx.release_expired_enqueue_claim(
            delivery_id, opaque_token("expiring-claim"), NOW
        )
        expired_at = NOW + timedelta(hours=73)
        assert await tx.expire_delivery(
            delivery_id, DeliveryState.PENDING, expired_at
        )
        assert not await tx.expire_delivery(
            delivery_id, DeliveryState.PENDING, expired_at
        )

        ready = await tx.upsert_runtime_heartbeat(
            RuntimeHeartbeatWrite(
                component=DeliveryRuntimeComponent.WORKER,
                instance_id=canonical_uuid4("worker-1"),
                ready=True,
                reason_code=None,
                heartbeat_at=NOW,
                last_success_at=NOW,
            )
        )
        assert ready.ready is True
        unavailable = await tx.upsert_runtime_heartbeat(
            RuntimeHeartbeatWrite(
                component=DeliveryRuntimeComponent.RECONCILER,
                instance_id=canonical_uuid4("reconciler-1"),
                ready=False,
                reason_code=DeliveryRuntimeReasonCode.JOBS_UNAVAILABLE,
                heartbeat_at=NOW,
                last_success_at=None,
            )
        )
        assert unavailable.reason_code is DeliveryRuntimeReasonCode.JOBS_UNAVAILABLE

    migration = await repository.get_migration_state()
    assert migration.first_canonical_activity_at is None
    assert migration.first_canonical_activity_kind is None

    heartbeats = await repository.list_runtime_heartbeats()
    assert len(heartbeats) == 2
    assert all(item.heartbeat_at.tzinfo is timezone.utc for item in heartbeats)

    await fixture.execute(
        """
        INSERT INTO admin_webhook_idempotency (
            lookup_digest, actor_id, operation, route, request_fingerprint,
            state, created_at, updated_at, expires_at
        ) VALUES (?, ?, ?, ?, ?, 'in_progress', ?, ?, ?)
        """,
        f"sha256:{opaque_token('expired-idempotency')}",
        "actor-1",
        "test",
        "/admin/webhooks/test",
        f"hmac-sha256:{opaque_token('expired-fingerprint')}",
        NOW - timedelta(days=1),
        NOW - timedelta(days=1),
        NOW - timedelta(seconds=1),
    )
    nonterminal_webhook_id, nonterminal_delivery_id = await _captured_delivery(
        repository,
        event_id="retention-nonterminal-event",
        command_id="retention-nonterminal-command",
        isolated=True,
    )

    purge_now = NOW + timedelta(days=34)
    cutoff = NOW + timedelta(days=4)
    first = await repository.purge_retained_rows(purge_now, cutoff, 1)
    second = await repository.purge_retained_rows(purge_now, cutoff, 1)
    third = await repository.purge_retained_rows(purge_now, cutoff, 1)
    fourth = await repository.purge_retained_rows(purge_now, cutoff, 1)
    fifth = await repository.purge_retained_rows(purge_now, cutoff, 1)
    assert isinstance(first, RetentionBatchResult)
    assert first.expired_idempotency == 1
    assert second.deliveries == 1
    assert third.events == 1
    assert fourth.heartbeats == 1
    assert fifth.heartbeats == 1
    assert sum(first.__dict__.values()) == 1
    assert sum(second.__dict__.values()) == 1
    assert sum(third.__dict__.values()) == 1
    assert sum(fourth.__dict__.values()) == 1
    assert sum(fifth.__dict__.values()) == 1
    assert await repository.get_delivery_for_registration(webhook_id, delivery_id) is None
    nonterminal = await repository.get_delivery_for_registration(
        nonterminal_webhook_id, nonterminal_delivery_id
    )
    assert nonterminal is not None and nonterminal.state is DeliveryState.PENDING
    assert (
        await fixture.fetchval(
            "SELECT COUNT(*) FROM admin_webhook_events WHERE id = ?",
            canonical_uuid4("retention-nonterminal-event"),
        )
        == 1
    )

    with pytest.raises(ValueError, match="batch_size"):
        await repository.purge_retained_rows(NOW, NOW, 201)


async def exercise_stale_recovery_and_cancellation(
    fixture: DeliveryRepositoryFixture,
) -> None:
    repository = fixture.repository
    webhook_id, stale_delivery_id = await _captured_delivery(
        repository,
        event_id="stale-event",
        command_id="stale-command",
    )
    async with repository.transaction() as tx:
        claim = await tx.claim_pending_delivery(
            opaque_token("stale-claim"),
            NOW + timedelta(minutes=1),
            NOW,
        )
        assert claim is not None
        assert await tx.attach_jobs_job(
            stale_delivery_id,
            opaque_token("stale-claim"),
            "stale-job",
            NOW,
        ) is not None
        reservation = await tx.reserve_jobs_attempt(
            stale_delivery_id,
            "stale-job",
            "stale-lease",
            canonical_uuid4("stale-attempt"),
            1,
            NOW,
            NOW + timedelta(seconds=1),
        )
        assert reservation is not None and reservation.reserved

    await fixture.execute(
        """
        UPDATE admin_webhook_delivery_attempts
        SET request_timeout_seconds = NULL
        WHERE id = ?
        """,
        canonical_uuid4("stale-attempt"),
    )
    async with repository.transaction() as tx:
        assert not await tx.close_stale_attempt_as_unknown(
            stale_delivery_id,
            canonical_uuid4("stale-attempt"),
            NOW + timedelta(seconds=29),
        )
        assert await tx.close_stale_attempt_as_unknown(
            stale_delivery_id,
            canonical_uuid4("stale-attempt"),
            NOW + timedelta(seconds=30),
        )
        assert await tx.expire_delivery(
            stale_delivery_id,
            DeliveryState.RETRY_WAIT,
            NOW + timedelta(hours=73),
        )

    generated: list[str] = []

    def delivery_factory() -> str:
        value = canonical_uuid4(f"cancel-delivery-{len(generated)}")
        generated.append(value)
        return value

    async with repository.transaction() as tx:
        captured = await tx.capture_event_and_expand(
            event_insert(
                event_id=canonical_uuid4("cancel-event"),
                source_identity="cancel-command",
            ),
            delivery_factory,
            NOW + timedelta(hours=72),
        )
    cancel_delivery = next(
        item for item in captured.deliveries if item.delivery.webhook_id == webhook_id
    )
    async with repository.transaction() as tx:
        claim = await tx.claim_pending_delivery(
            opaque_token("cancel-claim"),
            NOW + timedelta(minutes=1),
            NOW,
        )
        assert claim is not None
        assert await tx.attach_jobs_job(
            cancel_delivery.delivery.id,
            opaque_token("cancel-claim"),
            "cancel-job",
            NOW,
        ) is not None
        pending = await tx.cancel_registration_work(
            webhook_id,
            (2, 2),
            DeliveryReasonCode.CANCELED_DISABLED,
            lambda: "b" * 64,
            NOW + timedelta(minutes=2),
        )
        assert len(pending) == 1
        assert pending[0].attempt_id is None
        assert await tx.acknowledge_jobs_disposition(
            cancel_delivery.delivery.id,
            "b" * 64,
            "cancelled",
        )
        assert not await tx.expire_delivery(
            cancel_delivery.delivery.id,
            DeliveryState.CANCELED,
            NOW + timedelta(hours=73),
        )
    assert await repository.list_delivery_attempts(
        webhook_id,
        cancel_delivery.delivery.id,
    ) == ()


async def _queue_delivery(
    fixture: DeliveryRepositoryFixture,
    label: str,
) -> tuple[int, str, str]:
    webhook_id, delivery_id = await _captured_delivery(
        fixture.repository,
        event_id=f"{label}-event",
        command_id=f"{label}-command",
        isolated=True,
    )
    claim_token = opaque_token(f"{label}-claim")
    jobs_job_id = f"{label}-job"
    async with fixture.repository.transaction() as tx:
        claim = await tx.claim_pending_delivery(
            claim_token,
            NOW + timedelta(minutes=1),
            NOW,
        )
        assert claim is not None and claim.delivery.delivery.id == delivery_id
        assert await tx.attach_jobs_job(
            delivery_id, claim_token, jobs_job_id, NOW
        ) is not None
    return webhook_id, delivery_id, jobs_job_id


async def _prepare_retry_disposition(
    fixture: DeliveryRepositoryFixture,
    label: str,
) -> tuple[int, str, str, str]:
    webhook_id, delivery_id, jobs_job_id = await _queue_delivery(fixture, label)
    attempt_id = canonical_uuid4(f"{label}-attempt")
    lease_id = f"{label}-lease"
    disposition_token = opaque_token(f"{label}-disposition")
    async with fixture.repository.transaction() as tx:
        reservation = await tx.reserve_jobs_attempt(
            delivery_id,
            jobs_job_id,
            lease_id,
            attempt_id,
            10,
            NOW + timedelta(minutes=2),
            NOW + timedelta(minutes=2, seconds=10),
        )
        assert reservation is not None and reservation.reserved
        pending = await tx.finish_attempt_and_prepare_disposition(
            lease_id,
            AttemptCompletion(
                attempt_state=AttemptState.RETRYABLE,
                delivery_state=DeliveryState.RETRY_WAIT,
                disposition=JobsDispositionKind.RETRY,
                status_code=503,
                latency_ms=4,
                reason_code=None,
                requested_retry_delay_seconds=30,
                finished_at=NOW + timedelta(minutes=2, seconds=1),
                completed_after_config_change=False,
            ),
            disposition_token,
            NOW + timedelta(minutes=2, seconds=31),
        )
        assert pending is not None
    return webhook_id, delivery_id, attempt_id, disposition_token


async def exercise_cancellation_cas_and_processing_preservation(
    fixture: DeliveryRepositoryFixture,
) -> None:
    repository = fixture.repository
    for race_kind in ("attach", "reservation"):
        webhook_id, delivery_id = await _captured_delivery(
            repository,
            event_id=f"cancel-{race_kind}-race-event",
            command_id=f"cancel-{race_kind}-race-command",
            isolated=True,
        )
        claim_token = opaque_token(f"cancel-{race_kind}-race-claim")
        async with repository.transaction() as tx:
            claim = await tx.claim_pending_delivery(
                claim_token,
                NOW + timedelta(minutes=1),
                NOW,
            )
            assert claim is not None and claim.delivery.delivery.id == delivery_id
            if race_kind == "reservation":
                assert await tx.attach_jobs_job(
                    delivery_id, claim_token, "reservation-race-job", NOW
                ) is not None
        before = await repository.get_delivery_for_registration(webhook_id, delivery_id)
        assert before is not None

        with pytest.raises(
            WebhookRepositoryError,
            match="admin_webhook_delivery_state_stale",
        ):
            async with repository.transaction() as tx:
                original_fetchrow = tx._fetchrow
                injected = False

                async def race_fetchrow(
                    query: str,
                    params: tuple[object, ...] = (),
                    race_kind: str = race_kind,
                    delivery_id: str = delivery_id,
                    original_fetchrow=original_fetchrow,
                ) -> dict[str, object] | None:
                    nonlocal injected
                    if "admin_webhook_cancel_delivery_cas" in query and not injected:
                        injected = True
                        if race_kind == "attach":
                            await tx._execute(
                                """
                                UPDATE admin_webhook_deliveries
                                SET state = 'queued', jobs_job_id = ?,
                                    enqueue_claim_token = NULL,
                                    enqueue_claim_expires_at = NULL
                                WHERE id = ?
                                """,
                                ("attached-race-job", delivery_id),
                            )
                        else:
                            await tx._execute(
                                """
                                UPDATE admin_webhook_deliveries
                                SET state = 'processing', current_attempt_id = ?,
                                    attempt_count = attempt_count + 1
                                WHERE id = ?
                                """,
                                (
                                    canonical_uuid4("reservation-race-attempt"),
                                    delivery_id,
                                ),
                            )
                    return await original_fetchrow(query, params)

                tx._fetchrow = race_fetchrow  # type: ignore[method-assign]
                await tx.cancel_registration_work(
                    webhook_id,
                    (2, 2),
                    DeliveryReasonCode.CANCELED_DISABLED,
                    lambda race_kind=race_kind: opaque_token(
                        f"cancel-{race_kind}-race-disposition"
                    ),
                    NOW + timedelta(minutes=2),
                )
        assert injected
        assert await repository.get_delivery_for_registration(
            webhook_id, delivery_id
        ) == before

    webhook_id, delivery_id, jobs_job_id = await _queue_delivery(
        fixture, "processing-preservation"
    )
    attempt_id = canonical_uuid4("processing-preservation-attempt")
    lease_id = "processing-preservation-lease"
    async with repository.transaction() as tx:
        reservation = await tx.reserve_jobs_attempt(
            delivery_id,
            jobs_job_id,
            lease_id,
            attempt_id,
            10,
            NOW + timedelta(minutes=3),
            NOW + timedelta(minutes=3, seconds=10),
        )
        assert reservation is not None and reservation.reserved
        assert await tx.cancel_registration_work(
            webhook_id,
            (2, 2),
            DeliveryReasonCode.CANCELED_DISABLED,
            lambda: opaque_token("processing-cancel"),
            NOW + timedelta(minutes=4),
        ) == ()

    processing = await repository.get_delivery_for_registration(webhook_id, delivery_id)
    attempts = await repository.list_delivery_attempts(webhook_id, delivery_id)
    assert processing is not None and processing.state is DeliveryState.PROCESSING
    assert len(attempts) == 1 and attempts[0].state is AttemptState.PROCESSING

    async with repository.transaction() as tx:
        pending = await tx.finish_attempt_and_prepare_disposition(
            lease_id,
            AttemptCompletion(
                attempt_state=AttemptState.SUCCEEDED,
                delivery_state=DeliveryState.SUCCEEDED,
                disposition=JobsDispositionKind.COMPLETE,
                status_code=204,
                latency_ms=8,
                reason_code=None,
                requested_retry_delay_seconds=None,
                finished_at=NOW + timedelta(minutes=4, seconds=1),
                completed_after_config_change=True,
            ),
            opaque_token("processing-real-outcome"),
            None,
        )
        assert pending is not None and pending.attempt_id == attempt_id
    attempts = await repository.list_delivery_attempts(webhook_id, delivery_id)
    assert attempts[0].state is AttemptState.SUCCEEDED
    completed = await repository.get_delivery_bundle(delivery_id)
    assert completed is not None
    assert completed.delivery.delivery.state is DeliveryState.SUCCEEDED
    assert completed.delivery.completed_after_config_change is True


async def exercise_atomic_disposition_acknowledgement(
    fixture: DeliveryRepositoryFixture,
) -> None:
    repository = fixture.repository
    missing = await _prepare_retry_disposition(fixture, "missing-ack-attempt")
    missing_webhook, missing_delivery, _, missing_token = missing
    await fixture.execute(
        "UPDATE admin_webhook_deliveries SET current_attempt_id = ? WHERE id = ?",
        canonical_uuid4("missing-attempt"),
        missing_delivery,
    )
    async with repository.transaction() as tx:
        assert not await tx.acknowledge_jobs_disposition(
            missing_delivery, missing_token, "queued"
        )
    assert not bool(
        await fixture.fetchval(
            "SELECT jobs_disposition_applied FROM admin_webhook_deliveries WHERE id = ?",
            missing_delivery,
        )
    )
    assert await repository.get_delivery_for_registration(
        missing_webhook, missing_delivery
    ) is not None

    wrong = await _prepare_retry_disposition(fixture, "wrong-ack-attempt")
    _, wrong_delivery, wrong_attempt, wrong_token = wrong
    await fixture.execute(
        "UPDATE admin_webhook_deliveries SET current_attempt_id = ? WHERE id = ?",
        wrong_attempt,
        missing_delivery,
    )
    async with repository.transaction() as tx:
        assert not await tx.acknowledge_jobs_disposition(
            missing_delivery, missing_token, "queued"
        )
    assert not bool(
        await fixture.fetchval(
            "SELECT jobs_disposition_applied FROM admin_webhook_deliveries WHERE id = ?",
            missing_delivery,
        )
    )

    processing = await _prepare_retry_disposition(fixture, "processing-ack-attempt")
    _, processing_delivery, processing_attempt, processing_token = processing
    await fixture.execute(
        """
        UPDATE admin_webhook_delivery_attempts
        SET state = 'processing', finished_at = NULL,
            requested_retry_delay_seconds = NULL
        WHERE id = ?
        """,
        processing_attempt,
    )
    async with repository.transaction() as tx:
        assert not await tx.acknowledge_jobs_disposition(
            processing_delivery, processing_token, "queued"
        )
    assert not bool(
        await fixture.fetchval(
            "SELECT jobs_disposition_applied FROM admin_webhook_deliveries WHERE id = ?",
            processing_delivery,
        )
    )
    assert not bool(
        await fixture.fetchval(
            "SELECT jobs_disposition_applied FROM admin_webhook_delivery_attempts WHERE id = ?",
            processing_attempt,
        )
    )

    async with repository.transaction() as tx:
        assert await tx.acknowledge_jobs_disposition(
            wrong_delivery, wrong_token, "queued"
        )


async def exercise_acknowledgement_second_step_rollback(
    fixture: DeliveryRepositoryFixture,
) -> None:
    repository = fixture.repository
    _, delivery_id, attempt_id, disposition_token = await _prepare_retry_disposition(
        fixture, "ack-second-step-rollback"
    )
    assert not bool(
        await fixture.fetchval(
            "SELECT jobs_disposition_applied FROM admin_webhook_deliveries WHERE id = ?",
            delivery_id,
        )
    )
    assert not bool(
        await fixture.fetchval(
            "SELECT jobs_disposition_applied FROM admin_webhook_delivery_attempts WHERE id = ?",
            attempt_id,
        )
    )

    attempt_update_succeeded = False
    delivery_cas_lost = False
    replacement_token = opaque_token("ack-second-step-replacement")
    with pytest.raises(TransactionError, match="transaction"):
        async with repository.transaction() as tx:
            original_fetchrow = tx._fetchrow

            async def lose_delivery_cas(
                query: str,
                params: tuple[object, ...] = (),
            ) -> dict[str, object] | None:
                nonlocal attempt_update_succeeded, delivery_cas_lost
                if (
                    "UPDATE admin_webhook_deliveries" in query
                    and "SET jobs_disposition_applied = TRUE" in query
                ):
                    assert attempt_update_succeeded
                    await tx._execute(
                        """
                        UPDATE admin_webhook_deliveries
                        SET pending_jobs_disposition_token = ?
                        WHERE id = ?
                        """,
                        (replacement_token, delivery_id),
                    )
                    delivery_cas_lost = True
                row = await original_fetchrow(query, params)
                if (
                    "UPDATE admin_webhook_delivery_attempts" in query
                    and "SET jobs_disposition_applied = TRUE" in query
                ):
                    assert row is not None
                    attempt_update_succeeded = True
                return row

            tx._fetchrow = lose_delivery_cas  # type: ignore[method-assign]
            await tx.acknowledge_jobs_disposition(
                delivery_id, disposition_token, "queued"
            )

    assert attempt_update_succeeded
    assert delivery_cas_lost
    assert not bool(
        await fixture.fetchval(
            "SELECT jobs_disposition_applied FROM admin_webhook_deliveries WHERE id = ?",
            delivery_id,
        )
    )
    assert not bool(
        await fixture.fetchval(
            "SELECT jobs_disposition_applied FROM admin_webhook_delivery_attempts WHERE id = ?",
            attempt_id,
        )
    )
    assert (
        await fixture.fetchval(
            "SELECT pending_jobs_disposition_token FROM admin_webhook_deliveries WHERE id = ?",
            delivery_id,
        )
        == disposition_token
    )


async def exercise_malformed_persisted_coordinates(
    fixture: DeliveryRepositoryFixture,
) -> None:
    repository = fixture.repository
    webhook_id, delivery_id = await _captured_delivery(
        repository,
        event_id="malformed-persisted-event",
        command_id="malformed-persisted-command",
        isolated=True,
    )
    await fixture.execute(
        """
        UPDATE admin_webhook_deliveries
        SET state = 'enqueue_claimed', enqueue_claim_token = ?,
            enqueue_claim_expires_at = ?
        WHERE id = ?
        """,
        "A" * 64,
        NOW + timedelta(minutes=1),
        delivery_id,
    )
    with pytest.raises(ValueError, match="enqueue claim token"):
        await repository.get_delivery_for_registration(webhook_id, delivery_id)

    await fixture.execute(
        """
        INSERT INTO admin_webhook_runtime_heartbeats (
            component, instance_id, ready, reason_code, heartbeat_at,
            last_success_at, created_at, updated_at
        ) VALUES ('worker', ?, TRUE, NULL, ?, ?, ?, ?)
        """,
        canonical_uuid4("persisted-runtime-instance").upper(),
        NOW,
        NOW,
        NOW,
        NOW,
    )
    with pytest.raises(ValueError, match="runtime instance ID"):
        await repository.list_runtime_heartbeats()


async def exercise_persisted_coordinate_matrix(
    fixture: DeliveryRepositoryFixture,
) -> None:
    repository = fixture.repository

    event = event_insert(
        event_id=canonical_uuid4("persisted-event-probe"),
        source_identity="persisted-event-command",
        event_type="contract.persisted-event-coordinate",
    )
    malformed_event_id = canonical_uuid4("persisted-event-coordinate").upper()
    await fixture.execute(
        """
        INSERT INTO admin_webhook_events (
            id, event_type, api_version, source_kind, aggregate_type,
            aggregate_id, aggregate_version, source_command_id,
            source_component, source_request_id, body_ciphertext_json,
            body_key_id, body_size_bytes, created_at
        ) VALUES (?, ?, ?, ?, NULL, NULL, NULL, ?, ?, ?, ?, ?, ?, ?)
        """,
        malformed_event_id,
        event.event_type,
        event.api_version,
        event.source_kind.value,
        event.source_command_id,
        event.source_component,
        event.source_request_id,
        event.body.ciphertext_json,
        event.body.key_id,
        event.body_size_bytes,
        event.created_at,
    )
    async with repository.transaction() as tx:
        with pytest.raises(ValueError, match="persisted event ID"):
            await tx._event_by_source(event)
    await fixture.execute(
        "DELETE FROM admin_webhook_events WHERE id = ?", malformed_event_id
    )

    webhook_id, delivery_id = await _captured_delivery(
        repository,
        event_id="persisted-delivery-coordinate",
        command_id="persisted-delivery-command",
        isolated=True,
    )
    event_id = str(
        await fixture.fetchval(
            "SELECT event_id FROM admin_webhook_deliveries WHERE id = ?", delivery_id
        )
    )
    malformed_delivery_id = "00000000-0000-1000-8000-000000000002"
    await fixture.execute(
        "UPDATE admin_webhook_deliveries SET id = ? WHERE id = ?",
        malformed_delivery_id,
        delivery_id,
    )
    with pytest.raises(ValueError, match="persisted delivery ID"):
        await repository.list_delivery_history(webhook_id, limit=10)
    await fixture.execute(
        "UPDATE admin_webhook_deliveries SET id = ? WHERE id = ?",
        delivery_id,
        malformed_delivery_id,
    )

    test_delivery_id = canonical_uuid4("persisted-test-delivery")
    valid_test_token = opaque_token("persisted-test-token")
    test_attempt_id = canonical_uuid4("persisted-test-attempt")
    async with repository.transaction() as tx:
        await tx.insert_delivery(
            test_delivery_id,
            event_id=event_id,
            webhook_id=webhook_id,
            kind=DeliveryKind.TEST,
            expires_at=NOW + timedelta(hours=72),
            now=NOW,
        )
        reservation = await tx.reserve_test_attempt(
            valid_test_token,
            test_delivery_id,
            test_attempt_id,
            10,
            NOW,
        )
        assert reservation is not None and reservation.reserved
    await fixture.execute(
        "UPDATE admin_webhook_delivery_attempts SET test_attempt_token = ? WHERE id = ?",
        "A" * 64,
        test_attempt_id,
    )
    with pytest.raises(ValueError, match="persisted test attempt token"):
        await repository.list_delivery_attempts(webhook_id, test_delivery_id)
    await fixture.execute(
        "UPDATE admin_webhook_delivery_attempts SET test_attempt_token = ? WHERE id = ?",
        valid_test_token,
        test_attempt_id,
    )

    _, attempt_delivery_id, attempt_id, disposition_token = (
        await _prepare_retry_disposition(fixture, "persisted-attempt-coordinate")
    )
    malformed_attempt_id = canonical_uuid4("persisted-attempt-coordinate").upper()
    await fixture.execute(
        "UPDATE admin_webhook_delivery_attempts SET id = ? WHERE id = ?",
        malformed_attempt_id,
        attempt_id,
    )
    with pytest.raises(ValueError, match="persisted attempt ID"):
        await repository.list_delivery_attempts(
            int(
                await fixture.fetchval(
                    "SELECT webhook_id FROM admin_webhook_deliveries WHERE id = ?",
                    attempt_delivery_id,
                )
            ),
            attempt_delivery_id,
        )
    await fixture.execute(
        "UPDATE admin_webhook_delivery_attempts SET id = ? WHERE id = ?",
        attempt_id,
        malformed_attempt_id,
    )

    malformed_redelivery = canonical_uuid4("persisted-redelivery").upper()
    with pytest.raises(fixture.integrity_error):
        await fixture.execute(
            "UPDATE admin_webhook_deliveries SET redelivery_of_id = ? WHERE id = ?",
            malformed_redelivery,
            delivery_id,
        )
    assert (
        await fixture.fetchval(
            "SELECT redelivery_of_id FROM admin_webhook_deliveries WHERE id = ?",
            delivery_id,
        )
        is None
    )

    with pytest.raises(fixture.integrity_error):
        await fixture.execute(
            """
            UPDATE admin_webhook_deliveries
            SET pending_jobs_disposition_token = ?
            WHERE id = ?
            """,
            "A" * 64,
            attempt_delivery_id,
        )
    assert (
        await fixture.fetchval(
            "SELECT pending_jobs_disposition_token FROM admin_webhook_deliveries WHERE id = ?",
            attempt_delivery_id,
        )
        == disposition_token
    )


async def exercise_disposition_scheduling_persistence(
    fixture: DeliveryRepositoryFixture,
) -> None:
    cases = (
        (
            JobsDispositionKind.RETRY,
            AttemptState.RETRYABLE,
            DeliveryState.RETRY_WAIT,
            30,
            NOW + timedelta(minutes=10),
            "queued",
        ),
        (
            JobsDispositionKind.DEFER,
            AttemptState.OUTCOME_UNKNOWN,
            DeliveryState.RETRY_WAIT,
            None,
            NOW + timedelta(minutes=10),
            "queued",
        ),
        (
            JobsDispositionKind.COMPLETE,
            AttemptState.SUCCEEDED,
            DeliveryState.SUCCEEDED,
            None,
            None,
            "completed",
        ),
        (
            JobsDispositionKind.FAIL,
            AttemptState.FAILED,
            DeliveryState.DEAD,
            None,
            None,
            "failed",
        ),
        (
            JobsDispositionKind.CANCEL,
            AttemptState.CANCELED,
            DeliveryState.CANCELED,
            None,
            None,
            "cancelled",
        ),
    )
    for kind, attempt_state, delivery_state, delay, not_before, jobs_state in cases:
        label = f"schedule-{kind.value}"
        webhook_id, delivery_id, jobs_job_id = await _queue_delivery(fixture, label)
        attempt_id = canonical_uuid4(f"{label}-attempt")
        lease_id = f"{label}-lease"
        disposition_token = opaque_token(f"{label}-disposition")
        async with fixture.repository.transaction() as tx:
            reservation = await tx.reserve_jobs_attempt(
                delivery_id,
                jobs_job_id,
                lease_id,
                attempt_id,
                10,
                NOW + timedelta(minutes=8),
                NOW + timedelta(minutes=8, seconds=10),
            )
            assert reservation is not None and reservation.reserved
            pending = await tx.finish_attempt_and_prepare_disposition(
                lease_id,
                AttemptCompletion(
                    attempt_state=attempt_state,
                    delivery_state=delivery_state,
                    disposition=kind,
                    status_code=204 if kind is JobsDispositionKind.COMPLETE else 503,
                    latency_ms=5,
                    reason_code=None,
                    requested_retry_delay_seconds=delay,
                    finished_at=NOW + timedelta(minutes=8, seconds=1),
                    completed_after_config_change=False,
                ),
                disposition_token,
                not_before,
            )
            assert pending is not None
            assert pending.not_before_at == not_before
            assert pending.delay_seconds == delay
            assert await tx.acknowledge_jobs_disposition(
                delivery_id, disposition_token, jobs_state
            )
        stored = await fixture.repository.get_delivery_for_registration(
            webhook_id, delivery_id
        )
        assert stored is not None and stored.state is delivery_state
