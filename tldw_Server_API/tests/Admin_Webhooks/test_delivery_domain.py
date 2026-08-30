# The backend fixture functions must retain their imported names for pytest discovery.
# ruff: noqa: F401, F811

from __future__ import annotations

import base64
import importlib
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from dataclasses import fields, is_dataclass, replace
from datetime import datetime, timedelta, timezone
from types import ModuleType
from typing import Protocol

import pytest

from tldw_Server_API.app.core.Admin_Webhooks.crypto import (
    EVENT_BODY_MAX_BYTES,
    WebhookKeyLoadCode,
    WebhookKeyRing,
    WebhookKeyRingLoadResult,
)
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    DeliveryReasonCode,
    EventSourceKind,
    WebhookError,
    WebhookErrorCode,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import MandatoryAuditWriteError
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
    AdminWebhookUnitOfWork,
    RegistrationInsert,
    RegistrationTarget,
)
from tldw_Server_API.tests.Admin_Webhooks.test_repository_postgres import (
    PostgreSQLRepositoryFixture,
    pg_repo,
)
from tldw_Server_API.tests.Admin_Webhooks.test_repository_sqlite import (
    SQLiteRepositoryFixture,
    sqlite_repo,
)

pytest_plugins = ("tldw_Server_API.tests.AuthNZ.conftest",)

NOW = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)
API_VERSION = "2026-07-01"
KEY_ID = "key-2026-08"


class _RepositoryFixture(Protocol):
    repository: AdminWebhookRepository


class _DeterministicDependencies:
    def __init__(self, label: str) -> None:
        self._label = label
        self._event_ordinal = 0
        self._delivery_ordinal = 0
        self._clock_ordinal = 0
        self.delivery_ids: list[str] = []

    @staticmethod
    def _uuid4(label: str) -> str:
        import hashlib

        digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
        return (
            f"{digest[:8]}-{digest[8:12]}-4{digest[13:16]}-"
            f"8{digest[17:20]}-{digest[20:32]}"
        )

    def event_id(self) -> str:
        value = self._uuid4(f"{self._label}-event-{self._event_ordinal}")
        self._event_ordinal += 1
        return value

    def delivery_id(self) -> str:
        value = self._uuid4(f"{self._label}-delivery-{self._delivery_ordinal}")
        self._delivery_ordinal += 1
        self.delivery_ids.append(value)
        return value

    def now(self) -> datetime:
        value = NOW + timedelta(minutes=self._clock_ordinal)
        self._clock_ordinal += 1
        return value


def _delivery_module() -> ModuleType:
    return importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks.delivery"
    )


def _ring(*, key_byte: bytes = b"k") -> WebhookKeyRing:
    return WebhookKeyRing(
        {KEY_ID: base64.b64encode(key_byte * 32).decode("ascii")},
        primary_id=KEY_ID,
    )


def _available(ring: WebhookKeyRing) -> WebhookKeyRingLoadResult:
    return WebhookKeyRingLoadResult(
        ring=ring,
        code=WebhookKeyLoadCode.AVAILABLE,
    )


async def _complete_migration(repository: AdminWebhookRepository) -> None:
    current = await repository.get_migration_state()
    digest = "sha256:" + ("a" * 64)
    fingerprint = "hmac-sha256:" + ("b" * 64)
    async with repository.transaction() as tx:
        await tx.compare_and_set_migration_state(
            expected_revision=current.state_revision,
            updates={
                "phase": "complete",
                "import_operation_id": "whmig_" + ("c" * 32),
                "import_operator_id": 7,
                "import_started_at": NOW,
                "import_approved_at": NOW,
                "artifacts_ready_at": NOW,
                "database_committed_at": NOW,
                "fingerprint_key_id": KEY_ID,
                "completed_at": NOW,
                "active_primary_key_id": KEY_ID,
                "system_ops_webhook_fingerprint": fingerprint,
                "legacy_table_fingerprint": fingerprint,
                "redacted_report_digest": digest,
                "protected_backup_ciphertext_digest": digest,
                "active_report_path": "/srv/tldw/webhook-report.json",
                "active_backup_path": "/srv/tldw/webhook-backup.enc",
                "active_key_path": "/srv/tldw/webhook-backup.key",
                "staging_report_path": "/srv/tldw/webhook-report.json.staging",
                "staging_backup_path": "/srv/tldw/webhook-backup.enc.staging",
                "staging_key_path": "/srv/tldw/webhook-backup.key.staging",
                "report_owner_id": 1000,
                "report_group_id": 1000,
                "report_mode": 384,
                "report_file_identity": "1048576:41",
                "backup_owner_id": 1000,
                "backup_group_id": 1000,
                "backup_mode": 384,
                "backup_file_identity": "1048576:42",
                "rollback_key_owner_id": 1000,
                "rollback_key_group_id": 1000,
                "rollback_key_mode": 384,
                "rollback_key_file_identity": "1048576:43",
                "rollback_expires_at": NOW + timedelta(days=7),
                "rollback_retirement_phase": "retained",
                "expected_ciphertext_digest": digest,
            },
            at=NOW,
        )


async def _seed_registration(
    repository: AdminWebhookRepository,
    ring: WebhookKeyRing,
    *,
    event_types: tuple[str, ...],
    active: bool,
) -> int:
    async with repository.transaction() as tx:
        webhook_id = await tx.allocate_registration_id()
        target = ring.encrypt_text(
            purpose="registration.target",
            identity={"registration_id": webhook_id, "target_version": 1},
            plaintext="https://hooks.example.com/capture",
        )
        secret = ring.encrypt_text(
            purpose="registration.secret",
            identity={"registration_id": webhook_id, "secret_version": 1},
            plaintext="whsec_" + ("1" * 64),
        )
        await tx.insert_registration(
            RegistrationInsert(
                id=webhook_id,
                description="Synthetic capture receiver",
                target=RegistrationTarget(
                    protected=target,
                    hostname="hooks.example.com",
                    display="https://hooks.example.com",
                ),
                event_types=event_types,
                active=active,
                timeout_seconds=10,
                secret=secret,
                secret_rotation_required=False,
                actor_user_id=7,
                now=NOW - timedelta(minutes=1),
            )
        )
    return webhook_id


def _service(
    fixture: _RepositoryFixture,
    *,
    label: str,
    ring: WebhookKeyRing | None = None,
    key_ring_result: WebhookKeyRingLoadResult | None = None,
):
    module = _delivery_module()
    dependencies = _DeterministicDependencies(label)
    selected_ring = ring or _ring()
    service = module.AdminWebhookDeliveryService(
        repository=fixture.repository,
        key_ring_result=key_ring_result or _available(selected_ring),
        event_id_factory=dependencies.event_id,
        delivery_id_factory=dependencies.delivery_id,
        clock=dependencies.now,
    )
    return module, selected_ring, service, dependencies


def _command(
    module: ModuleType,
    *,
    event_type: str = "user.created",
    source_kind: EventSourceKind = EventSourceKind.AGGREGATE,
    source_identity: str = "user-7",
    source_component: str = "task6-tests",
    source_request_id: str | None = "source-request-1",
    data: dict[object, object] | None = None,
):
    return module.CaptureSyntheticEventCommand(
        actor_id=7,
        request_id="capture-audit-request-1",
        event_type=event_type,
        source_kind=source_kind,
        aggregate_type="user" if source_kind is EventSourceKind.AGGREGATE else None,
        aggregate_id=source_identity if source_kind is EventSourceKind.AGGREGATE else None,
        aggregate_version="3" if source_kind is EventSourceKind.AGGREGATE else None,
        source_command_id=(
            source_identity if source_kind is EventSourceKind.COMMAND else None
        ),
        source_component=source_component,
        source_request_id=source_request_id,
        data=data if data is not None else {"synthetic": True},
    )


def _recording_sink(records: list[object]):
    async def sink(record: object) -> None:
        records.append(record)

    return sink


class _CaptureUnitOfWorkProbe:
    def __init__(
        self,
        wrapped: AdminWebhookUnitOfWork,
        *,
        locked_updates: Mapping[str, object] | None,
        probe: _CaptureRepositoryProbe,
    ) -> None:
        self._wrapped = wrapped
        self._locked_updates = locked_updates
        self._probe = probe

    def __getattr__(self, name: str) -> object:
        return getattr(self._wrapped, name)

    async def lock_migration_state(self):
        state = await self._wrapped.lock_migration_state()
        if self._locked_updates is None:
            return state
        return await self._wrapped.compare_and_set_migration_state(
            expected_revision=state.state_revision,
            updates=self._locked_updates,
            at=NOW + timedelta(minutes=1),
        )

    async def capture_event_and_expand(self, *args: object):
        self._probe.capture_calls += 1
        return await self._wrapped.capture_event_and_expand(*args)


class _CaptureRepositoryProbe:
    def __init__(
        self,
        wrapped: AdminWebhookRepository,
        *,
        locked_updates: Mapping[str, object] | None = None,
    ) -> None:
        self._wrapped = wrapped
        self._locked_updates = locked_updates
        self.transaction_calls = 0
        self.capture_calls = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self._wrapped, name)

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[_CaptureUnitOfWorkProbe]:
        self.transaction_calls += 1
        async with self._wrapped.transaction() as tx:
            yield _CaptureUnitOfWorkProbe(
                tx,
                locked_updates=self._locked_updates,
                probe=self,
            )


@pytest.mark.unit
def test_capture_command_and_audit_are_frozen_closed_internal_records() -> None:
    module = _delivery_module()
    package = importlib.import_module(
        "tldw_Server_API.app.core.Admin_Webhooks"
    )
    aggregate = _command(module)
    command = _command(
        module,
        source_kind=EventSourceKind.COMMAND,
        source_identity="command-7",
    )

    assert is_dataclass(module.CaptureSyntheticEventCommand)
    assert package.__doc__ == "Canonical admin outgoing webhook contracts."
    for internal_name in (
        "AdminWebhookDeliveryService",
        "CaptureSyntheticEventCommand",
        "EventCaptureAudit",
        "EventCaptureAuditSink",
    ):
        assert internal_name not in package.__all__
        assert not hasattr(package, internal_name)
    assert module.CaptureSyntheticEventCommand.__dataclass_params__.frozen
    assert not hasattr(module.CaptureSyntheticEventCommand, "model_fields")
    assert {field.name for field in fields(aggregate)} == {
        "actor_id",
        "request_id",
        "event_type",
        "source_kind",
        "aggregate_type",
        "aggregate_id",
        "aggregate_version",
        "source_command_id",
        "source_component",
        "source_request_id",
        "data",
    }
    assert not {
        "event_id",
        "created_at",
        "delivery_id",
        "target",
        "headers",
        "method",
        "signing_input",
        "response",
    } & {field.name for field in fields(aggregate)}
    assert aggregate.aggregate_id == "user-7"
    assert command.source_command_id == "command-7"

    audit = module.EventCaptureAudit(
        event_type="user.created",
        event_id=None,
        fanout_count=0,
        actor_id=7,
        request_id="capture-audit-request-1",
        outcome="failed",
        reason_code=WebhookErrorCode.VALIDATION_FAILED,
    )
    assert audit.__dataclass_params__.frozen
    assert {field.name for field in fields(audit)} == {
        "event_type",
        "event_id",
        "fanout_count",
        "actor_id",
        "request_id",
        "outcome",
        "reason_code",
    }

    with pytest.raises(ValueError, match="source identity"):
        replace(aggregate, source_command_id="also-command")
    with pytest.raises(ValueError, match="source identity"):
        replace(command, source_command_id=None)
    with pytest.raises(ValueError, match="event type"):
        replace(aggregate, event_type="webhook.test")
    with pytest.raises(ValueError, match="source component"):
        replace(aggregate, source_component="x" * 65)
    with pytest.raises(ValueError, match="data"):
        replace(aggregate, data=[])


async def _exercise_capture_and_replay(fixture: _RepositoryFixture) -> None:
    await _complete_migration(fixture.repository)
    module, ring, service, dependencies = _service(fixture, label="capture")
    matching_id = await _seed_registration(
        fixture.repository,
        ring,
        event_types=("user.created",),
        active=True,
    )
    await _seed_registration(
        fixture.repository,
        ring,
        event_types=("user.deleted",),
        active=True,
    )
    records: list[object] = []
    command = _command(
        module,
        data={"unicode": "caf\u00e9", "a": 1},
    )

    captured = await service.capture_synthetic_event(
        command,
        audit_sink=_recording_sink(records),
    )

    expected_body = (
        b'{"api_version":"2026-07-01","created_at":"2026-08-23T12:00:00Z",'
        b'"data":{"a":1,"unicode":"caf\xc3\xa9"},"id":"'
        + captured.event.id.encode("ascii")
        + b'","type":"user.created"}'
    )
    plaintext = ring.decrypt_event_body(
        event_id=captured.event.id,
        api_version=API_VERSION,
        protected=captured.event.body,
    )
    assert plaintext == expected_body
    assert captured.event.body_size_bytes == len(expected_body)
    assert captured.inserted is True
    assert [item.delivery.webhook_id for item in captured.deliveries] == [matching_id]
    assert captured.event.aggregate_type == "user"
    assert captured.event.aggregate_id == "user-7"
    assert captured.event.aggregate_version == "3"
    assert captured.event.source_command_id is None
    assert captured.event.source_component == "task6-tests"
    assert captured.event.source_request_id == "source-request-1"
    assert records == [
        module.EventCaptureAudit(
            event_type="user.created",
            event_id=captured.event.id,
            fanout_count=1,
            actor_id=7,
            request_id="capture-audit-request-1",
            outcome="accepted",
            reason_code=None,
        )
    ]

    generated_before_replay = tuple(dependencies.delivery_ids)
    replayed = await service.capture_synthetic_event(
        command,
        audit_sink=_recording_sink(records),
    )
    assert replayed.inserted is False
    assert replayed.event.id == captured.event.id
    assert [item.delivery.id for item in replayed.deliveries] == [
        item.delivery.id for item in captured.deliveries
    ]
    assert tuple(dependencies.delivery_ids) == generated_before_replay
    assert records[-1].outcome == "accepted"
    assert records[-1].event_id == captured.event.id
    assert records[-1].fanout_count == 1

    unmatched = await service.capture_synthetic_event(
        _command(
            module,
            event_type="incident.notify",
            source_kind=EventSourceKind.COMMAND,
            source_identity="notify-command-1",
            data={"synthetic": True},
        ),
        audit_sink=_recording_sink(records),
    )
    assert unmatched.inserted is True
    assert unmatched.deliveries == ()
    assert unmatched.event.source_command_id == "notify-command-1"
    assert unmatched.event.aggregate_id is None

    migration = await fixture.repository.get_migration_state()
    assert migration.first_canonical_activity_kind == "event_capture"
    assert migration.first_canonical_activity_at == NOW

    registration = await fixture.repository.get_registration(matching_id)
    assert registration is not None
    delivery = captured.deliveries[0].delivery
    assert module.registration_work_lifecycle_reason(delivery, registration) is None
    assert module.registration_work_lifecycle_reason(
        delivery,
        replace(registration, active=False, delivery_config_version=2),
    ) is DeliveryReasonCode.CANCELED_DISABLED
    assert module.registration_work_lifecycle_reason(
        delivery,
        replace(registration, secret_version=2, delivery_config_version=2),
    ) is DeliveryReasonCode.CANCELED_SECRET_ROTATION
    assert module.registration_work_lifecycle_reason(
        delivery,
        replace(registration, delivery_config_version=2),
    ) is DeliveryReasonCode.SUPERSEDED_CONFIG
    assert module.registration_work_lifecycle_reason(
        delivery,
        replace(
            registration,
            active=False,
            secret_version=2,
            delivery_config_version=2,
            deleted_at=NOW,
        ),
    ) is DeliveryReasonCode.CANCELED_DELETED


@pytest.mark.unit
async def test_sqlite_synthetic_capture_replay_and_fanout(
    sqlite_repo: SQLiteRepositoryFixture,
) -> None:
    await _exercise_capture_and_replay(sqlite_repo)


@pytest.mark.unit
async def test_capture_metrics_emit_once_only_after_new_durable_commit(
    sqlite_repo: SQLiteRepositoryFixture,
) -> None:
    await _complete_migration(sqlite_repo.repository)
    module = _delivery_module()
    ring = _ring()
    dependencies = _DeterministicDependencies("metrics")
    observations: list[tuple[str, int]] = []

    class Metrics:
        def events_committed(self, *, event_type: str, fanout_count: int) -> None:
            observations.append((event_type, fanout_count))

    service = module.AdminWebhookDeliveryService(
        repository=sqlite_repo.repository,
        key_ring_result=_available(ring),
        event_id_factory=dependencies.event_id,
        delivery_id_factory=dependencies.delivery_id,
        clock=dependencies.now,
        metrics=Metrics(),
    )
    await _seed_registration(
        sqlite_repo.repository,
        ring,
        event_types=("user.created",),
        active=True,
    )
    command = _command(module)

    await service.capture_synthetic_event(command, audit_sink=_recording_sink([]))
    await service.capture_synthetic_event(command, audit_sink=_recording_sink([]))

    assert observations == [("user.created", 1)]


@pytest.mark.integration
@pytest.mark.postgres
async def test_postgres_synthetic_capture_replay_and_fanout(
    pg_repo: PostgreSQLRepositoryFixture,
) -> None:
    await _exercise_capture_and_replay(pg_repo)


@pytest.mark.unit
async def test_canonical_body_accepts_65536_bytes_and_rejects_one_more(
    sqlite_repo: SQLiteRepositoryFixture,
) -> None:
    await _complete_migration(sqlite_repo.repository)
    module, ring, service, _dependencies = _service(sqlite_repo, label="boundary")
    event_id = _DeterministicDependencies._uuid4("boundary-event-0")
    prefix = (
        b'{"api_version":"2026-07-01","created_at":"2026-08-23T12:00:00Z",'
        b'"data":{"blob":"'
    )
    suffix = (
        b'"},"id":"'
        + event_id.encode("ascii")
        + b'","type":"incident.notify"}'
    )
    blob_size = EVENT_BODY_MAX_BYTES - len(prefix) - len(suffix)
    records: list[object] = []

    accepted = await service.capture_synthetic_event(
        _command(
            module,
            event_type="incident.notify",
            source_kind=EventSourceKind.COMMAND,
            source_identity="boundary-accepted",
            data={"blob": "x" * blob_size},
        ),
        audit_sink=_recording_sink(records),
    )
    accepted_body = ring.decrypt_event_body(
        event_id=accepted.event.id,
        api_version=API_VERSION,
        protected=accepted.event.body,
    )
    assert accepted_body == prefix + (b"x" * blob_size) + suffix
    assert len(accepted_body) == EVENT_BODY_MAX_BYTES

    with pytest.raises(WebhookError) as oversized:
        await service.capture_synthetic_event(
            _command(
                module,
                event_type="incident.notify",
                source_kind=EventSourceKind.COMMAND,
                source_identity="boundary-rejected",
                data={"blob": "x" * (blob_size + 1)},
            ),
            audit_sink=_recording_sink(records),
        )
    assert oversized.value.code is WebhookErrorCode.VALIDATION_FAILED
    assert records[-1].outcome == "failed"
    assert records[-1].reason_code is WebhookErrorCode.VALIDATION_FAILED

    retried = await service.capture_synthetic_event(
        _command(
            module,
            event_type="incident.notify",
            source_kind=EventSourceKind.COMMAND,
            source_identity="boundary-rejected",
            data={"synthetic": True},
        ),
        audit_sink=_recording_sink(records),
    )
    assert retried.inserted is True


@pytest.mark.unit
@pytest.mark.parametrize(
    "data",
    (
        {"invalid": float("nan")},
        {1: "non-string-key"},
        {"invalid": {1, 2}},
    ),
)
async def test_non_json_event_data_fails_before_persistence_with_one_audit(
    sqlite_repo: SQLiteRepositoryFixture,
    data: dict[object, object],
) -> None:
    await _complete_migration(sqlite_repo.repository)
    module, _ring_value, service, _dependencies = _service(
        sqlite_repo,
        label="invalid-json",
    )
    records: list[object] = []

    with pytest.raises(WebhookError) as invalid:
        await service.capture_synthetic_event(
            _command(
                module,
                source_kind=EventSourceKind.COMMAND,
                source_identity="invalid-json-command",
                data=data,
            ),
            audit_sink=_recording_sink(records),
        )

    assert invalid.value.code is WebhookErrorCode.VALIDATION_FAILED
    assert len(records) == 1
    assert records[0].outcome == "failed"
    assert records[0].reason_code is WebhookErrorCode.VALIDATION_FAILED


@pytest.mark.unit
async def test_source_replay_verifies_body_and_every_persisted_source_field(
    sqlite_repo: SQLiteRepositoryFixture,
) -> None:
    await _complete_migration(sqlite_repo.repository)
    module, ring, service, _dependencies = _service(sqlite_repo, label="conflict")
    await _seed_registration(
        sqlite_repo.repository,
        ring,
        event_types=("user.created",),
        active=True,
    )
    records: list[object] = []
    original = _command(module)
    captured = await service.capture_synthetic_event(
        original,
        audit_sink=_recording_sink(records),
    )

    conflicting_commands = (
        replace(original, data={"synthetic": False}),
        replace(original, source_component="different-component"),
        replace(original, source_request_id="different-source-request"),
    )
    for command in conflicting_commands:
        with pytest.raises(WebhookError) as conflict:
            await service.capture_synthetic_event(
                command,
                audit_sink=_recording_sink(records),
            )
        assert conflict.value.code is WebhookErrorCode.IDEMPOTENCY_CONFLICT
        assert records[-1].outcome == "failed"
        assert records[-1].event_id == captured.event.id
        assert records[-1].reason_code is WebhookErrorCode.IDEMPOTENCY_CONFLICT

    wrong_ring_service = module.AdminWebhookDeliveryService(
        repository=sqlite_repo.repository,
        key_ring_result=_available(_ring(key_byte=b"z")),
        event_id_factory=_DeterministicDependencies("wrong-ring").event_id,
        delivery_id_factory=_DeterministicDependencies("wrong-ring").delivery_id,
        clock=lambda: NOW + timedelta(hours=1),
    )
    with pytest.raises(WebhookError) as decrypt_conflict:
        await wrong_ring_service.capture_synthetic_event(
            original,
            audit_sink=_recording_sink(records),
        )
    assert decrypt_conflict.value.code is WebhookErrorCode.IDEMPOTENCY_CONFLICT
    assert records[-1].outcome == "failed"

    history = await sqlite_repo.repository.list_delivery_history(
        captured.deliveries[0].delivery.webhook_id,
        limit=10,
        offset=0,
    )
    assert history.total == 1


async def _exercise_audit_rollback(fixture: _RepositoryFixture) -> None:
    await _complete_migration(fixture.repository)
    module, ring, service, _dependencies = _service(fixture, label="rollback")
    webhook_id = await _seed_registration(
        fixture.repository,
        ring,
        event_types=("user.created",),
        active=True,
    )
    command = _command(module)
    calls = 0

    async def unavailable(_record: object) -> None:
        nonlocal calls
        calls += 1
        raise MandatoryAuditWriteError("audit unavailable")

    with pytest.raises(WebhookError) as failed:
        await service.capture_synthetic_event(command, audit_sink=unavailable)
    assert failed.value.code is WebhookErrorCode.AUDIT_UNAVAILABLE
    assert calls == 1
    assert (
        await fixture.repository.list_delivery_history(
            webhook_id,
            limit=10,
            offset=0,
        )
    ).total == 0
    assert (
        await fixture.repository.get_migration_state()
    ).first_canonical_activity_at is None

    retried = await service.capture_synthetic_event(
        command,
        audit_sink=_recording_sink([]),
    )
    assert retried.inserted is True
    assert len(retried.deliveries) == 1


@pytest.mark.unit
async def test_sqlite_capture_audit_failure_rolls_back_event_fanout_and_activity(
    sqlite_repo: SQLiteRepositoryFixture,
) -> None:
    await _exercise_audit_rollback(sqlite_repo)


@pytest.mark.integration
@pytest.mark.postgres
async def test_postgres_capture_audit_failure_rolls_back_event_fanout_and_activity(
    pg_repo: PostgreSQLRepositoryFixture,
) -> None:
    await _exercise_audit_rollback(pg_repo)


@pytest.mark.unit
async def test_pretransaction_key_failures_emit_one_failed_audit(
    sqlite_repo: SQLiteRepositoryFixture,
) -> None:
    await _complete_migration(sqlite_repo.repository)
    module = _delivery_module()
    dependencies = _DeterministicDependencies("missing-key")
    service = module.AdminWebhookDeliveryService(
        repository=sqlite_repo.repository,
        key_ring_result=WebhookKeyRingLoadResult(
            ring=None,
            code=WebhookKeyLoadCode.KEY_UNAVAILABLE,
        ),
        event_id_factory=dependencies.event_id,
        delivery_id_factory=dependencies.delivery_id,
        clock=dependencies.now,
    )
    records: list[object] = []

    with pytest.raises(WebhookError) as unavailable:
        await service.capture_synthetic_event(
            _command(module),
            audit_sink=_recording_sink(records),
        )

    assert unavailable.value.code is WebhookErrorCode.KEY_UNAVAILABLE
    assert len(records) == 1
    assert records[0].outcome == "failed"
    assert records[0].event_id == _DeterministicDependencies._uuid4(
        "missing-key-event-0"
    )
    assert records[0].fanout_count == 0
    assert records[0].reason_code is WebhookErrorCode.KEY_UNAVAILABLE


async def _set_migration_state(
    repository: AdminWebhookRepository,
    updates: Mapping[str, object],
) -> None:
    current = await repository.get_migration_state()
    async with repository.transaction() as tx:
        await tx.compare_and_set_migration_state(
            expected_revision=current.state_revision,
            updates=updates,
            at=NOW + timedelta(minutes=1),
        )


async def _assert_capture_key_gate(
    sqlite_repo: SQLiteRepositoryFixture,
    *,
    reason_code: WebhookErrorCode,
    precheck_updates: Mapping[str, object] | None = None,
    locked_updates: Mapping[str, object] | None = None,
) -> None:
    await _complete_migration(sqlite_repo.repository)
    ring = _ring()
    webhook_id = await _seed_registration(
        sqlite_repo.repository,
        ring,
        event_types=("user.created",),
        active=True,
    )
    if precheck_updates is not None:
        await _set_migration_state(sqlite_repo.repository, precheck_updates)
    module = _delivery_module()
    dependencies = _DeterministicDependencies(f"key-gate-{reason_code.value}")
    repository = _CaptureRepositoryProbe(
        sqlite_repo.repository,
        locked_updates=locked_updates,
    )
    service = module.AdminWebhookDeliveryService(
        repository=repository,
        key_ring_result=_available(ring),
        event_id_factory=dependencies.event_id,
        delivery_id_factory=dependencies.delivery_id,
        clock=dependencies.now,
    )
    records: list[object] = []

    with pytest.raises(WebhookError) as rejected:
        await service.capture_synthetic_event(
            _command(module),
            audit_sink=_recording_sink(records),
        )

    assert rejected.value.code is reason_code
    assert len(records) == 1
    assert records[0].outcome == "failed"
    assert records[0].reason_code is reason_code
    assert records[0].fanout_count == 0
    assert not any(record.outcome == "accepted" for record in records)
    assert repository.transaction_calls == (1 if locked_updates is not None else 0)
    assert repository.capture_calls == 0
    assert (
        await sqlite_repo.repository.list_delivery_history(
            webhook_id,
            limit=10,
        )
    ).total == 0
    retention = await sqlite_repo.repository.purge_retained_rows(
        NOW + timedelta(days=100),
        NOW + timedelta(days=100),
        200,
    )
    assert retention.events == 0
    assert retention.deliveries == 0
    state = await sqlite_repo.repository.get_migration_state()
    assert state.first_canonical_activity_at is None
    assert state.first_canonical_activity_kind is None


@pytest.mark.parametrize(
    ("updates", "reason_code"),
    (
        (
            {
                "rotation_operation_id": "rotation-1",
                "rotation_source_key_id": KEY_ID,
                "rotation_target_key_id": "key-next",
                "rotation_phase": "rewriting",
                "rotation_started_at": NOW,
            },
            WebhookErrorCode.KEY_ROTATION_IN_PROGRESS,
        ),
        (
            {"active_primary_key_id": "key-other"},
            WebhookErrorCode.KEY_CONFIGURATION_MISMATCH,
        ),
    ),
    ids=("active-rotation", "primary-mismatch"),
)
@pytest.mark.unit
async def test_capture_rejects_invalid_key_state_before_transaction(
    sqlite_repo: SQLiteRepositoryFixture,
    updates: Mapping[str, object],
    reason_code: WebhookErrorCode,
) -> None:
    await _assert_capture_key_gate(
        sqlite_repo,
        reason_code=reason_code,
        precheck_updates=updates,
    )


@pytest.mark.parametrize(
    ("updates", "reason_code"),
    (
        (
            {
                "rotation_operation_id": "rotation-1",
                "rotation_source_key_id": KEY_ID,
                "rotation_target_key_id": "key-next",
                "rotation_phase": "rewriting",
                "rotation_started_at": NOW,
            },
            WebhookErrorCode.KEY_ROTATION_IN_PROGRESS,
        ),
        (
            {"active_primary_key_id": "key-other"},
            WebhookErrorCode.KEY_CONFIGURATION_MISMATCH,
        ),
    ),
    ids=("active-rotation", "primary-mismatch"),
)
@pytest.mark.unit
async def test_capture_rechecks_key_state_after_lock(
    sqlite_repo: SQLiteRepositoryFixture,
    updates: Mapping[str, object],
    reason_code: WebhookErrorCode,
) -> None:
    await _assert_capture_key_gate(
        sqlite_repo,
        reason_code=reason_code,
        locked_updates=updates,
    )
