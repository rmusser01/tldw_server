# The imported fixture must retain its name for pytest discovery.
# ruff: noqa: F401, F811

from __future__ import annotations

import asyncio
import base64
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Admin_Webhooks.catalog import EVENT_API_VERSION
from tldw_Server_API.app.core.Admin_Webhooks.config import (
    AdminWebhookMode,
    AdminWebhookSettings,
)
from tldw_Server_API.app.core.Admin_Webhooks.crypto import (
    WebhookKeyLoadCode,
    WebhookKeyRing,
    WebhookKeyRingLoadResult,
)
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    EventSourceKind,
    PendingIncidentWebhookMarker,
    WebhookError,
    WebhookErrorCode,
    build_idempotency_scope,
    canonical_request_hash,
    idempotency_lookup_digest,
)
from tldw_Server_API.app.core.Admin_Webhooks.incident_reconciler import (
    PendingIncidentEventReconciler,
)
from tldw_Server_API.app.core.Admin_Webhooks.producer import (
    AdminWebhookEventProducer,
    ProductionEventPreparation,
    build_incident_notify_data,
)
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
)
from tldw_Server_API.app.services import admin_system_ops_service as system_ops
from tldw_Server_API.tests.Admin_Webhooks.test_repository_sqlite import (
    SQLiteRepositoryFixture,
    sqlite_repo,
)

NOW = datetime(2026, 8, 31, 20, 0, tzinfo=timezone.utc)
KEY_ID = "key-2026-08"
CREATE_EVENT_ID = "10000000-0000-4000-8000-000000000001"
UPDATE_EVENT_ID = "10000000-0000-4000-8000-000000000002"
RESOLVE_EVENT_ID = "10000000-0000-4000-8000-000000000003"
REOPEN_EVENT_ID = "10000000-0000-4000-8000-000000000004"
NOTIFY_EVENT_ID = "10000000-0000-4000-8000-000000000005"


def _ring() -> WebhookKeyRing:
    return WebhookKeyRing(
        {KEY_ID: base64.b64encode(b"i" * 32).decode("ascii")},
        primary_id=KEY_ID,
    )


def _settings(mode: AdminWebhookMode = AdminWebhookMode.ON) -> AdminWebhookSettings:
    return AdminWebhookSettings(
        mode=mode,
        registration_limit=100,
        active_limit=25,
        allow_http_dev=False,
        idempotency_ttl_seconds=86_400,
        rollback_window_days=7,
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


def _producer(
    repository: AdminWebhookRepository,
    *,
    event_ids: list[str],
    ring: WebhookKeyRing | None = None,
    key_ring_result: WebhookKeyRingLoadResult | None = None,
) -> AdminWebhookEventProducer:
    ids = iter(event_ids)
    active_ring = ring or _ring()
    return AdminWebhookEventProducer(
        repository=repository,
        settings=_settings(),
        key_ring_result=key_ring_result
        or WebhookKeyRingLoadResult(
            ring=active_ring,
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        event_id_factory=lambda: next(ids),
        delivery_id_factory=lambda: "20000000-0000-4000-8000-000000000001",
        clock=lambda: NOW,
    )


def _configure_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    store_path = tmp_path / "system_ops.json"
    monkeypatch.setattr(system_ops, "_STORE_PATH", store_path)
    monkeypatch.setattr(system_ops, "_now_iso", lambda: NOW.isoformat())
    return store_path


def _markers(store_path: Path) -> list[PendingIncidentWebhookMarker]:
    store = json.loads(store_path.read_text(encoding="utf-8"))
    return [PendingIncidentWebhookMarker.from_store_record(record) for record in store["webhook_pending_events"]]


def _decrypt_marker(
    marker: PendingIncidentWebhookMarker,
    ring: WebhookKeyRing,
) -> dict[str, object]:
    plaintext = ring.decrypt_bytes(
        purpose=marker.envelope_purpose,
        identity=marker.envelope_identity,
        protected=marker.body,
    )
    return json.loads(plaintext)


@pytest.mark.unit
async def test_create_update_resolve_and_reopen_write_versioned_encrypted_markers(
    sqlite_repo: SQLiteRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_path = _configure_store(monkeypatch, tmp_path)
    ring = _ring()
    await _complete_migration(sqlite_repo.repository)
    producer = _producer(
        sqlite_repo.repository,
        event_ids=[CREATE_EVENT_ID, UPDATE_EVENT_ID, RESOLVE_EVENT_ID, REOPEN_EVENT_ID],
        ring=ring,
    )

    created = await system_ops.create_incident(
        title="Private title",
        status="investigating",
        severity="high",
        summary="Private summary",
        tags=["private-tag"],
        actor="alice_admin",
        webhook_event_producer=producer,
        source_request_id="request-create",
    )
    updated = await system_ops.update_incident(
        incident_id=created["id"],
        title="Still private",
        status=None,
        severity="critical",
        summary=None,
        tags=None,
        update_message="Private operator note",
        actor="alice_admin",
        webhook_event_producer=producer,
        source_request_id="request-update",
    )
    resolved = await system_ops.update_incident(
        incident_id=created["id"],
        title=None,
        status="resolved",
        severity=None,
        summary=None,
        tags=None,
        update_message=None,
        actor="alice_admin",
        webhook_event_producer=producer,
        source_request_id="request-resolve",
    )
    reopened = await system_ops.update_incident(
        incident_id=created["id"],
        title=None,
        status="investigating",
        severity=None,
        summary=None,
        tags=None,
        update_message=None,
        actor="alice_admin",
        webhook_event_producer=producer,
        source_request_id="request-reopen",
    )

    assert [created["version"], updated["version"], resolved["version"], reopened["version"]] == [1, 2, 3, 4]
    markers = _markers(store_path)
    assert [marker.event_type for marker in markers] == [
        "incident.created",
        "incident.updated",
        "incident.resolved",
        "incident.updated",
    ]
    assert [marker.aggregate_version for marker in markers] == ["1", "2", "3", "4"]
    assert {marker.aggregate_id for marker in markers} == {created["id"]}
    assert {marker.aggregate_type for marker in markers} == {"incident"}
    assert {marker.body.key_id for marker in markers} == {KEY_ID}

    approved_fields = {
        "incident_id",
        "state",
        "severity",
        "resource_version",
        "created_at",
        "updated_at",
        "resolved_at",
    }
    for marker in markers:
        record = marker.to_store_record()
        serialized = json.dumps(record)
        assert "body_ciphertext_json" in record
        assert "body_key_id" in record
        for forbidden in (
            "Private title",
            "Still private",
            "Private summary",
            "private-tag",
            "Private operator note",
            "timeline",
            "recipients",
        ):
            assert forbidden not in serialized
        body = _decrypt_marker(marker, ring)
        assert body["api_version"] == EVENT_API_VERSION
        assert set(body["data"]) == approved_fields


@pytest.mark.unit
async def test_noop_patch_does_not_write_bump_or_consume_event_identity(
    sqlite_repo: SQLiteRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_path = _configure_store(monkeypatch, tmp_path)
    await _complete_migration(sqlite_repo.repository)
    producer = _producer(
        sqlite_repo.repository,
        event_ids=[CREATE_EVENT_ID, RESOLVE_EVENT_ID, UPDATE_EVENT_ID],
    )
    created = await system_ops.create_incident(
        title="Queue backlog",
        status="open",
        severity="high",
        summary=None,
        tags=["queue"],
        actor="alice_admin",
        webhook_event_producer=producer,
    )
    before_noop = store_path.read_bytes()

    unchanged = await system_ops.update_incident(
        incident_id=created["id"],
        title="Queue backlog",
        status="open",
        severity="high",
        summary=None,
        tags=["queue"],
        update_message=None,
        actor="alice_admin",
        webhook_event_producer=producer,
    )
    assert store_path.read_bytes() == before_noop
    changed = await system_ops.add_incident_event(
        incident_id=created["id"],
        message="Operator timeline note",
        actor="alice_admin",
        webhook_event_producer=producer,
    )

    assert unchanged["version"] == 1
    assert changed["version"] == 2
    markers = _markers(store_path)
    assert [marker.event_id for marker in markers] == [CREATE_EVENT_ID, UPDATE_EVENT_ID]


@pytest.mark.unit
@pytest.mark.parametrize(
    ("failure", "expected_code"),
    (
        ("migration", WebhookErrorCode.MIGRATION_PENDING),
        ("missing_key", WebhookErrorCode.KEY_UNAVAILABLE),
        ("key_mismatch", WebhookErrorCode.KEY_CONFIGURATION_MISMATCH),
        ("rotation", WebhookErrorCode.KEY_ROTATION_IN_PROGRESS),
    ),
)
async def test_invalid_mode_on_preflight_never_enters_store_lock(
    sqlite_repo: SQLiteRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
    expected_code: WebhookErrorCode,
) -> None:
    _configure_store(monkeypatch, tmp_path)
    entered = False
    original = system_ops._locked_store

    def tracked_lock(*args: object, **kwargs: object):
        nonlocal entered
        entered = True
        return original(*args, **kwargs)

    monkeypatch.setattr(system_ops, "_locked_store", tracked_lock)
    if failure != "migration":
        await _complete_migration(sqlite_repo.repository)
    if failure == "missing_key":
        producer = _producer(
            sqlite_repo.repository,
            event_ids=[CREATE_EVENT_ID],
            key_ring_result=WebhookKeyRingLoadResult(
                ring=None,
                code=WebhookKeyLoadCode.KEY_UNAVAILABLE,
            ),
        )
    elif failure == "key_mismatch":
        mismatch_ring = WebhookKeyRing(
            {"key-2026-09": base64.b64encode(b"j" * 32).decode("ascii")},
            primary_id="key-2026-09",
        )
        producer = _producer(
            sqlite_repo.repository,
            event_ids=[CREATE_EVENT_ID],
            ring=mismatch_ring,
        )
    else:
        producer = _producer(sqlite_repo.repository, event_ids=[CREATE_EVENT_ID])
    if failure == "rotation":
        state = await sqlite_repo.repository.get_migration_state()
        async with sqlite_repo.repository.transaction() as tx:
            await tx.compare_and_set_migration_state(
                expected_revision=state.state_revision,
                updates={
                    "rotation_operation_id": "rotation-op-0123456789",
                    "rotation_source_key_id": KEY_ID,
                    "rotation_target_key_id": "key-2026-09",
                    "rotation_phase": "rewriting",
                    "rotation_table_cursor": "registration_targets",
                    "rotation_key_cursor": None,
                    "rotation_processed_count": 0,
                    "rotation_verified_count": 0,
                    "rotation_started_at": NOW,
                    "rotation_completed_at": None,
                },
                at=NOW,
            )

    with pytest.raises(WebhookError) as exc_info:
        await system_ops.create_incident(
            title="Must not persist",
            status="open",
            severity="high",
            summary=None,
            tags=[],
            actor="alice_admin",
            webhook_event_producer=producer,
        )

    assert exc_info.value.code is expected_code
    assert entered is False


@pytest.mark.unit
async def test_stakeholder_email_preflights_marker_before_sending(
    sqlite_repo: SQLiteRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from tldw_Server_API.app.core.AuthNZ import email_service as email_service_module

    store_path = _configure_store(monkeypatch, tmp_path)
    await _complete_migration(sqlite_repo.repository)
    created = await system_ops.create_incident(
        title="Queue backlog",
        status="open",
        severity="high",
        summary=None,
        tags=[],
        actor="alice_admin",
        webhook_event_producer=_producer(
            sqlite_repo.repository,
            event_ids=[CREATE_EVENT_ID],
        ),
    )
    before = store_path.read_bytes()
    sends: list[str] = []

    class RecordingEmailService:
        async def send_email(self, *, to_email: str, **kwargs: object) -> None:
            del kwargs
            sends.append(to_email)

    monkeypatch.setattr(
        email_service_module,
        "get_email_service",
        lambda: RecordingEmailService(),
    )
    unavailable_producer = _producer(
        sqlite_repo.repository,
        event_ids=[UPDATE_EVENT_ID],
        key_ring_result=WebhookKeyRingLoadResult(
            ring=None,
            code=WebhookKeyLoadCode.KEY_UNAVAILABLE,
        ),
    )

    with pytest.raises(WebhookError) as exc_info:
        await system_ops.notify_incident_stakeholders(
            incident_id=created["id"],
            recipients=["ops@example.com"],
            message="Investigating",
            actor="alice_admin",
            actor_id="user:7",
            idempotency_key="stakeholder-notify-preflight-0001",
            webhook_event_producer=unavailable_producer,
            source_request_id="request-stakeholder-notify",
        )

    assert exc_info.value.code is WebhookErrorCode.KEY_UNAVAILABLE
    assert sends == []
    assert store_path.read_bytes() == before


@pytest.mark.unit
async def test_atomic_publication_failure_preserves_previous_store_bytes(
    sqlite_repo: SQLiteRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_path = _configure_store(monkeypatch, tmp_path)
    original_store = system_ops._default_store()
    system_ops._atomic_write_store(store_path, original_store)
    before = store_path.read_bytes()
    await _complete_migration(sqlite_repo.repository)
    producer = _producer(sqlite_repo.repository, event_ids=[CREATE_EVENT_ID])

    def fail_publication(path: Path, store: dict[str, object]) -> None:
        del path, store
        raise OSError("injected atomic publication failure")

    monkeypatch.setattr(system_ops, "_atomic_write_store", fail_publication)
    with pytest.raises(OSError, match="injected atomic publication failure"):
        await system_ops.create_incident(
            title="Must roll back",
            status="open",
            severity="high",
            summary=None,
            tags=[],
            actor="alice_admin",
            webhook_event_producer=producer,
        )

    assert store_path.read_bytes() == before


@pytest.mark.unit
async def test_notify_command_replays_pending_marker_and_conflicts_on_new_narrative(
    sqlite_repo: SQLiteRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_path = _configure_store(monkeypatch, tmp_path)
    await _complete_migration(sqlite_repo.repository)
    producer = _producer(
        sqlite_repo.repository,
        event_ids=[
            CREATE_EVENT_ID,
            NOTIFY_EVENT_ID,
            REOPEN_EVENT_ID,
            RESOLVE_EVENT_ID,
        ],
    )
    created = await system_ops.create_incident(
        title="Private title",
        status="investigating",
        severity="high",
        summary="Private summary",
        tags=[],
        actor="alice_admin",
        webhook_event_producer=producer,
    )

    first = await system_ops.notify_incident_webhooks(
        incident_id=created["id"],
        narrative="Approved receiver narrative",
        expected_resource_version=int(created["version"]),
        actor_id="user:7",
        idempotency_key="incident-notify-key-0001",
        source_request_id="request-notify-1",
        webhook_event_producer=producer,
    )
    replay = await system_ops.notify_incident_webhooks(
        incident_id=created["id"],
        narrative="Approved receiver narrative",
        expected_resource_version=int(created["version"]),
        actor_id="user:7",
        idempotency_key="incident-notify-key-0001",
        source_request_id="request-notify-2",
        webhook_event_producer=producer,
    )

    assert first.accepted is True
    assert first.replayed is False
    assert replay.event_id == first.event_id
    assert replay.command_id == first.command_id
    assert replay.replayed is True
    assert len(_markers(store_path)) == 2

    with pytest.raises(WebhookError) as exc_info:
        await system_ops.notify_incident_webhooks(
            incident_id=created["id"],
            narrative="Different narrative",
            expected_resource_version=int(created["version"]),
            actor_id="user:7",
            idempotency_key="incident-notify-key-0001",
            source_request_id="request-notify-3",
            webhook_event_producer=producer,
        )
    assert exc_info.value.code is WebhookErrorCode.IDEMPOTENCY_CONFLICT
    assert len(_markers(store_path)) == 2


@pytest.mark.unit
async def test_notify_command_completes_audit_before_returning(
    sqlite_repo: SQLiteRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_store(monkeypatch, tmp_path)
    await _complete_migration(sqlite_repo.repository)
    producer = _producer(
        sqlite_repo.repository,
        event_ids=[CREATE_EVENT_ID, NOTIFY_EVENT_ID],
    )
    created = await system_ops.create_incident(
        title="Private title",
        status="investigating",
        severity="high",
        summary="Private summary",
        tags=[],
        actor="alice_admin",
        webhook_event_producer=producer,
    )
    audited: list[system_ops.IncidentWebhookCommandAcceptance] = []

    async def audit_sink(
        acceptance: system_ops.IncidentWebhookCommandAcceptance,
    ) -> None:
        audited.append(acceptance)

    result = await system_ops.notify_incident_webhooks(
        incident_id=created["id"],
        narrative="Approved receiver narrative",
        expected_resource_version=int(created["version"]),
        actor_id="user:7",
        idempotency_key="incident-notify-key-audit-0001",
        source_request_id="request-notify-audit-1",
        webhook_event_producer=producer,
        audit_sink=audit_sink,
    )

    assert audited == [result]


@pytest.mark.unit
async def test_notify_command_raises_typed_not_found_for_missing_incident(
    sqlite_repo: SQLiteRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_store(monkeypatch, tmp_path)
    await _complete_migration(sqlite_repo.repository)
    producer = _producer(sqlite_repo.repository, event_ids=[NOTIFY_EVENT_ID])

    with pytest.raises(WebhookError) as exc_info:
        await system_ops.notify_incident_webhooks(
            incident_id="inc-missing",
            narrative="Approved receiver narrative",
            expected_resource_version=1,
            actor_id="user:7",
            idempotency_key="incident-notify-key-missing-0001",
            source_request_id="request-notify-missing-1",
            webhook_event_producer=producer,
        )

    assert exc_info.value.code is WebhookErrorCode.NOT_FOUND


@pytest.mark.unit
async def test_async_incident_mutations_lock_store_off_event_loop(
    sqlite_repo: SQLiteRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_store(monkeypatch, tmp_path)
    await _complete_migration(sqlite_repo.repository)
    producer = _producer(
        sqlite_repo.repository,
        event_ids=[CREATE_EVENT_ID, UPDATE_EVENT_ID, REOPEN_EVENT_ID, NOTIFY_EVENT_ID],
    )
    original_locked_store = system_ops._locked_store

    @contextmanager
    def require_worker_thread(*args: object, **kwargs: object) -> Iterator[dict[str, object]]:
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        with original_locked_store(*args, **kwargs) as store:
            yield store

    monkeypatch.setattr(system_ops, "_locked_store", require_worker_thread)
    created = await system_ops.create_incident(
        title="Private title",
        status="investigating",
        severity="high",
        summary="Private summary",
        tags=[],
        actor="alice_admin",
        webhook_event_producer=producer,
    )
    updated = await system_ops.update_incident(
        incident_id=created["id"],
        title=None,
        status=None,
        severity="critical",
        summary=None,
        tags=None,
        update_message=None,
        actor="alice_admin",
        webhook_event_producer=producer,
    )
    timeline = await system_ops.add_incident_event(
        incident_id=created["id"],
        message="Operator note",
        actor="alice_admin",
        webhook_event_producer=producer,
    )
    accepted = await system_ops.notify_incident_webhooks(
        incident_id=created["id"],
        narrative="Approved receiver narrative",
        expected_resource_version=int(timeline["version"]),
        actor_id="user:7",
        idempotency_key="incident-notify-key-thread-0001",
        source_request_id="request-notify-thread-1",
        webhook_event_producer=producer,
    )

    assert updated["severity"] == "critical"
    assert timeline["timeline"][-1]["message"] == "Operator note"
    assert accepted.accepted is True


@pytest.mark.unit
async def test_notify_command_retains_marker_when_immediate_database_capture_fails(
    sqlite_repo: SQLiteRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_path = _configure_store(monkeypatch, tmp_path)
    await _complete_migration(sqlite_repo.repository)
    producer = _producer(
        sqlite_repo.repository,
        event_ids=[CREATE_EVENT_ID, NOTIFY_EVENT_ID, UPDATE_EVENT_ID],
    )
    created = await system_ops.create_incident(
        title="Private title",
        status="investigating",
        severity="high",
        summary="Private summary",
        tags=[],
        actor="alice_admin",
        webhook_event_producer=producer,
    )
    original_capture = producer.capture_incident_marker

    failure = RuntimeError("database unavailable")
    observed_log: dict[str, object] = {}

    class BoundLogger:
        def warning(self, message: str, *values: object) -> None:
            observed_log["message"] = message
            observed_log["values"] = values

    class Logger:
        def opt(self, *, exception: BaseException) -> BoundLogger:
            observed_log["exception"] = exception
            return BoundLogger()

    async def unavailable(_marker: PendingIncidentWebhookMarker) -> None:
        raise failure

    monkeypatch.setattr(producer, "capture_incident_marker", unavailable)
    monkeypatch.setattr(system_ops, "logger", Logger())
    with pytest.raises(WebhookError) as exc_info:
        await system_ops.notify_incident_webhooks(
            incident_id=created["id"],
            narrative="Approved receiver narrative",
            expected_resource_version=int(created["version"]),
            actor_id="user:7",
            idempotency_key="incident-notify-key-deferred-capture-0001",
            source_request_id="request-notify-deferred-1",
            webhook_event_producer=producer,
        )
    assert exc_info.value.code is WebhookErrorCode.OPERATION_FAILED
    assert any(marker.event_type == "incident.notify" for marker in _markers(store_path))
    assert observed_log == {
        "exception": failure,
        "message": (
            "Deferred incident webhook marker capture operation={} event_id={} source_request_id={} error_type={}"
        ),
        "values": (
            "incident.notify",
            NOTIFY_EVENT_ID,
            "request-notify-deferred-1",
            "RuntimeError",
        ),
    }

    monkeypatch.setattr(producer, "capture_incident_marker", original_capture)
    replay = await system_ops.notify_incident_webhooks(
        incident_id=created["id"],
        narrative="Approved receiver narrative",
        expected_resource_version=int(created["version"]),
        actor_id="user:7",
        idempotency_key="incident-notify-key-deferred-capture-0001",
        source_request_id="request-notify-deferred-2",
        webhook_event_producer=producer,
    )
    assert replay.event_id == NOTIFY_EVENT_ID
    assert replay.replayed is True
    with sqlite3.connect(sqlite_repo.path) as connection:
        event_count, stored_request_id = connection.execute(
            """
            SELECT COUNT(*), MAX(source_request_id)
            FROM admin_webhook_events
            WHERE event_type = 'incident.notify'
            """,
        ).fetchone()
    assert event_count == 1
    assert stored_request_id == "request-notify-deferred-1"


@pytest.mark.unit
async def test_notify_command_replays_reconciled_database_event_without_new_marker(
    sqlite_repo: SQLiteRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_path = _configure_store(monkeypatch, tmp_path)
    ring = _ring()
    await _complete_migration(sqlite_repo.repository)
    producer = _producer(
        sqlite_repo.repository,
        event_ids=[CREATE_EVENT_ID, NOTIFY_EVENT_ID, REOPEN_EVENT_ID, RESOLVE_EVENT_ID],
        ring=ring,
    )
    created = await system_ops.create_incident(
        title="Private title",
        status="investigating",
        severity="high",
        summary="Private summary",
        tags=[],
        actor="alice_admin",
        webhook_event_producer=producer,
    )
    first = await system_ops.notify_incident_webhooks(
        incident_id=created["id"],
        narrative="Approved receiver narrative",
        expected_resource_version=int(created["version"]),
        actor_id="user:7",
        idempotency_key="incident-notify-key-0002",
        source_request_id="request-notify-db-1",
        webhook_event_producer=producer,
    )
    reconciler = PendingIncidentEventReconciler(
        repository=sqlite_repo.repository,
        key_ring_result=WebhookKeyRingLoadResult(
            ring=ring,
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        settings=_settings(),
        delivery_id_factory=lambda: "20000000-0000-4000-8000-000000000097",
        store_path=store_path,
    )
    assert await reconciler.reconcile_once(limit=100) == 2

    replay = await system_ops.notify_incident_webhooks(
        incident_id=created["id"],
        narrative="Approved receiver narrative",
        expected_resource_version=int(created["version"]),
        actor_id="user:7",
        idempotency_key="incident-notify-key-0002",
        source_request_id="request-notify-db-2",
        webhook_event_producer=producer,
    )

    assert replay.event_id == first.event_id == NOTIFY_EVENT_ID
    assert replay.replayed is True
    assert all(marker.event_type != "incident.notify" for marker in _markers(store_path))
    with sqlite3.connect(sqlite_repo.path) as connection:
        event_count, stored_request_id = connection.execute(
            """
            SELECT COUNT(*), MAX(source_request_id)
            FROM admin_webhook_events
            WHERE event_type = 'incident.notify'
            """,
        ).fetchone()
    assert event_count == 1
    assert stored_request_id == "request-notify-db-1"

    with pytest.raises(WebhookError) as exc_info:
        await system_ops.notify_incident_webhooks(
            incident_id=created["id"],
            narrative="Different narrative",
            expected_resource_version=int(created["version"]),
            actor_id="user:7",
            idempotency_key="incident-notify-key-0002",
            source_request_id="request-notify-db-3",
            webhook_event_producer=producer,
        )
    assert exc_info.value.code is WebhookErrorCode.IDEMPOTENCY_CONFLICT


@pytest.mark.unit
async def test_notify_command_replays_after_incident_changes_and_rejects_stale_new_preview(
    sqlite_repo: SQLiteRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _configure_store(monkeypatch, tmp_path)
    await _complete_migration(sqlite_repo.repository)
    producer = _producer(
        sqlite_repo.repository,
        event_ids=[
            CREATE_EVENT_ID,
            NOTIFY_EVENT_ID,
            UPDATE_EVENT_ID,
            REOPEN_EVENT_ID,
            RESOLVE_EVENT_ID,
        ],
    )
    created = await system_ops.create_incident(
        title="Private title",
        status="investigating",
        severity="high",
        summary="Private summary",
        tags=[],
        actor="alice_admin",
        webhook_event_producer=producer,
    )
    expected_version = int(created["version"])
    first = await system_ops.notify_incident_webhooks(
        incident_id=created["id"],
        narrative="Approved receiver narrative",
        expected_resource_version=expected_version,
        actor_id="user:7",
        idempotency_key="incident-notify-key-mutation-0001",
        source_request_id="request-notify-before-mutation",
        webhook_event_producer=producer,
    )
    await system_ops.update_incident(
        incident_id=created["id"],
        title=None,
        status=None,
        severity="critical",
        summary=None,
        tags=None,
        update_message=None,
        actor="alice_admin",
        webhook_event_producer=producer,
    )

    replay = await system_ops.notify_incident_webhooks(
        incident_id=created["id"],
        narrative="Approved receiver narrative",
        expected_resource_version=expected_version,
        actor_id="user:7",
        idempotency_key="incident-notify-key-mutation-0001",
        source_request_id="request-notify-after-mutation",
        webhook_event_producer=producer,
    )

    assert replay.event_id == first.event_id
    assert replay.replayed is True
    with pytest.raises(WebhookError) as exc_info:
        await system_ops.notify_incident_webhooks(
            incident_id=created["id"],
            narrative="Approved receiver narrative",
            expected_resource_version=expected_version,
            actor_id="user:7",
            idempotency_key="incident-notify-key-stale-preview-0001",
            source_request_id="request-notify-stale-preview",
            webhook_event_producer=producer,
        )
    assert exc_info.value.code is WebhookErrorCode.PRECONDITION_FAILED


@pytest.mark.unit
async def test_notify_command_replays_after_incident_deletion(
    sqlite_repo: SQLiteRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_path = _configure_store(monkeypatch, tmp_path)
    ring = _ring()
    await _complete_migration(sqlite_repo.repository)
    producer = _producer(
        sqlite_repo.repository,
        event_ids=[CREATE_EVENT_ID, NOTIFY_EVENT_ID, UPDATE_EVENT_ID],
        ring=ring,
    )
    created = await system_ops.create_incident(
        title="Private title",
        status="investigating",
        severity="high",
        summary="Private summary",
        tags=[],
        actor="alice_admin",
        webhook_event_producer=producer,
    )
    first = await system_ops.notify_incident_webhooks(
        incident_id=created["id"],
        narrative="Approved receiver narrative",
        expected_resource_version=int(created["version"]),
        actor_id="user:7",
        idempotency_key="incident-notify-key-deletion-0001",
        source_request_id="request-notify-before-deletion",
        webhook_event_producer=producer,
    )
    reconciler = PendingIncidentEventReconciler(
        repository=sqlite_repo.repository,
        key_ring_result=WebhookKeyRingLoadResult(
            ring=ring,
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        settings=_settings(),
        delivery_id_factory=lambda: "20000000-0000-4000-8000-000000000098",
        store_path=store_path,
    )
    await reconciler.reconcile_once(limit=100)
    system_ops.delete_incident(incident_id=created["id"])

    replay = await system_ops.notify_incident_webhooks(
        incident_id=created["id"],
        narrative="Approved receiver narrative",
        expected_resource_version=int(created["version"]),
        actor_id="user:7",
        idempotency_key="incident-notify-key-deletion-0001",
        source_request_id="request-notify-after-deletion",
        webhook_event_producer=producer,
    )

    assert replay.event_id == first.event_id
    assert replay.replayed is True


@pytest.mark.unit
async def test_notify_marker_request_fingerprint_is_authenticated(
    sqlite_repo: SQLiteRepositoryFixture,
) -> None:
    ring = _ring()
    await _complete_migration(sqlite_repo.repository)
    producer = _producer(
        sqlite_repo.repository,
        event_ids=[NOTIFY_EVENT_ID],
        ring=ring,
    )
    scope = build_idempotency_scope(
        actor_id="user:7",
        operation="notify_incident",
        route="/admin/incidents/inc-authenticated/notify-webhooks",
    )
    key = "incident-notify-key-authenticated-0001"
    original_fingerprint = canonical_request_hash(
        key,
        scope=scope,
        body={
            "incident_id": "inc-authenticated",
            "narrative": "Original narrative",
            "expected_resource_version": 7,
        },
        conditional_version=7,
    )
    changed_fingerprint = canonical_request_hash(
        key,
        scope=scope,
        body={
            "incident_id": "inc-authenticated",
            "narrative": "Changed narrative",
            "expected_resource_version": 7,
        },
        conditional_version=7,
    )
    preparation = await producer.begin_capture(
        source_component="admin_system_ops",
        source_request_id="request-authenticated-marker",
    )
    assert preparation is not None
    marker = producer.prepare_incident_marker(
        preparation,
        event_type="incident.notify",
        source_kind=EventSourceKind.COMMAND,
        aggregate_type=None,
        aggregate_id=None,
        aggregate_version=None,
        source_command_id=idempotency_lookup_digest(key, scope),
        request_fingerprint=original_fingerprint,
        data=build_incident_notify_data(
            incident_id="inc-authenticated",
            state="investigating",
            severity="high",
            resource_version=7,
            created_at=NOW,
            updated_at=NOW,
            resolved_at=None,
            narrative="Original narrative",
        ),
    )
    relabeled = replace(marker, request_fingerprint=changed_fingerprint)

    with pytest.raises(WebhookError) as exc_info:
        producer.verify_incident_marker_replay(
            relabeled,
            request_fingerprint=changed_fingerprint,
            incident_id="inc-authenticated",
            narrative="Changed narrative",
            expected_resource_version=7,
        )
    assert exc_info.value.code is WebhookErrorCode.IDEMPOTENCY_CONFLICT

    with pytest.raises(WebhookError) as source_exc:
        producer.verify_incident_marker_replay(
            replace(marker, source_request_id="request-relabeled-marker"),
            request_fingerprint=original_fingerprint,
            incident_id="inc-authenticated",
            narrative="Original narrative",
            expected_resource_version=7,
        )
    assert source_exc.value.code is WebhookErrorCode.IDEMPOTENCY_CONFLICT


@pytest.mark.unit
async def test_incident_publication_rechecks_rotation_after_preflight(
    sqlite_repo: SQLiteRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_path = _configure_store(monkeypatch, tmp_path)
    await _complete_migration(sqlite_repo.repository)
    producer = _producer(
        sqlite_repo.repository,
        event_ids=[CREATE_EVENT_ID],
    )
    original_begin = producer.begin_capture

    async def begin_then_rotate(**kwargs):
        preparation = await original_begin(**kwargs)
        async with sqlite_repo.repository.transaction() as tx:
            state = await tx.lock_migration_state()
            await tx.compare_and_set_migration_state(
                expected_revision=state.state_revision,
                updates={
                    "rotation_operation_id": "rotation-race-0001",
                    "rotation_source_key_id": KEY_ID,
                    "rotation_target_key_id": "key-2026-09",
                    "rotation_phase": "rewriting",
                    "rotation_table_cursor": "registration_targets",
                    "rotation_started_at": NOW,
                },
                at=NOW,
            )
        return preparation

    monkeypatch.setattr(producer, "begin_capture", begin_then_rotate)

    with pytest.raises(WebhookError) as exc_info:
        await system_ops.create_incident(
            title="Rotation race",
            status="investigating",
            severity="high",
            summary="Must not publish",
            tags=[],
            actor="alice_admin",
            webhook_event_producer=producer,
        )

    assert exc_info.value.code is WebhookErrorCode.KEY_ROTATION_IN_PROGRESS
    assert not store_path.exists()


@pytest.mark.unit
@pytest.mark.parametrize("operation", ["update", "timeline", "notify"])
async def test_existing_incident_marker_publications_recheck_rotation(
    sqlite_repo: SQLiteRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    operation: str,
) -> None:
    store_path = _configure_store(monkeypatch, tmp_path)
    await _complete_migration(sqlite_repo.repository)
    producer = _producer(
        sqlite_repo.repository,
        event_ids=[CREATE_EVENT_ID, UPDATE_EVENT_ID, NOTIFY_EVENT_ID],
    )
    created = await system_ops.create_incident(
        title="Rotation race",
        status="investigating",
        severity="high",
        summary="Must remain unchanged",
        tags=[],
        actor="alice_admin",
        webhook_event_producer=producer,
    )
    before = store_path.read_bytes()
    original_begin = producer.begin_capture

    async def begin_then_rotate(**kwargs):
        preparation = await original_begin(**kwargs)
        async with sqlite_repo.repository.transaction() as tx:
            state = await tx.lock_migration_state()
            await tx.compare_and_set_migration_state(
                expected_revision=state.state_revision,
                updates={
                    "rotation_operation_id": "rotation-race-0002",
                    "rotation_source_key_id": KEY_ID,
                    "rotation_target_key_id": "key-2026-09",
                    "rotation_phase": "rewriting",
                    "rotation_table_cursor": "registration_targets",
                    "rotation_started_at": NOW,
                },
                at=NOW,
            )
        return preparation

    monkeypatch.setattr(producer, "begin_capture", begin_then_rotate)

    with pytest.raises(WebhookError) as exc_info:
        if operation == "update":
            await system_ops.update_incident(
                incident_id=created["id"],
                title=None,
                status=None,
                severity="critical",
                summary=None,
                tags=None,
                update_message=None,
                actor="alice_admin",
                webhook_event_producer=producer,
            )
        elif operation == "timeline":
            await system_ops.add_incident_event(
                incident_id=created["id"],
                message="Must not append",
                actor="alice_admin",
                webhook_event_producer=producer,
            )
        else:
            await system_ops.notify_incident_webhooks(
                incident_id=created["id"],
                narrative="Must not publish",
                expected_resource_version=int(created["version"]),
                actor_id="user:7",
                idempotency_key="incident-notify-key-rotation-race-0001",
                source_request_id="request-notify-rotation-race",
                webhook_event_producer=producer,
            )

    assert exc_info.value.code is WebhookErrorCode.KEY_ROTATION_IN_PROGRESS
    assert store_path.read_bytes() == before


@pytest.mark.unit
async def test_legacy_notify_marker_replays_and_preserves_original_request_id(
    sqlite_repo: SQLiteRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_path = _configure_store(monkeypatch, tmp_path)
    ring = _ring()
    await _complete_migration(sqlite_repo.repository)
    producer = _producer(
        sqlite_repo.repository,
        event_ids=[CREATE_EVENT_ID, UPDATE_EVENT_ID],
        ring=ring,
    )
    created = await system_ops.create_incident(
        title="Private title",
        status="investigating",
        severity="high",
        summary="Private summary",
        tags=[],
        actor="alice_admin",
        webhook_event_producer=producer,
    )
    key = "incident-notify-key-legacy-marker-0001"
    scope = build_idempotency_scope(
        actor_id="user:7",
        operation="notify_incident",
        route=f"/admin/incidents/{created['id']}/notify-webhooks",
    )
    fingerprint = canonical_request_hash(
        key,
        scope=scope,
        body={
            "incident_id": created["id"],
            "narrative": "Legacy narrative",
            "expected_resource_version": int(created["version"]),
        },
        conditional_version=int(created["version"]),
    )
    marker = producer.prepare_incident_marker(
        ProductionEventPreparation(
            event_id=NOTIFY_EVENT_ID,
            created_at=NOW,
            source_component="admin_system_ops",
            source_request_id="request-legacy-original",
        ),
        event_type="incident.notify",
        source_kind=EventSourceKind.COMMAND,
        aggregate_type=None,
        aggregate_id=None,
        aggregate_version=None,
        source_command_id=idempotency_lookup_digest(key, scope),
        request_fingerprint=fingerprint,
        data=system_ops._incident_webhook_data(
            created,
            event_type="incident.notify",
            narrative="Legacy narrative",
        ),
    )
    plaintext = ring.decrypt_bytes(
        purpose=marker.envelope_purpose,
        identity=marker.envelope_identity,
        protected=marker.body,
    )
    legacy = replace(
        marker,
        request_fingerprint=None,
        body=ring.encrypt_bytes(
            purpose=marker.envelope_purpose,
            identity=marker.legacy_envelope_identity,
            plaintext=plaintext,
        ),
    )
    legacy_record = legacy.to_store_record()
    legacy_record.pop("request_fingerprint")
    store = system_ops._load_store_strict(store_path)
    store["webhook_pending_events"].append(legacy_record)
    system_ops._atomic_write_store(store_path, store)

    replay = await system_ops.notify_incident_webhooks(
        incident_id=created["id"],
        narrative="Legacy narrative",
        expected_resource_version=int(created["version"]),
        actor_id="user:7",
        idempotency_key=key,
        source_request_id="request-legacy-retry",
        webhook_event_producer=producer,
    )

    assert replay.event_id == NOTIFY_EVENT_ID
    assert replay.replayed is True
    with sqlite3.connect(sqlite_repo.path) as connection:
        stored_request_id = connection.execute(
            "SELECT source_request_id FROM admin_webhook_events WHERE id = ?",
            (NOTIFY_EVENT_ID,),
        ).fetchone()[0]
    assert stored_request_id == "request-legacy-original"


@pytest.mark.unit
async def test_current_marker_shape_cannot_use_legacy_aad_fallback(
    sqlite_repo: SQLiteRepositoryFixture,
) -> None:
    ring = _ring()
    await _complete_migration(sqlite_repo.repository)
    producer = _producer(
        sqlite_repo.repository,
        event_ids=[NOTIFY_EVENT_ID],
        ring=ring,
    )
    fingerprint = f"hmac-sha256:{'a' * 64}"
    marker = producer.prepare_incident_marker(
        ProductionEventPreparation(
            event_id=NOTIFY_EVENT_ID,
            created_at=NOW,
            source_component="admin_system_ops",
            source_request_id="request-current-shape",
        ),
        event_type="incident.notify",
        source_kind=EventSourceKind.COMMAND,
        aggregate_type=None,
        aggregate_id=None,
        aggregate_version=None,
        source_command_id=f"sha256:{'b' * 64}",
        request_fingerprint=fingerprint,
        data=build_incident_notify_data(
            incident_id="incident-current-shape",
            state="investigating",
            severity="high",
            resource_version=7,
            created_at=NOW,
            updated_at=NOW,
            resolved_at=None,
            narrative="Approved narrative",
        ),
    )
    plaintext = ring.decrypt_bytes(
        purpose=marker.envelope_purpose,
        identity=marker.envelope_identity,
        protected=marker.body,
    )
    current_record_with_legacy_body = replace(
        marker,
        body=ring.encrypt_bytes(
            purpose=marker.envelope_purpose,
            identity=marker.legacy_envelope_identity,
            plaintext=plaintext,
        ),
    ).to_store_record()
    parsed = PendingIncidentWebhookMarker.from_store_record(
        current_record_with_legacy_body
    )

    with pytest.raises(WebhookError) as exc_info:
        producer.verify_incident_marker_replay(
            parsed,
            request_fingerprint=fingerprint,
            incident_id="incident-current-shape",
            narrative="Approved narrative",
            expected_resource_version=7,
        )

    assert exc_info.value.code is WebhookErrorCode.IDEMPOTENCY_CONFLICT


@pytest.mark.unit
async def test_notify_reconciler_race_converges_identical_marker(
    sqlite_repo: SQLiteRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_path = _configure_store(monkeypatch, tmp_path)
    ring = _ring()
    await _complete_migration(sqlite_repo.repository)
    primary = _producer(
        sqlite_repo.repository,
        event_ids=[CREATE_EVENT_ID, UPDATE_EVENT_ID],
        ring=ring,
    )
    competitor = _producer(
        sqlite_repo.repository,
        event_ids=[NOTIFY_EVENT_ID],
        ring=ring,
    )
    created = await system_ops.create_incident(
        title="Private title",
        status="investigating",
        severity="high",
        summary="Private summary",
        tags=[],
        actor="alice_admin",
        webhook_event_producer=primary,
    )
    expected_version = int(created["version"])
    key = "incident-notify-key-race-0001"
    original_find = primary.find_incident_command_replay

    competing_acceptance = None

    async def reconcile_between_lookup_and_publication(**kwargs):
        nonlocal competing_acceptance
        result = await original_find(**kwargs)
        if result is not None:
            return result
        competing_acceptance = await system_ops.notify_incident_webhooks(
            incident_id=created["id"],
            narrative="Approved receiver narrative",
            expected_resource_version=expected_version,
            actor_id="user:7",
            idempotency_key=key,
            source_request_id="request-racing-writer",
            webhook_event_producer=competitor,
        )
        reconciler = PendingIncidentEventReconciler(
            repository=sqlite_repo.repository,
            key_ring_result=WebhookKeyRingLoadResult(
                ring=ring,
                code=WebhookKeyLoadCode.AVAILABLE,
            ),
            settings=_settings(),
            delivery_id_factory=lambda: "20000000-0000-4000-8000-000000000099",
            store_path=store_path,
        )
        assert await reconciler.reconcile_once(limit=100) == 2
        return result

    monkeypatch.setattr(
        primary,
        "find_incident_command_replay",
        reconcile_between_lookup_and_publication,
    )
    accepted = await system_ops.notify_incident_webhooks(
        incident_id=created["id"],
        narrative="Approved receiver narrative",
        expected_resource_version=expected_version,
        actor_id="user:7",
        idempotency_key=key,
        source_request_id="request-racing-reader",
        webhook_event_producer=primary,
    )
    assert competing_acceptance is not None
    assert accepted.event_id == competing_acceptance.event_id
    assert accepted.replayed is True
    assert len(_markers(store_path)) == 1

    reconciler = PendingIncidentEventReconciler(
        repository=sqlite_repo.repository,
        key_ring_result=WebhookKeyRingLoadResult(
            ring=ring,
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        settings=_settings(),
        delivery_id_factory=lambda: "20000000-0000-4000-8000-000000000100",
        store_path=store_path,
    )
    assert await reconciler.reconcile_once(limit=100) == 1
    assert _markers(store_path) == []
    with sqlite3.connect(sqlite_repo.path) as connection:
        count, stored_request_id = connection.execute(
            """
            SELECT COUNT(*), MAX(source_request_id)
            FROM admin_webhook_events
            WHERE event_type = 'incident.notify'
            """,
        ).fetchone()
    assert count == 1
    assert stored_request_id == "request-racing-writer"


@pytest.mark.unit
async def test_notify_reconciler_conflicting_race_retains_losing_marker(
    sqlite_repo: SQLiteRepositoryFixture,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store_path = _configure_store(monkeypatch, tmp_path)
    ring = _ring()
    await _complete_migration(sqlite_repo.repository)
    primary = _producer(
        sqlite_repo.repository,
        event_ids=[CREATE_EVENT_ID, UPDATE_EVENT_ID],
        ring=ring,
    )
    competitor = _producer(
        sqlite_repo.repository,
        event_ids=[NOTIFY_EVENT_ID],
        ring=ring,
    )
    created = await system_ops.create_incident(
        title="Private title",
        status="investigating",
        severity="high",
        summary="Private summary",
        tags=[],
        actor="alice_admin",
        webhook_event_producer=primary,
    )
    expected_version = int(created["version"])
    key = "incident-notify-key-race-conflict-0001"
    original_find = primary.find_incident_command_replay

    async def reconcile_conflict_between_lookup_and_publication(**kwargs):
        result = await original_find(**kwargs)
        assert result is None
        await system_ops.notify_incident_webhooks(
            incident_id=created["id"],
            narrative="Competing narrative",
            expected_resource_version=expected_version,
            actor_id="user:7",
            idempotency_key=key,
            source_request_id="request-conflicting-writer",
            webhook_event_producer=competitor,
        )
        reconciler = PendingIncidentEventReconciler(
            repository=sqlite_repo.repository,
            key_ring_result=WebhookKeyRingLoadResult(
                ring=ring,
                code=WebhookKeyLoadCode.AVAILABLE,
            ),
            settings=_settings(),
            delivery_id_factory=lambda: "20000000-0000-4000-8000-000000000101",
            store_path=store_path,
        )
        await reconciler.reconcile_once(limit=100)
        return result

    monkeypatch.setattr(
        primary,
        "find_incident_command_replay",
        reconcile_conflict_between_lookup_and_publication,
    )
    with pytest.raises(WebhookError) as exc_info:
        await system_ops.notify_incident_webhooks(
            incident_id=created["id"],
            narrative="Primary narrative",
            expected_resource_version=expected_version,
            actor_id="user:7",
            idempotency_key=key,
            source_request_id="request-conflicting-reader",
            webhook_event_producer=primary,
        )
    assert exc_info.value.code is WebhookErrorCode.IDEMPOTENCY_CONFLICT
    markers = _markers(store_path)
    assert len(markers) == 1
    assert markers[0].source_request_id == "request-conflicting-reader"
    reconciler = PendingIncidentEventReconciler(
        repository=sqlite_repo.repository,
        key_ring_result=WebhookKeyRingLoadResult(
            ring=ring,
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        settings=_settings(),
        delivery_id_factory=lambda: "20000000-0000-4000-8000-000000000102",
        store_path=store_path,
    )
    with pytest.raises(WebhookError) as reconcile_exc:
        await reconciler.reconcile_once(limit=100)
    assert reconcile_exc.value.code is WebhookErrorCode.IDEMPOTENCY_CONFLICT
    assert _markers(store_path) == markers
