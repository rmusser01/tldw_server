# The imported fixture must retain its name for pytest discovery.
# ruff: noqa: F401, F811

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

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
)
from tldw_Server_API.app.core.Admin_Webhooks.incident_reconciler import (
    IncidentReconcileCrashPoint,
    PendingIncidentEventReconciler,
)
from tldw_Server_API.app.core.Admin_Webhooks.producer import (
    AdminWebhookEventProducer,
    ProductionEventPreparation,
    build_incident_created_data,
)
from tldw_Server_API.app.core.AuthNZ.exceptions import TransactionError
from tldw_Server_API.app.services import admin_system_ops_service as system_ops
from tldw_Server_API.tests.Admin_Webhooks.test_event_expansion import seed_registration
from tldw_Server_API.tests.Admin_Webhooks.test_incident_producers import (
    KEY_ID,
    _complete_migration,
    _ring,
    _settings,
)
from tldw_Server_API.tests.Admin_Webhooks.test_repository_sqlite import (
    SQLiteRepositoryFixture,
    sqlite_repo,
)

NOW = datetime(2026, 9, 1, 3, 0, tzinfo=timezone.utc)
EVENT_IDS = (
    "31000000-0000-4000-8000-000000000001",
    "31000000-0000-4000-8000-000000000002",
    "31000000-0000-4000-8000-000000000003",
)


def _ring_result(ring: WebhookKeyRing) -> WebhookKeyRingLoadResult:
    return WebhookKeyRingLoadResult(
        ring=ring,
        code=WebhookKeyLoadCode.AVAILABLE,
    )


def _marker(
    repository,
    ring: WebhookKeyRing,
    *,
    event_id: str,
    incident_id: str,
    created_at: datetime,
    severity: str = "high",
) -> PendingIncidentWebhookMarker:
    producer = AdminWebhookEventProducer(
        repository=repository,
        settings=_settings(),
        key_ring_result=_ring_result(ring),
        event_id_factory=lambda: event_id,
        delivery_id_factory=lambda: "32000000-0000-4000-8000-000000000001",
        clock=lambda: created_at,
    )
    return producer.prepare_incident_marker(
        ProductionEventPreparation(
            event_id=event_id,
            created_at=created_at,
            source_component="admin_system_ops",
            source_request_id="request-reconcile",
        ),
        event_type="incident.created",
        source_kind=EventSourceKind.AGGREGATE,
        aggregate_type="incident",
        aggregate_id=incident_id,
        aggregate_version="1",
        source_command_id=None,
        data=build_incident_created_data(
            incident_id=incident_id,
            state="investigating",
            severity=severity,
            resource_version=1,
            created_at=created_at,
            updated_at=created_at,
            resolved_at=None,
        ),
    )


def _write_markers(
    path: Path,
    markers: list[PendingIncidentWebhookMarker] | list[dict[str, object]],
) -> None:
    store = system_ops._default_store()
    store["webhook_pending_events"] = [
        marker.to_store_record() if isinstance(marker, PendingIncidentWebhookMarker) else marker for marker in markers
    ]
    system_ops._atomic_write_store(path, store)


def _stored_marker_records(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text(encoding="utf-8"))["webhook_pending_events"]


def _database_rows(path: Path) -> tuple[list[str], int]:
    with sqlite3.connect(path) as connection:
        event_ids = [
            row[0]
            for row in connection.execute("SELECT id FROM admin_webhook_events ORDER BY created_at, id").fetchall()
        ]
        delivery_count = connection.execute("SELECT COUNT(*) FROM admin_webhook_deliveries").fetchone()[0]
    return event_ids, int(delivery_count)


def _reconciler(
    fixture: SQLiteRepositoryFixture,
    ring: WebhookKeyRing,
    store_path: Path,
    *,
    crash_injector=None,
) -> PendingIncidentEventReconciler:
    delivery_ids = iter(f"33000000-0000-4000-8000-{value:012d}" for value in range(1, 20))
    return PendingIncidentEventReconciler(
        repository=fixture.repository,
        key_ring_result=_ring_result(ring),
        settings=_settings(),
        delivery_id_factory=lambda: next(delivery_ids),
        store_path=store_path,
        crash_injector=crash_injector,
    )


@pytest.mark.unit
async def test_reconciles_a_bounded_deterministic_page_with_automatic_fanout(
    sqlite_repo: SQLiteRepositoryFixture,
    tmp_path: Path,
) -> None:
    ring = _ring()
    store_path = tmp_path / "system_ops.json"
    await _complete_migration(sqlite_repo.repository)
    await seed_registration(
        sqlite_repo.repository,
        event_types=("incident.created",),
        active=True,
        now=NOW - timedelta(hours=1),
    )
    markers = [
        _marker(
            sqlite_repo.repository,
            ring,
            event_id=EVENT_IDS[2],
            incident_id="incident-latest",
            created_at=NOW + timedelta(minutes=2),
        ),
        _marker(
            sqlite_repo.repository,
            ring,
            event_id=EVENT_IDS[0],
            incident_id="incident-first",
            created_at=NOW,
        ),
        _marker(
            sqlite_repo.repository,
            ring,
            event_id=EVENT_IDS[1],
            incident_id="incident-middle",
            created_at=NOW + timedelta(minutes=1),
        ),
    ]
    _write_markers(store_path, markers)
    reconciler = _reconciler(sqlite_repo, ring, store_path)

    assert await reconciler.reconcile_once(limit=2) == 2
    assert [record["event_id"] for record in _stored_marker_records(store_path)] == [EVENT_IDS[2]]
    assert _database_rows(sqlite_repo.path) == ([EVENT_IDS[0], EVENT_IDS[1]], 2)

    assert await reconciler.reconcile_once(limit=100) == 1
    assert _stored_marker_records(store_path) == []
    assert _database_rows(sqlite_repo.path) == (list(EVENT_IDS), 3)
    with pytest.raises(ValueError, match="limit must be between 1 and 100"):
        await reconciler.reconcile_once(limit=0)
    with pytest.raises(ValueError, match="limit must be between 1 and 100"):
        await reconciler.reconcile_once(limit=101)


@pytest.mark.unit
@pytest.mark.parametrize("crash_point", list(IncidentReconcileCrashPoint))
async def test_each_crash_boundary_converges_without_duplicate_event_or_delivery(
    sqlite_repo: SQLiteRepositoryFixture,
    tmp_path: Path,
    crash_point: IncidentReconcileCrashPoint,
) -> None:
    ring = _ring()
    store_path = tmp_path / "system_ops.json"
    await _complete_migration(sqlite_repo.repository)
    await seed_registration(
        sqlite_repo.repository,
        event_types=("incident.created",),
        active=True,
    )
    marker = _marker(
        sqlite_repo.repository,
        ring,
        event_id=EVENT_IDS[0],
        incident_id="incident-crash",
        created_at=NOW,
    )
    _write_markers(store_path, [marker])
    injected = False

    def crash_once(point: IncidentReconcileCrashPoint) -> None:
        nonlocal injected
        if not injected and point is crash_point:
            injected = True
            raise RuntimeError("injected crash")

    expected_error = TransactionError if crash_point is IncidentReconcileCrashPoint.AFTER_EVENT_INSERT else RuntimeError
    with pytest.raises(expected_error):
        await _reconciler(
            sqlite_repo,
            ring,
            store_path,
            crash_injector=crash_once,
        ).reconcile_once()

    await _reconciler(sqlite_repo, ring, store_path).reconcile_once()

    assert _stored_marker_records(store_path) == []
    assert _database_rows(sqlite_repo.path) == ([EVENT_IDS[0]], 1)


@pytest.mark.unit
async def test_atomic_marker_save_failure_retries_after_database_commit_without_duplicate(
    sqlite_repo: SQLiteRepositoryFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ring = _ring()
    store_path = tmp_path / "system_ops.json"
    await _complete_migration(sqlite_repo.repository)
    await seed_registration(
        sqlite_repo.repository,
        event_types=("incident.created",),
        active=True,
    )
    marker = _marker(
        sqlite_repo.repository,
        ring,
        event_id=EVENT_IDS[0],
        incident_id="incident-save-failure",
        created_at=NOW,
    )
    _write_markers(store_path, [marker])
    before = store_path.read_bytes()
    original_write = system_ops._atomic_write_store
    calls = 0

    def fail_once(path: Path, store: dict[str, object]) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected atomic save failure")
        original_write(path, store)

    monkeypatch.setattr(system_ops, "_atomic_write_store", fail_once)
    reconciler = _reconciler(sqlite_repo, ring, store_path)

    with pytest.raises(OSError, match="injected atomic save failure"):
        await reconciler.reconcile_once()
    assert store_path.read_bytes() == before
    assert _database_rows(sqlite_repo.path) == ([EVENT_IDS[0]], 1)

    assert await reconciler.reconcile_once() == 1
    assert _stored_marker_records(store_path) == []
    assert _database_rows(sqlite_repo.path) == ([EVENT_IDS[0]], 1)


@pytest.mark.unit
async def test_in_flight_marker_replacement_is_preserved(
    sqlite_repo: SQLiteRepositoryFixture,
    tmp_path: Path,
) -> None:
    ring = _ring()
    store_path = tmp_path / "system_ops.json"
    await _complete_migration(sqlite_repo.repository)
    marker = _marker(
        sqlite_repo.repository,
        ring,
        event_id=EVENT_IDS[0],
        incident_id="incident-replaced",
        created_at=NOW,
    )
    _write_markers(store_path, [marker])
    replacement = marker.to_store_record()
    replacement["source_request_id"] = "request-replaced"

    def replace_after_commit(point: IncidentReconcileCrashPoint) -> None:
        if point is IncidentReconcileCrashPoint.AFTER_DB_COMMIT_BEFORE_REMOVE:
            _write_markers(store_path, [replacement])

    reconciler = _reconciler(
        sqlite_repo,
        ring,
        store_path,
        crash_injector=replace_after_commit,
    )

    with pytest.raises(WebhookError) as exc_info:
        await reconciler.reconcile_once()

    assert exc_info.value.code is WebhookErrorCode.PRECONDITION_FAILED
    assert _stored_marker_records(store_path) == [replacement]
    assert _database_rows(sqlite_repo.path) == ([EVENT_IDS[0]], 0)


@pytest.mark.unit
@pytest.mark.parametrize(
    "corruption",
    ["shape", "oversized", "ciphertext", "canonical_body", "source"],
)
async def test_corrupt_or_undecryptable_marker_fails_closed_without_file_mutation(
    sqlite_repo: SQLiteRepositoryFixture,
    tmp_path: Path,
    corruption: str,
) -> None:
    ring = _ring()
    store_path = tmp_path / "system_ops.json"
    await _complete_migration(sqlite_repo.repository)
    marker = _marker(
        sqlite_repo.repository,
        ring,
        event_id=EVENT_IDS[0],
        incident_id="incident-corrupt",
        created_at=NOW,
    )
    record = marker.to_store_record()
    if corruption == "shape":
        record["unexpected"] = "value"
    elif corruption == "oversized":
        record["body_size_bytes"] = 65_537
    elif corruption == "ciphertext":
        ciphertext = json.loads(str(record["body_ciphertext_json"]))
        ciphertext["ct"] = "not-base64"
        record["body_ciphertext_json"] = json.dumps(ciphertext)
    else:
        plaintext = ring.decrypt_bytes(
            purpose=marker.envelope_purpose,
            identity=marker.envelope_identity,
            protected=marker.body,
        )
        identity = marker.envelope_identity
        if corruption == "canonical_body":
            plaintext += b" "
            record["body_size_bytes"] = len(plaintext)
        else:
            identity = {
                "event_id": marker.event_id,
                "api_version": marker.api_version,
                "source_command_id": "invalid-created-source",
            }
            record.update(
                {
                    "source_kind": "command",
                    "aggregate_type": None,
                    "aggregate_id": None,
                    "aggregate_version": None,
                    "source_command_id": "invalid-created-source",
                }
            )
        protected = ring.encrypt_bytes(
            purpose=marker.envelope_purpose,
            identity=identity,
            plaintext=plaintext,
        )
        record["body_ciphertext_json"] = protected.ciphertext_json
        record["body_key_id"] = protected.key_id
    _write_markers(store_path, [record])
    before = store_path.read_bytes()

    with pytest.raises((ValueError, WebhookError)):
        await _reconciler(sqlite_repo, ring, store_path).reconcile_once()

    assert store_path.read_bytes() == before
    assert _database_rows(sqlite_repo.path) == ([], 0)


@pytest.mark.unit
async def test_existing_source_with_different_body_is_a_conflict_and_marker_remains(
    sqlite_repo: SQLiteRepositoryFixture,
    tmp_path: Path,
) -> None:
    ring = _ring()
    store_path = tmp_path / "system_ops.json"
    await _complete_migration(sqlite_repo.repository)
    marker = _marker(
        sqlite_repo.repository,
        ring,
        event_id=EVENT_IDS[0],
        incident_id="incident-source-conflict",
        created_at=NOW,
        severity="high",
    )
    _write_markers(store_path, [marker])
    producer = AdminWebhookEventProducer(
        repository=sqlite_repo.repository,
        settings=_settings(),
        key_ring_result=_ring_result(ring),
        event_id_factory=lambda: EVENT_IDS[1],
        delivery_id_factory=lambda: "34000000-0000-4000-8000-000000000001",
        clock=lambda: NOW,
    )
    preparation = await producer.begin_capture(
        source_component="admin_system_ops",
        source_request_id="request-reconcile",
    )
    assert preparation is not None
    async with sqlite_repo.repository.transaction() as tx:
        await producer.capture_in_transaction(
            preparation,
            tx=tx,
            event_type="incident.created",
            source_kind=EventSourceKind.AGGREGATE,
            aggregate_type="incident",
            aggregate_id="incident-source-conflict",
            aggregate_version="1",
            source_command_id=None,
            data=build_incident_created_data(
                incident_id="incident-source-conflict",
                state="investigating",
                severity="critical",
                resource_version=1,
                created_at=NOW,
                updated_at=NOW,
                resolved_at=None,
            ),
        )
    before = store_path.read_bytes()

    with pytest.raises(WebhookError) as exc_info:
        await _reconciler(sqlite_repo, ring, store_path).reconcile_once()

    assert exc_info.value.code is WebhookErrorCode.IDEMPOTENCY_CONFLICT
    assert store_path.read_bytes() == before
    assert _database_rows(sqlite_repo.path) == ([EVENT_IDS[1]], 0)


@pytest.mark.unit
async def test_unavailable_key_fails_closed_without_file_mutation(
    sqlite_repo: SQLiteRepositoryFixture,
    tmp_path: Path,
) -> None:
    ring = _ring()
    store_path = tmp_path / "system_ops.json"
    await _complete_migration(sqlite_repo.repository)
    marker = _marker(
        sqlite_repo.repository,
        ring,
        event_id=EVENT_IDS[0],
        incident_id="incident-key-loss",
        created_at=NOW,
    )
    _write_markers(store_path, [marker])
    before = store_path.read_bytes()
    unavailable = WebhookKeyRingLoadResult(
        ring=None,
        code=WebhookKeyLoadCode.KEY_UNAVAILABLE,
    )
    reconciler = PendingIncidentEventReconciler(
        repository=sqlite_repo.repository,
        key_ring_result=unavailable,
        settings=_settings(),
        delivery_id_factory=lambda: "35000000-0000-4000-8000-000000000001",
        store_path=store_path,
    )

    with pytest.raises(WebhookError) as exc_info:
        await reconciler.reconcile_once()

    assert exc_info.value.code is WebhookErrorCode.KEY_UNAVAILABLE
    assert store_path.read_bytes() == before
    assert _database_rows(sqlite_repo.path) == ([], 0)
