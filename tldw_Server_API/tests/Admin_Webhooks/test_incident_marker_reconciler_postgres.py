# The imported fixture must retain its name for pytest discovery.
# ruff: noqa: F401, F811

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tldw_Server_API.app.core.Admin_Webhooks.crypto import (
    WebhookKeyLoadCode,
    WebhookKeyRingLoadResult,
)
from tldw_Server_API.app.core.Admin_Webhooks.domain import EventSourceKind
from tldw_Server_API.app.core.Admin_Webhooks.incident_reconciler import (
    IncidentReconcileCrashPoint,
    PendingIncidentEventReconciler,
)
from tldw_Server_API.app.core.Admin_Webhooks.producer import (
    AdminWebhookEventProducer,
    ProductionEventPreparation,
    build_incident_created_data,
)
from tldw_Server_API.app.services import admin_system_ops_service as system_ops
from tldw_Server_API.tests.Admin_Webhooks.test_repository_postgres import (
    PostgreSQLRepositoryFixture,
    _complete_migration,
    pg_repo,
)
from tldw_Server_API.tests.Admin_Webhooks.test_user_producers_postgres import (
    _ring,
    _seed_registration,
    _settings,
)

pytest_plugins = ("tldw_Server_API.tests._plugins.authnz_full_fixtures",)
pytestmark = pytest.mark.postgres

NOW = datetime(2026, 9, 1, 4, 0, tzinfo=timezone.utc)
EVENT_ID = "36000000-0000-4000-8000-000000000001"


@pytest.mark.integration
async def test_postgres_replay_after_commit_converges_to_one_event_and_delivery(
    pg_repo: PostgreSQLRepositoryFixture,
    tmp_path: Path,
) -> None:
    ring = _ring()
    ring_result = WebhookKeyRingLoadResult(
        ring=ring,
        code=WebhookKeyLoadCode.AVAILABLE,
    )
    await _complete_migration(pg_repo.repository)
    await _seed_registration(pg_repo, ring, event_type="incident.created")
    producer = AdminWebhookEventProducer(
        repository=pg_repo.repository,
        settings=_settings(),
        key_ring_result=ring_result,
        event_id_factory=lambda: EVENT_ID,
        delivery_id_factory=lambda: "37000000-0000-4000-8000-000000000001",
        clock=lambda: NOW,
    )
    marker = producer.prepare_incident_marker(
        ProductionEventPreparation(
            event_id=EVENT_ID,
            created_at=NOW,
            source_component="admin_system_ops",
            source_request_id="postgres-incident-request",
        ),
        event_type="incident.created",
        source_kind=EventSourceKind.AGGREGATE,
        aggregate_type="incident",
        aggregate_id="postgres-incident",
        aggregate_version="1",
        source_command_id=None,
        data=build_incident_created_data(
            incident_id="postgres-incident",
            state="investigating",
            severity="high",
            resource_version=1,
            created_at=NOW,
            updated_at=NOW,
            resolved_at=None,
        ),
    )
    store_path = tmp_path / "system_ops.json"
    store = system_ops._default_store()
    store["webhook_pending_events"] = [marker.to_store_record()]
    system_ops._atomic_write_store(store_path, store)
    injected = False

    def crash_after_commit(point: IncidentReconcileCrashPoint) -> None:
        nonlocal injected
        if not injected and point is IncidentReconcileCrashPoint.AFTER_DB_COMMIT_BEFORE_REMOVE:
            injected = True
            raise RuntimeError("injected post-commit crash")

    delivery_ids = iter(
        (
            "38000000-0000-4000-8000-000000000001",
            "38000000-0000-4000-8000-000000000002",
        )
    )

    def reconciler(*, crash_injector=None) -> PendingIncidentEventReconciler:
        return PendingIncidentEventReconciler(
            repository=pg_repo.repository,
            key_ring_result=ring_result,
            settings=_settings(),
            delivery_id_factory=lambda: next(delivery_ids),
            store_path=store_path,
            crash_injector=crash_injector,
        )

    with pytest.raises(RuntimeError, match="injected post-commit crash"):
        await reconciler(crash_injector=crash_after_commit).reconcile_once()

    event_count = await pg_repo.pool.fetchval(
        "SELECT COUNT(*) FROM admin_webhook_events WHERE id = ?",
        EVENT_ID,
    )
    delivery_count = await pg_repo.pool.fetchval(
        "SELECT COUNT(*) FROM admin_webhook_deliveries WHERE event_id = ?",
        EVENT_ID,
    )
    assert int(event_count) == 1
    assert int(delivery_count) == 1
    assert len(json.loads(store_path.read_text(encoding="utf-8"))["webhook_pending_events"]) == 1

    assert await reconciler().reconcile_once() == 1
    assert json.loads(store_path.read_text(encoding="utf-8"))["webhook_pending_events"] == []
    assert (
        int(
            await pg_repo.pool.fetchval(
                "SELECT COUNT(*) FROM admin_webhook_events WHERE id = ?",
                EVENT_ID,
            )
        )
        == 1
    )
    assert (
        int(
            await pg_repo.pool.fetchval(
                "SELECT COUNT(*) FROM admin_webhook_deliveries WHERE event_id = ?",
                EVENT_ID,
            )
        )
        == 1
    )
