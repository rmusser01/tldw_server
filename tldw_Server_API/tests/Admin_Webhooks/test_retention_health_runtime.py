from __future__ import annotations

import asyncio
import base64
import importlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from tldw_Server_API.app.core.Admin_Webhooks.crypto import (
    WebhookKeyLoadCode,
    WebhookKeyRing,
    WebhookKeyRingLoadResult,
)
from tldw_Server_API.app.core.Admin_Webhooks.domain import (
    DeliveryBacklogCounts,
    DeliveryCapabilityStatus,
    DeliveryComponentStatus,
    DeliveryKind,
    DeliveryReasonCode,
    DeliveryRuntimeComponent,
    DeliveryRuntimeReasonCode,
    DeliveryState,
    WebhookErrorCode,
)
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.settings import Settings
from tldw_Server_API.app.core.DB_Management.admin_webhooks_repository import (
    AdminWebhookRepository,
    RuntimeHeartbeatWrite,
)
from tldw_Server_API.tests.Admin_Webhooks.test_event_expansion import (
    _captured_delivery,
    canonical_uuid4,
    opaque_token,
)

NOW = datetime(2026, 8, 29, 12, 0, tzinfo=timezone.utc)


@pytest_asyncio.fixture
async def repository(tmp_path: Path) -> AdminWebhookRepository:
    pool = DatabasePool(
        Settings(
            AUTH_MODE="single_user",
            DATABASE_URL=f"sqlite:///{tmp_path / 'task-11.db'}",
        )
    )
    await pool.initialize()
    try:
        yield AdminWebhookRepository(pool)
    finally:
        await pool.close()


@pytest.mark.unit
async def test_acquisition_preflight_does_not_require_its_own_worker_heartbeat() -> None:
    observability = importlib.import_module("tldw_Server_API.app.core.Admin_Webhooks.observability")
    domain = importlib.import_module("tldw_Server_API.app.core.Admin_Webhooks.domain")

    component_status = domain.DeliveryComponentStatus
    backlog_counts = domain.DeliveryBacklogCounts
    health_snapshot = domain.DeliveryHealthSnapshot
    jobs_status = observability.JobsCapabilityStatus

    snapshot = health_snapshot(
        canonical_schema_version=1,
        delivery_schema_ready=True,
        migration_complete=True,
        key_ready=True,
        key_primary_match=True,
        worker=component_status(
            component=DeliveryRuntimeComponent.WORKER,
            ready=False,
            reason_code=DeliveryRuntimeReasonCode.WORKER_UNAVAILABLE,
            heartbeat_age_seconds=None,
        ),
        reconciler=component_status(
            component=DeliveryRuntimeComponent.RECONCILER,
            ready=True,
            reason_code=None,
            heartbeat_age_seconds=1,
        ),
        retention=component_status(
            component=DeliveryRuntimeComponent.RETENTION,
            ready=False,
            reason_code=DeliveryRuntimeReasonCode.RETENTION_UNAVAILABLE,
            heartbeat_age_seconds=None,
        ),
        backlog=backlog_counts(pending=2),
        oldest_nonterminal_created_at=NOW - timedelta(seconds=9),
    )

    class Repository:
        async def get_delivery_health_snapshot(self, **_kwargs: object) -> object:
            return snapshot

    class JobsProbe:
        async def status(self) -> object:
            return jobs_status(
                database_ready=True,
                queue_ready=True,
                job_type_ready=True,
                backend="sqlite",
            )

    encoded = base64.b64encode(b"k" * 32).decode("ascii")
    capability = observability.AdminWebhookDeliveryCapability(
        repository=Repository(),
        key_ring_result=WebhookKeyRingLoadResult(
            ring=WebhookKeyRing({"key-1": encoded}, primary_id="key-1"),
            code=WebhookKeyLoadCode.AVAILABLE,
        ),
        jobs_probe=JobsProbe(),
        heartbeat_freshness_seconds=30,
    )

    status = await capability.status(NOW)

    assert status.acquisition_ready is True
    assert status.acquisition_reason_code is None
    assert status.worker.ready is False
    assert status.retention.ready is False
    assert status.delivery_capability_ready is False
    assert status.backlog.pending == 2
    assert status.oldest_nonterminal_age_seconds == 9
    assert "instance" not in repr(status).lower()


@pytest.mark.unit
async def test_retention_partial_batches_follow_ruled_order_without_starvation(
    repository: AdminWebhookRepository,
) -> None:
    webhook_id, delivery_id = await _captured_delivery(
        repository,
        event_id="task-11-retained-event",
        command_id="task-11-retained-command",
        isolated=True,
    )
    cutoff = NOW - timedelta(days=30)
    async with repository.transaction() as tx:
        await tx._execute(
            """
            UPDATE admin_webhook_deliveries
            SET state = 'succeeded', terminal_at = ?, updated_at = ?
            WHERE id = ?
            """,
            (cutoff, cutoff, delivery_id),
        )
        await tx._execute(
            "UPDATE admin_webhook_events SET created_at = ? WHERE id = ?",
            (cutoff, canonical_uuid4("task-11-retained-event")),
        )
        await tx._execute(
            """
            INSERT INTO admin_webhook_idempotency (
                lookup_digest, actor_id, operation, route, request_fingerprint,
                state, created_at, updated_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, 'in_progress', ?, ?, ?)
            """,
            (
                f"sha256:{opaque_token('task-11-expired-idempotency')}",
                "actor-1",
                "test",
                "/admin/webhooks/test",
                f"hmac-sha256:{opaque_token('task-11-expired-fingerprint')}",
                cutoff,
                cutoff,
                NOW,
            ),
        )
        await tx.upsert_runtime_heartbeat(
            RuntimeHeartbeatWrite(
                component=DeliveryRuntimeComponent.WORKER,
                instance_id=canonical_uuid4("task-11-stale-worker"),
                ready=False,
                reason_code=DeliveryRuntimeReasonCode.WORKER_UNAVAILABLE,
                heartbeat_at=cutoff,
                last_success_at=None,
            )
        )
        await tx._execute(
            """
            UPDATE admin_webhook_registrations
            SET deleted_at = ?, deleted_by_user_id = 1, active = 0, updated_at = ?
            WHERE id = ?
            """,
            (cutoff, cutoff, webhook_id),
        )

    batches = [await repository.purge_retained_rows(NOW, cutoff, 1) for _ in range(5)]

    assert [
        (
            batch.deliveries,
            batch.events,
            batch.expired_idempotency,
            batch.heartbeats,
            batch.registrations,
        )
        for batch in batches
    ] == [
        (1, 0, 0, 0, 0),
        (0, 1, 0, 0, 0),
        (0, 0, 1, 0, 0),
        (0, 0, 0, 1, 0),
        (0, 0, 0, 0, 1),
    ]


class _RecordingRegistry:
    def __init__(self) -> None:
        self.definitions: list[object] = []
        self.observations: list[tuple[str, float, dict[str, str]]] = []

    def register_metric(self, definition: object) -> bool:
        self.definitions.append(definition)
        return True

    def increment(
        self,
        name: str,
        value: float = 1,
        labels: dict[str, str] | None = None,
    ) -> None:
        self.observations.append((name, value, dict(labels or {})))


@pytest.mark.unit
def test_metrics_adapter_exposes_only_fixed_names_and_closed_labels() -> None:
    observability = importlib.import_module("tldw_Server_API.app.core.Admin_Webhooks.observability")
    registry = _RecordingRegistry()
    metrics = observability.AdminWebhookMetrics(registry=registry)

    metrics.delivery_committed(
        state=DeliveryState.SUCCEEDED,
        kind=DeliveryKind.AUTOMATIC,
        reason_code=None,
        status_code=204,
    )

    assert registry.observations == [
        (
            "admin_webhooks_deliveries_total",
            1,
            {
                "state": "succeeded",
                "kind": "automatic",
                "reason": "none",
                "status_class": "2xx",
            },
        )
    ]
    assert all(definition.name.startswith("admin_webhooks_") for definition in registry.definitions)
    assert all(
        set(definition.labels)
        <= {
            "state",
            "kind",
            "event_type",
            "reason",
            "status_class",
            "component",
            "backend",
        }
        for definition in registry.definitions
    )
    assert not hasattr(metrics, "increment")
    with pytest.raises(TypeError):
        metrics.delivery_committed(
            state="receiver.example",  # type: ignore[arg-type]
            kind=DeliveryKind.AUTOMATIC,
            reason_code=DeliveryReasonCode.TARGET_REJECTED,
            status_code=403,
        )


@pytest.mark.unit
def test_metrics_adapter_accepts_only_closed_admission_denial_codes() -> None:
    observability = importlib.import_module("tldw_Server_API.app.core.Admin_Webhooks.observability")
    registry = _RecordingRegistry()
    metrics = observability.AdminWebhookMetrics(registry=registry)

    metrics.admission_denied(WebhookErrorCode.ACTIVE_LIMIT)

    assert registry.observations == [
        (
            "admin_webhooks_admission_denials_total",
            1,
            {"reason": WebhookErrorCode.ACTIVE_LIMIT.value},
        )
    ]
    with pytest.raises(TypeError):
        metrics.admission_denied("receiver.example")  # type: ignore[arg-type]


@pytest.mark.unit
def test_metrics_registry_failures_never_change_delivery_control_flow() -> None:
    observability = importlib.import_module("tldw_Server_API.app.core.Admin_Webhooks.observability")

    class ExplodingRegistry:
        def register_metric(self, _definition: object) -> bool:
            raise RuntimeError("registry unavailable")

        def increment(
            self,
            _name: str,
            _value: float = 1,
            *,
            labels: dict[str, str] | None = None,
        ) -> None:
            del labels
            raise RuntimeError("registry unavailable")

    metrics = observability.AdminWebhookMetrics(registry=ExplodingRegistry())

    metrics.delivery_committed(
        state=DeliveryState.SUCCEEDED,
        kind=DeliveryKind.AUTOMATIC,
        reason_code=None,
        status_code=204,
    )


@pytest.mark.unit
def test_health_metrics_emit_closed_key_and_migration_errors() -> None:
    observability = importlib.import_module("tldw_Server_API.app.core.Admin_Webhooks.observability")
    registry = _RecordingRegistry()
    metrics = observability.AdminWebhookMetrics(registry=registry)
    unavailable = {
        component: DeliveryComponentStatus(
            component=component,
            ready=False,
            reason_code={
                DeliveryRuntimeComponent.WORKER: DeliveryRuntimeReasonCode.KEY_UNAVAILABLE,
                DeliveryRuntimeComponent.RECONCILER: DeliveryRuntimeReasonCode.MIGRATION_PENDING,
                DeliveryRuntimeComponent.RETENTION: DeliveryRuntimeReasonCode.RETENTION_UNAVAILABLE,
            }[component],
            heartbeat_age_seconds=None,
        )
        for component in DeliveryRuntimeComponent
    }
    status = DeliveryCapabilityStatus(
        canonical_schema_version=1,
        schema_ready=True,
        delivery_schema_ready=True,
        migration_complete=False,
        key_ready=False,
        key_primary_match=False,
        jobs_database_ready=True,
        queue_ready=True,
        job_type_ready=True,
        jobs_backend="sqlite",
        worker=unavailable[DeliveryRuntimeComponent.WORKER],
        reconciler=unavailable[DeliveryRuntimeComponent.RECONCILER],
        retention=unavailable[DeliveryRuntimeComponent.RETENTION],
        backlog=DeliveryBacklogCounts(),
        oldest_nonterminal_age_seconds=None,
        acquisition_ready=False,
        acquisition_reason_code=DeliveryRuntimeReasonCode.MIGRATION_PENDING,
        delivery_capability_ready=False,
    )

    metrics.health_snapshot(status)

    assert [item for item in registry.observations if "errors_total" in item[0]] == [
        (
            "admin_webhooks_key_errors_total",
            1,
            {"reason": DeliveryRuntimeReasonCode.KEY_UNAVAILABLE.value},
        ),
        (
            "admin_webhooks_migration_errors_total",
            1,
            {"reason": DeliveryRuntimeReasonCode.MIGRATION_PENDING.value},
        ),
    ]


@pytest.mark.unit
async def test_runtime_supervises_independent_loops_and_awaits_shutdown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = importlib.import_module("tldw_Server_API.app.services.admin_webhook_delivery_runtime")
    stop_event = asyncio.Event()
    all_started = asyncio.Event()
    reconciler_restarted = asyncio.Event()
    started: set[str] = set()
    finalized: set[str] = set()
    reconciler_starts = 0

    async def blocking_loop(name: str, stop: asyncio.Event) -> None:
        started.add(name)
        if len(started) == 3:
            all_started.set()
        try:
            await stop.wait()
        finally:
            finalized.add(name)

    async def worker(stop: asyncio.Event, _components: object) -> None:
        await blocking_loop("worker", stop)

    async def reconciler(stop: asyncio.Event, _components: object) -> None:
        nonlocal reconciler_starts
        reconciler_starts += 1
        if reconciler_starts == 1:
            started.add("reconciler")
            raise RuntimeError("bounded reconciler failure")
        reconciler_restarted.set()
        await blocking_loop("reconciler", stop)

    async def retention(stop: asyncio.Event, _components: object) -> None:
        await blocking_loop("retention", stop)

    async def build_components() -> object:
        return object()

    monkeypatch.setattr(runtime, "_build_runtime_components", build_components)
    monkeypatch.setattr(runtime, "_run_worker_loop", worker)
    monkeypatch.setattr(runtime, "_run_reconciler_loop", reconciler)
    monkeypatch.setattr(runtime, "_run_retention_loop", retention)
    monkeypatch.setattr(runtime, "_RESTART_DELAY_SECONDS", 0)

    task = asyncio.create_task(runtime.run_admin_webhook_delivery_runtime(stop_event))
    await asyncio.wait_for(all_started.wait(), timeout=1)
    await asyncio.wait_for(reconciler_restarted.wait(), timeout=1)
    assert task.done() is False

    stop_event.set()
    await asyncio.wait_for(task, timeout=1)

    assert reconciler_starts == 2
    assert finalized == {"worker", "reconciler", "retention"}


@pytest.mark.unit
async def test_worker_loop_propagates_sdk_exit_for_supervisor_restart() -> None:
    runtime = importlib.import_module("tldw_Server_API.app.services.admin_webhook_delivery_runtime")
    writes: list[RuntimeHeartbeatWrite] = []

    class WorkerSDK:
        async def run_prepared(self, **_kwargs: object) -> None:
            raise RuntimeError("prepared worker stopped")

        def stop(self) -> None:
            return None

    class Repository:
        async def upsert_runtime_heartbeat(
            self,
            write: RuntimeHeartbeatWrite,
        ) -> None:
            writes.append(write)

    components = SimpleNamespace(
        worker_sdk=WorkerSDK(),
        worker_handler=SimpleNamespace(
            handler_error_disposition=lambda *_args: None,
            on_disposition_applied=lambda *_args: None,
        ),
        worker_repository=Repository(),
        worker_instance_id=canonical_uuid4("task-11-worker-sdk-exit"),
        clock=lambda: NOW,
    )

    stop_event = asyncio.Event()
    task = asyncio.create_task(runtime._run_worker_loop(stop_event, components))
    done, _pending = await asyncio.wait({task}, timeout=0.25)
    if task not in done:
        stop_event.set()
        await asyncio.gather(task, return_exceptions=True)

    assert task in done
    with pytest.raises(RuntimeError, match="prepared worker stopped"):
        await task

    assert writes[-1].ready is False
    assert writes[-1].reason_code is DeliveryRuntimeReasonCode.WORKER_UNAVAILABLE


@pytest.mark.unit
async def test_runtime_builds_observable_recovery_when_jobs_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = importlib.import_module("tldw_Server_API.app.services.admin_webhook_delivery_runtime")
    settings = SimpleNamespace(
        mode=runtime.AdminWebhookMode.ON,
        route_selection=runtime.WebhookRouteSelection.CANONICAL,
        delivery_claim_ttl_seconds=60,
        delivery_loop_interval_seconds=1,
        delivery_heartbeat_freshness_seconds=30,
        allow_http_dev=False,
    )

    async def get_pool() -> object:
        return object()

    class Metrics:
        def enqueue_failure(self, *_args: object, **_kwargs: object) -> None:
            return None

        def enqueue_success(self, *_args: object, **_kwargs: object) -> None:
            return None

    monkeypatch.setattr(
        runtime.AdminWebhookSettings,
        "from_environment",
        staticmethod(lambda _environ: settings),
    )
    monkeypatch.setattr(runtime, "get_db_pool", get_pool)
    monkeypatch.setattr(
        runtime,
        "JobManager",
        lambda: (_ for _ in ()).throw(RuntimeError("Jobs unavailable")),
    )
    monkeypatch.setattr(runtime, "AdminWebhookMetrics", Metrics)
    monkeypatch.setattr(
        runtime,
        "load_webhook_key_ring",
        lambda: WebhookKeyRingLoadResult(
            ring=None,
            code=WebhookKeyLoadCode.KEY_UNAVAILABLE,
        ),
    )

    components = await runtime._build_runtime_components()
    jobs_status = await components.capability._jobs_probe.status()

    assert components.worker_sdk is None
    assert jobs_status.database_ready is False
    assert jobs_status.queue_ready is False
    assert jobs_status.job_type_ready is False
    assert jobs_status.backend == "unavailable"


@pytest.mark.unit
async def test_jobs_probe_checks_database_and_fixed_canonical_registration() -> None:
    observability = importlib.import_module("tldw_Server_API.app.core.Admin_Webhooks.observability")

    class Manager:
        backend = "sqlite"
        DOMAIN_ALLOWED_QUEUES = {"admin_webhooks": ("delivery",)}

        def get_job(self, _job_id: int) -> None:
            return None

        def admit_job(self, **_kwargs: object) -> None:
            return None

    status = await observability.JobManagerJobsCapabilityProbe(Manager()).status()

    assert status == observability.JobsCapabilityStatus(
        database_ready=True,
        queue_ready=True,
        job_type_ready=True,
        backend="sqlite",
    )


@pytest.mark.unit
async def test_jobs_probe_rejects_an_allowlist_without_canonical_job_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observability = importlib.import_module("tldw_Server_API.app.core.Admin_Webhooks.observability")

    class Manager:
        backend = "sqlite"
        DOMAIN_ALLOWED_QUEUES = {"admin_webhooks": ("delivery",)}

        def get_job(self, _job_id: int) -> None:
            return None

        def admit_job(self, **_kwargs: object) -> None:
            return None

    monkeypatch.setenv("JOBS_ALLOWED_JOB_TYPES_ADMIN_WEBHOOKS", "other")

    status = await observability.JobManagerJobsCapabilityProbe(Manager()).status()

    assert status.job_type_ready is False


@pytest.mark.unit
async def test_worker_preflight_writes_closed_unready_heartbeat() -> None:
    runtime = importlib.import_module("tldw_Server_API.app.services.admin_webhook_delivery_runtime")
    writes: list[RuntimeHeartbeatWrite] = []

    class Capability:
        async def status(self, _now: datetime) -> object:
            return SimpleNamespace(
                acquisition_ready=False,
                acquisition_reason_code=DeliveryRuntimeReasonCode.KEY_UNAVAILABLE,
            )

    class Repository:
        async def upsert_runtime_heartbeat(
            self,
            write: RuntimeHeartbeatWrite,
        ) -> None:
            writes.append(write)

    components = SimpleNamespace(
        capability=Capability(),
        worker_repository=Repository(),
        worker_instance_id=canonical_uuid4("task-11-worker-preflight"),
        clock=lambda: NOW,
    )

    assert await runtime._worker_pre_acquire(components) is False
    assert len(writes) == 1
    assert writes[0].ready is False
    assert writes[0].reason_code is DeliveryRuntimeReasonCode.KEY_UNAVAILABLE


@pytest.mark.unit
async def test_retention_failure_publishes_unready_heartbeat() -> None:
    runtime = importlib.import_module("tldw_Server_API.app.services.admin_webhook_delivery_runtime")
    stop_event = asyncio.Event()
    writes: list[RuntimeHeartbeatWrite] = []

    class Repository:
        async def purge_retained_rows(self, *_args: object) -> None:
            raise RuntimeError("retention write failed")

        async def upsert_runtime_heartbeat(
            self,
            write: RuntimeHeartbeatWrite,
        ) -> None:
            writes.append(write)
            stop_event.set()

    components = SimpleNamespace(
        retention_repository=Repository(),
        retention_instance_id=canonical_uuid4("task-11-retention-failure"),
        settings=SimpleNamespace(
            delivery_retention_days=30,
            delivery_loop_interval_seconds=1,
        ),
        metrics=SimpleNamespace(retention_committed=lambda _result: None),
        clock=lambda: NOW,
    )

    await runtime._run_retention_loop(stop_event, components)

    assert writes
    assert writes[0].ready is False
    assert writes[0].reason_code is DeliveryRuntimeReasonCode.RETENTION_UNAVAILABLE
