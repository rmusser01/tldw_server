from __future__ import annotations

import asyncio
import importlib
import sys

import pytest
from fastapi import FastAPI

from tldw_Server_API.app.services.lifecycle_worker_specs import (
    ShutdownPhase,
    WorkerLifecycleContext,
    WorkerStrategy,
)

pytestmark = pytest.mark.unit


def _import_startup_optional_workers():
    sys.modules.pop("tldw_Server_API.app.services.startup_optional_workers", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_optional_workers")


def _context() -> WorkerLifecycleContext:
    return WorkerLifecycleContext(
        app=FastAPI(),
        settings={},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )


def _specs_by_name(startup_workers):
    return {
        spec.name: spec
        for spec in startup_workers.provide_optional_worker_specs()
    }


def test_optional_worker_specs_match_legacy_worker_contract() -> None:
    startup_workers = _import_startup_optional_workers()

    specs = _specs_by_name(startup_workers)

    expected = {
        "jobs_metrics_reconcile_task": "jobs",
        "jobs_crypto_rotate_task": "jobs",
        "jobs_webhooks_task": "jobs",
        "meetings_webhook_dlq_task": "meetings",
        "workflows_dlq_task": "workflows",
        "workflows_gc_task": "workflows",
        "workflows_maint_task": "workflows",
        "jobs_integrity_task": "jobs",
        "persona_visual_generation_task": "persona",
        "persona_visual_portability_task": "persona",
    }
    assert set(specs) == set(expected)
    for name, category in expected.items():
        spec = specs[name]
        assert spec.task_name == name
        assert spec.category == category
        assert spec.phase is ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
        assert spec.timeout_sec == 5.0
        assert spec.strategy is WorkerStrategy.STOP_EVENT_TASK
        assert spec.factory is not None


def test_optional_worker_spec_factories_delegate_to_existing_worker_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_optional_workers()
    service_by_name = {
        "jobs_metrics_reconcile_task": "_run_jobs_metrics_reconcile_service",
        "jobs_crypto_rotate_task": "_run_jobs_crypto_rotate_service",
        "jobs_webhooks_task": "_run_jobs_webhooks_worker_service",
        "meetings_webhook_dlq_task": "_run_meetings_webhook_dlq_worker_service",
        "workflows_dlq_task": "_run_workflows_webhook_dlq_worker_service",
        "workflows_gc_task": "_run_workflows_artifact_gc_worker_service",
        "workflows_maint_task": "_run_workflows_db_maintenance_worker_service",
        "jobs_integrity_task": "_run_jobs_integrity_sweeper_service",
        "persona_visual_generation_task": "_run_persona_visual_generation_worker_service",
        "persona_visual_portability_task": "_run_persona_visual_portability_worker_service",
    }
    calls: list[tuple[str, object]] = []
    for worker_name, service_name in service_by_name.items():
        monkeypatch.setattr(
            startup_workers,
            service_name,
            lambda stop_event, worker_name=worker_name: (
                calls.append((worker_name, stop_event)) or f"{worker_name}-coro"
            ),
        )

    specs = _specs_by_name(startup_workers)

    for worker_name in service_by_name:
        assert specs[worker_name].factory(_context(), f"{worker_name}-stop") == f"{worker_name}-coro"

    assert calls == [
        (worker_name, f"{worker_name}-stop")
        for worker_name in service_by_name
    ]


def test_optional_worker_spec_predicates_match_legacy_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_optional_workers()
    enabled_envs: list[str] = []

    def _enabled(key: str) -> bool:
        enabled_envs.append(key)
        return key != "JOBS_WEBHOOKS_ENABLED"

    monkeypatch.setattr(startup_workers, "_env_flag_enabled", _enabled)
    monkeypatch.setenv("JOBS_WEBHOOKS_URL", "https://hooks.test/jobs")

    specs = _specs_by_name(startup_workers)

    assert specs["jobs_metrics_reconcile_task"].enabled(_context()) is True
    assert specs["jobs_webhooks_task"].enabled(_context()) is False
    assert enabled_envs == [
        "JOBS_METRICS_RECONCILE_ENABLE",
        "JOBS_WEBHOOKS_ENABLED",
    ]


@pytest.mark.asyncio
async def test_start_optional_workers_combines_handles_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_optional_workers()
    calls: list[str] = []

    async def _record_jobs_metrics(*, worker_inventory: object | None = None) -> tuple[str, str]:
        assert worker_inventory is None
        calls.append("jobs-metrics")
        return ("jobs-metrics-stop", "jobs-metrics-task")

    async def _record_jobs_crypto(*, worker_inventory: object | None = None) -> tuple[str, str]:
        assert worker_inventory is None
        calls.append("jobs-crypto")
        return ("jobs-crypto-stop", "jobs-crypto-task")

    async def _record_jobs_webhooks(*, worker_inventory: object | None = None) -> tuple[str, str]:
        assert worker_inventory is None
        calls.append("jobs-webhooks")
        return ("jobs-webhooks-stop", "jobs-webhooks-task")

    async def _record_meetings_dlq(*, worker_inventory: object | None = None) -> tuple[str, str]:
        assert worker_inventory is None
        calls.append("meetings-dlq")
        return ("meetings-dlq-stop", "meetings-dlq-task")

    async def _record_workflows_dlq(*, worker_inventory: object | None = None) -> tuple[str, str]:
        assert worker_inventory is None
        calls.append("workflows-dlq")
        return ("workflows-dlq-stop", "workflows-dlq-task")

    async def _record_workflows_gc(*, worker_inventory: object | None = None) -> tuple[str, str]:
        assert worker_inventory is None
        calls.append("workflows-gc")
        return ("workflows-gc-stop", "workflows-gc-task")

    async def _record_workflows_maint(*, worker_inventory: object | None = None) -> tuple[str, str]:
        assert worker_inventory is None
        calls.append("workflows-maint")
        return ("workflows-maint-stop", "workflows-maint-task")

    async def _record_jobs_integrity(*, worker_inventory: object | None = None) -> tuple[str, str]:
        assert worker_inventory is None
        calls.append("jobs-integrity")
        return ("jobs-integrity-stop", "jobs-integrity-task")

    async def _record_persona_generation(*, worker_inventory: object | None = None) -> tuple[str, str]:
        assert worker_inventory is None
        calls.append("persona-generation")
        return ("persona-generation-stop", "persona-generation-task")

    async def _record_persona_portability(*, worker_inventory: object | None = None) -> tuple[str, str]:
        assert worker_inventory is None
        calls.append("persona-portability")
        return ("persona-portability-stop", "persona-portability-task")

    monkeypatch.setattr(startup_workers, "_start_jobs_metrics_reconcile_worker", _record_jobs_metrics)
    monkeypatch.setattr(startup_workers, "_start_jobs_crypto_rotate_worker", _record_jobs_crypto)
    monkeypatch.setattr(startup_workers, "_start_jobs_webhooks_worker", _record_jobs_webhooks)
    monkeypatch.setattr(startup_workers, "_start_meetings_webhook_dlq_worker", _record_meetings_dlq)
    monkeypatch.setattr(startup_workers, "_start_workflows_webhook_dlq_worker", _record_workflows_dlq)
    monkeypatch.setattr(startup_workers, "_start_workflows_artifact_gc_worker", _record_workflows_gc)
    monkeypatch.setattr(startup_workers, "_start_workflows_db_maintenance_worker", _record_workflows_maint)
    monkeypatch.setattr(startup_workers, "_start_jobs_integrity_sweeper", _record_jobs_integrity)
    monkeypatch.setattr(startup_workers, "_start_persona_visual_generation_worker", _record_persona_generation)
    monkeypatch.setattr(startup_workers, "_start_persona_visual_portability_worker", _record_persona_portability)

    handles = await startup_workers.start_optional_workers()

    assert calls == [
        "jobs-metrics",
        "jobs-crypto",
        "jobs-webhooks",
        "meetings-dlq",
        "workflows-dlq",
        "workflows-gc",
        "workflows-maint",
        "jobs-integrity",
        "persona-generation",
        "persona-portability",
    ]
    assert handles.jobs_metrics_reconcile_stop == "jobs-metrics-stop"
    assert handles.jobs_metrics_reconcile_task == "jobs-metrics-task"
    assert handles.jobs_crypto_rotate_stop_event == "jobs-crypto-stop"
    assert handles.jobs_crypto_rotate_task == "jobs-crypto-task"
    assert handles.jobs_webhooks_stop_event == "jobs-webhooks-stop"
    assert handles.jobs_webhooks_task == "jobs-webhooks-task"
    assert handles.meetings_webhook_dlq_stop_event == "meetings-dlq-stop"
    assert handles.meetings_webhook_dlq_task == "meetings-dlq-task"
    assert handles.workflows_dlq_stop_event == "workflows-dlq-stop"
    assert handles.workflows_dlq_task == "workflows-dlq-task"
    assert handles.workflows_gc_stop_event == "workflows-gc-stop"
    assert handles.workflows_gc_task == "workflows-gc-task"
    assert handles.workflows_maint_stop_event == "workflows-maint-stop"
    assert handles.workflows_maint_task == "workflows-maint-task"
    assert handles.jobs_integrity_stop_event == "jobs-integrity-stop"
    assert handles.jobs_integrity_task == "jobs-integrity-task"
    assert handles.persona_visual_generation_stop_event == "persona-generation-stop"
    assert handles.persona_visual_generation_task == "persona-generation-task"
    assert handles.persona_visual_portability_stop_event == "persona-portability-stop"
    assert handles.persona_visual_portability_task == "persona-portability-task"


@pytest.mark.asyncio
async def test_start_jobs_webhooks_worker_skips_without_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_optional_workers()

    monkeypatch.setenv("JOBS_WEBHOOKS_ENABLED", "true")
    monkeypatch.delenv("JOBS_WEBHOOKS_URL", raising=False)
    monkeypatch.setattr(startup_workers, "_make_event", lambda: (_ for _ in ()).throw(AssertionError("no event")))
    monkeypatch.setattr(startup_workers, "_create_task", lambda coro: (_ for _ in ()).throw(AssertionError("no task")))

    stop_event, task = await startup_workers._start_jobs_webhooks_worker()

    assert stop_event is None
    assert task is None


@pytest.mark.asyncio
async def test_start_jobs_webhooks_worker_registers_background_inventory_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_optional_workers()
    from tldw_Server_API.app.services.lifecycle_workers import ShutdownPhase, WorkerRegistry

    app = FastAPI()
    worker_inventory = WorkerRegistry(app)
    observed_stop_events: list[asyncio.Event] = []

    async def _fake_webhooks_worker(stop_event: asyncio.Event) -> None:
        observed_stop_events.append(stop_event)
        await stop_event.wait()

    monkeypatch.setenv("JOBS_WEBHOOKS_ENABLED", "true")
    monkeypatch.setenv("JOBS_WEBHOOKS_URL", "https://example.test/jobs-webhooks")
    monkeypatch.setattr(
        startup_workers,
        "_run_jobs_webhooks_worker_service",
        _fake_webhooks_worker,
    )

    stop_event = None
    task = None
    try:
        stop_event, task = await startup_workers._start_jobs_webhooks_worker(
            worker_inventory=worker_inventory,
        )
        await asyncio.sleep(0)

        assert stop_event is not None
        assert task is not None
        assert task.get_name() == "jobs_webhooks_task"
        assert observed_stop_events == [stop_event]
        assert len(worker_inventory.handles) == 1
        handle = worker_inventory.handles[0]
        assert handle.name == "jobs_webhooks_task"
        assert handle.task is task
        assert handle.stop_event is stop_event
        assert handle.category == "jobs"
        assert handle.shutdown_phase is ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
        assert app.state._tldw_shutdown_worker_inventory == [
            {
                "name": "jobs_webhooks_task",
                "task_name": "jobs_webhooks_task",
                "has_stop_event": True,
                "timeout_sec": 5.0,
                "category": "jobs",
                "shutdown_phase": "background_worker_shutdown",
            }
        ]
        assert app.state._tldw_shutdown_job_poller_inventory == []
    finally:
        if stop_event is not None:
            stop_event.set()
        if task is not None:
            await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_start_optional_workers_passes_inventory_to_auxiliary_optional_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_optional_workers()
    worker_inventory = object()
    calls: list[tuple[str, object | None]] = []

    async def _record_jobs_metrics(*, worker_inventory: object | None = None) -> tuple[str, str]:
        calls.append(("jobs-metrics", worker_inventory))
        return ("jobs-metrics-stop", "jobs-metrics-task")

    async def _record_jobs_crypto(*, worker_inventory: object | None = None) -> tuple[str, str]:
        calls.append(("jobs-crypto", worker_inventory))
        return ("jobs-crypto-stop", "jobs-crypto-task")

    async def _record_jobs_webhooks(*, worker_inventory: object | None = None) -> tuple[str, str]:
        calls.append(("jobs-webhooks", worker_inventory))
        return ("jobs-webhooks-stop", "jobs-webhooks-task")

    async def _record_meetings_dlq(*, worker_inventory: object | None = None) -> tuple[str, str]:
        calls.append(("meetings-dlq", worker_inventory))
        return ("meetings-dlq-stop", "meetings-dlq-task")

    async def _record_workflows_dlq(*, worker_inventory: object | None = None) -> tuple[str, str]:
        calls.append(("workflows-dlq", worker_inventory))
        return ("workflows-dlq-stop", "workflows-dlq-task")

    async def _record_workflows_gc(*, worker_inventory: object | None = None) -> tuple[str, str]:
        calls.append(("workflows-gc", worker_inventory))
        return ("workflows-gc-stop", "workflows-gc-task")

    async def _record_workflows_maint(*, worker_inventory: object | None = None) -> tuple[str, str]:
        calls.append(("workflows-maint", worker_inventory))
        return ("workflows-maint-stop", "workflows-maint-task")

    async def _record_jobs_integrity(*, worker_inventory: object | None = None) -> tuple[str, str]:
        calls.append(("jobs-integrity", worker_inventory))
        return ("jobs-integrity-stop", "jobs-integrity-task")

    monkeypatch.setattr(startup_workers, "_start_jobs_metrics_reconcile_worker", _record_jobs_metrics)
    monkeypatch.setattr(startup_workers, "_start_jobs_crypto_rotate_worker", _record_jobs_crypto)
    monkeypatch.setattr(startup_workers, "_start_jobs_webhooks_worker", _record_jobs_webhooks)
    monkeypatch.setattr(startup_workers, "_start_meetings_webhook_dlq_worker", _record_meetings_dlq)
    monkeypatch.setattr(startup_workers, "_start_workflows_webhook_dlq_worker", _record_workflows_dlq)
    monkeypatch.setattr(startup_workers, "_start_workflows_artifact_gc_worker", _record_workflows_gc)
    monkeypatch.setattr(startup_workers, "_start_workflows_db_maintenance_worker", _record_workflows_maint)
    monkeypatch.setattr(startup_workers, "_start_jobs_integrity_sweeper", _record_jobs_integrity)

    handles = await startup_workers.start_optional_workers(worker_inventory=worker_inventory)

    assert calls == [
        ("jobs-metrics", worker_inventory),
        ("jobs-crypto", worker_inventory),
        ("jobs-webhooks", worker_inventory),
        ("meetings-dlq", worker_inventory),
        ("workflows-dlq", worker_inventory),
        ("workflows-gc", worker_inventory),
        ("workflows-maint", worker_inventory),
        ("jobs-integrity", worker_inventory),
    ]
    assert handles.meetings_webhook_dlq_task == "meetings-dlq-task"
    assert handles.workflows_dlq_task == "workflows-dlq-task"
    assert handles.workflows_gc_task == "workflows-gc-task"
    assert handles.workflows_maint_task == "workflows-maint-task"


@pytest.mark.parametrize(
    (
        "helper_name",
        "env_key",
        "service_name",
        "worker_name",
        "category",
    ),
    [
        (
            "_start_meetings_webhook_dlq_worker",
            "MEETINGS_WEBHOOK_DLQ_ENABLED",
            "_run_meetings_webhook_dlq_worker_service",
            "meetings_webhook_dlq_task",
            "meetings",
        ),
        (
            "_start_workflows_webhook_dlq_worker",
            "WORKFLOWS_WEBHOOK_DLQ_ENABLED",
            "_run_workflows_webhook_dlq_worker_service",
            "workflows_dlq_task",
            "workflows",
        ),
        (
            "_start_workflows_artifact_gc_worker",
            "WORKFLOWS_ARTIFACT_GC_ENABLED",
            "_run_workflows_artifact_gc_worker_service",
            "workflows_gc_task",
            "workflows",
        ),
        (
            "_start_workflows_db_maintenance_worker",
            "WORKFLOWS_DB_MAINTENANCE_ENABLED",
            "_run_workflows_db_maintenance_worker_service",
            "workflows_maint_task",
            "workflows",
        ),
    ],
)
@pytest.mark.asyncio
async def test_start_auxiliary_optional_worker_registers_background_inventory_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    helper_name: str,
    env_key: str,
    service_name: str,
    worker_name: str,
    category: str,
) -> None:
    startup_workers = _import_startup_optional_workers()
    from tldw_Server_API.app.services.lifecycle_workers import ShutdownPhase, WorkerRegistry

    app = FastAPI()
    worker_inventory = WorkerRegistry(app)
    observed_stop_events: list[asyncio.Event] = []

    async def _fake_worker(stop_event: asyncio.Event) -> None:
        observed_stop_events.append(stop_event)
        await stop_event.wait()

    monkeypatch.setenv(env_key, "true")
    monkeypatch.setattr(startup_workers, service_name, _fake_worker)

    stop_event = None
    task = None
    try:
        stop_event, task = await getattr(startup_workers, helper_name)(
            worker_inventory=worker_inventory,
        )
        await asyncio.sleep(0)

        assert stop_event is not None
        assert task is not None
        assert task.get_name() == worker_name
        assert observed_stop_events == [stop_event]
        assert len(worker_inventory.handles) == 1
        handle = worker_inventory.handles[0]
        assert handle.name == worker_name
        assert handle.task is task
        assert handle.stop_event is stop_event
        assert handle.category == category
        assert handle.shutdown_phase is ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
        assert app.state._tldw_shutdown_worker_inventory == [
            {
                "name": worker_name,
                "task_name": worker_name,
                "has_stop_event": True,
                "timeout_sec": 5.0,
                "category": category,
                "shutdown_phase": "background_worker_shutdown",
            }
        ]
        assert app.state._tldw_shutdown_job_poller_inventory == []
    finally:
        if stop_event is not None:
            stop_event.set()
        if task is not None:
            await asyncio.wait_for(task, timeout=1)


@pytest.mark.asyncio
async def test_start_jobs_metrics_reconcile_worker_starts_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_optional_workers()
    captured_stop_events: list[object] = []
    created_coroutines: list[object] = []

    monkeypatch.setenv("JOBS_METRICS_RECONCILE_ENABLE", "true")
    monkeypatch.setattr(startup_workers, "_make_event", lambda: "jobs-metrics-stop")
    monkeypatch.setattr(startup_workers, "_create_task", lambda coro: created_coroutines.append(coro) or "jobs-metrics-task")
    monkeypatch.setattr(
        startup_workers,
        "_run_jobs_metrics_reconcile_service",
        lambda stop_event: captured_stop_events.append(stop_event) or "jobs-metrics-coro",
    )

    stop_event, task = await startup_workers._start_jobs_metrics_reconcile_worker()

    assert stop_event == "jobs-metrics-stop"
    assert task == "jobs-metrics-task"
    assert captured_stop_events == ["jobs-metrics-stop"]
    assert created_coroutines == ["jobs-metrics-coro"]


@pytest.mark.asyncio
async def test_start_jobs_integrity_sweeper_handles_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_optional_workers()

    monkeypatch.setenv("JOBS_INTEGRITY_SWEEP_ENABLED", "true")
    monkeypatch.setattr(startup_workers, "_make_event", lambda: "jobs-integrity-stop")

    def _failing_create_task(coro):
        del coro
        raise RuntimeError("boom")

    monkeypatch.setattr(startup_workers, "_create_task", _failing_create_task)
    monkeypatch.setattr(startup_workers, "_run_jobs_integrity_sweeper_service", lambda stop_event: stop_event)

    stop_event, task = await startup_workers._start_jobs_integrity_sweeper()

    assert stop_event is None
    assert task is None


@pytest.mark.asyncio
async def test_start_jobs_integrity_sweeper_handles_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_optional_workers()

    monkeypatch.setenv("JOBS_INTEGRITY_SWEEP_ENABLED", "true")
    monkeypatch.setattr(startup_workers, "_make_event", lambda: "jobs-integrity-stop")

    def _failing_service(stop_event):
        del stop_event
        raise ImportError("missing jobs integrity service")

    monkeypatch.setattr(startup_workers, "_run_jobs_integrity_sweeper_service", _failing_service)

    stop_event, task = await startup_workers._start_jobs_integrity_sweeper()

    assert stop_event is None
    assert task is None
