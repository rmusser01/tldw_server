from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_startup_optional_workers():
    sys.modules.pop("tldw_Server_API.app.services.startup_optional_workers", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_optional_workers")


@pytest.mark.asyncio
async def test_start_optional_workers_combines_handles_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_optional_workers()
    calls: list[str] = []

    async def _record_jobs_metrics():
        calls.append("jobs-metrics")
        return ("jobs-metrics-stop", "jobs-metrics-task")

    async def _record_jobs_crypto():
        calls.append("jobs-crypto")
        return ("jobs-crypto-stop", "jobs-crypto-task")

    async def _record_jobs_webhooks():
        calls.append("jobs-webhooks")
        return ("jobs-webhooks-stop", "jobs-webhooks-task")

    async def _record_meetings_dlq():
        calls.append("meetings-dlq")
        return ("meetings-dlq-stop", "meetings-dlq-task")

    async def _record_workflows_dlq():
        calls.append("workflows-dlq")
        return ("workflows-dlq-stop", "workflows-dlq-task")

    async def _record_workflows_gc():
        calls.append("workflows-gc")
        return ("workflows-gc-stop", "workflows-gc-task")

    async def _record_workflows_maint():
        calls.append("workflows-maint")
        return ("workflows-maint-stop", "workflows-maint-task")

    async def _record_jobs_integrity():
        calls.append("jobs-integrity")
        return ("jobs-integrity-stop", "jobs-integrity-task")

    monkeypatch.setattr(startup_workers, "_start_jobs_metrics_reconcile_worker", _record_jobs_metrics)
    monkeypatch.setattr(startup_workers, "_start_jobs_crypto_rotate_worker", _record_jobs_crypto)
    monkeypatch.setattr(startup_workers, "_start_jobs_webhooks_worker", _record_jobs_webhooks)
    monkeypatch.setattr(startup_workers, "_start_meetings_webhook_dlq_worker", _record_meetings_dlq)
    monkeypatch.setattr(startup_workers, "_start_workflows_webhook_dlq_worker", _record_workflows_dlq)
    monkeypatch.setattr(startup_workers, "_start_workflows_artifact_gc_worker", _record_workflows_gc)
    monkeypatch.setattr(startup_workers, "_start_workflows_db_maintenance_worker", _record_workflows_maint)
    monkeypatch.setattr(startup_workers, "_start_jobs_integrity_sweeper", _record_jobs_integrity)

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
