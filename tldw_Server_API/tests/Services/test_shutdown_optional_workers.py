from __future__ import annotations

import asyncio
import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_shutdown_optional_workers():
    sys.modules.pop("tldw_Server_API.app.services.shutdown_optional_workers", None)
    return importlib.import_module("tldw_Server_API.app.services.shutdown_optional_workers")


class _FakeStopEvent:
    def __init__(self) -> None:
        self.is_set = False

    def set(self) -> None:
        self.is_set = True


class _FakeTask:
    def __init__(self) -> None:
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


@pytest.mark.asyncio
async def test_shutdown_optional_workers_runs_helpers_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_optional = _import_shutdown_optional_workers()
    calls: list[str] = []

    async def _record_jobs_crypto_rotate(**kwargs):
        del kwargs
        calls.append("jobs-crypto-rotate")

    async def _record_jobs_integrity(**kwargs):
        del kwargs
        calls.append("jobs-integrity")

    async def _record_jobs_webhooks(**kwargs):
        del kwargs
        calls.append("jobs-webhooks")

    async def _record_meetings_webhook_dlq(**kwargs):
        del kwargs
        calls.append("meetings-webhook-dlq")

    async def _record_workflows_dlq(**kwargs):
        del kwargs
        calls.append("workflows-dlq")

    async def _record_workflows_gc(**kwargs):
        del kwargs
        calls.append("workflows-gc")

    async def _record_workflows_maint(**kwargs):
        del kwargs
        calls.append("workflows-maint")

    monkeypatch.setattr(shutdown_optional, "_shutdown_jobs_crypto_rotate_worker", _record_jobs_crypto_rotate)
    monkeypatch.setattr(shutdown_optional, "_shutdown_jobs_integrity_worker", _record_jobs_integrity)
    monkeypatch.setattr(shutdown_optional, "_shutdown_jobs_webhooks_worker", _record_jobs_webhooks)
    monkeypatch.setattr(shutdown_optional, "_shutdown_meetings_webhook_dlq_worker", _record_meetings_webhook_dlq)
    monkeypatch.setattr(shutdown_optional, "_shutdown_workflows_webhook_dlq_worker", _record_workflows_dlq)
    monkeypatch.setattr(shutdown_optional, "_shutdown_workflows_artifact_gc_worker", _record_workflows_gc)
    monkeypatch.setattr(shutdown_optional, "_shutdown_workflows_db_maintenance_worker", _record_workflows_maint)

    handles = await shutdown_optional.shutdown_optional_workers(
        jobs_crypto_rotate_task="jobs-crypto-task",
        jobs_crypto_rotate_stop_event="jobs-crypto-stop",
        jobs_integrity_task="jobs-integrity-task",
        jobs_integrity_stop_event="jobs-integrity-stop",
        jobs_webhooks_task="jobs-webhooks-task",
        jobs_webhooks_stop_event="jobs-webhooks-stop",
        meetings_webhook_dlq_task="meetings-webhook-dlq-task",
        meetings_webhook_dlq_stop_event="meetings-webhook-dlq-stop",
        workflows_dlq_task="workflows-dlq-task",
        workflows_dlq_stop_event="workflows-dlq-stop",
        workflows_gc_task="workflows-gc-task",
        workflows_gc_stop_event="workflows-gc-stop",
        workflows_maint_task="workflows-maint-task",
        workflows_maint_stop_event="workflows-maint-stop",
        guard_exceptions=(RuntimeError,),
    )

    assert calls == [
        "jobs-crypto-rotate",
        "jobs-integrity",
        "jobs-webhooks",
        "meetings-webhook-dlq",
        "workflows-dlq",
        "workflows-gc",
        "workflows-maint",
    ]
    assert handles.jobs_crypto_rotate_task == "jobs-crypto-task"
    assert handles.jobs_integrity_task == "jobs-integrity-task"
    assert handles.jobs_webhooks_task == "jobs-webhooks-task"
    assert handles.meetings_webhook_dlq_task == "meetings-webhook-dlq-task"
    assert handles.workflows_dlq_task == "workflows-dlq-task"
    assert handles.workflows_gc_task == "workflows-gc-task"
    assert handles.workflows_maint_task == "workflows-maint-task"


@pytest.mark.asyncio
async def test_shutdown_jobs_webhooks_worker_stops_via_stop_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_optional = _import_shutdown_optional_workers()
    waits: list[tuple[object, float]] = []
    stop_event = _FakeStopEvent()

    async def _fake_wait(task, *, timeout):
        waits.append((task, timeout))

    monkeypatch.setattr(shutdown_optional, "_wait_for_task", _fake_wait)

    await shutdown_optional._shutdown_jobs_webhooks_worker(
        task="jobs-webhooks-task",
        stop_event=stop_event,
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is True
    assert waits == [("jobs-webhooks-task", 5.0)]


@pytest.mark.asyncio
async def test_shutdown_jobs_integrity_worker_cancels_on_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_optional = _import_shutdown_optional_workers()

    class _FakeTask:
        def __init__(self) -> None:
            self.cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    async def _failing_wait(_task, *, timeout):
        del timeout
        raise RuntimeError("boom")

    task = _FakeTask()
    stop_event = _FakeStopEvent()
    monkeypatch.setattr(shutdown_optional, "_wait_for_task", _failing_wait)

    await shutdown_optional._shutdown_jobs_integrity_worker(
        task=task,
        stop_event=stop_event,
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is True
    assert task.cancelled is True


@pytest.mark.asyncio
async def test_shutdown_jobs_webhooks_worker_cancels_on_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_optional = _import_shutdown_optional_workers()
    task = _FakeTask()
    stop_event = _FakeStopEvent()

    async def _timeout_wait(_task, *, timeout):
        del timeout
        raise asyncio.TimeoutError()

    monkeypatch.setattr(shutdown_optional, "_wait_for_task", _timeout_wait)

    await shutdown_optional._shutdown_jobs_webhooks_worker(
        task=task,
        stop_event=stop_event,
        guard_exceptions=(RuntimeError,),
    )

    assert stop_event.is_set is True
    assert task.cancelled is True


@pytest.mark.asyncio
async def test_shutdown_jobs_crypto_rotate_worker_waits_after_cancel_without_stop_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shutdown_optional = _import_shutdown_optional_workers()
    task = _FakeTask()
    waits: list[tuple[object, float]] = []

    async def _fake_wait(task_obj, *, timeout):
        waits.append((task_obj, timeout))

    monkeypatch.setattr(shutdown_optional, "_wait_for_task", _fake_wait)

    await shutdown_optional._shutdown_jobs_crypto_rotate_worker(
        task=task,
        stop_event=None,
        guard_exceptions=(RuntimeError,),
    )

    assert task.cancelled is True
    assert waits == [(task, 5.0)]
