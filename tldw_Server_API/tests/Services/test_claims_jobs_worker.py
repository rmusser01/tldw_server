from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tldw_Server_API.app.services import claims_jobs_worker
from tldw_Server_API.app.services.lifecycle_worker_specs import (
    ShutdownPhase,
    WorkerLifecycleContext,
)

pytestmark = pytest.mark.unit


def _context(settings: dict[str, Any] | None = None) -> WorkerLifecycleContext:
    return WorkerLifecycleContext(
        app=object(),
        settings=settings or {},
        test_mode=True,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )


def test_claims_jobs_worker_spec_is_job_poller() -> None:
    [spec] = claims_jobs_worker.provide_claims_jobs_worker_specs()

    assert spec.name == "claims_jobs_task"
    assert spec.task_name == "claims_jobs_task"
    assert spec.category == "jobs"
    assert spec.phase is ShutdownPhase.JOB_POLLER_QUIESCE
    assert spec.enabled(_context({"CLAIMS_JOBS_WORKER_ENABLED": True})) is True
    assert spec.enabled(_context({"CLAIMS_JOBS_WORKER_ENABLED": False})) is False


async def test_start_claims_jobs_worker_uses_worker_sdk_without_owner_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    class _FakeSDK:
        def __init__(self, manager: Any, config: Any) -> None:
            observed["manager"] = manager
            observed["config"] = config

        def stop(self) -> None:
            observed["stopped"] = True

        async def run(self, **kwargs: Any) -> None:
            observed["run_kwargs"] = kwargs

    monkeypatch.setattr(claims_jobs_worker, "WorkerSDK", _FakeSDK)
    monkeypatch.setattr(claims_jobs_worker, "jobs_manager_from_env", lambda: "manager")
    monkeypatch.delenv("CLAIMS_JOBS_QUEUE", raising=False)
    stop_event = asyncio.Event()
    stop_event.set()

    await claims_jobs_worker.start_claims_jobs_worker(stop_event=stop_event)

    assert observed["manager"] == "manager"
    assert observed["config"].domain == "claims"
    assert observed["config"].queue == "default"
    assert observed["run_kwargs"]["handler"] is claims_jobs_worker.process_claims_job
    assert "owner_user_id" not in observed["run_kwargs"]
    assert observed["stopped"] is True
