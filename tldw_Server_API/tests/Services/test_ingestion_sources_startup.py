"""Sources sync belongs to the normal opt-in lifecycle and quiescent shutdown."""

from __future__ import annotations

import asyncio

import pytest
from fastapi import FastAPI

from tldw_Server_API.app.services.lifecycle_worker_engine import LifecycleWorkerEngine
from tldw_Server_API.app.services.lifecycle_worker_specs import ShutdownPhase, WorkerLifecycleContext, WorkerSpec
from tldw_Server_API.app.services.startup_worker_groups import collect_startup_worker_specs

pytestmark = pytest.mark.unit


def _context(*, route: bool = True, sidecar: bool = False) -> WorkerLifecycleContext:
    return WorkerLifecycleContext(
        app=FastAPI(),
        settings={},
        test_mode=False,
        route_enabled=lambda key, **_kwargs: key == "ingestion-sources" and route,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
        sidecar_mode=sidecar,
    )


def _spec() -> WorkerSpec:
    return next(spec for spec in collect_startup_worker_specs(_context()) if spec.name == "ingestion_sources_task")


def test_sources_sync_worker_is_managed_by_startup() -> None:
    names = {spec.name for spec in collect_startup_worker_specs(_context())}
    assert "ingestion_sources_task" in names  # nosec B101 -- Pytest verifies the real startup catalog, not a runtime guard.


@pytest.mark.parametrize(
    "flag,route,sidecar,expected",
    [
        (None, True, False, False),
        ("0", True, False, False),
        ("1", True, False, True),
        ("1", False, False, False),
        ("1", True, True, False),
        ("0", True, True, False),
    ],
)
def test_sources_worker_requires_opt_in_route_and_inprocess_ownership(
    monkeypatch: pytest.MonkeyPatch, flag: str | None, route: bool, sidecar: bool, expected: bool
) -> None:
    if flag is None:
        monkeypatch.delenv("INGESTION_SOURCES_WORKER_ENABLED", raising=False)
    else:
        monkeypatch.setenv("INGESTION_SOURCES_WORKER_ENABLED", flag)
    assert _spec().enabled(_context(route=route, sidecar=sidecar)) is expected  # nosec B101 -- Pytest checks worker admission boundaries.


@pytest.mark.asyncio
async def test_sources_worker_receives_owned_stop_event_and_quiesces(monkeypatch: pytest.MonkeyPatch) -> None:
    from tldw_Server_API.app.services import ingestion_sources_worker

    started = asyncio.Event()
    stopped = asyncio.Event()
    received: list[asyncio.Event] = []

    async def worker(stop_event: asyncio.Event) -> None:
        received.append(stop_event)
        started.set()
        await stop_event.wait()
        stopped.set()

    monkeypatch.setenv("INGESTION_SOURCES_WORKER_ENABLED", "1")
    monkeypatch.setattr(ingestion_sources_worker, "run_ingestion_sources_worker", worker)
    engine = LifecycleWorkerEngine()
    spec = _spec()
    session = await engine.start(_context(), [spec])
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        handle = session.handles_by_name[spec.name]
        assert received == [handle.stop_event]  # nosec B101 -- Pytest verifies exact cooperative-stop ownership.
        assert spec.phase is ShutdownPhase.JOB_POLLER_QUIESCE  # nosec B101 -- Pytest checks shutdown ordering.
    finally:
        await engine.stop_phase(session, ShutdownPhase.JOB_POLLER_QUIESCE)
    assert stopped.is_set() and handle.task.done()  # nosec B101 -- Pytest requires the worker to retire.
