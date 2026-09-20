from __future__ import annotations

import asyncio

import pytest

from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDBError
from tldw_Server_API.app.services.lifecycle_worker_specs import (
    ShutdownPhase,
    WorkerLifecycleContext,
)
from tldw_Server_API.app.services.notes_graph_projection_worker import (
    provide_notes_graph_projection_worker_specs,
    run_notes_graph_projection_worker,
)
from tldw_Server_API.app.services.startup_worker_groups import (
    startup_worker_spec_providers,
)

pytestmark = pytest.mark.unit


def _context(*, test_mode: bool) -> WorkerLifecycleContext:
    return WorkerLifecycleContext(
        app=object(),
        settings={},
        test_mode=test_mode,
        route_enabled=lambda *_args, **_kwargs: True,
        logger=None,
        startup_guard_exceptions=(),
        import_exceptions=(),
    )


def test_projection_worker_is_registered_and_test_mode_defaults_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NOTES_GRAPH_PROJECTION_MAINTENANCE_ENABLED", raising=False)
    [spec] = provide_notes_graph_projection_worker_specs()

    assert provide_notes_graph_projection_worker_specs in startup_worker_spec_providers()
    assert spec.name == "notes_graph_projection_maintenance_task"
    assert spec.category == "notes-graph"
    assert spec.phase is ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
    assert spec.enabled(_context(test_mode=True)) is False
    assert spec.enabled(_context(test_mode=False)) is True


@pytest.mark.asyncio
async def test_projection_worker_uses_only_cached_owner_bound_instances(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_db = object()
    second_db = object()
    calls: list[object] = []
    stop_event = asyncio.Event()

    monkeypatch.setattr(
        "tldw_Server_API.app.services.notes_graph_projection_worker.snapshot_cached_chacha_db_instances",
        lambda: (first_db, second_db),
    )

    def run_once(db: object, **_kwargs: object) -> int:
        calls.append(db)
        if db is second_db:
            stop_event.set()
        return 0

    monkeypatch.setattr(
        "tldw_Server_API.app.services.notes_graph_projection_worker.run_notes_graph_projection_maintenance_once",
        run_once,
    )

    await run_notes_graph_projection_worker(stop_event, interval_seconds=0.01)

    assert calls == [first_db, second_db]


@pytest.mark.asyncio
async def test_projection_worker_isolates_one_owner_database_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_db = object()
    second_db = object()
    calls: list[object] = []
    stop_event = asyncio.Event()

    monkeypatch.setattr(
        "tldw_Server_API.app.services.notes_graph_projection_worker.snapshot_cached_chacha_db_instances",
        lambda: (first_db, second_db),
    )

    def run_once(db: object, **_kwargs: object) -> int:
        calls.append(db)
        if db is first_db:
            raise CharactersRAGDBError("injected owner DB failure")
        stop_event.set()
        return 0

    monkeypatch.setattr(
        "tldw_Server_API.app.services.notes_graph_projection_worker.run_notes_graph_projection_maintenance_once",
        run_once,
    )

    await run_notes_graph_projection_worker(stop_event, interval_seconds=0.01)

    assert calls == [first_db, second_db]
