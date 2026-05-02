from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


def _import_startup_claims_rebuild():
    sys.modules.pop("tldw_Server_API.app.services.startup_claims_rebuild", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_claims_rebuild")


@pytest.mark.asyncio
async def test_start_claims_rebuild_worker_skips_when_disabled() -> None:
    startup_claims = _import_startup_claims_rebuild()

    task = await startup_claims.start_claims_rebuild_worker(
        {"CLAIMS_REBUILD_ENABLED": False},
    )

    assert task is None


@pytest.mark.asyncio
async def test_start_claims_rebuild_worker_creates_task_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_claims = _import_startup_claims_rebuild()
    created_tasks = []

    def _record_create_task(coro):
        task = SimpleNamespace(coro=coro, cancel=lambda: None)
        created_tasks.append(task)
        coro.close()
        return task

    monkeypatch.setattr(startup_claims.asyncio, "create_task", _record_create_task)

    task = await startup_claims.start_claims_rebuild_worker(
        {
            "CLAIMS_REBUILD_ENABLED": True,
            "CLAIMS_REBUILD_INTERVAL_SEC": 17,
            "CLAIMS_REBUILD_POLICY": "stale",
        },
    )

    assert task is created_tasks[0]
    assert getattr(task, "_tldw_claims_rebuild_stop_event") is not None


@pytest.mark.asyncio
async def test_start_claims_rebuild_worker_registers_background_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_claims = _import_startup_claims_rebuild()
    worker_inventory = object()
    task = object()
    stop_event = object()
    calls: list[dict[str, object]] = []

    async def _fake_start_stop_event_worker(inventory, **kwargs):
        calls.append({"inventory": inventory, **kwargs})
        return task, stop_event

    monkeypatch.setattr(
        startup_claims,
        "start_stop_event_worker",
        _fake_start_stop_event_worker,
    )

    returned_task = await startup_claims.start_claims_rebuild_worker(
        {
            "CLAIMS_REBUILD_ENABLED": True,
            "CLAIMS_REBUILD_INTERVAL_SEC": 17,
            "CLAIMS_REBUILD_POLICY": "stale",
        },
        worker_inventory=worker_inventory,
    )

    assert returned_task is task
    assert len(calls) == 1
    assert calls[0]["inventory"] is worker_inventory
    assert calls[0]["name"] == "claims_rebuild"
    assert calls[0]["task_name"] == "claims_task"
    assert calls[0]["category"] == "claims"
    assert calls[0]["shutdown_phase"] == startup_claims.ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
    assert callable(calls[0]["coroutine_factory"])


def test_run_claims_rebuild_iteration_submits_selected_media_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_claims = _import_startup_claims_rebuild()
    submissions: list[tuple[int, str]] = []

    class _FakeService:
        def submit(self, *, media_id: int, db_path: str) -> None:
            submissions.append((media_id, db_path))

    @contextmanager
    def _fake_db_session(app_settings):
        assert app_settings["CLAIMS_STALE_DAYS"] == 11
        yield 1, "/tmp/media.db", object()

    def _fake_list_media_ids(db, *, policy, stale_days, compare_media_last_modified, limit):
        assert policy == "stale"
        assert stale_days == 11
        assert compare_media_last_modified is False
        assert limit == 25
        return [101, 202]

    monkeypatch.setattr(startup_claims, "_claims_rebuild_db_session", _fake_db_session)
    monkeypatch.setattr(startup_claims, "_list_claims_rebuild_media_ids", _fake_list_media_ids)

    startup_claims.run_claims_rebuild_iteration(
        {
            "CLAIMS_STALE_DAYS": 11,
        },
        _FakeService(),
        policy="stale",
    )

    assert submissions == [
        (101, "/tmp/media.db"),
        (202, "/tmp/media.db"),
    ]
