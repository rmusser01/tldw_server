from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest


pytestmark = pytest.mark.unit


def _import_startup_cleanup_workers():
    sys.modules.pop("tldw_Server_API.app.services.startup_cleanup_workers", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_cleanup_workers")


@pytest.mark.asyncio
async def test_start_cleanup_workers_combines_all_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_cleanup = _import_startup_cleanup_workers()
    calls: list[str] = []

    async def _fake_ephemeral(app_settings):
        calls.append("ephemeral")
        assert app_settings["SINGLE_USER_FIXED_ID"] == "7"
        return "ephemeral-task"

    async def _fake_chatbooks():
        calls.append("chatbooks")
        return ("chatbooks-task", "chatbooks-stop")

    async def _fake_storage(*, test_mode: bool):
        calls.append("storage")
        assert test_mode is True
        return "storage-service"

    monkeypatch.setattr(startup_cleanup, "_start_ephemeral_cleanup_worker", _fake_ephemeral)
    monkeypatch.setattr(startup_cleanup, "_start_chatbooks_cleanup_worker", _fake_chatbooks)
    monkeypatch.setattr(startup_cleanup, "_start_storage_cleanup_worker", _fake_storage)

    handles = await startup_cleanup.start_cleanup_workers(
        {"SINGLE_USER_FIXED_ID": "7"},
        test_mode=True,
    )

    assert calls == ["ephemeral", "chatbooks", "storage"]
    assert handles.cleanup_task == "ephemeral-task"
    assert handles.chatbooks_cleanup_task == "chatbooks-task"
    assert handles.chatbooks_cleanup_stop_event == "chatbooks-stop"
    assert handles.storage_cleanup_service == "storage-service"


@pytest.mark.asyncio
async def test_start_chatbooks_cleanup_worker_starts_when_interval_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_cleanup = _import_startup_cleanup_workers()
    monkeypatch.setenv("CHATBOOKS_CLEANUP_INTERVAL_SEC", "30")
    started_with: list[object] = []
    created_tasks = []

    async def _fake_runner(stop_event):
        started_with.append(stop_event)

    def _record_create_task(coro):
        task = SimpleNamespace(coro=coro, cancel=lambda: None)
        created_tasks.append(task)
        coro.close()
        return task

    monkeypatch.setattr(startup_cleanup.asyncio, "create_task", _record_create_task)
    monkeypatch.setattr(startup_cleanup, "_run_chatbooks_cleanup_loop", _fake_runner)

    task, stop_event = await startup_cleanup._start_chatbooks_cleanup_worker()

    assert task is created_tasks[0]
    assert stop_event is not None


@pytest.mark.asyncio
async def test_start_chatbooks_cleanup_worker_skips_when_interval_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_cleanup = _import_startup_cleanup_workers()
    monkeypatch.setenv("CHATBOOKS_CLEANUP_INTERVAL_SEC", "0")

    task, stop_event = await startup_cleanup._start_chatbooks_cleanup_worker()

    assert task is None
    assert stop_event is None


@pytest.mark.asyncio
async def test_start_storage_cleanup_worker_uses_test_mode_default_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_cleanup = _import_startup_cleanup_workers()
    monkeypatch.delenv("STORAGE_CLEANUP_ENABLED", raising=False)

    service = await startup_cleanup._start_storage_cleanup_worker(test_mode=True)

    assert service is None


@pytest.mark.asyncio
async def test_start_storage_cleanup_worker_starts_enabled_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_cleanup = _import_startup_cleanup_workers()
    monkeypatch.setenv("STORAGE_CLEANUP_ENABLED", "true")
    started: list[str] = []

    class _FakeService:
        async def start(self) -> None:
            started.append("start")

    monkeypatch.setattr(startup_cleanup, "_get_storage_cleanup_service", lambda: _FakeService())

    service = await startup_cleanup._start_storage_cleanup_worker(test_mode=False)

    assert service is not None
    assert started == ["start"]


@pytest.mark.asyncio
async def test_start_ephemeral_cleanup_worker_creates_task_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_cleanup = _import_startup_cleanup_workers()
    created_tasks = []

    def _record_create_task(coro):
        task = SimpleNamespace(coro=coro, cancel=lambda: None)
        created_tasks.append(task)
        coro.close()
        return task

    monkeypatch.setattr(startup_cleanup.asyncio, "create_task", _record_create_task)
    monkeypatch.setattr(startup_cleanup, "_create_evaluations_db", lambda db_path: SimpleNamespace())
    monkeypatch.setattr(startup_cleanup, "_create_vector_store_adapter", lambda settings, user_id: SimpleNamespace(initialize=lambda: None))
    monkeypatch.setattr(startup_cleanup, "_get_evaluations_db_path", lambda user_id: f"/tmp/evals-{user_id}.db")

    task = await startup_cleanup._start_ephemeral_cleanup_worker(
        {"SINGLE_USER_FIXED_ID": "11", "EPHEMERAL_CLEANUP_ENABLED": True, "EPHEMERAL_CLEANUP_INTERVAL_SEC": 9}
    )

    assert task is created_tasks[0]


@pytest.mark.asyncio
async def test_start_ephemeral_cleanup_worker_skips_when_disabled() -> None:
    startup_cleanup = _import_startup_cleanup_workers()

    task = await startup_cleanup._start_ephemeral_cleanup_worker(
        {"SINGLE_USER_FIXED_ID": "11", "EPHEMERAL_CLEANUP_ENABLED": False}
    )

    assert task is None
