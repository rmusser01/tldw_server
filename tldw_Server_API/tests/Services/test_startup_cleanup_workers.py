from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI

pytestmark = pytest.mark.unit


def _import_startup_cleanup_workers() -> Any:
    sys.modules.pop("tldw_Server_API.app.services.startup_cleanup_workers", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_cleanup_workers")


@pytest.mark.asyncio
async def test_start_cleanup_workers_combines_all_handles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_cleanup = _import_startup_cleanup_workers()
    calls: list[str] = []

    async def _fake_ephemeral(
        app_settings: dict[str, str],
        *,
        worker_inventory: object | None = None,
    ) -> str:
        assert worker_inventory is None
        calls.append("ephemeral")
        assert app_settings["SINGLE_USER_FIXED_ID"] == "7"
        return "ephemeral-task"

    async def _fake_chatbooks(*, worker_inventory: object | None = None) -> tuple[str, str]:
        assert worker_inventory is None
        calls.append("chatbooks")
        return ("chatbooks-task", "chatbooks-stop")

    async def _fake_storage(
        *,
        test_mode: bool,
        worker_inventory: object | None = None,
    ) -> str:
        assert worker_inventory is None
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
async def test_start_cleanup_workers_passes_worker_inventory_to_registered_cleanup_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_cleanup = _import_startup_cleanup_workers()
    worker_inventory = object()
    calls: list[tuple[str, object | None]] = []

    async def _fake_ephemeral(
        app_settings: dict[str, str],
        *,
        worker_inventory: object | None = None,
    ) -> str:
        assert app_settings["SINGLE_USER_FIXED_ID"] == "7"
        calls.append(("ephemeral", worker_inventory))
        return "ephemeral-task"

    async def _fake_chatbooks(*, worker_inventory: object | None = None) -> tuple[str, str]:
        calls.append(("chatbooks", worker_inventory))
        return ("chatbooks-task", "chatbooks-stop")

    async def _fake_storage(
        *,
        test_mode: bool,
        worker_inventory: object | None = None,
    ) -> None:
        assert test_mode is True
        calls.append(("storage", worker_inventory))
        return None

    monkeypatch.setattr(startup_cleanup, "_start_ephemeral_cleanup_worker", _fake_ephemeral)
    monkeypatch.setattr(startup_cleanup, "_start_chatbooks_cleanup_worker", _fake_chatbooks)
    monkeypatch.setattr(startup_cleanup, "_start_storage_cleanup_worker", _fake_storage)

    handles = await startup_cleanup.start_cleanup_workers(
        {"SINGLE_USER_FIXED_ID": "7"},
        test_mode=True,
        worker_inventory=worker_inventory,
    )

    assert calls == [
        ("ephemeral", worker_inventory),
        ("chatbooks", worker_inventory),
        ("storage", worker_inventory),
    ]
    assert handles.cleanup_task == "ephemeral-task"
    assert handles.chatbooks_cleanup_task == "chatbooks-task"
    assert handles.chatbooks_cleanup_stop_event == "chatbooks-stop"


@pytest.mark.asyncio
async def test_start_chatbooks_cleanup_worker_starts_when_interval_positive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_cleanup = _import_startup_cleanup_workers()
    monkeypatch.setenv("CHATBOOKS_CLEANUP_INTERVAL_SEC", "30")
    started_with: list[object] = []
    created_tasks: list[SimpleNamespace] = []

    async def _fake_runner(stop_event: object) -> None:
        started_with.append(stop_event)

    def _record_create_task(coro: Any, *, name: str | None = None) -> SimpleNamespace:
        task = SimpleNamespace(coro=coro, name=name, cancel=lambda: None)
        created_tasks.append(task)
        coro.close()
        return task

    monkeypatch.setattr(startup_cleanup.asyncio, "create_task", _record_create_task)
    monkeypatch.setattr(startup_cleanup, "_run_chatbooks_cleanup_loop", _fake_runner)

    task, stop_event = await startup_cleanup._start_chatbooks_cleanup_worker()

    assert task is created_tasks[0]
    assert created_tasks[0].name == "chatbooks_cleanup_task"
    assert stop_event is not None


@pytest.mark.asyncio
async def test_start_chatbooks_cleanup_worker_registers_background_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_cleanup = _import_startup_cleanup_workers()
    monkeypatch.setenv("CHATBOOKS_CLEANUP_INTERVAL_SEC", "30")
    calls: list[dict[str, object]] = []
    task = object()
    stop_event = object()

    async def _fake_start_stop_event_worker(
        inventory: object,
        **kwargs: object,
    ) -> tuple[object, object]:
        calls.append({"inventory": inventory, **kwargs})
        return task, stop_event

    worker_inventory = object()
    monkeypatch.setattr(startup_cleanup, "start_stop_event_worker", _fake_start_stop_event_worker)

    returned_task, returned_stop_event = await startup_cleanup._start_chatbooks_cleanup_worker(
        worker_inventory=worker_inventory,
    )

    assert returned_task is task
    assert returned_stop_event is stop_event
    assert calls == [
        {
            "inventory": worker_inventory,
            "name": "chatbooks_cleanup",
            "task_name": "chatbooks_cleanup_task",
            "coroutine_factory": startup_cleanup._run_chatbooks_cleanup_loop,
            "category": "cleanup",
            "shutdown_phase": startup_cleanup.ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN,
        }
    ]


@pytest.mark.asyncio
async def test_start_chatbooks_cleanup_worker_has_single_background_registry_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_cleanup = _import_startup_cleanup_workers()
    from tldw_Server_API.app.services.worker_registry import WorkerRegistry

    monkeypatch.setenv("CHATBOOKS_CLEANUP_INTERVAL_SEC", "30")
    runner_started = asyncio.Event()

    async def _fake_runner(stop_event: asyncio.Event) -> None:
        runner_started.set()
        await stop_event.wait()

    app = FastAPI()
    worker_inventory = WorkerRegistry(app)
    monkeypatch.setattr(startup_cleanup, "_run_chatbooks_cleanup_loop", _fake_runner)

    task, stop_event = await startup_cleanup._start_chatbooks_cleanup_worker(
        worker_inventory=worker_inventory,
    )
    try:
        await asyncio.wait_for(runner_started.wait(), timeout=1.0)

        handles = worker_inventory.handles_for_phase(
            startup_cleanup.ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
        )
        assert [handle.name for handle in handles].count("chatbooks_cleanup") == 1
        assert app.state._tldw_shutdown_job_poller_inventory == []
        assert [
            item
            for item in app.state._tldw_shutdown_worker_inventory
            if item["name"] == "chatbooks_cleanup"
        ] == [
            {
                "name": "chatbooks_cleanup",
                "task_name": "chatbooks_cleanup_task",
                "has_stop_event": True,
                "timeout_sec": 5.0,
                "category": "cleanup",
                "shutdown_phase": "background_worker_shutdown",
            }
        ]
    finally:
        if stop_event is not None:
            stop_event.set()
        if task is not None:
            await asyncio.wait_for(task, timeout=1.0)


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
async def test_start_storage_cleanup_worker_registers_background_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_cleanup = _import_startup_cleanup_workers()
    from tldw_Server_API.app.services.worker_registry import WorkerRegistry

    class _FakeService:
        def __init__(self) -> None:
            self.stop_event = asyncio.Event()
            self.task: asyncio.Task[None] | None = None
            self.stop_calls = 0

        async def _run(self) -> None:
            await self.stop_event.wait()

        async def start(self) -> None:
            self.task = asyncio.create_task(self._run(), name="storage_cleanup_service")

        async def stop(self) -> None:
            self.stop_calls += 1
            self.stop_event.set()
            if self.task is not None:
                await asyncio.wait_for(self.task, timeout=1)

    app = FastAPI()
    worker_inventory = WorkerRegistry(app)
    service = _FakeService()
    monkeypatch.setenv("STORAGE_CLEANUP_ENABLED", "true")
    monkeypatch.setattr(startup_cleanup, "_get_storage_cleanup_service", lambda: service)

    try:
        returned_service = await startup_cleanup._start_storage_cleanup_worker(
            test_mode=False,
            worker_inventory=worker_inventory,
        )

        assert returned_service is service
        assert service.task is not None
        assert len(worker_inventory.handles) == 1
        handle = worker_inventory.handles[0]
        assert handle.name == "storage_cleanup_service"
        assert handle.task is service.task
        assert handle.stop_event is None
        assert handle.shutdown_callback is not None
        assert handle.category == "cleanup"
        assert handle.shutdown_phase is startup_cleanup.ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
        assert app.state._tldw_shutdown_worker_inventory == [
            {
                "name": "storage_cleanup_service",
                "task_name": "storage_cleanup_service",
                "has_stop_event": False,
                "timeout_sec": 5.0,
                "category": "cleanup",
                "shutdown_phase": "background_worker_shutdown",
            }
        ]
        assert app.state._tldw_shutdown_job_poller_inventory == []

        await handle.shutdown_callback()
        assert service.stop_calls == 1
    finally:
        if service.task is not None and not service.task.done():
            await service.stop()


@pytest.mark.asyncio
async def test_start_ephemeral_cleanup_worker_creates_task_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    startup_cleanup = _import_startup_cleanup_workers()
    created_tasks: list[SimpleNamespace] = []

    def _record_create_task(coro: Any, *, name: str | None = None) -> SimpleNamespace:
        task = SimpleNamespace(coro=coro, name=name, cancel=lambda: None)
        created_tasks.append(task)
        coro.close()
        return task

    monkeypatch.setattr(startup_cleanup.asyncio, "create_task", _record_create_task)
    monkeypatch.setattr(startup_cleanup, "_create_evaluations_db", lambda db_path: SimpleNamespace())
    monkeypatch.setattr(
        startup_cleanup,
        "_create_vector_store_adapter",
        lambda settings, user_id: SimpleNamespace(initialize=lambda: None),
    )
    monkeypatch.setattr(
        startup_cleanup,
        "_get_evaluations_db_path",
        lambda user_id: str(tmp_path / f"evals-{user_id}.db"),
    )

    task = await startup_cleanup._start_ephemeral_cleanup_worker(
        {"SINGLE_USER_FIXED_ID": "11", "EPHEMERAL_CLEANUP_ENABLED": True, "EPHEMERAL_CLEANUP_INTERVAL_SEC": 9}
    )

    assert task is created_tasks[0]


@pytest.mark.asyncio
async def test_start_ephemeral_cleanup_worker_registers_background_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_cleanup = _import_startup_cleanup_workers()
    worker_inventory = object()
    task = object()
    stop_event = object()
    calls: list[dict[str, object]] = []

    async def _fake_start_stop_event_worker(
        inventory: object,
        **kwargs: object,
    ) -> tuple[object, object]:
        calls.append({"inventory": inventory, **kwargs})
        return task, stop_event

    monkeypatch.setattr(startup_cleanup, "start_stop_event_worker", _fake_start_stop_event_worker)
    monkeypatch.setattr(startup_cleanup, "_create_evaluations_db", lambda db_path: SimpleNamespace())
    monkeypatch.setattr(
        startup_cleanup,
        "_create_vector_store_adapter",
        lambda settings, user_id: SimpleNamespace(initialize=lambda: None),
    )
    monkeypatch.setattr(
        startup_cleanup,
        "_get_evaluations_db_path",
        lambda user_id: f"evals-{user_id}.db",
    )

    returned_task = await startup_cleanup._start_ephemeral_cleanup_worker(
        {"SINGLE_USER_FIXED_ID": "11", "EPHEMERAL_CLEANUP_ENABLED": True},
        worker_inventory=worker_inventory,
    )

    assert returned_task is task
    assert len(calls) == 1
    assert calls[0]["inventory"] is worker_inventory
    assert calls[0]["name"] == "ephemeral_cleanup_task"
    assert calls[0]["task_name"] == "ephemeral_cleanup_task"
    assert calls[0]["category"] == "cleanup"
    assert calls[0]["shutdown_phase"] == startup_cleanup.ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN
    assert callable(calls[0]["coroutine_factory"])


@pytest.mark.asyncio
async def test_run_ephemeral_cleanup_loop_exits_when_stop_event_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_cleanup = _import_startup_cleanup_workers()
    stop_event = asyncio.Event()
    stop_event.set()

    def _fail_create_evaluations_db(_db_path: str) -> None:
        raise AssertionError("cleanup loop should not initialize DB after stop")

    monkeypatch.setattr(startup_cleanup, "_create_evaluations_db", _fail_create_evaluations_db)

    await startup_cleanup._run_ephemeral_cleanup_loop(
        {"SINGLE_USER_FIXED_ID": "11", "EPHEMERAL_CLEANUP_INTERVAL_SEC": 1},
        stop_event=stop_event,
    )


@pytest.mark.asyncio
async def test_run_ephemeral_cleanup_loop_stops_delete_batch_after_stop_request(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    startup_cleanup = _import_startup_cleanup_workers()
    stop_event = asyncio.Event()
    deleted: list[str] = []
    marked: list[str] = []

    class _FakeDB:
        def list_expired_ephemeral_collections(self) -> list[str]:
            return ["expired-a", "expired-b"]

        def mark_ephemeral_deleted(self, collection_name: str) -> None:
            marked.append(collection_name)

    class _FakeAdapter:
        def initialize(self) -> None:
            return None

        async def delete_collection(self, collection_name: str) -> None:
            deleted.append(collection_name)
            stop_event.set()

    monkeypatch.setattr(startup_cleanup, "_create_evaluations_db", lambda db_path: _FakeDB())
    monkeypatch.setattr(
        startup_cleanup,
        "_create_vector_store_adapter",
        lambda settings, user_id: _FakeAdapter(),
    )

    await startup_cleanup._run_ephemeral_cleanup_loop(
        {"SINGLE_USER_FIXED_ID": "11", "EPHEMERAL_CLEANUP_INTERVAL_SEC": 1},
        single_uid=11,
        db_path=str(tmp_path / "evals.db"),
        interval_sec=1,
        stop_event=stop_event,
    )

    assert deleted == ["expired-a"]
    assert marked == ["expired-a"]


@pytest.mark.asyncio
async def test_start_ephemeral_cleanup_worker_skips_when_disabled() -> None:
    startup_cleanup = _import_startup_cleanup_workers()

    task = await startup_cleanup._start_ephemeral_cleanup_worker(
        {"SINGLE_USER_FIXED_ID": "11", "EPHEMERAL_CLEANUP_ENABLED": False}
    )

    assert task is None


@pytest.mark.asyncio
async def test_start_ephemeral_cleanup_worker_treats_false_string_as_disabled() -> None:
    startup_cleanup = _import_startup_cleanup_workers()

    task = await startup_cleanup._start_ephemeral_cleanup_worker(
        {"SINGLE_USER_FIXED_ID": "11", "EPHEMERAL_CLEANUP_ENABLED": "false"}
    )

    assert task is None
