from __future__ import annotations

import importlib
import sys

import pytest


pytestmark = pytest.mark.unit


def _import_startup_compactor_websub_workers():
    sys.modules.pop("tldw_Server_API.app.services.startup_compactor_websub_workers", None)
    return importlib.import_module("tldw_Server_API.app.services.startup_compactor_websub_workers")


@pytest.mark.asyncio
async def test_start_compactor_websub_workers_combines_handles_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_compactor_websub_workers()
    calls: list[str] = []

    async def _record_compactor(**kwargs):
        del kwargs
        calls.append("compactor")
        return ("compactor-stop", "compactor-task")

    async def _record_websub(**kwargs):
        del kwargs
        calls.append("websub")
        return "websub-task"

    monkeypatch.setattr(startup_workers, "_start_embeddings_vector_compactor", _record_compactor)
    monkeypatch.setattr(startup_workers, "_start_websub_renewal_worker", _record_websub)

    handles = await startup_workers.start_compactor_websub_workers(
        should_start_worker=lambda *args, **kwargs: False,
    )

    assert calls == ["compactor", "websub"]
    assert handles.embeddings_compactor_stop_event == "compactor-stop"
    assert handles.embeddings_compactor_task == "compactor-task"
    assert handles.websub_renewal_task == "websub-task"


@pytest.mark.asyncio
async def test_start_embeddings_vector_compactor_starts_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_compactor_websub_workers()
    captured_stop_events: list[object] = []
    created_coroutines: list[object] = []

    monkeypatch.setattr(
        startup_workers.os,
        "getenv",
        lambda key, default=None: "true" if key == "EMBEDDINGS_COMPACTOR_ENABLED" else default,
    )
    monkeypatch.setattr(startup_workers, "_make_event", lambda: "compactor-stop")
    monkeypatch.setattr(
        startup_workers,
        "_create_task",
        lambda coro: created_coroutines.append(coro) or "compactor-task",
    )
    monkeypatch.setattr(
        startup_workers,
        "_run_embeddings_vector_compactor_service",
        lambda stop_event: captured_stop_events.append(stop_event) or "compactor-coro",
    )

    stop_event, task = await startup_workers._start_embeddings_vector_compactor()

    assert stop_event == "compactor-stop"
    assert task == "compactor-task"
    assert captured_stop_events == ["compactor-stop"]
    assert created_coroutines == ["compactor-coro"]


@pytest.mark.asyncio
async def test_start_embeddings_vector_compactor_handles_guard_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_compactor_websub_workers()

    monkeypatch.setattr(
        startup_workers.os,
        "getenv",
        lambda key, default=None: "true" if key == "EMBEDDINGS_COMPACTOR_ENABLED" else default,
    )
    monkeypatch.setattr(startup_workers, "_make_event", lambda: "compactor-stop")

    def _failing_create_task(coro):
        del coro
        raise RuntimeError("boom")

    monkeypatch.setattr(startup_workers, "_create_task", _failing_create_task)
    monkeypatch.setattr(
        startup_workers,
        "_run_embeddings_vector_compactor_service",
        lambda stop_event: stop_event,
    )

    stop_event, task = await startup_workers._start_embeddings_vector_compactor()

    assert stop_event is None
    assert task is None


@pytest.mark.asyncio
async def test_start_websub_renewal_worker_starts_when_callback_and_worker_enabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_compactor_websub_workers()
    created_coroutines: list[object] = []

    monkeypatch.setattr(
        startup_workers.os,
        "getenv",
        lambda key, default=None: "http://callback.example" if key == "WEBSUB_CALLBACK_BASE_URL" else default,
    )
    monkeypatch.setattr(
        startup_workers,
        "_create_task",
        lambda coro: created_coroutines.append(coro) or "websub-task",
    )
    monkeypatch.setattr(startup_workers, "_run_websub_renewal_loop", lambda: "websub-coro")

    task = await startup_workers._start_websub_renewal_worker(
        should_start_worker=lambda flag, route, **kwargs: (flag, route, kwargs) == (
            "WEBSUB_RENEWAL_WORKER_ENABLED",
            "collections-websub",
            {},
        ),
    )

    assert task == "websub-task"
    assert created_coroutines == ["websub-coro"]


@pytest.mark.asyncio
async def test_start_websub_renewal_worker_skips_without_callback_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    startup_workers = _import_startup_compactor_websub_workers()
    monkeypatch.setattr(
        startup_workers.os,
        "getenv",
        lambda key, default=None: "" if key == "WEBSUB_CALLBACK_BASE_URL" else default,
    )
    monkeypatch.setattr(
        startup_workers,
        "_create_task",
        lambda coro: (_ for _ in ()).throw(AssertionError("no task")),
    )

    task = await startup_workers._start_websub_renewal_worker(
        should_start_worker=lambda *args, **kwargs: True,
    )

    assert task is None
