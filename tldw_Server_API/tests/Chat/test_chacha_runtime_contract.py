import asyncio
import contextlib
import inspect
import threading
import time
from pathlib import Path

import pytest
from fastapi import HTTPException, status

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.core.DB_Management.chacha.runtime import (
    ChaChaRuntimeManager,
    ChaChaRuntimeUnavailableError,
)


class _DummyConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, sql: str):
        self.statements.append(sql)
        return None


class _DummyPostgresDB:
    backend_type = BackendType.POSTGRESQL

    def __init__(self, connection: _DummyConnection) -> None:
        self._connection = connection

    def get_connection(self) -> _DummyConnection:
        return self._connection


def test_runtime_manager_exposes_explicit_resettable_surface():
    runtime = ChaChaRuntimeManager()

    assert hasattr(runtime, "get_or_create")
    assert hasattr(runtime, "shutdown")
    assert hasattr(runtime, "snapshot")
    assert hasattr(runtime, "schedule_default_character_ensure")
    runtime.reset_for_tests()


def test_dependency_module_preserves_compatibility_symbols():
    import tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps as deps

    assert hasattr(deps, "DEFAULT_CHARACTER_NAME")
    assert hasattr(deps, "DEFAULT_CHARACTER_DESCRIPTION")
    assert hasattr(deps, "resolve_chacha_user_base_dir")


@pytest.mark.unit
def test_runtime_health_check_skips_sqlite_tuning_for_non_sqlite_backend(monkeypatch):
    import tldw_Server_API.app.core.DB_Management.chacha.runtime as runtime
    import tldw_Server_API.app.core.DB_Management.sqlite_policy as sqlite_policy

    calls: list[dict[str, object]] = []
    conn = _DummyConnection()

    def fake_configure(connection, **kwargs):
        assert connection is conn
        calls.append(kwargs)

    monkeypatch.setattr(sqlite_policy, "configure_sqlite_connection", fake_configure)

    assert runtime._health_check_instance(_DummyPostgresDB(conn)) is True

    assert calls == []
    assert conn.statements == ["SELECT 1"]


@pytest.mark.asyncio
async def test_dependency_maps_runtime_unavailable_to_503(monkeypatch):
    import tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps as deps

    class _Runtime:
        async def get_or_create(self, *_args, **_kwargs):
            raise ChaChaRuntimeUnavailableError("ChaChaNotes shutdown in progress")

    monkeypatch.setattr(deps, "_CHACHA_RUNTIME", _Runtime())

    with pytest.raises(HTTPException) as exc:
        await deps.get_chacha_db_for_user_id(1, "1")

    assert exc.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "shutdown" in exc.value.detail.lower()


@pytest.mark.asyncio
async def test_timeout_keeps_init_gate_until_background_completion(monkeypatch, tmp_path):
    import tldw_Server_API.app.core.DB_Management.chacha.runtime as runtime

    runtime._reset_for_tests()

    started = threading.Event()
    proceed = threading.Event()
    init_calls: list[tuple[int, str]] = []

    class _FakeDB:
        def close_all_connections(self) -> None:
            return None

    def fake_get_user_base_directory(user_id: int) -> Path:
        return tmp_path / str(user_id)

    def fake_create_and_prepare_db(user_id: int, client_id: str):
        init_calls.append((user_id, client_id))
        started.set()
        proceed.wait(timeout=5)
        return _FakeDB()

    orig_wait_for = runtime.asyncio.wait_for
    timeout_triggered = False

    async def fake_wait_for(awaitable, timeout):  # type: ignore[no-untyped-def]
        nonlocal timeout_triggered
        if not timeout_triggered and not inspect.iscoroutine(awaitable):
            timeout_triggered = True
            raise asyncio.TimeoutError()
        return await orig_wait_for(awaitable, timeout=timeout)

    monkeypatch.setattr(runtime.DatabasePaths, "get_user_base_directory", fake_get_user_base_directory)
    monkeypatch.setattr(runtime, "_create_and_prepare_db", fake_create_and_prepare_db)
    monkeypatch.setattr(runtime.asyncio, "wait_for", fake_wait_for)

    second_task = None
    try:
        first_task = asyncio.create_task(runtime._get_or_init_db_instance(42, "42"))
        with pytest.raises(runtime.ChaChaRuntimeUnavailableError, match="timed out"):
            await first_task

        await asyncio.wait_for(asyncio.to_thread(started.wait), timeout=2.0)
        assert init_calls == [(42, "42")]

        second_task = asyncio.create_task(runtime._get_or_init_db_instance(42, "42"))
        await asyncio.sleep(0.05)
        assert init_calls == [(42, "42")]
        assert not second_task.done()

        proceed.set()
        result = await second_task

        assert isinstance(result, _FakeDB)
        assert init_calls == [(42, "42")]
        assert runtime._snapshot()["cached_instances"] == 1
    finally:
        proceed.set()
        if second_task is not None and not second_task.done():
            second_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await second_task
        runtime._reset_for_tests()


@pytest.mark.asyncio
async def test_cancelled_init_completion_releases_waiters():
    import tldw_Server_API.app.core.DB_Management.chacha.runtime as runtime

    runtime._reset_for_tests()
    cache_key = "cancelled-init"
    init_event = threading.Event()
    completed_future = asyncio.get_running_loop().create_future()
    completed_future.cancel()

    with runtime._STATE.db_lock:
        runtime._STATE.init_events[cache_key] = init_event

    runtime._handle_init_completion(
        cache_key,
        init_event,
        completed_future,
        time.perf_counter(),
    )

    with runtime._STATE.db_lock:
        assert cache_key not in runtime._STATE.init_events
        assert cache_key in runtime._STATE.init_errors

    assert init_event.is_set() is True
    assert isinstance(runtime._STATE.init_errors[cache_key], runtime.ChaChaRuntimeInitError)
    runtime._reset_for_tests()
