from __future__ import annotations

import asyncio
import importlib
import threading
from pathlib import Path

import pytest
from fastapi import HTTPException, status

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType


class _DummyConnection:
    def __init__(self) -> None:
        self.statements: list[str] = []

    def execute(self, sql: str):
        self.statements.append(sql)
        return None


class _DummySQLiteDB:
    backend_type = BackendType.SQLITE

    def __init__(self, connection: _DummyConnection) -> None:
        self._connection = connection

    def get_connection(self) -> _DummyConnection:
        return self._connection


@pytest.mark.unit
def test_chacha_dependency_tuning_uses_shared_sqlite_policy_helper(monkeypatch):
    import tldw_Server_API.app.core.DB_Management.sqlite_policy as sqlite_policy
    import tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps as deps

    calls: list[dict[str, object]] = []
    conn = _DummyConnection()

    def fake_configure(connection, **kwargs):
        assert connection is conn
        calls.append(kwargs)

    monkeypatch.setattr(sqlite_policy, "configure_sqlite_connection", fake_configure)
    deps = importlib.reload(deps)

    deps._apply_sqlite_tuning(_DummySQLiteDB(conn))

    assert calls == [
        {
            "use_wal": True,
            "synchronous": "NORMAL",
            "foreign_keys": True,
            "busy_timeout_ms": 10000,
            "temp_store": None,
        }
    ]


@pytest.mark.unit
def test_chacha_dependency_health_check_uses_shared_sqlite_policy_helper(monkeypatch):
    import tldw_Server_API.app.core.DB_Management.sqlite_policy as sqlite_policy
    import tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps as deps

    calls: list[dict[str, object]] = []
    conn = _DummyConnection()

    def fake_configure(connection, **kwargs):
        assert connection is conn
        calls.append(kwargs)

    monkeypatch.setattr(sqlite_policy, "configure_sqlite_connection", fake_configure)
    deps = importlib.reload(deps)

    assert deps._health_check_instance(_DummySQLiteDB(conn)) is True

    assert calls == [
        {
            "use_wal": False,
            "synchronous": None,
            "foreign_keys": True,
            "busy_timeout_ms": 1000,
            "temp_store": None,
        }
    ]
    assert conn.statements == ["SELECT 1"]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", [True, False, 0, -1])
async def test_get_chacha_db_for_user_id_rejects_bool_and_non_positive_ids(monkeypatch, user_id):
    import tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps as deps

    called = False

    async def fake_get_or_init_db_instance(_user_id, _client_id):
        nonlocal called
        called = True
        return object()

    monkeypatch.setattr(deps, "_get_or_init_db_instance", fake_get_or_init_db_instance)

    with pytest.raises(HTTPException) as exc_info:
        await deps.get_chacha_db_for_user_id(user_id)

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert exc_info.value.detail == "Invalid owner_user_id."
    assert called is False


@pytest.mark.unit
def test_close_all_chacha_db_instances_closes_instances_and_clears_stale_state():
    import tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps as deps

    deps = importlib.reload(deps)

    class _FakeDB:
        def __init__(self) -> None:
            self.closed = 0

        def close_all_connections(self) -> None:
            self.closed += 1

    fake_db = _FakeDB()

    deps._chacha_db_instances["user-1"] = fake_db
    deps._chacha_db_init_errors["stale-key"] = RuntimeError("stale init failure")

    deps.close_all_chacha_db_instances()

    assert fake_db.closed == 1
    assert deps._chacha_db_instances == {}
    assert deps._chacha_db_init_errors == {}
    assert deps._chacha_db_init_events == {}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_close_all_chacha_db_instances_wakes_waiters_with_shutdown_error(monkeypatch):
    import tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps as deps

    deps = importlib.reload(deps)

    class _BlockingInitEvent:
        def __init__(self) -> None:
            self.wait_called = threading.Event()
            self._release = threading.Event()

        def wait(self, timeout: float | None = None) -> bool:
            self.wait_called.set()
            return self._release.wait(timeout)

        def set(self) -> None:
            self._release.set()

    cache_key = "pending-user-1"
    blocking_event = _BlockingInitEvent()

    monkeypatch.setattr(
        deps.DatabasePaths,
        "get_user_base_directory",
        lambda user_id: Path(cache_key),
    )
    deps._chacha_db_init_events[cache_key] = blocking_event

    waiter_task = asyncio.create_task(deps._get_or_init_db_instance(1, "client-1"))
    await asyncio.to_thread(blocking_event.wait_called.wait, 1.0)

    deps.close_all_chacha_db_instances()

    with pytest.raises(HTTPException) as exc_info:
        await waiter_task

    assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert exc_info.value.detail == deps._CHACHA_SHUTDOWN_INIT_ERROR_DETAIL
    assert isinstance(deps._chacha_db_init_errors[cache_key], RuntimeError)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_shutdown_waiter_keeps_abort_sentinel_when_fresh_init_starts(monkeypatch):
    import tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps as deps

    deps = importlib.reload(deps)

    class _DeferredReleaseInitEvent:
        def __init__(self) -> None:
            self.wait_called = threading.Event()
            self.set_called = threading.Event()
            self._allow_wait_return = threading.Event()

        def wait(self, timeout: float | None = None) -> bool:
            self.wait_called.set()
            if not self.set_called.wait(timeout):
                return False
            return self._allow_wait_return.wait(timeout)

        def set(self) -> None:
            self.set_called.set()

        def allow_wait_return(self) -> None:
            self._allow_wait_return.set()

    cache_key = "pending-user-race"
    initial_event = _DeferredReleaseInitEvent()
    fresh_init_started = threading.Event()
    release_fresh_init = threading.Event()

    monkeypatch.setattr(
        deps.DatabasePaths,
        "get_user_base_directory",
        lambda user_id: Path(cache_key),
    )

    class _FreshDB:
        def close_all_connections(self) -> None:
            return None

    def _blocking_create(user_id: int, client_id: str):
        assert user_id == 1
        assert client_id == "client-1-fresh"
        fresh_init_started.set()
        assert release_fresh_init.wait(1.0)
        return _FreshDB()

    deps._chacha_db_init_events[cache_key] = initial_event
    monkeypatch.setattr(deps, "_create_and_prepare_db", _blocking_create)

    old_waiter_task = asyncio.create_task(deps._get_or_init_db_instance(1, "client-1"))
    await asyncio.to_thread(initial_event.wait_called.wait, 1.0)

    deps.close_all_chacha_db_instances()
    assert cache_key in deps._chacha_db_init_errors
    assert isinstance(deps._chacha_db_init_errors[cache_key], RuntimeError)

    fresh_init_task = asyncio.create_task(deps._get_or_init_db_instance(1, "client-1-fresh"))
    await asyncio.to_thread(fresh_init_started.wait, 1.0)

    initial_event.allow_wait_return()

    with pytest.raises(HTTPException) as exc_info:
        await old_waiter_task

    assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert exc_info.value.detail == deps._CHACHA_SHUTDOWN_INIT_ERROR_DETAIL

    release_fresh_init.set()
    fresh_db = await fresh_init_task
    assert isinstance(fresh_db, _FreshDB)
