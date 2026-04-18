from __future__ import annotations

import importlib

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

    class _Runtime:
        async def get_or_create(self, *_args, **_kwargs):
            raise AssertionError("runtime should not be called for invalid ids")

        def schedule_default_character_ensure(self, *_args, **_kwargs):
            raise AssertionError("runtime should not be called for invalid ids")

        def is_shutting_down(self) -> bool:
            return False

    monkeypatch.setattr(deps, "_CHACHA_RUNTIME", _Runtime())

    with pytest.raises(HTTPException) as exc_info:
        await deps.get_chacha_db_for_user_id(user_id)

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert exc_info.value.detail == "Invalid owner_user_id."


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_chacha_db_for_user_id_maps_runtime_unavailable_to_503(monkeypatch):
    import tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps as deps
    from tldw_Server_API.app.core.DB_Management.chacha.runtime import ChaChaRuntimeUnavailableError

    class _Runtime:
        def __init__(self) -> None:
            self.scheduled = 0

        async def get_or_create(self, *_args, **_kwargs):
            raise ChaChaRuntimeUnavailableError("ChaChaNotes shutdown in progress")

        def schedule_default_character_ensure(self, *_args, **_kwargs):
            self.scheduled += 1

        def is_shutting_down(self) -> bool:
            return False

    monkeypatch.setattr(deps, "_CHACHA_RUNTIME", _Runtime())

    with pytest.raises(HTTPException) as exc_info:
        await deps.get_chacha_db_for_user_id(1, "1")

    assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "shutdown" in exc_info.value.detail.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_chacha_db_for_user_id_schedules_default_character_ensure_when_not_shutting_down(monkeypatch):
    import tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps as deps

    class _Runtime:
        def __init__(self) -> None:
            self.get_calls: list[tuple[int, str | None]] = []
            self.schedule_calls: list[tuple[object, int]] = []

        async def get_or_create(self, user_id, client_id):
            self.get_calls.append((user_id, client_id))
            return object()

        def schedule_default_character_ensure(self, db_instance, user_id):
            self.schedule_calls.append((db_instance, user_id))

        def is_shutting_down(self) -> bool:
            return False

    runtime = _Runtime()
    monkeypatch.setattr(deps, "_CHACHA_RUNTIME", runtime)

    db_instance = await deps.get_chacha_db_for_user_id(7, None)

    assert runtime.get_calls == [(7, "7")]
    assert runtime.schedule_calls == [(db_instance, 7)]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_chacha_db_for_user_does_not_call_warm_for_user_after_get_or_create(monkeypatch):
    import tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps as deps

    class _Runtime:
        def __init__(self) -> None:
            self.get_calls: list[tuple[int, str | None]] = []
            self.schedule_calls: list[tuple[object, int]] = []
            self.warm_calls: list[tuple[int, str | None]] = []

        async def get_or_create(self, user_id, client_id):
            self.get_calls.append((user_id, client_id))
            return object()

        def schedule_default_character_ensure(self, db_instance, user_id):
            self.schedule_calls.append((db_instance, user_id))

        async def warm_for_user(self, user_id, client_id=None):
            self.warm_calls.append((user_id, client_id))

        def is_shutting_down(self) -> bool:
            return False

    runtime = _Runtime()
    monkeypatch.setattr(deps, "_CHACHA_RUNTIME", runtime)

    class _User:
        id = 11

    await deps.get_chacha_db_for_user(_User())

    assert runtime.get_calls == [(11, "11")]
    assert len(runtime.schedule_calls) == 1
    assert runtime.warm_calls == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_chacha_db_for_user_id_maps_generic_init_failure_to_500(monkeypatch):
    import tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps as deps
    from tldw_Server_API.app.core.DB_Management.chacha.runtime import ChaChaRuntimeInitError

    class _Runtime:
        async def get_or_create(self, *_args, **_kwargs):
            raise ChaChaRuntimeInitError("Could not initialize character & notes database for user: boom")

        def schedule_default_character_ensure(self, *_args, **_kwargs):
            raise AssertionError("schedule should not be called on init failure")

        def is_shutting_down(self) -> bool:
            return False

    monkeypatch.setattr(deps, "_CHACHA_RUNTIME", _Runtime())

    with pytest.raises(HTTPException) as exc_info:
        await deps.get_chacha_db_for_user_id(1, "1")

    assert exc_info.value.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
    assert "boom" in exc_info.value.detail


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("owner_user_id", [True, False, 0, -1])
async def test_get_chacha_db_for_owner_rejects_bool_and_non_positive_ids(monkeypatch, owner_user_id):
    import tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps as deps

    class _Runtime:
        async def get_or_create(self, *_args, **_kwargs):
            raise AssertionError("runtime should not be called for invalid owner ids")

    monkeypatch.setattr(deps, "_CHACHA_RUNTIME", _Runtime())

    with pytest.raises(HTTPException) as exc_info:
        await deps.get_chacha_db_for_owner(owner_user_id)

    assert exc_info.value.status_code == status.HTTP_400_BAD_REQUEST
    assert exc_info.value.detail == "Invalid owner_user_id."
