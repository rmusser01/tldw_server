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
@pytest.mark.asyncio
async def test_get_chacha_db_for_user_id_maps_runtime_unavailable_to_503(monkeypatch):
    import tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps as deps
    from tldw_Server_API.app.core.DB_Management.chacha.runtime import ChaChaRuntimeUnavailableError

    class _Runtime:
        async def get_or_create(self, *_args, **_kwargs):
            raise ChaChaRuntimeUnavailableError("ChaChaNotes shutdown in progress")

    monkeypatch.setattr(deps, "_CHACHA_RUNTIME", _Runtime())

    with pytest.raises(HTTPException) as exc_info:
        await deps.get_chacha_db_for_user_id(1, "1")

    assert exc_info.value.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert "shutdown" in exc_info.value.detail.lower()
