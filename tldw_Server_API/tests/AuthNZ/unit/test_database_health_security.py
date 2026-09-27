from __future__ import annotations

import io
from types import SimpleNamespace

import pytest

from tldw_Server_API.app.core.AuthNZ import database


@pytest.mark.asyncio
async def test_database_health_check_sanitizes_error_response_and_log() -> None:
    marker = "postgresql://admin:secret@private-host/authnz"

    class _Acquire:
        async def __aenter__(self):
            raise RuntimeError(marker)

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            del exc_type, exc, tb
            return False

    class _Pool:
        def acquire(self) -> _Acquire:
            return _Acquire()

    db_pool = database.DatabasePool.__new__(database.DatabasePool)
    db_pool.pool = _Pool()
    output = io.StringIO()
    sink = database.logger.add(output, format="{message} {extra}")
    try:
        result = await db_pool.health_check()
    finally:
        database.logger.remove(sink)

    assert result == {
        "status": "unhealthy",
        "type": "postgresql",
        "error": "database_unavailable",
    }
    assert marker not in output.getvalue()
    assert "secret" not in output.getvalue()


@pytest.mark.asyncio
async def test_sqlite_health_check_sanitizes_driver_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    marker = "/private/authnz.sqlite?token=secret"

    def _failing_connect(*_args, **_kwargs):
        raise database.sqlite3.OperationalError(marker)

    db_pool = database.DatabasePool.__new__(database.DatabasePool)
    db_pool.pool = None
    db_pool.db_path = marker
    db_pool._sqlite_uri = False
    monkeypatch.setattr(database.aiosqlite, "connect", _failing_connect)
    output = io.StringIO()
    sink = database.logger.add(output, format="{message} {extra}")
    try:
        result = await db_pool.health_check()
    finally:
        database.logger.remove(sink)

    assert result == {
        "status": "unhealthy",
        "type": "sqlite",
        "error": "database_unavailable",
    }
    assert marker not in output.getvalue()
    assert "secret" not in output.getvalue()


@pytest.mark.asyncio
async def test_pool_reconfiguration_log_never_contains_database_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_url = "postgresql://old-user:old-secret@old-host/authnz"
    current_url = "postgresql://new-user:new-secret@new-host/authnz"

    class _ExistingPool:
        def __init__(self) -> None:
            self._initialized = True
            self.settings = SimpleNamespace(
                AUTH_MODE="multi_user",
                DATABASE_URL=previous_url,
            )

        async def close(self) -> None:
            return None

    class _ReplacementPool:
        def __init__(self, settings) -> None:
            self._initialized = False
            self.settings = settings

        async def initialize(self) -> None:
            self._initialized = True

    current_settings = SimpleNamespace(
        AUTH_MODE="multi_user",
        DATABASE_URL=current_url,
    )
    monkeypatch.setattr(database, "_db_pool", _ExistingPool())
    monkeypatch.setattr(database, "get_settings", lambda: current_settings)
    monkeypatch.setattr(database, "DatabasePool", _ReplacementPool)
    output = io.StringIO()
    sink = database.logger.add(output, format="{message} {extra}")
    try:
        result = await database.get_db_pool()
    finally:
        database.logger.remove(sink)

    assert isinstance(result, _ReplacementPool)
    assert previous_url not in output.getvalue()
    assert current_url not in output.getvalue()
    assert "old-secret" not in output.getvalue()
    assert "new-secret" not in output.getvalue()
