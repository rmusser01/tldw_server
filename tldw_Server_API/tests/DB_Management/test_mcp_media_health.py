"""MCP health probes against real SQLite and official temporary PostgreSQL DBs."""

import asyncio

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType, DatabaseConfig
from tldw_Server_API.app.core.DB_Management.backends.factory import DatabaseBackendFactory
from tldw_Server_API.app.core.DB_Management.media_db.errors import DatabaseError
from tldw_Server_API.app.core.DB_Management.media_db.native_class import MediaDatabase
from tldw_Server_API.app.core.MCP_unified.modules.base import ModuleConfig
from tldw_Server_API.app.core.MCP_unified.modules.implementations import media_module


@pytest.fixture(params=["sqlite", "postgresql"])
def health_module(request, tmp_path, monkeypatch):
    """Keep the real DB adapter/transactions; isolate only the disk-space signal."""
    db_path = str(tmp_path / "media.db")
    config = (
        request.getfixturevalue("pg_database_config")
        if request.param == "postgresql"
        else DatabaseConfig(backend_type=BackendType.SQLITE, sqlite_path=db_path)
    )
    backend = DatabaseBackendFactory.create_backend(config)
    db = MediaDatabase(db_path=db_path, client_id="mcp-health-test", backend=backend)
    module = media_module.MediaModule(ModuleConfig(name="media", settings={"db_path": db_path}))
    module.db = db
    monkeypatch.setattr(media_module, "get_free_disk_space_gb", lambda _path: 5)
    try:
        yield module
    finally:
        db.close_connection()
        backend.get_pool().close_all()


def test_health_repeated_write_probe_succeeds_without_leaving_rows(health_module):
    for _ in range(2):
        assert asyncio.run(health_module.check_health()) == {
            "database_connection": True,
            "database_writable": True,
            "disk_space": True,
            "service_available": True,
        }
        assert health_module.db.execute_query("SELECT k FROM _mcp_healthcheck").fetchall() == []


def test_health_does_not_replace_or_delete_another_probe_row(health_module):
    db = health_module.db
    db.execute_query("CREATE TABLE _mcp_healthcheck (k TEXT PRIMARY KEY, v TEXT)")
    db.execute_query("INSERT INTO _mcp_healthcheck(k, v) VALUES (?, ?)", ("ping", "other probe"))
    assert asyncio.run(health_module.check_health())["database_writable"] is True
    assert dict(db.execute_query("SELECT k, v FROM _mcp_healthcheck").fetchone()) == {
        "k": "ping", "v": "other probe",
    }


def test_health_reports_real_insert_failure_and_releases_transaction(health_module):
    db = health_module.db
    db.execute_query("CREATE TABLE _mcp_healthcheck (k TEXT PRIMARY KEY, v TEXT CHECK (v IS NULL))")
    checks = asyncio.run(health_module.check_health())
    assert checks["database_connection"] is True
    assert checks["database_writable"] is False
    assert db.execute_query("SELECT k FROM _mcp_healthcheck").fetchall() == []
    # The failed transaction must not poison a subsequent database write.
    db.execute_query("INSERT INTO _mcp_healthcheck(k, v) VALUES (?, ?)", ("recovered", None))
    assert db.execute_query("SELECT k FROM _mcp_healthcheck").fetchone()["k"] == "recovered"


def test_health_cleanup_failure_rolls_back_probe_and_is_not_healthy(health_module, monkeypatch):
    db = health_module.db
    db.execute_query("CREATE TABLE _mcp_healthcheck (k TEXT PRIMARY KEY, v TEXT)")
    execute = db.execute_query

    def fail_cleanup(query, *args, **kwargs):
        if query.startswith("DELETE FROM _mcp_healthcheck"):
            raise DatabaseError("Cleanup unavailable")
        return execute(query, *args, **kwargs)

    monkeypatch.setattr(db, "execute_query", fail_cleanup)
    assert asyncio.run(health_module.check_health())["database_writable"] is False
    assert execute("SELECT k FROM _mcp_healthcheck").fetchall() == []


def test_health_reports_real_read_failure_without_escaping(health_module, monkeypatch):
    db = health_module.db
    execute = db.execute_query

    def fail_read(query, *args, **kwargs):
        if query == "SELECT 1":
            return execute("SELECT k FROM missing_mcp_health_read_table")
        return execute(query, *args, **kwargs)

    monkeypatch.setattr(db, "execute_query", fail_read)
    checks = asyncio.run(health_module.check_health())
    assert checks["database_connection"] is False
    assert checks["database_writable"] is True
