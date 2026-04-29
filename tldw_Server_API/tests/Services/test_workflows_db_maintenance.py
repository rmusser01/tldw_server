import asyncio
from typing import Any

import pytest

from tldw_Server_API.app.core.DB_Management.backends.base import BackendType
from tldw_Server_API.app.services import workflows_db_maintenance as service


pytestmark = pytest.mark.unit


class _LoggerStub:
    def __init__(self) -> None:
        self.debugs: list[str] = []
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.binds: list[dict[str, Any]] = []

    def bind(self, **kwargs: Any):
        self.binds.append(kwargs)
        return self

    def debug(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.debugs.append(message.format(*args) if args else message)

    def info(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.infos.append(message.format(*args) if args else message)

    def warning(self, message: str, *args: Any, **kwargs: Any) -> None:
        self.warnings.append(message.format(*args) if args else message)


class _FakeSqliteConnection:
    def __init__(self, failures: dict[str, str]) -> None:
        self.failures = failures
        self.queries: list[str] = []

    def execute(self, query: str) -> None:
        self.queries.append(query)
        for prefix, error in self.failures.items():
            if query.startswith(prefix):
                raise RuntimeError(error)


class _FakeSqliteDB:
    backend = None

    def __init__(self, conn: _FakeSqliteConnection) -> None:
        self._conn = conn


async def _run_one_maintenance_iteration(monkeypatch: pytest.MonkeyPatch, db: Any, logger: _LoggerStub) -> None:
    stop_event = asyncio.Event()
    monkeypatch.setenv("WORKFLOWS_DB_MAINTENANCE_INTERVAL_SEC", "1")
    monkeypatch.setattr(service, "logger", logger)
    monkeypatch.setattr(service, "get_content_backend_instance", lambda: object())
    monkeypatch.setattr(service, "create_workflows_database", lambda backend: db)

    async def _fake_wait_for(awaitable: Any, timeout: float) -> None:
        if hasattr(awaitable, "close"):
            awaitable.close()
        stop_event.set()

    monkeypatch.setattr(service.asyncio, "wait_for", _fake_wait_for)

    await service.run_workflows_db_maintenance(stop_event)


@pytest.mark.asyncio
async def test_sqlite_wal_checkpoint_failure_log_is_sanitized(monkeypatch):
    logger = _LoggerStub()
    db = _FakeSqliteDB(
        _FakeSqliteConnection(
            {"PRAGMA wal_checkpoint": "wal checkpoint leaked /tmp/workflows-wal-secret sk-live-wal"}
        )
    )

    await _run_one_maintenance_iteration(monkeypatch, db, logger)

    assert logger.debugs == ["Workflows DB maintenance: WAL checkpoint skipped"]
    assert logger.binds[-1] == {"error_type": "RuntimeError"}
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/workflows-wal-secret" not in rendered
    assert "sk-live-wal" not in rendered
    assert "wal checkpoint leaked" not in rendered


@pytest.mark.asyncio
async def test_sqlite_optimize_failure_log_is_sanitized(monkeypatch):
    logger = _LoggerStub()
    db = _FakeSqliteDB(
        _FakeSqliteConnection(
            {"PRAGMA optimize": "optimize leaked /tmp/workflows-optimize-secret sk-live-optimize"}
        )
    )

    await _run_one_maintenance_iteration(monkeypatch, db, logger)

    assert logger.debugs == ["Workflows DB maintenance: PRAGMA optimize skipped"]
    assert logger.binds[-1] == {"error_type": "RuntimeError"}
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/workflows-optimize-secret" not in rendered
    assert "sk-live-optimize" not in rendered
    assert "optimize leaked" not in rendered


@pytest.mark.asyncio
async def test_sqlite_vacuum_failure_log_is_sanitized(monkeypatch):
    logger = _LoggerStub()
    db = _FakeSqliteDB(
        _FakeSqliteConnection({"VACUUM": "vacuum leaked /tmp/workflows-vacuum-secret sk-live-vacuum"})
    )
    monkeypatch.setenv("WORKFLOWS_SQLITE_VACUUM", "true")

    await _run_one_maintenance_iteration(monkeypatch, db, logger)

    assert logger.warnings == ["Workflows DB maintenance: SQLite VACUUM failed"]
    assert logger.binds[-1] == {"error_type": "RuntimeError"}
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/workflows-vacuum-secret" not in rendered
    assert "sk-live-vacuum" not in rendered
    assert "vacuum leaked" not in rendered


@pytest.mark.asyncio
async def test_sqlite_outer_failure_log_is_sanitized(monkeypatch):
    logger = _LoggerStub()
    db = _FakeSqliteDB(_FakeSqliteConnection({}))
    real_getenv = service.os.getenv

    def _fake_getenv(name: str, default: str = "") -> str:
        if name == "WORKFLOWS_DB_MAINTENANCE_INTERVAL_SEC":
            return "1"
        if name == "WORKFLOWS_SQLITE_CHECKPOINT":
            raise RuntimeError("sqlite outer leaked /tmp/workflows-sqlite-outer-secret sk-live-sqlite")
        return real_getenv(name, default)

    monkeypatch.setattr(service.os, "getenv", _fake_getenv)

    await _run_one_maintenance_iteration(monkeypatch, db, logger)

    assert logger.warnings == ["Workflows DB maintenance (SQLite) failed"]
    assert logger.binds[-1] == {"error_type": "RuntimeError"}
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/workflows-sqlite-outer-secret" not in rendered
    assert "sk-live-sqlite" not in rendered
    assert "sqlite outer leaked" not in rendered


@pytest.mark.asyncio
async def test_outer_loop_failure_log_is_sanitized(monkeypatch):
    class _FakeDB:
        backend = object()

        @property
        def backend_type(self):
            raise RuntimeError("outer loop leaked /tmp/workflows-loop-secret sk-live-loop")

    logger = _LoggerStub()

    await _run_one_maintenance_iteration(monkeypatch, _FakeDB(), logger)

    assert logger.warnings == ["Workflows DB maintenance loop error"]
    assert logger.binds[-1] == {"error_type": "RuntimeError"}
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/workflows-loop-secret" not in rendered
    assert "sk-live-loop" not in rendered
    assert "outer loop leaked" not in rendered


@pytest.mark.asyncio
async def test_postgres_vacuum_failure_log_is_sanitized(monkeypatch):
    class _FakeCursor:
        def execute(self, query: str) -> None:
            raise RuntimeError("postgres vacuum leaked /tmp/workflows-pg-secret sk-live-pg")

    class _FakeConnection:
        autocommit = False

        def cursor(self) -> _FakeCursor:
            return _FakeCursor()

    class _FakeBackend:
        def connect(self) -> _FakeConnection:
            return _FakeConnection()

        def disconnect(self, connection: _FakeConnection) -> None:
            pass

        def escape_identifier(self, table: str) -> str:
            return f'"{table}"'

    class _FakePostgresDB:
        backend = _FakeBackend()
        backend_type = BackendType.POSTGRESQL

    logger = _LoggerStub()
    monkeypatch.setenv("WORKFLOWS_POSTGRES_VACUUM", "true")

    await _run_one_maintenance_iteration(monkeypatch, _FakePostgresDB(), logger)

    assert logger.warnings == ["Workflows DB maintenance: Postgres VACUUM failed"]
    assert logger.binds[-1] == {"error_type": "RuntimeError"}
    rendered = "\n".join(logger.debugs + logger.infos + logger.warnings)
    assert "/tmp/workflows-pg-secret" not in rendered
    assert "sk-live-pg" not in rendered
    assert "postgres vacuum leaked" not in rendered
