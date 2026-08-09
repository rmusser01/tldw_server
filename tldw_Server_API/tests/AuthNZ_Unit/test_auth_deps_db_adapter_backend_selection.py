from __future__ import annotations

import asyncio
import io
import traceback
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import aiosqlite
import pytest
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.core.AuthNZ.database import (
    DatabasePool,
    _GuardedSQLiteConnection,
)
from tldw_Server_API.app.core.AuthNZ.exceptions import RollbackSignal, TransactionError
from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import (
    ProfileUserWriteRejected,
    _guard_sql,
    _mint_profile_user_sql,
    _revoke_profile_user_sql,
)
from tldw_Server_API.app.core.AuthNZ.profile_version import normalize_profile_version
from tldw_Server_API.app.services import auth_service

_RAW_RELEASE_TEXT = "adapter release secret=/tmp/authnz-users.db token=release-secret"


class _AcquireCM:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:
        return False


class _PoolStub:
    def __init__(self, conn: Any, *, pool_marker: Any) -> None:
        self._conn = conn
        self.pool = pool_marker

    def acquire(self) -> _AcquireCM:
        return _AcquireCM(self._conn)


class _ReleaseFailingBackend:
    def __init__(self) -> None:
        class _Connection:
            def transaction(self) -> _AcquireCM:
                return _AcquireCM(self)

        self.connection = _Connection()
        self.release_calls = 0

    async def acquire(self, *, timeout: float | None = None) -> Any:  # noqa: ARG002
        return self.connection

    async def release(
        self,
        connection: Any,
        *,
        timeout: float | None = None,  # noqa: ARG002
    ) -> None:
        assert connection is self.connection
        self.release_calls += 1
        raise RuntimeError(_RAW_RELEASE_TEXT)


class _RecordingAcquireContext:
    def __init__(self, context: Any) -> None:
        self._context = context
        self.exit_args: tuple[Any, Any, Any] | None = None
        self.received_exact_traceback = False

    async def __aenter__(self) -> Any:
        return await self._context.__aenter__()

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        self.exit_args = (exc_type, exc, tb)
        self.received_exact_traceback = exc is None or tb is exc.__traceback__
        return await self._context.__aexit__(exc_type, exc, tb)


class _RecordingAdapterPool:
    def __init__(self) -> None:
        backend = _ReleaseFailingBackend()
        database_pool = DatabasePool.__new__(DatabasePool)
        database_pool._initialized = True
        database_pool.pool = backend
        database_pool.db_path = ":memory:"
        database_pool._sqlite_uri = False
        self.pool = backend
        self.backend = backend
        self.context = _RecordingAcquireContext(database_pool.acquire())

    def acquire(self) -> _RecordingAcquireContext:
        return self.context


async def _adapter_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, _RecordingAdapterPool]:
    pool = _RecordingAdapterPool()

    async def _fake_get_db_pool() -> _RecordingAdapterPool:
        return pool

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setattr(auth_deps, "get_db_pool", _fake_get_db_pool)
    generator = auth_deps.get_db_transaction()
    await generator.__anext__()
    return generator, pool


class _PostgresConnWithoutFetchrowProbe:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.commit_calls = 0
        self.transaction_enter_calls = 0
        self.transaction_exit_args: list[tuple[Any, Any, Any]] = []

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - failure guard only
        if name == "fetchrow":
            raise AssertionError("adapter should not inspect connection capability via fetchrow")
        raise AttributeError(name)

    async def execute(self, query: str, *args: Any) -> str:
        self.execute_calls.append((str(query), tuple(args)))
        return "OK"

    async def commit(self) -> None:
        self.commit_calls += 1

    def transaction(self) -> _AcquireCM:
        connection = self

        class _TransactionContext(_AcquireCM):
            async def __aenter__(self) -> Any:
                connection.transaction_enter_calls += 1
                return await super().__aenter__()

            async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
                connection.transaction_exit_args.append((exc_type, exc, tb))
                return False

        return _TransactionContext(self)


class _SqliteConnWithFetchrowCapability:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, Any]] = []
        self.commit_calls = 0
        self.rollback_calls = 0

    async def fetchrow(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - sqlite path should ignore
        raise AssertionError("sqlite adapter path should not call fetchrow")

    async def execute(self, query: str, params: Any) -> Any:
        self.execute_calls.append((str(query), params))
        return SimpleNamespace()

    async def commit(self) -> None:
        self.commit_calls += 1

    async def rollback(self) -> None:
        self.rollback_calls += 1


@pytest.mark.asyncio
async def test_get_db_transaction_adapter_uses_pool_backend_for_postgres(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_get_db_pool() -> _PoolStub:
        return _PoolStub(_PostgresConnWithoutFetchrowProbe(), pool_marker=object())

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setattr(auth_deps, "get_db_pool", _fake_get_db_pool)

    agen = auth_deps.get_db_transaction()
    adapter = await agen.__anext__()
    try:
        assert not hasattr(adapter, "transaction")
        assert not hasattr(adapter, "create_savepoint")
        assert not hasattr(adapter, "rollback_savepoint")
        assert not hasattr(adapter, "release_savepoint")
        await adapter.execute("SELECT $1", 1)
        conn = adapter._conn  # noqa: SLF001 - test verifies adapter behavior
        assert conn.execute_calls == [("SELECT $1", (1,))]
        assert conn.commit_calls == 0
    finally:
        await agen.aclose()

    assert conn.transaction_enter_calls == 1
    assert len(conn.transaction_exit_args) == 1


@pytest.mark.asyncio
async def test_postgres_test_adapter_rolls_back_one_request_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _PostgresConnWithoutFetchrowProbe()

    async def _fake_get_db_pool() -> _PoolStub:
        return _PoolStub(connection, pool_marker=object())

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setattr(auth_deps, "get_db_pool", _fake_get_db_pool)

    generator = auth_deps.get_db_transaction()
    await generator.__anext__()
    failure = RuntimeError("request failed")
    with pytest.raises(RuntimeError, match="request failed"):
        await generator.athrow(failure)

    assert connection.transaction_enter_calls == 1
    assert len(connection.transaction_exit_args) == 1
    assert connection.transaction_exit_args[0][0] is RuntimeError
    assert connection.transaction_exit_args[0][1] is failure


@pytest.mark.asyncio
async def test_get_db_transaction_adapter_uses_pool_backend_for_sqlite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _SqliteConnWithFetchrowCapability()

    async def _fake_get_db_pool() -> _PoolStub:
        return _PoolStub(connection, pool_marker=None)

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setattr(auth_deps, "get_db_pool", _fake_get_db_pool)

    agen = auth_deps.get_db_transaction()
    adapter = await agen.__anext__()
    await adapter.execute("SELECT $1", 1)
    await adapter.commit()

    assert connection.execute_calls and connection.execute_calls[0][0] == "SELECT ?"
    assert connection.commit_calls == 0
    assert connection.rollback_calls == 0

    with pytest.raises(StopAsyncIteration):
        await agen.__anext__()

    assert connection.commit_calls == 1
    assert connection.rollback_calls == 0


@pytest.mark.asyncio
async def test_test_adapter_releases_connection_when_guard_marker_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _InvalidMarkerConnection:
        _authnz_profile_user_backend = "invalid"

    class _RecordingContext(_AcquireCM):
        def __init__(self) -> None:
            super().__init__(_InvalidMarkerConnection())
            self.exit_calls: list[tuple[Any, Any, Any]] = []

        async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
            self.exit_calls.append((exc_type, exc, tb))
            return False

    context = _RecordingContext()

    class _Pool:
        pool = None

        def acquire(self) -> _RecordingContext:
            return context

    async def _fake_get_db_pool() -> _Pool:
        return _Pool()

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setattr(auth_deps, "get_db_pool", _fake_get_db_pool)

    generator = auth_deps.get_db_transaction()
    with pytest.raises(ProfileUserWriteRejected) as raised:
        await generator.__anext__()

    assert len(context.exit_calls) == 1
    assert context.exit_calls[0][0] is type(raised.value)
    assert context.exit_calls[0][1] is raised.value
    assert context.exit_calls[0][2] is not None


@pytest.mark.asyncio
async def test_sqlite_test_adapter_preserves_capability_and_guard_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = object()

    class _GuardedConnection:
        _authnz_profile_user_guard_identity = identity

        def __init__(self) -> None:
            self.calls: list[tuple[str, Any]] = []

        async def execute(self, query: Any, parameters: Any) -> Any:
            guarded = _guard_sql(
                query,
                backend="sqlite",
                connection_identity=identity,
                operation="execute",
            )
            self.calls.append((guarded, parameters))
            return SimpleNamespace()

        async def commit(self) -> None:
            return None

    connection = _GuardedConnection()

    async def _fake_get_db_pool() -> _PoolStub:
        return _PoolStub(connection, pool_marker=None)

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setattr(auth_deps, "get_db_pool", _fake_get_db_pool)
    capability = _mint_profile_user_sql(
        "UPDATE users SET email = ? WHERE id = ?",
        backend="sqlite",
        connection_identity=identity,
        operation="update",
        columns=("email",),
    )

    generator = auth_deps.get_db_transaction()
    adapter = await generator.__anext__()
    try:
        assert adapter._authnz_profile_user_guard_identity is identity
        await adapter.execute(capability, "adapter@example.com", 7)
    finally:
        _revoke_profile_user_sql(capability)

    with pytest.raises(StopAsyncIteration):
        await generator.__anext__()

    assert connection.calls == [
        (
            "UPDATE users SET email = ? WHERE id = ?",
            ("adapter@example.com", 7),
        )
    ]


@pytest.mark.asyncio
async def test_sqlite_test_adapter_runs_versioned_user_gateway_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_connection = await aiosqlite.connect(":memory:")
    initial_version = "2026-01-01T00:00:00.000000Z"
    try:
        await raw_connection.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                last_login TEXT,
                profile_version TEXT NOT NULL
            );
            CREATE TABLE org_members (
                user_id INTEGER NOT NULL,
                org_id INTEGER NOT NULL,
                status TEXT
            );
            CREATE TABLE team_members (
                user_id INTEGER NOT NULL,
                team_id INTEGER NOT NULL,
                status TEXT
            );
            CREATE TABLE user_config_overrides (
                user_id INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE org_config_overrides (
                org_id INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE team_config_overrides (
                team_id INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT INTO users (id, profile_version)
            VALUES (1, '2026-01-01T00:00:00.000000Z');
            """
        )
        await raw_connection.commit()
        guarded_connection = _GuardedSQLiteConnection(raw_connection)

        async def _fake_get_db_pool() -> _PoolStub:
            return _PoolStub(guarded_connection, pool_marker=None)

        monkeypatch.setenv("TEST_MODE", "1")
        monkeypatch.setattr(auth_deps, "get_db_pool", _fake_get_db_pool)
        generator = auth_deps.get_db_transaction()
        adapter = await generator.__anext__()
        assert adapter._authnz_profile_user_backend == "sqlite"
        await auth_service.update_user_last_login(
            adapter,
            1,
            datetime(2026, 8, 1, 12, 30, tzinfo=timezone.utc),
        )
        with pytest.raises(StopAsyncIteration):
            await generator.__anext__()

        cursor = await raw_connection.execute(
            "SELECT last_login, profile_version FROM users WHERE id = 1"
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row[0] is not None
        assert normalize_profile_version(row[1]) > normalize_profile_version(
            initial_version
        )
    finally:
        await raw_connection.close()


@pytest.mark.asyncio
async def test_sqlite_test_adapter_rolls_back_update_when_version_touch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_connection = await aiosqlite.connect(":memory:")

    class _FailingVersionTouchConnection:
        def __init__(self) -> None:
            self.commit_calls = 0
            self.rollback_calls = 0

        async def execute(self, query: str, *args: Any) -> Any:
            normalized = " ".join(query.lower().split())
            if normalized.startswith("update main.users set profile_version"):
                raise RuntimeError("injected profile-version touch failure")
            return await raw_connection.execute(query, *args)

        async def commit(self) -> None:
            self.commit_calls += 1
            await raw_connection.commit()

        async def rollback(self) -> None:
            self.rollback_calls += 1
            await raw_connection.rollback()

    try:
        await raw_connection.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                last_login TEXT,
                profile_version TEXT NOT NULL
            );
            CREATE TABLE org_members (
                user_id INTEGER NOT NULL,
                org_id INTEGER NOT NULL,
                status TEXT
            );
            CREATE TABLE team_members (
                user_id INTEGER NOT NULL,
                team_id INTEGER NOT NULL,
                status TEXT
            );
            CREATE TABLE user_config_overrides (
                user_id INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE org_config_overrides (
                org_id INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE team_config_overrides (
                team_id INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            );
            INSERT INTO users (id, profile_version)
            VALUES (1, '2026-01-01T00:00:00.000000Z');
            """
        )
        await raw_connection.commit()
        failing_connection = _FailingVersionTouchConnection()
        guarded_connection = _GuardedSQLiteConnection(failing_connection)

        async def _fake_get_db_pool() -> _PoolStub:
            return _PoolStub(guarded_connection, pool_marker=None)

        monkeypatch.setenv("TEST_MODE", "1")
        monkeypatch.setattr(auth_deps, "get_db_pool", _fake_get_db_pool)
        generator = auth_deps.get_db_transaction()
        adapter = await generator.__anext__()

        with pytest.raises(RuntimeError) as raised:
            await auth_service.update_user_last_login(
                adapter,
                1,
                datetime(2026, 8, 1, 12, 30, tzinfo=timezone.utc),
            )

        with pytest.raises(RuntimeError) as propagated:
            await generator.athrow(raised.value)

        assert propagated.value is raised.value
        cursor = await raw_connection.execute(
            "SELECT last_login, profile_version FROM users WHERE id = 1"
        )
        row = await cursor.fetchone()
        assert row == (None, "2026-01-01T00:00:00.000000Z")
        assert failing_connection.commit_calls == 0
        assert failing_connection.rollback_calls == 1
    finally:
        await raw_connection.close()


@pytest.mark.asyncio
async def test_test_adapter_cancellation_wins_over_release_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator, pool = await _adapter_generator(monkeypatch)
    cancellation = asyncio.CancelledError("adapter-cancelled")
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")
    try:
        with pytest.raises(asyncio.CancelledError) as raised:
            await generator.athrow(cancellation)
    finally:
        logger.remove(sink_id)

    assert raised.value is cancellation
    assert pool.context.exit_args is not None
    assert pool.context.exit_args[:2] == (asyncio.CancelledError, cancellation)
    assert pool.context.received_exact_traceback is True
    assert pool.backend.release_calls == 1
    assert sink.getvalue() == ""


@pytest.mark.asyncio
async def test_test_adapter_rollback_signal_wins_over_release_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator, pool = await _adapter_generator(monkeypatch)
    rollback = RollbackSignal("adapter-rollback")
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")
    try:
        with pytest.raises(RollbackSignal) as raised:
            await generator.athrow(rollback)
    finally:
        logger.remove(sink_id)

    assert raised.value is rollback
    assert pool.context.exit_args is not None
    assert pool.context.exit_args[:2] == (RollbackSignal, rollback)
    assert pool.context.received_exact_traceback is True
    assert pool.backend.release_calls == 1
    assert sink.getvalue() == ""


@pytest.mark.asyncio
async def test_test_adapter_ordinary_body_failure_wins_over_release_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator, pool = await _adapter_generator(monkeypatch)
    body_failure = ValueError("adapter-body-failure")
    sink = io.StringIO()
    sink_id = logger.add(sink, format="{message} {extra}")
    try:
        with pytest.raises(ValueError) as raised:
            await generator.athrow(body_failure)
    finally:
        logger.remove(sink_id)

    logs = sink.getvalue()
    assert raised.value is body_failure
    assert pool.context.exit_args is not None
    assert pool.context.exit_args[:2] == (ValueError, body_failure)
    assert pool.context.received_exact_traceback is True
    assert pool.backend.release_calls == 1
    assert logs.count("PostgreSQL connection release failed") == 1
    assert _RAW_RELEASE_TEXT not in logs


@pytest.mark.asyncio
async def test_test_adapter_no_primary_release_failure_is_sanitized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator, pool = await _adapter_generator(monkeypatch)
    sink = io.StringIO()
    sink_id = logger.add(sink, format="{message} {extra}")
    try:
        with pytest.raises(TransactionError) as raised:
            await generator.asend(None)
    finally:
        logger.remove(sink_id)

    rendered = "".join(traceback.format_exception(raised.value))
    logs = sink.getvalue()
    assert pool.context.exit_args == (None, None, None)
    assert pool.backend.release_calls == 1
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True
    assert _RAW_RELEASE_TEXT not in str(raised.value)
    assert _RAW_RELEASE_TEXT not in rendered
    assert _RAW_RELEASE_TEXT not in logs
    assert logs.count("PostgreSQL connection release failed") == 1
