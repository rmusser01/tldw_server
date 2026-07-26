from __future__ import annotations

import asyncio
import io
import traceback
from types import SimpleNamespace
from typing import Any

import pytest
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps import auth_deps
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.exceptions import RollbackSignal, TransactionError

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
        self.connection = object()
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

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - failure guard only
        if name == "fetchrow":
            raise AssertionError("adapter should not inspect connection capability via fetchrow")
        raise AttributeError(name)

    async def execute(self, query: str, *args: Any) -> str:
        self.execute_calls.append((str(query), tuple(args)))
        return "OK"

    async def commit(self) -> None:
        self.commit_calls += 1


class _SqliteConnWithFetchrowCapability:
    def __init__(self) -> None:
        self.execute_calls: list[tuple[str, Any]] = []
        self.commit_calls = 0

    async def fetchrow(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover - sqlite path should ignore
        raise AssertionError("sqlite adapter path should not call fetchrow")

    async def execute(self, query: str, params: Any) -> Any:
        self.execute_calls.append((str(query), params))
        return SimpleNamespace()

    async def commit(self) -> None:
        self.commit_calls += 1


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
        await adapter.execute("SELECT $1", 1)
        conn = adapter._conn  # noqa: SLF001 - test verifies adapter behavior
        assert conn.execute_calls == [("SELECT $1", (1,))]
        assert conn.commit_calls == 0
    finally:
        await agen.aclose()


@pytest.mark.asyncio
async def test_get_db_transaction_adapter_uses_pool_backend_for_sqlite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _fake_get_db_pool() -> _PoolStub:
        return _PoolStub(_SqliteConnWithFetchrowCapability(), pool_marker=None)

    monkeypatch.setenv("TEST_MODE", "1")
    monkeypatch.setattr(auth_deps, "get_db_pool", _fake_get_db_pool)

    agen = auth_deps.get_db_transaction()
    adapter = await agen.__anext__()
    try:
        await adapter.execute("SELECT $1", 1)
        conn = adapter._conn  # noqa: SLF001 - test verifies adapter behavior
        assert conn.execute_calls and conn.execute_calls[0][0] == "SELECT ?"
        assert conn.commit_calls == 1
    finally:
        await agen.aclose()


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
