from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest


class _FakeConnection:
    async def fetchval(self, query: str) -> str:
        assert query == "SELECT version()"
        return "PostgreSQL test"

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any]:
        assert query == "SELECT $1 AS value"
        return {"value": args[0]}


class _AcquireContext:
    def __init__(self, pool: _FakeAsyncpgPool, timeout: float | None) -> None:
        self._pool = pool
        self._timeout = timeout
        self._connection: Any = None

    def __await__(self):
        return self._acquire().__await__()

    async def _acquire(self):
        self._pool.acquire_timeouts.append(self._timeout)
        self._connection = self._pool.connection
        return self._connection

    async def __aenter__(self):
        return await self._acquire()

    async def __aexit__(self, _exc_type, _exc, _traceback) -> None:
        await self._pool.release(self._connection)


class _FakeAsyncpgPool:
    def __init__(self, name: str, close_order: list[str]) -> None:
        self.name = name
        self.connection = _FakeConnection() if name == "main" else object()
        self.close_order = close_order
        self.acquire_timeouts: list[float | None] = []
        self.release_calls: list[tuple[Any, float | None]] = []

    def acquire(self, *, timeout: float | None = None) -> _AcquireContext:
        return _AcquireContext(self, timeout)

    async def release(self, connection: Any, *, timeout: float | None = None) -> None:
        self.release_calls.append((connection, timeout))

    async def close(self) -> None:
        self.close_order.append(self.name)


class _GatedAcquireContext(_AcquireContext):
    async def _acquire(self):
        self._pool.acquire_timeouts.append(self._timeout)
        self._pool.acquire_started.set()
        await self._pool.allow_acquire_return.wait()
        self._connection = self._pool.connection
        return self._connection


class _GatedClosePool(_FakeAsyncpgPool):
    def __init__(self, name: str, close_order: list[str]) -> None:
        super().__init__(name, close_order)
        self.acquire_started = asyncio.Event()
        self.allow_acquire_return = asyncio.Event()
        self.close_started = asyncio.Event()
        self.allow_close_return = asyncio.Event()

    def acquire(self, *, timeout: float | None = None) -> _GatedAcquireContext:
        return _GatedAcquireContext(self, timeout)

    async def close(self) -> None:
        self.close_started.set()
        await self.allow_close_return.wait()
        await super().close()


def _postgres_settings() -> SimpleNamespace:
    return SimpleNamespace(
        AUTH_MODE="multi_user",
        DATABASE_URL="postgresql://test:test@localhost/test",
        DATABASE_POOL_MIN_SIZE=2,
        DATABASE_POOL_MAX_SIZE=12,
        DATABASE_MAX_QUERIES=321,
        DATABASE_MAX_INACTIVE_CONNECTION_LIFETIME=45,
    )


@pytest.mark.asyncio
async def test_postgres_initializes_and_closes_bounded_dedicated_lock_pool(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import database

    close_order: list[str] = []
    main_pool = _FakeAsyncpgPool("main", close_order)
    lock_pool = _FakeAsyncpgPool("lock", close_order)
    pools = iter((main_pool, lock_pool))
    create_calls: list[dict[str, Any]] = []

    async def _create_pool(*args, **kwargs):
        create_calls.append({"args": args, **kwargs})
        return next(pools)

    async def _create_schema(self) -> None:
        return None

    monkeypatch.setattr(database.asyncpg, "create_pool", _create_pool)
    monkeypatch.setattr(database.DatabasePool, "_create_postgresql_schema", _create_schema)

    pool = database.DatabasePool(_postgres_settings())
    await pool.initialize()

    assert len(create_calls) == 2
    assert create_calls[1]["args"] == (_postgres_settings().DATABASE_URL,)
    assert create_calls[1]["min_size"] == 0
    assert create_calls[1]["max_size"] == database.OPENAI_CREDENTIAL_LOCK_POOL_MAX_SIZE == 4
    assert create_calls[1]["max_queries"] == 321
    assert create_calls[1]["max_inactive_connection_lifetime"] == 45
    assert create_calls[1]["command_timeout"] == 60

    async with pool.acquire_openai_credential_lock_connection(timeout=1.5) as connection:
        assert connection is lock_pool.connection

    assert lock_pool.acquire_timeouts == [1.5]
    assert lock_pool.release_calls == [(lock_pool.connection, 1.5)]

    await pool.close()

    assert close_order == ["lock", "main"]
    assert pool._openai_credential_lock_pool is None


@pytest.mark.asyncio
async def test_dedicated_lock_pool_creation_failure_closes_main_pool(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import database

    close_order: list[str] = []
    main_pool = _FakeAsyncpgPool("main", close_order)
    create_count = 0

    async def _create_pool(*_args, **_kwargs):
        nonlocal create_count
        create_count += 1
        if create_count == 1:
            return main_pool
        raise RuntimeError("lock pool unavailable")

    async def _create_schema(self) -> None:
        return None

    monkeypatch.setattr(database.asyncpg, "create_pool", _create_pool)
    monkeypatch.setattr(database.DatabasePool, "_create_postgresql_schema", _create_schema)

    pool = database.DatabasePool(_postgres_settings())

    with pytest.raises(database.DatabaseError):
        await pool.initialize()

    assert create_count == 2
    assert close_order == ["main"]
    assert pool.pool is None
    assert pool._openai_credential_lock_pool is None
    assert not pool._initialized


@pytest.mark.asyncio
async def test_postgres_schema_failure_closes_both_pools_in_lock_first_order(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import database

    close_order: list[str] = []
    pools = iter(
        (
            _FakeAsyncpgPool("main", close_order),
            _FakeAsyncpgPool("lock", close_order),
        )
    )

    async def _create_pool(*_args, **_kwargs):
        return next(pools)

    async def _create_schema(self) -> None:
        raise RuntimeError("schema failed")

    monkeypatch.setattr(database.asyncpg, "create_pool", _create_pool)
    monkeypatch.setattr(database.DatabasePool, "_create_postgresql_schema", _create_schema)

    pool = database.DatabasePool(_postgres_settings())

    with pytest.raises(database.DatabaseError):
        await pool.initialize()

    assert close_order == ["lock", "main"]
    assert pool.pool is None
    assert pool._openai_credential_lock_pool is None
    assert not pool._initialized


@pytest.mark.asyncio
async def test_dedicated_lock_acquisition_fails_closed_when_pool_is_unavailable():
    from tldw_Server_API.app.core.AuthNZ import database

    pool = database.DatabasePool(_postgres_settings())
    pool._initialized = True
    pool.pool = _FakeAsyncpgPool("main", [])

    with pytest.raises(database.DatabaseError, match="credential lock pool unavailable"):
        async with pool.acquire_openai_credential_lock_connection():
            raise AssertionError("unavailable lock pool entered protected body")


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_close_keeps_main_pool_attached_until_checked_out_connection_returns():
    from tldw_Server_API.app.core.AuthNZ import database

    close_order: list[str] = []
    main_pool = _GatedClosePool("main", close_order)
    main_pool.allow_acquire_return.set()
    manager = database.DatabasePool(_postgres_settings())
    manager.pool = main_pool
    manager._initialized = True
    close_task: asyncio.Task[None] | None = None

    try:
        async with manager.acquire() as connection:
            assert connection is main_pool.connection
            close_task = asyncio.create_task(manager.close())
            await asyncio.wait_for(main_pool.close_started.wait(), timeout=1)

            assert manager.pool is main_pool
            assert not close_task.done()

        assert main_pool.release_calls == [(main_pool.connection, None)]
        main_pool.allow_close_return.set()
        await asyncio.wait_for(close_task, timeout=1)

        assert manager.pool is None
        assert close_order == ["main"]
    finally:
        main_pool.allow_close_return.set()
        if close_task is not None and not close_task.done():
            await asyncio.gather(close_task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_cancelled_close_drains_both_pools_before_detaching_them():
    from tldw_Server_API.app.core.AuthNZ import database

    close_order: list[str] = []
    main_pool = _GatedClosePool("main", close_order)
    lock_pool = _GatedClosePool("lock", close_order)
    manager = database.DatabasePool(_postgres_settings())
    manager.pool = main_pool
    manager._openai_credential_lock_pool = lock_pool
    manager._initialized = True
    close_task = asyncio.create_task(manager.close())

    try:
        await asyncio.wait_for(lock_pool.close_started.wait(), timeout=1)
        close_task.cancel()
        await asyncio.sleep(0)

        assert not close_task.done()
        assert manager._openai_credential_lock_pool is lock_pool
        assert manager.pool is main_pool
        assert manager._initialized

        lock_pool.allow_close_return.set()
        await asyncio.wait_for(main_pool.close_started.wait(), timeout=1)

        assert manager._openai_credential_lock_pool is None
        assert manager.pool is main_pool
        assert manager._initialized
        assert close_order == ["lock"]

        main_pool.allow_close_return.set()
        with pytest.raises(asyncio.CancelledError):
            await close_task

        assert manager._openai_credential_lock_pool is None
        assert manager.pool is None
        assert not manager._initialized
        assert close_order == ["lock", "main"]
    finally:
        lock_pool.allow_close_return.set()
        main_pool.allow_close_return.set()
        if not close_task.done():
            await asyncio.gather(close_task, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_reset_waits_for_inflight_postgres_query_before_detaching_pool(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import database

    close_order: list[str] = []
    main_pool = _GatedClosePool("main", close_order)
    manager = database.DatabasePool(_postgres_settings())
    manager.pool = main_pool
    manager._initialized = True
    monkeypatch.setattr(database, "_db_pool", manager)
    query_task = asyncio.create_task(manager.fetchone("SELECT ? AS value", 7))
    reset_task: asyncio.Task[None] | None = None

    try:
        await asyncio.wait_for(main_pool.acquire_started.wait(), timeout=1)
        reset_task = asyncio.create_task(database.reset_db_pool())
        await asyncio.wait_for(main_pool.close_started.wait(), timeout=1)

        assert database._db_pool is manager
        assert manager.pool is main_pool

        main_pool.allow_acquire_return.set()
        assert await asyncio.wait_for(query_task, timeout=1) == {"value": 7}
        assert main_pool.release_calls == [(main_pool.connection, None)]

        main_pool.allow_close_return.set()
        await asyncio.wait_for(reset_task, timeout=5)

        assert database._db_pool is None
        assert manager.pool is None
        assert close_order == ["main"]
    finally:
        main_pool.allow_acquire_return.set()
        main_pool.allow_close_return.set()
        await asyncio.gather(query_task, return_exceptions=True)
        if reset_task is not None and not reset_task.done():
            await asyncio.gather(reset_task, return_exceptions=True)
        database._db_pool = None


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_cancelled_reset_preserves_inflight_postgres_helper_branch(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import database

    close_order: list[str] = []
    main_pool = _GatedClosePool("main", close_order)
    manager = database.DatabasePool(_postgres_settings())
    manager.pool = main_pool
    manager._initialized = True
    monkeypatch.setattr(database, "_db_pool", manager)
    query_task = asyncio.create_task(manager.fetchone("SELECT ? AS value", 11))
    reset_task: asyncio.Task[None] | None = None

    try:
        await asyncio.wait_for(main_pool.acquire_started.wait(), timeout=1)
        reset_task = asyncio.create_task(database.reset_db_pool())
        await asyncio.wait_for(main_pool.close_started.wait(), timeout=1)
        reset_task.cancel()
        await asyncio.sleep(0)

        main_pool.allow_acquire_return.set()
        assert await asyncio.wait_for(query_task, timeout=1) == {"value": 11}
        assert main_pool.release_calls == [(main_pool.connection, None)]
        assert database._db_pool is manager
        assert manager.pool is main_pool
        assert manager._initialized

        main_pool.allow_close_return.set()
        with pytest.raises(asyncio.CancelledError):
            await reset_task

        assert database._db_pool is None
        assert manager.pool is None
        assert not manager._initialized
        assert close_order == ["main"]
    finally:
        main_pool.allow_acquire_return.set()
        main_pool.allow_close_return.set()
        await asyncio.gather(query_task, return_exceptions=True)
        if reset_task is not None and not reset_task.done():
            await asyncio.gather(reset_task, return_exceptions=True)
        database._db_pool = None
