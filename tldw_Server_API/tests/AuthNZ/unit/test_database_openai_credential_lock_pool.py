from __future__ import annotations

import asyncio
import io
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
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
async def test_fresh_postgres_initialize_executes_trusted_schema_as_guarded_statements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_Server_API.app.core.AuthNZ import database
    from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import _guard_sql

    executed: list[str] = []
    readiness_connections: list[object] = []
    candidate_repair_connections: list[object] = []
    candidate_validation_connections: list[object] = []

    class _Transaction:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, _exc_type, _exc, _traceback) -> bool:
            return False

    class _FreshConnection:
        _authnz_profile_user_backend = "postgres"

        def __init__(self) -> None:
            self._authnz_profile_user_guard_identity = self

        def transaction(self) -> _Transaction:
            return _Transaction()

        async def fetchval(self, query: str) -> str | bool:
            if query == "SELECT version()":
                return "PostgreSQL test"
            if "information_schema.tables" in query:
                return False
            raise AssertionError(f"Unexpected query: {query}")

        async def execute(self, query: object) -> str:
            concrete = _guard_sql(
                query,
                backend="postgres",
                connection_identity=self,
                operation="execute",
            )
            executed.append(concrete)
            return "CREATE"

    close_order: list[str] = []
    main_pool = _FakeAsyncpgPool("main", close_order)
    main_pool.connection = _FreshConnection()
    lock_pool = _FakeAsyncpgPool("lock", close_order)
    pools = iter((main_pool, lock_pool))

    async def _create_pool(*_args: Any, **_kwargs: Any) -> _FakeAsyncpgPool:
        return next(pools)

    async def _ensure_profile_version(conn: object) -> None:
        readiness_connections.append(conn)

    async def _repair_candidates(conn: object) -> None:
        candidate_repair_connections.append(conn)

    async def _validate_candidates(conn: object) -> None:
        candidate_validation_connections.append(conn)

    monkeypatch.setattr(database.asyncpg, "create_pool", _create_pool)
    monkeypatch.setattr(
        database,
        "ensure_postgres_profile_version_on_connection",
        _ensure_profile_version,
    )
    monkeypatch.setattr(
        database,
        "repair_postgres_profile_candidate_timestamps",
        _repair_candidates,
    )
    monkeypatch.setattr(
        database,
        "validate_postgres_profile_candidate_schema",
        _validate_candidates,
    )

    pool = database.DatabasePool(_postgres_settings())
    await pool.initialize()

    assert pool._initialized is True
    assert not any("CREATE EXTENSION" in sql.upper() for sql in executed)
    assert any("CREATE TABLE IF NOT EXISTS public.users" in sql for sql in executed)
    users_index = next(
        index
        for index, sql in enumerate(executed)
        if "CREATE TABLE IF NOT EXISTS public.users" in sql
    )
    assert "DEFAULT gen_random_uuid()" in executed[users_index]
    assert len(executed) > 1
    assert readiness_connections == [main_pool.connection]
    assert candidate_repair_connections == [main_pool.connection]
    assert candidate_validation_connections == [main_pool.connection]

    await pool.close()


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
    assert all(
        call["connection_class"] is database._GuardedAsyncpgConnection
        for call in create_calls
    )
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
async def test_postgres_database_error_closes_both_pools_in_lock_first_order(monkeypatch):
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
        raise database.DatabaseError("schema failed")

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
async def test_get_db_pool_publishes_only_successfully_initialized_candidate(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import database

    candidates: list[object] = []

    class _CandidatePool:
        def __init__(self, settings) -> None:
            self.settings = settings
            self._initialized = False
            candidates.append(self)

        async def initialize(self) -> None:
            if len(candidates) == 1:
                raise database.DatabaseError("schema failed")
            self._initialized = True

    monkeypatch.setattr(database, "_db_pool", None)
    monkeypatch.setattr(database, "DatabasePool", _CandidatePool)
    monkeypatch.setattr(database, "get_settings", _postgres_settings)

    with pytest.raises(database.DatabaseError):
        await database.get_db_pool()

    assert database._db_pool is None

    pool = await database.get_db_pool()

    assert pool is candidates[1]
    assert pool._initialized is True
    assert database._db_pool is pool


def test_get_db_pool_serializes_initialization_across_event_loops(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import database

    allow_initialize = threading.Event()
    first_candidate_started = threading.Event()
    second_candidate_started = threading.Event()
    candidates: list[object] = []
    candidates_guard = threading.Lock()

    class _CandidatePool:
        def __init__(self, settings) -> None:
            self.settings = settings
            self._initialized = False
            with candidates_guard:
                candidates.append(self)
                if len(candidates) == 1:
                    first_candidate_started.set()
                else:
                    second_candidate_started.set()

        async def initialize(self) -> None:
            await asyncio.to_thread(allow_initialize.wait)
            self._initialized = True

    monkeypatch.setattr(database, "_db_pool", None)
    monkeypatch.setattr(database, "DatabasePool", _CandidatePool)
    monkeypatch.setattr(database, "get_settings", _postgres_settings)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(lambda: asyncio.run(database.get_db_pool()))
        assert first_candidate_started.wait(timeout=1)
        second = executor.submit(lambda: asyncio.run(database.get_db_pool()))
        try:
            assert not second_candidate_started.wait(timeout=0.2)
        finally:
            allow_initialize.set()
        first_result = first.result(timeout=2)
        second_result = second.result(timeout=2)

    assert first_result is second_result is candidates[0]
    assert len(candidates) == 1
    database._db_pool = None


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_cancelled_lifecycle_waiter_exits_without_stranding_process_lock(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import database

    initialize_started = asyncio.Event()
    allow_initialize = asyncio.Event()

    class _CandidatePool:
        def __init__(self, settings) -> None:
            self.settings = settings
            self._initialized = False

        async def initialize(self) -> None:
            initialize_started.set()
            await allow_initialize.wait()
            self._initialized = True

    monkeypatch.setattr(database, "_db_pool", None)
    monkeypatch.setattr(database, "DatabasePool", _CandidatePool)
    monkeypatch.setattr(database, "get_settings", _postgres_settings)

    first = asyncio.create_task(database.get_db_pool())
    waiter: asyncio.Task[object] | None = None
    try:
        await asyncio.wait_for(initialize_started.wait(), timeout=1)
        waiter = asyncio.create_task(database.get_db_pool())
        await asyncio.sleep(0)
        waiter.cancel()
        await asyncio.sleep(0)
        assert waiter.done()
        with pytest.raises(asyncio.CancelledError):
            await waiter

        allow_initialize.set()
        first_result = await asyncio.wait_for(first, timeout=1)
        assert await asyncio.wait_for(database.get_db_pool(), timeout=1) is first_result
    finally:
        allow_initialize.set()
        pending = [first]
        if waiter is not None:
            pending.append(waiter)
        await asyncio.gather(*pending, return_exceptions=True)
        database._db_pool = None


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_lifecycle_waiters_do_not_saturate_default_executor(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import database

    initialize_started = asyncio.Event()
    allow_executor_work = asyncio.Event()

    class _CandidatePool:
        def __init__(self, settings) -> None:
            self.settings = settings
            self._initialized = False

        async def initialize(self) -> None:
            initialize_started.set()
            await allow_executor_work.wait()
            assert await asyncio.to_thread(lambda: "executor-available") == "executor-available"
            self._initialized = True

    monkeypatch.setattr(database, "_db_pool", None)
    monkeypatch.setattr(database, "DatabasePool", _CandidatePool)
    monkeypatch.setattr(database, "get_settings", _postgres_settings)
    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(executor)

    first = asyncio.create_task(database.get_db_pool())
    waiters: list[asyncio.Task[object]] = []
    try:
        await asyncio.wait_for(initialize_started.wait(), timeout=1)
        waiters = [asyncio.create_task(database.get_db_pool()) for _ in range(8)]
        await asyncio.sleep(0.01)
        allow_executor_work.set()

        first_result = await asyncio.wait_for(first, timeout=1)
        waiter_results = await asyncio.wait_for(asyncio.gather(*waiters), timeout=1)
        assert all(result is first_result for result in waiter_results)
    finally:
        allow_executor_work.set()
        await asyncio.gather(first, *waiters, return_exceptions=True)
        database._db_pool = None
        executor.shutdown(wait=True)


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_concurrent_get_db_pool_calls_publish_one_candidate(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import database

    initialize_started = asyncio.Event()
    allow_initialize = asyncio.Event()
    candidates: list[object] = []

    class _CandidatePool:
        def __init__(self, settings) -> None:
            self.settings = settings
            self._initialized = False
            candidates.append(self)

        async def initialize(self) -> None:
            initialize_started.set()
            await allow_initialize.wait()
            self._initialized = True

    monkeypatch.setattr(database, "_db_pool", None)
    monkeypatch.setattr(database, "DatabasePool", _CandidatePool)
    monkeypatch.setattr(database, "get_settings", _postgres_settings)

    first = asyncio.create_task(database.get_db_pool())
    second: asyncio.Task[object] | None = None
    try:
        await asyncio.wait_for(initialize_started.wait(), timeout=1)
        second = asyncio.create_task(database.get_db_pool())
        await asyncio.sleep(0)

        assert len(candidates) == 1

        allow_initialize.set()
        first_result, second_result = await asyncio.gather(first, second)
        assert first_result is second_result is candidates[0]
        assert database._db_pool is candidates[0]
    finally:
        allow_initialize.set()
        pending = [first]
        if second is not None:
            pending.append(second)
        await asyncio.gather(*pending, return_exceptions=True)
        database._db_pool = None


@pytest.mark.asyncio
@pytest.mark.concurrent
async def test_reset_db_pool_waits_for_inflight_singleton_initialization(monkeypatch):
    from tldw_Server_API.app.core.AuthNZ import (
        api_key_manager,
        database,
        llm_provider_overrides,
    )
    from tldw_Server_API.app.core.DB_Management import Users_DB
    from tldw_Server_API.app.core.MCP_unified import server as mcp_server

    initialize_started = asyncio.Event()
    allow_initialize = asyncio.Event()
    close_calls: list[object] = []

    class _CandidatePool:
        def __init__(self, settings) -> None:
            self.settings = settings
            self._initialized = False

        async def initialize(self) -> None:
            initialize_started.set()
            await allow_initialize.wait()
            self._initialized = True

        async def close(self) -> None:
            close_calls.append(self)
            self._initialized = False

    async def _noop_async() -> None:
        return None

    monkeypatch.setattr(database, "_db_pool", None)
    monkeypatch.setattr(database, "DatabasePool", _CandidatePool)
    monkeypatch.setattr(database, "get_settings", _postgres_settings)
    monkeypatch.setattr(mcp_server, "reset_mcp_server", _noop_async)
    monkeypatch.setattr(api_key_manager, "reset_api_key_manager", _noop_async)
    monkeypatch.setattr(Users_DB, "reset_users_db", _noop_async)
    monkeypatch.setattr(
        llm_provider_overrides,
        "set_llm_provider_overrides_cache_for_tests",
        lambda _value: None,
    )
    get_task = asyncio.create_task(database.get_db_pool())
    reset_task: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(initialize_started.wait(), timeout=1)
        reset_task = asyncio.create_task(database.reset_db_pool())
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert not reset_task.done()

        allow_initialize.set()
        initialized_pool, _ = await asyncio.gather(get_task, reset_task)
        assert close_calls == [initialized_pool]
        assert database._db_pool is None
    finally:
        allow_initialize.set()
        pending: list[asyncio.Task[Any]] = [get_task]
        if reset_task is not None:
            pending.append(reset_task)
        await asyncio.gather(*pending, return_exceptions=True)
        database._db_pool = None


@pytest.mark.asyncio
async def test_sqlite_initialization_error_is_sanitized_database_error(
    monkeypatch,
    tmp_path,
):
    from tldw_Server_API.app.core.AuthNZ import database

    marker = "sqlite failed at /private/users.db?token=secret"
    settings = SimpleNamespace(
        AUTH_MODE="single_user",
        DATABASE_URL=f"sqlite:///{tmp_path / 'users.db'}",
    )

    async def _fail_sqlite_schema(_self) -> None:
        raise sqlite3.OperationalError(marker)

    monkeypatch.setattr(database.DatabasePool, "_create_sqlite_schema", _fail_sqlite_schema)
    output = io.StringIO()
    sink = database.logger.add(output, format="{message} {extra}")
    try:
        with pytest.raises(database.DatabaseError) as exc_info:
            await database.DatabasePool(settings).initialize()
    finally:
        database.logger.remove(sink)

    assert str(exc_info.value) == "Database initialization failed"
    assert exc_info.value.__cause__ is None
    assert marker not in output.getvalue()
    assert "/private/" not in output.getvalue()


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
async def test_initialize_cleanup_cancellation_wins_over_ordinary_failure():
    from tldw_Server_API.app.core.AuthNZ import database

    close_order: list[str] = []
    main_pool = _GatedClosePool("main", close_order)
    manager = database.DatabasePool(_postgres_settings())
    manager.pool = main_pool
    manager._openai_credential_lock_pool = None

    def _fail_backend_selection() -> bool:
        raise RuntimeError("private initialization failure")

    manager._should_use_postgres = _fail_backend_selection  # type: ignore[method-assign]
    initialize_task = asyncio.create_task(manager.initialize())

    try:
        await asyncio.wait_for(main_pool.close_started.wait(), timeout=1)
        initialize_task.cancel()
        await asyncio.sleep(0)
        assert not initialize_task.done()

        main_pool.allow_close_return.set()
        with pytest.raises(asyncio.CancelledError):
            await initialize_task

        assert manager.pool is None
        assert close_order == ["main"]
    finally:
        main_pool.allow_close_return.set()
        if not initialize_task.done():
            await asyncio.gather(initialize_task, return_exceptions=True)


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
