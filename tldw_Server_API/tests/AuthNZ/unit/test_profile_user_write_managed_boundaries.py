from __future__ import annotations

import gc
import sqlite3
import weakref
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import asyncpg
import pytest

from tldw_Server_API.app.core.AuthNZ.database import (
    DatabasePool,
    _GuardedAsyncpgConnection,
)
from tldw_Server_API.app.core.AuthNZ.exceptions import DatabaseError as AuthNZDatabaseError
from tldw_Server_API.app.core.AuthNZ.profile_user_sync_boundary import (
    _guard_authnz_sync_backend,
)
from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import (
    ProfileUserWriteRejected,
    _guard_sql,
    _mint_profile_user_sql,
    _profile_user_backend,
    _profile_user_connection_identity,
    _revoke_profile_user_sql,
)
from tldw_Server_API.app.core.DB_Management.backends.base import (
    BackendType,
    DatabaseConfig,
    DatabaseError,
)
from tldw_Server_API.app.core.DB_Management.backends.factory import (
    close_all_backends,
    is_factory_managed_backend,
    reset_managed_sqlite_backends,
)
from tldw_Server_API.app.core.DB_Management.UserDatabase_v2 import UserDatabase
from tldw_Server_API.app.core.UserProfiles.version_gateway import ProfileVersionGateway


def _initialized_sqlite_pool(path: Path) -> DatabasePool:
    pool = DatabasePool.__new__(DatabasePool)
    pool.pool = None
    pool.db_path = str(path)
    pool._sqlite_uri = False
    pool._initialized = True
    return pool


class _TrackingSyncCursor:
    def __init__(self, raw_calls: list[str]) -> None:
        self._raw_calls = raw_calls
        self.rowcount = 0
        self.lastrowid = None
        self.description = None
        self.arraysize = 1

    def execute(self, query: str, parameters: Any = None) -> _TrackingSyncCursor:
        del query, parameters
        self._raw_calls.append("cursor.execute")
        return self

    def executemany(self, query: str, parameters: Any) -> _TrackingSyncCursor:
        del query, parameters
        self._raw_calls.append("cursor.executemany")
        return self

    def executescript(self, query: str) -> _TrackingSyncCursor:
        del query
        self._raw_calls.append("cursor.executescript")
        return self

    def fetchone(self) -> tuple[int]:
        self._raw_calls.append("cursor.fetchone")
        return (1,)

    def fetchmany(self, size: int | None = None) -> list[tuple[int]]:
        del size
        self._raw_calls.append("cursor.fetchmany")
        return [(1,)]

    def fetchall(self) -> list[tuple[int]]:
        self._raw_calls.append("cursor.fetchall")
        return [(1,)]

    def close(self) -> None:
        self._raw_calls.append("cursor.close")

    def __iter__(self) -> Any:
        self._raw_calls.append("cursor.__iter__")
        return iter([(1,)])

    def __enter__(self) -> _TrackingSyncCursor:
        self._raw_calls.append("cursor.__enter__")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        del exc_type, exc, traceback
        self._raw_calls.append("cursor.__exit__")
        return False


class _TrackingSyncConnection:
    def __init__(self, raw_calls: list[str]) -> None:
        self._raw_calls = raw_calls
        self.in_transaction = False
        self.closed = False

    def cursor(self, *args: Any, **kwargs: Any) -> _TrackingSyncCursor:
        del args, kwargs
        self._raw_calls.append("connection.cursor")
        return _TrackingSyncCursor(self._raw_calls)

    def commit(self) -> None:
        self._raw_calls.append("connection.commit")

    def rollback(self) -> None:
        self._raw_calls.append("connection.rollback")

    def close(self) -> None:
        self._raw_calls.append("connection.close")
        self.closed = True

    def __enter__(self) -> _TrackingSyncConnection:
        self._raw_calls.append("connection.__enter__")
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        del exc_type, exc, traceback
        self._raw_calls.append("connection.__exit__")
        return False


class _TrackingSyncPool:
    def __init__(self, raw_connection: _TrackingSyncConnection) -> None:
        self._raw_connection = raw_connection
        self.return_count = 0

    def get_connection(self) -> _TrackingSyncConnection:
        return self._raw_connection

    def return_connection(self, connection: _TrackingSyncConnection) -> None:
        assert connection is self._raw_connection
        self.return_count += 1

    @contextmanager
    def connection(self) -> Any:
        connection = self.get_connection()
        try:
            yield connection
        finally:
            self.return_connection(connection)

    def close_all(self) -> None:
        pass

    def get_stats(self) -> dict[str, int]:
        return {}

    def clear_thread_local_connection(self) -> None:
        pass


class _TrackingSyncBackend:
    backend_type = SimpleNamespace(value="sqlite")
    config = SimpleNamespace()
    features = SimpleNamespace()

    def __init__(self, raw_calls: list[str]) -> None:
        self.pool = _TrackingSyncPool(_TrackingSyncConnection(raw_calls))

    def get_pool(self) -> _TrackingSyncPool:
        return self.pool


@pytest.mark.asyncio
async def test_sqlite_acquire_guards_connection_and_returned_cursor_writes(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "authnz.db"
    with sqlite3.connect(db_path) as setup:
        setup.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT, password_hash TEXT)"
        )
        setup.execute(
            "INSERT INTO users (id, email, password_hash) VALUES (1, 'before', 'hash')"
        )
    pool = _initialized_sqlite_pool(db_path)

    async with pool.acquire() as conn:
        assert conn._authnz_profile_user_backend == "sqlite"
        with pytest.raises(ProfileUserWriteRejected):
            await conn.execute(
                "INSERT INTO users (email, password_hash) VALUES ('raw', 'hash')"
            )
        with pytest.raises(ProfileUserWriteRejected):
            await conn.execute(
                """
                CREATE TRIGGER rewrite_user AFTER UPDATE ON users
                BEGIN
                    UPDATE users SET email = 'trigger' WHERE id = NEW.id;
                END
                """
            )
        with pytest.raises(ProfileUserWriteRejected):
            await conn.execute("UPDATE users SET email = 'execute' WHERE id = 1")
        with pytest.raises(ProfileUserWriteRejected):
            await conn.execute("PRAGMA writable_schema=1")
        with pytest.raises(ProfileUserWriteRejected):
            await conn.execute(
                "UPDATE sqlite_master SET sql = 'invalid' WHERE name = 'users'"
            )
        with pytest.raises(ProfileUserWriteRejected):
            await conn.executemany(
                "UPDATE users SET email = ? WHERE id = ?",
                [("many", 1)],
            )
        with pytest.raises(ProfileUserWriteRejected):
            await conn.executescript(
                "SELECT 1; UPDATE users SET email = 'script' WHERE id = 1"
            )
        with pytest.raises(ProfileUserWriteRejected):
            await conn.execute_insert(
                "INSERT INTO users (email, password_hash) VALUES (?, ?)",
                ("insert", "hash"),
            )

        cursor = await conn.execute("SELECT email FROM users WHERE id = 1")
        assert cursor.connection is conn
        assert (await cursor.fetchone())[0] == "before"
        with pytest.raises(ProfileUserWriteRejected):
            await cursor.execute("UPDATE users SET email = 'cursor' WHERE id = 1")
        with pytest.raises(ProfileUserWriteRejected):
            await cursor.execute("PRAGMA writable_schema(1)")
        with pytest.raises(ProfileUserWriteRejected):
            await cursor.executemany(
                "UPDATE users SET email = ? WHERE id = ?",
                [("cursor-many", 1)],
            )
        with pytest.raises(ProfileUserWriteRejected):
            await cursor.executescript(
                "UPDATE users SET email = 'cursor-script' WHERE id = 1"
            )

        rows = await conn.execute_fetchall("SELECT email FROM users WHERE id = 1")
        assert rows[0][0] == "before"
        count_rows = await conn.execute_fetchall("SELECT COUNT(*) FROM users")
        assert count_rows[0][0] == 1
        trigger_rows = await conn.execute_fetchall(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        )
        assert trigger_rows == []


@pytest.mark.asyncio
async def test_profile_version_touch_capability_reaches_async_managed_boundary(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "async-authnz-profile-touch.db"
    with sqlite3.connect(db_path) as setup:
        setup.executescript(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                profile_version TEXT NOT NULL
            );
            INSERT INTO users VALUES (1, '2026-01-01T00:00:00.000000Z');
            """
        )
    pool = _initialized_sqlite_pool(db_path)

    async with pool.transaction() as connection:
        await ProfileVersionGateway(pool).touch(
            connection,
            1,
            datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc),
        )

    with sqlite3.connect(db_path) as check:
        value = check.execute(
            "SELECT profile_version FROM users WHERE id = 1"
        ).fetchone()[0]
    assert value == "2026-08-01T12:00:00.000000Z"

@pytest.mark.asyncio
async def test_asyncpg_connection_guards_all_raw_query_and_copy_entrypoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegated: list[str] = []

    async def _unexpected_delegate(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        delegated.append("called")
        return None

    for method_name in (
        "execute",
        "executemany",
        "fetch",
        "fetchrow",
        "fetchval",
        "prepare",
        "copy_from_query",
        "copy_to_table",
        "copy_records_to_table",
    ):
        monkeypatch.setattr(asyncpg.Connection, method_name, _unexpected_delegate)

    connection = object.__new__(_GuardedAsyncpgConnection)
    connection._authnz_profile_user_guard_identity = object()
    connection._aborted = True
    assert connection._authnz_profile_user_backend == "postgres"
    protected = "UPDATE users SET email = $1 WHERE id = $2"

    calls = (
        lambda: connection.execute(protected),
        lambda: connection.executemany(protected, []),
        lambda: connection.fetch(protected),
        lambda: connection.fetchrow(protected),
        lambda: connection.fetchval(protected),
        lambda: connection.prepare(protected),
        lambda: connection.copy_from_query(protected, output=object()),
        lambda: connection.copy_to_table("users", source=object()),
        lambda: connection.copy_to_table('"users"', source=object()),
        lambda: connection.copy_records_to_table("public.users", records=[]),
        lambda: connection.copy_to_table("org_members", source=object()),
        lambda: connection.copy_to_table(
            '"team_members"',
            source=object(),
        ),
        lambda: connection.copy_records_to_table(
            "public.org_members",
            records=[],
        ),
        lambda: connection.copy_records_to_table(
            "team_members",
            records=[],
        ),
    )
    for call in calls:
        with pytest.raises(ProfileUserWriteRejected):
            await call()

    assert delegated == []


@pytest.mark.asyncio
async def test_asyncpg_connection_reset_bypasses_guard_for_driver_cleanup_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reset_query = "SELECT pg_advisory_unlock_all();\nCLOSE ALL;\nRESET ALL;"
    delegated: list[tuple[str, str | None, float | None]] = []

    async def _reset(connection: Any) -> None:
        del connection
        delegated.append(("reset", None, None))

    def _get_reset_query(connection: Any) -> str:
        del connection
        return reset_query

    async def _execute(
        connection: Any,
        query: str,
        *args: Any,
        timeout: float | None = None,
    ) -> str:
        del connection, args
        delegated.append(("execute", query, timeout))
        return "RESET"

    monkeypatch.setattr(asyncpg.Connection, "_reset", _reset)
    monkeypatch.setattr(asyncpg.Connection, "get_reset_query", _get_reset_query)
    monkeypatch.setattr(asyncpg.Connection, "execute", _execute)
    connection = object.__new__(_GuardedAsyncpgConnection)
    connection._authnz_profile_user_guard_identity = object()
    connection._aborted = True

    await connection.reset(timeout=2.0)

    assert delegated == [
        ("reset", None, None),
        ("execute", reset_query, 2.0),
    ]
    with pytest.raises(ProfileUserWriteRejected):
        await connection.execute(reset_query)


@pytest.mark.asyncio
async def test_asyncpg_connection_allows_unprotected_copy_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegated: list[tuple[str, str, str | None]] = []

    async def _copy_to_table(
        connection: Any,
        table_name: str,
        *,
        schema_name: str | None = None,
        **kwargs: Any,
    ) -> str:
        del connection, kwargs
        delegated.append(("copy_to_table", table_name, schema_name))
        return "COPY 1"

    async def _copy_records_to_table(
        connection: Any,
        table_name: str,
        *,
        schema_name: str | None = None,
        **kwargs: Any,
    ) -> str:
        del connection, kwargs
        delegated.append(("copy_records_to_table", table_name, schema_name))
        return "COPY 1"

    monkeypatch.setattr(asyncpg.Connection, "copy_to_table", _copy_to_table)
    monkeypatch.setattr(
        asyncpg.Connection,
        "copy_records_to_table",
        _copy_records_to_table,
    )
    connection = object.__new__(_GuardedAsyncpgConnection)
    connection._authnz_profile_user_guard_identity = object()

    assert (
        await connection.copy_to_table(
            "audit_events",
            schema_name="public",
            source=object(),
        )
        == "COPY 1"
    )
    assert (
        await connection.copy_records_to_table(
            "api_keys",
            records=[],
        )
        == "COPY 1"
    )
    assert delegated == [
        ("copy_to_table", "audit_events", "public"),
        ("copy_records_to_table", "api_keys", None),
    ]


@pytest.mark.asyncio
async def test_asyncpg_connection_rejects_stored_routine_ddl_without_delegation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegated: list[str] = []

    async def _unexpected_delegate(*args: Any, **kwargs: Any) -> str:
        del args, kwargs
        delegated.append("called")
        return "CREATE"

    monkeypatch.setattr(asyncpg.Connection, "execute", _unexpected_delegate)
    connection = object.__new__(_GuardedAsyncpgConnection)
    connection._authnz_profile_user_guard_identity = object()
    connection._aborted = True

    for statement in (
        """
        CREATE FUNCTION rewrite_user() RETURNS void LANGUAGE SQL AS $$
            UPDATE users SET email = 'function@example.com' WHERE id = 1
        $$
        """,
        """
        CREATE OR REPLACE PROCEDURE rewrite_user() LANGUAGE plpgsql AS $$
        BEGIN
            UPDATE users SET email = 'procedure@example.com' WHERE id = 1;
        END
        $$
        """,
        """
        CREATE TRIGGER rewrite_user BEFORE UPDATE ON users
        FOR EACH ROW EXECUTE FUNCTION rewrite_user()
        """,
        """
        CREATE RULE rewrite_user AS ON UPDATE TO api_keys
        DO ALSO UPDATE users SET email = 'rule@example.com' WHERE id = NEW.user_id
        """,
    ):
        with pytest.raises(ProfileUserWriteRejected):
            await connection.execute(statement)

    assert delegated == []


@pytest.mark.asyncio
async def test_asyncpg_pool_proxy_and_connection_share_capability_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegated: list[tuple[str, tuple[Any, ...]]] = []

    async def _execute(
        connection: Any,
        query: str,
        *args: Any,
        timeout: float | None = None,
    ) -> str:
        del connection, timeout
        delegated.append((query, args))
        return "UPDATE 1"

    monkeypatch.setattr(asyncpg.Connection, "execute", _execute)
    connection = object.__new__(_GuardedAsyncpgConnection)
    connection._authnz_profile_user_guard_identity = object()
    connection._aborted = True

    class _PoolProxy:
        def __init__(self, target: _GuardedAsyncpgConnection) -> None:
            self._target = target

        def __getattr__(self, name: str) -> Any:
            return getattr(self._target, name)

    proxy = _PoolProxy(connection)
    capability = _mint_profile_user_sql(
        "UPDATE users SET email = $1 WHERE id = $2",
        backend="postgres",
        connection_identity=_profile_user_connection_identity(proxy),
        operation="update",
        columns=("email",),
    )
    try:
        assert await proxy.execute(capability, "pool@example.com", 9) == "UPDATE 1"
    finally:
        _revoke_profile_user_sql(capability)

    assert delegated == [
        (
            "UPDATE users SET email = $1 WHERE id = $2",
            ("pool@example.com", 9),
        )
    ]


def test_asyncpg_pool_proxy_resolves_guarded_backend_and_identity() -> None:
    connection = object.__new__(_GuardedAsyncpgConnection)
    connection._authnz_profile_user_guard_identity = object()
    connection._aborted = True
    proxy = object.__new__(asyncpg.pool.PoolConnectionProxy)
    proxy._con = connection
    proxy._holder = None

    assert _profile_user_backend(proxy) == "postgres"
    assert _profile_user_connection_identity(proxy) is (
        connection._authnz_profile_user_guard_identity
    )


def test_released_asyncpg_pool_proxy_rejects_managed_boundary_resolution() -> None:
    proxy = object.__new__(asyncpg.pool.PoolConnectionProxy)
    proxy._con = None
    proxy._holder = None

    with pytest.raises(ProfileUserWriteRejected):
        _profile_user_backend(proxy)
    with pytest.raises(ProfileUserWriteRejected):
        _profile_user_connection_identity(proxy)


def test_asyncpg_cursor_factory_rejects_protected_query_without_delegation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delegated: list[str] = []

    def _cursor(*args: Any, **kwargs: Any) -> object:
        del args, kwargs
        delegated.append("called")
        return object()

    monkeypatch.setattr(asyncpg.Connection, "cursor", _cursor)
    connection = object.__new__(_GuardedAsyncpgConnection)
    connection._authnz_profile_user_guard_identity = object()
    connection._aborted = True

    with pytest.raises(ProfileUserWriteRejected):
        connection.cursor(
            "UPDATE users SET email = $1 WHERE id = $2",
            "cursor@example.com",
            4,
            prefetch=50,
        )

    assert delegated == []


def test_user_database_backend_rejects_raw_visible_write_and_remains_factory_managed(
    tmp_path: Path,
) -> None:
    close_all_backends()
    db_path = tmp_path / "sync-authnz.db"
    try:
        db = UserDatabase(
            config=DatabaseConfig(
                backend_type=BackendType.SQLITE,
                sqlite_path=str(db_path),
            )
        )
        with sqlite3.connect(db_path) as setup:
            setup.execute(
                """
                INSERT INTO users (uuid, username, email, password_hash)
                VALUES ('sync-user', 'sync-user', 'before@example.com', 'hash')
                """
            )

        assert db.backend._authnz_profile_user_backend == "sqlite"

        assert is_factory_managed_backend(db.backend)
        with pytest.raises(ProfileUserWriteRejected):
            db.backend.execute(
                "UPDATE users SET email = ? WHERE username = ?",
                ("after@example.com", "sync-user"),
            )

        with sqlite3.connect(db_path) as check:
            email = check.execute(
                "SELECT email FROM users WHERE username = 'sync-user'"
            ).fetchone()[0]
        assert email == "before@example.com"
    finally:
        close_all_backends()


def test_user_database_connections_and_cursors_do_not_expose_raw_write_bypasses(
    tmp_path: Path,
) -> None:
    close_all_backends()
    db_path = tmp_path / "sync-authnz-connections.db"
    try:
        db = UserDatabase(
            config=DatabaseConfig(
                backend_type=BackendType.SQLITE,
                sqlite_path=str(db_path),
            )
        )
        with sqlite3.connect(db_path) as setup:
            setup.execute(
                """
                INSERT INTO users (uuid, username, email, password_hash)
                VALUES ('connection-user', 'connection-user', 'before@example.com', 'hash')
                """
            )

        with pytest.raises(ProfileUserWriteRejected):
            db.backend.execute_many(
                "UPDATE users SET email = ? WHERE username = ?",
                [("backend-many@example.com", "connection-user")],
            )

        with db.backend.transaction() as transaction_connection:
            assert transaction_connection._authnz_profile_user_backend == "sqlite"
            with pytest.raises(ProfileUserWriteRejected):
                transaction_connection.execute(
                    "UPDATE users SET email = 'transaction@example.com' "
                    "WHERE username = 'connection-user'"
                )
            cursor = transaction_connection.cursor()
            with pytest.raises(ProfileUserWriteRejected):
                cursor.execute(
                    "UPDATE users SET email = ? WHERE username = ?",
                    ("cursor@example.com", "connection-user"),
                )
            with pytest.raises(ProfileUserWriteRejected):
                cursor.executemany(
                    "UPDATE users SET email = ? WHERE username = ?",
                    [("cursor-many@example.com", "connection-user")],
                )
            with pytest.raises(ProfileUserWriteRejected):
                cursor.executescript(
                    "SELECT 1; UPDATE users SET email = 'script@example.com' "
                    "WHERE username = 'connection-user'"
                )

        pool = db.backend.get_pool()
        pooled_connection = pool.get_connection()
        try:
            assert pooled_connection._authnz_profile_user_backend == "sqlite"
            with pytest.raises(ProfileUserWriteRejected):
                pooled_connection.execute(
                    "UPDATE users SET email = 'pool@example.com' "
                    "WHERE username = 'connection-user'"
                )
        finally:
            pool.return_connection(pooled_connection)

        direct_connection = db.backend.connect()
        try:
            assert direct_connection._authnz_profile_user_backend == "sqlite"
            with pytest.raises(ProfileUserWriteRejected):
                direct_connection.execute(
                    "UPDATE users SET email = 'direct@example.com' "
                    "WHERE username = 'connection-user'"
                )
        finally:
            db.backend.disconnect(direct_connection)

        with sqlite3.connect(db_path) as check:
            email = check.execute(
                "SELECT email FROM users WHERE username = 'connection-user'"
            ).fetchone()[0]
        assert email == "before@example.com"
    finally:
        close_all_backends()


def test_user_database_gateway_capability_reaches_sync_managed_boundary(
    tmp_path: Path,
) -> None:
    close_all_backends()
    try:
        db = UserDatabase(
            config=DatabaseConfig(
                backend_type=BackendType.SQLITE,
                sqlite_path=str(tmp_path / "sync-authnz-gateway.db"),
            )
        )
        user_id = db.create_user(
            username="gateway-user",
            email="before@example.com",
            password_hash="hash",
        )

        assert db.update_user(user_id, email="after@example.com") is True
        assert db.get_user(user_id=user_id)["email"] == "after@example.com"
    finally:
        close_all_backends()


def test_profile_version_touch_capability_reaches_sync_managed_boundary(
    tmp_path: Path,
) -> None:
    close_all_backends()
    try:
        db = UserDatabase(
            config=DatabaseConfig(
                backend_type=BackendType.SQLITE,
                sqlite_path=str(tmp_path / "sync-authnz-profile-touch.db"),
            )
        )
        user_id = db.create_user(
            username="profile-touch-user",
            email="profile-touch@example.com",
            password_hash="hash",
        )
        gateway = ProfileVersionGateway(
            SimpleNamespace(backend_type="sqlite", pool=None)
        )

        with db.backend.transaction() as connection:
            gateway.touch_sync(
                db.backend,
                connection,
                user_id,
                datetime(2026, 8, 1, 12, 0, tzinfo=timezone.utc),
            )

        assert db.get_user(user_id=user_id)["profile_version"] == (
            "2026-08-01T12:00:00.000000Z"
        )
    finally:
        close_all_backends()


def test_guarded_sync_pool_does_not_retain_returned_connection_wrappers(
    tmp_path: Path,
) -> None:
    close_all_backends()
    try:
        db = UserDatabase(
            config=DatabaseConfig(
                backend_type=BackendType.SQLITE,
                sqlite_path=str(tmp_path / "sync-authnz-wrapper-lifetime.db"),
            )
        )
        pool = db.backend.get_pool()
        connection = pool.get_connection()
        connection_ref = weakref.ref(connection)

        pool.return_connection(connection)
        del connection
        gc.collect()

        assert connection_ref() is None
    finally:
        close_all_backends()


def test_guarded_sync_pool_return_retires_wrapper_and_existing_cursor() -> None:
    raw_calls: list[str] = []
    backend = _guard_authnz_sync_backend(_TrackingSyncBackend(raw_calls))
    pool = backend.get_pool()
    connection = pool.get_connection()
    cursor = connection.cursor()
    old_identity = connection._authnz_profile_user_guard_identity

    pool.return_connection(connection)
    raw_calls.clear()

    with pytest.raises(RuntimeError, match="lease is no longer active"):
        connection.execute("SELECT 1")
    with pytest.raises(RuntimeError, match="lease is no longer active"):
        cursor.fetchone()
    assert raw_calls == []

    next_connection = pool.get_connection()
    assert next_connection._authnz_profile_user_guard_identity is not old_identity
    assert next_connection.execute("SELECT 1").fetchone() == (1,)
    pool.return_connection(next_connection)


def test_guarded_sync_pool_context_exit_retires_wrapper_and_cursor() -> None:
    raw_calls: list[str] = []
    raw_backend = _TrackingSyncBackend(raw_calls)
    pool = _guard_authnz_sync_backend(raw_backend).get_pool()

    with pool.connection() as connection:
        cursor = connection.cursor()

    assert raw_backend.pool.return_count == 1
    raw_calls.clear()
    with pytest.raises(RuntimeError, match="lease is no longer active"):
        connection.cursor()
    with pytest.raises(RuntimeError, match="lease is no longer active"):
        cursor.fetchall()
    with pytest.raises(RuntimeError, match="already released"):
        pool.return_connection(connection)
    assert raw_backend.pool.return_count == 1
    assert raw_calls == []


def test_guarded_sync_cursor_context_exit_retires_only_that_cursor() -> None:
    raw_calls: list[str] = []
    pool = _guard_authnz_sync_backend(_TrackingSyncBackend(raw_calls)).get_pool()
    connection = pool.get_connection()

    with connection.cursor() as cursor:
        assert cursor.fetchone() == (1,)

    raw_calls.clear()
    with pytest.raises(RuntimeError, match="cursor is no longer active"):
        cursor.fetchone()
    assert connection.execute("SELECT 1").fetchone() == (1,)
    pool.return_connection(connection)


def test_retired_sync_lease_capabilities_fail_before_raw_connection_reuse() -> None:
    raw_calls: list[str] = []
    pool = _guard_authnz_sync_backend(_TrackingSyncBackend(raw_calls)).get_pool()
    connection = pool.get_connection()
    identity = connection._authnz_profile_user_guard_identity
    stale_wrapper_capability = _mint_profile_user_sql(
        "UPDATE users SET email = ? WHERE id = ?",
        backend="sqlite",
        connection_identity=identity,
        operation="update",
        columns=("email",),
    )
    reused_raw_capability = _mint_profile_user_sql(
        "UPDATE users SET email = ? WHERE id = ?",
        backend="sqlite",
        connection_identity=identity,
        operation="update",
        columns=("email",),
    )

    try:
        pool.return_connection(connection)
        next_connection = pool.get_connection()
        raw_calls.clear()

        with pytest.raises(RuntimeError, match="lease is no longer active"):
            connection.execute(stale_wrapper_capability, ("stale@example.com", 1))
        with pytest.raises(ProfileUserWriteRejected):
            next_connection.execute(reused_raw_capability, ("stale@example.com", 1))
        assert raw_calls == []
        pool.return_connection(next_connection)
    finally:
        _revoke_profile_user_sql(stale_wrapper_capability)
        _revoke_profile_user_sql(reused_raw_capability)


def test_user_database_guarded_backend_selected_reset_retires_factory_owner(
    tmp_path: Path,
) -> None:
    close_all_backends()
    db = UserDatabase(
        config=DatabaseConfig(
            backend_type=BackendType.SQLITE,
            sqlite_path=str(tmp_path / "sync-authnz-reset.db"),
        )
    )
    pool = db.backend.get_pool()
    pool.get_connection()

    reset_managed_sqlite_backends(mode="hard", backends=[db.backend])

    assert not is_factory_managed_backend(db.backend)
    with pytest.raises(DatabaseError):
        db.backend.get_pool()
    with pytest.raises(DatabaseError):
        pool.get_connection()


@pytest.mark.asyncio
async def test_postgres_bootstrap_executes_static_ddl_as_guarded_single_statements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed: list[str] = []
    readiness_connections: list[object] = []
    candidate_repair_connections: list[object] = []
    candidate_validation_connections: list[object] = []

    async def _ensure_readiness(connection: object) -> None:
        readiness_connections.append(connection)

    async def _repair_candidates(connection: object) -> None:
        candidate_repair_connections.append(connection)

    async def _validate_candidates(connection: object) -> None:
        candidate_validation_connections.append(connection)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.database.ensure_postgres_profile_version_on_connection",
        _ensure_readiness,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.database.repair_postgres_profile_candidate_timestamps",
        _repair_candidates,
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.database.validate_postgres_profile_candidate_schema",
        _validate_candidates,
    )

    class _Transaction:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            del exc_type, exc, tb
            return False

    class _Connection:
        _authnz_profile_user_backend = "postgres"

        def __init__(self) -> None:
            self._authnz_profile_user_guard_identity = self

        async def fetchval(self, query: str) -> bool:
            assert "table_schema = 'public'" in query
            del query
            return False

        def transaction(self) -> _Transaction:
            return _Transaction()

        async def execute(self, query: object) -> str:
            concrete = _guard_sql(
                query,
                backend="postgres",
                connection_identity=self,
                operation="execute",
            )
            executed.append(concrete)
            return "CREATE"

    class _Acquire:
        async def __aenter__(self) -> _Connection:
            return _Connection()

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            del exc_type, exc, tb
            return False

    class _Pool:
        def acquire(self) -> _Acquire:
            return _Acquire()

    database_pool = DatabasePool.__new__(DatabasePool)
    database_pool.pool = _Pool()

    await database_pool._create_postgresql_schema()

    assert len(readiness_connections) == 1
    assert candidate_repair_connections == readiness_connections
    assert candidate_validation_connections == readiness_connections
    assert len(executed) > 1
    assert all(statement.count(";") <= 1 for statement in executed)
    assert any(
        "CREATE TABLE IF NOT EXISTS public.users" in statement
        for statement in executed
    )


@pytest.mark.asyncio
async def test_postgres_bootstrap_propagates_sanitized_readiness_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "postgres://user:password@example.invalid/authnz"

    async def _fail_readiness(connection: object) -> None:
        del connection
        raise RuntimeError(secret)

    monkeypatch.setattr(
        "tldw_Server_API.app.core.AuthNZ.database.ensure_postgres_profile_version_on_connection",
        _fail_readiness,
    )

    class _Transaction:
        async def __aenter__(self) -> None:
            return None

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            del exc_type, exc, tb
            return False

    class _Connection:
        async def fetchval(self, query: str) -> bool:
            assert "table_schema = 'public'" in query
            return True

        def transaction(self) -> _Transaction:
            return _Transaction()

    class _Acquire:
        async def __aenter__(self) -> _Connection:
            return _Connection()

        async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
            del exc_type, exc, tb
            return False

    class _Pool:
        def acquire(self) -> _Acquire:
            return _Acquire()

    database_pool = DatabasePool.__new__(DatabasePool)
    database_pool.pool = _Pool()

    with pytest.raises(
        AuthNZDatabaseError,
        match="PostgreSQL AuthNZ schema readiness failed",
    ) as exc_info:
        await database_pool._create_postgresql_schema()

    assert secret not in str(exc_info.value)
    assert exc_info.value.__cause__ is None
