"""Sync managed-connection boundary for the legacy AuthNZ user database."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from threading import RLock
from typing import Any

from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import _guard_sql


def _backend_dialect(backend: Any) -> str:
    value = getattr(getattr(backend, "backend_type", None), "value", None)
    if value == "sqlite":
        return "sqlite"
    if value in {"postgres", "postgresql"}:
        return "postgres"
    raise RuntimeError("Unsupported managed AuthNZ backend")


class _SyncLeaseState:
    """Generation identity and lifecycle shared by one leased wrapper tree."""

    def __init__(self, *, identity: object | None = None) -> None:
        self.identity = identity if identity is not None else object()
        self._active = True
        self._released = False
        self._lock = RLock()

    @contextmanager
    def access(self) -> Generator[None, None, None]:
        with self._lock:
            if not self._active:
                raise RuntimeError("Managed AuthNZ connection lease is no longer active")
            yield

    @contextmanager
    def release(self) -> Generator[None, None, None]:
        with self._lock:
            if self._released:
                raise RuntimeError("Managed AuthNZ connection lease was already released")
            self._active = False
            self._released = True
            yield

    def retire(self) -> None:
        with self._lock:
            self._active = False


class _GuardedSyncCursor:
    def __init__(
        self,
        cursor: Any,
        *,
        connection: _GuardedSyncConnection,
        dialect: str,
    ) -> None:
        self._cursor = cursor
        self._connection = connection
        self._dialect = dialect
        self._active = True

    @contextmanager
    def _access(self) -> Generator[None, None, None]:
        with self._connection._lease.access():
            if not self._active:
                raise RuntimeError("Managed AuthNZ cursor is no longer active")
            yield

    @property
    def connection(self) -> _GuardedSyncConnection:
        return self._connection

    @property
    def rowcount(self) -> int:
        with self._access():
            return self._cursor.rowcount

    @property
    def lastrowid(self) -> Any:
        with self._access():
            return getattr(self._cursor, "lastrowid", None)

    @property
    def description(self) -> Any:
        with self._access():
            return self._cursor.description

    @property
    def arraysize(self) -> int:
        with self._access():
            return self._cursor.arraysize

    @arraysize.setter
    def arraysize(self, value: int) -> None:
        with self._access():
            self._cursor.arraysize = value

    def _guard(self, query: Any, *, operation: str) -> str:
        return _guard_sql(
            query,
            backend=self._dialect,
            connection_identity=self._connection._authnz_profile_user_guard_identity,
            operation=operation,
        )

    def _execute_guarded(
        self,
        guarded: str,
        parameters: Any = None,
    ) -> _GuardedSyncCursor:
        if parameters is None:
            self._cursor.execute(guarded)
        else:
            self._cursor.execute(guarded, parameters)
        return self

    def execute(self, query: Any, parameters: Any = None) -> _GuardedSyncCursor:
        with self._access():
            return self._execute_guarded(
                self._guard(query, operation="execute"),
                parameters,
            )

    def executemany(self, query: Any, parameters: Any) -> _GuardedSyncCursor:
        with self._access():
            self._cursor.executemany(
                self._guard(query, operation="executemany"),
                parameters,
            )
            return self

    def executescript(self, query: Any) -> _GuardedSyncCursor:
        with self._access():
            self._cursor.executescript(
                self._guard(query, operation="executescript")
            )
            return self

    def copy(self, query: Any, *args: Any, **kwargs: Any) -> Any:
        with self._access():
            return self._cursor.copy(
                self._guard(query, operation="copy"),
                *args,
                **kwargs,
            )

    def fetchone(self) -> Any:
        with self._access():
            return self._cursor.fetchone()

    def fetchmany(self, size: int | None = None) -> Any:
        with self._access():
            if size is None:
                return self._cursor.fetchmany()
            return self._cursor.fetchmany(size)

    def fetchall(self) -> Any:
        with self._access():
            return self._cursor.fetchall()

    def close(self) -> None:
        with self._access():
            try:
                self._cursor.close()
            finally:
                self._active = False

    def __iter__(self) -> Any:
        return self

    def __next__(self) -> Any:
        with self._access():
            return next(self._cursor)

    def __enter__(self) -> _GuardedSyncCursor:
        with self._access():
            self._cursor.__enter__()
            return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
        with self._access():
            try:
                return self._cursor.__exit__(exc_type, exc, traceback)
            finally:
                self._active = False


class _GuardedSyncConnection:
    def __init__(
        self,
        connection: Any,
        *,
        owner: _GuardedAuthNZBackend,
        identity: object | None = None,
    ) -> None:
        self._connection = connection
        self._owner = owner
        self._dialect = owner._dialect
        self._authnz_profile_user_backend = self._dialect
        self._lease = _SyncLeaseState(identity=identity)

    @property
    def _authnz_profile_user_guard_identity(self) -> object:
        with self._lease.access():
            return self._lease.identity

    def _retire(self) -> None:
        self._lease.retire()

    def _guard(self, query: Any, *, operation: str) -> str:
        return _guard_sql(
            query,
            backend=self._dialect,
            connection_identity=self._lease.identity,
            operation=operation,
        )

    def cursor(self, *args: Any, **kwargs: Any) -> _GuardedSyncCursor:
        with self._lease.access():
            return _GuardedSyncCursor(
                self._connection.cursor(*args, **kwargs),
                connection=self,
                dialect=self._dialect,
            )

    def execute(self, query: Any, parameters: Any = None) -> _GuardedSyncCursor:
        with self._lease.access():
            guarded = self._guard(query, operation="execute")
            cursor = _GuardedSyncCursor(
                self._connection.cursor(),
                connection=self,
                dialect=self._dialect,
            )
            return cursor._execute_guarded(guarded, parameters)

    def executemany(self, query: Any, parameters: Any) -> _GuardedSyncCursor:
        with self._lease.access():
            guarded = self._guard(query, operation="executemany")
            cursor = _GuardedSyncCursor(
                self._connection.cursor(),
                connection=self,
                dialect=self._dialect,
            )
            cursor._cursor.executemany(guarded, parameters)
            return cursor

    def executescript(self, query: Any) -> _GuardedSyncCursor:
        with self._lease.access():
            guarded = self._guard(query, operation="executescript")
            cursor = _GuardedSyncCursor(
                self._connection.cursor(),
                connection=self,
                dialect=self._dialect,
            )
            cursor._cursor.executescript(guarded)
            return cursor

    def commit(self) -> None:
        with self._lease.access():
            self._connection.commit()

    def rollback(self) -> None:
        with self._lease.access():
            self._connection.rollback()

    def close(self) -> None:
        with self._lease.access():
            try:
                self._connection.close()
            finally:
                self._retire()

    @property
    def in_transaction(self) -> bool:
        with self._lease.access():
            return bool(getattr(self._connection, "in_transaction", False))

    @property
    def closed(self) -> Any:
        with self._lease.access():
            return getattr(self._connection, "closed", False)

    def __enter__(self) -> _GuardedSyncConnection:
        with self._lease.access():
            self._connection.__enter__()
            return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
        with self._lease.access():
            try:
                return self._connection.__exit__(exc_type, exc, traceback)
            finally:
                self._retire()


class _GuardedSyncPool:
    def __init__(self, pool: Any, *, owner: _GuardedAuthNZBackend) -> None:
        self._pool = pool
        self._owner = owner

    def _wrap(self, connection: Any) -> _GuardedSyncConnection:
        return _GuardedSyncConnection(connection, owner=self._owner)

    def get_connection(self) -> _GuardedSyncConnection:
        return self._wrap(self._pool.get_connection())

    def return_connection(self, connection: _GuardedSyncConnection) -> None:
        if type(connection) is not _GuardedSyncConnection or connection._owner is not self._owner:
            raise RuntimeError("Connection does not belong to this managed AuthNZ pool")
        with connection._lease.release():
            self._pool.return_connection(connection._connection)

    @contextmanager
    def connection(self) -> Generator[_GuardedSyncConnection, None, None]:
        with self._pool.connection() as connection:
            wrapper = self._wrap(connection)
            try:
                yield wrapper
            finally:
                with wrapper._lease.release():
                    pass

    def close_all(self) -> None:
        self._pool.close_all()

    def get_stats(self) -> dict[str, Any]:
        return self._pool.get_stats()

    def clear_thread_local_connection(self) -> None:
        self._pool.clear_thread_local_connection()


class _GuardedAuthNZBackend:
    """Composition guard retaining the factory-owned backend lifecycle."""

    def __init__(self, backend: Any) -> None:
        self._authnz_wrapped_backend = backend
        self._dialect = _backend_dialect(backend)
        self._authnz_profile_user_backend = self._dialect
        self._identity = object()
        self._pool: _GuardedSyncPool | None = None

    @property
    def backend_type(self) -> Any:
        return self._authnz_wrapped_backend.backend_type

    @property
    def config(self) -> Any:
        return self._authnz_wrapped_backend.config

    @property
    def features(self) -> Any:
        return self._authnz_wrapped_backend.features

    @contextmanager
    def _connection_parts(
        self,
        connection: Any,
    ) -> Generator[tuple[Any, object], None, None]:
        if connection is None:
            yield None, self._identity
            return
        if type(connection) is _GuardedSyncConnection and connection._owner is self:
            with connection._lease.access():
                yield connection._connection, connection._lease.identity
            return
        yield connection, connection

    def execute(
        self,
        query: Any,
        params: Any = None,
        connection: Any = None,
    ) -> Any:
        with self._connection_parts(connection) as (raw_connection, identity):
            guarded = _guard_sql(
                query,
                backend=self._dialect,
                connection_identity=identity,
                operation="execute",
            )
            return self._authnz_wrapped_backend.execute(
                guarded,
                params,
                connection=raw_connection,
            )

    def execute_many(
        self,
        query: Any,
        params_list: Any,
        connection: Any = None,
    ) -> Any:
        with self._connection_parts(connection) as (raw_connection, identity):
            guarded = _guard_sql(
                query,
                backend=self._dialect,
                connection_identity=identity,
                operation="execute_many",
            )
            return self._authnz_wrapped_backend.execute_many(
                guarded,
                params_list,
                connection=raw_connection,
            )

    @contextmanager
    def transaction(self, connection: Any = None) -> Generator[_GuardedSyncConnection, None, None]:
        with self._connection_parts(connection) as (raw_connection, identity):
            with self._authnz_wrapped_backend.transaction(
                connection=raw_connection
            ) as transaction_connection:
                if (
                    type(connection) is _GuardedSyncConnection
                    and transaction_connection is raw_connection
                ):
                    yield connection
                else:
                    wrapper = _GuardedSyncConnection(
                        transaction_connection,
                        owner=self,
                        identity=identity if connection is not None else None,
                    )
                    try:
                        yield wrapper
                    finally:
                        with wrapper._lease.release():
                            pass

    def connect(self) -> _GuardedSyncConnection:
        return _GuardedSyncConnection(
            self._authnz_wrapped_backend.connect(),
            owner=self,
        )

    def disconnect(self, connection: _GuardedSyncConnection) -> None:
        if type(connection) is not _GuardedSyncConnection or connection._owner is not self:
            raise RuntimeError("Connection does not belong to this managed AuthNZ backend")
        with connection._lease.release():
            self._authnz_wrapped_backend.disconnect(connection._connection)

    def get_pool(self) -> _GuardedSyncPool:
        raw_pool = self._authnz_wrapped_backend.get_pool()
        if self._pool is None or self._pool._pool is not raw_pool:
            self._pool = _GuardedSyncPool(raw_pool, owner=self)
        return self._pool


def _guard_authnz_sync_backend(backend: Any) -> _GuardedAuthNZBackend:
    if type(backend) is _GuardedAuthNZBackend:
        return backend
    return _GuardedAuthNZBackend(backend)
