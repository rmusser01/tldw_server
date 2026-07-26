from __future__ import annotations

import asyncio
import inspect
import io
import traceback
from contextlib import asynccontextmanager
from typing import Any

import pytest
from loguru import logger

import tldw_Server_API.app.core.AuthNZ.exceptions as authnz_exceptions
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    AuthnzMonitoringError,
    TransactionError,
    UserNotFoundError,
)

pytestmark = pytest.mark.unit

RollbackSignal = getattr(authnz_exceptions, "RollbackSignal", Exception)
DatabaseConcurrencyConflict = getattr(
    authnz_exceptions,
    "DatabaseConcurrencyConflict",
    type("MissingDatabaseConcurrencyConflict", (Exception,), {}),
)

_RAW_BACKEND_TEXT = "secret=/tmp/authnz-users.db token=postgres-secret"
_ROLLBACK_CLEANUP_TEXT = "rollback secret=/tmp/rollback.db token=rollback-secret"
_RELEASE_CLEANUP_TEXT = "release secret=/tmp/release.db token=release-secret"


class TestRollback(RollbackSignal):
    pass


class _CleanupBarrier:
    def __init__(self, *, exit_error: BaseException | None = None) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.completed = False
        self._exit_error = exit_error

    async def run(self) -> None:
        self.started.set()
        await self.release.wait()
        self.completed = True
        if self._exit_error is not None:
            raise self._exit_error


async def _event_loop_turn() -> None:
    reached = asyncio.Event()
    asyncio.get_running_loop().call_soon(reached.set)
    await reached.wait()


async def _capture_task_failure(operation: Any) -> BaseException:
    async def _capture() -> BaseException:
        try:
            await operation
        except BaseException as exc:  # noqa: BLE001 - safely capture control flow in-task
            return exc
        raise AssertionError("transaction unexpectedly succeeded")

    return await asyncio.create_task(_capture())


class _SqlstateError(RuntimeError):
    def __init__(self, sqlstate: str) -> None:
        super().__init__(_RAW_BACKEND_TEXT)
        self.sqlstate = sqlstate


class _AcquireContext:
    def __init__(
        self,
        conn: _PostgresConnection,
        events: list[str],
        exit_error: BaseException | None,
        exit_probe: _CleanupBarrier | None,
    ) -> None:
        self._conn = conn
        self._events = events
        self._exit_error = exit_error
        self._exit_probe = exit_probe

    async def __aenter__(self) -> _PostgresConnection:
        self._events.append("acquired")
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ARG002
        self._events.append("released")
        if self._exit_probe is not None:
            await self._exit_probe.run()
        if self._exit_error is not None:
            raise self._exit_error
        return False


class _TransactionContext:
    def __init__(
        self,
        events: list[str],
        exit_error: BaseException | None,
        rollback_exit_error: BaseException | None,
        rollback_probe: _CleanupBarrier | None,
    ) -> None:
        self._events = events
        self._exit_error = exit_error
        self._rollback_exit_error = rollback_exit_error
        self._rollback_probe = rollback_probe

    async def __aenter__(self) -> None:
        self._events.append("transaction_entered")

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ARG002
        self._events.append("transaction_exited")
        if self._exit_error is not None and exc_type is None:
            raise self._exit_error
        if self._rollback_probe is not None and exc_type is not None:
            await self._rollback_probe.run()
        if self._rollback_exit_error is not None and exc_type is not None:
            raise self._rollback_exit_error
        return False


class _PostgresConnection:
    def __init__(
        self,
        events: list[str],
        *,
        statement_error: BaseException | None = None,
        exit_error: BaseException | None = None,
        rollback_exit_error: BaseException | None = None,
        rollback_probe: _CleanupBarrier | None = None,
    ) -> None:
        self._events = events
        self._statement_error = statement_error
        self._exit_error = exit_error
        self._rollback_exit_error = rollback_exit_error
        self._rollback_probe = rollback_probe

    def transaction(self) -> _TransactionContext:
        return _TransactionContext(
            self._events,
            self._exit_error,
            self._rollback_exit_error,
            self._rollback_probe,
        )

    async def execute(self, _query: str) -> None:
        if self._statement_error is not None:
            raise self._statement_error


class _PostgresPool:
    def __init__(
        self,
        *,
        statement_error: BaseException | None = None,
        exit_error: BaseException | None = None,
        rollback_exit_error: BaseException | None = None,
        release_error: BaseException | None = None,
        rollback_probe: _CleanupBarrier | None = None,
        release_probe: _CleanupBarrier | None = None,
    ) -> None:
        self.events: list[str] = []
        self.acquire_timeouts: list[float | None] = []
        self._release_error = release_error
        self._release_probe = release_probe
        self.conn = _PostgresConnection(
            self.events,
            statement_error=statement_error,
            exit_error=exit_error,
            rollback_exit_error=rollback_exit_error,
            rollback_probe=rollback_probe,
        )

    def acquire(self, *, timeout: float | None = None) -> _AcquireContext:
        self.acquire_timeouts.append(timeout)
        return _AcquireContext(
            self.conn,
            self.events,
            self._release_error,
            self._release_probe,
        )


def _database_pool(backend: Any) -> DatabasePool:
    db_pool = DatabasePool.__new__(DatabasePool)
    db_pool._initialized = True
    db_pool.pool = backend
    db_pool.db_path = ":memory:"
    db_pool._sqlite_uri = False
    return db_pool


def _nested_sqlstate_error(sqlstate: str) -> RuntimeError:
    try:
        raise _SqlstateError(sqlstate)
    except _SqlstateError as conflict:
        try:
            raise RuntimeError(_RAW_BACKEND_TEXT) from conflict
        except RuntimeError as wrapper:
            return wrapper


def _natural_sqlstate_error(sqlstate: str) -> RuntimeError:
    try:
        raise _SqlstateError(sqlstate)
    except _SqlstateError:
        try:
            raise RuntimeError(_RAW_BACKEND_TEXT)
        except RuntimeError as wrapper:
            return wrapper


def _suppressed_sqlstate_context(sqlstate: str) -> RuntimeError:
    try:
        raise _SqlstateError(sqlstate)
    except _SqlstateError:
        try:
            raise RuntimeError(_RAW_BACKEND_TEXT) from None
        except RuntimeError as wrapper:
            return wrapper


def _hidden_sqlstate_context_behind_explicit_cause(sqlstate: str) -> RuntimeError:
    explicit_cause = OSError(_RAW_BACKEND_TEXT)
    try:
        raise _SqlstateError(sqlstate)
    except _SqlstateError:
        try:
            raise RuntimeError(_RAW_BACKEND_TEXT) from explicit_cause
        except RuntimeError as wrapper:
            return wrapper


def test_transaction_signal_types_are_public_and_sanitized() -> None:
    assert hasattr(authnz_exceptions, "RollbackSignal")
    assert hasattr(authnz_exceptions, "DatabaseConcurrencyConflict")
    assert issubclass(authnz_exceptions.RollbackSignal, Exception)
    assert issubclass(
        authnz_exceptions.DatabaseConcurrencyConflict,
        authnz_exceptions.DatabaseError,
    )
    assert _RAW_BACKEND_TEXT not in str(authnz_exceptions.DatabaseConcurrencyConflict())


def test_transaction_is_the_only_public_transaction_context_api() -> None:
    signature = inspect.signature(DatabasePool.transaction)
    parameter = signature.parameters["acquire_timeout_seconds"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert hasattr(DatabasePool, "_transaction_context")


@pytest.mark.asyncio
async def test_transaction_delegates_to_private_backend_context() -> None:
    pool = _database_pool(None)
    observed: list[float | None] = []
    sentinel = object()

    @asynccontextmanager
    async def _transaction_context(timeout: float | None):
        observed.append(timeout)
        yield sentinel

    pool._transaction_context = _transaction_context  # type: ignore[method-assign]

    async with pool.transaction(acquire_timeout_seconds=1.25) as conn:
        assert conn is sentinel

    assert observed == [1.25]


@pytest.mark.asyncio
async def test_transaction_rethrows_rollback_signal_unchanged_without_logs() -> None:
    backend = _PostgresPool()
    pool = _database_pool(backend)
    signal = TestRollback()
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")
    try:
        with pytest.raises(TestRollback) as raised:
            async with pool.transaction(acquire_timeout_seconds=5.0):
                raise signal
    finally:
        logger.remove(sink_id)

    assert raised.value is signal
    assert raised.value.__cause__ is None
    assert backend.events == ["acquired", "transaction_entered", "transaction_exited", "released"]
    assert sink.getvalue() == ""


@pytest.mark.asyncio
async def test_postgres_rollback_failure_does_not_mask_rollback_signal() -> None:
    backend = _PostgresPool(
        rollback_exit_error=RuntimeError(_ROLLBACK_CLEANUP_TEXT),
    )
    pool = _database_pool(backend)
    signal = TestRollback()
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")
    try:
        with pytest.raises(TestRollback) as raised:
            async with pool.transaction():
                raise signal
    finally:
        logger.remove(sink_id)

    assert raised.value is signal
    assert raised.value.__cause__ is None
    assert backend.events == ["acquired", "transaction_entered", "transaction_exited", "released"]
    assert sink.getvalue() == ""


@pytest.mark.asyncio
async def test_postgres_rollback_failure_does_not_mask_cancellation() -> None:
    backend = _PostgresPool(
        rollback_exit_error=RuntimeError(_ROLLBACK_CLEANUP_TEXT),
    )
    pool = _database_pool(backend)
    cancellation = asyncio.CancelledError()
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")
    try:
        with pytest.raises(asyncio.CancelledError) as raised:
            async with pool.transaction():
                raise cancellation
    finally:
        logger.remove(sink_id)

    assert raised.value is cancellation
    assert raised.value.__cause__ is None
    assert backend.events == ["acquired", "transaction_entered", "transaction_exited", "released"]
    assert sink.getvalue() == ""


@pytest.mark.asyncio
async def test_postgres_release_failure_does_not_mask_rollback_signal() -> None:
    backend = _PostgresPool(release_error=RuntimeError(_RELEASE_CLEANUP_TEXT))
    pool = _database_pool(backend)
    signal = TestRollback()
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")
    try:
        with pytest.raises(TestRollback) as raised:
            async with pool.transaction():
                raise signal
    finally:
        logger.remove(sink_id)

    assert raised.value is signal
    assert raised.value.__cause__ is None
    assert backend.events == ["acquired", "transaction_entered", "transaction_exited", "released"]
    assert sink.getvalue() == ""


@pytest.mark.asyncio
async def test_postgres_cancellation_finishes_rollback_then_release() -> None:
    rollback_probe = _CleanupBarrier(
        exit_error=RuntimeError(_ROLLBACK_CLEANUP_TEXT),
    )
    release_probe = _CleanupBarrier()
    backend = _PostgresPool(
        rollback_probe=rollback_probe,
        release_probe=release_probe,
    )
    pool = _database_pool(backend)
    signal = TestRollback()
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")

    async def _run_transaction() -> None:
        async with pool.transaction():
            raise signal

    task = asyncio.create_task(_run_transaction())
    try:
        await rollback_probe.started.wait()
        task.cancel("first-cancellation")
        await _event_loop_turn()
        assert task.done() is False
        assert rollback_probe.completed is False
        assert release_probe.started.is_set() is False

        task.cancel("repeated-during-rollback")
        await _event_loop_turn()
        assert task.done() is False
        assert rollback_probe.completed is False

        rollback_probe.release.set()
        await release_probe.started.wait()
        assert rollback_probe.completed is True
        assert task.done() is False
        assert release_probe.completed is False

        task.cancel("repeated-during-release")
        await _event_loop_turn()
        assert task.done() is False
        assert release_probe.completed is False

        release_probe.release.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
    finally:
        logger.remove(sink_id)

    assert raised.value.args == ("first-cancellation",)
    assert raised.value.__cause__ is None
    assert rollback_probe.completed is True
    assert release_probe.completed is True
    assert backend.events == ["acquired", "transaction_entered", "transaction_exited", "released"]
    assert sink.getvalue() == ""


@pytest.mark.asyncio
async def test_postgres_cancellation_waits_for_release_after_success() -> None:
    release_probe = _CleanupBarrier()
    backend = _PostgresPool(release_probe=release_probe)
    pool = _database_pool(backend)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")

    async def _run_transaction() -> None:
        async with pool.transaction():
            pass

    task = asyncio.create_task(_run_transaction())
    try:
        await release_probe.started.wait()
        task.cancel("first-cancellation")
        await _event_loop_turn()
        assert task.done() is False
        assert release_probe.completed is False

        task.cancel("repeated-cancellation")
        await _event_loop_turn()
        assert task.done() is False
        assert release_probe.completed is False

        release_probe.release.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
    finally:
        logger.remove(sink_id)

    assert raised.value.args == ("first-cancellation",)
    assert raised.value.__cause__ is None
    assert release_probe.completed is True
    assert backend.events == ["acquired", "transaction_entered", "transaction_exited", "released"]
    assert sink.getvalue() == ""


@pytest.mark.asyncio
async def test_postgres_release_cancellation_replaces_ordinary_primary() -> None:
    cleanup_cancellation = asyncio.CancelledError()
    backend = _PostgresPool(release_error=cleanup_cancellation)
    pool = _database_pool(backend)
    primary = RuntimeError(_RAW_BACKEND_TEXT)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")
    try:
        with pytest.raises(asyncio.CancelledError) as raised:
            async with pool.transaction():
                raise primary
    finally:
        logger.remove(sink_id)

    assert raised.value is cleanup_cancellation
    assert raised.value.__cause__ is None
    assert backend.events == ["acquired", "transaction_entered", "transaction_exited", "released"]
    assert sink.getvalue() == ""


@pytest.mark.asyncio
async def test_postgres_cleanup_cancellation_preserves_existing_cancellation_identity() -> None:
    body_cancellation = asyncio.CancelledError()
    backend = _PostgresPool(
        rollback_exit_error=asyncio.CancelledError(),
        release_error=asyncio.CancelledError(),
    )
    pool = _database_pool(backend)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")
    try:
        with pytest.raises(asyncio.CancelledError) as raised:
            async with pool.transaction():
                raise body_cancellation
    finally:
        logger.remove(sink_id)

    assert raised.value is body_cancellation
    assert raised.value.__cause__ is None
    assert backend.events == ["acquired", "transaction_entered", "transaction_exited", "released"]
    assert sink.getvalue() == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
@pytest.mark.parametrize("primary_kind", ["ordinary", "trusted"])
async def test_postgres_cleanup_control_replaces_exception_primary(
    cleanup_type: type[BaseException],
    primary_kind: str,
) -> None:
    cleanup = cleanup_type("cleanup-control")
    primary = (
        RuntimeError(_RAW_BACKEND_TEXT)
        if primary_kind == "ordinary"
        else UserNotFoundError("user-42")
    )
    backend = _PostgresPool(rollback_exit_error=cleanup)
    pool = _database_pool(backend)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")

    async def _run_transaction() -> None:
        async with pool.transaction():
            raise primary

    try:
        raised = await _capture_task_failure(_run_transaction())
    finally:
        logger.remove(sink_id)

    assert raised is cleanup
    assert raised.__cause__ is None
    assert backend.events == ["acquired", "transaction_entered", "transaction_exited", "released"]
    assert sink.getvalue() == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("primary_type", "cleanup_type"),
    [
        (KeyboardInterrupt, SystemExit),
        (SystemExit, GeneratorExit),
        (GeneratorExit, KeyboardInterrupt),
    ],
)
async def test_postgres_existing_control_primary_wins_over_cleanup_control(
    primary_type: type[BaseException],
    cleanup_type: type[BaseException],
) -> None:
    primary = primary_type("primary-control")
    cleanup = cleanup_type("cleanup-control")
    backend = _PostgresPool(rollback_exit_error=cleanup)
    pool = _database_pool(backend)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")

    async def _run_transaction() -> None:
        async with pool.transaction():
            raise primary

    try:
        raised = await _capture_task_failure(_run_transaction())
    finally:
        logger.remove(sink_id)

    assert raised is primary
    assert raised.__cause__ is None
    assert backend.events == ["acquired", "transaction_entered", "transaction_exited", "released"]
    assert sink.getvalue() == ""


@pytest.mark.asyncio
async def test_postgres_release_failure_without_primary_is_sanitized() -> None:
    backend = _PostgresPool(release_error=RuntimeError(_RELEASE_CLEANUP_TEXT))
    pool = _database_pool(backend)
    sink = io.StringIO()
    sink_id = logger.add(sink, format="{message} {extra}")
    try:
        with pytest.raises(TransactionError) as raised:
            async with pool.transaction():
                pass
    finally:
        logger.remove(sink_id)

    rendered = "".join(traceback.format_exception(raised.value))
    logs = sink.getvalue()
    assert backend.events == ["acquired", "transaction_entered", "transaction_exited", "released"]
    assert raised.value.__cause__ is None
    assert _RELEASE_CLEANUP_TEXT not in str(raised.value)
    assert _RELEASE_CLEANUP_TEXT not in rendered
    assert _RELEASE_CLEANUP_TEXT not in logs
    assert "'backend': 'postgresql'" in logs
    assert "'operation': 'release'" in logs
    assert "'error_type': 'RuntimeError'" in logs


@pytest.mark.asyncio
@pytest.mark.parametrize("sqlstate", ["40P01", "40001"])
async def test_postgres_release_sqlstate_after_commit_is_not_concurrency_conflict(
    sqlstate: str,
) -> None:
    backend = _PostgresPool(release_error=_SqlstateError(sqlstate))
    pool = _database_pool(backend)
    sink = io.StringIO()
    sink_id = logger.add(sink, format="{message} {extra}")
    try:
        with pytest.raises(TransactionError) as raised:
            async with pool.transaction():
                pass
    finally:
        logger.remove(sink_id)

    rendered = "".join(traceback.format_exception(raised.value))
    logs = sink.getvalue()
    assert backend.events == ["acquired", "transaction_entered", "transaction_exited", "released"]
    assert raised.value.__cause__ is None
    assert _RAW_BACKEND_TEXT not in str(raised.value)
    assert _RAW_BACKEND_TEXT not in rendered
    assert _RAW_BACKEND_TEXT not in logs
    assert "'backend': 'postgresql'" in logs
    assert "'operation': 'release'" in logs
    assert "'error_type': '_SqlstateError'" in logs


@pytest.mark.asyncio
async def test_postgres_body_cannot_inject_concurrency_translation() -> None:
    backend = _PostgresPool()
    pool = _database_pool(backend)
    conflict = DatabaseConcurrencyConflict()

    with pytest.raises(TransactionError) as raised:
        async with pool.transaction():
            raise conflict

    assert raised.value is not conflict
    assert raised.value.__cause__ is None
    assert backend.events == ["acquired", "transaction_entered", "transaction_exited", "released"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "domain_error",
    [
        UserNotFoundError("user-42"),
        AuthnzMonitoringError("write_metric"),
    ],
    ids=["user-not-found", "established-database-error"],
)
async def test_postgres_domain_exceptions_propagate_unchanged_without_logs(
    domain_error: Exception,
) -> None:
    backend = _PostgresPool()
    pool = _database_pool(backend)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")
    try:
        with pytest.raises(type(domain_error)) as raised:
            async with pool.transaction():
                raise domain_error
    finally:
        logger.remove(sink_id)

    assert raised.value is domain_error
    assert raised.value.__cause__ is None
    assert backend.events == ["acquired", "transaction_entered", "transaction_exited", "released"]
    assert sink.getvalue() == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("sqlstate", ["40P01", "40001"])
@pytest.mark.parametrize("failure_point", ["statement", "commit"])
async def test_postgres_concurrency_sqlstate_is_sanitized_after_transaction_exit(
    sqlstate: str,
    failure_point: str,
) -> None:
    raw_error = _nested_sqlstate_error(sqlstate)
    backend = _PostgresPool(
        statement_error=raw_error if failure_point == "statement" else None,
        exit_error=raw_error if failure_point == "commit" else None,
    )
    pool = _database_pool(backend)

    with pytest.raises(DatabaseConcurrencyConflict) as raised:
        async with pool.transaction() as conn:
            await conn.execute("SELECT 1")

    rendered = "".join(traceback.format_exception(raised.value))
    assert backend.events == ["acquired", "transaction_entered", "transaction_exited", "released"]
    assert raised.value.__cause__ is None
    assert _RAW_BACKEND_TEXT not in str(raised.value)
    assert _RAW_BACKEND_TEXT not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("sqlstate", ["40P01", "40001"])
async def test_postgres_statement_sqlstate_wins_over_rollback_and_release_failures(
    sqlstate: str,
) -> None:
    backend = _PostgresPool(
        statement_error=_nested_sqlstate_error(sqlstate),
        rollback_exit_error=RuntimeError(_ROLLBACK_CLEANUP_TEXT),
        release_error=RuntimeError(_RELEASE_CLEANUP_TEXT),
    )
    pool = _database_pool(backend)
    sink = io.StringIO()
    sink_id = logger.add(sink, format="{message} {extra}")
    try:
        with pytest.raises(DatabaseConcurrencyConflict) as raised:
            async with pool.transaction() as conn:
                await conn.execute("SELECT 1")
    finally:
        logger.remove(sink_id)

    rendered = "".join(traceback.format_exception(raised.value))
    logs = sink.getvalue()
    assert backend.events == ["acquired", "transaction_entered", "transaction_exited", "released"]
    assert raised.value.__cause__ is None
    assert _RAW_BACKEND_TEXT not in rendered
    assert _ROLLBACK_CLEANUP_TEXT not in rendered
    assert _RELEASE_CLEANUP_TEXT not in rendered
    assert _RAW_BACKEND_TEXT not in logs
    assert _ROLLBACK_CLEANUP_TEXT not in logs
    assert _RELEASE_CLEANUP_TEXT not in logs
    assert "'backend': 'postgresql'" in logs
    assert "'operation': 'rollback'" in logs
    assert "'operation': 'release'" in logs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("statement_error", "expected_error"),
    [
        (_suppressed_sqlstate_context("40P01"), TransactionError),
        (_hidden_sqlstate_context_behind_explicit_cause("40001"), TransactionError),
        (_nested_sqlstate_error("40P01"), DatabaseConcurrencyConflict),
        (_natural_sqlstate_error("40001"), DatabaseConcurrencyConflict),
    ],
    ids=[
        "suppressed-context",
        "explicit-non-conflict-cause",
        "explicit-conflict-cause",
        "natural-conflict-context",
    ],
)
async def test_postgres_sqlstate_uses_python_effective_exception_chain(
    statement_error: BaseException,
    expected_error: type[BaseException],
) -> None:
    backend = _PostgresPool(statement_error=statement_error)
    pool = _database_pool(backend)

    with pytest.raises(expected_error) as raised:
        async with pool.transaction() as conn:
            await conn.execute("SELECT 1")

    rendered = "".join(traceback.format_exception(raised.value))
    assert raised.value.__cause__ is None
    assert backend.events == ["acquired", "transaction_entered", "transaction_exited", "released"]
    assert _RAW_BACKEND_TEXT not in rendered


@pytest.mark.asyncio
async def test_postgres_ordinary_failure_is_sanitized_without_raw_chain() -> None:
    backend = _PostgresPool(statement_error=RuntimeError(_RAW_BACKEND_TEXT))
    pool = _database_pool(backend)

    with pytest.raises(TransactionError) as raised:
        async with pool.transaction() as conn:
            await conn.execute("SELECT 1")

    rendered = "".join(traceback.format_exception(raised.value))
    assert raised.value.__cause__ is None
    assert _RAW_BACKEND_TEXT not in str(raised.value)
    assert _RAW_BACKEND_TEXT not in rendered


@pytest.mark.asyncio
async def test_postgres_cancellation_in_body_propagates_after_exit() -> None:
    backend = _PostgresPool()
    pool = _database_pool(backend)
    cancelled = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError) as raised:
        async with pool.transaction():
            raise cancelled

    assert raised.value is cancelled
    assert backend.events == ["acquired", "transaction_entered", "transaction_exited", "released"]
