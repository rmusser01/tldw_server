import asyncio
import io
import traceback
from types import SimpleNamespace

import pytest
from loguru import logger

import tldw_Server_API.app.core.AuthNZ.database as database_mod
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    AuthnzMonitoringError,
    DatabaseConcurrencyConflict,
    RollbackSignal,
    TransactionError,
    UserNotFoundError,
)

_RAW_BACKEND_TEXT = "sqlite secret=/tmp/authnz-users.db token=sqlite-secret"


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


async def _capture_task_failure(operation) -> BaseException:
    async def _capture() -> BaseException:
        try:
            await operation
        except BaseException as exc:  # noqa: BLE001 - safely capture control flow in-task
            return exc
        raise AssertionError("transaction unexpectedly succeeded")

    return await asyncio.create_task(_capture())


class _RecordingAsyncConnection:
    def __init__(
        self,
        *,
        commit_error: BaseException | None = None,
        rollback_error: BaseException | None = None,
        close_error: BaseException | None = None,
        rollback_probe: _CleanupBarrier | None = None,
        close_probe: _CleanupBarrier | None = None,
    ) -> None:
        self.statements: list[str] = []
        self.row_factory = None
        self.closed = False
        self.committed = False
        self.rolled_back = False
        self.rollback_calls = 0
        self.close_calls = 0
        self._commit_error = commit_error
        self._rollback_error = rollback_error
        self._close_error = close_error
        self._rollback_probe = rollback_probe
        self._close_probe = close_probe

    async def execute(self, sql: str, *args):
        self.statements.append(sql)
        return _RecordingAsyncCursor(sql)

    async def commit(self) -> None:
        if self._commit_error is not None:
            raise self._commit_error
        self.committed = True

    async def rollback(self) -> None:
        self.rollback_calls += 1
        if self._rollback_probe is not None:
            await self._rollback_probe.run()
        if self._rollback_error is not None:
            raise self._rollback_error
        self.rolled_back = True

    async def close(self) -> None:
        self.close_calls += 1
        if self._close_probe is not None:
            await self._close_probe.run()
        if self._close_error is not None:
            raise self._close_error
        self.closed = True


class _RecordingAsyncCursor:
    def __init__(self, sql: str) -> None:
        self.sql = sql

    async def fetchall(self):
        if self.sql.strip().upper() == "PRAGMA DATABASE_LIST":
            return [(0, "main", "")]
        return []


@pytest.mark.asyncio
async def test_sqlite_transaction_uses_begin_immediate(monkeypatch):
    conn = _RecordingAsyncConnection()

    async def _fake_connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(database_mod.aiosqlite, "connect", _fake_connect)

    pool = DatabasePool(settings=SimpleNamespace())
    pool._initialized = True
    pool.pool = None
    pool.db_path = ":memory:"
    pool._sqlite_uri = False

    async with pool.transaction():
        pass

    assert conn.statements[:6] == [
        "PRAGMA database_list",
        "PRAGMA synchronous=NORMAL",
        "PRAGMA foreign_keys=ON",
        "PRAGMA busy_timeout=5000",
        "PRAGMA temp_store=MEMORY",
        "BEGIN IMMEDIATE",
    ]
    assert conn.committed is True
    assert conn.closed is True


@pytest.mark.asyncio
async def test_sqlite_acquire_applies_runtime_pragmas(tmp_path):
    pool = DatabasePool(settings=SimpleNamespace())
    pool._initialized = True
    pool.pool = None
    pool.db_path = str(tmp_path / "users.db")
    pool._sqlite_uri = False

    async with pool.acquire() as conn:
        journal_mode = await (await conn.execute("PRAGMA journal_mode")).fetchone()
        synchronous = await (await conn.execute("PRAGMA synchronous")).fetchone()
        foreign_keys = await (await conn.execute("PRAGMA foreign_keys")).fetchone()
        busy_timeout = await (await conn.execute("PRAGMA busy_timeout")).fetchone()
        temp_store = await (await conn.execute("PRAGMA temp_store")).fetchone()

    assert str(journal_mode[0]).lower() == "wal"
    assert int(synchronous[0]) == 1
    assert int(foreign_keys[0]) == 1
    assert int(busy_timeout[0]) == 5000
    assert int(temp_store[0]) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("primary_type", [asyncio.CancelledError, KeyboardInterrupt])
async def test_sqlite_acquire_close_runtime_error_does_not_mask_control_primary(
    monkeypatch,
    primary_type: type[BaseException],
):
    primary = primary_type("body-control")
    conn = _RecordingAsyncConnection(close_error=RuntimeError(_RAW_BACKEND_TEXT))

    async def _fake_connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(database_mod.aiosqlite, "connect", _fake_connect)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")

    async def _run_context() -> None:
        async with _sqlite_pool().acquire():
            raise primary

    try:
        raised = await _capture_task_failure(_run_context())
    finally:
        logger.remove(sink_id)

    assert raised is primary
    assert raised.__cause__ is None
    assert conn.close_calls == 1
    assert sink.getvalue() == ""


@pytest.mark.asyncio
async def test_sqlite_acquire_existing_keyboard_interrupt_wins_over_close_system_exit(
    monkeypatch,
):
    primary = KeyboardInterrupt("body-control")
    cleanup = SystemExit("close-control")
    conn = _RecordingAsyncConnection(close_error=cleanup)

    async def _fake_connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(database_mod.aiosqlite, "connect", _fake_connect)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")

    async def _run_context() -> None:
        async with _sqlite_pool().acquire():
            raise primary

    try:
        raised = await _capture_task_failure(_run_context())
    finally:
        logger.remove(sink_id)

    assert raised is primary
    assert raised.__cause__ is None
    assert conn.close_calls == 1
    assert sink.getvalue() == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("primary_kind", ["ordinary", "trusted"])
async def test_sqlite_acquire_close_system_exit_replaces_exception_primary(
    monkeypatch,
    primary_kind: str,
):
    primary = (
        RuntimeError(_RAW_BACKEND_TEXT)
        if primary_kind == "ordinary"
        else UserNotFoundError("user-42")
    )
    cleanup = SystemExit("close-control")
    conn = _RecordingAsyncConnection(close_error=cleanup)

    async def _fake_connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(database_mod.aiosqlite, "connect", _fake_connect)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")

    async def _run_context() -> None:
        async with _sqlite_pool().acquire():
            raise primary

    try:
        raised = await _capture_task_failure(_run_context())
    finally:
        logger.remove(sink_id)

    assert raised is cleanup
    assert raised.__cause__ is None
    assert conn.close_calls == 1
    assert sink.getvalue() == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("primary_kind", ["ordinary", "trusted"])
async def test_sqlite_acquire_ordinary_close_failure_preserves_exception_chain(
    monkeypatch,
    primary_kind: str,
):
    cause = ValueError("body-cause")
    primary = (
        RuntimeError(_RAW_BACKEND_TEXT)
        if primary_kind == "ordinary"
        else UserNotFoundError("user-42")
    )
    primary.__cause__ = cause
    conn = _RecordingAsyncConnection(close_error=RuntimeError(_RAW_BACKEND_TEXT))

    async def _fake_connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(database_mod.aiosqlite, "connect", _fake_connect)
    sink = io.StringIO()
    sink_id = logger.add(sink, format="{message} {extra}")

    async def _run_context() -> None:
        async with _sqlite_pool().acquire():
            raise primary

    try:
        raised = await _capture_task_failure(_run_context())
    finally:
        logger.remove(sink_id)

    logs = sink.getvalue()
    assert raised is primary
    assert raised.__cause__ is cause
    assert conn.close_calls == 1
    assert _RAW_BACKEND_TEXT not in logs
    if primary_kind == "ordinary":
        assert logs.count("SQLite connection close failed") == 1
        assert "'backend': 'sqlite'" in logs
        assert "'operation': 'close'" in logs
        assert "'error_type': 'RuntimeError'" in logs
    else:
        assert logs == ""


@pytest.mark.asyncio
async def test_sqlite_acquire_close_failure_after_success_is_sanitized(monkeypatch):
    conn = _RecordingAsyncConnection(close_error=RuntimeError(_RAW_BACKEND_TEXT))

    async def _fake_connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(database_mod.aiosqlite, "connect", _fake_connect)
    sink = io.StringIO()
    sink_id = logger.add(sink, format="{message} {extra}")
    try:
        with pytest.raises(TransactionError) as raised:
            async with _sqlite_pool().acquire():
                pass
    finally:
        logger.remove(sink_id)

    rendered = "".join(traceback.format_exception(raised.value))
    logs = sink.getvalue()
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None
    assert raised.value.__suppress_context__ is True
    assert _RAW_BACKEND_TEXT not in str(raised.value)
    assert _RAW_BACKEND_TEXT not in rendered
    assert _RAW_BACKEND_TEXT not in logs
    assert logs.count("SQLite connection close failed") == 1
    assert "'backend': 'sqlite'" in logs
    assert "'operation': 'close'" in logs
    assert "'error_type': 'RuntimeError'" in logs
    assert conn.close_calls == 1


@pytest.mark.asyncio
async def test_sqlite_acquire_close_control_after_success_propagates_unchanged(
    monkeypatch,
):
    cleanup = GeneratorExit("close-control")
    conn = _RecordingAsyncConnection(close_error=cleanup)

    async def _fake_connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(database_mod.aiosqlite, "connect", _fake_connect)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")

    async def _run_context() -> None:
        async with _sqlite_pool().acquire():
            pass

    try:
        raised = await _capture_task_failure(_run_context())
    finally:
        logger.remove(sink_id)

    assert raised is cleanup
    assert raised.__cause__ is None
    assert conn.close_calls == 1
    assert sink.getvalue() == ""


@pytest.mark.asyncio
async def test_sqlite_acquire_cancellation_waits_for_close_completion(monkeypatch):
    close_probe = _CleanupBarrier()
    conn = _RecordingAsyncConnection(close_probe=close_probe)

    async def _fake_connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(database_mod.aiosqlite, "connect", _fake_connect)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")

    async def _run_context() -> None:
        async with _sqlite_pool().acquire():
            pass

    task = asyncio.create_task(_run_context())
    try:
        await close_probe.started.wait()
        task.cancel("first-cancellation")
        await _event_loop_turn()
        assert task.done() is False
        assert close_probe.completed is False

        task.cancel("repeated-cancellation")
        await _event_loop_turn()
        assert task.done() is False
        assert close_probe.completed is False

        close_probe.release.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
    finally:
        close_probe.release.set()
        if not task.done():
            await asyncio.gather(task, return_exceptions=True)
        logger.remove(sink_id)

    assert raised.value.args == ("first-cancellation",)
    assert raised.value.__cause__ is None
    assert close_probe.completed is True
    assert conn.closed is True
    assert conn.close_calls == 1
    assert sink.getvalue() == ""


def _sqlite_pool() -> DatabasePool:
    pool = DatabasePool(settings=SimpleNamespace())
    pool._initialized = True
    pool.pool = None
    pool.db_path = ":memory:"
    pool._sqlite_uri = False
    return pool


@pytest.mark.asyncio
async def test_sqlite_body_failure_rolls_back_closes_and_hides_raw_chain(monkeypatch):
    conn = _RecordingAsyncConnection()

    async def _fake_connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(database_mod.aiosqlite, "connect", _fake_connect)

    with pytest.raises(TransactionError) as raised:
        async with _sqlite_pool().transaction():
            raise RuntimeError(_RAW_BACKEND_TEXT)

    rendered = "".join(traceback.format_exception(raised.value))
    assert conn.rolled_back is True
    assert conn.closed is True
    assert raised.value.__cause__ is None
    assert _RAW_BACKEND_TEXT not in str(raised.value)
    assert _RAW_BACKEND_TEXT not in rendered


@pytest.mark.asyncio
async def test_sqlite_commit_failure_rolls_back_and_closes(monkeypatch):
    conn = _RecordingAsyncConnection(commit_error=RuntimeError(_RAW_BACKEND_TEXT))

    async def _fake_connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(database_mod.aiosqlite, "connect", _fake_connect)

    with pytest.raises(TransactionError) as raised:
        async with _sqlite_pool().transaction():
            pass

    assert conn.committed is False
    assert conn.rolled_back is True
    assert conn.closed is True
    assert raised.value.__cause__ is None
    assert _RAW_BACKEND_TEXT not in str(raised.value)


@pytest.mark.asyncio
async def test_sqlite_cancellation_rolls_back_and_closes_before_propagating(monkeypatch):
    conn = _RecordingAsyncConnection()
    cancellation = asyncio.CancelledError()

    async def _fake_connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(database_mod.aiosqlite, "connect", _fake_connect)

    with pytest.raises(asyncio.CancelledError) as raised:
        async with _sqlite_pool().transaction():
            raise cancellation

    assert raised.value is cancellation
    assert conn.rolled_back is True
    assert conn.closed is True


@pytest.mark.asyncio
async def test_sqlite_close_failure_after_success_is_sanitized(monkeypatch):
    conn = _RecordingAsyncConnection(close_error=RuntimeError(_RAW_BACKEND_TEXT))

    async def _fake_connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(database_mod.aiosqlite, "connect", _fake_connect)
    sink = io.StringIO()
    sink_id = logger.add(sink, format="{message} {extra}")
    try:
        with pytest.raises(TransactionError) as raised:
            async with _sqlite_pool().transaction():
                pass
    finally:
        logger.remove(sink_id)

    rendered = "".join(traceback.format_exception(raised.value))
    assert conn.committed is True
    assert conn.close_calls == 1
    assert raised.value.__cause__ is None
    assert _RAW_BACKEND_TEXT not in str(raised.value)
    assert _RAW_BACKEND_TEXT not in rendered
    assert _RAW_BACKEND_TEXT not in sink.getvalue()
    assert "'backend': 'sqlite'" in sink.getvalue()
    assert "'operation': 'close'" in sink.getvalue()
    assert "'error_type': 'RuntimeError'" in sink.getvalue()


class _RollbackForTest(RollbackSignal):
    pass


@pytest.mark.asyncio
async def test_sqlite_cleanup_failures_do_not_mask_rollback_signal(monkeypatch):
    conn = _RecordingAsyncConnection(
        rollback_error=RuntimeError(_RAW_BACKEND_TEXT),
        close_error=RuntimeError(_RAW_BACKEND_TEXT),
    )
    signal = _RollbackForTest()

    async def _fake_connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(database_mod.aiosqlite, "connect", _fake_connect)
    sink = io.StringIO()
    sink_id = logger.add(sink, format="{message} {extra}")
    try:
        with pytest.raises(_RollbackForTest) as raised:
            async with _sqlite_pool().transaction():
                raise signal
    finally:
        logger.remove(sink_id)

    assert raised.value is signal
    assert raised.value.__cause__ is None
    assert conn.rollback_calls == 1
    assert conn.close_calls == 1
    assert sink.getvalue() == ""


@pytest.mark.asyncio
async def test_sqlite_rollback_and_close_failures_do_not_mask_cancellation(monkeypatch):
    conn = _RecordingAsyncConnection(
        rollback_error=RuntimeError(_RAW_BACKEND_TEXT),
        close_error=RuntimeError(_RAW_BACKEND_TEXT),
    )
    cancellation = asyncio.CancelledError()

    async def _fake_connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(database_mod.aiosqlite, "connect", _fake_connect)
    sink = io.StringIO()
    sink_id = logger.add(sink, format="{message} {extra}")
    try:
        with pytest.raises(asyncio.CancelledError) as raised:
            async with _sqlite_pool().transaction():
                raise cancellation
    finally:
        logger.remove(sink_id)

    assert raised.value is cancellation
    assert raised.value.__cause__ is None
    assert conn.rollback_calls == 1
    assert conn.close_calls == 1
    assert _RAW_BACKEND_TEXT not in sink.getvalue()


@pytest.mark.asyncio
async def test_sqlite_cancellation_finishes_rollback_then_close(
    monkeypatch,
):
    rollback_probe = _CleanupBarrier(exit_error=RuntimeError(_RAW_BACKEND_TEXT))
    close_probe = _CleanupBarrier()
    conn = _RecordingAsyncConnection(
        rollback_probe=rollback_probe,
        close_probe=close_probe,
    )
    signal = _RollbackForTest()

    async def _fake_connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(database_mod.aiosqlite, "connect", _fake_connect)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")

    async def _run_transaction() -> None:
        async with _sqlite_pool().transaction():
            raise signal

    task = asyncio.create_task(_run_transaction())
    try:
        await rollback_probe.started.wait()
        task.cancel("first-cancellation")
        await _event_loop_turn()
        assert task.done() is False
        assert rollback_probe.completed is False
        assert close_probe.started.is_set() is False

        task.cancel("repeated-during-rollback")
        await _event_loop_turn()
        assert task.done() is False
        assert rollback_probe.completed is False

        rollback_probe.release.set()
        await close_probe.started.wait()
        assert rollback_probe.completed is True
        assert task.done() is False
        assert close_probe.completed is False

        task.cancel("repeated-during-close")
        await _event_loop_turn()
        assert task.done() is False
        assert close_probe.completed is False

        close_probe.release.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
    finally:
        logger.remove(sink_id)

    assert raised.value.args == ("first-cancellation",)
    assert raised.value.__cause__ is None
    assert rollback_probe.completed is True
    assert close_probe.completed is True
    assert conn.rollback_calls == 1
    assert conn.close_calls == 1
    assert sink.getvalue() == ""


@pytest.mark.asyncio
async def test_sqlite_cancellation_waits_for_close_after_success(monkeypatch):
    close_probe = _CleanupBarrier()
    conn = _RecordingAsyncConnection(close_probe=close_probe)

    async def _fake_connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(database_mod.aiosqlite, "connect", _fake_connect)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")

    async def _run_transaction() -> None:
        async with _sqlite_pool().transaction():
            pass

    task = asyncio.create_task(_run_transaction())
    try:
        await close_probe.started.wait()
        task.cancel("first-cancellation")
        await _event_loop_turn()
        assert task.done() is False
        assert close_probe.completed is False

        task.cancel("repeated-cancellation")
        await _event_loop_turn()
        assert task.done() is False
        assert close_probe.completed is False

        close_probe.release.set()
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
    finally:
        logger.remove(sink_id)

    assert raised.value.args == ("first-cancellation",)
    assert raised.value.__cause__ is None
    assert close_probe.completed is True
    assert conn.committed is True
    assert conn.rollback_calls == 0
    assert conn.close_calls == 1
    assert sink.getvalue() == ""


@pytest.mark.asyncio
async def test_sqlite_close_cancellation_replaces_trusted_domain_primary(monkeypatch):
    cleanup_cancellation = asyncio.CancelledError()
    conn = _RecordingAsyncConnection(close_error=cleanup_cancellation)
    domain_error = UserNotFoundError("user-42")

    async def _fake_connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(database_mod.aiosqlite, "connect", _fake_connect)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")
    try:
        with pytest.raises(asyncio.CancelledError) as raised:
            async with _sqlite_pool().transaction():
                raise domain_error
    finally:
        logger.remove(sink_id)

    assert raised.value is cleanup_cancellation
    assert raised.value.__cause__ is None
    assert conn.rollback_calls == 1
    assert conn.close_calls == 1
    assert sink.getvalue() == ""


@pytest.mark.asyncio
async def test_sqlite_cleanup_cancellation_preserves_existing_cancellation_identity(
    monkeypatch,
):
    body_cancellation = asyncio.CancelledError()
    conn = _RecordingAsyncConnection(
        rollback_error=asyncio.CancelledError(),
        close_error=asyncio.CancelledError(),
    )

    async def _fake_connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(database_mod.aiosqlite, "connect", _fake_connect)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")
    try:
        with pytest.raises(asyncio.CancelledError) as raised:
            async with _sqlite_pool().transaction():
                raise body_cancellation
    finally:
        logger.remove(sink_id)

    assert raised.value is body_cancellation
    assert raised.value.__cause__ is None
    assert conn.rollback_calls == 1
    assert conn.close_calls == 1
    assert sink.getvalue() == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("cleanup_type", [KeyboardInterrupt, SystemExit, GeneratorExit])
@pytest.mark.parametrize("primary_kind", ["ordinary", "trusted"])
async def test_sqlite_cleanup_control_replaces_exception_primary(
    monkeypatch,
    cleanup_type: type[BaseException],
    primary_kind: str,
):
    cleanup = cleanup_type("cleanup-control")
    primary = (
        RuntimeError(_RAW_BACKEND_TEXT)
        if primary_kind == "ordinary"
        else UserNotFoundError("user-42")
    )
    conn = _RecordingAsyncConnection(rollback_error=cleanup)

    async def _fake_connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(database_mod.aiosqlite, "connect", _fake_connect)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")

    async def _run_transaction() -> None:
        async with _sqlite_pool().transaction():
            raise primary

    try:
        raised = await _capture_task_failure(_run_transaction())
    finally:
        logger.remove(sink_id)

    assert raised is cleanup
    assert raised.__cause__ is None
    assert conn.rollback_calls == 1
    assert conn.close_calls == 1
    assert conn.closed is True
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
async def test_sqlite_existing_control_primary_wins_over_cleanup_control(
    monkeypatch,
    primary_type: type[BaseException],
    cleanup_type: type[BaseException],
):
    primary = primary_type("primary-control")
    cleanup = cleanup_type("cleanup-control")
    conn = _RecordingAsyncConnection(rollback_error=cleanup)

    async def _fake_connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(database_mod.aiosqlite, "connect", _fake_connect)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")

    async def _run_transaction() -> None:
        async with _sqlite_pool().transaction():
            raise primary

    try:
        raised = await _capture_task_failure(_run_transaction())
    finally:
        logger.remove(sink_id)

    assert raised is primary
    assert raised.__cause__ is None
    assert conn.rollback_calls == 1
    assert conn.close_calls == 1
    assert conn.closed is True
    assert sink.getvalue() == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "domain_error",
    [
        UserNotFoundError("user-42"),
        AuthnzMonitoringError("write_metric"),
    ],
    ids=["user-not-found", "established-database-error"],
)
async def test_sqlite_domain_exceptions_propagate_unchanged_without_logs(
    monkeypatch,
    domain_error: Exception,
):
    conn = _RecordingAsyncConnection()

    async def _fake_connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(database_mod.aiosqlite, "connect", _fake_connect)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")
    try:
        with pytest.raises(type(domain_error)) as raised:
            async with _sqlite_pool().transaction():
                raise domain_error
    finally:
        logger.remove(sink_id)

    assert raised.value is domain_error
    assert raised.value.__cause__ is None
    assert conn.rollback_calls == 1
    assert conn.close_calls == 1
    assert sink.getvalue() == ""


@pytest.mark.asyncio
async def test_sqlite_body_cannot_inject_concurrency_translation(monkeypatch):
    conn = _RecordingAsyncConnection()
    conflict = DatabaseConcurrencyConflict()

    async def _fake_connect(*args, **kwargs):
        return conn

    monkeypatch.setattr(database_mod.aiosqlite, "connect", _fake_connect)

    with pytest.raises(TransactionError) as raised:
        async with _sqlite_pool().transaction():
            raise conflict

    assert raised.value is not conflict
    assert raised.value.__cause__ is None
    assert conn.rollback_calls == 1
    assert conn.close_calls == 1
