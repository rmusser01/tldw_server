from __future__ import annotations

import asyncio
import configparser
import io
import math
import traceback
from types import SimpleNamespace
from typing import Any

import asyncpg
import pytest
from loguru import logger

from tldw_Server_API.app.core.AuthNZ import settings as authnz_settings_module
from tldw_Server_API.app.core.AuthNZ.database import DatabasePool
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    ConnectionPoolExhaustedError,
    TransactionError,
    UserNotFoundError,
)

try:
    from tldw_Server_API.app.core.AuthNZ import transaction_policy as transaction_policy_module

    AuthnzTransactionPolicy = transaction_policy_module.AuthnzTransactionPolicy
    get_authnz_transaction_policy = getattr(
        transaction_policy_module,
        "get_authnz_transaction_policy",
        None,
    )
except ImportError:
    AuthnzTransactionPolicy = None  # type: ignore[assignment,misc]
    get_authnz_transaction_policy = None


pytestmark = pytest.mark.unit

_RAW_BACKEND_TEXT = "pool secret=/tmp/authnz-users.db token=acquire-secret"


class _AcquireFailureContext:
    def __init__(self, error: BaseException) -> None:
        self._error = error

    async def __aenter__(self) -> Any:
        raise self._error

    async def __aexit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001, ARG002
        return False


class _AcquireFailurePool:
    def __init__(self, error: BaseException) -> None:
        self._error = error
        self.acquire_timeouts: list[float | None] = []

    def acquire(self, *, timeout: float | None = None) -> _AcquireFailureContext:
        self.acquire_timeouts.append(timeout)
        return _AcquireFailureContext(self._error)


class _ReleaseBarrier:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.completed = False

    async def run(self) -> None:
        self.started.set()
        await self.release.wait()
        self.completed = True


class _DirectAcquirePool:
    def __init__(
        self,
        *,
        acquire_error: BaseException | None = None,
        release_error: BaseException | None = None,
        release_barrier: _ReleaseBarrier | None = None,
    ) -> None:
        self.connection = object()
        self.acquire_timeouts: list[float | None] = []
        self.release_calls: list[tuple[Any, float | None]] = []
        self.released = False
        self._acquire_error = acquire_error
        self._release_error = release_error
        self._release_barrier = release_barrier

    async def acquire(self, *, timeout: float | None = None) -> Any:
        self.acquire_timeouts.append(timeout)
        if self._acquire_error is not None:
            raise self._acquire_error
        return self.connection

    async def release(self, connection: Any, *, timeout: float | None = None) -> None:
        self.release_calls.append((connection, timeout))
        if self._release_barrier is not None:
            await self._release_barrier.run()
        self.released = True
        if self._release_error is not None:
            raise self._release_error


def _database_pool(backend: Any) -> DatabasePool:
    db_pool = DatabasePool.__new__(DatabasePool)
    db_pool._initialized = True
    db_pool.pool = backend
    db_pool._openai_credential_lock_pool = backend
    db_pool.db_path = ":memory:"
    db_pool._sqlite_uri = False
    return db_pool


def _acquisition_context(
    pool: DatabasePool,
    context_name: str,
    *,
    timeout: float = 1.25,
) -> Any:
    if context_name == "main":
        return pool.acquire(timeout=timeout)
    return pool.acquire_openai_credential_lock_connection(timeout=timeout)


async def _capture_task_failure(operation: Any) -> BaseException:
    async def _capture() -> BaseException:
        try:
            await operation
        except BaseException as exc:  # noqa: BLE001 - safely capture control flow in-task
            return exc
        raise AssertionError("acquisition context unexpectedly succeeded")

    return await asyncio.create_task(_capture())


async def _event_loop_turn() -> None:
    reached = asyncio.Event()
    asyncio.get_running_loop().call_soon(reached.set)
    await reached.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "backend_error",
    [
        asyncio.TimeoutError(_RAW_BACKEND_TEXT),
        asyncpg.exceptions.TooManyConnectionsError(_RAW_BACKEND_TEXT),
    ],
    ids=["timeout", "explicit-exhaustion"],
)
async def test_postgres_acquire_failures_map_to_sanitized_pool_exhaustion(
    backend_error: BaseException,
) -> None:
    backend = _AcquireFailurePool(backend_error)
    pool = _database_pool(backend)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")
    try:
        with pytest.raises(ConnectionPoolExhaustedError) as raised:
            async with pool.transaction(acquire_timeout_seconds=2.75):
                pass
    finally:
        logger.remove(sink_id)

    rendered = "".join(traceback.format_exception(raised.value))
    assert backend.acquire_timeouts == [2.75]
    assert raised.value.__cause__ is None
    assert _RAW_BACKEND_TEXT not in str(raised.value)
    assert _RAW_BACKEND_TEXT not in rendered
    assert _RAW_BACKEND_TEXT not in sink.getvalue()


@pytest.mark.asyncio
async def test_postgres_acquire_cancellation_propagates_unchanged() -> None:
    cancellation = asyncio.CancelledError()
    backend = _AcquireFailurePool(cancellation)
    pool = _database_pool(backend)

    with pytest.raises(asyncio.CancelledError) as raised:
        async with pool.transaction(acquire_timeout_seconds=1.5):
            pass

    assert raised.value is cancellation
    assert backend.acquire_timeouts == [1.5]


@pytest.mark.asyncio
@pytest.mark.parametrize("context_name", ["main", "credential-lock"])
@pytest.mark.parametrize("primary_type", [asyncio.CancelledError, KeyboardInterrupt])
async def test_acquisition_release_runtime_error_does_not_mask_control_primary(
    context_name: str,
    primary_type: type[BaseException],
) -> None:
    primary = primary_type("body-control")
    backend = _DirectAcquirePool(release_error=RuntimeError(_RAW_BACKEND_TEXT))
    pool = _database_pool(backend)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")

    async def _run_context() -> None:
        async with _acquisition_context(pool, context_name):
            raise primary

    try:
        raised = await _capture_task_failure(_run_context())
    finally:
        logger.remove(sink_id)

    assert raised is primary
    assert raised.__cause__ is None
    assert backend.acquire_timeouts == [1.25]
    assert backend.release_calls == [(backend.connection, 1.25)]
    assert sink.getvalue() == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("context_name", ["main", "credential-lock"])
async def test_acquisition_existing_keyboard_interrupt_wins_over_release_system_exit(
    context_name: str,
) -> None:
    primary = KeyboardInterrupt("body-control")
    cleanup = SystemExit("release-control")
    backend = _DirectAcquirePool(release_error=cleanup)
    pool = _database_pool(backend)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")

    async def _run_context() -> None:
        async with _acquisition_context(pool, context_name):
            raise primary

    try:
        raised = await _capture_task_failure(_run_context())
    finally:
        logger.remove(sink_id)

    assert raised is primary
    assert raised.__cause__ is None
    assert backend.release_calls == [(backend.connection, 1.25)]
    assert sink.getvalue() == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("context_name", ["main", "credential-lock"])
@pytest.mark.parametrize("primary_kind", ["ordinary", "trusted"])
async def test_acquisition_release_system_exit_replaces_exception_primary(
    context_name: str,
    primary_kind: str,
) -> None:
    primary = (
        RuntimeError(_RAW_BACKEND_TEXT)
        if primary_kind == "ordinary"
        else UserNotFoundError("user-42")
    )
    cleanup = SystemExit("release-control")
    backend = _DirectAcquirePool(release_error=cleanup)
    pool = _database_pool(backend)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")

    async def _run_context() -> None:
        async with _acquisition_context(pool, context_name):
            raise primary

    try:
        raised = await _capture_task_failure(_run_context())
    finally:
        logger.remove(sink_id)

    assert raised is cleanup
    assert raised.__cause__ is None
    assert backend.release_calls == [(backend.connection, 1.25)]
    assert sink.getvalue() == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("context_name", ["main", "credential-lock"])
@pytest.mark.parametrize("primary_kind", ["ordinary", "trusted"])
async def test_acquisition_ordinary_release_failure_preserves_exception_chain(
    context_name: str,
    primary_kind: str,
) -> None:
    cause = ValueError("body-cause")
    primary = (
        RuntimeError(_RAW_BACKEND_TEXT)
        if primary_kind == "ordinary"
        else UserNotFoundError("user-42")
    )
    primary.__cause__ = cause
    backend = _DirectAcquirePool(release_error=RuntimeError(_RAW_BACKEND_TEXT))
    pool = _database_pool(backend)
    sink = io.StringIO()
    sink_id = logger.add(sink, format="{message} {extra}")

    async def _run_context() -> None:
        async with _acquisition_context(pool, context_name):
            raise primary

    try:
        raised = await _capture_task_failure(_run_context())
    finally:
        logger.remove(sink_id)

    logs = sink.getvalue()
    assert raised is primary
    assert raised.__cause__ is cause
    assert backend.release_calls == [(backend.connection, 1.25)]
    assert _RAW_BACKEND_TEXT not in logs
    if primary_kind == "ordinary":
        assert logs.count("PostgreSQL connection release failed") == 1
        assert "'backend': 'postgresql'" in logs
        assert "'operation': 'release'" in logs
        assert "'error_type': 'RuntimeError'" in logs
    else:
        assert logs == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("context_name", ["main", "credential-lock"])
async def test_acquisition_release_failure_after_success_is_sanitized(
    context_name: str,
) -> None:
    backend = _DirectAcquirePool(release_error=RuntimeError(_RAW_BACKEND_TEXT))
    pool = _database_pool(backend)
    sink = io.StringIO()
    sink_id = logger.add(sink, format="{message} {extra}")
    try:
        with pytest.raises(TransactionError) as raised:
            async with _acquisition_context(pool, context_name):
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
    assert logs.count("PostgreSQL connection release failed") == 1
    assert "'backend': 'postgresql'" in logs
    assert "'operation': 'release'" in logs
    assert "'error_type': 'RuntimeError'" in logs
    assert backend.release_calls == [(backend.connection, 1.25)]


@pytest.mark.asyncio
@pytest.mark.parametrize("context_name", ["main", "credential-lock"])
async def test_acquisition_release_control_after_success_propagates_unchanged(
    context_name: str,
) -> None:
    cleanup = SystemExit("release-control")
    backend = _DirectAcquirePool(release_error=cleanup)
    pool = _database_pool(backend)
    sink = io.StringIO()
    sink_id = logger.add(sink, level="DEBUG")

    async def _run_context() -> None:
        async with _acquisition_context(pool, context_name):
            pass

    try:
        raised = await _capture_task_failure(_run_context())
    finally:
        logger.remove(sink_id)

    assert raised is cleanup
    assert raised.__cause__ is None
    assert backend.release_calls == [(backend.connection, 1.25)]
    assert sink.getvalue() == ""


@pytest.mark.asyncio
@pytest.mark.parametrize("context_name", ["main", "credential-lock"])
async def test_acquisition_cancellation_waits_for_release_completion(
    context_name: str,
) -> None:
    release_barrier = _ReleaseBarrier()
    backend = _DirectAcquirePool(release_barrier=release_barrier)
    pool = _database_pool(backend)

    async def _run_context() -> None:
        async with _acquisition_context(pool, context_name):
            pass

    task = asyncio.create_task(_run_context())
    await release_barrier.started.wait()
    task.cancel("first-cancellation")
    await _event_loop_turn()
    assert task.done() is False
    assert release_barrier.completed is False

    task.cancel("repeated-cancellation")
    await _event_loop_turn()
    assert task.done() is False
    assert release_barrier.completed is False

    release_barrier.release.set()
    with pytest.raises(asyncio.CancelledError) as raised:
        await task

    assert raised.value.args == ("first-cancellation",)
    assert raised.value.__cause__ is None
    assert release_barrier.completed is True
    assert backend.released is True
    assert backend.release_calls == [(backend.connection, 1.25)]


@pytest.mark.asyncio
@pytest.mark.parametrize("context_name", ["main", "credential-lock"])
async def test_acquisition_context_success_yields_and_releases_once(
    context_name: str,
) -> None:
    backend = _DirectAcquirePool()
    pool = _database_pool(backend)

    async with _acquisition_context(pool, context_name, timeout=2.5) as connection:
        assert connection is backend.connection

    assert backend.acquire_timeouts == [2.5]
    assert backend.release_calls == [(backend.connection, 2.5)]
    assert backend.released is True


@pytest.mark.asyncio
@pytest.mark.parametrize("context_name", ["main", "credential-lock"])
async def test_acquisition_context_explicit_exhaustion_remains_sanitized(
    context_name: str,
) -> None:
    backend_error = asyncpg.exceptions.TooManyConnectionsError(_RAW_BACKEND_TEXT)
    backend = _DirectAcquirePool(acquire_error=backend_error)
    pool = _database_pool(backend)

    with pytest.raises(ConnectionPoolExhaustedError) as raised:
        async with _acquisition_context(pool, context_name):
            raise AssertionError("exhausted pool yielded a connection")

    assert raised.value.__cause__ is None
    assert _RAW_BACKEND_TEXT not in str(raised.value)
    assert backend.acquire_timeouts == [1.25]
    assert backend.release_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("context_name", ["main", "credential-lock"])
async def test_acquisition_context_timeout_identity_remains_unchanged(
    context_name: str,
) -> None:
    timeout = asyncio.TimeoutError(_RAW_BACKEND_TEXT)
    backend = _DirectAcquirePool(acquire_error=timeout)
    pool = _database_pool(backend)

    with pytest.raises(asyncio.TimeoutError) as raised:
        async with _acquisition_context(pool, context_name):
            raise AssertionError("timed-out pool yielded a connection")

    assert raised.value is timeout
    assert backend.acquire_timeouts == [1.25]
    assert backend.release_calls == []


def test_transaction_policy_defaults() -> None:
    assert AuthnzTransactionPolicy is not None

    policy = AuthnzTransactionPolicy.from_mapping({})

    assert policy.sqlite_lock_max_retries == 2
    assert policy.sqlite_lock_retry_base_seconds == 0.05
    assert policy.sqlite_lock_retry_max_seconds == 0.25
    assert policy.busy_retry_after_seconds == 1
    assert policy.db_pool_acquire_timeout_seconds == 5.0


def test_transaction_policy_invalid_values_are_safely_normalized_without_logging() -> None:
    assert AuthnzTransactionPolicy is not None
    secret = "invalid-/tmp/authnz-users.db-token"
    sink = io.StringIO()
    sink_id = logger.add(sink, level="WARNING")
    try:
        policy = AuthnzTransactionPolicy.from_mapping(
            {
                "AUTHNZ_SQLITE_LOCK_MAX_RETRIES": "-4",
                "AUTHNZ_SQLITE_LOCK_RETRY_BASE_SECONDS": "nan",
                "AUTHNZ_SQLITE_LOCK_RETRY_MAX_SECONDS": "-1",
                "AUTHNZ_SQLITE_LOCK_RETRY_AFTER_SECONDS": "-3",
                "AUTHNZ_DB_POOL_ACQUIRE_TIMEOUT_SECONDS": "inf",
                "UNRELATED_SECRET": secret,
            }
        )
    finally:
        logger.remove(sink_id)

    assert policy.sqlite_lock_max_retries == 0
    assert math.isfinite(policy.sqlite_lock_retry_base_seconds)
    assert policy.sqlite_lock_retry_base_seconds == 0.05
    assert policy.sqlite_lock_retry_max_seconds == policy.sqlite_lock_retry_base_seconds
    assert policy.busy_retry_after_seconds == 0
    assert policy.db_pool_acquire_timeout_seconds == 5.0
    assert secret not in sink.getvalue()
    text_policy = AuthnzTransactionPolicy.from_mapping(
        {
            "AUTHNZ_SQLITE_LOCK_MAX_RETRIES": secret,
            "AUTHNZ_SQLITE_LOCK_RETRY_BASE_SECONDS": secret,
            "AUTHNZ_SQLITE_LOCK_RETRY_MAX_SECONDS": secret,
            "AUTHNZ_SQLITE_LOCK_RETRY_AFTER_SECONDS": secret,
            "AUTHNZ_DB_POOL_ACQUIRE_TIMEOUT_SECONDS": secret,
        }
    )
    assert text_policy == AuthnzTransactionPolicy()
    assert secret not in sink.getvalue()


def test_transaction_policy_observes_env_set_change_and_delete_without_settings_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert get_authnz_transaction_policy is not None
    monkeypatch.setenv("AUTHNZ_DB_POOL_ACQUIRE_TIMEOUT_SECONDS", "1.25")
    first = get_authnz_transaction_policy()

    monkeypatch.setenv("AUTHNZ_DB_POOL_ACQUIRE_TIMEOUT_SECONDS", "2.5")
    changed = get_authnz_transaction_policy()

    monkeypatch.delenv("AUTHNZ_DB_POOL_ACQUIRE_TIMEOUT_SECONDS")
    deleted = get_authnz_transaction_policy()

    assert first.db_pool_acquire_timeout_seconds == 1.25
    assert changed.db_pool_acquire_timeout_seconds == 2.5
    assert deleted.db_pool_acquire_timeout_seconds == 5.0


def test_transaction_policy_accepts_config_backed_settings_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert AuthnzTransactionPolicy is not None
    config = configparser.ConfigParser()
    config["AuthNZ"] = {
        "authnz_sqlite_lock_max_retries": "6",
        "authnz_sqlite_lock_retry_base_seconds": "0.4",
        "authnz_sqlite_lock_retry_max_seconds": "0.8",
        "authnz_sqlite_lock_retry_after_seconds": "9",
        "authnz_db_pool_acquire_timeout_seconds": "3.75",
    }
    for name in config["AuthNZ"]:
        monkeypatch.delenv(name.upper(), raising=False)
    monkeypatch.setattr(
        authnz_settings_module,
        "load_comprehensive_config",
        lambda: config,
    )

    settings_values = authnz_settings_module._load_overrides_from_config()
    policy = AuthnzTransactionPolicy.from_settings(
        SimpleNamespace(**settings_values),
        environ={},
    )

    assert policy.sqlite_lock_max_retries == 6
    assert policy.sqlite_lock_retry_base_seconds == 0.4
    assert policy.sqlite_lock_retry_max_seconds == 0.8
    assert policy.busy_retry_after_seconds == 9
    assert policy.db_pool_acquire_timeout_seconds == 3.75


def test_transaction_policy_accepts_settings_object_without_process_env() -> None:
    assert AuthnzTransactionPolicy is not None
    settings = SimpleNamespace(
        AUTHNZ_SQLITE_LOCK_MAX_RETRIES="4",
        AUTHNZ_SQLITE_LOCK_RETRY_BASE_SECONDS="0.2",
        AUTHNZ_SQLITE_LOCK_RETRY_MAX_SECONDS="0.6",
        AUTHNZ_SQLITE_LOCK_RETRY_AFTER_SECONDS="8",
        AUTHNZ_DB_POOL_ACQUIRE_TIMEOUT_SECONDS="4.5",
    )

    policy = AuthnzTransactionPolicy.from_settings(settings, environ={})

    assert policy.sqlite_lock_max_retries == 4
    assert policy.sqlite_lock_retry_base_seconds == 0.2
    assert policy.sqlite_lock_retry_max_seconds == 0.6
    assert policy.busy_retry_after_seconds == 8
    assert policy.db_pool_acquire_timeout_seconds == 4.5


def test_live_transaction_policy_layers_env_over_non_stale_config_and_dotenv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    assert get_authnz_transaction_policy is not None
    config = configparser.ConfigParser()
    config["AuthNZ"] = {
        "authnz_db_pool_acquire_timeout_seconds": "3.75",
    }
    env_path = tmp_path / ".env"
    env_path.write_text(
        "AUTHNZ_DB_POOL_ACQUIRE_TIMEOUT_SECONDS=3.25\n",
        encoding="utf-8",
    )
    stale_settings = SimpleNamespace(
        AUTHNZ_DB_POOL_ACQUIRE_TIMEOUT_SECONDS="1.25",
    )
    authnz_settings_module.reset_settings()
    monkeypatch.setattr(authnz_settings_module, "_settings", stale_settings)
    monkeypatch.setattr(authnz_settings_module, "AUTHNZ_DEFAULT_ENV_FILE", env_path)
    monkeypatch.setattr(
        authnz_settings_module,
        "load_comprehensive_config",
        lambda: config,
    )

    monkeypatch.setenv("AUTHNZ_DB_POOL_ACQUIRE_TIMEOUT_SECONDS", "1.25")
    initial = get_authnz_transaction_policy()
    monkeypatch.setenv("AUTHNZ_DB_POOL_ACQUIRE_TIMEOUT_SECONDS", "2.5")
    changed = get_authnz_transaction_policy()
    monkeypatch.delenv("AUTHNZ_DB_POOL_ACQUIRE_TIMEOUT_SECONDS")
    deleted = get_authnz_transaction_policy()

    assert authnz_settings_module.get_settings() is stale_settings
    assert initial.db_pool_acquire_timeout_seconds == 1.25
    assert changed.db_pool_acquire_timeout_seconds == 2.5
    assert deleted.db_pool_acquire_timeout_seconds == 3.25


def test_live_transaction_policy_refreshes_config_only_after_settings_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert get_authnz_transaction_policy is not None
    config = configparser.ConfigParser()
    config["AuthNZ"] = {
        "authnz_db_pool_acquire_timeout_seconds": "3.75",
    }
    monkeypatch.setattr(
        authnz_settings_module,
        "load_comprehensive_config",
        lambda: config,
    )
    monkeypatch.delenv("AUTHNZ_DB_POOL_ACQUIRE_TIMEOUT_SECONDS", raising=False)
    authnz_settings_module.reset_settings()

    initial = get_authnz_transaction_policy()
    config["AuthNZ"]["authnz_db_pool_acquire_timeout_seconds"] = "4.5"
    cached = get_authnz_transaction_policy()
    monkeypatch.setenv("AUTHNZ_DB_POOL_ACQUIRE_TIMEOUT_SECONDS", "1.25")
    env_initial = get_authnz_transaction_policy()
    monkeypatch.setenv("AUTHNZ_DB_POOL_ACQUIRE_TIMEOUT_SECONDS", "2.5")
    env_changed = get_authnz_transaction_policy()
    monkeypatch.delenv("AUTHNZ_DB_POOL_ACQUIRE_TIMEOUT_SECONDS")
    env_deleted = get_authnz_transaction_policy()
    authnz_settings_module.reset_settings()
    refreshed = get_authnz_transaction_policy()

    assert initial.db_pool_acquire_timeout_seconds == 3.75
    assert cached.db_pool_acquire_timeout_seconds == 3.75
    assert env_initial.db_pool_acquire_timeout_seconds == 1.25
    assert env_changed.db_pool_acquire_timeout_seconds == 2.5
    assert env_deleted.db_pool_acquire_timeout_seconds == 3.75
    assert refreshed.db_pool_acquire_timeout_seconds == 4.5
