from __future__ import annotations

import asyncio
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.exceptions import (
    ConnectionPoolExhaustedError,
    DatabaseConcurrencyConflict,
    DatabaseLockError,
    RollbackSignal,
    TransactionError,
)
from tldw_Server_API.app.core.AuthNZ.transaction_policy import AuthnzTransactionPolicy
from tldw_Server_API.app.core.UserProfiles.transaction_gateway import (
    ProfileDatabaseBusy,
    ProfileTransactionFailed,
    ProfileTransactionGateway,
    ProfileUpdateConcurrencyConflict,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "exception_type",
    [
        ProfileDatabaseBusy,
        ProfileTransactionFailed,
        ProfileUpdateConcurrencyConflict,
    ],
)
def test_transaction_failures_are_owned_by_core_exception_taxonomy(
    exception_type: type[Exception],
) -> None:
    assert exception_type.__module__ == "tldw_Server_API.app.core.exceptions"


class _TransactionContext:
    def __init__(self, pool: _Pool, outcome: Any) -> None:
        self.pool = pool
        self.outcome = outcome

    async def __aenter__(self) -> object:
        self.pool.enter_count += 1
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.pool.conn

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        self.pool.exit_triples.append((exc_type, exc, traceback))
        if isinstance(self.outcome, _ExitFailure):
            raise self.outcome.error
        return False


class _ExitFailure:
    def __init__(self, error: BaseException) -> None:
        self.error = error


class _Pool:
    def __init__(
        self,
        outcomes: list[Any],
        *,
        postgres: bool,
        backend_type: Any = None,
    ) -> None:
        self.pool = object() if postgres else None
        self.backend_type = (
            backend_type
            if backend_type is not None
            else ("postgres" if postgres else "sqlite")
        )
        self.outcomes = list(outcomes)
        self.conn = object()
        self.timeouts: list[float | None] = []
        self.enter_count = 0
        self.exit_triples: list[tuple[Any, Any, Any]] = []

    def transaction(self, *, acquire_timeout_seconds: float | None = None):
        self.timeouts.append(acquire_timeout_seconds)
        return _TransactionContext(self, self.outcomes.pop(0))


def policy(**overrides: Any) -> AuthnzTransactionPolicy:
    values = {
        "sqlite_lock_max_retries": 2,
        "sqlite_lock_retry_base_seconds": 0.05,
        "sqlite_lock_retry_max_seconds": 0.20,
        "busy_retry_after_seconds": 7,
        "db_pool_acquire_timeout_seconds": 3.25,
    }
    values.update(overrides)
    return AuthnzTransactionPolicy(**values)


@pytest.mark.asyncio
async def test_success_uses_one_connection_and_forwards_exact_pool_deadline() -> None:
    pool = _Pool([None], postgres=True)
    seen: list[object] = []

    async def operation(conn: object) -> str:
        seen.append(conn)
        return "done"

    result = await ProfileTransactionGateway(pool, policy=policy()).run(operation)

    assert result == "done"
    assert seen == [pool.conn]
    assert pool.timeouts == [3.25]
    assert pool.enter_count == 1


@pytest.mark.asyncio
async def test_sqlite_retries_initial_attempt_plus_configured_max_with_exponential_backoff() -> None:
    pool = _Pool(
        [DatabaseLockError(), DatabaseLockError(), None],
        postgres=False,
    )
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    result = await ProfileTransactionGateway(
        pool,
        policy=policy(),
        sleep=sleep,
    ).run(lambda _conn: _return("ok"))

    assert result == "ok"
    assert pool.enter_count == 3
    assert delays == [0.05, 0.10]
    assert pool.timeouts == [3.25, 3.25, 3.25]


@pytest.mark.asyncio
async def test_sqlite_backoff_is_capped() -> None:
    pool = _Pool(
        [DatabaseLockError(), DatabaseLockError(), DatabaseLockError(), None],
        postgres=False,
    )
    delays: list[float] = []

    async def sleep(delay: float) -> None:
        delays.append(delay)

    await ProfileTransactionGateway(
        pool,
        policy=policy(
            sqlite_lock_max_retries=3,
            sqlite_lock_retry_base_seconds=0.15,
            sqlite_lock_retry_max_seconds=0.20,
        ),
        sleep=sleep,
    ).run(lambda _conn: _return(None))

    assert delays == [0.15, 0.20, 0.20]


@pytest.mark.asyncio
async def test_exhausted_sqlite_contention_has_exact_bounded_busy_metadata() -> None:
    pool = _Pool([DatabaseLockError(), DatabaseLockError()], postgres=False)

    with pytest.raises(ProfileDatabaseBusy) as raised:
        await ProfileTransactionGateway(
            pool,
            policy=policy(sqlite_lock_max_retries=1),
            sleep=lambda _delay: _return(None),
        ).run(lambda _conn: _return("never"))

    assert raised.value.code == "database_busy"
    assert raised.value.retry_after_seconds == 7
    assert raised.value.__cause__ is None
    assert pool.enter_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ConnectionPoolExhaustedError(), TimeoutError()])
async def test_postgres_acquisition_failure_maps_to_database_busy_without_retry(
    error: Exception,
) -> None:
    pool = _Pool([error], postgres=True)

    with pytest.raises(ProfileDatabaseBusy) as raised:
        await ProfileTransactionGateway(pool, policy=policy()).run(
            lambda _conn: _return("never")
        )

    assert raised.value.code == "database_busy"
    assert raised.value.retry_after_seconds == 7
    assert pool.enter_count == 1
    assert pool.timeouts == [3.25]


@pytest.mark.asyncio
async def test_lazy_postgres_backend_type_maps_timeout_before_pool_creation() -> None:
    pool = _Pool(
        [TimeoutError()],
        postgres=False,
        backend_type="postgres",
    )

    with pytest.raises(ProfileDatabaseBusy) as raised:
        await ProfileTransactionGateway(pool, policy=policy()).run(
            lambda _conn: _return("never")
        )

    assert pool.pool is None
    assert raised.value.code == "database_busy"
    assert pool.enter_count == 1


@pytest.mark.asyncio
async def test_unknown_backend_type_fails_closed() -> None:
    pool = _Pool([None], postgres=False, backend_type="mysql")

    with pytest.raises(ProfileTransactionFailed) as raised:
        await ProfileTransactionGateway(pool, policy=policy()).run(
            lambda _conn: _return("must-not-run")
        )

    assert raised.value.code == "profile_update_failed"
    assert pool.enter_count == 0


class _HostileBackendIdentifier:
    def __eq__(self, _other: object) -> bool:
        raise RuntimeError("secret backend discriminator")


@pytest.mark.asyncio
async def test_hostile_backend_identifier_is_sanitized_by_transaction_gateway() -> None:
    pool = _Pool(
        [None],
        postgres=False,
        backend_type=_HostileBackendIdentifier(),
    )

    with pytest.raises(ProfileTransactionFailed) as raised:
        await ProfileTransactionGateway(pool, policy=policy()).run(
            lambda _conn: _return("must-not-run")
        )

    assert str(raised.value) == "Profile update transaction failed"
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True
    assert pool.enter_count == 0


@pytest.mark.asyncio
async def test_postgres_concurrency_conflict_maps_after_exit_and_is_never_retried() -> None:
    pool = _Pool([DatabaseConcurrencyConflict(), None], postgres=True)
    calls = 0

    async def operation(_conn: object) -> None:
        nonlocal calls
        calls += 1

    with pytest.raises(ProfileUpdateConcurrencyConflict) as raised:
        await ProfileTransactionGateway(pool, policy=policy()).run(operation)

    assert raised.value.code == "profile_update_concurrency_conflict"
    assert raised.value.retry_after_seconds is None
    assert pool.enter_count == 1
    assert calls == 0


@pytest.mark.asyncio
async def test_rollback_signal_passes_unchanged_after_exit() -> None:
    pool = _Pool([None], postgres=True)
    signal = RollbackSignal("sanitized")

    async def operation(_conn: object) -> None:
        raise signal

    with pytest.raises(RollbackSignal) as raised:
        await ProfileTransactionGateway(pool, policy=policy()).run(operation)

    assert raised.value is signal
    assert pool.exit_triples[0][1] is signal


@pytest.mark.asyncio
async def test_cancellation_passes_unchanged_after_exit() -> None:
    pool = _Pool([None], postgres=True)
    cancellation = asyncio.CancelledError("stop")

    async def operation(_conn: object) -> None:
        raise cancellation

    with pytest.raises(asyncio.CancelledError) as raised:
        await ProfileTransactionGateway(pool, policy=policy()).run(operation)

    assert raised.value is cancellation
    assert pool.exit_triples[0][1] is cancellation


@pytest.mark.asyncio
async def test_non_exception_baseexception_passes_unchanged() -> None:
    pool = _Pool([None], postgres=True)
    control = KeyboardInterrupt()

    async def operation(_conn: object) -> None:
        raise control

    with pytest.raises(KeyboardInterrupt) as raised:
        await ProfileTransactionGateway(pool, policy=policy()).run(operation)

    assert raised.value is control


@pytest.mark.asyncio
async def test_unexpected_failure_is_sanitized_without_raw_backend_chain() -> None:
    pool = _Pool([TransactionError("raw-host secret")], postgres=True)

    with pytest.raises(ProfileTransactionFailed) as raised:
        await ProfileTransactionGateway(pool, policy=policy()).run(
            lambda _conn: _return("never")
        )

    assert raised.value.code == "profile_update_failed"
    assert "secret" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True


class _DetailedConcurrencyConflict(DatabaseConcurrencyConflict):
    def __init__(self) -> None:
        Exception.__init__(self, "secret exit-time conflict")


@pytest.mark.asyncio
async def test_exit_time_commit_conflict_never_returns_success_or_retries() -> None:
    conflict = _DetailedConcurrencyConflict()
    pool = _Pool([_ExitFailure(conflict), None], postgres=True)
    calls = 0

    async def operation(_conn: object) -> str:
        nonlocal calls
        calls += 1
        return "must-not-escape"

    with pytest.raises(ProfileUpdateConcurrencyConflict) as raised:
        await ProfileTransactionGateway(pool, policy=policy()).run(operation)

    assert str(raised.value) == "Profile update conflicted"
    assert "secret" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True
    assert calls == 1
    assert pool.enter_count == 1
    assert len(pool.exit_triples) == 1
    assert len(pool.outcomes) == 1


@pytest.mark.asyncio
async def test_exit_time_rollback_conflict_maps_after_body_failure_without_retry() -> None:
    conflict = _DetailedConcurrencyConflict()
    body_error = RuntimeError("secret body failure")
    pool = _Pool([_ExitFailure(conflict), None], postgres=True)
    calls = 0

    async def operation(_conn: object) -> None:
        nonlocal calls
        calls += 1
        raise body_error

    with pytest.raises(ProfileUpdateConcurrencyConflict) as raised:
        await ProfileTransactionGateway(pool, policy=policy()).run(operation)

    assert str(raised.value) == "Profile update conflicted"
    assert "secret" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True
    assert calls == 1
    assert pool.enter_count == 1
    assert pool.exit_triples[0][1] is body_error
    assert len(pool.outcomes) == 1


@pytest.mark.asyncio
async def test_commit_failure_never_returns_body_success() -> None:
    pool = _Pool([_ExitFailure(TransactionError("commit secret"))], postgres=True)
    body_completed = asyncio.Event()

    async def operation(_conn: object) -> str:
        body_completed.set()
        return "must-not-escape"

    with pytest.raises(ProfileTransactionFailed):
        await ProfileTransactionGateway(pool, policy=policy()).run(operation)

    assert body_completed.is_set()
    assert len(pool.exit_triples) == 1


async def _return(value: Any) -> Any:
    return value
