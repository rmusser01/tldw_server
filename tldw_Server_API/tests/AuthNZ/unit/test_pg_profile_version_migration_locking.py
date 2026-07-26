from __future__ import annotations

import asyncio
from types import TracebackType

import pytest

from tldw_Server_API.app.core.AuthNZ.pg_migrations_extra import (
    ensure_user_profile_version_pg,
)

pytestmark = pytest.mark.unit


class _MigrationState:
    def __init__(self, *, metadata_failure: BaseException | None = None) -> None:
        self.column_exists = False
        self.metadata_failure = metadata_failure
        self.unlocked_metadata_reads = 0
        self.unlocked_metadata_barrier = asyncio.Event()
        self.advisory_lock = asyncio.Lock()
        self.active_lock_holders = 0
        self.max_lock_holders = 0
        self.lock_acquisitions = 0
        self.lock_releases = 0
        self.lock_keys: list[int] = []
        self.add_attempts = 0
        self.commits = 0
        self.rollbacks = 0
        self.events: list[tuple[int, str]] = []


class _Transaction:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    async def __aenter__(self) -> None:
        self.connection.in_transaction = True
        self.connection.state.events.append(
            (self.connection.identifier, "transaction_enter")
        )

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc, traceback
        state = self.connection.state
        state.events.append(
            (
                self.connection.identifier,
                "transaction_rollback" if exc_type is not None else "transaction_commit",
            )
        )
        if exc_type is None:
            state.commits += 1
        else:
            state.rollbacks += 1
        if self.connection.holds_advisory_lock:
            self.connection.holds_advisory_lock = False
            state.active_lock_holders -= 1
            state.lock_releases += 1
            state.advisory_lock.release()
        self.connection.in_transaction = False
        return False


class _Connection:
    def __init__(self, state: _MigrationState, identifier: int) -> None:
        self.state = state
        self.identifier = identifier
        self.in_transaction = False
        self.holds_advisory_lock = False
        self.initial_profile_metadata_read = False

    def transaction(self) -> _Transaction:
        return _Transaction(self)

    async def fetchval(self, query: str, *args: object) -> object:
        normalized = " ".join(query.split())
        if "pg_advisory_xact_lock" in normalized:
            assert self.in_transaction
            assert normalized == "SELECT pg_advisory_xact_lock($1)"
            assert len(args) == 1 and isinstance(args[0], int)
            self.state.events.append((self.identifier, "lock_wait"))
            await self.state.advisory_lock.acquire()
            self.holds_advisory_lock = True
            self.state.active_lock_holders += 1
            self.state.max_lock_holders = max(
                self.state.max_lock_holders,
                self.state.active_lock_holders,
            )
            self.state.lock_acquisitions += 1
            self.state.lock_keys.append(args[0])
            self.state.events.append((self.identifier, "lock_acquired"))
            return None
        if "information_schema.tables" in normalized:
            self.state.events.append((self.identifier, "users_metadata"))
            if self.state.metadata_failure is not None:
                raise self.state.metadata_failure
            return True
        if "column_name = 'updated_at'" in normalized:
            return "timestamp with time zone"
        if "SELECT COUNT(*)" in normalized:
            return 0
        raise AssertionError(f"Unexpected fetchval query: {normalized}")

    async def fetchrow(self, query: str, *args: object) -> object:
        del args
        normalized = " ".join(query.split())
        if "column_name = 'profile_version'" not in normalized:
            raise AssertionError(f"Unexpected fetchrow query: {normalized}")
        if not self.initial_profile_metadata_read:
            self.initial_profile_metadata_read = True
            self.state.events.append((self.identifier, "profile_metadata"))
            observed_exists = self.state.column_exists
            if not self.holds_advisory_lock:
                self.state.unlocked_metadata_reads += 1
                if self.state.unlocked_metadata_reads == 2:
                    self.state.unlocked_metadata_barrier.set()
                await self.state.unlocked_metadata_barrier.wait()
            if not observed_exists:
                return None
        return {
            "data_type": "timestamp with time zone",
            "is_nullable": "NO",
        }

    async def execute(self, query: str, *args: object) -> None:
        del args
        normalized = " ".join(query.split())
        if "ADD COLUMN profile_version" in normalized:
            self.state.add_attempts += 1
            if self.state.column_exists:
                raise RuntimeError("duplicate column profile_version")
            self.state.column_exists = True
        self.state.events.append((self.identifier, f"execute:{normalized}"))


class _Acquire:
    def __init__(self, connection: _Connection) -> None:
        self.connection = connection

    async def __aenter__(self) -> _Connection:
        return self.connection

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool:
        del exc_type, exc, traceback
        return False


class _ConcurrentPool:
    def __init__(self, state: _MigrationState) -> None:
        self.pool = object()
        self.state = state
        self.connections: list[_Connection] = []

    def acquire(self) -> _Acquire:
        connection = _Connection(self.state, len(self.connections) + 1)
        self.connections.append(connection)
        return _Acquire(connection)


@pytest.mark.asyncio
async def test_profile_version_migration_serializes_concurrent_startup() -> None:
    state = _MigrationState()
    pool = _ConcurrentPool(state)

    results = await asyncio.gather(
        ensure_user_profile_version_pg(pool),
        ensure_user_profile_version_pg(pool),
    )

    assert results == [True, True]
    assert state.add_attempts == 1
    assert state.lock_acquisitions == 2
    assert state.lock_releases == 2
    assert state.max_lock_holders == 1
    assert state.lock_keys == [0x544C44575F505631, 0x544C44575F505631]
    assert state.commits == 2
    assert state.rollbacks == 0
    assert not state.advisory_lock.locked()
    for connection in pool.connections:
        labels = [
            label
            for identifier, label in state.events
            if identifier == connection.identifier
        ]
        assert labels.index("transaction_enter") < labels.index("lock_acquired")
        assert labels.index("lock_acquired") < labels.index("users_metadata")
        assert labels[-1] == "transaction_commit"


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(RuntimeError("metadata failed"), id="runtime-error"),
        pytest.param(asyncio.CancelledError(), id="cancellation"),
    ],
)
@pytest.mark.asyncio
async def test_profile_version_migration_failure_rolls_back_advisory_lock(
    failure: BaseException,
) -> None:
    state = _MigrationState(metadata_failure=failure)
    pool = _ConcurrentPool(state)

    with pytest.raises(type(failure)) as caught:
        await ensure_user_profile_version_pg(pool)

    assert caught.value is failure
    assert state.lock_acquisitions == 1
    assert state.lock_releases == 1
    assert state.commits == 0
    assert state.rollbacks == 1
    assert not state.advisory_lock.locked()
    labels = [label for _identifier, label in state.events]
    assert labels.index("transaction_enter") < labels.index("lock_acquired")
    assert labels.index("lock_acquired") < labels.index("users_metadata")
    assert labels[-1] == "transaction_rollback"
