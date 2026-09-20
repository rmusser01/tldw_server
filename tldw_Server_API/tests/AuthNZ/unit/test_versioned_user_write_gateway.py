from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import (
    _active_capability_count,
    _guard_sql,
)
from tldw_Server_API.app.core.AuthNZ.profile_version import (
    ProfileVersionInvalid,
    ProfileVersionNotFound,
    UserVersionOwnership,
    VersionedUserWriteGateway,
)

BASE = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)


class _Cursor:
    def __init__(
        self,
        *,
        rowcount: Any = 1,
        lastrowid: Any = 41,
    ) -> None:
        self.rowcount = rowcount
        self.lastrowid = lastrowid


class _SQLiteConnection:
    def __init__(self, *, rowcount: Any = 1, lastrowid: Any = 41) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self.rowcount = rowcount
        self.lastrowid = lastrowid

    async def execute(self, statement: Any, parameters: tuple[Any, ...]) -> _Cursor:
        guarded = _guard_sql(
            statement,
            backend="sqlite",
            connection_identity=self,
            operation="execute",
        )
        self.calls.append((guarded, parameters))
        return _Cursor(rowcount=self.rowcount, lastrowid=self.lastrowid)


class _PostgresConnection:
    def __init__(
        self,
        *,
        status: Any = "UPDATE 1",
        inserted_id: Any = 51,
    ) -> None:
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.fetchval_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.status = status
        self.inserted_id = inserted_id

    async def execute(self, statement: Any, *parameters: Any) -> Any:
        guarded = _guard_sql(
            statement,
            backend="postgres",
            connection_identity=self,
            operation="execute",
        )
        self.execute_calls.append((guarded, parameters))
        return self.status

    async def fetchval(self, statement: Any, *parameters: Any) -> Any:
        guarded = _guard_sql(
            statement,
            backend="postgres",
            connection_identity=self,
            operation="fetchval",
        )
        self.fetchval_calls.append((guarded, parameters))
        return self.inserted_id


class _ProfileVersionGateway:
    def __init__(
        self,
        *floors: datetime,
        read_failure: BaseException | None = None,
        touch_failure: BaseException | None = None,
    ) -> None:
        self.floors = list(floors)
        self.read_failure = read_failure
        self.touch_failure = touch_failure
        self.events: list[tuple[Any, ...]] = []

    async def read_in_transaction(
        self,
        conn: Any,
        user_id: int,
        *,
        lock_user: bool,
    ) -> datetime:
        self.events.append(("read", conn, user_id, lock_user))
        if self.read_failure is not None:
            raise self.read_failure
        return self.floors.pop(0)

    async def touch(self, conn: Any, user_id: int, value: datetime) -> None:
        self.events.append(("touch", conn, user_id, value))
        if self.touch_failure is not None:
            raise self.touch_failure

    def read_in_transaction_sync(
        self,
        executor: Any,
        conn: Any,
        user_id: int,
        *,
        lock_user: bool,
    ) -> datetime:
        self.events.append(("read_sync", executor, conn, user_id, lock_user))
        if self.read_failure is not None:
            raise self.read_failure
        return self.floors.pop(0)

    def touch_sync(
        self,
        executor: Any,
        conn: Any,
        user_id: int,
        value: datetime,
    ) -> None:
        self.events.append(("touch_sync", executor, conn, user_id, value))
        if self.touch_failure is not None:
            raise self.touch_failure


class _SyncResult:
    def __init__(
        self,
        *,
        rowcount: Any = 1,
        lastrowid: Any = 61,
        rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.rowcount = rowcount
        self.lastrowid = lastrowid
        self.rows = rows or []


class _SyncExecutor:
    def __init__(
        self,
        *,
        rowcount: Any = 1,
        lastrowid: Any = 61,
        backend: str = "sqlite",
    ) -> None:
        self.rowcount = rowcount
        self.lastrowid = lastrowid
        self.backend = backend
        self.calls: list[tuple[str, tuple[Any, ...], Any]] = []

    def execute(
        self,
        statement: Any,
        parameters: tuple[Any, ...],
        *,
        connection: Any,
    ) -> _SyncResult:
        guarded = _guard_sql(
            statement,
            backend=self.backend,
            connection_identity=connection,
            operation="execute",
        )
        self.calls.append((guarded, parameters, connection))
        return _SyncResult(rowcount=self.rowcount, lastrowid=self.lastrowid)


@pytest.mark.asyncio
async def test_gateway_owned_update_locks_captures_pre_and_post_and_touches_once() -> None:
    conn = _SQLiteConnection()
    versions = _ProfileVersionGateway(BASE, BASE + timedelta(seconds=2))
    gateway = VersionedUserWriteGateway(
        "sqlite",
        profile_version_gateway=versions,
        clock=lambda: BASE,
    )

    result = await gateway.execute_update(
        conn,
        user_id=7,
        profile_visible_fields=("email",),
        statement="UPDATE users SET email = ? WHERE id = ?",
        parameters=("new@example.com", 7),
        ownership=UserVersionOwnership.GATEWAY_OWNS_ANCHOR,
    )

    assert result.affected_user_ids == (7,)
    assert result.version_floor == BASE + timedelta(seconds=2)
    assert versions.events == [
        ("read", conn, 7, True),
        ("read", conn, 7, False),
        ("touch", conn, 7, BASE + timedelta(seconds=2, microseconds=1)),
    ]
    assert conn.calls == [
        ("UPDATE main.users SET email = ? WHERE id = ?", ("new@example.com", 7))
    ]


@pytest.mark.asyncio
async def test_postgres_gateway_qualifies_users_target_to_public_schema() -> None:
    conn = _PostgresConnection()
    versions = _ProfileVersionGateway(BASE, BASE)
    gateway = VersionedUserWriteGateway(
        "postgres",
        profile_version_gateway=versions,
        clock=lambda: BASE,
    )

    await gateway.execute_update(
        conn,
        user_id=7,
        profile_visible_fields=("email",),
        statement="UPDATE users SET email = $1 WHERE id = $2",
        parameters=("new@example.com", 7),
    )

    assert conn.execute_calls[0][0] == (
        "UPDATE public.users SET email = $1 WHERE id = $2"
    )


@pytest.mark.asyncio
async def test_gateway_capability_survives_adapter_with_shared_guard_identity() -> None:
    identity = object()

    class _GuardedConnection(_SQLiteConnection):
        _authnz_profile_user_guard_identity = identity

        async def execute(
            self,
            statement: Any,
            parameters: tuple[Any, ...],
        ) -> _Cursor:
            guarded = _guard_sql(
                statement,
                backend="sqlite",
                connection_identity=identity,
                operation="execute",
            )
            self.calls.append((guarded, parameters))
            return _Cursor()

    class _Adapter:
        _authnz_profile_user_guard_identity = identity

        def __init__(self, connection: _GuardedConnection) -> None:
            self.connection = connection

        async def execute(self, statement: Any, *parameters: Any) -> _Cursor:
            normalized = (
                parameters[0]
                if len(parameters) == 1 and isinstance(parameters[0], tuple)
                else tuple(parameters)
            )
            return await self.connection.execute(statement, normalized)

    connection = _GuardedConnection()
    adapter = _Adapter(connection)
    versions = _ProfileVersionGateway(BASE, BASE)

    await VersionedUserWriteGateway(
        "sqlite",
        profile_version_gateway=versions,
        clock=lambda: BASE,
    ).execute_update(
        adapter,
        user_id=71,
        profile_visible_fields=("email",),
        statement="UPDATE users SET email = ? WHERE id = ?",
        parameters=("adapter@example.com", 71),
    )

    assert connection.calls == [
        ("UPDATE main.users SET email = ? WHERE id = ?", ("adapter@example.com", 71))
    ]


@pytest.mark.asyncio
async def test_same_clock_update_strictly_advances_by_one_microsecond() -> None:
    conn = _SQLiteConnection()
    versions = _ProfileVersionGateway(BASE, BASE)
    gateway = VersionedUserWriteGateway(
        "sqlite",
        profile_version_gateway=versions,
        clock=lambda: BASE,
    )

    await gateway.execute_update(
        conn,
        user_id=8,
        profile_visible_fields=("is_active",),
        statement="UPDATE users SET is_active = ? WHERE id = ?",
        parameters=(0, 8),
    )

    assert versions.events[-1] == (
        "touch",
        conn,
        8,
        BASE + timedelta(microseconds=1),
    )


@pytest.mark.asyncio
async def test_future_inherited_override_sets_the_strict_touch_floor() -> None:
    inherited_floor = BASE + timedelta(days=3)
    conn = _PostgresConnection()
    versions = _ProfileVersionGateway(BASE, inherited_floor)
    gateway = VersionedUserWriteGateway(
        "postgres",
        profile_version_gateway=versions,
        clock=lambda: BASE,
    )

    await gateway.execute_update(
        conn,
        user_id=9,
        profile_visible_fields=("storage_quota_mb",),
        statement="UPDATE users SET storage_quota_mb = $1 WHERE id = $2",
        parameters=(2048, 9),
    )

    assert versions.events[-1] == (
        "touch",
        conn,
        9,
        inherited_floor + timedelta(microseconds=1),
    )


@pytest.mark.asyncio
async def test_caller_owned_update_does_not_touch_and_returns_final_touch_floor() -> None:
    conn = _PostgresConnection()
    versions = _ProfileVersionGateway(BASE, BASE + timedelta(seconds=4))
    gateway = VersionedUserWriteGateway(
        "postgres",
        profile_version_gateway=versions,
        clock=lambda: BASE,
    )

    result = await gateway.execute_update(
        conn,
        user_id=10,
        profile_visible_fields=("username",),
        statement="UPDATE users SET username = $1 WHERE id = $2",
        parameters=("renamed", 10),
        ownership=UserVersionOwnership.CALLER_OWNS_ANCHOR,
    )

    assert result.version_floor == BASE + timedelta(seconds=4)
    assert [event[0] for event in versions.events] == ["read", "read"]


@pytest.mark.asyncio
async def test_caller_final_touch_resnapshots_and_touches_exactly_once() -> None:
    conn = _SQLiteConnection()
    versions = _ProfileVersionGateway(BASE + timedelta(seconds=5))
    gateway = VersionedUserWriteGateway(
        "sqlite",
        profile_version_gateway=versions,
        clock=lambda: BASE,
    )

    result = await gateway.final_touch(
        conn,
        user_id=11,
        version_floor=BASE,
    )

    assert result.affected_user_ids == (11,)
    assert result.version_floor == BASE + timedelta(seconds=5)
    assert versions.events == [
        ("read", conn, 11, False),
        ("touch", conn, 11, BASE + timedelta(seconds=5, microseconds=1)),
    ]


@pytest.mark.asyncio
async def test_secret_only_update_does_not_read_or_advance_profile_version() -> None:
    conn = _SQLiteConnection()
    versions = _ProfileVersionGateway()
    gateway = VersionedUserWriteGateway(
        "sqlite",
        profile_version_gateway=versions,
        clock=lambda: BASE,
    )

    result = await gateway.execute_update(
        conn,
        user_id=12,
        profile_visible_fields=(),
        statement="UPDATE users SET password_hash = ? WHERE id = ?",
        parameters=("secret-hash", 12),
    )

    assert result.affected_user_ids == ()
    assert result.version_floor == BASE
    assert versions.events == []
    assert len(conn.calls) == 1


@pytest.mark.asyncio
async def test_secret_and_visible_update_advances_profile_version() -> None:
    conn = _SQLiteConnection()
    versions = _ProfileVersionGateway(BASE, BASE)
    gateway = VersionedUserWriteGateway(
        "sqlite",
        profile_version_gateway=versions,
        clock=lambda: BASE,
    )

    await gateway.execute_update(
        conn,
        user_id=13,
        profile_visible_fields=("two_factor_enabled",),
        statement=(
            "UPDATE users SET totp_secret = ?, two_factor_enabled = ?, "
            "backup_codes = ? WHERE id = ?"
        ),
        parameters=("secret", 1, "[]", 13),
    )

    assert [event[0] for event in versions.events] == ["read", "read", "touch"]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
async def test_insert_explicitly_initializes_profile_version(backend: str) -> None:
    conn: Any
    if backend == "sqlite":
        conn = _SQLiteConnection(lastrowid=41)
    else:
        conn = _PostgresConnection(inserted_id=51)
    gateway = VersionedUserWriteGateway(backend, clock=lambda: BASE)

    result = await gateway.insert_user(
        conn,
        values={
            "username": "new-user",
            "email": "new@example.com",
            "password_hash": "secret-hash",
            "is_active": True,
        },
    )

    expected_id = 41 if backend == "sqlite" else 51
    assert result.affected_user_ids == (expected_id,)
    assert result.version_floor == BASE
    if backend == "sqlite":
        statement, parameters = conn.calls[0]
        assert "profile_version" in statement
        assert statement.count("?") == len(parameters)
        assert parameters[-1] == "2026-07-26T12:00:00.000000Z"
    else:
        statement, parameters = conn.fetchval_calls[0]
        assert "profile_version" in statement
        assert "RETURNING id" in statement
        assert parameters[-1] == BASE


@pytest.mark.asyncio
async def test_missing_user_stops_before_mutation() -> None:
    missing = ProfileVersionNotFound()
    conn = _SQLiteConnection()
    versions = _ProfileVersionGateway(read_failure=missing)
    gateway = VersionedUserWriteGateway(
        "sqlite",
        profile_version_gateway=versions,
        clock=lambda: BASE,
    )

    with pytest.raises(ProfileVersionNotFound) as exc_info:
        await gateway.execute_update(
            conn,
            user_id=404,
            profile_visible_fields=("email",),
            statement="UPDATE users SET email = ? WHERE id = ?",
            parameters=("missing@example.com", 404),
        )

    assert exc_info.value is missing
    assert conn.calls == []


@pytest.mark.asyncio
async def test_mutation_failure_propagates_without_post_read_or_touch() -> None:
    storage_failure = RuntimeError("write failed")

    class _FailingConnection(_SQLiteConnection):
        async def execute(self, statement: str, parameters: tuple[Any, ...]) -> _Cursor:
            raise storage_failure

    conn = _FailingConnection()
    versions = _ProfileVersionGateway(BASE)
    gateway = VersionedUserWriteGateway(
        "sqlite",
        profile_version_gateway=versions,
        clock=lambda: BASE,
    )

    with pytest.raises(RuntimeError) as exc_info:
        await gateway.execute_update(
            conn,
            user_id=14,
            profile_visible_fields=("email",),
            statement="UPDATE users SET email = ? WHERE id = ?",
            parameters=("new@example.com", 14),
        )

    assert exc_info.value is storage_failure
    assert [event[0] for event in versions.events] == ["read"]
    assert _active_capability_count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "control",
    [asyncio.CancelledError(), KeyboardInterrupt()],
)
async def test_base_exception_propagates_by_identity_without_touch(
    control: BaseException,
) -> None:
    class _FailingConnection(_PostgresConnection):
        async def execute(self, statement: str, *parameters: Any) -> Any:
            raise control

    conn = _FailingConnection()
    versions = _ProfileVersionGateway(BASE)
    gateway = VersionedUserWriteGateway(
        "postgres",
        profile_version_gateway=versions,
        clock=lambda: BASE,
    )

    with pytest.raises(BaseException) as exc_info:
        await gateway.execute_update(
            conn,
            user_id=15,
            profile_visible_fields=("last_login",),
            statement="UPDATE users SET last_login = $1 WHERE id = $2",
            parameters=(BASE, 15),
        )

    assert exc_info.value is control
    assert [event[0] for event in versions.events] == ["read"]
    assert _active_capability_count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "metadata"),
    [
        ("sqlite", True),
        ("sqlite", -1),
        ("postgres", "UPDATE -1"),
        ("postgres", "UPDATE 1 extra"),
    ],
)
async def test_malformed_update_metadata_fails_closed(
    backend: str,
    metadata: Any,
) -> None:
    conn: Any
    if backend == "sqlite":
        conn = _SQLiteConnection(rowcount=metadata)
        statement = "UPDATE users SET email = ? WHERE id = ?"
        parameters = ("new@example.com", 16)
    else:
        conn = _PostgresConnection(status=metadata)
        statement = "UPDATE users SET email = $1 WHERE id = $2"
        parameters = ("new@example.com", 16)
    versions = _ProfileVersionGateway(BASE)
    gateway = VersionedUserWriteGateway(
        backend,
        profile_version_gateway=versions,
        clock=lambda: BASE,
    )

    with pytest.raises(ProfileVersionInvalid):
        await gateway.execute_update(
            conn,
            user_id=16,
            profile_visible_fields=("email",),
            statement=statement,
            parameters=parameters,
        )

    assert [event[0] for event in versions.events] == ["read"]


@pytest.mark.asyncio
async def test_declared_fields_must_exactly_match_visible_statement_columns() -> None:
    conn = _SQLiteConnection()
    gateway = VersionedUserWriteGateway("sqlite", clock=lambda: BASE)

    with pytest.raises(ProfileVersionInvalid):
        await gateway.execute_update(
            conn,
            user_id=17,
            profile_visible_fields=(),
            statement="UPDATE users SET email = ? WHERE id = ?",
            parameters=("hidden@example.com", 17),
        )

    assert conn.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("backend", "statement", "parameters"),
    [
        ("sqlite", "UPDATE users SET email = ? WHERE id = ?", ("x", 99)),
        ("postgres", "UPDATE users SET email = $1 WHERE id = $2", ("x", 99)),
        ("sqlite", "UPDATE users SET email = ? WHERE username = ?", ("x", 17)),
    ],
)
async def test_async_update_rejects_statement_not_bound_to_declared_user(
    backend: str,
    statement: str,
    parameters: tuple[Any, ...],
) -> None:
    conn = _PostgresConnection() if backend == "postgres" else _SQLiteConnection()
    versions = _ProfileVersionGateway(BASE, BASE)

    with pytest.raises(ProfileVersionInvalid):
        await VersionedUserWriteGateway(
            backend,
            profile_version_gateway=versions,
            clock=lambda: BASE,
        ).execute_update(
            conn,
            user_id=17,
            profile_visible_fields=("email",),
            statement=statement,
            parameters=parameters,
        )

    assert versions.events == []


@pytest.mark.asyncio
async def test_async_update_accepts_matching_target_with_additional_and_predicates(
) -> None:
    conn = _PostgresConnection()
    versions = _ProfileVersionGateway(BASE, BASE)

    result = await VersionedUserWriteGateway(
        "postgres",
        profile_version_gateway=versions,
        clock=lambda: BASE,
    ).execute_update(
        conn,
        user_id=17,
        profile_visible_fields=("email",),
        statement=(
            "UPDATE users SET email = $1 WHERE id = $2 "
            "AND lower(email) = lower($3)"
        ),
        parameters=("new@example.com", 17, "old@example.com"),
    )

    assert result.affected_user_ids == (17,)


@pytest.mark.asyncio
async def test_async_update_rejects_or_that_can_escape_target_user() -> None:
    conn = _PostgresConnection()
    versions = _ProfileVersionGateway(BASE, BASE)

    with pytest.raises(ProfileVersionInvalid):
        await VersionedUserWriteGateway(
            "postgres",
            profile_version_gateway=versions,
            clock=lambda: BASE,
        ).execute_update(
            conn,
            user_id=17,
            profile_visible_fields=("email",),
            statement=(
                "UPDATE users SET email = $1 WHERE id = $2 "
                "OR lower(email) = lower($3)"
            ),
            parameters=("new@example.com", 17, "old@example.com"),
        )

    assert versions.events == []


@pytest.mark.parametrize(
    ("backend", "statement", "parameters"),
    [
        ("sqlite", "UPDATE users SET email = ? WHERE id = ?", ("x", 99)),
        ("postgres", "UPDATE users SET email = %s WHERE id = %s", ("x", 99)),
        ("sqlite", "UPDATE users SET email = ? WHERE username = ?", ("x", 17)),
    ],
)
def test_sync_update_rejects_statement_not_bound_to_declared_user(
    backend: str,
    statement: str,
    parameters: tuple[Any, ...],
) -> None:
    executor = _SyncExecutor(backend=backend)
    versions = _ProfileVersionGateway(BASE, BASE)
    connection = object()

    with pytest.raises(ProfileVersionInvalid):
        VersionedUserWriteGateway(
            backend,
            profile_version_gateway=versions,
            clock=lambda: BASE,
        ).execute_update_sync(
            executor,
            connection,
            user_id=17,
            profile_visible_fields=("email",),
            statement=statement,
            parameters=parameters,
        )

    assert versions.events == []
    assert executor.calls == []


def test_sync_gateway_owned_update_uses_same_snapshot_and_touch_semantics() -> None:
    conn = object()
    executor = _SyncExecutor()
    versions = _ProfileVersionGateway(BASE, BASE + timedelta(seconds=6))
    gateway = VersionedUserWriteGateway(
        "sqlite",
        profile_version_gateway=versions,
        clock=lambda: BASE,
    )

    result = gateway.execute_update_sync(
        executor,
        conn,
        user_id=18,
        profile_visible_fields=("email",),
        statement="UPDATE users SET email = ? WHERE id = ?",
        parameters=("sync@example.com", 18),
    )

    assert result.affected_user_ids == (18,)
    assert result.version_floor == BASE + timedelta(seconds=6)
    assert versions.events == [
        ("read_sync", executor, conn, 18, True),
        ("read_sync", executor, conn, 18, False),
        (
            "touch_sync",
            executor,
            conn,
            18,
            BASE + timedelta(seconds=6, microseconds=1),
        ),
    ]
    assert executor.calls == [
        (
            "UPDATE main.users SET email = ? WHERE id = ?",
            ("sync@example.com", 18),
            conn,
        )
    ]


def test_sync_caller_owned_update_returns_floor_without_touch() -> None:
    conn = object()
    executor = _SyncExecutor(backend="postgres")
    versions = _ProfileVersionGateway(BASE, BASE + timedelta(seconds=7))
    gateway = VersionedUserWriteGateway(
        "postgres",
        profile_version_gateway=versions,
        clock=lambda: BASE,
    )

    result = gateway.execute_update_sync(
        executor,
        conn,
        user_id=19,
        profile_visible_fields=("role",),
        statement="UPDATE users SET role = %s WHERE id = %s",
        parameters=("admin", 19),
        ownership=UserVersionOwnership.CALLER_OWNS_ANCHOR,
    )

    assert result.version_floor == BASE + timedelta(seconds=7)
    assert [event[0] for event in versions.events] == ["read_sync", "read_sync"]


@pytest.mark.parametrize("backend", ["sqlite", "postgres"])
def test_sync_insert_explicitly_initializes_profile_version(backend: str) -> None:
    conn = object()
    executor = _SyncExecutor(lastrowid=61, backend=backend)
    gateway = VersionedUserWriteGateway(backend, clock=lambda: BASE)

    result = gateway.insert_user_sync(
        executor,
        conn,
        values={
            "username": "sync-user",
            "email": "sync@example.com",
            "password_hash": "secret-hash",
            "is_active": True,
        },
    )

    assert result.affected_user_ids == (61,)
    assert result.version_floor == BASE
    statement, parameters, used_connection = executor.calls[0]
    assert used_connection is conn
    assert "profile_version" in statement
    if backend == "sqlite":
        assert statement.count("?") == len(parameters)
        assert parameters[-1] == "2026-07-26T12:00:00.000000Z"
    else:
        assert statement.count("%s") == len(parameters)
        assert parameters[-1] == BASE


@pytest.mark.parametrize("metadata", [True, -1, 2])
def test_sync_update_rejects_malformed_or_multirow_metadata(metadata: Any) -> None:
    conn = object()
    executor = _SyncExecutor(rowcount=metadata)
    versions = _ProfileVersionGateway(BASE)
    gateway = VersionedUserWriteGateway(
        "sqlite",
        profile_version_gateway=versions,
        clock=lambda: BASE,
    )

    with pytest.raises(ProfileVersionInvalid):
        gateway.execute_update_sync(
            executor,
            conn,
            user_id=20,
            profile_visible_fields=("email",),
            statement="UPDATE users SET email = ? WHERE id = ?",
            parameters=("sync@example.com", 20),
        )

    assert [event[0] for event in versions.events] == ["read_sync"]
