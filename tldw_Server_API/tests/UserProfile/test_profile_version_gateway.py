from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone, tzinfo
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_Server_API.app.core.AuthNZ.profile_version import (
    ProfileVersionInvalid,
    ProfileVersionNotFound,
    ProfileVersionReadFailed,
    compute_touch_value,
    normalize_profile_version,
)
from tldw_Server_API.app.core.UserProfiles.backend import (
    ProfileBackendUnavailable,
    resolve_profile_backend,
)
from tldw_Server_API.app.core.UserProfiles.service import UserProfileService
from tldw_Server_API.app.core.UserProfiles.version_gateway import (
    ProfileVersionCandidates,
    ProfileVersionGateway,
)

UTC = timezone.utc

UTC_CONVERSION_OVERFLOW_VALUES = (
    pytest.param(
        "9999-12-31T23:59:59.999999-23:59",
        id="upper-string",
    ),
    pytest.param(
        datetime.max.replace(
            tzinfo=timezone(-timedelta(hours=23, minutes=59))
        ),
        id="upper-datetime",
    ),
    pytest.param(
        "0001-01-01T00:00:00+23:59",
        id="lower-string",
    ),
    pytest.param(
        datetime.min.replace(
            tzinfo=timezone(timedelta(hours=23, minutes=59))
        ),
        id="lower-datetime",
    ),
)


def utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class _Cursor:
    def __init__(self, rows: list[Any], *, rowcount: Any = -1) -> None:
        self._rows = rows
        self.rowcount = rowcount

    async def fetchall(self) -> list[Any]:
        return self._rows


class _SQLiteConnection:
    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows
        self.calls: list[tuple[str, Any]] = []
        self.update_rowcount = 1

    async def execute(self, sql: str, params: Any = ()) -> _Cursor:
        self.calls.append((sql, params))
        if sql.lstrip().upper().startswith("UPDATE"):
            return _Cursor([], rowcount=self.update_rowcount)
        return _Cursor(self.rows)


class _PostgresConnection:
    def __init__(self, rows: list[Any]) -> None:
        self.rows = rows
        self.fetch_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.execute_calls: list[tuple[str, tuple[Any, ...]]] = []
        self.update_result = "UPDATE 1"

    async def fetch(self, sql: str, *params: Any) -> list[Any]:
        self.fetch_calls.append((sql, params))
        return self.rows

    async def execute(self, sql: str, *params: Any) -> str:
        self.execute_calls.append((sql, params))
        return self.update_result


class _Pool:
    def __init__(
        self,
        conn: Any,
        *,
        postgres: bool = False,
        backend_type: Any = None,
    ) -> None:
        self.conn = conn
        self.pool = object() if postgres else None
        self.backend_type = (
            backend_type
            if backend_type is not None
            else ("postgres" if postgres else "sqlite")
        )
        self.acquire_calls = 0

    @asynccontextmanager
    async def acquire(self):
        self.acquire_calls += 1
        yield self.conn


def _candidate_rows(*values: tuple[str, int | None, Any]) -> list[dict[str, Any]]:
    return [
        {"source_tag": source, "source_id": source_id, "candidate_value": value}
        for source, source_id, value in values
    ]


def test_profile_version_candidates_are_immutable_and_compute_maximum() -> None:
    older = utc("2026-01-01T00:00:00.000000Z")
    newer = utc("2026-01-02T00:00:00.000000Z")
    candidates = ProfileVersionCandidates(user_exists=True, values=(older, newer))

    assert candidates.maximum == newer
    with pytest.raises(AttributeError):
        candidates.user_exists = False  # type: ignore[misc]


@pytest.mark.parametrize(
    "candidates,error_type",
    [
        (ProfileVersionCandidates(user_exists=False, values=()), ProfileVersionNotFound),
        (ProfileVersionCandidates(user_exists=True, values=()), ProfileVersionNotFound),
    ],
)
def test_profile_version_candidates_fail_when_user_or_values_are_missing(
    candidates: ProfileVersionCandidates,
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        _ = candidates.maximum


def test_compute_touch_value_exceeds_future_floor() -> None:
    now = utc("2026-01-01T00:00:00.000000Z")
    floor = utc("2026-01-02T00:00:00.999999Z")

    assert compute_touch_value(now, floor) == utc("2026-01-02T00:00:01.000000Z")


def test_compute_touch_value_uses_aware_utc_clock_when_it_is_later() -> None:
    now = utc("2026-01-02T00:00:01.000000Z")
    floor = utc("2026-01-01T00:00:00.000000Z")

    assert compute_touch_value(now, floor) is now


def test_compute_touch_value_sanitizes_maximum_datetime_overflow() -> None:
    maximum = datetime.max.replace(tzinfo=UTC)

    with pytest.raises(ProfileVersionInvalid) as raised:
        compute_touch_value(maximum, maximum)

    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True


class _TimestampString(str):
    def strip(self, _chars: str | None = None) -> str:
        raise RuntimeError("secret timestamp string")


class _TimestampDatetime(datetime):
    pass


class _HostileTimezone(tzinfo):
    def utcoffset(self, _dt: datetime | None) -> timedelta | None:
        raise RuntimeError("secret timezone")

    def dst(self, _dt: datetime | None) -> timedelta | None:
        return None

    def tzname(self, _dt: datetime | None) -> str | None:
        return "hostile"


UNTRUSTED_TIMESTAMP_VALUES = (
    pytest.param(
        _TimestampString("2026-01-01T00:00:00Z"),
        id="str-subclass",
    ),
    pytest.param(
        _TimestampDatetime(2026, 1, 1, tzinfo=UTC),
        id="datetime-subclass",
    ),
    pytest.param(
        datetime(2026, 1, 1, tzinfo=_HostileTimezone()),
        id="hostile-timezone",
    ),
)


@pytest.mark.parametrize("value", UNTRUSTED_TIMESTAMP_VALUES)
def test_normalize_profile_version_contains_untrusted_timestamp_types(
    value: Any,
) -> None:
    with pytest.raises(ProfileVersionInvalid) as raised:
        normalize_profile_version(value)

    assert str(raised.value) == "Stored profile version is invalid"
    assert "secret" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True


@pytest.mark.parametrize("value", UNTRUSTED_TIMESTAMP_VALUES)
def test_compute_touch_value_contains_untrusted_timestamp_types(value: Any) -> None:
    with pytest.raises(ProfileVersionInvalid) as raised:
        compute_touch_value(value, utc("2026-01-01T00:00:00Z"))

    assert str(raised.value) == "Stored profile version is invalid"
    assert "secret" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True


class _ControlFlowTimezone(tzinfo):
    def __init__(self, signal: BaseException) -> None:
        self.signal = signal

    def utcoffset(self, _dt: datetime | None) -> timedelta | None:
        raise self.signal

    def dst(self, _dt: datetime | None) -> timedelta | None:
        return None

    def tzname(self, _dt: datetime | None) -> str | None:
        return "control-flow"


def test_normalize_profile_version_does_not_swallow_baseexception() -> None:
    signal = KeyboardInterrupt("stop")
    value = datetime(2026, 1, 1, tzinfo=_ControlFlowTimezone(signal))

    with pytest.raises(KeyboardInterrupt) as raised:
        normalize_profile_version(value)

    assert raised.value is signal


@pytest.mark.parametrize("value", UTC_CONVERSION_OVERFLOW_VALUES)
def test_normalize_profile_version_sanitizes_utc_conversion_overflow(
    value: Any,
) -> None:
    with pytest.raises(ProfileVersionInvalid) as raised:
        normalize_profile_version(value)

    assert str(raised.value) == "Stored profile version is invalid"
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True


@pytest.mark.parametrize("value", UTC_CONVERSION_OVERFLOW_VALUES)
def test_compute_touch_value_sanitizes_utc_conversion_overflow(value: Any) -> None:
    with pytest.raises(ProfileVersionInvalid) as raised:
        compute_touch_value(value, utc("2026-01-01T00:00:00Z"))

    assert str(raised.value) == "Stored profile version is invalid"
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True


@pytest.mark.parametrize(
    "now,floor",
    [
        (datetime(2026, 1, 1), utc("2026-01-01T00:00:00Z")),
        (utc("2026-01-01T00:00:00Z"), datetime(2026, 1, 1)),
        ("not-a-date", utc("2026-01-01T00:00:00Z")),
    ],
)
def test_compute_touch_value_rejects_invalid_or_naive_inputs(now: Any, floor: Any) -> None:
    with pytest.raises(ProfileVersionInvalid):
        compute_touch_value(now, floor)


@pytest.mark.asyncio
async def test_stale_read_acquires_once_and_uses_one_complete_sqlite_statement() -> None:
    rows = _candidate_rows(
        ("user", 7, "2026-01-01 00:00:00"),
        ("org_membership", 11, None),
        ("team_membership", 13, None),
        ("user_override", None, "2026-01-01T00:00:01.000000Z"),
        ("org_override", 11, "2025-12-31T23:00:00-01:00"),
        ("team_override", 13, "2026-01-01T00:00:03.000000+00:00"),
    )
    conn = _SQLiteConnection(rows)
    pool = _Pool(conn)

    version = await ProfileVersionGateway(pool).read(7)

    assert version == utc("2026-01-01T00:00:03.000000Z")
    assert pool.acquire_calls == 1
    assert len(conn.calls) == 1
    sql, params = conn.calls[0]
    assert params == (7,)
    assert "users.profile_version" in sql
    assert "user_config_overrides" in sql
    assert "org_members" in sql and "org_config_overrides" in sql
    assert "team_members" in sql and "team_config_overrides" in sql
    assert "source_tag" in sql
    assert "COALESCE(om.status, 'active') = 'active'" in sql
    assert "COALESCE(tm.status, 'active') = 'active'" in sql


@pytest.mark.asyncio
async def test_transaction_read_uses_only_supplied_connection() -> None:
    pooled_conn = _SQLiteConnection([])
    supplied_conn = _SQLiteConnection(
        _candidate_rows(("user", 4, "2026-02-03T04:05:06.123456Z"))
    )
    pool = _Pool(pooled_conn)

    version = await ProfileVersionGateway(pool).read_in_transaction(
        supplied_conn,
        4,
        lock_user=True,
    )

    assert version == utc("2026-02-03T04:05:06.123456Z")
    assert pool.acquire_calls == 0
    assert len(supplied_conn.calls) == 1
    assert pooled_conn.calls == []


@pytest.mark.asyncio
async def test_missing_user_fails_closed_without_raw_cause() -> None:
    pool = _Pool(_SQLiteConnection([]))

    with pytest.raises(ProfileVersionNotFound) as raised:
        await ProfileVersionGateway(pool).read(999)

    assert raised.value.code == "profile_update_not_found"
    assert raised.value.__cause__ is None
    assert "999" not in str(raised.value)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row",
    [
        {"source_tag": "user", "source_id": 1, "candidate_value": None},
        {"source_tag": "user_override", "source_id": None, "candidate_value": None},
        {"source_tag": "org_override", "source_id": 4, "candidate_value": "not-a-date"},
        {"source_tag": "team_override", "source_id": 5, "candidate_value": "2026-01-01"},
        {"source_tag": "team_membership", "source_id": None, "candidate_value": None},
    ],
)
async def test_null_actual_or_malformed_component_fails_closed(row: dict[str, Any]) -> None:
    rows = _candidate_rows(("user", 1, "2026-01-01T00:00:00Z"))
    rows.append(row)

    with pytest.raises(ProfileVersionInvalid) as raised:
        await ProfileVersionGateway(_Pool(_SQLiteConnection(rows))).read(1)

    assert raised.value.code == "profile_version_invalid"
    assert "not-a-date" not in str(raised.value)


class _SourceTag(str):
    pass


class _HostileSourceTag:
    def __hash__(self) -> int:
        raise RuntimeError("secret source-tag hash")

    def __eq__(self, _other: object) -> bool:
        raise RuntimeError("secret source-tag equality")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_tag",
    [
        pytest.param(["user"], id="unhashable"),
        pytest.param(_HostileSourceTag(), id="hostile"),
        pytest.param(_SourceTag("user"), id="str-subclass"),
    ],
)
async def test_candidate_source_tag_requires_exact_closed_string(
    source_tag: Any,
) -> None:
    rows = _candidate_rows(
        (source_tag, 1, "2026-01-01T00:00:00Z"),
    )

    with pytest.raises(ProfileVersionInvalid) as raised:
        await ProfileVersionGateway(_Pool(_SQLiteConnection(rows))).read(1)

    assert str(raised.value) == "Stored profile version is invalid"
    assert "secret" not in str(raised.value)
    assert raised.value.__cause__ is None


class _IntIdentifier(int):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row",
    [
        pytest.param(
            ("user", True, "2026-01-01T00:00:00Z"),
            id="user-bool",
        ),
        pytest.param(("org_membership", True, None), id="org-membership-bool"),
        pytest.param(
            ("team_membership", _IntIdentifier(2), None),
            id="team-membership-int-subclass",
        ),
        pytest.param(
            ("org_override", "3", "2026-01-01T00:00:00Z"),
            id="org-override-string",
        ),
        pytest.param(
            ("team_override", 4.0, "2026-01-01T00:00:00Z"),
            id="team-override-float",
        ),
        pytest.param(
            ("user_override", 0, "2026-01-01T00:00:00Z"),
            id="user-override-nonnull",
        ),
        pytest.param(
            ("org_membership", 5, "2026-01-01T00:00:00Z"),
            id="org-membership-candidate",
        ),
        pytest.param(
            ("team_membership", 6, "2026-01-01T00:00:00Z"),
            id="team-membership-candidate",
        ),
    ],
)
async def test_candidate_source_id_and_value_shapes_are_exact(
    row: tuple[str, Any, Any],
) -> None:
    rows = []
    if row[0] != "user":
        rows.extend(_candidate_rows(("user", 1, "2026-01-01T00:00:00Z")))
    rows.extend(_candidate_rows(row))

    with pytest.raises(ProfileVersionInvalid) as raised:
        await ProfileVersionGateway(_Pool(_SQLiteConnection(rows))).read(1)

    assert str(raised.value) == "Stored profile version is invalid"
    assert raised.value.__cause__ is None


@pytest.mark.asyncio
async def test_postgres_rejects_naive_backend_datetime() -> None:
    rows = _candidate_rows(("user", 1, datetime(2026, 1, 1)))

    with pytest.raises(ProfileVersionInvalid):
        await ProfileVersionGateway(_Pool(_PostgresConnection(rows), postgres=True)).read(1)


@pytest.mark.asyncio
async def test_lazy_postgres_backend_type_is_used_before_pool_creation() -> None:
    conn = _PostgresConnection(
        _candidate_rows(("user", 1, utc("2026-01-01T00:00:00Z")))
    )
    pool = _Pool(conn, backend_type="postgres")

    version = await ProfileVersionGateway(pool).read(1)

    assert pool.pool is None
    assert version == utc("2026-01-01T00:00:00Z")
    assert len(conn.fetch_calls) == 1


@pytest.mark.asyncio
async def test_unknown_backend_type_fails_closed() -> None:
    pool = _Pool(
        _SQLiteConnection(
            _candidate_rows(("user", 1, "2026-01-01T00:00:00Z"))
        ),
        backend_type="mysql",
    )

    with pytest.raises(ProfileVersionReadFailed) as raised:
        await ProfileVersionGateway(pool).read(1)

    assert raised.value.code == "profile_version_read_failed"
    assert raised.value.__cause__ is None


class _HostileBackendIdentifier:
    def __eq__(self, _other: object) -> bool:
        raise RuntimeError("secret backend discriminator")


class _BackendString(str):
    pass


@pytest.mark.parametrize(
    "backend_type",
    [_HostileBackendIdentifier(), _BackendString("postgres")],
    ids=["raising-equality", "string-subclass"],
)
def test_backend_resolver_rejects_non_exact_string_discriminators(
    backend_type: Any,
) -> None:
    with pytest.raises(ProfileBackendUnavailable) as raised:
        resolve_profile_backend(SimpleNamespace(backend_type=backend_type))

    assert str(raised.value) == "Profile storage backend is unavailable"
    assert raised.value.__cause__ is None
    assert raised.value.__context__ is None


@pytest.mark.asyncio
async def test_hostile_backend_identifier_is_sanitized_by_version_gateway() -> None:
    pool = _Pool(_SQLiteConnection([]), backend_type=_HostileBackendIdentifier())

    with pytest.raises(ProfileVersionReadFailed) as raised:
        await ProfileVersionGateway(pool).read(1)

    assert str(raised.value) == "Profile version could not be read"
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True
    assert pool.acquire_calls == 0


@pytest.mark.asyncio
async def test_backend_failure_is_sanitized_without_raw_detail_or_chain() -> None:
    class FailingConnection:
        async def fetch(self, _sql: str, *_params: Any) -> list[Any]:
            raise RuntimeError("secret backend host and SQL")

    gateway = ProfileVersionGateway(_Pool(FailingConnection(), postgres=True))

    with pytest.raises(ProfileVersionReadFailed) as raised:
        await gateway.read(1)

    assert raised.value.code == "profile_version_read_failed"
    assert "secret" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True


@pytest.mark.asyncio
async def test_transaction_read_preserves_sanitized_postgres_conflict_signal() -> None:
    class PostgresConflict(RuntimeError):
        sqlstate = "40001"

    class FailingConnection:
        async def fetch(self, _sql: str, *_params: Any) -> list[Any]:
            try:
                raise PostgresConflict("secret serialization detail")
            except PostgresConflict as conflict:
                raise RuntimeError("secret driver wrapper") from conflict

    conn = FailingConnection()
    gateway = ProfileVersionGateway(_Pool(conn, postgres=True))

    with pytest.raises(ProfileVersionReadFailed) as raised:
        await gateway.read_in_transaction(conn, 1, lock_user=True)

    assert raised.value.sqlstate == "40001"
    assert "secret" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True


class _SqlstateString(str):
    pass


class _HostileSqlstate:
    def __hash__(self) -> int:
        return hash("40001")

    def __eq__(self, _other: object) -> bool:
        raise RuntimeError("secret sqlstate comparison")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        pytest.param("sqlstate", ["40001"], id="unhashable-sqlstate"),
        pytest.param("pgcode", {"40001"}, id="unhashable-pgcode"),
        pytest.param("sqlstate", _SqlstateString("40001"), id="string-subclass"),
        pytest.param("sqlstate", _HostileSqlstate(), id="raising-equality"),
    ],
)
async def test_malformed_postgres_conflict_signal_is_sanitized_by_gateway(
    attribute: str,
    value: Any,
) -> None:
    error = RuntimeError("secret backend failure")
    setattr(error, attribute, value)

    class FailingConnection:
        async def fetch(self, _sql: str, *_params: Any) -> list[Any]:
            raise error

    conn = FailingConnection()
    gateway = ProfileVersionGateway(_Pool(conn, postgres=True))

    with pytest.raises(ProfileVersionReadFailed) as raised:
        await gateway.read_in_transaction(conn, 1, lock_user=True)

    assert str(raised.value) == "Profile version could not be read"
    assert not hasattr(raised.value, "sqlstate")
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True


class _RaisingSqlstateWithPgcode(RuntimeError):
    pgcode = "40001"

    @property
    def sqlstate(self) -> str:
        raise RuntimeError("secret sqlstate accessor")


class _MalformedSqlstateWithPgcode(RuntimeError):
    sqlstate = ["invalid"]
    pgcode = "40P01"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected_sqlstate"),
    [
        pytest.param(
            _RaisingSqlstateWithPgcode("secret backend failure"),
            "40001",
            id="raising-sqlstate-valid-pgcode",
        ),
        pytest.param(
            _MalformedSqlstateWithPgcode("secret backend failure"),
            "40P01",
            id="malformed-sqlstate-valid-pgcode",
        ),
    ],
)
async def test_postgres_conflict_accessors_are_validated_independently(
    error: BaseException,
    expected_sqlstate: str,
) -> None:
    class FailingConnection:
        async def fetch(self, _sql: str, *_params: Any) -> list[Any]:
            raise error

    conn = FailingConnection()

    with pytest.raises(ProfileVersionReadFailed) as raised:
        await ProfileVersionGateway(_Pool(conn, postgres=True)).read_in_transaction(
            conn,
            1,
            lock_user=True,
        )

    assert raised.value.sqlstate == expected_sqlstate
    assert str(raised.value) == "Profile version could not be read"
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True


class _RaisingConflictAccessors(RuntimeError):
    @property
    def sqlstate(self) -> str:
        raise RuntimeError("secret sqlstate accessor")

    @property
    def pgcode(self) -> str:
        raise RuntimeError("secret pgcode accessor")


@pytest.mark.asyncio
@pytest.mark.parametrize("chain", ["cause", "context"])
async def test_postgres_conflict_accessor_errors_do_not_stop_chain_traversal(
    chain: str,
) -> None:
    class NestedConflict(RuntimeError):
        sqlstate = "40001"

    class FailingConnection:
        async def fetch(self, _sql: str, *_params: Any) -> list[Any]:
            try:
                raise NestedConflict("secret nested conflict")
            except NestedConflict as conflict:
                wrapper = _RaisingConflictAccessors("secret wrapper")
                if chain == "cause":
                    raise wrapper from conflict
                raise wrapper  # noqa: B904 - exercise implicit context traversal

    conn = FailingConnection()

    with pytest.raises(ProfileVersionReadFailed) as raised:
        await ProfileVersionGateway(_Pool(conn, postgres=True)).read_in_transaction(
            conn,
            1,
            lock_user=True,
        )

    assert raised.value.sqlstate == "40001"
    assert str(raised.value) == "Profile version could not be read"
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True


@pytest.mark.asyncio
async def test_one_statement_snapshot_never_hybridizes_old_and_new_components() -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    old_rows = _candidate_rows(
        ("user", 1, "2026-01-01T00:00:00Z"),
        ("user_override", None, "2026-01-01T00:00:01Z"),
    )
    new_rows = _candidate_rows(
        ("user", 1, "2026-01-02T00:00:00Z"),
        ("user_override", None, "2026-01-02T00:00:01Z"),
    )
    state = {"rows": old_rows}

    class SnapshotConnection:
        calls = 0

        async def fetch(self, _sql: str, *_params: Any) -> list[Any]:
            self.calls += 1
            snapshot = list(state["rows"])
            entered.set()
            await release.wait()
            return snapshot

    conn = SnapshotConnection()
    task = asyncio.create_task(ProfileVersionGateway(_Pool(conn, postgres=True)).read(1))
    await entered.wait()
    state["rows"] = new_rows
    release.set()

    assert await task == utc("2026-01-01T00:00:01Z")
    assert conn.calls == 1


@pytest.mark.asyncio
async def test_postgres_lock_query_locks_user_cte_before_candidate_union() -> None:
    conn = _PostgresConnection(
        _candidate_rows(("user", 8, utc("2026-01-01T00:00:00Z")))
    )
    gateway = ProfileVersionGateway(_Pool(conn, postgres=True))

    await gateway.read_in_transaction(conn, 8, lock_user=True)
    locked_sql = conn.fetch_calls[-1][0]
    await gateway.read_in_transaction(conn, 8, lock_user=False)
    unlocked_sql = conn.fetch_calls[-1][0]

    assert locked_sql.index("FOR UPDATE") < locked_sql.index("UNION ALL")
    assert "FOR UPDATE" not in unlocked_sql


@pytest.mark.asyncio
async def test_touch_writes_exact_backend_value_and_requires_one_user_row() -> None:
    value = utc("2026-01-01T00:00:00.123456Z")
    sqlite_conn = _SQLiteConnection([])
    postgres_conn = _PostgresConnection([])

    await ProfileVersionGateway(_Pool(sqlite_conn)).touch(sqlite_conn, 3, value)
    await ProfileVersionGateway(_Pool(postgres_conn, postgres=True)).touch(
        postgres_conn,
        3,
        value,
    )

    assert sqlite_conn.calls == [
        (
            "UPDATE users SET profile_version = ? WHERE id = ?",
            ("2026-01-01T00:00:00.123456Z", 3),
        )
    ]
    assert postgres_conn.execute_calls == [
        ("UPDATE users SET profile_version = $1 WHERE id = $2", (value, 3))
    ]

    sqlite_conn.update_rowcount = 0
    with pytest.raises(ProfileVersionNotFound):
        await ProfileVersionGateway(_Pool(sqlite_conn)).touch(sqlite_conn, 3, value)


class _IntRowcount(int):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "rowcount",
    [
        pytest.param(True, id="bool"),
        pytest.param("1", id="string"),
        pytest.param(_IntRowcount(1), id="int-subclass"),
    ],
)
async def test_sqlite_touch_rejects_malformed_rowcount(rowcount: Any) -> None:
    conn = _SQLiteConnection([])
    conn.update_rowcount = rowcount

    with pytest.raises(ProfileVersionReadFailed) as raised:
        await ProfileVersionGateway(_Pool(conn)).touch(
            conn,
            3,
            utc("2026-01-01T00:00:00Z"),
        )

    assert str(raised.value) == "Profile version could not be read"
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True


class _CommandTag(str):
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        pytest.param("DELETE 1", id="wrong-command"),
        pytest.param("UPDATE x", id="nonnumeric-count"),
        pytest.param("prefix 1", id="arbitrary-prefix"),
        pytest.param(" UPDATE 1", id="leading-whitespace"),
        pytest.param("UPDATE 1 ", id="trailing-whitespace"),
        pytest.param("UPDATE  1", id="repeated-whitespace"),
        pytest.param("UPDATE\t1", id="tab-whitespace"),
        pytest.param(_CommandTag("UPDATE 1"), id="str-subclass"),
    ],
)
async def test_postgres_touch_rejects_malformed_command_tag(result: Any) -> None:
    conn = _PostgresConnection([])
    conn.update_result = result

    with pytest.raises(ProfileVersionReadFailed) as raised:
        await ProfileVersionGateway(_Pool(conn, postgres=True)).touch(
            conn,
            3,
            utc("2026-01-01T00:00:00Z"),
        )

    assert str(raised.value) == "Profile version could not be read"
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "acknowledged"),
    [
        pytest.param("UPDATE 0", False, id="zero"),
        pytest.param("UPDATE 1", True, id="one"),
    ],
)
async def test_postgres_touch_accepts_exact_update_command_tags(
    result: str,
    acknowledged: bool,
) -> None:
    conn = _PostgresConnection([])
    conn.update_result = result
    gateway = ProfileVersionGateway(_Pool(conn, postgres=True))

    if acknowledged:
        await gateway.touch(conn, 3, utc("2026-01-01T00:00:00Z"))
    else:
        with pytest.raises(ProfileVersionNotFound):
            await gateway.touch(conn, 3, utc("2026-01-01T00:00:00Z"))


@pytest.mark.asyncio
@pytest.mark.parametrize("value", UTC_CONVERSION_OVERFLOW_VALUES)
async def test_touch_sanitizes_utc_conversion_overflow(value: Any) -> None:
    conn = _SQLiteConnection([])

    with pytest.raises(ProfileVersionInvalid) as raised:
        await ProfileVersionGateway(_Pool(conn)).touch(conn, 3, value)

    assert str(raised.value) == "Stored profile version is invalid"
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True
    assert conn.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("value", UNTRUSTED_TIMESTAMP_VALUES)
async def test_touch_contains_untrusted_timestamp_types(value: Any) -> None:
    conn = _SQLiteConnection([])

    with pytest.raises(ProfileVersionInvalid) as raised:
        await ProfileVersionGateway(_Pool(conn)).touch(conn, 3, value)

    assert str(raised.value) == "Stored profile version is invalid"
    assert "secret" not in str(raised.value)
    assert raised.value.__cause__ is None
    assert raised.value.__suppress_context__ is True
    assert conn.calls == []


@pytest.mark.asyncio
async def test_service_delegates_version_reads_and_build_profile_propagates_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Gateway:
        def __init__(self) -> None:
            self.calls: list[tuple[str, Any]] = []

        async def read(self, user_id: int) -> datetime:
            self.calls.append(("read", user_id))
            raise ProfileVersionReadFailed()

        async def read_in_transaction(
            self,
            conn: Any,
            user_id: int,
            *,
            lock_user: bool,
        ) -> datetime:
            self.calls.append(("transaction", conn, user_id, lock_user))
            return utc("2026-01-01T00:00:00Z")

        async def touch(self, conn: Any, user_id: int, value: datetime) -> None:
            raise AssertionError("not used")

    gateway = Gateway()
    service = UserProfileService(SimpleNamespace(pool=None), profile_version_gateway=gateway)
    supplied = object()

    assert await service.get_profile_version(
        user_id=7,
        user_updated_at=utc("1999-01-01T00:00:00Z"),
        db_conn=supplied,
        lock_user=True,
    ) == utc("2026-01-01T00:00:00Z")
    assert gateway.calls == [("transaction", supplied, 7, True)]

    monkeypatch.setattr(
        "tldw_Server_API.app.core.UserProfiles.service.load_user_profile_catalog",
        lambda: SimpleNamespace(version="test", entries=[]),
    )
    with pytest.raises(ProfileVersionReadFailed):
        await service.build_profile(user={"id": 7}, sections=set())
