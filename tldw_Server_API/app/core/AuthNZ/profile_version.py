"""Sanitized profile-version values and failures shared with UserProfiles."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable

from tldw_Server_API.app.core.AuthNZ.profile_user_fields import (
    PROFILE_VISIBLE_USER_FIELDS,
)
from tldw_Server_API.app.core.AuthNZ.profile_user_write_guard import (
    _canonical_users_table,
    _mint_profile_user_sql,
    _profile_user_connection_identity,
    _revoke_profile_user_sql,
)

_PROFILE_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"
    r"(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})?$"
)
_POSTGRES_CONCURRENCY_SQLSTATES = frozenset({"40P01", "40001"})

_WRITABLE_USER_FIELDS = PROFILE_VISIBLE_USER_FIELDS | frozenset(
    {
        "id",
        "password_hash",
        "totp_secret",
        "backup_codes",
        "created_at",
        "updated_at",
        "created_by",
        "email_verified",
        "password_changed_at",
        "failed_login_attempts",
        "is_locked",
        "locked_until",
        "metadata",
    }
)
_UPDATE_USERS_PATTERN = re.compile(
    r"^\s*UPDATE\s+(?P<table>(?:(?:main|public)\.)?users)\s+"
    r"SET\s+(?P<assignments>.+?)\s+"
    r"WHERE\s+(?P<predicate>.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
_ASSIGNMENT_COLUMN_PATTERN = re.compile(
    r'^\s*(?:"(?P<quoted>[A-Za-z_][A-Za-z0-9_]*)"|'
    r"(?P<plain>[A-Za-z_][A-Za-z0-9_]*))\s*=",
)


class UserVersionOwnership(str, Enum):
    """Identify the component responsible for the final anchor touch."""

    GATEWAY_OWNS_ANCHOR = "gateway_owns_anchor"
    CALLER_OWNS_ANCHOR = "caller_owns_anchor"


@dataclass(frozen=True)
class UserWriteResult:
    """Users changed by a write and the complete post-write version floor."""

    affected_user_ids: tuple[int, ...]
    version_floor: datetime


@dataclass(frozen=True)
class _BackendMarker:
    backend_type: str


class ProfileVersionError(RuntimeError):
    """Base class for transport-neutral profile-version failures."""

    code = "profile_version_failed"


class ProfileVersionNotFound(ProfileVersionError):
    """The target user or its durable profile-version anchor is absent."""

    code = "profile_update_not_found"

    def __init__(self) -> None:
        super().__init__("Target profile was not found")


class ProfileVersionInvalid(ProfileVersionError):
    """A stored or supplied profile-version value is invalid."""

    code = "profile_version_invalid"

    def __init__(self) -> None:
        super().__init__("Stored profile version is invalid")


class ProfileVersionReadFailed(ProfileVersionError):
    """The complete profile-version snapshot could not be read."""

    code = "profile_version_read_failed"

    def __init__(self, *, sqlstate: str | None = None) -> None:
        super().__init__("Profile version could not be read")
        if (
            type(sqlstate) is str
            and sqlstate in _POSTGRES_CONCURRENCY_SQLSTATES
        ):
            self.sqlstate = sqlstate

    @classmethod
    def from_storage_error(
        cls,
        error: BaseException,
    ) -> ProfileVersionReadFailed:
        """Preserve only a safe PostgreSQL conflict signal from storage errors."""
        return cls(sqlstate=_postgres_concurrency_sqlstate(error))


def normalize_profile_version(value: Any, *, allow_naive: bool = False) -> datetime:
    """Return one aware UTC timestamp or fail with a sanitized domain error."""
    if type(value) is datetime:
        parsed = value
    elif type(value) is str:
        candidate = value.strip()
        if not _PROFILE_TIMESTAMP_PATTERN.fullmatch(candidate):
            raise ProfileVersionInvalid()
        try:
            parsed = datetime.fromisoformat(candidate.replace("Z", "+00:00"))
        except ValueError:
            raise ProfileVersionInvalid() from None
    else:
        raise ProfileVersionInvalid() from None

    try:
        parsed_tzinfo = parsed.tzinfo
    except Exception:  # noqa: BLE001 - tzinfo implementations are untrusted
        raise ProfileVersionInvalid() from None
    if parsed_tzinfo is None:
        if not allow_naive:
            raise ProfileVersionInvalid()
        parsed = parsed.replace(tzinfo=timezone.utc)
    try:
        return parsed.astimezone(timezone.utc)
    except Exception:  # noqa: BLE001 - tzinfo implementations are untrusted
        raise ProfileVersionInvalid() from None


def compute_touch_value(clock_now_utc: Any, version_floor: Any) -> datetime:
    """Compute the exact monotonic value for the final profile-version touch."""
    now = normalize_profile_version(clock_now_utc)
    floor = normalize_profile_version(version_floor)
    try:
        next_floor = floor + timedelta(microseconds=1)
    except OverflowError:
        raise ProfileVersionInvalid() from None
    return max(now, next_floor)


class VersionedUserWriteGateway:
    """Apply profile-visible users writes on a caller-owned connection."""

    def __init__(
        self,
        backend: str,
        *,
        profile_version_gateway: Any | None = None,
        clock: Callable[[], Any] | None = None,
    ) -> None:
        if type(backend) is not str or backend not in {"sqlite", "postgres"}:
            raise ProfileVersionInvalid()
        self._backend = backend
        if profile_version_gateway is None:
            # Local import avoids a cycle: the reader shares errors from this module.
            from tldw_Server_API.app.core.UserProfiles.version_gateway import (
                ProfileVersionGateway,
            )

            profile_version_gateway = ProfileVersionGateway(_BackendMarker(backend))
        self._profile_versions = profile_version_gateway
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    @property
    def backend(self) -> str:
        """Return the exact backend contract selected for SQL generation."""
        return self._backend

    async def execute_update(
        self,
        conn: Any,
        *,
        user_id: int,
        profile_visible_fields: Sequence[str],
        statement: str,
        parameters: Sequence[Any],
        ownership: UserVersionOwnership = UserVersionOwnership.GATEWAY_OWNS_ANCHOR,
    ) -> UserWriteResult:
        """Execute one users update and, when owned, touch its anchor once."""
        bound_parameters = _validate_update_parameters(parameters)
        fields = _validate_update(
            user_id,
            profile_visible_fields,
            statement,
            ownership,
            bound_parameters,
            self._backend,
        )
        execution_statement = _qualify_update_statement(statement, self._backend)
        operation_clock = self._operation_clock()
        if not fields:
            changed = await self._execute_async(
                conn,
                execution_statement,
                bound_parameters,
            )
            _require_single_or_noop(changed)
            return UserWriteResult((), operation_clock)

        pre_floor = await self._profile_versions.read_in_transaction(
            conn,
            user_id,
            lock_user=True,
        )
        capability = _mint_profile_user_sql(
            execution_statement,
            backend=self._backend,
            connection_identity=_profile_user_connection_identity(conn),
            operation="update",
            columns=tuple(sorted(_update_columns(execution_statement))),
        )
        try:
            changed = await self._execute_async(conn, capability, bound_parameters)
        finally:
            _revoke_profile_user_sql(capability)
        _require_single_or_noop(changed)
        if changed == 0:
            return UserWriteResult((), pre_floor)
        post_floor = await self._profile_versions.read_in_transaction(
            conn,
            user_id,
            lock_user=False,
        )
        version_floor = max(pre_floor, post_floor)
        if ownership is UserVersionOwnership.GATEWAY_OWNS_ANCHOR:
            await self._profile_versions.touch(
                conn,
                user_id,
                compute_touch_value(operation_clock, version_floor),
            )
        return UserWriteResult((user_id,), version_floor)

    def execute_update_sync(
        self,
        executor: Any,
        conn: Any,
        *,
        user_id: int,
        profile_visible_fields: Sequence[str],
        statement: str,
        parameters: Sequence[Any],
        ownership: UserVersionOwnership = UserVersionOwnership.GATEWAY_OWNS_ANCHOR,
    ) -> UserWriteResult:
        """Synchronous counterpart using only the supplied executor/connection."""
        bound_parameters = _validate_update_parameters(parameters)
        fields = _validate_update(
            user_id,
            profile_visible_fields,
            statement,
            ownership,
            bound_parameters,
            self._backend,
        )
        execution_statement = _qualify_update_statement(statement, self._backend)
        operation_clock = self._operation_clock()
        if not fields:
            result = executor.execute(
                execution_statement,
                bound_parameters,
                connection=conn,
            )
            changed = _sync_changed_rows(result)
            _require_single_or_noop(changed)
            return UserWriteResult((), operation_clock)

        pre_floor = self._profile_versions.read_in_transaction_sync(
            executor,
            conn,
            user_id,
            lock_user=True,
        )
        capability = _mint_profile_user_sql(
            execution_statement,
            backend=self._backend,
            connection_identity=_profile_user_connection_identity(conn),
            operation="update",
            columns=tuple(sorted(_update_columns(execution_statement))),
        )
        try:
            result = executor.execute(
                capability,
                bound_parameters,
                connection=conn,
            )
        finally:
            _revoke_profile_user_sql(capability)
        changed = _sync_changed_rows(result)
        _require_single_or_noop(changed)
        if changed == 0:
            return UserWriteResult((), pre_floor)
        post_floor = self._profile_versions.read_in_transaction_sync(
            executor,
            conn,
            user_id,
            lock_user=False,
        )
        version_floor = max(pre_floor, post_floor)
        if ownership is UserVersionOwnership.GATEWAY_OWNS_ANCHOR:
            self._profile_versions.touch_sync(
                executor,
                conn,
                user_id,
                compute_touch_value(operation_clock, version_floor),
            )
        return UserWriteResult((user_id,), version_floor)

    async def final_touch(
        self,
        conn: Any,
        *,
        user_id: int,
        version_floor: Any,
    ) -> UserWriteResult:
        """Perform the caller-owned final touch after one fresh snapshot."""
        floor = normalize_profile_version(version_floor)
        post_floor = await self._profile_versions.read_in_transaction(
            conn,
            user_id,
            lock_user=False,
        )
        final_floor = max(floor, post_floor)
        await self._profile_versions.touch(
            conn,
            user_id,
            compute_touch_value(self._operation_clock(), final_floor),
        )
        return UserWriteResult((user_id,), final_floor)

    async def capture_floor(
        self,
        conn: Any,
        *,
        user_id: int,
        lock_user: bool = True,
    ) -> datetime:
        """Capture a complete floor for a caller-owned compound write."""
        return await self._profile_versions.read_in_transaction(
            conn,
            user_id,
            lock_user=lock_user,
        )

    def final_touch_sync(
        self,
        executor: Any,
        conn: Any,
        *,
        user_id: int,
        version_floor: Any,
    ) -> UserWriteResult:
        """Synchronous caller-owned final touch on the same transaction."""
        floor = normalize_profile_version(version_floor)
        post_floor = self._profile_versions.read_in_transaction_sync(
            executor,
            conn,
            user_id,
            lock_user=False,
        )
        final_floor = max(floor, post_floor)
        self._profile_versions.touch_sync(
            executor,
            conn,
            user_id,
            compute_touch_value(self._operation_clock(), final_floor),
        )
        return UserWriteResult((user_id,), final_floor)

    def capture_floor_sync(
        self,
        executor: Any,
        conn: Any,
        *,
        user_id: int,
        lock_user: bool = True,
    ) -> datetime:
        """Synchronously capture a floor for a caller-owned compound write."""
        return self._profile_versions.read_in_transaction_sync(
            executor,
            conn,
            user_id,
            lock_user=lock_user,
        )

    async def insert_user(
        self,
        conn: Any,
        *,
        values: Mapping[str, Any],
        ignore_conflict: bool = False,
    ) -> UserWriteResult:
        """Insert one user with an explicit initial profile-version value."""
        operation_clock = self._operation_clock()
        statement, parameters = _build_insert(
            self._backend,
            values,
            operation_clock,
            ignore_conflict=ignore_conflict,
            async_postgres=True,
        )
        columns = tuple(values.keys()) + ("profile_version",)
        if self._backend == "postgres":
            capability = _mint_profile_user_sql(
                statement,
                backend=self._backend,
                connection_identity=_profile_user_connection_identity(conn),
                operation="insert",
                columns=columns,
                execution_mode="fetchval",
            )
            try:
                user_id = await conn.fetchval(capability, *parameters)
            finally:
                _revoke_profile_user_sql(capability)
            if user_id is None and ignore_conflict:
                return UserWriteResult((), operation_clock)
        else:
            capability = _mint_profile_user_sql(
                statement,
                backend=self._backend,
                connection_identity=_profile_user_connection_identity(conn),
                operation="insert",
                columns=columns,
            )
            try:
                cursor = await conn.execute(capability, parameters)
            finally:
                _revoke_profile_user_sql(capability)
            changed = _sqlite_rowcount(cursor)
            if changed == 0 and ignore_conflict:
                return UserWriteResult((), operation_clock)
            _require_exactly_one(changed)
            user_id = _inserted_id(cursor)
        return UserWriteResult((_validate_inserted_id(user_id),), operation_clock)

    def insert_user_sync(
        self,
        executor: Any,
        conn: Any,
        *,
        values: Mapping[str, Any],
        ignore_conflict: bool = False,
    ) -> UserWriteResult:
        """Synchronously insert one explicitly versioned user."""
        operation_clock = self._operation_clock()
        statement, parameters = _build_insert(
            self._backend,
            values,
            operation_clock,
            ignore_conflict=ignore_conflict,
            async_postgres=False,
        )
        capability = _mint_profile_user_sql(
            statement,
            backend=self._backend,
            connection_identity=_profile_user_connection_identity(conn),
            operation="insert",
            columns=tuple(values.keys()) + ("profile_version",),
        )
        try:
            result = executor.execute(capability, parameters, connection=conn)
        finally:
            _revoke_profile_user_sql(capability)
        changed = _sync_changed_rows(result)
        if changed == 0 and ignore_conflict:
            return UserWriteResult((), operation_clock)
        _require_exactly_one(changed)
        user_id = _sync_inserted_id(result)
        return UserWriteResult((_validate_inserted_id(user_id),), operation_clock)

    async def _execute_async(
        self,
        conn: Any,
        statement: Any,
        parameters: Sequence[Any],
    ) -> int:
        if self._backend == "postgres":
            status = await conn.execute(statement, *tuple(parameters))
            return _postgres_update_count(status)
        cursor = await conn.execute(statement, tuple(parameters))
        return _sqlite_rowcount(cursor)

    def _operation_clock(self) -> datetime:
        try:
            value = self._clock()
        except Exception:  # noqa: BLE001 - clocks are an injected boundary
            raise ProfileVersionInvalid() from None
        return normalize_profile_version(value)


def _validate_update(
    user_id: int,
    declared_fields: Sequence[str],
    statement: str,
    ownership: UserVersionOwnership,
    parameters: tuple[Any, ...],
    backend: str,
) -> frozenset[str]:
    if type(user_id) is not int or user_id <= 0:
        raise ProfileVersionInvalid()
    if type(ownership) is not UserVersionOwnership:
        raise ProfileVersionInvalid()
    if type(statement) is not str:
        raise ProfileVersionInvalid()
    if isinstance(declared_fields, (str, bytes)):
        raise ProfileVersionInvalid()
    try:
        declared = tuple(declared_fields)
    except Exception:  # noqa: BLE001 - caller-supplied iterables are untrusted metadata
        raise ProfileVersionInvalid() from None
    if any(type(field) is not str for field in declared):
        raise ProfileVersionInvalid()
    if len(set(declared)) != len(declared):
        raise ProfileVersionInvalid()
    declared_set = frozenset(declared)
    if not declared_set <= PROFILE_VISIBLE_USER_FIELDS:
        raise ProfileVersionInvalid()
    statement_fields = _update_columns(statement)
    if not statement_fields <= _WRITABLE_USER_FIELDS:
        raise ProfileVersionInvalid()
    if statement_fields & PROFILE_VISIBLE_USER_FIELDS != declared_set:
        raise ProfileVersionInvalid()
    _validate_update_target(statement, parameters, backend, user_id)
    return declared_set


def _validate_update_parameters(parameters: Sequence[Any]) -> tuple[Any, ...]:
    if isinstance(parameters, (str, bytes)):
        raise ProfileVersionInvalid()
    try:
        return tuple(parameters)
    except Exception:  # noqa: BLE001 - caller metadata is untrusted
        raise ProfileVersionInvalid() from None


def _validate_update_target(
    statement: str,
    parameters: tuple[Any, ...],
    backend: str,
    user_id: int,
) -> None:
    match = _UPDATE_USERS_PATTERN.fullmatch(statement.rstrip().removesuffix(";"))
    if match is None or not parameters:
        raise ProfileVersionInvalid()
    if match.group("table").casefold() not in {
        "users",
        _canonical_users_table(backend),
    }:
        raise ProfileVersionInvalid()
    predicate = match.group("predicate").strip()
    predicate_match = re.fullmatch(
        r'(?:users\.)?(?:"id"|id)\s*=\s*'
        r'(?P<placeholder>\?|%s|\$(?P<index>[1-9][0-9]*))'
        r'(?P<remainder>\s+AND\s+.+)?',
        predicate,
        re.IGNORECASE | re.DOTALL,
    )
    if predicate_match is None:
        raise ProfileVersionInvalid()
    remainder = predicate_match.group("remainder")
    if remainder is not None and re.search(r"\bOR\b", remainder, re.IGNORECASE):
        raise ProfileVersionInvalid()
    placeholder = predicate_match.group("placeholder")
    placeholder_end = (
        match.start("predicate") + predicate_match.end("placeholder")
    )
    if backend == "sqlite":
        if placeholder != "?":
            raise ProfileVersionInvalid()
        index = statement[:placeholder_end].count("?") - 1
    elif backend == "postgres" and placeholder == "%s":
        index = statement[:placeholder_end].count("%s") - 1
    elif backend == "postgres" and predicate_match.group("index") is not None:
        index = int(predicate_match.group("index")) - 1
    else:
        raise ProfileVersionInvalid()
    if index < 0 or index >= len(parameters):
        raise ProfileVersionInvalid()
    bound_user_id = parameters[index]
    if type(bound_user_id) is not int or bound_user_id != user_id:
        raise ProfileVersionInvalid()


def _qualify_update_statement(statement: str, backend: str) -> str:
    stripped = statement.rstrip().removesuffix(";")
    match = _UPDATE_USERS_PATTERN.fullmatch(stripped)
    if match is None:
        raise ProfileVersionInvalid()
    target = match.group("table").casefold()
    canonical = _canonical_users_table(backend)
    if target not in {"users", canonical}:
        raise ProfileVersionInvalid()
    start, end = match.span("table")
    return stripped[:start] + canonical + stripped[end:]


def _update_columns(statement: str) -> frozenset[str]:
    match = _UPDATE_USERS_PATTERN.fullmatch(statement.rstrip().removesuffix(";"))
    if match is None:
        raise ProfileVersionInvalid()
    assignments = _split_assignments(match.group("assignments"))
    if not assignments:
        raise ProfileVersionInvalid()
    columns: list[str] = []
    for assignment in assignments:
        column_match = _ASSIGNMENT_COLUMN_PATTERN.match(assignment)
        if column_match is None:
            raise ProfileVersionInvalid()
        columns.append(column_match.group("quoted") or column_match.group("plain"))
    if len(set(columns)) != len(columns):
        raise ProfileVersionInvalid()
    return frozenset(columns)


def _split_assignments(value: str) -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote: str | None = None
    index = 0
    while index < len(value):
        char = value[index]
        if quote is not None:
            if char == quote:
                if index + 1 < len(value) and value[index + 1] == quote:
                    index += 1
                else:
                    quote = None
        elif char in {"'", '"'}:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ProfileVersionInvalid()
        elif char == "," and depth == 0:
            parts.append(value[start:index])
            start = index + 1
        index += 1
    if quote is not None or depth != 0:
        raise ProfileVersionInvalid()
    parts.append(value[start:])
    if any(not part.strip() for part in parts):
        raise ProfileVersionInvalid()
    return tuple(parts)


def _build_insert(
    backend: str,
    values: Mapping[str, Any],
    operation_clock: datetime,
    *,
    ignore_conflict: bool,
    async_postgres: bool,
) -> tuple[str, tuple[Any, ...]]:
    if not isinstance(values, Mapping):
        raise ProfileVersionInvalid()
    columns = tuple(values.keys())
    if not columns or any(type(column) is not str for column in columns):
        raise ProfileVersionInvalid()
    if len(set(columns)) != len(columns):
        raise ProfileVersionInvalid()
    if "profile_version" in columns or not set(columns) <= _WRITABLE_USER_FIELDS:
        raise ProfileVersionInvalid()
    all_columns = columns + ("profile_version",)
    stored_clock: Any
    if backend == "sqlite":
        placeholder = "?"
        placeholders = ", ".join(placeholder for _ in all_columns)
        prefix = "INSERT OR IGNORE" if ignore_conflict else "INSERT"
        suffix = ""
        stored_clock = _serialize_sqlite(operation_clock)
    else:
        if async_postgres:
            placeholders = ", ".join(
                f"${index}" for index in range(1, len(all_columns) + 1)
            )
        else:
            placeholders = ", ".join("%s" for _ in all_columns)
        prefix = "INSERT"
        suffix = " ON CONFLICT DO NOTHING" if ignore_conflict else ""
        suffix += " RETURNING id"
        stored_clock = operation_clock
    statement = (
        f"{prefix} INTO {_canonical_users_table(backend)} "
        f"({', '.join(all_columns)}) "
        f"VALUES ({placeholders}){suffix}"
    )
    return statement, tuple(values[column] for column in columns) + (stored_clock,)


def _postgres_update_count(status: Any) -> int:
    if type(status) is not str:
        raise ProfileVersionInvalid()
    match = re.fullmatch(r"UPDATE ([0-9]+)", status)
    if match is None:
        raise ProfileVersionInvalid()
    return int(match.group(1))


def _sqlite_rowcount(result: Any) -> int:
    rowcount = getattr(result, "rowcount", None)
    if type(rowcount) is not int:
        raise ProfileVersionInvalid()
    return rowcount


def _sync_changed_rows(result: Any) -> int:
    return _sqlite_rowcount(result)


def _require_single_or_noop(changed: int) -> None:
    if changed not in {0, 1}:
        raise ProfileVersionInvalid()


def _require_exactly_one(changed: int) -> None:
    if changed != 1:
        raise ProfileVersionInvalid()


def _inserted_id(result: Any) -> Any:
    try:
        return result.lastrowid
    except Exception:  # noqa: BLE001 - backend result objects are untrusted metadata
        raise ProfileVersionInvalid() from None


def _sync_inserted_id(result: Any) -> Any:
    try:
        rows = result.rows
    except Exception:  # noqa: BLE001 - backend result objects are untrusted metadata
        raise ProfileVersionInvalid() from None
    if rows:
        if len(rows) != 1:
            raise ProfileVersionInvalid()
        row = rows[0]
        if isinstance(row, dict):
            value = row.get("id")
        else:
            try:
                value = row[0]
            except (KeyError, IndexError, TypeError):
                raise ProfileVersionInvalid() from None
        return value
    return _inserted_id(result)


def _validate_inserted_id(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise ProfileVersionInvalid()
    return value


def _serialize_sqlite(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _postgres_concurrency_sqlstate(error: BaseException) -> str | None:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and len(seen) < 32:
        identity = id(current)
        if identity in seen:
            break
        seen.add(identity)
        sqlstate = _exception_attribute(current, "sqlstate")
        pgcode = _exception_attribute(current, "pgcode")
        for candidate in (sqlstate, pgcode):
            if (
                type(candidate) is str
                and candidate in _POSTGRES_CONCURRENCY_SQLSTATES
            ):
                return candidate
        cause = _exception_attribute(current, "__cause__")
        context = _exception_attribute(current, "__context__")
        suppress_context = _exception_attribute(current, "__suppress_context__")
        if isinstance(cause, BaseException):
            current = cause
        elif suppress_context is not True and isinstance(context, BaseException):
            current = context
        else:
            current = None
    return None


def _exception_attribute(error: BaseException, name: str) -> Any:
    try:
        return getattr(error, name, None)
    except Exception:  # noqa: BLE001 - backend exceptions are untrusted
        return None


__all__ = [
    "PROFILE_VISIBLE_USER_FIELDS",
    "ProfileVersionError",
    "ProfileVersionInvalid",
    "ProfileVersionNotFound",
    "ProfileVersionReadFailed",
    "UserVersionOwnership",
    "UserWriteResult",
    "VersionedUserWriteGateway",
    "compute_touch_value",
    "normalize_profile_version",
]
