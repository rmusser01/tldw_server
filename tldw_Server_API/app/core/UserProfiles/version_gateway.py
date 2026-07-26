"""Fail-closed profile-version reads over one backend snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from tldw_Server_API.app.core.AuthNZ.profile_version import (
    ProfileVersionInvalid,
    ProfileVersionNotFound,
    ProfileVersionReadFailed,
    normalize_profile_version,
)
from tldw_Server_API.app.core.UserProfiles.backend import (
    ProfileBackendUnavailable,
    resolve_profile_backend,
)

_SQLITE_CANDIDATES_SQL = """
WITH target_user AS (
    SELECT users.id, users.profile_version
    FROM users
    WHERE users.id = ?
),
org_memberships AS (
    SELECT om.org_id
    FROM org_members AS om
    JOIN target_user AS u ON u.id = om.user_id
    WHERE COALESCE(om.status, 'active') = 'active'
),
team_memberships AS (
    SELECT tm.team_id
    FROM team_members AS tm
    JOIN target_user AS u ON u.id = tm.user_id
    WHERE COALESCE(tm.status, 'active') = 'active'
)
SELECT 'user' AS source_tag, id AS source_id, profile_version AS candidate_value
FROM target_user
UNION ALL
SELECT 'org_membership', org_id, NULL
FROM org_memberships
UNION ALL
SELECT 'team_membership', team_id, NULL
FROM team_memberships
UNION ALL
SELECT 'user_override', NULL, uco.updated_at
FROM user_config_overrides AS uco
JOIN target_user AS u ON u.id = uco.user_id
UNION ALL
SELECT 'org_override', oco.org_id, oco.updated_at
FROM org_config_overrides AS oco
JOIN org_memberships AS om ON om.org_id = oco.org_id
UNION ALL
SELECT 'team_override', tco.team_id, tco.updated_at
FROM team_config_overrides AS tco
JOIN team_memberships AS tm ON tm.team_id = tco.team_id
""".strip()


def _postgres_candidates_sql(*, lock_user: bool) -> str:
    # This closed boolean choice cannot introduce user-controlled SQL.
    lock_clause = " FOR UPDATE" if lock_user else ""
    query = f"""
WITH locked_user AS (
    SELECT users.id, users.profile_version
    FROM users
    WHERE users.id = $1{lock_clause}
),
org_memberships AS (
    SELECT om.org_id
    FROM org_members AS om
    JOIN locked_user AS u ON u.id = om.user_id
    WHERE COALESCE(om.status, 'active') = 'active'
),
team_memberships AS (
    SELECT tm.team_id
    FROM team_members AS tm
    JOIN locked_user AS u ON u.id = tm.user_id
    WHERE COALESCE(tm.status, 'active') = 'active'
)
SELECT 'user' AS source_tag, id AS source_id, profile_version AS candidate_value
FROM locked_user
UNION ALL
SELECT 'org_membership', org_id, NULL::TIMESTAMPTZ
FROM org_memberships
UNION ALL
SELECT 'team_membership', team_id, NULL::TIMESTAMPTZ
FROM team_memberships
UNION ALL
SELECT 'user_override', NULL::INTEGER, uco.updated_at
FROM user_config_overrides AS uco
JOIN locked_user AS u ON u.id = uco.user_id
UNION ALL
SELECT 'org_override', oco.org_id, oco.updated_at
FROM org_config_overrides AS oco
JOIN org_memberships AS om ON om.org_id = oco.org_id
UNION ALL
SELECT 'team_override', tco.team_id, tco.updated_at
FROM team_config_overrides AS tco
JOIN team_memberships AS tm ON tm.team_id = tco.team_id
""".strip()  # nosec B608
    return query


@dataclass(frozen=True, slots=True)
class ProfileVersionCandidates:
    """Immutable, normalized timestamp candidates from one database snapshot."""

    user_exists: bool
    values: tuple[datetime, ...]

    @property
    def maximum(self) -> datetime:
        if not self.user_exists or not self.values:
            raise ProfileVersionNotFound()
        return max(self.values)


class ProfileVersionGatewayProtocol(Protocol):
    async def read(self, user_id: int) -> datetime: ...

    async def read_in_transaction(
        self,
        conn: Any,
        user_id: int,
        *,
        lock_user: bool,
    ) -> datetime: ...

    async def touch(self, conn: Any, user_id: int, value: datetime) -> None: ...


class ProfileVersionGateway:
    """Read and advance the durable composite profile-version anchor."""

    def __init__(self, db_pool: Any) -> None:
        self._db_pool = db_pool

    async def read(self, user_id: int) -> datetime:
        """Acquire exactly one connection and read one complete snapshot."""
        try:
            is_postgres = self._is_postgres_backend()
            async with self._db_pool.acquire() as conn:
                return await self._read(
                    conn,
                    user_id,
                    lock_user=False,
                    is_postgres=is_postgres,
                )
        except (ProfileVersionNotFound, ProfileVersionInvalid):
            raise
        except Exception as exc:  # noqa: BLE001 - sanitize storage failures
            raise self._storage_failure(exc) from None

    async def read_in_transaction(
        self,
        conn: Any,
        user_id: int,
        *,
        lock_user: bool,
    ) -> datetime:
        """Read through only the caller-supplied transaction connection."""
        try:
            is_postgres = self._is_postgres_backend()
            return await self._read(
                conn,
                user_id,
                lock_user=lock_user,
                is_postgres=is_postgres,
            )
        except (ProfileVersionNotFound, ProfileVersionInvalid):
            raise
        except Exception as exc:  # noqa: BLE001 - sanitize storage failures
            raise self._storage_failure(exc) from None

    async def touch(self, conn: Any, user_id: int, value: datetime) -> None:
        """Write one explicit UTC anchor value on the supplied connection."""
        normalized = normalize_profile_version(value)
        try:
            is_postgres = self._is_postgres_backend()
            if is_postgres:
                result = await conn.execute(
                    "UPDATE users SET profile_version = $1 WHERE id = $2",
                    normalized,
                    user_id,
                )
                changed = _postgres_changed_rows(result)
            else:
                serialized = normalized.astimezone(timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%S.%fZ"
                )
                cursor = await conn.execute(
                    "UPDATE users SET profile_version = ? WHERE id = ?",
                    (serialized, user_id),
                )
                changed = _sqlite_changed_rows(cursor)
        except (ProfileVersionNotFound, ProfileVersionInvalid):
            raise
        except Exception as exc:  # noqa: BLE001 - sanitize storage failures
            raise self._storage_failure(exc) from None

        if changed != 1:
            raise ProfileVersionNotFound()

    async def _read(
        self,
        conn: Any,
        user_id: int,
        *,
        lock_user: bool,
        is_postgres: bool,
    ) -> datetime:
        if is_postgres:
            rows = await conn.fetch(
                _postgres_candidates_sql(lock_user=lock_user),
                user_id,
            )
        else:
            cursor = await conn.execute(_SQLITE_CANDIDATES_SQL, (user_id,))
            rows = await cursor.fetchall()
        return _parse_candidates(rows, allow_naive=not is_postgres).maximum

    def _storage_failure(self, error: BaseException) -> ProfileVersionReadFailed:
        try:
            is_postgres = self._is_postgres_backend()
        except ProfileVersionReadFailed:
            is_postgres = False
        if is_postgres:
            return ProfileVersionReadFailed.from_storage_error(error)
        return ProfileVersionReadFailed()

    def _is_postgres_backend(self) -> bool:
        try:
            return resolve_profile_backend(self._db_pool) == "postgres"
        except ProfileBackendUnavailable:
            raise ProfileVersionReadFailed() from None


def _parse_candidates(
    rows: Any,
    *,
    allow_naive: bool,
) -> ProfileVersionCandidates:
    user_exists = False
    values: list[datetime] = []
    timestamp_sources = {
        "user",
        "user_override",
        "org_override",
        "team_override",
    }
    membership_sources = {"org_membership", "team_membership"}

    for row in rows:
        source_tag = _row_value(row, "source_tag", 0)
        source_id = _row_value(row, "source_id", 1)
        candidate = _row_value(row, "candidate_value", 2)
        if type(source_tag) is not str:
            raise ProfileVersionInvalid() from None
        if source_tag not in timestamp_sources | membership_sources:
            raise ProfileVersionInvalid()
        if source_tag == "user_override":
            if source_id is not None:
                raise ProfileVersionInvalid() from None
        elif type(source_id) is not int:
            raise ProfileVersionInvalid() from None
        if source_tag in membership_sources:
            if candidate is not None:
                raise ProfileVersionInvalid()
            continue
        if source_tag == "user":
            if user_exists:
                raise ProfileVersionInvalid()
            user_exists = True
        if candidate is None:
            raise ProfileVersionInvalid()
        values.append(
            normalize_profile_version(candidate, allow_naive=allow_naive)
        )

    return ProfileVersionCandidates(user_exists=user_exists, values=tuple(values))


def _row_value(row: Any, key: str, index: int) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    try:
        return row[key]
    except (KeyError, IndexError, TypeError):
        try:
            return row[index]
        except (KeyError, IndexError, TypeError):
            raise ProfileVersionInvalid() from None


def _postgres_changed_rows(result: Any) -> int:
    if type(result) is not str or not result.startswith("UPDATE "):
        raise ProfileVersionReadFailed()
    count = result.removeprefix("UPDATE ")
    if not count or not count.isascii() or not count.isdecimal():
        raise ProfileVersionReadFailed()
    return int(count)


def _sqlite_changed_rows(cursor: Any) -> int:
    rowcount = getattr(cursor, "rowcount", None)
    if type(rowcount) is not int:
        raise ProfileVersionReadFailed()
    return rowcount


__all__ = [
    "ProfileVersionCandidates",
    "ProfileVersionGateway",
    "ProfileVersionGatewayProtocol",
]
