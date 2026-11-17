"""AuthNZ-owned retained-byte reservations; legacy counters never include these bytes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from tldw_Server_API.app.core.AuthNZ.exceptions import StorageError
from tldw_Server_API.app.core.DB_Management.vn_generated_files_queries import VNGeneratedFilesQueries

if TYPE_CHECKING:
    from tldw_Server_API.app.core.AuthNZ.database import DatabasePool


def native_quota_schema_statements(*, postgres: bool) -> tuple[str, ...]:
    """Return additive schema shared by packaged and runtime AuthNZ bootstrap."""
    integer = "BIGINT" if postgres else "INTEGER"
    return (
        f"""CREATE TABLE IF NOT EXISTS native_chat_quota_reservations (
            user_id {integer} NOT NULL,
            user_uuid TEXT NOT NULL,
            owner_key TEXT NOT NULL,
            candidate_key TEXT NOT NULL,
            scope_type TEXT NOT NULL CHECK (scope_type IN ('global', 'workspace')),
            workspace_id TEXT,
            org_id {integer},
            team_id {integer},
            size_bytes {integer} NOT NULL CHECK (size_bytes >= 0),
            state TEXT NOT NULL CHECK (state IN ('active', 'released')),
            PRIMARY KEY (user_id, candidate_key),
            CHECK (team_id IS NULL OR org_id IS NOT NULL),
            CHECK ((scope_type = 'global' AND workspace_id IS NULL) OR (scope_type = 'workspace' AND workspace_id IS NOT NULL))
        )""",
        "CREATE INDEX IF NOT EXISTS native_chat_quota_user ON native_chat_quota_reservations(user_id, user_uuid, state)",
        "CREATE INDEX IF NOT EXISTS native_chat_quota_org ON native_chat_quota_reservations(org_id, state)",
        "CREATE INDEX IF NOT EXISTS native_chat_quota_team ON native_chat_quota_reservations(team_id, state)",
    )


class NativeQuotaConflictError(StorageError):
    """Original owner/scope/size identity conflicts with the retained allocation."""


class NativeQuotaUnavailableError(StorageError):
    """Unavailable retained usage cannot be reported as an empty native ledger."""


@dataclass(frozen=True)
class NativeQuotaOwner:
    """Authenticated destination identity, never request-supplied quota selectors."""

    user_id: int
    owner_key: str
    user_uuid: str
    scope_type: str = "global"
    workspace_id: str | None = None
    org_id: int | None = None
    team_id: int | None = None

    def __post_init__(self) -> None:
        """Reject ambiguous IDs before any quota query."""
        from tldw_Server_API.app.core.Chat.native_fork_schemas import NativeScopeV1

        scope = {"kind": self.scope_type}
        if self.workspace_id is not None:
            scope["workspace_id"] = self.workspace_id
        NativeScopeV1.model_validate(scope)
        if (
            type(self.user_id) is not int
            or self.user_id < 1
            or not isinstance(self.owner_key, str)
            or not 1 <= len(self.owner_key) <= 256
            or not isinstance(self.user_uuid, str)
            or not 1 <= len(self.user_uuid) <= 64
            or any(value is not None and (type(value) is not int or value < 1) for value in (self.org_id, self.team_id))
            or self.team_id is not None
            and self.org_id is None
        ):
            raise ValueError("native_quota_owner_required")


@dataclass(frozen=True)
class NativeQuotaReservation:
    """Immutable original allocation; released keys remain terminal."""

    owner: NativeQuotaOwner
    candidate_key: str
    size_bytes: int
    state: str


class NativeQuotaReservationStore:
    """Queries run on the existing AuthNZ pool or its owning transaction connection."""

    def __init__(self, db_pool: DatabasePool) -> None:
        self.db_pool = db_pool

    @staticmethod
    def require_key(candidate_key: str) -> None:
        """Keep original candidate identity bounded and exact."""
        if not isinstance(candidate_key, str) or not 1 <= len(candidate_key) <= 256:
            raise ValueError("native_quota_candidate_required")

    @asynccontextmanager
    async def transaction(self, owner: NativeQuotaOwner) -> AsyncIterator[NativeQuotaReservationStore]:
        """Reuse user/org/team lock order; SQLite owns BEGIN IMMEDIATE."""
        from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import _TransactionBoundPool

        async with self.db_pool.transaction() as conn:
            postgres = getattr(self.db_pool, "pool", None) is not None
            await VNGeneratedFilesQueries(conn, postgres=postgres).lock_quota_scopes(
                user_id=owner.user_id,
                org_id=owner.org_id,
                team_id=owner.team_id,
            )
            yield NativeQuotaReservationStore(_TransactionBoundPool(conn, postgres=postgres))

    async def user(self, owner: NativeQuotaOwner) -> dict[str, Any]:
        """Bind an allocation to the current account incarnation without cascade release."""
        row = await self.db_pool.fetchone(
            "SELECT uuid, is_active, storage_used_mb FROM users WHERE id = ?", owner.user_id
        )
        if row is None or str(row.get("uuid") or "") != owner.user_uuid:
            raise NativeQuotaConflictError("native_quota_user_unavailable")
        return row

    async def validate_scope(self, owner: NativeQuotaOwner) -> None:
        """Require active authenticated membership in every applicable destination pool."""
        if owner.org_id is not None:
            row = await self.db_pool.fetchone(
                "SELECT o.id FROM organizations o JOIN org_members m ON m.org_id = o.id "
                "WHERE o.id = ? AND m.user_id = ? AND o.is_active = ? AND COALESCE(m.status, 'active') = 'active'",
                owner.org_id,
                owner.user_id,
                True,
            )
            if row is None:
                raise NativeQuotaConflictError("native_quota_scope_unavailable")
        if owner.team_id is not None:
            row = await self.db_pool.fetchone(
                "SELECT t.id FROM teams t JOIN team_members m ON m.team_id = t.id "
                "WHERE t.id = ? AND t.org_id = ? AND m.user_id = ? AND t.is_active = ? "
                "AND COALESCE(m.status, 'active') = 'active'",
                owner.team_id,
                owner.org_id,
                owner.user_id,
                True,
            )
            if row is None:
                raise NativeQuotaConflictError("native_quota_scope_unavailable")

    async def resolve(self, owner: NativeQuotaOwner, candidate_key: str) -> NativeQuotaReservation | None:
        """Resolve only the exact key; differing frozen scope is a conflict."""
        self.require_key(candidate_key)
        row = await self.db_pool.fetchone(
            "SELECT * FROM native_chat_quota_reservations WHERE user_id = ? AND candidate_key = ?",
            owner.user_id,
            candidate_key,
        )
        if row is None:
            return None
        if (
            row["owner_key"] != owner.owner_key
            or row["user_uuid"] != owner.user_uuid
            or row["org_id"] != owner.org_id
            or row["team_id"] != owner.team_id
            or row["scope_type"] != owner.scope_type
            or row["workspace_id"] != owner.workspace_id
        ):
            raise NativeQuotaConflictError("native_quota_identity_conflict")
        return NativeQuotaReservation(owner, candidate_key, int(row["size_bytes"]), row["state"])

    async def insert(self, owner: NativeQuotaOwner, candidate_key: str, size_bytes: int) -> NativeQuotaReservation:
        """Allocate once after the caller has checked its serialized current budget."""
        await self.db_pool.execute(
            "INSERT INTO native_chat_quota_reservations "
            "(user_id,user_uuid,owner_key,candidate_key,scope_type,workspace_id,org_id,team_id,size_bytes,state) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'active')",
            owner.user_id,
            owner.user_uuid,
            owner.owner_key,
            candidate_key,
            owner.scope_type,
            owner.workspace_id,
            owner.org_id,
            owner.team_id,
            size_bytes,
        )
        return NativeQuotaReservation(owner, candidate_key, size_bytes, "active")

    async def release(self, owner: NativeQuotaOwner, candidate_key: str) -> NativeQuotaReservation | None:
        """Keep terminal identity; an absent allocation remains unknown."""
        await self.db_pool.execute(
            "UPDATE native_chat_quota_reservations SET state = 'released' "
            "WHERE user_id = ? AND owner_key = ? AND candidate_key = ? AND state = 'active'",
            owner.user_id,
            owner.owner_key,
            candidate_key,
        )
        return await self.resolve(owner, candidate_key)

    async def org_member_usage(self, org_id: int) -> int:
        """Preserve Billing's active-member user aggregation independently of pool attribution."""
        try:
            row = await self.db_pool.fetchone(
                "SELECT COALESCE(SUM(n.size_bytes),0) AS bytes FROM native_chat_quota_reservations n "
                "JOIN users u ON u.id = n.user_id AND CAST(u.uuid AS TEXT) = n.user_uuid "
                "JOIN org_members m ON m.user_id = u.id "
                "WHERE m.org_id = ? AND m.status = 'active' AND n.state = 'active'",
                org_id,
            )
            if (
                row is None
                or type(row.get("bytes")) not in (int, Decimal)
                or row["bytes"] < 0
                or int(row["bytes"]) != row["bytes"]
            ):
                raise ValueError("invalid_native_quota_usage")
            return int(row["bytes"])
        except Exception as exc:
            raise NativeQuotaUnavailableError("Native retained storage usage is unavailable") from exc

    async def usage(
        self, *, user_id: int | None = None, org_id: int | None = None, team_id: int | None = None
    ) -> dict[str, int]:
        """Read fresh active bytes; SQL/read failure is never treated as zero."""
        if user_id is not None:
            query = (
                "SELECT COALESCE(SUM(n.size_bytes),0) AS bytes, COUNT(*) AS count "
                "FROM native_chat_quota_reservations n JOIN users u ON u.id = n.user_id AND CAST(u.uuid AS TEXT) = n.user_uuid "
                "WHERE u.id = ? AND n.state = 'active'"
            )
            identity = user_id
        elif org_id is not None:
            query = "SELECT COALESCE(SUM(size_bytes),0) AS bytes, COUNT(*) AS count FROM native_chat_quota_reservations WHERE org_id = ? AND state = 'active'"
            identity = org_id
        elif team_id is not None:
            query = "SELECT COALESCE(SUM(size_bytes),0) AS bytes, COUNT(*) AS count FROM native_chat_quota_reservations WHERE team_id = ? AND state = 'active'"
            identity = team_id
        else:
            raise ValueError("native_quota_owner_required")
        try:
            row = await self.db_pool.fetchone(query, identity)
            if (
                row is None
                or type(row.get("bytes")) not in (int, Decimal)
                or type(row.get("count")) is not int
                or row["bytes"] < 0
                or int(row["bytes"]) != row["bytes"]
            ):
                raise ValueError("invalid_native_quota_usage")
            return {"bytes": int(row["bytes"]), "count": row["count"]}
        except Exception as exc:
            raise NativeQuotaUnavailableError("Native retained storage usage is unavailable") from exc


async def native_storage_system_stats(conn: Any, *, postgres: bool) -> dict[str, Any]:
    """Read user aggregate storage on the current admin connection without raw-L totals."""
    from tldw_Server_API.app.core.AuthNZ.repos.generated_files_repo import _TransactionBoundPool

    try:
        row = await _TransactionBoundPool(conn, postgres=postgres).fetchone(
            """SELECT SUM(u.storage_used_mb + COALESCE(n.bytes, 0) / 1048576.0) AS total_used_mb,
                       AVG(u.storage_used_mb + COALESCE(n.bytes, 0) / 1048576.0) AS avg_used_mb,
                       MAX(u.storage_used_mb + COALESCE(n.bytes, 0) / 1048576.0) AS max_used_mb
                FROM users u LEFT JOIN (
                    SELECT user_id, user_uuid, SUM(size_bytes) AS bytes
                    FROM native_chat_quota_reservations WHERE state = 'active'
                    GROUP BY user_id, user_uuid
                ) n ON n.user_id = u.id AND n.user_uuid = CAST(u.uuid AS TEXT)
                WHERE u.is_active = ?""",
            True,
        )
    except Exception as exc:
        raise NativeQuotaUnavailableError("Native retained storage usage is unavailable") from exc
    if row is None or not {"total_used_mb", "avg_used_mb", "max_used_mb"}.issubset(row):
        raise NativeQuotaUnavailableError("Native retained storage usage is unavailable")
    return row
