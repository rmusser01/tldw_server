"""DB-owned statements for profile-visible AuthNZ user writes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from tldw_Server_API.app.core.AuthNZ.profile_version import (
    UserVersionOwnership,
    UserWriteResult,
    VersionedUserWriteGateway,
)

# SET right-hand sides read the old row on both backends.
_UPDATE_EMAIL_SQLITE = (
    "UPDATE users SET email = ?, previous_email = CASE WHEN previous_email IS NOT NULL"
    " AND email_changed_at > datetime('now', '-7 days') THEN previous_email ELSE NULLIF(email, '') END,"
    " email_changed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP"
    " WHERE id = ? AND LOWER(email) <> LOWER(?)"
)
_UPDATE_EMAIL_POSTGRES = (
    "UPDATE users SET email = $1, previous_email = CASE WHEN previous_email IS NOT NULL"
    " AND email_changed_at > CURRENT_TIMESTAMP - INTERVAL '7 days' THEN previous_email ELSE NULLIF(email, '') END,"
    " email_changed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP"
    " WHERE id = $2 AND LOWER(email) <> LOWER($3)"
)


_RESTORE_PREVIOUS_EMAIL_SQLITE = (
    "UPDATE users SET email = ?, previous_email = NULL, email_changed_at = CURRENT_TIMESTAMP,"
    " password_hash = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ? AND LOWER(previous_email) = LOWER(?)"
)
_RESTORE_PREVIOUS_EMAIL_POSTGRES = (
    "UPDATE users SET email = $1, previous_email = NULL, email_changed_at = CURRENT_TIMESTAMP,"
    " password_hash = $2, updated_at = CURRENT_TIMESTAMP WHERE id = $3 AND LOWER(previous_email) = LOWER($4)"
)


async def restore_previous_email(
    connection: Any,
    *,
    backend: str,
    user_id: int,
    previous_email: str,
    password_hash: str,
    gateway: VersionedUserWriteGateway | None = None,
    ownership: UserVersionOwnership = UserVersionOwnership.GATEWAY_OWNS_ANCHOR,
) -> UserWriteResult:
    """Make the previous address the account's email again and replace the password (account take-back).

    Only when ``previous_email`` is still the stored previous address (any letter case); otherwise
    nothing changes (empty ``affected_user_ids``). TASK-15679.
    """
    statement = _RESTORE_PREVIOUS_EMAIL_POSTGRES if backend == "postgres" else _RESTORE_PREVIOUS_EMAIL_SQLITE
    return await (gateway or VersionedUserWriteGateway(backend)).execute_update(
        connection,
        user_id=user_id,
        profile_visible_fields=("email",),
        statement=statement,
        parameters=(previous_email, password_hash, user_id, previous_email),
        ownership=ownership,
    )


async def update_user_email(
    connection: Any,
    *,
    backend: str,
    user_id: int,
    email: str,
    gateway: VersionedUserWriteGateway | None = None,
    ownership: UserVersionOwnership = UserVersionOwnership.GATEWAY_OWNS_ANCHOR,
) -> UserWriteResult:
    """Change one user's email and record the previous address.

    An address equal in any letter case is no change (empty ``affected_user_ids``).
    Within 7 days of the last change the original previous address is kept, so
    repeated changes cannot push the owner's address out (PR 3 sends lock mail to it).
    """
    statement = _UPDATE_EMAIL_POSTGRES if backend == "postgres" else _UPDATE_EMAIL_SQLITE
    return await (gateway or VersionedUserWriteGateway(backend)).execute_update(
        connection,
        user_id=user_id,
        profile_visible_fields=("email",),
        statement=statement,
        parameters=(email, user_id, email),
        ownership=ownership,
    )


async def read_lock_state_for_identity_write(connection: Any, *, backend: str, user_id: int) -> dict[str, Any] | None:
    """The user's ``security_locked_at`` and ``credentials_valid_after``, read on the caller's transaction.

    On PostgreSQL the row stays locked until that transaction ends, so an account lock or take-back (whose
    first write updates this row) waits for it. ``FOR NO KEY UPDATE``, not ``FOR UPDATE``: the latter also
    blocks the ``KEY SHARE`` of a foreign-key insert on another connection, and the password reset revokes
    sessions (blacklist rows) on one while it holds this lock. On SQLite the request transaction already holds
    the write lock. None when there is no row (TASK-15679).
    """
    if backend == "postgres":
        row = await connection.fetchrow(
            "SELECT security_locked_at, credentials_valid_after FROM users WHERE id = $1 FOR NO KEY UPDATE",
            user_id,
        )
    else:
        cursor = await connection.execute(
            "SELECT security_locked_at, credentials_valid_after FROM users WHERE id = ?", (user_id,)
        )
        row = await cursor.fetchone()
    if not row:
        return None
    if isinstance(row, dict):
        return {key: row.get(key) for key in ("security_locked_at", "credentials_valid_after")}
    return {"security_locked_at": row[0], "credentials_valid_after": row[1]}


async def email_recently_released_by_another_user(
    db_conn: Any, emails: Sequence[str], *, user_id: int, is_postgres: bool
) -> bool:
    """Whether another user's ``previous_email`` is one of ``emails`` (any case), changed in the last 7 days.

    Reads on the caller's connection only. Its owner may still take such an address back (TASK-15679).
    """
    addresses = sorted({str(value).strip().lower() for value in emails if str(value).strip()})
    if not addresses:
        return False
    if is_postgres:
        found = await db_conn.fetchval(
            "SELECT 1 FROM users WHERE id <> $1 AND LOWER(previous_email) = ANY($2::text[])"
            " AND email_changed_at > CURRENT_TIMESTAMP - INTERVAL '7 days' LIMIT 1",
            user_id,
            addresses,
        )
        return found is not None
    placeholders = ", ".join("?" for _ in addresses)
    cursor = await db_conn.execute(
        f"SELECT 1 FROM users WHERE id <> ? AND LOWER(previous_email) IN ({placeholders})"  # nosec B608 - placeholders only
        " AND datetime(email_changed_at) > datetime('now', '-7 days') LIMIT 1",
        (user_id, *addresses),
    )
    return await cursor.fetchone() is not None
