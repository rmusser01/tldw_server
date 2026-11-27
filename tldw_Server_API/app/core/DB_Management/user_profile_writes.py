"""DB-owned statements for profile-visible AuthNZ user writes."""

from __future__ import annotations

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
    nothing changes (empty ``affected_user_ids``). TASK-15671.
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
