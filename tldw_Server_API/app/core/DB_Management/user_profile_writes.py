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
    " AND email_changed_at > datetime('now', '-7 days') THEN previous_email ELSE email END,"
    " email_changed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP"
    " WHERE id = ? AND LOWER(email) <> LOWER(?)"
)
_UPDATE_EMAIL_POSTGRES = (
    "UPDATE users SET email = $1, previous_email = CASE WHEN previous_email IS NOT NULL"
    " AND email_changed_at > CURRENT_TIMESTAMP - INTERVAL '7 days' THEN previous_email ELSE email END,"
    " email_changed_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP"
    " WHERE id = $2 AND LOWER(email) <> LOWER($3)"
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
