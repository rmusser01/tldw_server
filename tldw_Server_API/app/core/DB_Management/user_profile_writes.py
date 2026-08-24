"""DB-owned statements for profile-visible AuthNZ user writes."""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.AuthNZ.profile_version import (
    UserWriteResult,
    VersionedUserWriteGateway,
)


async def update_user_email(
    connection: Any,
    *,
    backend: str,
    user_id: int,
    email: str,
) -> UserWriteResult:
    """Update one user's email through the versioned write boundary."""
    statement = (
        "UPDATE users SET email = $1, updated_at = CURRENT_TIMESTAMP WHERE id = $2"
        if backend == "postgres"
        else "UPDATE users SET email = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?"
    )
    return await VersionedUserWriteGateway(backend).execute_update(
        connection,
        user_id=user_id,
        profile_visible_fields=("email",),
        statement=statement,
        parameters=(email, user_id),
    )
