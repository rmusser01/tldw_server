"""Create AuthNZ users in tests without tripping the profile-write guard.

``username``, ``email`` and ``is_active`` are profile-visible columns. Since
``feat(authnz): version profile-visible user writes`` (5f31630280),
``profile_user_write_guard`` rejects raw writes touching them on a managed
AuthNZ connection, so the ``INSERT OR IGNORE INTO users (...)`` that test
helpers used to do now raises ``ProfileUserWriteRejected``.

Seed through this instead. It goes via ``UsersDB``, which is the path
production uses and the one the guard sanctions.
"""

from __future__ import annotations

import uuid as _uuid
from typing import Any


async def ensure_test_user(
    pool: Any,
    username: str,
    email: str | None = None,
    *,
    role: str = "user",
    password_hash: str = "x",
    is_active: bool = True,
    is_superuser: bool = False,
) -> int:
    """Return the id of ``username``, creating the user if it is not there yet.

    Idempotent, matching the ``INSERT OR IGNORE`` semantics of the raw inserts
    this replaces: callers seed the same fixed username across tests in a
    session and expect the second call to be a no-op.

    Args:
        pool: An initialized AuthNZ connection pool.
        username: Login name to look up or create.
        email: Address for a newly created user. Defaults to
            ``<username>@example.com``.
        role: Role for a newly created user.
        password_hash: Stored verbatim; these users never authenticate.
        is_active: Active flag for a newly created user.
        is_superuser: Superuser flag for a newly created user.

    Returns:
        The user's integer id.
    """
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    existing = await pool.fetchval("SELECT id FROM users WHERE username = ?", username)
    if existing is not None:
        return int(existing)

    users_db = UsersDB(pool)
    await users_db.initialize()
    created = await users_db.create_user(
        username=username,
        email=email or f"{username}@example.com",
        password_hash=password_hash,
        role=role,
        is_active=is_active,
        is_superuser=is_superuser,
        uuid_value=_uuid.uuid4(),
    )
    return int(created["id"])
