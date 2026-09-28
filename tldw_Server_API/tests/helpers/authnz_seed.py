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
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any


async def ensure_test_user(
    pool: Any,
    username: str,
    email: str | None = None,
    *,
    role: str = "user",
    password_hash: str = "x",
    is_active: bool = True,
    is_verified: bool = False,
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
        is_verified: Verified flag for a newly created user.
        is_superuser: Superuser flag for a newly created user.

    Returns:
        The user's integer id.
    """
    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    users_db = UsersDB(pool)
    await users_db.initialize()

    existing = await users_db.get_user_by_username(username)
    if existing is not None:
        return int(existing["id"])

    created = await users_db.create_user(
        username=username,
        email=email or f"{username}@example.com",
        password_hash=password_hash,
        role=role,
        is_active=is_active,
        is_verified=is_verified,
        is_superuser=is_superuser,
        uuid_value=_uuid.uuid4(),
    )
    return int(created["id"])


@asynccontextmanager
async def unmanaged_authnz_pg_connection(database: str | None = None) -> AsyncIterator[Any]:
    """Open a plain asyncpg connection to the AuthNZ Postgres test database.

    For tests that must fabricate state the app itself can never write: a
    database left behind by an older release, or one damaged out of band
    (legacy columns, NULL activity flags, views or tables shadowing
    ``users``). The guarded ``DatabasePool`` rightly refuses those writes, so
    such tests set the state up the way an old server or an operator would,
    on a connection the app does not manage. Everything else seeds through
    :func:`ensure_test_user`.

    Args:
        database: Database to connect to. Defaults to the shared
            ``TEST_DB_NAME`` behind ``test_db_pool``; pass the per-test name
            from ``isolated_test_environment`` to target that database.
    """
    import asyncpg

    from tldw_Server_API.tests.AuthNZ.conftest import (
        TEST_DB_HOST,
        TEST_DB_NAME,
        TEST_DB_PASSWORD,
        TEST_DB_PORT,
        TEST_DB_USER,
    )

    conn = await asyncpg.connect(
        host=TEST_DB_HOST,
        port=TEST_DB_PORT,
        user=TEST_DB_USER,
        password=TEST_DB_PASSWORD,
        database=database or TEST_DB_NAME,
    )
    try:
        yield conn
    finally:
        await conn.close()
