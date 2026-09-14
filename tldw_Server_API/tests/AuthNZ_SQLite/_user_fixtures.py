"""Authorized user-record setup helpers for SQLite AuthNZ tests."""

from __future__ import annotations

import uuid
from typing import Any

from tldw_Server_API.app.core.AuthNZ.profile_version import VersionedUserWriteGateway


async def create_authnz_test_user(
    pool: Any,
    *,
    username: str,
    email: str,
    password_hash: str = "x",
    role: str = "user",
    is_active: bool = True,
    is_verified: bool = False,
    is_superuser: bool = False,
    user_id: int | None = None,
    ignore_conflict: bool = False,
) -> int:
    """Create a test user through the production versioned-write gateway."""

    backend = "postgres" if getattr(pool, "pool", None) is not None else "sqlite"
    values: dict[str, Any] = {
        "uuid": str(uuid.uuid4()),
        "username": username,
        "email": email.lower(),
        "password_hash": password_hash,
        "role": role,
        "is_active": is_active if backend == "postgres" else int(is_active),
        "is_verified": is_verified if backend == "postgres" else int(is_verified),
        "is_superuser": is_superuser if backend == "postgres" else int(is_superuser),
        "storage_quota_mb": 5120,
    }
    if user_id is not None:
        values["id"] = user_id

    async with pool.transaction() as conn:
        result = await VersionedUserWriteGateway(backend).insert_user(
            conn,
            values=values,
            ignore_conflict=ignore_conflict,
        )

    if result.affected_user_ids:
        return int(result.affected_user_ids[0])
    if ignore_conflict and user_id is not None:
        return user_id
    raise AssertionError("authorized test-user insert did not return a user id")


async def set_authnz_test_user_active(pool: Any, user_id: int, active: bool) -> None:
    """Update test-user activity through the production user abstraction."""

    from tldw_Server_API.app.core.DB_Management.Users_DB import UsersDB

    users_db = UsersDB(pool)
    await users_db.initialize(ensure_schema=False)
    await users_db.update_user(user_id, is_active=active)
