"""Create a tenant (user + organization + membership) in one transaction (TASK-13317).

Moved out of the admin tenant-provisioning endpoint, which now only maps errors to HTTP.
"""

from __future__ import annotations

from typing import Any

from tldw_Server_API.app.core.AuthNZ.exceptions import DuplicateUserError
from tldw_Server_API.app.core.AuthNZ.profile_version import VersionedUserWriteGateway


async def provision_tenant(
    pool: Any,
    *,
    username: str,
    email: str,
    password_hash: str,
    org_name: str,
    role: str,
) -> tuple[int, int]:
    """Return (user_id, org_id). Raises DuplicateUserError("username") if the name is taken."""
    is_postgres = getattr(pool, "pool", None) is not None
    backend = "postgres" if is_postgres else "sqlite"
    async with pool.transaction() as conn:
        if is_postgres:
            existing = await conn.fetchrow("SELECT id FROM public.users WHERE username = $1", username)
        else:
            cur = await conn.execute("SELECT id FROM main.users WHERE username = ?", (username,))
            existing = await cur.fetchone()
        if existing:
            raise DuplicateUserError("username")

        insert_result = await VersionedUserWriteGateway(backend).insert_user(
            conn,
            values={"username": username, "email": email, "password_hash": password_hash, "is_active": True},
        )
        user_id = insert_result.affected_user_ids[0]

        if is_postgres:
            row = await conn.fetchrow(
                "INSERT INTO public.organizations (name, owner_user_id) VALUES ($1, $2) RETURNING id",
                org_name,
                user_id,
            )
            if not row:
                raise RuntimeError("Tenant organization insert returned no id")
            org_id = int(row["id"])
            await conn.execute(
                "INSERT INTO public.org_members (org_id, user_id, role) VALUES ($1, $2, $3)",
                org_id,
                user_id,
                role,
            )
        else:
            cur = await conn.execute(
                "INSERT INTO main.organizations (name, owner_user_id) VALUES (?, ?)", (org_name, user_id)
            )
            org_id = int(cur.lastrowid)
            await conn.execute(
                "INSERT INTO main.org_members (org_id, user_id, role) VALUES (?, ?, ?)", (org_id, user_id, role)
            )
    return user_id, org_id
