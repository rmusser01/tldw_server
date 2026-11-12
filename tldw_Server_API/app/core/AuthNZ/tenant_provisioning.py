"""Create a tenant (user + organization + membership) in one transaction (TASK-13317).

Moved out of the admin tenant-provisioning endpoint, which now only maps errors to HTTP.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from tldw_Server_API.app.core.AuthNZ.exceptions import DuplicateUserError
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    ActorMembershipWriteContext,
    AnchorOwnership,
    MembershipAuthority,
    MembershipWriter,
    MembershipWriterContractError,
)
from tldw_Server_API.app.core.AuthNZ.profile_version import VersionedUserWriteGateway
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import AuthnzOrgsTeamsRepo
from tldw_Server_API.app.core.AuthNZ.transaction_policy import get_authnz_transaction_policy


async def provision_tenant(
    pool: Any,
    *,
    actor_user_id: int,
    username: str,
    email: str,
    password_hash: str,
    org_name: str,
    role: Literal["owner"],
) -> tuple[int, int]:
    """Return (user_id, org_id). Raises DuplicateUserError("username") if the name is taken."""
    if role != "owner":
        raise MembershipWriterContractError()
    context = ActorMembershipWriteContext(
        actor_user_id=actor_user_id,
        required_authority=MembershipAuthority.PLATFORM_ADMIN,
    )
    is_postgres = getattr(pool, "pool", None) is not None
    backend = "postgres" if is_postgres else "sqlite"
    async with pool.transaction(
        acquire_timeout_seconds=get_authnz_transaction_policy().db_pool_acquire_timeout_seconds,
    ) as conn:
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

        await MembershipWriter(pool).authorize_organization_creation(
            conn=conn,
            context=context,
            owner_user_id=user_id,
        )

        if is_postgres:
            row = await conn.fetchrow(
                "INSERT INTO public.organizations (name, owner_user_id) VALUES ($1, $2) RETURNING id",
                org_name,
                user_id,
            )
            if not row:
                raise RuntimeError("Tenant organization insert returned no id")
            org_id = int(row["id"])
        else:
            cur = await conn.execute(
                "INSERT INTO main.organizations (name, owner_user_id) VALUES (?, ?)", (org_name, user_id)
            )
            org_id = int(cur.lastrowid)
        await AuthnzOrgsTeamsRepo(pool).provision_org_membership_on_connection(
            conn=conn,
            org_id=org_id,
            user_id=user_id,
            org_role=role,
            team_id=None,
            team_role=None,
            team_failure_is_best_effort=False,
            context=context,
            anchor_ownership=AnchorOwnership.WRITER_OWNS_ANCHOR,
            operation_time=datetime.now(timezone.utc),
        )
    return user_id, org_id
