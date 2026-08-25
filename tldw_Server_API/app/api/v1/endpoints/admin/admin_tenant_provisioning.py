"""
Tenant provisioning endpoint.

Provides a single API call that chains user creation, org creation,
and role assignment into one atomic provisioning operation.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    ConnectionPoolExhaustedError,
    DatabaseLockError,
)
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    ActorMembershipWriteContext,
    AnchorOwnership,
    MembershipAuthority,
    MembershipAuthorizationError,
    MembershipTargetNotFound,
    MembershipWriter,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.profile_version import VersionedUserWriteGateway
from tldw_Server_API.app.core.AuthNZ.repos.orgs_teams_repo import (
    AuthnzOrgsTeamsRepo,
)
from tldw_Server_API.app.core.AuthNZ.transaction_policy import (
    get_authnz_transaction_policy,
)

router = APIRouter(prefix="/provisioning", tags=["admin-provisioning"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class TenantProvisionRequest(BaseModel):
    """Request body for provisioning a new tenant."""

    username: str = Field(..., min_length=1, max_length=150)
    email: str = Field(..., min_length=3, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    org_name: str = Field(..., min_length=1, max_length=255)
    role: Literal["owner"] = Field(
        default="owner",
        description="The initial tenant user is always the organization owner.",
    )


class TenantProvisionResponse(BaseModel):
    """Response body after successful tenant provisioning."""

    user_id: int
    username: str
    org_id: int
    org_name: str
    role: str
    message: str = "Tenant provisioned successfully"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/tenants", response_model=TenantProvisionResponse)
async def provision_tenant(
    payload: TenantProvisionRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> TenantProvisionResponse:
    """Create a new tenant: user + org + default role in one call.

    This endpoint is restricted to admin users (enforced by the parent
    ``/admin`` router dependency).

    Steps:
    1. Create user account
    2. Create organisation
    3. Add user as org member with requested role
    """
    if type(principal.user_id) is not int or principal.user_id <= 0:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to provision tenants",
        )
    try:
        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

        pool = await get_db_pool()
        is_postgres = getattr(pool, "pool", None) is not None
        backend = "postgres" if is_postgres else "sqlite"

        async with pool.transaction(
            acquire_timeout_seconds=(
                get_authnz_transaction_policy().db_pool_acquire_timeout_seconds
            ),
        ) as conn:
            # 1. Create user
            # Check for duplicate username
            if is_postgres:
                existing = await conn.fetchrow(
                    "SELECT id FROM public.users WHERE username = $1",
                    payload.username,
                )
            else:
                cur = await conn.execute(
                    "SELECT id FROM main.users WHERE username = ?",
                    (payload.username,),
                )
                existing = await cur.fetchone()
            if existing:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail=f"Username '{payload.username}' already exists",
                )

            # Hash password
            from tldw_Server_API.app.core.AuthNZ.password_service import get_password_service

            pw_svc = get_password_service()
            hashed = pw_svc.hash_password(payload.password)

            insert_result = await VersionedUserWriteGateway(backend).insert_user(
                conn,
                values={
                    "username": payload.username,
                    "email": payload.email,
                    "password_hash": hashed,
                    "is_active": True,
                },
            )
            user_id = insert_result.affected_user_ids[0]

            creation_context = ActorMembershipWriteContext(
                actor_user_id=principal.user_id,
                required_authority=MembershipAuthority.PLATFORM_ADMIN,
            )
            await MembershipWriter(pool).authorize_organization_creation(
                conn=conn,
                context=creation_context,
                owner_user_id=user_id,
            )

            # 2. Create organization owned by the new user.
            if is_postgres:
                row = await conn.fetchrow(
                    "INSERT INTO public.organizations (name, owner_user_id) "
                    "VALUES ($1, $2) RETURNING id",
                    payload.org_name,
                    user_id,
                )
                if not row:
                    raise RuntimeError("Tenant organization insert returned no id")
                org_id = int(row["id"])
            else:
                cur = await conn.execute(
                    "INSERT INTO main.organizations (name, owner_user_id) VALUES (?, ?)",
                    (payload.org_name, user_id),
                )
                org_id = int(cur.lastrowid)
            await AuthnzOrgsTeamsRepo(
                pool
            ).provision_org_membership_on_connection(
                conn=conn,
                org_id=org_id,
                user_id=user_id,
                org_role=payload.role,
                team_id=None,
                team_role=None,
                team_failure_is_best_effort=False,
                context=creation_context,
                anchor_ownership=AnchorOwnership.WRITER_OWNS_ANCHOR,
                operation_time=datetime.now(timezone.utc),
            )

        logger.info(
            "Tenant provisioned: user_id={}, org_id={}, role={}, by admin={}",
            user_id,
            org_id,
            payload.role,
            principal.user_id,
        )

        return TenantProvisionResponse(
            user_id=user_id,
            username=payload.username,
            org_id=org_id,
            org_name=payload.org_name,
            role=payload.role,
        )

    except (MembershipAuthorizationError, MembershipTargetNotFound):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to provision tenants",
        ) from None
    except (ConnectionPoolExhaustedError, DatabaseLockError, TimeoutError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication database is busy. Please retry shortly.",
            headers={
                "Retry-After": str(
                    get_authnz_transaction_policy().busy_retry_after_seconds
                ),
            },
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Tenant provisioning failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tenant provisioning failed",
        ) from exc
