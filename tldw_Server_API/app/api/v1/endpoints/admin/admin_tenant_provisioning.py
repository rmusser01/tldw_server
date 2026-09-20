"""
Tenant provisioning endpoint.

Provides a single API call that chains user creation, org creation,
and role assignment into one atomic provisioning operation.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.profile_version import VersionedUserWriteGateway

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
    role: str = Field(
        default="owner",
        pattern=r"^(owner|admin|lead|member)$",
        description="Role to assign the user within the new org.",
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
    try:
        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

        pool = await get_db_pool()
        is_postgres = getattr(pool, "pool", None) is not None
        backend = "postgres" if is_postgres else "sqlite"

        async with pool.transaction() as conn:
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
                await conn.execute(
                    "INSERT INTO public.org_members (org_id, user_id, role) "
                    "VALUES ($1, $2, $3)",
                    org_id,
                    user_id,
                    payload.role,
                )
            else:
                cur = await conn.execute(
                    "INSERT INTO main.organizations (name, owner_user_id) VALUES (?, ?)",
                    (payload.org_name, user_id),
                )
                org_id = int(cur.lastrowid)
                await conn.execute(
                    "INSERT INTO main.org_members (org_id, user_id, role) "
                    "VALUES (?, ?, ?)",
                    (org_id, user_id, payload.role),
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

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Tenant provisioning failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Tenant provisioning failed",
        ) from exc
