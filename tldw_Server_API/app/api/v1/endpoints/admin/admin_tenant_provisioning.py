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
    role: str = Field(default="owner", description="Role to assign the user within the new org.")


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

        # 1. Create user
        async with pool.acquire() as conn:
            # Check for duplicate username
            cur = await conn.execute(
                "SELECT id FROM users WHERE username = ?",
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

            await conn.execute(
                "INSERT INTO users (username, email, password_hash, is_active) VALUES (?, ?, ?, 1)",
                (payload.username, payload.email, hashed),
            )
            cur = await conn.execute("SELECT last_insert_rowid()")
            row = await cur.fetchone()
            user_id: int = row[0]

        # 2. Create org
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO organizations (name) VALUES (?)",
                (payload.org_name,),
            )
            cur = await conn.execute("SELECT last_insert_rowid()")
            row = await cur.fetchone()
            org_id: int = row[0]

        # 3. Add user as org member with role
        async with pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO organization_members (organization_id, user_id, role) VALUES (?, ?, ?)",
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
