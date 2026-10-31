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
from tldw_Server_API.app.core.AuthNZ.exceptions import DuplicateUserError
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.tenant_provisioning import provision_tenant as create_tenant_records

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

        from tldw_Server_API.app.core.AuthNZ.password_service import get_password_service

        try:
            user_id, org_id = await create_tenant_records(
                pool,
                username=payload.username,
                email=payload.email,
                password_hash=get_password_service().hash_password(payload.password),
                org_name=payload.org_name,
                role=payload.role,
            )
        except DuplicateUserError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Username '{payload.username}' already exists",
            ) from exc

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
