"""
Tenant provisioning endpoint.

Provides a single API call that chains user creation, org creation,
and role assignment into one atomic provisioning operation.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.exceptions import (
    ConnectionPoolExhaustedError,
    DatabaseLockError,
    DuplicateUserError,
)
from tldw_Server_API.app.core.AuthNZ.membership_writer import (
    MembershipAuthorizationError,
    MembershipTargetNotFound,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.tenant_provisioning import provision_tenant as create_tenant_records
from tldw_Server_API.app.core.AuthNZ.transaction_policy import get_authnz_transaction_policy

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

        from tldw_Server_API.app.core.AuthNZ.password_service import get_password_service

        try:
            user_id, org_id = await create_tenant_records(
                pool,
                actor_user_id=principal.user_id,
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
