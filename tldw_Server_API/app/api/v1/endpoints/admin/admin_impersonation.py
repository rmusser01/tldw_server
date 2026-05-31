"""
Admin impersonation endpoint.

Allows administrators to generate short-lived JWT tokens that act on behalf
of another user.  The token includes an ``impersonated_by`` claim for full
audit traceability.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

router = APIRouter(prefix="/impersonate", tags=["admin-impersonation"])

# Default impersonation token TTL in minutes
_IMPERSONATION_TTL_MINUTES = 15


# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


class ImpersonationTokenResponse(BaseModel):
    """Response containing the impersonation JWT."""

    token: str
    token_type: str = "bearer"
    expires_in_minutes: int = _IMPERSONATION_TTL_MINUTES
    impersonated_user_id: int
    impersonated_by: int | None = None


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post("/{user_id}/token", response_model=ImpersonationTokenResponse)
async def create_impersonation_token(
    user_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> ImpersonationTokenResponse:
    """Generate a temporary JWT token for impersonating a user.

    The token has a short TTL (15 minutes) and includes an ``impersonated_by``
    claim for audit trailing.  Only administrators can call this endpoint
    (enforced by the parent ``/admin`` router dependency).
    """
    try:
        from tldw_Server_API.app.core.AuthNZ.database import get_db_pool

        pool = await get_db_pool()

        # Verify the target user exists
        async with pool.acquire() as conn:
            cur = await conn.execute(
                "SELECT id, username, is_active FROM users WHERE id = ?",
                (user_id,),
            )
            row = await cur.fetchone()

        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user_id} not found",
            )

        target_user_id = row[0]
        target_username = row[1]
        target_is_active = row[2]

        if not target_is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"User {user_id} is not active",
            )

        # Determine the target user's role
        async with pool.acquire() as conn:
            cur = await conn.execute(
                "SELECT role FROM user_roles WHERE user_id = ? LIMIT 1",
                (user_id,),
            )
            role_row = await cur.fetchone()
        target_role = role_row[0] if role_row else "user"

        # Generate a short-lived access token with impersonation claim
        from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service

        jwt_svc = get_jwt_service()
        token = jwt_svc.create_access_token(
            user_id=target_user_id,
            username=target_username,
            role=target_role,
            additional_claims={
                "impersonated_by": principal.user_id,
                "impersonation": True,
            },
        )

        logger.info(
            "Impersonation token created: admin_user_id={} -> target_user_id={}",
            principal.user_id,
            target_user_id,
        )

        return ImpersonationTokenResponse(
            token=token,
            impersonated_user_id=target_user_id,
            impersonated_by=principal.user_id,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Impersonation token creation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Impersonation token creation failed",
        ) from exc
