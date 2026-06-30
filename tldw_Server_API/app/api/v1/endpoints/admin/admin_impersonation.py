"""
Admin impersonation endpoint.

Allows administrators to generate short-lived JWT tokens that act on behalf
of another user.  The token includes an ``impersonated_by`` claim for full
audit traceability.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.Audit.unified_audit_service import MandatoryAuditWriteError
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.rbac_repo import AuthnzRbacRepo
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.services.admin_audit_service import emit_impersonation_issuance_audit_event

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


def _first_role_name(role_rows: list[Any]) -> str | None:
    if not role_rows:
        return None
    first = role_rows[0]
    if isinstance(first, dict):
        value = first.get("name") or first.get("role") or first.get("role_name")
        return str(value) if value else None
    value = (
        getattr(first, "name", None)
        or getattr(first, "role", None)
        or getattr(first, "role_name", None)
    )
    return str(value) if value else str(first)


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
        repo = await AuthnzUsersRepo.from_pool()
        row = await repo.get_user_by_id(user_id)

        if not row:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user_id} not found",
            )

        target_user_id = int(row["id"])
        target_username = str(row["username"])
        target_is_active = bool(row.get("is_active", False))

        if not target_is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"User {user_id} is not active",
            )

        # Determine the target user's role
        try:
            role_rows = AuthnzRbacRepo().get_user_roles(target_user_id)
        except Exception:
            logger.warning(
                "Unable to load RBAC roles for impersonation target; falling back to user row role"
            )
            role_rows = []
        target_role = _first_role_name(role_rows) or str(row.get("role") or "user")

        # Generate a short-lived access token with impersonation claim
        token = get_jwt_service().create_impersonation_access_token(
            user_id=target_user_id,
            username=target_username,
            role=target_role,
            impersonated_by=principal.user_id,
            expires_delta=timedelta(minutes=_IMPERSONATION_TTL_MINUTES),
        )

        try:
            await emit_impersonation_issuance_audit_event(
                actor_id=principal.user_id,
                target_user_id=target_user_id,
                expires_in_minutes=_IMPERSONATION_TTL_MINUTES,
            )
        except MandatoryAuditWriteError as exc:
            logger.error("Mandatory audit write failed while creating impersonation token")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Mandatory audit persistence unavailable",
            ) from exc

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
