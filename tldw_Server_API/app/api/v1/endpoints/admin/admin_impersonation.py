"""
Admin impersonation endpoint.

Allows administrators to generate short-lived JWT tokens that act on behalf
of another user.  The token includes an ``impersonated_by`` claim for full
audit traceability.
"""

from __future__ import annotations

from datetime import timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_auth_principal
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditEventCategory,
    AuditEventType,
    MandatoryAuditWriteError,
)
from tldw_Server_API.app.core.AuthNZ.jwt_service import get_jwt_service
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.repos.users_repo import AuthnzUsersRepo
from tldw_Server_API.app.services.admin_audit_service import (
    emit_admin_account_audit_event as _emit_admin_account_audit_event,
)

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
        users_repo = await AuthnzUsersRepo.from_pool()
        target_user = await users_repo.get_user_by_id(user_id)
        if not target_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User {user_id} not found",
            )

        if isinstance(target_user, dict):
            target_user_id = int(target_user["id"])
            target_username = str(target_user["username"])
            target_is_active = bool(target_user.get("is_active", True))
            target_role = str(target_user.get("role") or "user")
        else:
            target_user_id = int(target_user.id)
            target_username = str(target_user.username)
            target_is_active = bool(getattr(target_user, "is_active", True))
            target_role = str(getattr(target_user, "role", None) or "user")

        if not target_is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"User {user_id} is not active",
            )
        # Generate a short-lived access token with impersonation claim
        jwt_svc = get_jwt_service()
        token = jwt_svc.create_access_token(
            user_id=target_user_id,
            username=target_username,
            role=target_role,
            expires_delta=timedelta(minutes=_IMPERSONATION_TTL_MINUTES),
            additional_claims={
                "impersonated_by": principal.user_id,
                "impersonation": True,
            },
        )

        await _emit_admin_account_audit_event(
            actor_id=principal.user_id,
            target_user_id=target_user_id,
            event_type=AuditEventType.AUTH_TOKEN_CREATED,
            category=AuditEventCategory.AUTHORIZATION,
            resource_type="user_impersonation",
            resource_id=str(target_user_id),
            action="admin.impersonation.token.create",
            metadata={
                "impersonated_by": principal.user_id,
                "impersonated_user_id": target_user_id,
                "expires_in_minutes": _IMPERSONATION_TTL_MINUTES,
                "impersonation": True,
            },
            raise_on_failure=True,
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
    except MandatoryAuditWriteError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": {
                    "message": "Mandatory audit persistence unavailable",
                    "type": "audit_persistence_failure",
                    "code": "audit_persistence_failure",
                }
            },
        ) from exc
    except Exception as exc:
        logger.error("Impersonation token creation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Impersonation token creation failed",
        ) from exc
