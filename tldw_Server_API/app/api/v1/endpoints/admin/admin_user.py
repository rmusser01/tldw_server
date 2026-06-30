from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import PlainTextResponse
from loguru import logger
from pydantic import BaseModel, EmailStr, Field

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_auth_principal,
    get_db_transaction,
    get_password_service_dep,
    get_registration_service_dep,
)
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    AdminMfaRequirementRequest,
    AdminMfaRequirementResponse,
    AdminPasswordResetRequest,
    AdminPasswordResetResponse,
    AdminPrivilegedActionRequest,
    AdminUserCreateRequest,
    UserDetailResponse,
    UserListResponse,
    UserSummary,
    UserUpdateRequest,
)
from tldw_Server_API.app.api.v1.schemas.auth_schemas import MessageResponse
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.testing import is_test_mode
from tldw_Server_API.app.services import admin_users_service
from tldw_Server_API.app.services.admin_system_ops_service import (
    create_invitation as svc_create_invitation,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    list_invitations as svc_list_invitations,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    resend_invitation as svc_resend_invitation,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    revoke_invitation as svc_revoke_invitation,
)
from tldw_Server_API.app.services.admin_system_ops_service import (
    update_invitation_email_status as svc_update_invitation_email_status,
)

# ──────────────────────────────────────────────────────────────────────────────
# Invitation Schemas
# ──────────────────────────────────────────────────────────────────────────────


class InviteUserRequest(BaseModel):
    email: EmailStr
    role: str = Field(default="user", pattern=r"^(user|admin|service|viewer)$")
    expiry_days: int = Field(default=7, ge=1, le=365)


class InvitationItem(BaseModel):
    id: str
    email: str
    role: str
    status: str
    token: str | None = None
    invited_by: str | None = None
    created_at: str | None = None
    expires_at: str | None = None
    accepted_at: str | None = None
    email_sent: bool = False
    email_error: str | None = None
    resend_count: int = 0
    last_resent_at: str | None = None


class InvitationListResponse(BaseModel):
    items: list[InvitationItem]
    total: int


router = APIRouter()


@router.post("/users", response_model=UserSummary)
async def admin_create_user(
    payload: AdminUserCreateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    registration_service=Depends(get_registration_service_dep),
) -> UserSummary:
    """
    Create a new user as an admin.
    """
    return await admin_users_service.create_user(
        payload=payload,
        principal=principal,
        registration_service=registration_service,
    )


@router.get("/users", response_model=UserListResponse)
async def list_users(
    request: Request,
    response: Response,
    principal: AuthPrincipal = Depends(get_auth_principal),
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    role: str | None = None,
    admin_capable: bool = Query(False, description="Restrict to admin-capable assignees"),
    is_active: bool | None = None,
    mfa_enabled: bool | None = Query(None, description="Filter by MFA status: true=enabled, false=disabled"),
    search: str | None = None,
    org_id: int | None = Query(None, description="Restrict to a specific organization"),
) -> UserListResponse:
    """
    List all users with pagination and filters

    Args:
        page: Page number (1-based)
        limit: Items per page
        role: Filter by role
        is_active: Filter by active status
        search: Search in username/email

    Returns:
        Paginated list of users
    """
    # TEST_MODE diagnostics: annotate DB backend and admin dependency success
    if is_test_mode():
        try:
            pool = await get_db_pool()
            db_backend = "postgres" if getattr(pool, "pool", None) is not None else "sqlite"
            response.headers["X-TLDW-Admin-DB"] = db_backend
            response.headers["X-TLDW-Admin-Req"] = "ok"
            auth_hdr = request.headers.get("Authorization")
            logger.info(
                "Admin list_users TEST_MODE: Authorization present={}",
                bool(auth_hdr),
            )
        except Exception as diag_exc:  # noqa: BLE001 - diagnostics only, do not fail request
            response.headers["X-TLDW-Admin-Diag-Error"] = str(diag_exc)
            logger.debug("Admin list_users TEST_MODE diagnostics failed")

    try:
        users, total = await admin_users_service.list_users(
            principal,
            page=page,
            limit=limit,
            role=role,
            admin_capable=admin_capable,
            is_active=is_active,
            mfa_enabled=mfa_enabled,
            search=search,
            org_id=org_id,
        )
        return UserListResponse(
            users=users,
            total=total,
            page=page,
            limit=limit,
            pages=(total + limit - 1) // limit if limit else 0,
            pagination=build_offset_pagination_meta(
                total=total,
                limit=limit,
                offset=(page - 1) * limit,
                count=len(users),
            ),
        )
    except HTTPException as e:
        try:
            if is_test_mode():
                response.headers["X-TLDW-Admin-Error"] = str(e)
        except Exception:
            logger.debug("TEST_MODE header assignment failed")
        raise
    except Exception as e:
        logger.error("Failed to list users")
        try:
            if is_test_mode():
                response.headers["X-TLDW-Admin-Error"] = str(e)
        except Exception:
            logger.debug("TEST_MODE header assignment failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users",
        ) from e


@router.get(
    "/users/export",
    responses={
        200: {
            "description": "User export as JSON or CSV.",
            "content": {
                "application/json": {},
                "text/csv": {},
            },
        },
    },
)
async def export_users(
    role: str | None = None,
    is_active: bool | None = None,
    search: str | None = None,
    org_id: int | None = Query(None, description="Restrict to a specific organization"),
    limit: int = Query(10000, ge=1, le=50000),
    offset: int = Query(0, ge=0),
    format: str = Query("csv", pattern="^(csv|json)$"),
    filename: str | None = Query(None, description="Optional filename for Content-Disposition"),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> Response:
    try:
        content, media_type, default_name = await admin_users_service.export_users(
            principal,
            role=role,
            is_active=is_active,
            search=search,
            org_id=org_id,
            limit=limit,
            offset=offset,
            format=format,
        )
        if media_type == "application/json":
            resp = Response(content=content, media_type=media_type)
        else:
            resp = PlainTextResponse(content=content, media_type=media_type)
        if not filename:
            filename = default_name
        if filename:
            safe = filename.replace("\n", " ").replace("\r", " ").replace('"', "_")
            resp.headers["Content-Disposition"] = f'attachment; filename="{safe}"'
        return resp
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to export users")
        raise HTTPException(status_code=500, detail="Failed to export users") from exc


# ──────────────────────────────────────────────────────────────────────────────
# User Invitations
# NOTE: These static routes MUST be registered before /users/{user_id}
# to avoid FastAPI matching "invite" or "invitations" as a user_id.
# ──────────────────────────────────────────────────────────────────────────────


@router.post("/users/invite", response_model=InvitationItem)
async def invite_user(
    payload: InviteUserRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> InvitationItem:
    """
    Create a user invitation and optionally send an invite email.

    The invitation is always created even if email delivery fails.
    Check `email_sent` in the response to determine delivery status.
    """
    actor = getattr(principal, "username", None) or str(getattr(principal, "user_id", "admin"))

    try:
        invitation = svc_create_invitation(
            email=payload.email,
            role=payload.role,
            invited_by=actor,
            expiry_days=payload.expiry_days,
        )
    except ValueError as exc:
        error_key = str(exc)
        if error_key == "duplicate_pending_invitation":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A pending invitation already exists for this email address.",
            ) from exc
        if error_key == "too_many_pending_invitations":
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many pending invitations. Please revoke some before creating new ones.",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid invitation parameters: {error_key}",
        ) from exc

    # Attempt email delivery (best-effort)
    email_sent = False
    email_error: str | None = None
    try:
        from tldw_Server_API.app.core.AuthNZ.email_service import get_email_service

        email_service = get_email_service()
        email_sent = await email_service.send_user_invitation_email(
            to_email=invitation["email"],
            invite_token=invitation["token"],
            role=invitation["role"],
            expiry_days=payload.expiry_days,
        )
        if not email_sent:
            email_error = "Email delivery returned false"
    except Exception as exc:
        email_error = str(exc)
        logger.warning("Failed to send invitation email")

    svc_update_invitation_email_status(
        invitation_id=invitation["id"],
        email_sent=email_sent,
        email_error=email_error,
    )

    invitation["email_sent"] = email_sent
    invitation["email_error"] = email_error

    logger.info(
        "User invitation created for {} by {} (email_sent={})",
        invitation["email"],
        actor,
        email_sent,
    )

    return InvitationItem(**{k: v for k, v in invitation.items() if k in InvitationItem.model_fields})


@router.get("/users/invitations", response_model=InvitationListResponse)
async def list_user_invitations(
    principal: AuthPrincipal = Depends(get_auth_principal),
    invitation_status: str | None = Query(None, alias="status"),
) -> InvitationListResponse:
    """List all user invitations with optional status filter."""
    invitations = svc_list_invitations(status=invitation_status)
    items = []
    for inv in invitations:
        try:
            item = InvitationItem(**{k: v for k, v in inv.items() if k in InvitationItem.model_fields})
        except (AttributeError, TypeError, ValueError):
            logger.warning("Skipping invalid invitation record")
        else:
            items.append(item)
    return InvitationListResponse(items=items, total=len(items))


@router.delete("/users/invitations/{invitation_id}", response_model=InvitationItem)
async def revoke_user_invitation(
    invitation_id: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> InvitationItem:
    """Revoke a pending invitation."""
    try:
        result = svc_revoke_invitation(invitation_id=invitation_id)
    except ValueError as exc:
        error_key = str(exc)
        if error_key == "not_found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invitation not found.",
            ) from exc
        if error_key == "not_pending":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Only pending invitations can be revoked.",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    actor = getattr(principal, "username", None) or str(getattr(principal, "user_id", "admin"))
    logger.info("Invitation {} revoked by {}", invitation_id, actor)

    return InvitationItem(**{k: v for k, v in result.items() if k in InvitationItem.model_fields})


@router.post("/users/invitations/{invitation_id}/resend", response_model=InvitationItem)
async def resend_user_invitation(
    invitation_id: str,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> InvitationItem:
    """Resend a pending invitation with a new token and extended expiry.

    Rate-limited to 3 resends per invitation.
    """
    try:
        invitation = svc_resend_invitation(invitation_id=invitation_id)
    except ValueError as exc:
        error_key = str(exc)
        if error_key == "not_found":
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invitation not found.",
            ) from exc
        if error_key == "not_pending":
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Only pending invitations can be resent.",
            ) from exc
        if error_key == "resend_limit_reached":
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Maximum resend limit (3) reached for this invitation.",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    # Attempt email re-delivery (best-effort)
    email_sent = False
    email_error: str | None = None
    try:
        from tldw_Server_API.app.core.AuthNZ.email_service import get_email_service

        email_service = get_email_service()
        email_sent = await email_service.send_user_invitation_email(
            to_email=invitation["email"],
            invite_token=invitation["token"],
            role=invitation["role"],
            expiry_days=7,
        )
        if not email_sent:
            email_error = "Email delivery returned false"
    except Exception as exc:
        email_error = str(exc)
        logger.warning("Failed to resend invitation email")

    svc_update_invitation_email_status(
        invitation_id=invitation["id"],
        email_sent=email_sent,
        email_error=email_error,
    )

    invitation["email_sent"] = email_sent
    invitation["email_error"] = email_error

    actor = getattr(principal, "username", None) or str(getattr(principal, "user_id", "admin"))
    logger.info(
        "Invitation {} resent by {} (attempt {}, email_sent={})",
        invitation_id,
        actor,
        invitation.get("resend_count", 0),
        email_sent,
    )

    return InvitationItem(**{k: v for k, v in invitation.items() if k in InvitationItem.model_fields})


# ──────────────────────────────────────────────────────────────────────────────
# User Detail Endpoints (dynamic user_id routes)
# ──────────────────────────────────────────────────────────────────────────────


@router.get("/users/{user_id}", response_model=UserDetailResponse)
async def get_user_details(
    user_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> UserDetailResponse:
    return await admin_users_service.get_user_details(principal, user_id)


def _get_is_pg_fn():
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    return admin_mod._is_postgres_backend


@router.put("/users/{user_id}", response_model=MessageResponse)
async def update_user(
    user_id: int,
    request: UserUpdateRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_db_transaction),
    password_service=Depends(get_password_service_dep),
) -> MessageResponse:
    return await admin_users_service.update_user(
        principal,
        user_id,
        request,
        db,
        password_service,
        is_pg_fn=_get_is_pg_fn(),
    )


@router.post("/users/{user_id}/reset-password", response_model=AdminPasswordResetResponse)
async def reset_user_password(
    user_id: int,
    request: AdminPasswordResetRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_db_transaction),
    password_service=Depends(get_password_service_dep),
) -> AdminPasswordResetResponse:
    result = await admin_users_service.reset_user_password(
        principal,
        user_id,
        request,
        db,
        password_service,
        is_pg_fn=_get_is_pg_fn(),
    )
    return AdminPasswordResetResponse(**result)


@router.post("/users/{user_id}/mfa/require", response_model=AdminMfaRequirementResponse)
async def set_user_mfa_requirement(
    user_id: int,
    request: AdminMfaRequirementRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_db_transaction),
    password_service=Depends(get_password_service_dep),
) -> AdminMfaRequirementResponse:
    result = await admin_users_service.set_user_mfa_requirement(
        principal,
        user_id,
        request,
        db,
        password_service,
        is_pg_fn=_get_is_pg_fn(),
    )
    return AdminMfaRequirementResponse(**result)


@router.delete("/users/{user_id}", response_model=MessageResponse)
async def delete_user(
    user_id: int,
    request: AdminPrivilegedActionRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_db_transaction),
    password_service=Depends(get_password_service_dep),
) -> MessageResponse:
    return await admin_users_service.delete_user(
        principal,
        user_id,
        request,
        db,
        password_service,
        is_pg_fn=_get_is_pg_fn(),
    )
