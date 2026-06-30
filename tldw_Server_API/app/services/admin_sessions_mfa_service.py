from __future__ import annotations

from typing import Any

from fastapi import HTTPException
from loguru import logger

from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    AdminPrivilegedActionRequest,
    unwrap_optional_secret,
)
from tldw_Server_API.app.api.v1.schemas.auth_schemas import MessageResponse, SessionResponse
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditEventCategory,
    AuditEventType,
)
from tldw_Server_API.app.core.AuthNZ.mfa_service import get_mfa_service
from tldw_Server_API.app.core.AuthNZ.password_service import PasswordService
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.AuthNZ.session_manager import SessionManager
from tldw_Server_API.app.services import admin_scope_service
from tldw_Server_API.app.services.admin_audit_service import (
    emit_admin_account_audit_event as _emit_admin_account_audit_event,
)
from tldw_Server_API.app.services.admin_guardrails_service import verify_privileged_action

_ADMIN_SESSIONS_NONCRITICAL_EXCEPTIONS = (
    AttributeError,
    LookupError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
)


async def list_user_sessions(
    principal: AuthPrincipal,
    user_id: int,
    session_manager: SessionManager,
) -> list[SessionResponse]:
    """List active sessions for a user (admin scope)."""
    try:
        await admin_scope_service.enforce_admin_user_scope(
            principal,
            user_id,
            require_hierarchy=False,
        )
        sessions = await session_manager.get_user_sessions(user_id)
        return [
            SessionResponse(
                id=session["id"],
                ip_address=session.get("ip_address"),
                user_agent=session.get("user_agent"),
                created_at=session["created_at"],
                last_activity=session["last_activity"],
                expires_at=session["expires_at"],
            )
            for session in sessions
        ]
    except HTTPException:
        raise
    except _ADMIN_SESSIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to list sessions")
        raise HTTPException(status_code=500, detail="Failed to list sessions") from exc


async def revoke_user_session(
    principal: AuthPrincipal,
    user_id: int,
    session_id: int,
    session_manager: SessionManager,
    db: Any,
    password_service: PasswordService,
    request: AdminPrivilegedActionRequest,
) -> MessageResponse:
    """Revoke a specific session for a user (admin scope)."""
    try:
        await admin_scope_service.enforce_admin_user_scope(
            principal,
            user_id,
            require_hierarchy=True,
        )
        reason = await verify_privileged_action(
            principal,
            db,
            password_service,
            reason=getattr(request, "reason", None),
            admin_password=unwrap_optional_secret(getattr(request, "admin_password", None)),
            admin_reauth_token=unwrap_optional_secret(getattr(request, "admin_reauth_token", None)),
        )
        await session_manager.revoke_session(session_id=session_id, revoked_by=principal.user_id)
        await _emit_admin_account_audit_event(
            actor_id=principal.user_id,
            target_user_id=user_id,
            event_type=AuditEventType.AUTH_TOKEN_REVOKED,
            category=AuditEventCategory.AUTHENTICATION,
            resource_type="user_session",
            resource_id=str(session_id),
            action="admin.user.session.revoke",
            metadata={
                "reason": reason,
                "session_id": session_id,
            },
        )
        return MessageResponse(message="Session revoked")
    except HTTPException:
        raise
    except _ADMIN_SESSIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to revoke session")
        raise HTTPException(status_code=500, detail="Failed to revoke session") from exc


async def revoke_all_user_sessions(
    principal: AuthPrincipal,
    user_id: int,
    session_manager: SessionManager,
    db: Any,
    password_service: PasswordService,
    request: AdminPrivilegedActionRequest,
) -> MessageResponse:
    """Revoke all sessions for a user (admin scope)."""
    try:
        await admin_scope_service.enforce_admin_user_scope(
            principal,
            user_id,
            require_hierarchy=True,
        )
        reason = await verify_privileged_action(
            principal,
            db,
            password_service,
            reason=getattr(request, "reason", None),
            admin_password=unwrap_optional_secret(getattr(request, "admin_password", None)),
            admin_reauth_token=unwrap_optional_secret(getattr(request, "admin_reauth_token", None)),
        )
        await session_manager.revoke_all_user_sessions(user_id=user_id)
        await _emit_admin_account_audit_event(
            actor_id=principal.user_id,
            target_user_id=user_id,
            event_type=AuditEventType.AUTH_TOKEN_REVOKED,
            category=AuditEventCategory.AUTHENTICATION,
            resource_type="user_session",
            resource_id=str(user_id),
            action="admin.user.sessions.revoke_all",
            metadata={
                "reason": reason,
                "scope": "all",
            },
        )
        return MessageResponse(message="All sessions revoked")
    except HTTPException:
        raise
    except _ADMIN_SESSIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to revoke all sessions")
        raise HTTPException(status_code=500, detail="Failed to revoke sessions") from exc


async def get_user_mfa_status(
    principal: AuthPrincipal,
    user_id: int,
) -> dict[str, Any]:
    """Fetch MFA status for a user (admin scope)."""
    try:
        await admin_scope_service.enforce_admin_user_scope(
            principal,
            user_id,
            require_hierarchy=False,
        )
        mfa_service = get_mfa_service()
        return await mfa_service.get_user_mfa_status(user_id)
    except HTTPException:
        raise
    except _ADMIN_SESSIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to fetch MFA status")
        raise HTTPException(status_code=500, detail="Failed to fetch MFA status") from exc


async def disable_user_mfa(
    principal: AuthPrincipal,
    user_id: int,
    db: Any,
    password_service: PasswordService,
    request: AdminPrivilegedActionRequest,
) -> MessageResponse:
    """Disable MFA for a user (admin scope)."""
    try:
        await admin_scope_service.enforce_admin_user_scope(
            principal,
            user_id,
            require_hierarchy=True,
        )
        reason = await verify_privileged_action(
            principal,
            db,
            password_service,
            reason=getattr(request, "reason", None),
            admin_password=unwrap_optional_secret(getattr(request, "admin_password", None)),
            admin_reauth_token=unwrap_optional_secret(getattr(request, "admin_reauth_token", None)),
        )
        mfa_service = get_mfa_service()
        success = await mfa_service.disable_mfa(user_id)
        if not success:
            raise HTTPException(status_code=404, detail="MFA not enabled")
        await _emit_admin_account_audit_event(
            actor_id=principal.user_id,
            target_user_id=user_id,
            event_type=AuditEventType.CONFIG_CHANGED,
            category=AuditEventCategory.SECURITY,
            resource_type="user_mfa",
            resource_id=str(user_id),
            action="admin.user.mfa.disable",
            metadata={"reason": reason},
        )
        return MessageResponse(message="MFA disabled")
    except HTTPException:
        raise
    except _ADMIN_SESSIONS_NONCRITICAL_EXCEPTIONS as exc:
        logger.error("Failed to disable MFA")
        raise HTTPException(status_code=500, detail="Failed to disable MFA") from exc
