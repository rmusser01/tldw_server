from __future__ import annotations

from typing import Any

from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import (
    get_or_create_audit_service_for_user_id_optional,
)
from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditContext,
    AuditEventCategory,
    AuditEventType,
    MandatoryAuditWriteError,
)


async def emit_admin_account_audit_event(
    *,
    actor_id: int | None,
    target_user_id: int,
    event_type: AuditEventType,
    category: AuditEventCategory,
    resource_type: str,
    resource_id: str,
    action: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Persist a durable audit event for privileged admin account mutations."""
    try:
        svc = await get_or_create_audit_service_for_user_id_optional(actor_id)
        ctx = AuditContext(
            user_id=str(actor_id) if actor_id is not None else None,
            endpoint="/api/v1/admin/users",
            method="INTERNAL",
        )
        await svc.log_event(
            event_type=event_type,
            category=category,
            context=ctx,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            metadata={
                "actor_id": actor_id,
                "target_user_id": target_user_id,
                "resource_type": resource_type,
                "resource_id": resource_id,
                "action": action,
                **(metadata or {}),
            },
        )
        await svc.flush(raise_on_failure=False)
    except Exception as exc:
        logger.warning("Admin audit emission failed for action={}: {}", action, exc)


async def emit_impersonation_issuance_audit_event(
    *,
    actor_id: int | None,
    target_user_id: int,
    expires_in_minutes: int,
) -> None:
    try:
        svc = await get_or_create_audit_service_for_user_id_optional(actor_id)
        ctx = AuditContext(
            user_id=str(actor_id) if actor_id is not None else None,
            endpoint="/api/v1/admin/impersonate/{user_id}/token",
            method="POST",
        )
        await svc.log_event(
            event_type=AuditEventType.USER_UPDATED,
            category=AuditEventCategory.AUTHENTICATION,
            context=ctx,
            resource_type="user_impersonation",
            resource_id=str(target_user_id),
            action="admin.impersonation.token_issued",
            metadata={
                "actor_id": actor_id,
                "target_user_id": target_user_id,
                "expires_in_minutes": expires_in_minutes,
                "impersonation": True,
            },
        )
        await svc.flush(raise_on_failure=True)
    except MandatoryAuditWriteError:
        raise
    except Exception as exc:
        logger.warning("Mandatory impersonation audit emission failed: {}", exc)
        raise MandatoryAuditWriteError("Mandatory audit persistence unavailable") from exc
