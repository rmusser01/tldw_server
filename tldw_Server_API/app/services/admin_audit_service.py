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
)
from tldw_Server_API.app.core.AuthNZ.transaction_hooks import (
    defer_until_after_commit,
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
    """Persist now or defer until the owning AuthNZ transaction commits."""
    payload = {
        "actor_id": actor_id,
        "target_user_id": target_user_id,
        "event_type": event_type,
        "category": category,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "action": action,
        "metadata": dict(metadata or {}),
    }

    async def _deferred_emit() -> None:
        await _persist_admin_account_audit_event(**payload)

    if defer_until_after_commit(_deferred_emit):
        return
    await _persist_admin_account_audit_event(**payload)


async def _persist_admin_account_audit_event(
    *,
    actor_id: int | None,
    target_user_id: int,
    event_type: AuditEventType,
    category: AuditEventCategory,
    resource_type: str,
    resource_id: str,
    action: str,
    metadata: dict[str, Any],
) -> None:
    """Write one best-effort durable audit event."""
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
                **metadata,
            },
        )
        await svc.flush(raise_on_failure=False)
    except Exception as exc:
        logger.bind(exception_type=type(exc).__name__).warning(
            "Admin audit emission failed"
        )
