"""Strict unified-audit helpers for API key management operations."""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from loguru import logger

from tldw_Server_API.app.core.Audit.unified_audit_service import (
    AuditContext,
    AuditEventCategory,
    AuditEventType,
    MandatoryAuditWriteError,
    UnifiedAuditService,
)


async def _get_or_create_audit_service(user_id: int) -> UnifiedAuditService:
    """Return the cached audit service for the target user."""
    from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import (
        get_or_create_audit_service_for_user_id,
    )

    return await get_or_create_audit_service_for_user_id(user_id)


async def _create_isolated_audit_service(user_id: int) -> UnifiedAuditService:
    """Create an isolated audit service instance for fail-closed writes."""
    from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import (
        _create_audit_service_for_user,
    )

    return await _create_audit_service_for_user(user_id)


async def emit_mandatory_api_key_management_audit(
    *,
    user_id: int,
    event_type: AuditEventType | str,
    category: AuditEventCategory | str,
    action: str,
    resource_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    actor_user_id: int | None = None,
    actor_subject: str | None = None,
    actor_kind: str | None = None,
    actor_roles: list[str] | None = None,
) -> None:
    """Emit a mandatory unified audit event for API-key management operations."""
    merged_metadata = dict(metadata or {})
    merged_metadata["target_user_id"] = int(user_id)
    if actor_user_id is not None:
        merged_metadata["actor_user_id"] = int(actor_user_id)
    if actor_subject:
        merged_metadata["actor_subject"] = str(actor_subject)
    if actor_kind:
        merged_metadata["actor_kind"] = str(actor_kind)
    if actor_roles:
        merged_metadata["actor_roles"] = [str(role) for role in actor_roles if str(role).strip()]

    audit_service: UnifiedAuditService | None = None
    try:
        audit_service = await _create_isolated_audit_service(int(user_id))
        await audit_service.log_event(
            event_type=event_type,
            category=category,
            context=AuditContext(user_id=str(user_id)),
            resource_type="api_key",
            resource_id=resource_id,
            action=action,
            metadata=merged_metadata,
        )
        await audit_service.flush(raise_on_failure=True)
    except MandatoryAuditWriteError:
        raise
    except Exception as exc:
        logger.opt(exception=True).error(
            "Mandatory API key management audit write failed (action={}, user_id={}, resource_id={}): {}",
            action,
            user_id,
            resource_id,
            type(exc).__name__,
        )
        raise MandatoryAuditWriteError("Mandatory audit persistence unavailable") from exc
    finally:
        if audit_service is not None:
            with suppress(Exception):
                await audit_service.stop()
