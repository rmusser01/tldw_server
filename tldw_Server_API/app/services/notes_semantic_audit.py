"""Resolve runtime audit dependencies for Notes semantic events."""

from typing import Any

from tldw_Server_API.app.core.Notes_Graph.semantic_observability import (
    emit_semantic_audit_event as write_semantic_audit_event,
)


async def emit_semantic_audit_event(*, owner_user_id: str, **fields: Any) -> None:
    """Write and flush an event using the existing owner-bound audit service."""
    from tldw_Server_API.app.api.v1.API_Deps.Audit_DB_Deps import (
        get_or_create_audit_service_for_user_id_optional,
    )

    await write_semantic_audit_event(
        owner_user_id=owner_user_id,
        audit_service_factory=get_or_create_audit_service_for_user_id_optional,
        **fields,
    )
