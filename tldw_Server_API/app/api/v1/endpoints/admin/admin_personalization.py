from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequireRole,
    check_rate_limit,
)
from tldw_Server_API.app.services import admin_personalization_service

router = APIRouter()

@router.post(
    "/personalization/consolidate",
    response_model=dict,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def trigger_personalization_consolidation(
    user_id: str | None = Query(None, description="User ID to consolidate; defaults to single-user id"),
) -> dict:
    """
    Trigger personalization consolidation (admin scope).

    Args:
        user_id: User ID to consolidate; defaults to the single-user id.
        principal: Authenticated principal used for admin authorization.

    Returns:
        Consolidation trigger result payload.
    """
    return await admin_personalization_service.trigger_consolidation(user_id=user_id)


@router.get(
    "/personalization/status",
    response_model=dict,
    dependencies=[Depends(check_rate_limit), Depends(RequireRole("admin"))],
)
async def get_personalization_status() -> dict:
    """Return personalization consolidation status (admin scope)."""
    return await admin_personalization_service.get_status()
