from __future__ import annotations

from fastapi import APIRouter, Request

from tldw_Server_API.app.api.v1.schemas.admin_schemas import (
    AdminStartupWarningItem,
    AdminStartupWarningsResponse,
    AdminStartupWarningSummary,
)


router = APIRouter()


@router.get(
    "/startup-warnings",
    response_model=AdminStartupWarningsResponse,
)
async def get_startup_warnings(request: Request) -> AdminStartupWarningsResponse:
    """Return current-process startup warning records and grouped counts for admin inspection."""
    registry = getattr(request.app.state, "startup_warning_registry", None)
    if registry is None:
        return AdminStartupWarningsResponse(
            startup_id="uninitialized",
            warnings_present=False,
            blocking_present=False,
            summary=AdminStartupWarningSummary(total=0),
            items=[],
        )

    records = registry.list_warnings()
    summary_data = registry.summary()
    items = [
        AdminStartupWarningItem.model_validate(record)
        for record in records
    ]
    return AdminStartupWarningsResponse(
        startup_id=str(registry.startup_id),
        warnings_present=bool(records),
        blocking_present=bool(summary_data["has_blocking"]),
        summary=AdminStartupWarningSummary.model_validate(summary_data),
        items=items,
    )
