from __future__ import annotations

from collections import Counter

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
    severity_counts = Counter(record.severity for record in records)
    items = [
        AdminStartupWarningItem.model_validate(record)
        for record in records
    ]
    return AdminStartupWarningsResponse(
        startup_id=str(registry.startup_id),
        warnings_present=bool(records),
        blocking_present=registry.should_block_startup(),
        summary=AdminStartupWarningSummary(
            total=len(records),
            by_component=dict(registry.summary().get("by_component") or {}),
            by_severity={
                severity: severity_counts[severity]
                for severity in sorted(severity_counts)
            },
        ),
        items=items,
    )
