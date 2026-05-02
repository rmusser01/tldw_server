from __future__ import annotations

from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, Request, UploadFile
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, rbac_rate_limit, RequirePermission, User

from tldw_Server_API.app.api.v1.API_Deps.billing_deps import require_within_limit
from tldw_Server_API.app.api.v1.API_Deps.storage_quota_guard import guard_storage_quota
from tldw_Server_API.app.core.Billing.enforcement import LimitCategory
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
from tldw_Server_API.app.api.v1.API_Deps.media_add_deps import get_add_media_form
from tldw_Server_API.app.api.v1.API_Deps.personalization_deps import (
    UsageEventLogger,
    get_usage_event_logger,
)
from tldw_Server_API.app.api.v1.schemas.media_request_models import AddMediaForm
from tldw_Server_API.app.core.AuthNZ.permissions import MEDIA_CREATE
from tldw_Server_API.app.core.Ingestion_Media_Processing.persistence import (
    add_media_persist,
)

router = APIRouter()


@router.post(
    "/add",
    # Status code is determined dynamically based on per-item results.
    dependencies=[
        Depends(RequirePermission(MEDIA_CREATE)),
        Depends(rbac_rate_limit("media.create")),
        Depends(guard_storage_quota),
        Depends(require_within_limit(LimitCategory.STORAGE_MB, 1)),
        Depends(require_within_limit(LimitCategory.API_CALLS_DAY, 1)),
    ],
    summary="Add media (URLs/files) with processing and persistence",
    tags=["Media Ingestion & Persistence"],
)
async def add_media(
    request: Request,
    background_tasks: BackgroundTasks,
    form_data: AddMediaForm = Depends(get_add_media_form),
    files: list[UploadFile] | None = File(
        None,
        description="List of files to upload",
    ),
    db: Any = Depends(get_media_db_for_user),
    current_user: User = Depends(get_request_user),
    usage_log: UsageEventLogger = Depends(get_usage_event_logger),
):
    """
    Thin wrapper for the `/media/add` endpoint.

    This keeps the HTTP contract, auth/RBAC behavior, and side effects
    identical to the legacy implementation while routing through the
    modular `media` package. Processing and persistence logic is
    implemented in the core `persistence.add_media_orchestrate`
    helper.
    """

    return await add_media_persist(
        background_tasks=background_tasks,
        form_data=form_data,
        files=files,
        db=db,
        current_user=current_user,
        usage_log=usage_log,
        response=None,
        request=request,
    )


__all__ = ["router"]
