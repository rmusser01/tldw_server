from __future__ import annotations

import contextlib
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_auth_principal,
    get_db_transaction,
)
from tldw_Server_API.app.api.v1.schemas.user_profile_schemas import (
    UserProfileUpdateRequest,
)
from tldw_Server_API.app.api.v1.endpoints.users import (
    _USERS_AUDIT_EXCEPTIONS,
    _emit_user_profile_audit_event,
    _require_principal_active_verified,
)
from tldw_Server_API.app.core.AuthNZ.database import get_db_pool
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.core.UserProfiles.command_service import ProfileCommandService
from tldw_Server_API.app.core.UserProfiles.contracts import (
    ProfileContractMode,
    ProfileUpdateCommand,
)

router = APIRouter()


class UserProfileV2UpdateResponse(BaseModel):
    profile_version: datetime = Field(..., description="Profile version timestamp")
    applied: list[str] = Field(default_factory=list)


@router.patch("/users/me/profile", response_model=UserProfileV2UpdateResponse)
async def update_current_user_profile_v2(
    payload: UserProfileUpdateRequest,
    http_request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_db_transaction),
) -> UserProfileV2UpdateResponse:
    user_context = await _require_principal_active_verified(principal)
    user_id = int(user_context["id"])

    db_pool = await get_db_pool()
    command_service = ProfileCommandService(db_pool=db_pool)
    command = ProfileUpdateCommand(
        actor_user_id=user_id,
        target_user_id=user_id,
        updates=tuple((entry.key, entry.value) for entry in payload.updates),
        roles=frozenset({"user"}),
        dry_run=payload.dry_run,
        expected_profile_version=payload.profile_version,
        contract_mode=ProfileContractMode.CLEAN_V2,
    )
    result = await command_service.apply(command, db_conn=db, scope=None)
    if result.status_code != status.HTTP_200_OK:
        raise HTTPException(
            status_code=result.status_code,
            detail=result.detail or result.error_code or "profile_update_failed",
        )

    with contextlib.suppress(_USERS_AUDIT_EXCEPTIONS):
        if not payload.dry_run:
            await _emit_user_profile_audit_event(
                http_request,
                user_id=user_id,
                update_keys=[entry.key for entry in payload.updates],
                applied_count=len(result.applied),
                skipped_count=0,
                dry_run=False,
            )

    return UserProfileV2UpdateResponse(
        profile_version=result.profile_version,
        applied=list(result.applied),
    )
