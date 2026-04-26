from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_auth_principal,
    get_db_transaction,
    get_session_manager_dep,
)
from tldw_Server_API.app.api.v1.schemas.user_profile_schemas import (
    UserProfileBatchResponse,
    UserProfileBulkUpdateRequest,
    UserProfileBulkUpdateResponse,
    UserProfileResponse,
    UserProfileUpdateRequest,
    UserProfileUpdateResponse,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import admin_profiles_service

router = APIRouter()


def _get_emit_admin_audit_event():
    from tldw_Server_API.app.api.v1.endpoints import admin as admin_mod

    return admin_mod._emit_admin_audit_event


@router.get(
    "/users/profile",
    response_model=UserProfileBatchResponse,
    response_model_exclude_none=True,
)
async def admin_list_user_profiles(
    http_request: Request,
    sections: str | None = Query(None, description="Comma-separated list of sections to include"),
    include_sources: bool = Query(False, description="Include per-field source attribution"),
    include_raw: bool = Query(False, description="Include raw stored overrides"),
    mask_secrets: bool = Query(True, description="Mask secret values in the response"),
    user_ids: str | None = Query(None, description="Comma-separated list of user IDs to include"),
    org_id: int | None = Query(None, description="Restrict to a specific organization"),
    team_id: int | None = Query(None, description="Restrict to a specific team"),
    role: str | None = None,
    is_active: bool | None = None,
    search: str | None = None,
    page: int = Query(1, ge=1),
    limit: int = Query(25, ge=1, le=100),
    principal: AuthPrincipal = Depends(get_auth_principal),
    session_manager: Any = Depends(get_session_manager_dep),
) -> UserProfileBatchResponse:
    """Get batch profile summaries within admin scope."""
    response, audit_info = await admin_profiles_service.list_user_profiles(
        principal=principal,
        sections=sections,
        include_sources=include_sources,
        include_raw=include_raw,
        mask_secrets=mask_secrets,
        user_ids=user_ids,
        org_id=org_id,
        team_id=team_id,
        role=role,
        is_active=is_active,
        search=search,
        page=page,
        limit=limit,
        session_manager=session_manager,
    )
    if audit_info:
        try:
            await _get_emit_admin_audit_event()(http_request, principal, **audit_info)
        except Exception:
            logger.warning("Admin audit emission failed")
    return response


@router.get(
    "/users/{user_id}/profile",
    response_model=UserProfileResponse,
    response_model_exclude_none=True,
)
async def admin_get_user_profile(
    user_id: int,
    http_request: Request,
    sections: str | None = Query(None, description="Comma-separated list of sections to include"),
    include_sources: bool = Query(False, description="Include per-field source attribution"),
    include_raw: bool = Query(False, description="Include raw stored overrides"),
    mask_secrets: bool = Query(True, description="Mask secret values in the response"),
    principal: AuthPrincipal = Depends(get_auth_principal),
    session_manager: Any = Depends(get_session_manager_dep),
) -> UserProfileResponse:
    """Get a unified user profile (admin scope)."""
    response, audit_info = await admin_profiles_service.get_user_profile(
        user_id=user_id,
        principal=principal,
        sections=sections,
        include_sources=include_sources,
        include_raw=include_raw,
        mask_secrets=mask_secrets,
        session_manager=session_manager,
    )
    if audit_info:
        try:
            await _get_emit_admin_audit_event()(http_request, principal, **audit_info)
        except Exception:
            logger.warning("Admin audit emission failed")
    return response


@router.patch("/users/{user_id}/profile", response_model=UserProfileUpdateResponse)
async def admin_update_user_profile(
    user_id: int,
    payload: UserProfileUpdateRequest,
    http_request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db: Any = Depends(get_db_transaction),
) -> UserProfileUpdateResponse:
    """Update a user's profile (admin scope)."""
    response, audit_info = await admin_profiles_service.update_user_profile(
        user_id=user_id,
        payload=payload,
        principal=principal,
        db=db,
    )
    if audit_info:
        try:
            await _get_emit_admin_audit_event()(http_request, principal, **audit_info)
        except Exception:
            logger.warning("Admin audit emission failed")
    return response


@router.post("/users/profile/bulk", response_model=UserProfileBulkUpdateResponse)
async def admin_bulk_update_user_profiles(
    payload: UserProfileBulkUpdateRequest,
    http_request: Request,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> UserProfileBulkUpdateResponse:
    """Bulk update user profiles (admin scope)."""
    response, audit_info = await admin_profiles_service.bulk_update_user_profiles(
        payload=payload,
        principal=principal,
    )
    if audit_info:
        try:
            await _get_emit_admin_audit_event()(http_request, principal, **audit_info)
        except Exception:
            logger.warning("Admin audit emission failed")
    return response
