from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    get_auth_principal,
    get_db_transaction,
    get_password_service_dep,
    get_session_manager_dep,
)
from tldw_Server_API.app.api.v1.schemas.admin_schemas import AdminPrivilegedActionRequest
from tldw_Server_API.app.api.v1.schemas.auth_schemas import MessageResponse, SessionResponse
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal
from tldw_Server_API.app.services import admin_sessions_mfa_service

router = APIRouter()


@router.get("/users/{user_id}/sessions", response_model=list[SessionResponse])
async def admin_list_user_sessions(
    user_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
    session_manager=Depends(get_session_manager_dep),
) -> list[SessionResponse]:
    """List active sessions for a user (admin scope)."""
    return await admin_sessions_mfa_service.list_user_sessions(
        principal,
        user_id,
        session_manager,
    )


@router.delete("/users/{user_id}/sessions/{session_id}", response_model=MessageResponse)
async def admin_revoke_user_session(
    user_id: int,
    session_id: int,
    request: AdminPrivilegedActionRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    session_manager=Depends(get_session_manager_dep),
    db=Depends(get_db_transaction),
    password_service=Depends(get_password_service_dep),
) -> MessageResponse:
    """Revoke a specific session for a user (admin scope)."""
    return await admin_sessions_mfa_service.revoke_user_session(
        principal,
        user_id,
        session_id,
        session_manager,
        db,
        password_service,
        request,
    )


@router.post("/users/{user_id}/sessions/revoke-all", response_model=MessageResponse)
async def admin_revoke_all_user_sessions(
    user_id: int,
    request: AdminPrivilegedActionRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    session_manager=Depends(get_session_manager_dep),
    db=Depends(get_db_transaction),
    password_service=Depends(get_password_service_dep),
) -> MessageResponse:
    """Revoke all sessions for a user (admin scope)."""
    return await admin_sessions_mfa_service.revoke_all_user_sessions(
        principal,
        user_id,
        session_manager,
        db,
        password_service,
        request,
    )


@router.get("/users/{user_id}/mfa", response_model=dict[str, Any])
async def admin_get_user_mfa_status(
    user_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> dict[str, Any]:
    """Fetch MFA status for a user (admin scope)."""
    return await admin_sessions_mfa_service.get_user_mfa_status(
        principal,
        user_id,
    )


class BulkMfaStatusResponse(BaseModel):
    mfa_status: dict[str, bool] = Field(default_factory=dict)
    failed_user_ids: list[int] = Field(default_factory=list)


@router.get("/users/mfa/bulk", response_model=BulkMfaStatusResponse)
async def admin_get_bulk_mfa_status(
    ids: str = Query(..., description="Comma-separated user IDs"),
    principal: AuthPrincipal = Depends(get_auth_principal),
) -> BulkMfaStatusResponse:
    """Fetch MFA status for multiple users in a single request."""
    raw_ids = [uid.strip() for uid in ids.split(",") if uid.strip()]
    if not raw_ids:
        raise HTTPException(status_code=400, detail="Invalid user IDs")
    if len(raw_ids) > 200:
        raise HTTPException(status_code=400, detail="Too many user IDs")
    if any(not uid.isdigit() for uid in raw_ids):
        raise HTTPException(status_code=400, detail="Invalid user IDs")
    user_ids = [int(uid) for uid in raw_ids]

    async def _fetch_one(uid: int) -> tuple[int, bool | None]:
        try:
            status = await admin_sessions_mfa_service.get_user_mfa_status(principal, uid)
            enabled = bool(status.get("enabled", False)) if isinstance(status, dict) else False
            return uid, enabled
        except Exception:
            logger.warning("bulk MFA status: failed for user")
            return uid, None

    pairs = await asyncio.gather(*[_fetch_one(uid) for uid in user_ids])
    mfa_status = {str(uid): enabled for uid, enabled in pairs if enabled is not None}
    failed_user_ids = [uid for uid, enabled in pairs if enabled is None]
    return BulkMfaStatusResponse(mfa_status=mfa_status, failed_user_ids=failed_user_ids)


@router.post("/users/{user_id}/mfa/disable", response_model=MessageResponse)
async def admin_disable_user_mfa(
    user_id: int,
    request: AdminPrivilegedActionRequest,
    principal: AuthPrincipal = Depends(get_auth_principal),
    db=Depends(get_db_transaction),
    password_service=Depends(get_password_service_dep),
) -> MessageResponse:
    """Disable MFA for a user (admin scope)."""
    return await admin_sessions_mfa_service.disable_user_mfa(
        principal,
        user_id,
        db,
        password_service,
        request,
    )
