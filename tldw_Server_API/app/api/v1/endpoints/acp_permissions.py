"""ACP Permissions sub-module.

Provides CRUD endpoints for ACP permission policies and persisted
permission decisions (the "remember" pattern).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User

from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB

router = APIRouter(prefix="/acp/permissions", tags=["acp-permissions"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_acp_db() -> ACPSessionsDB:
    """Return the shared ACPSessionsDB singleton.

    Re-uses the same DB instance that the ACP session store already holds.
    """
    from tldw_Server_API.app.services.admin_acp_sessions_service import (
        _store as _module_store,
    )

    # If the async singleton has already been initialized, reuse its DB
    if _module_store is not None:
        try:
            return _module_store.get_db()
        except Exception:
            return ACPSessionsDB()

    # Fallback: instantiate a fresh one (default path)
    return ACPSessionsDB()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class PermissionDecisionCreate(BaseModel):
    tool_pattern: str = Field(..., description="fnmatch pattern for tool names (e.g. 'bash', 'file_*')")
    decision: str = Field(..., description="'allow' or 'deny'")
    scope: str = Field("session", description="'session' or 'global'")
    session_id: str | None = Field(None, description="Required when scope is 'session'")
    persona_id: str | None = None
    reason: str | None = None


class PermissionDecisionOut(BaseModel):
    id: str
    user_id: int
    tool_pattern: str
    decision: str
    scope: str
    session_id: str | None = None
    persona_id: str | None = None
    created_at: str
    expires_at: str | None = None
    reason: str | None = None


class PermissionDecisionDeleteResponse(BaseModel):
    status: str
    id: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/decisions",
    response_model=list[PermissionDecisionOut],
    summary="List persisted permission decisions",
)
async def list_permission_decisions(
    user: User = Depends(get_request_user),
) -> list[dict[str, Any]]:
    """Return all non-expired remembered permission decisions for the current user."""
    db = _get_acp_db()
    return db.list_permission_decisions(user_id=user.id)


@router.post(
    "/decisions",
    response_model=PermissionDecisionOut,
    status_code=201,
    summary="Create a permission decision",
)
async def create_permission_decision(
    body: PermissionDecisionCreate,
    user: User = Depends(get_request_user),
) -> dict[str, Any]:
    """Manually add a remembered permission decision."""
    if body.decision not in ("allow", "deny"):
        raise HTTPException(status_code=400, detail="decision must be 'allow' or 'deny'")
    if body.scope not in ("session", "global"):
        raise HTTPException(status_code=400, detail="scope must be 'session' or 'global'")
    if body.scope == "session" and not body.session_id:
        raise HTTPException(
            status_code=422,
            detail="session_id is required when scope is 'session'",
        )

    from tldw_Server_API.app.core.Agent_Client_Protocol.permission_decision_service import (
        PermissionDecisionService,
    )

    db = _get_acp_db()
    svc = PermissionDecisionService(db)
    decision_id = svc.persist(
        user_id=user.id,
        tool_pattern=body.tool_pattern,
        decision=body.decision,
        scope=body.scope,
        session_id=body.session_id,
        persona_id=body.persona_id,
        reason=body.reason,
    )
    return db.get_permission_decision(decision_id)  # type: ignore[return-value]


@router.delete(
    "/decisions/{decision_id}",
    summary="Revoke a permission decision",
    response_model=PermissionDecisionDeleteResponse,
)
async def revoke_permission_decision(
    decision_id: str,
    user: User = Depends(get_request_user),
) -> PermissionDecisionDeleteResponse:
    """Revoke (delete) a remembered permission decision."""
    db = _get_acp_db()
    # Verify ownership
    existing = db.get_permission_decision(decision_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Decision not found")
    if existing["user_id"] != user.id:
        raise HTTPException(status_code=403, detail="Not your decision to revoke")
    db.delete_permission_decision(decision_id)
    return PermissionDecisionDeleteResponse(status="deleted", id=decision_id)
