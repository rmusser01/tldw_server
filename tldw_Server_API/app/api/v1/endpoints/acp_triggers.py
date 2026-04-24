"""ACP Triggers sub-module.

Provides:
- ``POST /acp/triggers/webhook/{trigger_id}`` -- inbound webhook receiver
  (NO auth required; uses HMAC verification instead)
- CRUD endpoints for managing webhook trigger configurations
  (auth required via ``get_request_user``)
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import (
    User,
    get_request_user,
)

router = APIRouter(prefix="/acp/triggers", tags=["acp-triggers"])

_WEBHOOK_ERROR_STATUS = {
    "trigger_not_found": 404,
    "verification_failed": 403,
    "signature_invalid": 403,
    "rate_limit_exceeded": 429,
}

_WEBHOOK_CLIENT_ERROR_CODES = frozenset(
    {
        "trigger_not_found",
        "trigger_disabled",
        "verification_failed",
        "signature_invalid",
        "rate_limit_exceeded",
    }
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class TriggerCreateRequest(BaseModel):
    name: str = Field(..., description="Human-readable trigger name")
    source_type: str = Field(
        "generic",
        description="Webhook source: 'github', 'slack', or 'generic'",
    )
    secret: str = Field(..., description="HMAC shared secret (will be encrypted at rest)")
    agent_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Agent configuration (cwd, agent_type, model, etc.)",
    )
    prompt_template: str = Field(
        "",
        description="Prompt template with {payload} and {event_type} placeholders",
    )
    enabled: bool = Field(True, description="Whether the trigger is active")


class TriggerUpdateRequest(BaseModel):
    name: str | None = None
    source_type: str | None = None
    secret: str | None = None
    agent_config: dict[str, Any] | None = None
    prompt_template: str | None = None
    enabled: bool | None = None


class TriggerResponse(BaseModel):
    id: str
    name: str
    source_type: str
    owner_user_id: int
    agent_config: dict[str, Any] = Field(default_factory=dict)
    prompt_template: str = ""
    enabled: bool = True
    created_at: str | None = None
    updated_at: str | None = None


class WebhookResult(BaseModel):
    task_id: str | None = None
    status: str
    error: str | None = None


class TriggerDeleteResponse(BaseModel):
    status: str
    id: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_trigger_manager():
    """Lazily construct and return an ACPTriggerManager.

    Uses the same ``ACPSessionsDB`` instance backing the ACP session store
    and a ``TriggerSecretManager`` initialized from the environment.
    """
    from tldw_Server_API.app.core.Agent_Client_Protocol.triggers import (
        ACPTriggerManager,
        TriggerSecretManager,
    )
    from tldw_Server_API.app.core.DB_Management.ACP_Sessions_DB import ACPSessionsDB

    # Re-use the DB instance from the session store when possible
    try:
        from tldw_Server_API.app.services.admin_acp_sessions_service import (
            _store as _module_store,
        )
        db: ACPSessionsDB | None = None
        if _module_store is not None:
            db = _module_store.get_db()
    except Exception:
        db = None
    if db is None:
        db = ACPSessionsDB()

    try:
        secret_mgr = TriggerSecretManager()
    except (ImportError, ValueError) as exc:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=503,
            detail=f"Webhook trigger encryption not configured: {exc}",
        )
    return ACPTriggerManager(db=db, secret_manager=secret_mgr)


def _sanitize_webhook_error_detail(result: dict[str, Any]) -> tuple[int, dict[str, str]]:
    status_value = str(result.get("status") or "error")
    raw_error = str(result.get("error") or "").strip()
    if raw_error in _WEBHOOK_CLIENT_ERROR_CODES:
        error_code = raw_error
        status_code = _WEBHOOK_ERROR_STATUS.get(
            error_code,
            400 if status_value == "rejected" else 503,
        )
    else:
        error_code = "internal_error"
        status_code = 503
    return status_code, {"status": status_value, "error": error_code}


def _sanitize_webhook_success_result(result: dict[str, Any]) -> dict[str, str | None]:
    """Return the minimal public success payload for accepted webhooks."""
    return {
        "status": "accepted",
        "task_id": str(result.get("task_id") or "").strip() or None,
    }


# ---------------------------------------------------------------------------
# Inbound webhook endpoint (NO AUTH -- uses HMAC)
# ---------------------------------------------------------------------------


@router.post("/webhook/{trigger_id}", response_model=WebhookResult)
async def receive_webhook(trigger_id: str, request: Request) -> dict[str, Any]:
    """Receive an inbound webhook and trigger an ACP agent run.

    This endpoint does NOT require authentication.  Instead, the request
    body is verified against the trigger's stored HMAC secret using the
    provider-specific signing scheme.
    """
    payload_body = await request.body()
    # Collect headers as a lowercase dict
    headers = {k.lower(): v for k, v in request.headers.items()}

    mgr = _get_trigger_manager()
    result = await mgr.handle_webhook(trigger_id, payload_body, headers)

    if result.get("status") != "accepted":
        status_code, safe_detail = _sanitize_webhook_error_detail(result)
        raise HTTPException(status_code=status_code, detail=safe_detail)

    return _sanitize_webhook_success_result(result)


# ---------------------------------------------------------------------------
# CRUD endpoints (AUTH required)
# ---------------------------------------------------------------------------


@router.post("", response_model=TriggerResponse, status_code=201)
async def create_trigger(
    body: TriggerCreateRequest,
    user: User = Depends(get_request_user),
) -> dict[str, Any]:
    """Create a new webhook trigger."""
    if body.source_type not in ("github", "slack", "generic"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid source_type '{body.source_type}'; must be github, slack, or generic",
        )

    mgr = _get_trigger_manager()
    trigger_id = mgr.create_trigger(
        name=body.name,
        source_type=body.source_type,
        secret=body.secret,
        owner_user_id=user.id,
        agent_config=body.agent_config,
        prompt_template=body.prompt_template,
        enabled=body.enabled,
    )

    trigger = mgr.get_trigger(trigger_id)
    if trigger is None:
        raise HTTPException(status_code=500, detail="Failed to create trigger")
    # Strip secret from response
    trigger.pop("secret_encrypted", None)
    return trigger


@router.get("", response_model=list[TriggerResponse])
async def list_triggers(
    user: User = Depends(get_request_user),
) -> list[dict[str, Any]]:
    """List triggers for the current user."""
    mgr = _get_trigger_manager()
    return mgr.list_triggers(user.id)


@router.get("/{trigger_id}", response_model=TriggerResponse)
async def get_trigger(
    trigger_id: str,
    user: User = Depends(get_request_user),
) -> dict[str, Any]:
    """Get trigger details."""
    mgr = _get_trigger_manager()
    trigger = mgr.get_trigger(trigger_id)
    if trigger is None:
        raise HTTPException(status_code=404, detail="Trigger not found")
    if trigger.get("owner_user_id") != user.id:
        raise HTTPException(status_code=403, detail="Not your trigger")
    # Strip secret from response
    trigger.pop("secret_encrypted", None)
    return trigger


@router.put("/{trigger_id}", response_model=TriggerResponse)
async def update_trigger(
    trigger_id: str,
    body: TriggerUpdateRequest,
    user: User = Depends(get_request_user),
) -> dict[str, Any]:
    """Update a trigger."""
    mgr = _get_trigger_manager()
    existing = mgr.get_trigger(trigger_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Trigger not found")
    if existing.get("owner_user_id") != user.id:
        raise HTTPException(status_code=403, detail="Not your trigger")

    if body.source_type is not None and body.source_type not in ("github", "slack", "generic"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid source_type '{body.source_type}'",
        )

    update_kwargs: dict[str, Any] = {}
    if body.name is not None:
        update_kwargs["name"] = body.name
    if body.source_type is not None:
        update_kwargs["source_type"] = body.source_type
    if body.secret is not None:
        update_kwargs["secret"] = body.secret
    if body.agent_config is not None:
        update_kwargs["agent_config"] = body.agent_config
    if body.prompt_template is not None:
        update_kwargs["prompt_template"] = body.prompt_template
    if body.enabled is not None:
        update_kwargs["enabled"] = body.enabled

    if update_kwargs:
        mgr.update_trigger(trigger_id, **update_kwargs)

    updated = mgr.get_trigger(trigger_id)
    if updated is None:
        raise HTTPException(status_code=500, detail="Trigger disappeared after update")
    updated.pop("secret_encrypted", None)
    return updated


@router.delete("/{trigger_id}", response_model=TriggerDeleteResponse)
async def delete_trigger(
    trigger_id: str,
    user: User = Depends(get_request_user),
) -> TriggerDeleteResponse:
    """Delete a trigger."""
    mgr = _get_trigger_manager()
    existing = mgr.get_trigger(trigger_id)
    if existing is None:
        raise HTTPException(status_code=404, detail="Trigger not found")
    if existing.get("owner_user_id") != user.id:
        raise HTTPException(status_code=403, detail="Not your trigger")

    mgr.delete_trigger(trigger_id)
    return TriggerDeleteResponse(status="deleted", id=trigger_id)
