"""
Watchlist Alert Rules API

CRUD endpoints for user-defined alert rules that trigger notifications
based on watchlist run statistics.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User

from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
from tldw_Server_API.app.core.Watchlists.alert_rules import (
    ALERT_CONDITION_TYPE_VALUES,
    create_alert_rule,
    delete_alert_rule,
    ensure_alert_rules_table,
    list_alert_rules,
    update_alert_rule,
)

router = APIRouter(prefix="/watchlists/alert-rules", tags=["watchlists"])


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AlertRuleCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    condition_type: str  # no_items, error_rate_above, items_below, items_above, run_failed
    condition_value: dict[str, Any] | None = None
    job_id: int | None = None
    severity: str = "warning"


class AlertRuleUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = None
    enabled: bool | None = None
    condition_type: str | None = None
    condition_value: dict[str, Any] | None = None
    job_id: int | None = None
    severity: str | None = None


class AlertRuleResponse(BaseModel):
    id: int
    user_id: str
    job_id: int | None
    name: str
    enabled: bool
    condition_type: str
    condition_value: str
    severity: str
    created_at: str
    updated_at: str


class AlertRuleListResponse(BaseModel):
    items: list[AlertRuleResponse]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_db_path(user_id: str) -> str:
    """Resolve the per-user ChaChaNotes DB path for alert rules storage."""
    base_override = os.environ.get("TLDW_USER_DB_DIR") or None
    user_dir = DatabasePaths.get_user_base_directory(
        user_id,
        base_dir_override=base_override,
    )
    db_path = user_dir / DatabasePaths.CHACHA_DB_NAME
    return str(db_path)


def _user_id_text(current_user: User) -> str:
    raw = getattr(current_user, "id", None)
    if raw is None:
        raw = getattr(current_user, "id_int", None)
    return str(raw)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=AlertRuleListResponse)
async def list_rules(
    job_id: int | None = None,
    current_user: User = Depends(get_request_user),
):
    """List all alert rules for the current user, optionally filtered by job."""
    user_id = _user_id_text(current_user)
    db_path = _get_db_path(user_id)
    await asyncio.to_thread(ensure_alert_rules_table, db_path)
    rules = await asyncio.to_thread(list_alert_rules, db_path, user_id, job_id=job_id)
    return AlertRuleListResponse(
        items=[AlertRuleResponse(**vars(r)) for r in rules]
    )


@router.post("", response_model=AlertRuleResponse, status_code=201)
async def create_rule(
    body: AlertRuleCreate,
    current_user: User = Depends(get_request_user),
):
    """Create a new alert rule."""
    if body.condition_type not in ALERT_CONDITION_TYPE_VALUES:
        raise HTTPException(
            400,
            f"Invalid condition_type. Must be one of: {', '.join(sorted(ALERT_CONDITION_TYPE_VALUES))}",
        )
    user_id = _user_id_text(current_user)
    db_path = _get_db_path(user_id)
    await asyncio.to_thread(ensure_alert_rules_table, db_path)
    rule = await asyncio.to_thread(
        create_alert_rule,
        db_path,
        user_id=user_id,
        name=body.name,
        condition_type=body.condition_type,
        condition_value=body.condition_value,
        job_id=body.job_id,
        severity=body.severity,
    )
    return AlertRuleResponse(**vars(rule))


@router.patch("/{rule_id}", response_model=AlertRuleResponse)
async def update_rule(
    rule_id: int,
    body: AlertRuleUpdate,
    current_user: User = Depends(get_request_user),
):
    """Update an existing alert rule."""
    if body.condition_type is not None and body.condition_type not in ALERT_CONDITION_TYPE_VALUES:
        raise HTTPException(
            400,
            f"Invalid condition_type. Must be one of: {', '.join(sorted(ALERT_CONDITION_TYPE_VALUES))}",
        )
    user_id = _user_id_text(current_user)
    db_path = _get_db_path(user_id)
    await asyncio.to_thread(ensure_alert_rules_table, db_path)
    try:
        rule = await asyncio.to_thread(
            update_alert_rule,
            db_path,
            rule_id,
            user_id,
            **body.model_dump(exclude_none=True),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not rule:
        raise HTTPException(404, "Rule not found or no changes applied")
    return AlertRuleResponse(**vars(rule))


@router.delete("/{rule_id}")
async def delete_rule(
    rule_id: int,
    current_user: User = Depends(get_request_user),
):
    """Delete an alert rule."""
    user_id = _user_id_text(current_user)
    db_path = _get_db_path(user_id)
    await asyncio.to_thread(ensure_alert_rules_table, db_path)
    deleted = await asyncio.to_thread(delete_alert_rule, db_path, rule_id, user_id)
    if not deleted:
        raise HTTPException(404, "Rule not found")
    return {"deleted": True}
