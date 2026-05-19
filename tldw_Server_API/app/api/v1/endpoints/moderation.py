# moderation.py
# Description: Moderation configuration endpoints gated by admin role +
# SYSTEM_CONFIGURE permission (per-user overrides and blocklist)

from __future__ import annotations

import asyncio
from functools import partial
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    CurrentPrincipal,
    RequirePermission,
    RequireRole,
)
from tldw_Server_API.app.api.v1.schemas.moderation_schemas import (
    BlocklistAppendRequest,
    BlocklistAppendResponse,
    BlocklistDeleteResponse,
    BlocklistLintItem,
    BlocklistLintRequest,
    BlocklistLintResponse,
    BlocklistManagedItem,
    BlocklistManagedResponse,
    ModerationBlocklistUpdate,
    ModerationReviewAuditResponse,
    ModerationReviewBulkDecisionRequest,
    ModerationReviewBulkDecisionResponse,
    ModerationReviewDecisionRequest,
    ModerationReviewDecisionResponse,
    ModerationReviewItem,
    ModerationReviewListResponse,
    ModerationReviewUndoRequest,
    ModerationSettingsResponse,
    ModerationSettingsUpdate,
    ModerationTestRequest,
    ModerationTestResponse,
    ModerationUserOverride,
    ModerationUserOverrideLookupResponse,
    ModerationUserOverridesResponse,
)
from tldw_Server_API.app.core.AuthNZ.permissions import (
    MODERATION_AUDIT_READ,
    MODERATION_REVIEW_BULK_DECIDE,
    MODERATION_REVIEW_DECIDE,
    MODERATION_REVIEW_READ,
    SYSTEM_CONFIGURE,
)
from tldw_Server_API.app.core.Moderation.moderation_service import get_moderation_service
from tldw_Server_API.app.core.Moderation.review_service import get_moderation_review_service
from tldw_Server_API.app.core.Moderation.supervised_policy import (
    GuardianModerationProxy,
    bootstrap_guardian_moderation_runtime,
)

router = APIRouter()

rules_router = APIRouter(
    dependencies=[
        Depends(RequireRole("admin")),
        Depends(RequirePermission(SYSTEM_CONFIGURE)),
    ]
)

review_router = APIRouter()


def _normalize_etag_list(value: str | None) -> list[str]:
    if not value:
        return []
    tokens: list[str] = []
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        # RFC 7232 wildcard token for If-Match.
        if token == "*":  # nosec B105
            tokens.append(token)
            continue
        if token.lower().startswith("w/"):
            token = token[2:].strip()
        if len(token) >= 2 and token[0] == token[-1] == '"':
            token = token[1:-1]
        if token:
            tokens.append(token)
    return tokens


def _normalize_optional_identifier(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


def _normalize_chat_type(value: str | None) -> str:
    normalized = str(value or "regular").strip().lower()
    return normalized or "regular"


async def _run_review_store_call(func: Any, /, *args: Any, **kwargs: Any) -> Any:
    return await asyncio.to_thread(partial(func, *args, **kwargs))


@rules_router.get("/moderation/users", response_model=ModerationUserOverridesResponse, summary="List all per-user moderation overrides", tags=["moderation"])
async def list_user_overrides() -> ModerationUserOverridesResponse:
    """List all per-user moderation override entries."""
    svc = get_moderation_service()
    return {"overrides": svc.list_user_overrides()}


@rules_router.get(
    "/moderation/users/{user_id}",
    response_model=ModerationUserOverrideLookupResponse,
    summary="Get per-user moderation override",
    tags=["moderation"],
)
async def get_user_override(user_id: str) -> ModerationUserOverrideLookupResponse:
    """Return the per-user moderation override for the given user id."""
    svc = get_moderation_service()
    data = svc.list_user_overrides().get(str(user_id))
    if data is None:
        return ModerationUserOverrideLookupResponse(exists=False, override={})
    return ModerationUserOverrideLookupResponse(exists=True, override=data)


@rules_router.put("/moderation/users/{user_id}", response_model=dict, summary="Set per-user moderation override", tags=["moderation"])
async def set_user_override(user_id: str, override: ModerationUserOverride) -> dict[str, Any]:
    """Set or replace a per-user moderation override entry."""
    svc = get_moderation_service()
    status_info = svc.set_user_override(user_id, override.model_dump(exclude_none=True))
    status_dict = status_info if isinstance(status_info, dict) else {}
    if not status_dict.get("ok"):
        error_detail = status_dict.get("error", "Failed to persist override")
        logger.error("Moderation override persist failed")
        error_type = str(status_dict.get("error_type", "")).strip().lower()
        if error_type == "validation":
            status_code = status.HTTP_400_BAD_REQUEST
        elif error_type == "persistence":
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        else:
            # Backward-compatible fallback for older service responses.
            err_text = str(error_detail).lower()
            is_validation_like = any(token in err_text for token in ("invalid", "dangerous", "required"))
            status_code = status.HTTP_400_BAD_REQUEST if is_validation_like else status.HTTP_500_INTERNAL_SERVER_ERROR
        raise HTTPException(
            status_code=status_code,
            detail=error_detail if status_code == status.HTTP_400_BAD_REQUEST else "Failed to persist override",
        )
    data = svc.list_user_overrides().get(str(user_id), {})
    # Surface whether the change was persisted
    if isinstance(data, dict):
        data = {**data, "persisted": bool(status_dict.get("persisted", False))}
    return data


@rules_router.delete("/moderation/users/{user_id}", summary="Delete per-user moderation override", tags=["moderation"])
async def delete_user_override(user_id: str) -> dict[str, Any]:
    """Delete a per-user moderation override entry."""
    svc = get_moderation_service()
    status_info = svc.delete_user_override(user_id)
    status_dict = status_info if isinstance(status_info, dict) else {}
    if not status_dict.get("ok"):
        detail = status_dict.get("error", "Override not found or failed to delete")
        code = (
            status.HTTP_404_NOT_FOUND
            if status_dict.get("error") == "not found"
            else status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        raise HTTPException(status_code=code, detail=detail)
    return {"status": "deleted", "persisted": bool(status_dict.get("persisted", False))}


@rules_router.get("/moderation/blocklist", response_model=list, summary="Get current blocklist lines", tags=["moderation"])
async def get_blocklist() -> list[str]:
    """Return the current moderation blocklist lines."""
    svc = get_moderation_service()
    return svc.get_blocklist_lines()


@rules_router.put("/moderation/blocklist", summary="Replace blocklist with provided lines", tags=["moderation"])
async def update_blocklist(data: ModerationBlocklistUpdate) -> dict[str, Any]:
    """Replace the entire blocklist with the provided lines."""
    svc = get_moderation_service()
    lines = data.lines or []
    try:
        lint = svc.lint_blocklist_lines(lines)
    except Exception as exc:
        logger.exception("Failed to lint blocklist lines")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to lint blocklist lines",
        ) from exc
    if int(lint.get("invalid_count", 0)) > 0:
        invalid_items = [it for it in (lint.get("items") or []) if not it.get("ok")]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid blocklist lines", "invalid_items": invalid_items},
        )
    ok = svc.set_blocklist_lines(lines)
    if not ok:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to persist blocklist")
    return {"status": "ok", "count": len(lines)}


@rules_router.get(
    "/moderation/policy/effective",
    summary="Inspect effective moderation policy for a user",
    tags=["moderation"],
)
async def get_effective_policy(user_id: str | None = Query(None, description="User ID to compute effective policy; optional")) -> dict[str, Any]:
    """Return the effective moderation policy snapshot for an optional user."""
    svc = get_moderation_service()
    try:
        snapshot = svc.effective_policy_snapshot(user_id)
    except Exception as exc:
        logger.exception("Failed to compute effective moderation policy")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute effective policy",
        ) from exc
    else:
        return snapshot


@rules_router.post(
    "/moderation/reload",
    summary="Reload moderation configuration from disk",
    tags=["moderation"],
)
async def reload_moderation() -> dict[str, Any]:
    """Reload moderation configuration from disk."""
    svc = get_moderation_service()
    try:
        svc.reload()
    except Exception as exc:
        logger.exception("Failed to reload moderation configuration")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reload moderation",
        ) from exc
    else:
        return {"status": "ok"}


@rules_router.get(
    "/moderation/settings",
    response_model=ModerationSettingsResponse,
    summary="Get runtime moderation settings and effective state",
    tags=["moderation"],
)
async def get_moderation_settings() -> ModerationSettingsResponse:
    """Return runtime moderation settings and effective state."""
    svc = get_moderation_service()
    try:
        data = svc.get_settings()
    except Exception as exc:
        logger.exception("Failed to get moderation settings")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get moderation settings",
        ) from exc
    else:
        return data


@rules_router.put(
    "/moderation/settings",
    response_model=ModerationSettingsResponse,
    summary="Update runtime moderation settings (non-persistent)",
    tags=["moderation"],
)
async def update_moderation_settings(body: ModerationSettingsUpdate) -> ModerationSettingsResponse:
    """Update runtime moderation settings without persisting by default."""
    svc = get_moderation_service()
    try:
        fields_set = getattr(body, "model_fields_set", set())
        clear_pii = ("pii_enabled" in fields_set and body.pii_enabled is None)
        clear_categories = ("categories_enabled" in fields_set and body.categories_enabled is None)
        data = svc.update_settings(
            pii_enabled=body.pii_enabled,
            categories_enabled=body.categories_enabled,
            persist=bool(body.persist),
            clear_pii=clear_pii,
            clear_categories=clear_categories,
        )
        if isinstance(data, dict) and data.get("ok") is False:
            error_detail = str(data.get("error", "Failed to update moderation settings"))
            error_type = str(data.get("error_type", "")).strip().lower()
            if error_type == "validation":
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_detail,
                )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update moderation settings",
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Failed to update moderation settings")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update moderation settings",
        ) from exc
    else:
        return data


@rules_router.get(
    "/moderation/blocklist/managed",
    response_model=BlocklistManagedResponse,
    summary="Managed blocklist listing with version",
    tags=["moderation"],
)
async def get_blocklist_managed(response: Response) -> BlocklistManagedResponse:
    """Return the managed blocklist with version metadata for concurrency control."""
    svc = get_moderation_service()
    state = svc.get_blocklist_state()
    # Set ETag header for clients to use with If-Match
    version = state.get("version", "")
    response.headers["ETag"] = f"\"{version}\"" if version else ""
    items = [BlocklistManagedItem(**it) for it in (state.get("items") or [])]
    return BlocklistManagedResponse(version=state.get("version", ""), items=items)


@rules_router.post(
    "/moderation/blocklist/append",
    response_model=BlocklistAppendResponse,
    summary="Append a blocklist line (optimistic concurrency)",
    tags=["moderation"],
)
async def append_blocklist_line(
    payload: BlocklistAppendRequest,
    response: Response,
    if_match: str | None = Header(None, alias="If-Match"),
) -> BlocklistAppendResponse:
    """Append a blocklist line using optimistic concurrency via If-Match."""
    tokens = _normalize_etag_list(if_match)
    if not tokens:
        raise HTTPException(status_code=428, detail="If-Match header is required")
    svc = get_moderation_service()
    try:
        lint = svc.lint_blocklist_lines([payload.line])
    except Exception as exc:
        logger.exception("Failed to lint blocklist line")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to lint blocklist line",
        ) from exc
    if int(lint.get("invalid_count", 0)) > 0:
        invalid_items = [it for it in (lint.get("items") or []) if not it.get("ok")]
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"error": "Invalid blocklist line", "invalid_items": invalid_items},
        )
    expected_version = ""
    if "*" not in tokens:
        current_state = svc.get_blocklist_state()
        current_version = str(current_state.get("version", ""))
        if current_version not in tokens:
            raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail="Version conflict")
        expected_version = current_version
    ok, state = svc.append_blocklist_line(expected_version, payload.line)
    if not ok:
        if state.get("conflict"):
            raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail="Version conflict")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to append blocklist line")
    version = str(state.get("version", ""))
    items = state.get("items") or []
    # New index is last
    index = len(items) - 1
    response.headers["ETag"] = f"\"{version}\"" if version else ""
    return BlocklistAppendResponse(version=version, index=index, count=len(items))


@rules_router.delete(
    "/moderation/blocklist/{item_id}",
    response_model=BlocklistDeleteResponse,
    summary="Delete a blocklist entry by index (optimistic concurrency)",
    tags=["moderation"],
)
async def delete_blocklist_item(
    item_id: int,
    response: Response,
    if_match: str | None = Header(None, alias="If-Match"),
) -> BlocklistDeleteResponse:
    """Delete a blocklist entry by index using optimistic concurrency."""
    tokens = _normalize_etag_list(if_match)
    if not tokens:
        raise HTTPException(status_code=428, detail="If-Match header is required")
    svc = get_moderation_service()
    expected_version = ""
    if "*" not in tokens:
        current_state = svc.get_blocklist_state()
        current_version = str(current_state.get("version", ""))
        if current_version not in tokens:
            raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail="Version conflict")
        expected_version = current_version
    ok, state = svc.delete_blocklist_index(expected_version, item_id)
    if not ok:
        if state.get("conflict"):
            raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED, detail="Version conflict")
        detail = state.get("error", "Unknown error")
        code = status.HTTP_400_BAD_REQUEST if detail == "index out of range" else status.HTTP_500_INTERNAL_SERVER_ERROR
        if code == status.HTTP_500_INTERNAL_SERVER_ERROR:
            detail = "Failed to delete blocklist line"
        raise HTTPException(status_code=code, detail=detail)
    version = str(state.get("version", ""))
    items = state.get("items") or []
    response.headers["ETag"] = f"\"{version}\"" if version else ""
    return BlocklistDeleteResponse(version=version, count=len(items))


@rules_router.post(
    "/moderation/blocklist/lint",
    response_model=BlocklistLintResponse,
    summary="Validate blocklist lines without persisting",
    tags=["moderation"],
)
async def lint_blocklist(
    payload: BlocklistLintRequest,
) -> BlocklistLintResponse:
    """Validate blocklist lines without persisting changes."""
    svc = get_moderation_service()
    lines = []
    if payload.lines:
        lines = payload.lines
    elif payload.line:
        lines = [payload.line]
    else:
        raise HTTPException(status_code=400, detail="Provide 'lines' or 'line'")
    try:
        res = svc.lint_blocklist_lines(lines)
        items = [BlocklistLintItem(**it) for it in (res.get("items") or [])]
    except Exception as exc:
        logger.exception("Failed to lint blocklist lines")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to lint blocklist lines",
        ) from exc
    else:
        return BlocklistLintResponse(
            items=items,
            valid_count=int(res.get("valid_count", 0)),
            invalid_count=int(res.get("invalid_count", 0)),
        )


@rules_router.post(
    "/moderation/test",
    response_model=ModerationTestResponse,
    summary="Test moderation against sample text for a user",
    tags=["moderation"],
)
async def test_moderation(payload: ModerationTestRequest) -> ModerationTestResponse:
    """Evaluate sample text against the effective moderation policy for a user."""
    base_service = get_moderation_service()
    service = base_service
    user_id = _normalize_optional_identifier(payload.user_id)

    supervised_engine = None
    guardian_chat_type = "regular"
    guardian_dep_uid = None
    if payload.apply_guardian_overlay:
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id is required when apply_guardian_overlay=true",
            )
        dependent_user_id = _normalize_optional_identifier(payload.dependent_user_id) or user_id
        if dependent_user_id != user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="dependent_user_id must match user_id for live-chat guardian simulation",
            )
        guardian_chat_type = _normalize_chat_type(payload.chat_type)
        runtime = bootstrap_guardian_moderation_runtime(
            user_id=user_id,
            dependent_user_id=dependent_user_id,
            chat_type=guardian_chat_type,
        )
        guardian_dep_uid = runtime.dependent_user_id
        if runtime.supervised_engine is not None:
            supervised_engine = runtime.supervised_engine
            service = GuardianModerationProxy(
                base_service,
                runtime.supervised_engine,
                runtime.dependent_user_id,
                chat_type=runtime.chat_type,
            )

    effective_policy = service.get_effective_policy(user_id)
    result = service.evaluate_text(payload.text, effective_policy, payload.phase)

    action = result.action
    category = result.category

    # When guardian overlay is active, run a direct supervised-engine pass
    # so notify-only rules are preserved (the proxy coerces notify→warn).
    if supervised_engine is not None and guardian_dep_uid is not None:
        supervised_result = supervised_engine.check_text(
            payload.text,
            guardian_dep_uid,
            phase=payload.phase,
            chat_type=guardian_chat_type,
        )
        if supervised_result.action == "notify" and action == "pass":
            # The base pipeline saw nothing actionable but the supervised
            # engine matched a notify-only rule — surface it.
            action = "notify"
            category = supervised_result.matched_category or category
        elif supervised_result.action == "notify" and action == "warn":
            # The proxy coerced notify→warn; restore the original action
            # when the base policy itself didn't produce the warn.
            base_result = base_service.evaluate_text(
                payload.text,
                base_service.get_effective_policy(user_id),
                payload.phase,
            )
            if base_result.action == "pass":
                action = "notify"
                category = supervised_result.matched_category or category

    return ModerationTestResponse(
        flagged=action != "pass",
        action=action,
        sample=result.sample,
        redacted_text=result.redacted_text,
        effective=effective_policy.to_dict(),
        category=category,
    )


@review_router.get(
    "/moderation/review/items",
    response_model=ModerationReviewListResponse,
    summary="List sanitized moderation review items",
    tags=["moderation-review"],
    dependencies=[Depends(RequirePermission(MODERATION_REVIEW_READ))],
)
async def list_review_items(
    status_filter: str | None = Query(None, alias="status"),
    category: str | None = Query(None),
    severity: str | None = Query(None),
    source_type: str | None = Query(None),
    source_id: str | None = Query(None),
    user_id: str | None = Query(None),
    q: str | None = Query(None),
    sort: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    cursor: str | None = Query(None),
) -> ModerationReviewListResponse:
    """List sanitized moderation review items with optional filters and pagination."""
    service = get_moderation_review_service()
    return await _run_review_store_call(
        service.list_items,
        status=status_filter,
        category=category,
        severity=severity,
        source_type=source_type,
        source_id=source_id,
        user_id=user_id,
        q=q,
        sort=sort,
        limit=limit,
        cursor=cursor,
    )


@review_router.get(
    "/moderation/review/items/{item_id}",
    response_model=ModerationReviewItem,
    summary="Get sanitized moderation review item detail",
    tags=["moderation-review"],
    dependencies=[Depends(RequirePermission(MODERATION_REVIEW_READ))],
)
async def get_review_item(item_id: str) -> ModerationReviewItem:
    """Return a sanitized moderation review item with decision history."""
    service = get_moderation_review_service()
    item = await _run_review_store_call(service.get_item, item_id)
    if item is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review item not found")
    return item


@review_router.post(
    "/moderation/review/items/{item_id}/decision",
    response_model=ModerationReviewDecisionResponse,
    summary="Record a moderation review decision",
    tags=["moderation-review"],
    dependencies=[Depends(RequirePermission(MODERATION_REVIEW_DECIDE))],
)
async def decide_review_item(
    item_id: str,
    payload: ModerationReviewDecisionRequest,
    principal: CurrentPrincipal,
) -> ModerationReviewDecisionResponse:
    """Record a reviewer decision for a moderation review item."""
    service = get_moderation_review_service()
    try:
        return await _run_review_store_call(
            service.record_decision,
            item_id,
            action=payload.action,
            actor_id=principal.principal_id,
            reason=payload.reason,
            request_actor_id=payload.actor_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Review item not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@review_router.post(
    "/moderation/review/items/{item_id}/undo",
    response_model=ModerationReviewItem,
    summary="Undo a moderation review decision",
    tags=["moderation-review"],
    dependencies=[Depends(RequirePermission(MODERATION_REVIEW_DECIDE))],
)
async def undo_review_decision(
    item_id: str,
    payload: ModerationReviewUndoRequest,
    principal: CurrentPrincipal,
) -> ModerationReviewItem:
    """Undo a recent reviewer decision when its undo token is still valid."""
    service = get_moderation_review_service()
    try:
        return await _run_review_store_call(
            service.undo_decision,
            item_id,
            undo_token=payload.undo_token,
            actor_id=principal.principal_id,
        )
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Undo token not found") from exc
    except ValueError as exc:
        detail = str(exc) or "Undo is no longer available"
        code = status.HTTP_410_GONE if "expired" in detail.casefold() else status.HTTP_409_CONFLICT
        raise HTTPException(status_code=code, detail=detail) from exc


@review_router.post(
    "/moderation/review/bulk-decision",
    response_model=ModerationReviewBulkDecisionResponse,
    summary="Record a bulk moderation review decision",
    tags=["moderation-review"],
    dependencies=[Depends(RequirePermission(MODERATION_REVIEW_BULK_DECIDE))],
)
async def bulk_decide_review_items(
    payload: ModerationReviewBulkDecisionRequest,
    principal: CurrentPrincipal,
) -> ModerationReviewBulkDecisionResponse:
    """Record the same reviewer decision for multiple moderation review items."""
    service = get_moderation_review_service()
    try:
        return await _run_review_store_call(
            service.bulk_decision,
            item_ids=payload.item_ids,
            action=payload.action,
            actor_id=principal.principal_id,
            reason=payload.reason,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@review_router.get(
    "/moderation/review/audit",
    response_model=ModerationReviewAuditResponse,
    summary="List sanitized moderation review audit events",
    tags=["moderation-review"],
    dependencies=[Depends(RequirePermission(MODERATION_AUDIT_READ))],
)
async def list_review_audit(
    item_id: str | None = Query(None),
    decision_id: str | None = Query(None),
    actor_id: str | None = Query(None, alias="actor"),
    action: str | None = Query(None),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    cursor: str | None = Query(None),
) -> ModerationReviewAuditResponse:
    """List sanitized moderation review audit events."""
    service = get_moderation_review_service()
    return await _run_review_store_call(
        service.list_audit,
        item_id=item_id,
        decision_id=decision_id,
        actor_id=actor_id,
        action=action,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
        cursor=cursor,
    )


router.include_router(rules_router)
router.include_router(review_router)
