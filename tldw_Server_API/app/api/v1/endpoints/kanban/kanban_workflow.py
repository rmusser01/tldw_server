# app/api/v1/endpoints/kanban/kanban_workflow.py
"""Kanban workflow-control API endpoints."""

from __future__ import annotations

import asyncio
from typing import Any, Callable, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.kanban_deps import (
    get_kanban_db_for_user,
    kanban_rate_limit,
)
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.schemas.kanban_schemas import (
    WorkflowApprovalDecisionRequest,
    WorkflowClaimRequest,
    WorkflowControlResponse,
    WorkflowEventsListResponse,
    WorkflowForceReassignRequest,
    WorkflowPolicyResponse,
    WorkflowPolicyUpsertRequest,
    WorkflowReleaseRequest,
    WorkflowStaleClaimsListResponse,
    WorkflowStatePatchRequest,
    WorkflowStateResponse,
    WorkflowStatusesListResponse,
    WorkflowTransitionsListResponse,
    WorkflowTransitionRequest,
)
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import get_request_user, User
from tldw_Server_API.app.core.DB_Management.Kanban_DB import (
    ConflictError,
    InputError,
    KanbanDB,
    KanbanDBError,
    NotFoundError,
)

router = APIRouter(tags=["Kanban Workflow"])

_KNOWN_WORKFLOW_ERROR_CODES = {
    "version_conflict",
    "lease_required",
    "lease_mismatch",
    "policy_paused",
    "transition_not_allowed",
    "approval_required",
    "projection_failed",
    "idempotency_conflict",
}

T = TypeVar("T")


def _extract_workflow_error_code(exc: Exception) -> str:
    """Extract a stable workflow conflict code from an exception."""
    explicit_code = getattr(exc, "code", None)
    if isinstance(explicit_code, str) and explicit_code in _KNOWN_WORKFLOW_ERROR_CODES:
        return explicit_code

    message = str(exc).strip()
    if not message:
        return "workflow_error"

    code = message.split("(", 1)[0].strip().split(" ", 1)[0].strip()
    if code in _KNOWN_WORKFLOW_ERROR_CODES:
        return code
    return "workflow_error"


def _workflow_http_error(
    exc: Exception,
    *,
    operation: str,
    context: dict[str, Any] | None = None,
) -> HTTPException:
    """Map domain errors to HTTP errors and log contextual diagnostics."""
    log = logger.bind(operation=operation, **(context or {}))

    if isinstance(exc, NotFoundError):
        log.warning("Workflow request failed with not_found: {}", exc)
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "not_found", "message": str(exc)},
        )
    if isinstance(exc, InputError):
        log.warning("Workflow request failed with invalid_request: {}", exc)
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "invalid_request", "message": str(exc)},
        )
    if isinstance(exc, ConflictError):
        error_code = _extract_workflow_error_code(exc)
        log.warning("Workflow request failed with conflict code {}: {}", error_code, exc)
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": error_code, "message": str(exc)},
        )
    if isinstance(exc, KanbanDBError):
        log.exception("Workflow request failed with Kanban DB error")
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"code": "kanban_db_error", "message": "Workflow database error"},
        )

    log.exception("Workflow request failed with unexpected error")
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail={"code": "internal_error", "message": "An unexpected error occurred"},
    )


async def _run_db_call(
    *,
    operation: str,
    func: Callable[..., T],
    context: dict[str, Any] | None = None,
    **kwargs: Any,
) -> T:
    """Execute synchronous KanbanDB calls in a worker thread with mapped errors."""
    try:
        return await asyncio.to_thread(func, **kwargs)
    except (NotFoundError, InputError, ConflictError, KanbanDBError) as exc:
        raise _workflow_http_error(exc, operation=operation, context=context) from exc


def _require_admin(current_user: User) -> None:
    """Reject non-admin callers for privileged workflow operations."""
    if not getattr(current_user, "is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "forbidden", "message": "Admin privileges required"},
        )


@router.get(
    "/workflow/boards/{board_id}/policy",
    response_model=WorkflowPolicyResponse,
    dependencies=[Depends(kanban_rate_limit("kanban.workflow.policy.get"))],
)
async def get_workflow_policy(
    board_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user),
) -> WorkflowPolicyResponse:
    """Return workflow policy for a board."""
    policy = await _run_db_call(
        operation="workflow.policy.get",
        func=db.get_workflow_policy,
        context={"board_id": board_id},
        board_id=board_id,
    )
    if policy is None:
        raise _workflow_http_error(
            NotFoundError("Workflow policy not found", entity="workflow_policy", entity_id=board_id),
            operation="workflow.policy.get",
            context={"board_id": board_id},
        )
    return WorkflowPolicyResponse(**policy)


@router.put(
    "/workflow/boards/{board_id}/policy",
    response_model=WorkflowPolicyResponse,
    dependencies=[Depends(kanban_rate_limit("kanban.workflow.policy.upsert"))],
)
async def upsert_workflow_policy(
    board_id: int,
    policy_in: WorkflowPolicyUpsertRequest,
    db: KanbanDB = Depends(get_kanban_db_for_user),
) -> WorkflowPolicyResponse:
    """Create or replace workflow policy configuration for a board."""
    upsert_kwargs: dict[str, Any] = {
        "board_id": board_id,
        "statuses": [status.model_dump() for status in policy_in.statuses] if policy_in.statuses is not None else None,
        "transitions": [transition.model_dump() for transition in policy_in.transitions]
        if policy_in.transitions is not None
        else None,
        "is_paused": policy_in.is_paused,
        "is_draining": policy_in.is_draining,
        "default_lease_ttl_sec": policy_in.default_lease_ttl_sec,
        "strict_projection": policy_in.strict_projection,
    }
    if "metadata" in policy_in.model_fields_set:
        upsert_kwargs["metadata"] = policy_in.metadata

    policy = await _run_db_call(
        operation="workflow.policy.upsert",
        func=db.upsert_workflow_policy,
        context={"board_id": board_id},
        **upsert_kwargs,
    )
    return WorkflowPolicyResponse(**policy)


@router.get(
    "/workflow/boards/{board_id}/statuses",
    response_model=WorkflowStatusesListResponse,
    dependencies=[Depends(kanban_rate_limit("kanban.workflow.statuses.list"))],
)
async def list_workflow_statuses(
    board_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user),
) -> WorkflowStatusesListResponse:
    """List active workflow statuses for a board."""
    statuses = await _run_db_call(
        operation="workflow.statuses.list",
        func=db.list_workflow_statuses,
        context={"board_id": board_id},
        board_id=board_id,
    )
    return WorkflowStatusesListResponse(statuses=statuses)


@router.get(
    "/workflow/boards/{board_id}/transitions",
    response_model=WorkflowTransitionsListResponse,
    dependencies=[Depends(kanban_rate_limit("kanban.workflow.transitions.list"))],
)
async def list_workflow_transitions(
    board_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user),
) -> WorkflowTransitionsListResponse:
    """List active workflow transitions for a board."""
    transitions = await _run_db_call(
        operation="workflow.transitions.list",
        func=db.list_workflow_transitions,
        context={"board_id": board_id},
        board_id=board_id,
    )
    return WorkflowTransitionsListResponse(transitions=transitions)


@router.get(
    "/workflow/cards/{card_id}/state",
    response_model=WorkflowStateResponse,
    dependencies=[Depends(kanban_rate_limit("kanban.workflow.task.state.get"))],
)
async def get_card_workflow_state(
    card_id: int,
    db: KanbanDB = Depends(get_kanban_db_for_user),
) -> WorkflowStateResponse:
    """Return workflow runtime state for a card."""
    state = await _run_db_call(
        operation="workflow.task.state.get",
        func=db.get_card_workflow_state,
        context={"card_id": card_id},
        card_id=card_id,
    )
    return WorkflowStateResponse(**state)


@router.patch(
    "/workflow/cards/{card_id}/state",
    response_model=WorkflowStateResponse,
    dependencies=[Depends(kanban_rate_limit("kanban.workflow.task.state.patch"))],
)
async def patch_card_workflow_state(
    card_id: int,
    state_in: WorkflowStatePatchRequest,
    current_user: User = Depends(get_request_user),
    db: KanbanDB = Depends(get_kanban_db_for_user),
) -> WorkflowStateResponse:
    """Apply a privileged direct patch to workflow runtime state."""
    _require_admin(current_user)
    state = await _run_db_call(
        operation="workflow.task.state.patch",
        func=db.patch_card_workflow_state,
        context={"card_id": card_id},
        card_id=card_id,
        workflow_status_key=state_in.workflow_status_key,
        expected_version=state_in.expected_version,
        lease_owner=state_in.lease_owner,
        idempotency_key=state_in.idempotency_key,
        correlation_id=state_in.correlation_id,
        last_actor=state_in.actor,
    )
    return WorkflowStateResponse(**state)


@router.post(
    "/workflow/cards/{card_id}/claim",
    response_model=WorkflowStateResponse,
    dependencies=[Depends(kanban_rate_limit("kanban.workflow.task.claim"))],
)
async def claim_card_workflow(
    card_id: int,
    claim_in: WorkflowClaimRequest,
    db: KanbanDB = Depends(get_kanban_db_for_user),
) -> WorkflowStateResponse:
    """Claim workflow lease ownership for a card."""
    state = await _run_db_call(
        operation="workflow.task.claim",
        func=db.claim_card_workflow,
        context={"card_id": card_id, "owner": claim_in.owner},
        card_id=card_id,
        owner=claim_in.owner,
        lease_ttl_sec=claim_in.lease_ttl_sec,
        idempotency_key=claim_in.idempotency_key,
        correlation_id=claim_in.correlation_id,
    )
    return WorkflowStateResponse(**state)


@router.post(
    "/workflow/cards/{card_id}/release",
    response_model=WorkflowStateResponse,
    dependencies=[Depends(kanban_rate_limit("kanban.workflow.task.release"))],
)
async def release_card_workflow(
    card_id: int,
    release_in: WorkflowReleaseRequest,
    db: KanbanDB = Depends(get_kanban_db_for_user),
) -> WorkflowStateResponse:
    """Release workflow lease ownership for a card."""
    state = await _run_db_call(
        operation="workflow.task.release",
        func=db.release_card_workflow,
        context={"card_id": card_id, "owner": release_in.owner},
        card_id=card_id,
        owner=release_in.owner,
        idempotency_key=release_in.idempotency_key,
        correlation_id=release_in.correlation_id,
    )
    return WorkflowStateResponse(**state)


@router.post(
    "/workflow/cards/{card_id}/transition",
    response_model=WorkflowStateResponse,
    dependencies=[Depends(kanban_rate_limit("kanban.workflow.task.transition"))],
)
async def transition_card_workflow(
    card_id: int,
    transition_in: WorkflowTransitionRequest,
    db: KanbanDB = Depends(get_kanban_db_for_user),
) -> WorkflowStateResponse:
    """Execute a policy-enforced workflow transition for a card."""
    state = await _run_db_call(
        operation="workflow.task.transition",
        func=db.transition_card_workflow,
        context={"card_id": card_id, "to_status_key": transition_in.to_status_key},
        card_id=card_id,
        to_status_key=transition_in.to_status_key,
        actor=transition_in.actor,
        expected_version=transition_in.expected_version,
        idempotency_key=transition_in.idempotency_key,
        correlation_id=transition_in.correlation_id,
        reason=transition_in.reason,
    )
    return WorkflowStateResponse(**state)


@router.post(
    "/workflow/cards/{card_id}/approval",
    response_model=WorkflowStateResponse,
    dependencies=[Depends(kanban_rate_limit("kanban.workflow.task.approval.decide"))],
)
async def decide_card_workflow_approval(
    card_id: int,
    approval_in: WorkflowApprovalDecisionRequest,
    db: KanbanDB = Depends(get_kanban_db_for_user),
) -> WorkflowStateResponse:
    """Record approval decision for a pending transition."""
    state = await _run_db_call(
        operation="workflow.task.approval.decide",
        func=db.decide_card_workflow_approval,
        context={"card_id": card_id, "decision": approval_in.decision},
        card_id=card_id,
        reviewer=approval_in.reviewer,
        decision=approval_in.decision,
        expected_version=approval_in.expected_version,
        idempotency_key=approval_in.idempotency_key,
        correlation_id=approval_in.correlation_id,
        reason=approval_in.reason,
    )
    return WorkflowStateResponse(**state)


@router.get(
    "/workflow/cards/{card_id}/events",
    response_model=WorkflowEventsListResponse,
    dependencies=[Depends(kanban_rate_limit("kanban.workflow.task.events.list"))],
)
async def list_card_workflow_events(
    card_id: int,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: KanbanDB = Depends(get_kanban_db_for_user),
) -> WorkflowEventsListResponse:
    """List append-only workflow events for a card."""
    rows = await _run_db_call(
        operation="workflow.task.events.list",
        func=db.list_card_workflow_events,
        context={"card_id": card_id, "limit": limit, "offset": offset},
        card_id=card_id,
        limit=limit + 1,
        offset=offset,
    )
    has_more = len(rows) > limit
    events = rows[:limit]
    return WorkflowEventsListResponse(
        events=events,
        pagination=build_offset_pagination_meta(
            limit=limit,
            offset=offset,
            total=None,
            count=len(events),
            has_more=has_more,
        ),
    )


@router.get(
    "/workflow/recovery/stale-claims",
    response_model=WorkflowStaleClaimsListResponse,
    dependencies=[Depends(kanban_rate_limit("kanban.workflow.recovery.list_stale_claims"))],
)
async def list_stale_workflow_claims(
    board_id: int | None = Query(None),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: KanbanDB = Depends(get_kanban_db_for_user),
) -> WorkflowStaleClaimsListResponse:
    """List cards with expired active lease claims."""
    rows = await _run_db_call(
        operation="workflow.recovery.list_stale_claims",
        func=db.list_stale_workflow_claims,
        context={"board_id": board_id, "limit": limit, "offset": offset},
        board_id=board_id,
        limit=limit + 1,
        offset=offset,
    )
    has_more = len(rows) > limit
    claims = rows[:limit]
    return WorkflowStaleClaimsListResponse(
        stale_claims=claims,
        pagination=build_offset_pagination_meta(
            limit=limit,
            offset=offset,
            total=None,
            count=len(claims),
            has_more=has_more,
        ),
    )


@router.post(
    "/workflow/recovery/cards/{card_id}/force-reassign",
    response_model=WorkflowStateResponse,
    dependencies=[Depends(kanban_rate_limit("kanban.workflow.recovery.force_reassign"))],
)
async def force_reassign_workflow_claim(
    card_id: int,
    request_in: WorkflowForceReassignRequest,
    current_user: User = Depends(get_request_user),
    db: KanbanDB = Depends(get_kanban_db_for_user),
) -> WorkflowStateResponse:
    """Force-reassign workflow lease ownership for a card (admin only)."""
    _require_admin(current_user)
    state = await _run_db_call(
        operation="workflow.recovery.force_reassign",
        func=db.force_reassign_workflow_claim,
        context={"card_id": card_id, "new_owner": request_in.new_owner},
        card_id=card_id,
        new_owner=request_in.new_owner,
        idempotency_key=request_in.idempotency_key,
        correlation_id=request_in.correlation_id,
        reason=request_in.reason,
    )
    return WorkflowStateResponse(**state)


@router.post(
    "/workflow/control/boards/{board_id}/pause",
    response_model=WorkflowControlResponse,
    dependencies=[Depends(kanban_rate_limit("kanban.workflow.control.pause"))],
)
async def pause_workflow_policy(
    board_id: int,
    current_user: User = Depends(get_request_user),
    db: KanbanDB = Depends(get_kanban_db_for_user),
) -> WorkflowControlResponse:
    """Pause workflow transitions for a board (admin only)."""
    _require_admin(current_user)
    policy = await _run_db_call(
        operation="workflow.control.pause",
        func=db.update_workflow_policy_flags,
        context={"board_id": board_id},
        board_id=board_id,
        is_paused=True,
    )
    return WorkflowControlResponse(success=True, policy=WorkflowPolicyResponse(**policy))


@router.post(
    "/workflow/control/boards/{board_id}/resume",
    response_model=WorkflowControlResponse,
    dependencies=[Depends(kanban_rate_limit("kanban.workflow.control.resume"))],
)
async def resume_workflow_policy(
    board_id: int,
    current_user: User = Depends(get_request_user),
    db: KanbanDB = Depends(get_kanban_db_for_user),
) -> WorkflowControlResponse:
    """Resume workflow transitions for a board (admin only)."""
    _require_admin(current_user)
    policy = await _run_db_call(
        operation="workflow.control.resume",
        func=db.update_workflow_policy_flags,
        context={"board_id": board_id},
        board_id=board_id,
        is_paused=False,
    )
    return WorkflowControlResponse(success=True, policy=WorkflowPolicyResponse(**policy))


@router.post(
    "/workflow/control/boards/{board_id}/drain",
    response_model=WorkflowControlResponse,
    dependencies=[Depends(kanban_rate_limit("kanban.workflow.control.drain"))],
)
async def drain_workflow_policy(
    board_id: int,
    current_user: User = Depends(get_request_user),
    db: KanbanDB = Depends(get_kanban_db_for_user),
) -> WorkflowControlResponse:
    """Enable workflow draining mode for a board (admin only)."""
    _require_admin(current_user)
    policy = await _run_db_call(
        operation="workflow.control.drain",
        func=db.update_workflow_policy_flags,
        context={"board_id": board_id},
        board_id=board_id,
        is_draining=True,
    )
    return WorkflowControlResponse(success=True, policy=WorkflowPolicyResponse(**policy))
