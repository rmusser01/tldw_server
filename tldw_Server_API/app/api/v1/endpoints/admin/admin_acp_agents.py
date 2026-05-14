"""Admin endpoints for ACP session management, agent configuration, and permission policies.

Provides cross-user visibility into agent sessions, CRUD for custom agent
configurations, and tool permission policy management.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, cast, get_args

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_auth_principal
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.schemas.agent_client_protocol import (
    ACPAgentConfigCreate,
    ACPAgentConfigListResponse,
    ACPAgentConfigResponse,
    ACPAgentMetrics,
    ACPAgentMetricsListResponse,
    ACPAgentUsageItem,
    ACPAgentUsageResponse,
    ACPExecutionHealthAgentSummary,
    ACPExecutionHealthCompatibilitySummary,
    ACPExecutionHealthFailureBuckets,
    ACPExecutionHealthRedactionSummary,
    ACPExecutionHealthRetentionSummary,
    ACPExecutionHealthSessionSummary,
    ACPExecutionHealthSummaryResponse,
    ACPSupportState,
    ACPVerificationLevel,
    ACPPermissionPolicyCreate,
    ACPPermissionPolicyListResponse,
    ACPPermissionPolicyResponse,
    ACPSessionBudgetRequest,
    ACPSessionBudgetResponse,
    ACPSessionInfo,
    ACPSessionListResponse,
    ACPSessionUsageResponse,
    ACPTokenUsage,
)
from tldw_Server_API.app.core.Usage.pricing_catalog import compute_token_cost
from tldw_Server_API.app.services.admin_acp_sessions_service import get_acp_session_store

router = APIRouter(tags=["admin-acp"])

_NONCRITICAL = (
    AssertionError,
    AttributeError,
    ConnectionError,
    ImportError,
    KeyError,
    RuntimeError,
    TypeError,
    ValueError,
)

_REVIEW_REJECTION_REASONS = frozenset({
    "review_rejected",
    "reviewer_rejected",
    "review_rejected_retry",
    "manual_review_rejected_retry",
    "review_rejected_max_attempts",
    "manual_review_rejected_max_attempts",
})
_REVIEW_FAILURE_REASONS = frozenset({
    "reviewer_failed",
    "review_decision_invalid",
})
_GOVERNANCE_DENIAL_REASONS = frozenset({
    "governance_denied",
    "permission_denied",
    "policy_denied",
    "tool_denied",
    "denied",
})
_SETUP_BLOCKER_REASONS = frozenset({
    "setup_blocked",
    "runner_missing",
    "binary_missing",
    "api_key_missing",
    "adapter_required",
})
_SUPPORT_STATES = frozenset(get_args(ACPSupportState))
_VERIFICATION_LEVELS = frozenset(get_args(ACPVerificationLevel))


async def _get_available_agents():
    """Return configured ACP agents through the public ACP endpoint helper."""
    from tldw_Server_API.app.api.v1.endpoints.agent_client_protocol import (
        _get_available_agents as _acp_get_available_agents,
    )

    return await _acp_get_available_agents()


def _now_iso() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    """Parse an ISO timestamp and return None for absent or malformed values."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _session_within_range(record: Any, *, since: datetime) -> bool:
    """Return whether a session record falls inside the requested lookback."""
    created = _parse_iso(getattr(record, "created_at", None))
    if created is None:
        return True
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    return created >= since


def _collect_reason_codes(payload: Any, reasons: set[str]) -> None:
    """Collect normalized failure/status codes from nested ACP payload data."""
    if isinstance(payload, dict):
        for key in ("reason_code", "error_type", "status", "code"):
            value = payload.get(key)
            if value is not None:
                reasons.add(str(value).lower())
        for value in payload.values():
            _collect_reason_codes(value, reasons)
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            _collect_reason_codes(item, reasons)


def _iter_session_reason_codes(record: Any) -> set[str]:
    """Extract outcome reason codes from non-user session messages."""
    reasons: set[str] = set()
    for message in getattr(record, "messages", []) or []:
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").lower() == "user":
            continue
        _collect_reason_codes(message.get("content"), reasons)
        _collect_reason_codes(message.get("raw_result"), reasons)
    return reasons


def _increment_failure_buckets(record: Any, buckets: dict[str, int]) -> None:
    """Increment normalized execution-health buckets for one ACP session."""
    status_value = str(getattr(record, "status", "") or "").lower()
    reasons = _iter_session_reason_codes(record)
    if status_value == "error" or any("runner" in reason or "session_failed" in reason for reason in reasons):
        buckets["runner_session_failures"] += 1
    if reasons & _REVIEW_REJECTION_REASONS:
        buckets["reviewer_rejections"] += 1
    if reasons & _REVIEW_FAILURE_REASONS:
        buckets["reviewer_failures"] += 1
    if reasons & _GOVERNANCE_DENIAL_REASONS:
        buckets["governance_denials"] += 1
    if reasons & _SETUP_BLOCKER_REASONS:
        buckets["setup_blockers"] += 1


def _agent_setup_blocked(agent: Any) -> tuple[bool, str | None]:
    """Return whether an agent contributes an admin-visible setup blocker."""
    entrypoint = getattr(agent, "entrypoint", None)
    primary_blocker = getattr(entrypoint, "primary_blocker", None)
    probe_state = getattr(entrypoint, "probe_state", None)
    is_configured = bool(getattr(agent, "is_configured", False))
    setup_blocked = (not is_configured) or bool(primary_blocker) or str(probe_state) == "blocked"
    return setup_blocked, str(primary_blocker) if primary_blocker else None


def _coerce_support_state(value: Any) -> ACPSupportState:
    """Validate support state values and fail closed to documented-unverified."""
    support_state = str(value or "documented_unverified")
    if support_state in _SUPPORT_STATES:
        return cast(ACPSupportState, support_state)
    return "documented_unverified"


def _coerce_verification_level(value: Any) -> ACPVerificationLevel:
    """Validate verification level values and fail closed to documented-only."""
    verification_level = str(value or "documented_only")
    if verification_level in _VERIFICATION_LEVELS:
        return cast(ACPVerificationLevel, verification_level)
    return "documented_only"


def _retention_summary() -> ACPExecutionHealthRetentionSummary:
    """Return the configured ACP retention posture, falling back to defaults."""
    try:
        from tldw_Server_API.app.core.Agent_Client_Protocol.config import load_acp_sandbox_config

        config = load_acp_sandbox_config()
        return ACPExecutionHealthRetentionSummary(
            session_retention_days=int(getattr(config, "session_retention_days", 30)),
            audit_retention_days=int(getattr(config, "audit_retention_days", 30)),
        )
    except _NONCRITICAL:
        return ACPExecutionHealthRetentionSummary()


# ---------------------------------------------------------------------------
# ACP Session Admin Endpoints
# ---------------------------------------------------------------------------

@router.get("/acp/sessions", response_model=ACPSessionListResponse)
async def admin_list_acp_sessions(
    user_id: int | None = Query(default=None, description="Filter by user ID"),
    status_filter: str | None = Query(default=None, alias="status"),
    agent_type: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> ACPSessionListResponse:
    """Admin cross-user view of all ACP sessions."""
    store = await get_acp_session_store()

    has_ws_fn = _get_ws_checker()
    records, total = await store.list_sessions(
        user_id=user_id,
        status=status_filter,
        agent_type=agent_type,
        limit=limit,
        offset=offset,
    )
    sessions = [
        ACPSessionInfo(**rec.to_info_dict(
            has_websocket=has_ws_fn(rec.session_id),
        ))
        for rec in records
    ]
    return ACPSessionListResponse(
        sessions=sessions,
        total=total,
        pagination=build_offset_pagination_meta(
            total=total,
            limit=limit,
            offset=offset,
            count=len(sessions),
        ),
    )


@router.get("/acp/sessions/{session_id}/usage", response_model=ACPSessionUsageResponse)
async def admin_acp_session_usage(session_id: str) -> ACPSessionUsageResponse:
    """Get token usage for any ACP session (admin view)."""
    store = await get_acp_session_store()
    rec = await store.get_session(session_id)
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session_not_found")
    return ACPSessionUsageResponse(
        session_id=rec.session_id,
        user_id=rec.user_id,
        agent_type=rec.agent_type,
        usage=ACPTokenUsage(**rec.usage.to_dict()),
        message_count=rec.message_count,
        created_at=rec.created_at,
        last_activity_at=rec.last_activity_at,
        model=rec.model,
        estimated_cost_usd=compute_token_cost(
            model=rec.model,
            prompt_tokens=rec.usage.prompt_tokens,
            completion_tokens=rec.usage.completion_tokens,
        ),
    )


@router.post("/acp/sessions/{session_id}/close")
async def admin_close_acp_session(session_id: str) -> dict[str, str]:
    """Force-close an ACP session (admin action)."""
    store = await get_acp_session_store()
    rec = await store.get_session(session_id)
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session_not_found")
    # Try to close on the runner client as well
    try:
        from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import get_runner_client
        client = await get_runner_client()
        await client.close_session(session_id)
    except _NONCRITICAL:
        pass
    await store.close_session(session_id)
    return {"status": "ok", "session_id": session_id}


@router.patch("/acp/sessions/{session_id}/budget", response_model=ACPSessionBudgetResponse)
async def admin_set_session_budget(
    session_id: str,
    body: ACPSessionBudgetRequest,
) -> ACPSessionBudgetResponse:
    """Set or update the token budget for an ACP session.

    Setting token_budget to null removes the budget (unlimited).
    When auto_terminate_at_budget is True, the session will automatically
    close once total_tokens >= token_budget.
    """
    store = await get_acp_session_store()
    rec = await store.update_session_budget(
        session_id,
        token_budget=body.token_budget,
        auto_terminate_at_budget=body.auto_terminate_at_budget,
    )
    if not rec:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="session_not_found")

    budget_remaining = None
    if rec.token_budget is not None:
        budget_remaining = max(0, rec.token_budget - rec.usage.total_tokens)

    return ACPSessionBudgetResponse(
        session_id=rec.session_id,
        token_budget=rec.token_budget,
        auto_terminate_at_budget=rec.auto_terminate_at_budget,
        budget_exhausted=rec.budget_exhausted,
        total_tokens=rec.usage.total_tokens,
        budget_remaining=budget_remaining,
    )


# ---------------------------------------------------------------------------
# Agent Usage Stats
# ---------------------------------------------------------------------------

@router.get("/acp/agents/usage", response_model=ACPAgentUsageResponse)
async def admin_get_agent_usage(
    range_days: int = Query(7, ge=1, le=90),
    _: object = Depends(get_auth_principal),
    __: None = Depends(check_rate_limit),
) -> ACPAgentUsageResponse:
    """Aggregated per-agent token usage from ACP sessions."""
    store = await get_acp_session_store()
    rows = await store.get_agent_usage_stats(range_days=range_days)
    return ACPAgentUsageResponse(
        agents=[ACPAgentUsageItem(**r) for r in rows],
        range_days=range_days,
    )


# ---------------------------------------------------------------------------
# Agent Configuration CRUD
# ---------------------------------------------------------------------------

@router.get("/acp/agents", response_model=ACPAgentConfigListResponse)
async def admin_list_agent_configs(
    org_id: int | None = Query(default=None),
    team_id: int | None = Query(default=None),
    enabled_only: bool = Query(default=False),
) -> ACPAgentConfigListResponse:
    """List all custom agent configurations."""
    store = await get_acp_session_store()
    configs = await store.list_agent_configs(org_id=org_id, team_id=team_id, enabled_only=enabled_only)
    return ACPAgentConfigListResponse(
        agents=[ACPAgentConfigResponse(**c.to_dict()) for c in configs],
        total=len(configs),
    )


@router.get("/acp/agents/metrics", response_model=ACPAgentMetricsListResponse)
async def get_acp_agent_metrics() -> ACPAgentMetricsListResponse:
    """Aggregate runtime metrics per ACP agent type.

    Returns per-agent totals for sessions, active sessions, tokens,
    messages, and the timestamp of the most recent activity.
    """
    store = await get_acp_session_store()
    metrics = await store.get_agent_metrics()
    return ACPAgentMetricsListResponse(
        items=[ACPAgentMetrics(**m) for m in metrics],
    )


@router.get("/acp/execution-health/summary", response_model=ACPExecutionHealthSummaryResponse)
async def get_acp_execution_health_summary(
    range_days: int = Query(30, ge=1, le=180),
    _: object = Depends(get_auth_principal),
    __: None = Depends(check_rate_limit),
) -> ACPExecutionHealthSummaryResponse:
    """Return a compact ACP execution-health rollup for admin reporting."""
    store = await get_acp_session_store()
    records, _total = await store.list_sessions(limit=1000, offset=0)
    since = datetime.now(timezone.utc) - timedelta(days=range_days)

    session_records = [
        await store.get_session(record.session_id)
        for record in records
        if _session_within_range(record, since=since)
    ]
    sessions = [record for record in session_records if record is not None]

    status_counts: dict[str, int] = {}
    buckets = {
        "setup_blockers": 0,
        "runner_session_failures": 0,
        "reviewer_rejections": 0,
        "reviewer_failures": 0,
        "governance_denials": 0,
    }
    for record in sessions:
        status_value = str(getattr(record, "status", "unknown") or "unknown")
        status_counts[status_value] = status_counts.get(status_value, 0) + 1
        _increment_failure_buckets(record, buckets)

    try:
        agents, _default_agent = await _get_available_agents()
    except _NONCRITICAL as exc:
        logger.warning("Unable to include ACP agent compatibility in execution-health summary: {}", exc)
        agents = []

    agent_summaries: list[ACPExecutionHealthAgentSummary] = []
    support_counts: dict[str, int] = {}
    documented_unverified: list[str] = []
    for agent in agents:
        support_state = _coerce_support_state(getattr(agent, "support_state", "documented_unverified"))
        verification_level = _coerce_verification_level(getattr(agent, "verification_level", "documented_only"))
        support_counts[support_state] = support_counts.get(support_state, 0) + 1
        if support_state == "documented_unverified":
            documented_unverified.append(str(getattr(agent, "type", "")))
        setup_blocked, primary_blocker = _agent_setup_blocked(agent)
        if setup_blocked:
            buckets["setup_blockers"] += 1
        agent_summaries.append(
            ACPExecutionHealthAgentSummary(
                agent_type=str(getattr(agent, "type", "custom")),
                name=str(getattr(agent, "name", "")),
                is_configured=bool(getattr(agent, "is_configured", False)),
                support_state=support_state,
                verification_level=verification_level,
                setup_blocked=setup_blocked,
                primary_blocker=primary_blocker,
            )
        )

    return ACPExecutionHealthSummaryResponse(
        timestamp=_now_iso(),
        range_days=range_days,
        sessions=ACPExecutionHealthSessionSummary(
            total=len(sessions),
            by_status=status_counts,
        ),
        failure_buckets=ACPExecutionHealthFailureBuckets(**buckets),
        agents=agent_summaries,
        compatibility=ACPExecutionHealthCompatibilitySummary(
            by_support_state=support_counts,
            documented_unverified_agents=documented_unverified,
            live_certification_required=bool(documented_unverified),
        ),
        retention=_retention_summary(),
        redaction=ACPExecutionHealthRedactionSummary(),
    )


@router.post("/acp/agents", response_model=ACPAgentConfigResponse, status_code=status.HTTP_201_CREATED)
async def admin_create_agent_config(payload: ACPAgentConfigCreate) -> ACPAgentConfigResponse:
    """Create a new custom agent configuration."""
    store = await get_acp_session_store()
    config = await store.create_agent_config(payload.model_dump())
    return ACPAgentConfigResponse(**config.to_dict())


@router.get("/acp/agents/{config_id}", response_model=ACPAgentConfigResponse)
async def admin_get_agent_config(config_id: int) -> ACPAgentConfigResponse:
    """Get a specific agent configuration."""
    store = await get_acp_session_store()
    config = await store.get_agent_config(config_id)
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="agent_config_not_found")
    return ACPAgentConfigResponse(**config.to_dict())


@router.put("/acp/agents/{config_id}", response_model=ACPAgentConfigResponse)
async def admin_update_agent_config(config_id: int, payload: ACPAgentConfigCreate) -> ACPAgentConfigResponse:
    """Update an agent configuration."""
    store = await get_acp_session_store()
    config = await store.update_agent_config(config_id, payload.model_dump())
    if not config:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="agent_config_not_found")
    return ACPAgentConfigResponse(**config.to_dict())


@router.delete("/acp/agents/{config_id}")
async def admin_delete_agent_config(config_id: int) -> dict[str, str]:
    """Delete an agent configuration."""
    store = await get_acp_session_store()
    deleted = await store.delete_agent_config(config_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="agent_config_not_found")
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Permission Policy CRUD
# ---------------------------------------------------------------------------

@router.get("/acp/permission-policies", response_model=ACPPermissionPolicyListResponse)
async def admin_list_permission_policies(
    org_id: int | None = Query(default=None),
    team_id: int | None = Query(default=None),
) -> ACPPermissionPolicyListResponse:
    """List tool permission policies."""
    store = await get_acp_session_store()
    policies = await store.list_permission_policies(org_id=org_id, team_id=team_id)
    return ACPPermissionPolicyListResponse(
        policies=[
            ACPPermissionPolicyResponse(
                id=p.id,
                name=p.name,
                description=p.description,
                rules=[{"tool_pattern": r.tool_pattern, "tier": r.tier} for r in p.rules],
                org_id=p.org_id,
                team_id=p.team_id,
                priority=p.priority,
                created_at=p.created_at,
                updated_at=p.updated_at,
            )
            for p in policies
        ],
        total=len(policies),
    )


@router.post("/acp/permission-policies", response_model=ACPPermissionPolicyResponse, status_code=status.HTTP_201_CREATED)
async def admin_create_permission_policy(payload: ACPPermissionPolicyCreate) -> ACPPermissionPolicyResponse:
    """Create a new tool permission policy."""
    store = await get_acp_session_store()
    policy = await store.create_permission_policy(payload.model_dump())
    return ACPPermissionPolicyResponse(
        id=policy.id,
        name=policy.name,
        description=policy.description,
        rules=[{"tool_pattern": r.tool_pattern, "tier": r.tier} for r in policy.rules],
        org_id=policy.org_id,
        team_id=policy.team_id,
        priority=policy.priority,
        created_at=policy.created_at,
        updated_at=policy.updated_at,
    )


@router.put("/acp/permission-policies/{policy_id}", response_model=ACPPermissionPolicyResponse)
async def admin_update_permission_policy(policy_id: int, payload: ACPPermissionPolicyCreate) -> ACPPermissionPolicyResponse:
    """Update a permission policy."""
    store = await get_acp_session_store()
    policy = await store.update_permission_policy(policy_id, payload.model_dump())
    if not policy:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="policy_not_found")
    return ACPPermissionPolicyResponse(
        id=policy.id,
        name=policy.name,
        description=policy.description,
        rules=[{"tool_pattern": r.tool_pattern, "tier": r.tier} for r in policy.rules],
        org_id=policy.org_id,
        team_id=policy.team_id,
        priority=policy.priority,
        created_at=policy.created_at,
        updated_at=policy.updated_at,
    )


@router.delete("/acp/permission-policies/{policy_id}")
async def admin_delete_permission_policy(policy_id: int) -> dict[str, str]:
    """Delete a permission policy."""
    store = await get_acp_session_store()
    deleted = await store.delete_permission_policy(policy_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="policy_not_found")
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_ws_checker():
    """Return a function to check WebSocket connections, best-effort."""
    try:
        from tldw_Server_API.app.core.Agent_Client_Protocol.runner_client import (
            _runner_client,
            _sandbox_client,
        )
        active_client = _sandbox_client or _runner_client
        if active_client and hasattr(active_client, "has_websocket_connections"):
            return active_client.has_websocket_connections
    except _NONCRITICAL:
        pass
    return lambda _sid: False
