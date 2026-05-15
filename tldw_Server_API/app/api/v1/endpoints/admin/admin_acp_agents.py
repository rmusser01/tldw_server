"""Admin endpoints for ACP session management, agent configuration, and permission policies.

Provides cross-user visibility into agent sessions, CRUD for custom agent
configurations, and tool permission policy management.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Query, status
from loguru import logger

from tldw_Server_API.app.api.v1.API_Deps.auth_deps import check_rate_limit, get_auth_principal
from tldw_Server_API.app.api.v1.endpoints._pagination_utils import build_offset_pagination_meta
from tldw_Server_API.app.api.v1.schemas.agent_client_protocol import (
    ACPAgentConfigCreate,
    ACPAgentConfigListResponse,
    ACPAgentConfigResponse,
    ACPAgentInfo,
    ACPAgentMetrics,
    ACPAgentMetricsListResponse,
    ACPAgentUsageItem,
    ACPAgentUsageResponse,
    ACPExecutionHealthFailureBuckets,
    ACPExecutionHealthRedactionSummary,
    ACPExecutionHealthRetentionSummary,
    ACPExecutionHealthSessionSummary,
    ACPExecutionHealthSetupSummary,
    ACPExecutionHealthSummaryResponse,
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
from tldw_Server_API.app.core.Agent_Client_Protocol.execution_health import (
    summarize_execution_health,
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


async def _get_available_agents() -> tuple[list[ACPAgentInfo], str | None]:
    """Return configured ACP agents through the public ACP endpoint helper."""
    from tldw_Server_API.app.api.v1.endpoints.agent_client_protocol import (
        _get_available_agents as _acp_get_available_agents,
    )

    return await _acp_get_available_agents()


def _now_iso() -> str:
    """Return the current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _retention_summary() -> ACPExecutionHealthRetentionSummary:
    """Return the configured ACP retention posture, falling back to defaults."""
    try:
        from tldw_Server_API.app.core.Agent_Client_Protocol.config import load_acp_sandbox_config

        config = load_acp_sandbox_config()
        return ACPExecutionHealthRetentionSummary(
            session_retention_days=int(getattr(config, "session_retention_days", 30)),
            audit_retention_days=int(getattr(config, "audit_retention_days", 30)),
        )
    except _NONCRITICAL as exc:
        logger.warning(
            "Unable to load ACP retention config for execution-health summary; using defaults: {}",
            type(exc).__name__,
        )
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
    since = datetime.now(timezone.utc) - timedelta(days=range_days)
    sessions = await store.list_sessions_with_messages_since(since=since, page_size=1000)

    try:
        agents, _default_agent = await _get_available_agents()
    except _NONCRITICAL as exc:
        logger.warning(
            "Unable to include ACP agent compatibility in execution-health summary: {}",
            exc,
        )
        agents = []

    summary = summarize_execution_health(sessions=sessions, agents=agents)

    return ACPExecutionHealthSummaryResponse(
        timestamp=_now_iso(),
        range_days=range_days,
        sessions=ACPExecutionHealthSessionSummary(**summary["sessions"]),
        failure_buckets=ACPExecutionHealthFailureBuckets(**summary["failure_buckets"]),
        setup_health=ACPExecutionHealthSetupSummary(**summary["setup_health"]),
        agents=summary["agents"],
        compatibility=summary["compatibility"],
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
