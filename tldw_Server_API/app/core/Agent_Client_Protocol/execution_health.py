"""ACP execution-health summary aggregation.

The admin endpoint owns routing and auth. This module owns the reusable
classification and aggregation rules for ACP session health reporting.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

REASON_KEYS = ("reason_code", "error_type", "status", "code")
MAX_REASON_NODES = 10_000

SUPPORT_STATES = frozenset({
    "supported",
    "supported_with_caveats",
    "experimental",
    "documented_unverified",
    "unsupported",
})
VERIFICATION_LEVELS = frozenset({
    "documented_only",
    "stub_smoke_tested",
    "live_e2e_tested",
    "sandbox_tested",
    "production_supported",
})

REVIEW_REJECTION_REASONS = frozenset({
    "review_rejected",
    "reviewer_rejected",
    "review_rejected_retry",
    "manual_review_rejected_retry",
    "review_rejected_max_attempts",
    "manual_review_rejected_max_attempts",
})
REVIEW_FAILURE_REASONS = frozenset({
    "reviewer_failed",
    "review_decision_invalid",
})
GOVERNANCE_DENIAL_REASONS = frozenset({
    "governance_denied",
    "permission_denied",
    "policy_denied",
    "tool_denied",
    "denied",
})
SETUP_BLOCKER_REASONS = frozenset({
    "setup_blocked",
    "runner_missing",
    "binary_missing",
    "api_key_missing",
    "adapter_required",
})
RUNNER_SESSION_FAILURE_REASONS = frozenset({
    "agent_run_failed",
    "run_failed",
    "runner_crashed",
    "runner_error",
    "runner_failed",
    "runner_timeout",
    "session_error",
    "session_failed",
    "session_failed_retry",
    "task_failed",
})
STRUCTURED_COMPLETION_FAILURE_REASONS = frozenset({
    "completion_schema_invalid",
    "completion_signal_invalid",
    "completion_signal_missing",
    "completion_validation_failed",
    "invalid_completion_signal",
    "missing_completion_signal",
    "structured_completion_failed",
    "structured_completion_invalid",
})
SANDBOX_RUNTIME_ERROR_REASONS = frozenset({
    "runtime_error",
    "runtime_unavailable",
    "sandbox_launch_failed",
    "sandbox_runner_failed",
    "sandbox_runtime_error",
    "sandbox_start_failed",
    "sandbox_timeout",
    "sandbox_unavailable",
})
RETENTION_REDACTION_ACTION_REASONS = frozenset({
    "audit_retention_purged",
    "redacted_view_requested",
    "redaction_applied",
    "retention_applied",
    "retention_purged",
    "session_retention_purged",
})
MCP_INJECTION_FAILURE_REASONS = frozenset({
    "mcp_injection_failed",
    "mcp_server_unavailable",
    "mcp_tool_registration_failed",
})
SCHEDULER_TRIGGER_FAILURE_REASONS = frozenset({
    "schedule_enqueue_failed",
    "scheduler_trigger_failed",
    "trigger_dispatch_failed",
    "webhook_trigger_failed",
})


def _get_field(obj: Any, key: str, default: Any = None) -> Any:
    """Read a key from dict-like or attribute-like objects."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_utc(value: datetime) -> datetime:
    """Return an aware UTC datetime."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def parse_iso(value: Any) -> datetime | None:
    """Parse an ISO timestamp, including trailing Z, and fail closed on errors."""
    if isinstance(value, datetime):
        return _normalize_utc(value)
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = f"{normalized[:-1]}+00:00"
    try:
        return _normalize_utc(datetime.fromisoformat(normalized))
    except (TypeError, ValueError):
        return None


def session_reference_datetime(record: Any) -> datetime | None:
    """Return the best timestamp available for lookback filtering."""
    parsed_values: list[datetime] = []
    for field_name in ("created_at", "last_activity_at"):
        parsed = parse_iso(_get_field(record, field_name))
        if parsed is not None:
            parsed_values.append(parsed)
    return max(parsed_values) if parsed_values else None


def session_within_range(record: Any, *, since: datetime) -> bool:
    """Return whether a session falls inside the requested lookback window."""
    reference_time = session_reference_datetime(record)
    if reference_time is None:
        return False
    return reference_time >= _normalize_utc(since)


def collect_reason_codes(payload: Any, reasons: set[str]) -> None:
    """Collect normalized failure/status codes from nested ACP payload data."""
    stack: list[Any] = [payload]
    seen_containers: set[int] = set()
    visited = 0

    while stack and visited < MAX_REASON_NODES:
        current = stack.pop()
        visited += 1

        if isinstance(current, dict):
            identity = id(current)
            if identity in seen_containers:
                continue
            seen_containers.add(identity)
            for key in REASON_KEYS:
                value = current.get(key)
                if value is not None:
                    reasons.add(str(value).strip().lower())
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            identity = id(current)
            if identity in seen_containers:
                continue
            seen_containers.add(identity)
            stack.extend(current)


def iter_session_reason_codes(record: Any) -> set[str]:
    """Extract outcome reason codes from non-user session messages."""
    reasons: set[str] = set()
    for message in _get_field(record, "messages", []) or []:
        if not isinstance(message, dict):
            continue
        if str(message.get("role") or "").lower() == "user":
            continue
        collect_reason_codes(message.get("content"), reasons)
        collect_reason_codes(message.get("raw_result"), reasons)
        collect_reason_codes(message.get("raw_data"), reasons)
    return reasons


def default_failure_buckets() -> dict[str, int]:
    """Return zeroed ACP execution-health failure buckets."""
    return {
        "setup_blockers": 0,
        "runner_session_failures": 0,
        "reviewer_rejections": 0,
        "reviewer_failures": 0,
        "governance_denials": 0,
        "structured_completion_failures": 0,
        "sandbox_runtime_errors": 0,
        "retention_redaction_actions": 0,
    }


def _has_runner_session_failure(status_value: str, reasons: set[str]) -> bool:
    if status_value == "error":
        return True
    if reasons & RUNNER_SESSION_FAILURE_REASONS:
        return True
    return any(
        reason.startswith("runner_") and reason not in SETUP_BLOCKER_REASONS
        for reason in reasons
    )


def increment_failure_buckets(
    *,
    status_value: str,
    reasons: set[str],
    buckets: dict[str, int],
) -> None:
    """Increment normalized execution-health buckets for one ACP session."""
    if _has_runner_session_failure(status_value, reasons):
        buckets["runner_session_failures"] += 1
    if reasons & REVIEW_REJECTION_REASONS:
        buckets["reviewer_rejections"] += 1
    if reasons & REVIEW_FAILURE_REASONS:
        buckets["reviewer_failures"] += 1
    if reasons & GOVERNANCE_DENIAL_REASONS:
        buckets["governance_denials"] += 1
    if reasons & SETUP_BLOCKER_REASONS:
        buckets["setup_blockers"] += 1
    if reasons & STRUCTURED_COMPLETION_FAILURE_REASONS:
        buckets["structured_completion_failures"] += 1
    if reasons & SANDBOX_RUNTIME_ERROR_REASONS:
        buckets["sandbox_runtime_errors"] += 1
    if reasons & RETENTION_REDACTION_ACTION_REASONS:
        buckets["retention_redaction_actions"] += 1


def agent_setup_blocked(agent: Any) -> tuple[bool, str | None]:
    """Return whether an agent contributes an admin-visible setup blocker."""
    entrypoint = _get_field(agent, "entrypoint")
    primary_blocker = _get_field(entrypoint, "primary_blocker") if entrypoint else None
    probe_state = _get_field(entrypoint, "probe_state") if entrypoint else None
    is_configured = bool(_get_field(agent, "is_configured", False))
    setup_blocked = (
        (not is_configured)
        or bool(primary_blocker)
        or str(probe_state) == "blocked"
    )
    return setup_blocked, str(primary_blocker) if primary_blocker else None


def coerce_support_state(value: Any) -> str:
    """Validate support state values and fail closed to documented-unverified."""
    support_state = str(value or "documented_unverified")
    if support_state in SUPPORT_STATES:
        return support_state
    return "documented_unverified"


def coerce_verification_level(value: Any) -> str:
    """Validate verification level values and fail closed to documented-only."""
    verification_level = str(value or "documented_only")
    if verification_level in VERIFICATION_LEVELS:
        return verification_level
    return "documented_only"


def _setup_dimension(
    status_value: str,
    *,
    blockers: Iterable[str] = (),
    evidence_count: int = 0,
) -> dict[str, Any]:
    """Build one setup-health dimension summary."""
    return {
        "status": status_value,
        "blockers": sorted({blocker for blocker in blockers if blocker}),
        "evidence_count": max(0, int(evidence_count)),
    }


def _session_has_workspace_evidence(record: Any) -> bool:
    return bool(
        _get_field(record, "workspace_id")
        or _get_field(record, "workspace_group_id")
        or _get_field(record, "scope_snapshot_id")
    )


def _session_has_mcp_evidence(record: Any) -> bool:
    return bool(_get_field(record, "mcp_servers") or [])


def build_setup_health_summary(
    *,
    sessions: list[Any],
    agent_summaries: list[dict[str, Any]],
    buckets: dict[str, int],
    reason_sets: list[set[str]],
) -> dict[str, Any]:
    """Summarize readiness blockers across ACP setup dimensions."""
    agent_blockers = [
        str(agent.get("primary_blocker"))
        for agent in agent_summaries
        if agent.get("setup_blocked") and agent.get("primary_blocker")
    ]
    blocked_agent_count = sum(
        1 for agent in agent_summaries if agent.get("setup_blocked")
    )
    session_setup_blocked = buckets.get("setup_blockers", 0) > blocked_agent_count
    agent_status = (
        "blocked"
        if blocked_agent_count or session_setup_blocked
        else "ready"
        if agent_summaries
        else "unknown"
    )
    if session_setup_blocked:
        agent_blockers.append("session_setup_blockers")

    workspace_evidence = sum(
        1 for session in sessions if _session_has_workspace_evidence(session)
    )
    mcp_evidence = sum(1 for session in sessions if _session_has_mcp_evidence(session))
    mcp_failures = sum(
        1 for reasons in reason_sets if reasons & MCP_INJECTION_FAILURE_REASONS
    )
    scheduler_failures = sum(
        1 for reasons in reason_sets if reasons & SCHEDULER_TRIGGER_FAILURE_REASONS
    )

    return {
        "agent": _setup_dimension(
            agent_status,
            blockers=agent_blockers,
            evidence_count=(
                blocked_agent_count
                if agent_status == "blocked"
                else len(agent_summaries)
            ),
        ),
        "workspace": _setup_dimension(
            "ready" if workspace_evidence else "unknown",
            evidence_count=workspace_evidence,
        ),
        "sandbox_runtime": _setup_dimension(
            "blocked" if buckets.get("sandbox_runtime_errors", 0) else "unknown",
            blockers=(
                ["sandbox_runtime_errors"]
                if buckets.get("sandbox_runtime_errors", 0)
                else ()
            ),
            evidence_count=buckets.get("sandbox_runtime_errors", 0),
        ),
        "mcp_injection": _setup_dimension(
            "blocked" if mcp_failures else "ready" if mcp_evidence else "unknown",
            blockers=["mcp_injection_failures"] if mcp_failures else (),
            evidence_count=mcp_failures or mcp_evidence,
        ),
        "scheduler_trigger_path": _setup_dimension(
            "blocked" if scheduler_failures else "unknown",
            blockers=["scheduler_or_trigger_failures"] if scheduler_failures else (),
            evidence_count=scheduler_failures,
        ),
    }


def summarize_execution_health(
    *,
    sessions: Iterable[Any],
    agents: Iterable[Any],
) -> dict[str, Any]:
    """Aggregate ACP sessions and agent metadata into the admin summary contract."""
    session_list = list(sessions)
    buckets = default_failure_buckets()
    status_counts: dict[str, int] = {}
    reason_sets: list[set[str]] = []

    for record in session_list:
        status_value = str(_get_field(record, "status", "unknown") or "unknown")
        status_counts[status_value] = status_counts.get(status_value, 0) + 1
        normalized_status = status_value.lower()
        reasons = iter_session_reason_codes(record)
        reason_sets.append(reasons)
        increment_failure_buckets(
            status_value=normalized_status,
            reasons=reasons,
            buckets=buckets,
        )

    agent_summaries: list[dict[str, Any]] = []
    support_counts: dict[str, int] = {}
    documented_unverified: list[str] = []

    for agent in agents:
        support_state = coerce_support_state(
            _get_field(agent, "support_state", "documented_unverified")
        )
        verification_level = coerce_verification_level(
            _get_field(agent, "verification_level", "documented_only")
        )
        support_counts[support_state] = support_counts.get(support_state, 0) + 1
        agent_type = str(_get_field(agent, "type", "custom"))
        if support_state == "documented_unverified":
            documented_unverified.append(agent_type)
        setup_blocked, primary_blocker = agent_setup_blocked(agent)
        if setup_blocked:
            buckets["setup_blockers"] += 1
        agent_summaries.append({
            "agent_type": agent_type,
            "name": str(_get_field(agent, "name", "")),
            "is_configured": bool(_get_field(agent, "is_configured", False)),
            "support_state": support_state,
            "verification_level": verification_level,
            "setup_blocked": setup_blocked,
            "primary_blocker": primary_blocker,
        })

    return {
        "sessions": {
            "total": len(session_list),
            "by_status": status_counts,
        },
        "failure_buckets": buckets,
        "agents": agent_summaries,
        "compatibility": {
            "by_support_state": support_counts,
            "documented_unverified_agents": documented_unverified,
            "live_certification_required": bool(documented_unverified),
        },
        "setup_health": build_setup_health_summary(
            sessions=session_list,
            agent_summaries=agent_summaries,
            buckets=buckets,
            reason_sets=reason_sets,
        ),
    }
