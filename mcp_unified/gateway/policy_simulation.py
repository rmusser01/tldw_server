"""Read-only policy simulation harness for the standalone MCP gateway.

The harness mirrors the runtime tool-call decision pipeline by reusing the
same legacy policy resolution, permission-rule enforcement, and TTL grant
merge code the gateway executes, so simulated verdicts cannot drift from
runtime behavior. It never writes audit events or consumes leases.
"""

from __future__ import annotations

from typing import Any

from mcp_unified.policy_grants import PolicyGrantStore
from mcp_unified.profiles.models import MCPProfile
from mcp_unified.profiles.permission_rules import (
    compile_permission_rules,
    evaluate_permission_rule_decision,
)
from mcp_unified.profiles.resolution import build_effective_policy_result
from mcp_unified.profiles.subjects import (
    PermissionSubjectLimitError,
    extract_permission_rule_subjects,
)

from .profile_runtime import (
    _enforce_permission_rules_for_tool_call,
    _policy_with_ttl_path_grants,
)
from .runtime import GatewayPolicyDenied


def simulate_tool_call_policy(
    profile: MCPProfile,
    tool_name: str,
    arguments: dict[str, Any],
    *,
    capability: str | None = None,
    policy_grant_store: PolicyGrantStore | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Simulate one gateway tool-call policy decision without executing it."""

    result: dict[str, Any] = {
        "profile_id": profile.id,
        "tool_name": tool_name,
        "legacy_policy": None,
        "subjects": [],
        "approval_grant_markers": [],
        "path_scopes": [],
        "denial": None,
        "overall": {"status": "allowed", "reason_code": None},
    }

    legacy_result = build_effective_policy_result(
        profile,
        tool_name=tool_name,
        capability=capability,
    )
    result["legacy_policy"] = {
        "status": legacy_result.status,
        "reason_code": legacy_result.reason_code,
    }
    if legacy_result.status != "resolved":
        result["overall"] = {
            "status": "denied",
            "reason_code": legacy_result.reason_code,
        }
        return result

    try:
        rules = (
            tuple(compile_permission_rules(profile.policy_document))
            if profile.policy_document is not None
            else ()
        )
    except ValueError:
        result["overall"] = {
            "status": "denied",
            "reason_code": "invalid_permission_rules",
        }
        return result

    result["subjects"] = _subject_details(tool_name, arguments, rules)

    try:
        result["approval_grant_markers"] = _enforce_permission_rules_for_tool_call(
            profile,
            tool_name,
            arguments,
            rules,
            policy_grant_store=policy_grant_store,
            session_id=session_id,
        )
    except GatewayPolicyDenied as exc:
        result["denial"] = exc.to_error_data()
        result["overall"] = {"status": exc.status, "reason_code": exc.reason_code}

    merged_policy = _policy_with_ttl_path_grants(
        legacy_result.policy,
        policy_grant_store,
        profile_id=profile.id,
        session_id=session_id,
    )
    if merged_policy is not None:
        result["path_scopes"] = [dict(scope) for scope in merged_policy.path_scopes or []]
    return result


def _subject_details(
    tool_name: str,
    arguments: dict[str, Any],
    rules: tuple[Any, ...],
) -> list[dict[str, Any]]:
    """Return per-subject decision detail for one simulated tool call."""

    if not rules:
        return []
    try:
        subjects = extract_permission_rule_subjects(tool_name, arguments)
    except PermissionSubjectLimitError:
        return []

    details: list[dict[str, Any]] = []
    for subject_type, value, argv in subjects:
        try:
            decision = evaluate_permission_rule_decision(
                rules,
                subject_type=subject_type,
                value=value,
                argv=argv,
            )
        except ValueError:
            details.append(
                {
                    "subject_type": subject_type,
                    "value": value,
                    "outcome": "error",
                    "reason_code": "invalid_permission_rule_subject",
                    "matched_rules": [],
                }
            )
            continue
        effective_outcome = decision.outcome if decision.matched_rules else "allow"
        details.append(
            {
                "subject_type": subject_type,
                "value": value,
                "outcome": effective_outcome,
                "reason_code": decision.reason_code,
                "matched_rules": [
                    matched_rule.model_dump(mode="json", exclude_none=True)
                    for matched_rule in decision.matched_rules
                ],
            }
        )
    return details


__all__ = ["simulate_tool_call_policy"]
