"""Structured profile-resolution and policy result primitives."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .decisions import PolicyDecision, evaluate_profile_tool_decision
from .models import MCPProfile

_WORKSPACE_BINDING_KEYS = (
    "workspace_binding",
    "workspace_id",
    "workspace_root",
    "path_scopes",
    "scope",
    "workspace",
)

ProfileResolutionStatus = Literal[
    "resolved",
    "profile_required",
    "profile_not_found",
    "profile_disabled",
    "store_unavailable",
]

EffectivePolicyStatus = Literal[
    "resolved",
    "denied",
    "approval_required",
    "degraded",
]


class ProfileResolutionResult(BaseModel):
    """Machine-readable result for resolving an MCP profile."""

    status: ProfileResolutionStatus
    reason_code: str
    profile: MCPProfile | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    warnings: list[dict[str, Any]] = Field(default_factory=list)


class EffectivePolicy(BaseModel):
    """Caller-owned effective policy document derived from a profile."""

    profile_id: str
    allowed_tools: list[str] = Field(default_factory=list)
    denied_tools: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    denied_capabilities: list[str] = Field(default_factory=list)
    resource_constraints: dict[str, Any] = Field(default_factory=dict)
    approval_policy: dict[str, Any] = Field(default_factory=dict)
    path_scopes: list[dict[str, Any]] = Field(default_factory=list)
    external_server_grants: list[dict[str, Any]] = Field(default_factory=list)
    credential_grants: list[dict[str, Any]] = Field(default_factory=list)


class EffectivePolicyResult(BaseModel):
    """Machine-readable result for deriving an MCP profile policy."""

    status: EffectivePolicyStatus
    reason_code: str
    policy: EffectivePolicy | None = None
    decision: PolicyDecision | None = None
    provenance: dict[str, Any] = Field(default_factory=dict)
    warnings: list[dict[str, Any]] = Field(default_factory=list)


def build_effective_policy_result(
    profile: MCPProfile,
    *,
    host_caps: dict[str, Any] | None = None,
    assignment_binding: dict[str, Any] | None = None,
    tool_name: str | None = None,
    capability: str | None = None,
) -> EffectivePolicyResult:
    """Build a package-local effective policy result for a profile.

    This helper only derives and validates policy metadata. It intentionally does
    not enforce runtime execution or call host-specific registries.
    """
    if profile is None:
        raise ValueError("profile cannot be None")

    policy_document = profile.policy_document
    if policy_document is None:
        raise ValueError("profile.policy_document cannot be None")

    provenance = {
        "profile_id": profile.id,
        "preset_id": profile.preset_id,
        "resolver": "build_effective_policy_result",
    }

    if _requires_workspace_binding(profile) and not _has_workspace_binding(
        profile,
        host_caps=host_caps,
        assignment_binding=assignment_binding,
    ):
        return EffectivePolicyResult(
            status="denied",
            reason_code="workspace_scope_required",
            provenance=provenance,
        )

    allowed_tools = list(policy_document.allowed_tools or [])
    denied_tools = list(policy_document.denied_tools or [])
    capabilities = list(policy_document.capabilities or [])
    denied_capabilities = list(policy_document.denied_capabilities or [])

    decision: PolicyDecision | None = None
    if tool_name is not None:
        decision = evaluate_profile_tool_decision(
            profile,
            tool_name,
            capability=capability,
        )
        if decision.outcome == "deny":
            return EffectivePolicyResult(
                status="denied",
                reason_code=decision.reason_code,
                decision=decision,
                provenance={**provenance, "tool_name": tool_name},
            )
        if decision.outcome == "ask":
            return EffectivePolicyResult(
                status="approval_required",
                reason_code=decision.reason_code,
                decision=decision,
                provenance={**provenance, "tool_name": tool_name},
            )

    if capability is not None:
        if capability in denied_capabilities:
            return EffectivePolicyResult(
                status="denied",
                reason_code="capability_denied",
                decision=decision,
                provenance={**provenance, "capability": capability},
            )
        if not _capability_allowed(capability, capabilities):
            return EffectivePolicyResult(
                status="denied",
                reason_code="capability_not_allowed",
                decision=decision,
                provenance={**provenance, "capability": capability},
            )

    return EffectivePolicyResult(
        status="resolved",
        reason_code="resolved",
        policy=EffectivePolicy(
            profile_id=profile.id,
            allowed_tools=allowed_tools,
            denied_tools=denied_tools,
            capabilities=capabilities,
            denied_capabilities=denied_capabilities,
            resource_constraints=(policy_document.resource_constraints or {}).copy(),
            approval_policy=(profile.approval_policy or {}).copy(),
            path_scopes=[scope.copy() for scope in (profile.path_scopes or [])],
            external_server_grants=[
                grant.copy() for grant in (profile.external_server_grants or [])
            ],
            credential_grants=[
                grant.copy() for grant in (profile.credential_grants or [])
            ],
        ),
        decision=decision,
        provenance=provenance,
    )


def _requires_workspace_binding(profile: MCPProfile) -> bool:
    """Return whether a profile needs workspace binding before policy use."""
    policy_document = profile.policy_document
    constraints = policy_document.resource_constraints or {}
    if constraints.get("requires_workspace_binding") is True:
        return True

    capabilities = policy_document.capabilities or []
    if any(
        any(token in str(capability).lower() for token in ("write", "mutate", "delete"))
        for capability in capabilities
    ):
        return True

    risk_classes = set(policy_document.risk_classes or [])
    return bool(risk_classes & {"mutating", "destructive_filesystem"})


def _has_workspace_binding(
    profile: MCPProfile,
    *,
    host_caps: dict[str, Any] | None,
    assignment_binding: dict[str, Any] | None,
) -> bool:
    """Return whether profile, host, or assignment data binds workspace scope."""
    if _has_workspace_binding_value(profile.path_scopes or []):
        return True
    if _mapping_has_workspace_binding(assignment_binding):
        return True
    return _mapping_has_workspace_binding(host_caps)


def _mapping_has_workspace_binding(mapping: dict[str, Any] | None) -> bool:
    """Return whether a mapping carries a recognized non-empty workspace binding."""
    if not mapping:
        return False
    return any(
        _has_workspace_binding_value(mapping.get(key))
        for key in _WORKSPACE_BINDING_KEYS
    )


def _has_workspace_binding_value(value: Any) -> bool:
    """Return whether a value is meaningful workspace-binding data."""
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(_has_workspace_binding_value(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_workspace_binding_value(item) for item in value)
    return bool(value)


def _capability_allowed(capability: str, capabilities: list[str]) -> bool:
    """Return whether a requested capability is allowed by profile policy."""
    return capability in capabilities
