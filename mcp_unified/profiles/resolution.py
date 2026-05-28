"""Structured profile-resolution and policy result primitives."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .models import MCPProfile

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
    policy_document = profile.policy_document
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

    allowed_tools = list(policy_document.allowed_tools)
    denied_tools = list(policy_document.denied_tools)
    capabilities = list(policy_document.capabilities)
    denied_capabilities = list(policy_document.denied_capabilities)

    if tool_name is not None:
        if tool_name in denied_tools:
            return EffectivePolicyResult(
                status="denied",
                reason_code="tool_denied",
                provenance={**provenance, "tool_name": tool_name},
            )
        if not _tool_allowed(tool_name, allowed_tools, capability):
            return EffectivePolicyResult(
                status="denied",
                reason_code="tool_not_allowed",
                provenance={**provenance, "tool_name": tool_name},
            )

    if capability is not None:
        if capability in denied_capabilities:
            return EffectivePolicyResult(
                status="denied",
                reason_code="capability_denied",
                provenance={**provenance, "capability": capability},
            )
        if not _capability_allowed(capability, capabilities):
            return EffectivePolicyResult(
                status="denied",
                reason_code="capability_not_allowed",
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
            resource_constraints=policy_document.resource_constraints.copy(),
            approval_policy=profile.approval_policy.copy(),
            path_scopes=[scope.copy() for scope in profile.path_scopes],
            external_server_grants=[
                grant.copy() for grant in profile.external_server_grants
            ],
            credential_grants=[grant.copy() for grant in profile.credential_grants],
        ),
        provenance=provenance,
    )


def _requires_workspace_binding(profile: MCPProfile) -> bool:
    """Return whether a profile needs workspace binding before policy use."""
    policy_document = profile.policy_document
    constraints = policy_document.resource_constraints
    if constraints.get("requires_workspace_binding") is True:
        return True

    capability_text = " ".join(policy_document.capabilities).lower()
    if any(token in capability_text for token in ("write", "mutate", "delete")):
        return True

    risk_classes = set(policy_document.risk_classes)
    return bool(risk_classes & {"mutating", "destructive_filesystem"})


def _has_workspace_binding(
    profile: MCPProfile,
    *,
    host_caps: dict[str, Any] | None,
    assignment_binding: dict[str, Any] | None,
) -> bool:
    """Return whether profile, host, or assignment data binds workspace scope."""
    if profile.path_scopes:
        return True
    if assignment_binding:
        return True
    if not host_caps:
        return False

    binding_keys = (
        "workspace_binding",
        "workspace_id",
        "workspace_root",
        "path_scopes",
    )
    return any(bool(host_caps.get(key)) for key in binding_keys)


def _tool_allowed(
    tool_name: str,
    allowed_tools: list[str],
    capability: str | None,
) -> bool:
    """Return whether a requested tool is allowed by explicit profile policy."""
    if allowed_tools:
        return tool_name in allowed_tools
    return capability is not None


def _capability_allowed(capability: str, capabilities: list[str]) -> bool:
    """Return whether a requested capability is allowed by profile policy."""
    return capability in capabilities
