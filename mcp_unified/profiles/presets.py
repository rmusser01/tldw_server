"""Built-in MCP profile presets for front-end modes."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from .models import MCPProfile, ProfilePolicy

PRESET_VERSION = "2026.05.27"

_PROCESS_CAPABILITIES = {
    "command.run",
    "process.execute",
    "shell.execute",
}
_DESTRUCTIVE_FILESYSTEM_CAPABILITIES = {
    "filesystem.delete",
    "filesystem.destructive",
    "filesystem.write_unscoped",
}
_EXTERNAL_NETWORK_CAPABILITIES = {
    "external.network",
}
_HIGH_RISK_RISK_CLASSES = {
    "credential_use",
    "destructive_filesystem",
    "external_network",
    "process_execution",
}


class ProfilePreset(BaseModel):
    """Immutable template for a built-in MCP profile preset."""

    model_config = ConfigDict(frozen=True)

    id: str
    version: str
    profile: MCPProfile


def _policy(
    *,
    capabilities: list[str],
    risk_classes: list[str] | None = None,
    allowed_tools: list[str] | None = None,
    tool_patterns: list[str] | None = None,
    module_patterns: list[str] | None = None,
) -> ProfilePolicy:
    return ProfilePolicy(
        allowed_tools=allowed_tools or [],
        capabilities=capabilities,
        tool_patterns=tool_patterns or [],
        module_patterns=module_patterns or [],
        risk_classes=risk_classes or [],
    )


def _profile(
    *,
    preset_id: str,
    name: str,
    description: str,
    capabilities: list[str],
    risk_classes: list[str] | None = None,
    approval_policy: dict[str, Any] | None = None,
    external_server_grants: list[dict[str, Any]] | None = None,
    credential_grants: list[dict[str, Any]] | None = None,
    provenance: dict[str, Any] | None = None,
    agent_metadata: dict[str, Any] | None = None,
) -> MCPProfile:
    metadata = {
        "agent_metadata": {
            "ui_label": name,
            **(agent_metadata or {}),
        }
    }
    return MCPProfile(
        id=preset_id,
        name=name,
        description=description,
        preset_id=preset_id,
        preset_version=PRESET_VERSION,
        policy_document=_policy(
            capabilities=capabilities,
            risk_classes=risk_classes,
        ),
        approval_policy=approval_policy or {},
        external_server_grants=external_server_grants or [],
        credential_grants=credential_grants or [],
        metadata=metadata,
        provenance={
            "source": "builtin_preset",
            "preset_id": preset_id,
            "preset_version": PRESET_VERSION,
            **(provenance or {}),
        },
    )


def _preset(
    *,
    preset_id: str,
    name: str,
    description: str,
    capabilities: list[str],
    risk_classes: list[str] | None = None,
    approval_policy: dict[str, Any] | None = None,
    external_server_grants: list[dict[str, Any]] | None = None,
    provenance: dict[str, Any] | None = None,
    agent_metadata: dict[str, Any] | None = None,
) -> ProfilePreset:
    return ProfilePreset(
        id=preset_id,
        version=PRESET_VERSION,
        profile=_profile(
            preset_id=preset_id,
            name=name,
            description=description,
            capabilities=capabilities,
            risk_classes=risk_classes,
            approval_policy=approval_policy,
            external_server_grants=external_server_grants,
            provenance=provenance,
            agent_metadata=agent_metadata,
        ),
    )


_WRITE_APPROVAL_POLICY: dict[str, Any] = {
    "required_for": ["write", "mutating"],
    "reason": "Preset includes scoped write-oriented capabilities.",
}

_BUILTIN_PRESETS: tuple[ProfilePreset, ...] = (
    _preset(
        preset_id="orchestrator",
        name="Orchestrator",
        description="Coordinates workflows with broad read access and approval-gated scoped writes.",
        capabilities=["workflow.plan", "task.coordinate", "workspace.read", "workspace.write_scoped"],
        risk_classes=["mutating"],
        approval_policy=_WRITE_APPROVAL_POLICY,
    ),
    _preset(
        preset_id="product-owner",
        name="Product Owner",
        description="Plans user stories, requirements, and documentation without process execution.",
        capabilities=["issues.plan", "stories.write", "docs.read", "docs.write_scoped"],
        risk_classes=["mutating"],
        approval_policy=_WRITE_APPROVAL_POLICY,
    ),
    _preset(
        preset_id="architect",
        name="Architect",
        description="Reviews architecture with code search, documentation, and read-only filesystem access.",
        capabilities=["code_search", "docs.read", "filesystem.read"],
    ),
    _preset(
        preset_id="merge-conflict-resolver",
        name="Merge Conflict Resolver",
        description="Inspects git state and writes repo-scoped conflict resolutions with approval gates.",
        capabilities=["git.status", "git.diff", "filesystem.read", "repo.write_scoped"],
        risk_classes=["mutating"],
        approval_policy={
            "required_for": ["repo.write_scoped", "git.destructive"],
            "reason": "Conflict resolution writes are repo-scoped and mutating.",
        },
    ),
    _preset(
        preset_id="documentation-writer",
        name="Documentation Writer",
        description="Reads and writes documentation within scoped workspace paths.",
        capabilities=["docs.read", "docs.write_scoped", "filesystem.read"],
        risk_classes=["mutating"],
        approval_policy=_WRITE_APPROVAL_POLICY,
    ),
    _preset(
        preset_id="project-researcher",
        name="Project Researcher",
        description="Searches and reads the codebase without write or execution capabilities.",
        capabilities=["code_search", "filesystem.read", "docs.read"],
    ),
    _preset(
        preset_id="deep-researcher",
        name="Deep Researcher",
        description="Uses research and citation tools with explicit external network permission.",
        capabilities=["research.web", "citations.write", "external.network"],
        risk_classes=["external_network"],
        external_server_grants=[
            {
                "server_id": "research",
                "grant": "network_research",
                "provenance": "Deep research requires explicitly granted external network access.",
            }
        ],
        provenance={
            "external_network": "Deep research preset intentionally grants web research capability.",
            "high_risk": {
                "external_network": "External network is limited to research/citation tools.",
            },
        },
    ),
    _preset(
        preset_id="code-reviewer",
        name="Code Reviewer",
        description="Reviews diffs, code, and test results without write access.",
        capabilities=["code_search", "diff.read", "tests.read", "filesystem.read"],
    ),
    _preset(
        preset_id="devops-engineer",
        name="DevOps Engineer",
        description="Inspects deployment, logs, and infrastructure state with approval for mutating actions.",
        capabilities=["deploy.inspect", "logs.read", "infra.read", "infra.mutate_scoped"],
        risk_classes=["mutating"],
        approval_policy={
            "required_for": ["infra.mutate_scoped", "deployment.mutate"],
            "reason": "Infrastructure changes require explicit approval.",
        },
    ),
    _preset(
        preset_id="backend-engineer",
        name="Backend Engineer",
        description="Reads and writes backend source with scoped test/build commands behind approval.",
        capabilities=["source.write_scoped", "code_search", "tests.request", "backend.inspect"],
        risk_classes=["mutating"],
        approval_policy=_WRITE_APPROVAL_POLICY,
    ),
    _preset(
        preset_id="frontend-engineer",
        name="Frontend Engineer",
        description="Works on scoped frontend source and browser/debug flows without broad process execution.",
        capabilities=["source.write_scoped", "browser.debug", "frontend.inspect", "tests.request"],
        risk_classes=["mutating"],
        approval_policy=_WRITE_APPROVAL_POLICY,
    ),
    _preset(
        preset_id="qa-engineer",
        name="QA Engineer",
        description="Debugs running applications with browser, logs, screenshots, and read-only app state.",
        capabilities=["browser.debug", "logs.read", "screenshots.capture", "app_state.read"],
    ),
    _preset(
        preset_id="sdet",
        name="Software Development Engineer in Test",
        description="Authors and runs automated tests through scoped write and approval-gated runner requests.",
        capabilities=["tests.write_scoped", "tests.request", "code_search", "filesystem.read"],
        risk_classes=["mutating"],
        approval_policy=_WRITE_APPROVAL_POLICY,
    ),
    _preset(
        preset_id="memory-keeper",
        name="Memory Keeper",
        description="Maintains graph and memory tools without shell or process access.",
        capabilities=["memory.graph.read", "memory.graph.write_scoped", "notes.read"],
        risk_classes=["mutating"],
        approval_policy=_WRITE_APPROVAL_POLICY,
        agent_metadata={"memory_provider": "graphiti"},
    ),
)

_BUILTIN_BY_ID = {preset.id: preset for preset in _BUILTIN_PRESETS}


def list_builtin_presets() -> tuple[ProfilePreset, ...]:
    """Return copies of the bundled profile presets."""
    return tuple(preset.model_copy(deep=True) for preset in _BUILTIN_PRESETS)


def get_builtin_preset(preset_id: str) -> ProfilePreset | None:
    """Return a copy of a bundled preset by id, or None when unknown."""
    preset = _BUILTIN_BY_ID.get(preset_id)
    if preset is None:
        return None
    return preset.model_copy(deep=True)


def duplicate_builtin_preset(
    preset_id: str,
    *,
    profile_id: str | None = None,
    name: str | None = None,
) -> MCPProfile:
    """Copy a bundled preset into an editable user profile."""
    preset = get_builtin_preset(preset_id)
    if preset is None:
        raise ValueError(f"Unknown MCP profile preset: {preset_id}")

    profile = preset.profile.model_copy(deep=True)
    profile.id = profile_id or f"{preset.id}-copy"
    if name is not None:
        profile.name = name
    profile.preset_id = preset.id
    profile.preset_version = preset.version
    profile.provenance = {
        **profile.provenance,
        "source": "builtin_preset",
        "preset_id": preset.id,
        "preset_version": preset.version,
        "duplicated": True,
    }
    return profile


def validate_preset_safety(preset: ProfilePreset) -> list[str]:
    """Return safety-baseline violation codes for a preset template."""
    profile = preset.profile
    policy = profile.policy_document
    capabilities = set(policy.capabilities)
    risk_classes = set(policy.risk_classes)
    violations: list[str] = []

    if capabilities & _PROCESS_CAPABILITIES or "process_execution" in risk_classes:
        if not _approval_required_for(profile, "process_execution"):
            violations.append("process_execution_requires_approval")
        if not _has_high_risk_provenance(profile, "process_execution"):
            violations.append("high_risk_capability_requires_provenance")

    if capabilities & _DESTRUCTIVE_FILESYSTEM_CAPABILITIES or "destructive_filesystem" in risk_classes:
        if not _approval_required_for(profile, "destructive_filesystem"):
            violations.append("destructive_filesystem_requires_approval")
        if not _has_high_risk_provenance(profile, "destructive_filesystem"):
            violations.append("high_risk_capability_requires_provenance")

    if profile.credential_grants or "credential_use" in risk_classes:
        if not _has_high_risk_provenance(profile, "credential_use"):
            violations.append("credential_grant_requires_provenance")

    if capabilities & _EXTERNAL_NETWORK_CAPABILITIES or "external_network" in risk_classes:
        if not profile.external_server_grants:
            violations.append("external_network_requires_explicit_grant")
        if not _has_high_risk_provenance(profile, "external_network"):
            violations.append("high_risk_capability_requires_provenance")

    unknown_high_risk = risk_classes - _HIGH_RISK_RISK_CLASSES - {"mutating"}
    if unknown_high_risk:
        violations.append("unknown_high_risk_requires_review")

    return sorted(set(violations))


def _approval_required_for(profile: MCPProfile, risk_class: str) -> bool:
    approval_policy = profile.approval_policy
    required_for = approval_policy.get("required_for", [])
    if approval_policy.get("required") is True:
        return True
    if isinstance(required_for, str):
        required_for = [required_for]
    return risk_class in required_for or "mutating" in required_for or "write" in required_for


def _has_high_risk_provenance(profile: MCPProfile, risk_class: str) -> bool:
    high_risk = profile.provenance.get("high_risk")
    if isinstance(high_risk, dict) and high_risk.get(risk_class):
        return True
    return bool(profile.provenance.get(risk_class))
