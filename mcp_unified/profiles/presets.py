"""Built-in MCP profile presets for front-end modes."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from .models import MCPProfile, ProfilePolicy
from .tooling import (
    browser_server_recommendation,
    issue_tracker_server_recommendation,
    tooling_metadata,
    web_search_server_recommendation,
)

PRESET_RELEASE_DATE = date(2026, 5, 27)
PRESET_VERSION = PRESET_RELEASE_DATE.strftime("%Y.%m.%d")
PRESET_CREATED_AT = datetime(
    PRESET_RELEASE_DATE.year,
    PRESET_RELEASE_DATE.month,
    PRESET_RELEASE_DATE.day,
    tzinfo=timezone.utc,
)

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
    resource_constraints: dict[str, Any] | None = None,
) -> ProfilePolicy:
    """Build a conservative package-local policy document for a preset."""
    return ProfilePolicy(
        allowed_tools=allowed_tools or [],
        capabilities=capabilities,
        tool_patterns=tool_patterns or [],
        module_patterns=module_patterns or [],
        risk_classes=risk_classes or [],
        resource_constraints=resource_constraints or {},
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
    requires_workspace_binding: bool = False,
    provenance: dict[str, Any] | None = None,
    agent_metadata: dict[str, Any] | None = None,
    tooling_metadata_document: dict[str, Any] | None = None,
) -> MCPProfile:
    """Build an MCP profile template with preset provenance metadata."""
    metadata: dict[str, Any] = {
        "agent_metadata": {
            "ui_label": name,
            **(agent_metadata or {}),
        }
    }
    if tooling_metadata_document is not None:
        metadata["tooling"] = tooling_metadata_document

    return MCPProfile(
        id=preset_id,
        name=name,
        description=description,
        preset_id=preset_id,
        preset_version=PRESET_VERSION,
        policy_document=_policy(
            capabilities=capabilities,
            risk_classes=risk_classes,
            resource_constraints={
                "requires_workspace_binding": True,
                "binding_stage": "assignment",
            }
            if requires_workspace_binding
            else None,
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
        created_at=PRESET_CREATED_AT,
        updated_at=PRESET_CREATED_AT,
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
    requires_workspace_binding: bool = False,
    provenance: dict[str, Any] | None = None,
    agent_metadata: dict[str, Any] | None = None,
    tooling_metadata_document: dict[str, Any] | None = None,
) -> ProfilePreset:
    """Build an immutable preset wrapper for a bundled MCP profile template."""
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
            requires_workspace_binding=requires_workspace_binding,
            provenance=provenance,
            agent_metadata=agent_metadata,
            tooling_metadata_document=tooling_metadata_document,
        ),
    )


_WRITE_APPROVAL_POLICY: dict[str, Any] = {
    "required_for": ["write", "mutating"],
    "reason": "Preset includes scoped write-oriented capabilities.",
}

_TOOL_DISCOVERY_TOOLS = [
    "tool_categories.list",
    "tool_search",
    "tool_describe",
    "profile.tools.list",
]
_FILES_READ_TOOLS = ["fs.list", "fs.read_text"]
_FILES_WRITE_TOOLS = [*_FILES_READ_TOOLS, "fs.write_text"]
_CODE_READ_TOOLS = ["code.search", "code.symbols", "code.references"]
_DOCS_READ_TOOLS = ["docs.search", "docs.read"]
_DOCS_WRITE_TOOLS = [*_DOCS_READ_TOOLS, "docs.write"]
_GIT_READ_TOOLS = ["git.status", "git.diff", "git.conflicts.list", "git.conflicts.read"]
_TEST_READ_TOOLS = ["tests.results.read", "tests.logs.read"]
_TEST_REQUEST_TOOLS = [*_TEST_READ_TOOLS, "tests.request"]

_BUILTIN_PRESETS: tuple[ProfilePreset, ...] = (
    _preset(
        preset_id="orchestrator",
        name="Orchestrator",
        description="Coordinates workflows with broad read access and approval-gated scoped writes.",
        capabilities=["workflow.plan", "task.coordinate", "workspace.read", "workspace.write_scoped"],
        risk_classes=["mutating"],
        approval_policy=_WRITE_APPROVAL_POLICY,
        requires_workspace_binding=True,
    ),
    _preset(
        preset_id="product-owner",
        name="Product Owner",
        description="Plans user stories, requirements, and documentation without process execution.",
        capabilities=["issues.plan", "stories.write", "docs.read", "docs.write_scoped"],
        risk_classes=["mutating"],
        approval_policy=_WRITE_APPROVAL_POLICY,
        requires_workspace_binding=True,
        tooling_metadata_document=tooling_metadata(
            enabled_tools=[
                *_TOOL_DISCOVERY_TOOLS,
                "fs.list",
                "fs.read_text",
                "fs.write_text",
                "kanban.cards.create",
                "memory.recall",
            ],
            enabled_capabilities=[
                "filesystem.read",
                "filesystem.write_scoped",
                "issues.plan",
                "stories.write",
                "memory.read",
            ],
            direct_categories=["files", "tool_discovery", "issues", "memory"],
            deferred_categories=[
                "issue_tracker",
                "docs_search",
                "browser",
            ],
            recommended_servers=[
                web_search_server_recommendation(),
                browser_server_recommendation(),
                issue_tracker_server_recommendation(),
            ],
        ),
    ),
    _preset(
        preset_id="architect",
        name="Architect",
        description="Reviews architecture with code search, documentation, and read-only filesystem access.",
        capabilities=["code_search", "docs.read", "filesystem.read"],
        tooling_metadata_document=tooling_metadata(
            enabled_tools=[
                *_TOOL_DISCOVERY_TOOLS,
                *_FILES_READ_TOOLS,
                *_CODE_READ_TOOLS,
                *_DOCS_READ_TOOLS,
            ],
            enabled_capabilities=[
                "code_search",
                "docs.read",
                "filesystem.read",
                "architecture.review",
            ],
            direct_categories=["files", "tool_discovery", "code", "docs"],
            deferred_categories=["web_search", "browser", "diagramming"],
            recommended_servers=[
                web_search_server_recommendation(),
                browser_server_recommendation(),
            ],
        ),
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
        requires_workspace_binding=True,
        tooling_metadata_document=tooling_metadata(
            enabled_tools=[
                *_TOOL_DISCOVERY_TOOLS,
                *_GIT_READ_TOOLS,
                *_FILES_WRITE_TOOLS,
            ],
            enabled_capabilities=[
                "git.status",
                "git.diff",
                "filesystem.read",
                "repo.write_scoped",
            ],
            direct_categories=["git", "files", "tool_discovery"],
            deferred_categories=["safe_test_runner", "issue_tracker", "browser"],
            recommended_servers=[
                browser_server_recommendation(),
                issue_tracker_server_recommendation(),
            ],
        ),
    ),
    _preset(
        preset_id="documentation-writer",
        name="Documentation Writer",
        description="Reads and writes documentation within scoped workspace paths.",
        capabilities=["docs.read", "docs.write_scoped", "filesystem.read"],
        risk_classes=["mutating"],
        approval_policy=_WRITE_APPROVAL_POLICY,
        requires_workspace_binding=True,
        tooling_metadata_document=tooling_metadata(
            enabled_tools=[
                *_TOOL_DISCOVERY_TOOLS,
                *_FILES_WRITE_TOOLS,
                *_DOCS_WRITE_TOOLS,
                "memory.recall",
            ],
            enabled_capabilities=[
                "docs.read",
                "docs.write_scoped",
                "filesystem.read",
                "filesystem.write_scoped",
                "memory.read",
            ],
            direct_categories=["files", "docs", "tool_discovery", "memory"],
            deferred_categories=["web_search", "browser", "issue_tracker"],
            recommended_servers=[
                web_search_server_recommendation(),
                browser_server_recommendation(),
                issue_tracker_server_recommendation(),
            ],
        ),
    ),
    _preset(
        preset_id="project-researcher",
        name="Project Researcher",
        description="Searches and reads the codebase without write or execution capabilities.",
        capabilities=["code_search", "filesystem.read", "docs.read"],
        tooling_metadata_document=tooling_metadata(
            enabled_tools=[
                *_TOOL_DISCOVERY_TOOLS,
                *_FILES_READ_TOOLS,
                *_CODE_READ_TOOLS,
                *_DOCS_READ_TOOLS,
            ],
            enabled_capabilities=["code_search", "filesystem.read", "docs.read"],
            direct_categories=["files", "code", "docs", "tool_discovery"],
            deferred_categories=["web_search", "browser", "citations"],
            recommended_servers=[
                web_search_server_recommendation(),
                browser_server_recommendation(),
            ],
        ),
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
        requires_workspace_binding=True,
    ),
    _preset(
        preset_id="code-reviewer",
        name="Code Reviewer",
        description="Reviews diffs, code, and test results without write access.",
        capabilities=["code_search", "diff.read", "tests.read", "filesystem.read"],
        tooling_metadata_document=tooling_metadata(
            enabled_tools=[
                *_TOOL_DISCOVERY_TOOLS,
                *_FILES_READ_TOOLS,
                *_CODE_READ_TOOLS,
                *_GIT_READ_TOOLS,
                *_TEST_READ_TOOLS,
            ],
            enabled_capabilities=[
                "code_search",
                "diff.read",
                "tests.read",
                "filesystem.read",
            ],
            direct_categories=["files", "code", "git", "tests", "tool_discovery"],
            deferred_categories=["browser", "issue_tracker", "safe_test_runner"],
            recommended_servers=[
                browser_server_recommendation(),
                issue_tracker_server_recommendation(),
            ],
        ),
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
        requires_workspace_binding=True,
        tooling_metadata_document=tooling_metadata(
            enabled_tools=[
                *_TOOL_DISCOVERY_TOOLS,
                "deploy.status",
                "deploy.logs.read",
                "infra.inspect",
                "logs.search",
            ],
            enabled_capabilities=["deploy.inspect", "logs.read", "infra.read"],
            direct_categories=["deployments", "logs", "infra", "tool_discovery"],
            deferred_categories=["issue_tracker", "safe_test_runner", "browser"],
            recommended_servers=[
                browser_server_recommendation(),
                issue_tracker_server_recommendation(),
            ],
        ),
    ),
    _preset(
        preset_id="backend-engineer",
        name="Backend Engineer",
        description="Reads and writes backend source with scoped test/build commands behind approval.",
        capabilities=["source.write_scoped", "code_search", "tests.request", "backend.inspect"],
        risk_classes=["mutating"],
        approval_policy=_WRITE_APPROVAL_POLICY,
        requires_workspace_binding=True,
        tooling_metadata_document=tooling_metadata(
            enabled_tools=[
                *_TOOL_DISCOVERY_TOOLS,
                *_FILES_WRITE_TOOLS,
                *_CODE_READ_TOOLS,
                *_TEST_REQUEST_TOOLS,
                "api.schema.inspect",
            ],
            enabled_capabilities=[
                "source.write_scoped",
                "code_search",
                "tests.request",
                "backend.inspect",
            ],
            direct_categories=[
                "files",
                "code",
                "tests",
                "backend",
                "tool_discovery",
            ],
            deferred_categories=[
                "safe_test_runner",
                "issue_tracker",
                "browser",
                "web_search",
            ],
            recommended_servers=[
                browser_server_recommendation(),
                issue_tracker_server_recommendation(),
                web_search_server_recommendation(),
            ],
        ),
    ),
    _preset(
        preset_id="frontend-engineer",
        name="Frontend Engineer",
        description="Works on scoped frontend source and browser/debug flows without broad process execution.",
        capabilities=["source.write_scoped", "browser.debug", "frontend.inspect", "tests.request"],
        risk_classes=["mutating"],
        approval_policy=_WRITE_APPROVAL_POLICY,
        requires_workspace_binding=True,
        tooling_metadata_document=tooling_metadata(
            enabled_tools=[
                *_TOOL_DISCOVERY_TOOLS,
                *_FILES_WRITE_TOOLS,
                *_CODE_READ_TOOLS,
                *_TEST_REQUEST_TOOLS,
                "ui.components.inspect",
            ],
            enabled_capabilities=[
                "source.write_scoped",
                "frontend.inspect",
                "tests.request",
                "code_search",
            ],
            direct_categories=[
                "files",
                "code",
                "tests",
                "frontend",
                "tool_discovery",
            ],
            deferred_categories=[
                "browser",
                "safe_test_runner",
                "issue_tracker",
                "web_search",
            ],
            recommended_servers=[
                browser_server_recommendation(),
                issue_tracker_server_recommendation(),
                web_search_server_recommendation(),
            ],
        ),
    ),
    _preset(
        preset_id="qa-engineer",
        name="QA Engineer",
        description="Debugs running applications with browser, logs, screenshots, and read-only app state.",
        capabilities=["browser.debug", "logs.read", "screenshots.capture", "app_state.read"],
        tooling_metadata_document=tooling_metadata(
            enabled_tools=[
                *_TOOL_DISCOVERY_TOOLS,
                "logs.search",
                "screenshots.list",
                "app_state.read",
                "test_cases.read",
            ],
            enabled_capabilities=["logs.read", "screenshots.capture", "app_state.read"],
            direct_categories=["logs", "screenshots", "app_state", "tool_discovery"],
            deferred_categories=["browser", "safe_test_runner", "issue_tracker"],
            recommended_servers=[
                browser_server_recommendation(),
                issue_tracker_server_recommendation(),
            ],
        ),
    ),
    _preset(
        preset_id="sdet",
        name="Software Development Engineer in Test",
        description="Authors and runs automated tests through scoped write and approval-gated runner requests.",
        capabilities=["tests.write_scoped", "tests.request", "code_search", "filesystem.read"],
        risk_classes=["mutating"],
        approval_policy=_WRITE_APPROVAL_POLICY,
        requires_workspace_binding=True,
        tooling_metadata_document=tooling_metadata(
            enabled_tools=[
                *_TOOL_DISCOVERY_TOOLS,
                *_FILES_WRITE_TOOLS,
                *_CODE_READ_TOOLS,
                *_TEST_REQUEST_TOOLS,
            ],
            enabled_capabilities=[
                "tests.write_scoped",
                "tests.request",
                "code_search",
                "filesystem.read",
                "filesystem.write_scoped",
            ],
            direct_categories=["files", "code", "tests", "tool_discovery"],
            deferred_categories=["safe_test_runner", "browser", "issue_tracker"],
            recommended_servers=[
                browser_server_recommendation(),
                issue_tracker_server_recommendation(),
            ],
        ),
    ),
    _preset(
        preset_id="memory-keeper",
        name="Memory Keeper",
        description="Maintains graph and memory tools without shell or process access.",
        capabilities=["memory.graph.read", "memory.graph.write_scoped", "notes.read"],
        risk_classes=["mutating"],
        approval_policy=_WRITE_APPROVAL_POLICY,
        agent_metadata={"memory_provider": "graphiti"},
        requires_workspace_binding=True,
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
    now = datetime.now(timezone.utc)
    profile.id = profile_id or f"{preset.id}-{uuid4().hex[:8]}"
    if name is not None:
        profile.name = name
    profile.preset_id = preset.id
    profile.preset_version = preset.version
    profile.created_at = now
    profile.updated_at = now
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
    """Return whether a profile requires approval for the given risk class."""
    approval_policy = profile.approval_policy
    if approval_policy.get("required") is True:
        return True

    raw_required_for = approval_policy.get("required_for", [])
    if raw_required_for is None:
        required_for: list[str] = []
    elif isinstance(raw_required_for, str):
        required_for = [raw_required_for]
    elif isinstance(raw_required_for, (list, tuple, set)):
        required_for = [item for item in raw_required_for if isinstance(item, str)]
    else:
        required_for = []

    if risk_class == "process_execution":
        return "process_execution" in required_for

    return risk_class in required_for or "mutating" in required_for or "write" in required_for


def _has_high_risk_provenance(profile: MCPProfile, risk_class: str) -> bool:
    """Return whether high-risk capability provenance is recorded."""
    high_risk = profile.provenance.get("high_risk")
    if isinstance(high_risk, dict) and high_risk.get(risk_class):
        return True
    return bool(profile.provenance.get(risk_class))
