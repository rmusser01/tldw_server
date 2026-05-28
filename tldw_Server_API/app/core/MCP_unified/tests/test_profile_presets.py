"""Tests for package-local MCP profile preset primitives."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
from pathlib import Path

import mcp_unified.profiles.presets as presets
from mcp_unified.profiles.models import MCPProfile

EXPECTED_PRESET_IDS = {
    "orchestrator",
    "product-owner",
    "architect",
    "merge-conflict-resolver",
    "documentation-writer",
    "project-researcher",
    "deep-researcher",
    "code-reviewer",
    "devops-engineer",
    "backend-engineer",
    "frontend-engineer",
    "qa-engineer",
    "sdet",
    "memory-keeper",
}

WORKSPACE_BOUND_PRESET_IDS = {
    "orchestrator",
    "product-owner",
    "merge-conflict-resolver",
    "documentation-writer",
    "deep-researcher",
    "devops-engineer",
    "backend-engineer",
    "frontend-engineer",
    "sdet",
    "memory-keeper",
}


def _tldw_imports_for(path: Path) -> list[str]:
    """Return imports from a Python file that cross into the host package."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(
                alias.name
                for alias in node.names
                if alias.name == "tldw_Server_API"
                or alias.name.startswith("tldw_Server_API.")
            )
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "tldw_Server_API" or node.module.startswith("tldw_Server_API."):
                imports.append(node.module)
    return imports


def test_profile_presets_module_has_no_tldw_server_imports() -> None:
    package_root = Path(presets.__file__).resolve().parent
    offenders: dict[str, list[str]] = {}
    for path in package_root.rglob("*.py"):
        imports = _tldw_imports_for(path)
        if imports:
            offenders[str(path)] = imports
    assert offenders == {}


def test_builtin_presets_cover_initial_mode_ids_and_pass_safety_baseline() -> None:
    bundled = presets.list_builtin_presets()
    assert {preset.id for preset in bundled} == EXPECTED_PRESET_IDS
    assert all(preset.version == "2026.05.27" for preset in bundled)

    safety_violations = {
        preset.id: presets.validate_preset_safety(preset)
        for preset in bundled
        if presets.validate_preset_safety(preset)
    }
    assert safety_violations == {}


def test_write_capable_presets_advertise_workspace_binding_requirement() -> None:
    bundled = presets.list_builtin_presets()
    workspace_bound = {
        preset.id
        for preset in bundled
        if preset.profile.policy_document.resource_constraints.get(
            "requires_workspace_binding"
        )
        is True
    }
    assignment_bound = {
        preset.id
        for preset in bundled
        if preset.profile.policy_document.resource_constraints.get("binding_stage")
        == "assignment"
    }

    assert workspace_bound == WORKSPACE_BOUND_PRESET_IDS
    assert assignment_bound == WORKSPACE_BOUND_PRESET_IDS


def test_get_builtin_preset_returns_stable_profile_template() -> None:
    preset = presets.get_builtin_preset("architect")

    assert preset is not None
    assert preset.id == "architect"
    assert preset.profile.id == "architect"
    assert preset.profile.preset_id == "architect"
    assert preset.profile.preset_version == preset.version
    assert preset.profile.metadata["agent_metadata"]["ui_label"] == "Architect"
    assert "code_search" in preset.profile.policy_document.capabilities
    assert preset.profile.credential_grants == []


def test_duplicate_builtin_preset_returns_editable_profile_with_provenance() -> None:
    profile = presets.duplicate_builtin_preset(
        "architect",
        profile_id="workspace-architect",
        name="Workspace Architect",
    )

    assert isinstance(profile, MCPProfile)
    assert profile.id == "workspace-architect"
    assert profile.name == "Workspace Architect"
    assert profile.preset_id == "architect"
    assert profile.preset_version == "2026.05.27"
    assert profile.provenance["source"] == "builtin_preset"
    assert profile.provenance["preset_id"] == "architect"

    profile.policy_document.allowed_tools.append("custom.search")
    original = presets.get_builtin_preset("architect")
    assert original is not None
    assert "custom.search" not in original.profile.policy_document.allowed_tools


def test_duplicate_builtin_preset_default_ids_are_unique() -> None:
    first = presets.duplicate_builtin_preset("architect")
    second = presets.duplicate_builtin_preset("architect")

    assert first.id != second.id
    assert first.id.startswith("architect-")
    assert second.id.startswith("architect-")


def test_duplicate_builtin_preset_refreshes_timestamps() -> None:
    template = presets.get_builtin_preset("architect")
    assert template is not None

    before_duplicate = datetime.now(timezone.utc)
    profile = presets.duplicate_builtin_preset("architect")

    assert profile.created_at >= before_duplicate
    assert profile.updated_at >= before_duplicate
    assert profile.created_at != template.profile.created_at
    assert profile.updated_at != template.profile.updated_at


def test_safety_validation_rejects_unsafe_unapproved_process_capability() -> None:
    unsafe_profile = MCPProfile(
        id="unsafe",
        name="Unsafe",
        policy_document={
            "capabilities": ["process.execute"],
            "risk_classes": ["process_execution"],
        },
    )
    unsafe_preset = presets.ProfilePreset(
        id="unsafe",
        version="test",
        profile=unsafe_profile,
    )

    violations = presets.validate_preset_safety(unsafe_preset)

    assert "process_execution_requires_approval" in violations
    assert "high_risk_capability_requires_provenance" in violations


def test_safety_validation_requires_explicit_process_execution_approval() -> None:
    unsafe_profile = MCPProfile(
        id="unsafe-process",
        name="Unsafe Process",
        policy_document={
            "capabilities": ["process.execute"],
            "risk_classes": ["process_execution"],
        },
        approval_policy={
            "required_for": ["mutating", "write"],
        },
        provenance={
            "high_risk": {
                "process_execution": "Process execution is intentionally present.",
            },
        },
    )
    unsafe_preset = presets.ProfilePreset(
        id="unsafe-process",
        version="test",
        profile=unsafe_profile,
    )

    violations = presets.validate_preset_safety(unsafe_preset)

    assert "process_execution_requires_approval" in violations
    assert "high_risk_capability_requires_provenance" not in violations


def test_safety_validation_handles_null_required_for_policy() -> None:
    unsafe_profile = MCPProfile(
        id="unsafe-null-required-for",
        name="Unsafe Null Required For",
        policy_document={
            "capabilities": ["process.execute"],
            "risk_classes": ["process_execution"],
        },
        approval_policy={
            "required_for": None,
        },
        provenance={
            "high_risk": {
                "process_execution": "Process execution is intentionally present.",
            },
        },
    )
    unsafe_preset = presets.ProfilePreset(
        id="unsafe-null-required-for",
        version="test",
        profile=unsafe_profile,
    )

    violations = presets.validate_preset_safety(unsafe_preset)

    assert "process_execution_requires_approval" in violations
