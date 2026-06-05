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

TOOLING_PRESET_IDS = {
    "product-owner",
    "architect",
    "merge-conflict-resolver",
    "documentation-writer",
    "project-researcher",
    "code-reviewer",
    "devops-engineer",
    "backend-engineer",
    "frontend-engineer",
    "qa-engineer",
    "sdet",
}

TOOL_CATEGORY_BY_PREFIX = {
    "app_state": "app_state",
    "api": "backend",
    "browser": "browser",
    "code": "code",
    "deploy": "deployments",
    "docs": "docs",
    "fs": "files",
    "git": "git",
    "infra": "infra",
    "kanban": "issues",
    "logs": "logs",
    "memory": "memory",
    "profile": "tool_discovery",
    "screenshots": "screenshots",
    "tests": "tests",
    "test_cases": "tests",
    "tool_categories": "tool_discovery",
    "tool_describe": "tool_discovery",
    "tool_search": "tool_discovery",
    "ui": "frontend",
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
    assert all(preset.version == "2026.06.04" for preset in bundled)

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


def test_role_presets_include_tooling_metadata() -> None:
    bundled_by_id = {preset.id: preset for preset in presets.list_builtin_presets()}

    for preset_id in TOOLING_PRESET_IDS:
        preset = bundled_by_id[preset_id]
        tooling = preset.profile.metadata["tooling"]

        assert tooling["enabled_tools"]
        assert tooling["enabled_capabilities"]
        assert tooling["recommended_tools"]
        assert tooling["recommended_servers"]
        assert all(
            item["id"] not in preset.profile.policy_document.allowed_tools
            for item in tooling["recommended_tools"]
        )
        assert tooling["recommendation_catalog_patchable"] is True
        assert tooling["progressive_disclosure"]["max_direct_tools"] <= 24


def test_tool_category_prefix_mapping_covers_all_enabled_tool_prefixes() -> None:
    missing_by_preset: dict[str, list[str]] = {}
    for preset in presets.list_builtin_presets():
        tooling = preset.profile.metadata.get("tooling")
        if not isinstance(tooling, dict):
            continue
        missing_prefixes = sorted(
            {
                str(tool).split(".", 1)[0]
                for tool in tooling.get("enabled_tools") or []
                if str(tool).split(".", 1)[0] not in TOOL_CATEGORY_BY_PREFIX
            }
        )
        if missing_prefixes:
            missing_by_preset[preset.id] = missing_prefixes

    assert missing_by_preset == {}  # nosec B101


def test_preset_direct_categories_do_not_exceed_max_direct_tools() -> None:
    bundled_by_id = {preset.id: preset for preset in presets.list_builtin_presets()}

    for preset_id in TOOLING_PRESET_IDS:
        tooling = bundled_by_id[preset_id].profile.metadata["tooling"]
        progressive_disclosure = tooling["progressive_disclosure"]
        direct_categories = set(progressive_disclosure["direct_categories"])
        direct_tools = [
            tool
            for tool in tooling["enabled_tools"]
            if TOOL_CATEGORY_BY_PREFIX[tool.split(".", 1)[0]] in direct_categories
        ]

        assert len(direct_tools) <= progressive_disclosure["max_direct_tools"], preset_id  # nosec B101


def test_filesystem_read_presets_include_helper_tools() -> None:
    expected = {"fs.stat", "fs.glob", "fs.grep"}
    for preset in presets.list_builtin_presets():
        tooling = preset.profile.metadata.get("tooling")
        if not isinstance(tooling, dict):
            continue
        enabled_tools = set(tooling.get("enabled_tools") or [])
        if {"fs.list", "fs.read_text"} <= enabled_tools:
            assert expected <= enabled_tools


def test_web_search_is_recommended_unavailable_not_enabled() -> None:
    product_owner = presets.get_builtin_preset("product-owner")
    assert product_owner is not None

    tooling = product_owner.profile.metadata["tooling"]
    progressive_disclosure = tooling["progressive_disclosure"]
    web_search_markers = {"web.search", "web_search"}
    web_search_recommendations = [
        item
        for item in tooling["recommended_servers"]
        if item["category"] == "web_search"
    ]

    assert "web.search" not in product_owner.profile.policy_document.allowed_tools
    assert web_search_markers.isdisjoint(tooling["enabled_tools"])
    assert web_search_markers.isdisjoint(tooling["enabled_capabilities"])
    assert all(
        item.get("id") not in web_search_markers
        and item.get("category") not in web_search_markers
        for item in tooling["recommended_tools"]
    )
    assert "web_search" not in progressive_disclosure["direct_categories"]
    assert "web_search" not in progressive_disclosure["deferred_categories"]
    assert web_search_recommendations
    assert all(item["required"] is False for item in web_search_recommendations)


def test_cdp_browser_exact_target_is_documented() -> None:
    frontend = presets.get_builtin_preset("frontend-engineer")
    assert frontend is not None

    browser_servers = [
        server
        for server in frontend.profile.metadata["tooling"]["recommended_servers"]
        if server["category"] == "browser"
    ]

    assert browser_servers
    assert any(
        option["id"] == "chrome-devtools-mcp"
        and option["install_target"] == "ChromeDevTools/chrome-devtools-mcp"
        and option["maturity"] == "exact_target"
        for server in browser_servers
        for option in server["binding_options"]
    )


def test_browser_capable_presets_include_native_cdp_read_tools() -> None:
    browser_read_tools = {
        "browser.status",
        "browser.pages.list",
        "browser.snapshot",
        "browser.page_state",
        "browser.screenshot",
        "browser.console",
        "browser.network",
    }

    for preset_id in ("frontend-engineer", "qa-engineer", "sdet"):
        preset = presets.get_builtin_preset(preset_id)
        assert preset is not None  # nosec B101

        tooling = preset.profile.metadata["tooling"]
        policy_caps = set(preset.profile.policy_document.capabilities)
        assert browser_read_tools <= set(tooling["enabled_tools"])  # nosec B101
        assert {"browser.inspect", "browser.debug"} <= set(tooling["enabled_capabilities"])  # nosec B101
        assert "browser.inspect" in policy_caps  # nosec B101
        for capability in {"browser.debug", "screenshots.capture", "app_state.read"}:
            if capability in tooling["enabled_capabilities"]:
                assert capability in policy_caps  # nosec B101
        assert "browser" in tooling["progressive_disclosure"]["direct_categories"]  # nosec B101
        assert "browser" not in tooling["progressive_disclosure"]["deferred_categories"]  # nosec B101


def test_git_capable_presets_include_native_read_tools() -> None:
    git_read_tools = {
        "git.status",
        "git.diff",
        "git.log",
        "git.blame",
        "git.branches",
        "git.conflicts.list",
        "git.conflicts.read",
    }

    for preset_id in (
        "architect",
        "merge-conflict-resolver",
        "code-reviewer",
        "devops-engineer",
        "backend-engineer",
        "qa-engineer",
    ):
        preset = presets.get_builtin_preset(preset_id)
        assert preset is not None  # nosec B101

        tooling = preset.profile.metadata["tooling"]
        assert git_read_tools <= set(tooling["enabled_tools"])  # nosec B101
        assert "git.read" in tooling["enabled_capabilities"]  # nosec B101
        assert "git.read" in preset.profile.policy_document.capabilities  # nosec B101
        assert "git" in tooling["progressive_disclosure"]["direct_categories"]  # nosec B101

    for preset_id in ("frontend-engineer", "sdet"):
        preset = presets.get_builtin_preset(preset_id)
        assert preset is not None  # nosec B101

        tooling = preset.profile.metadata["tooling"]
        assert git_read_tools <= set(tooling["enabled_tools"])  # nosec B101
        assert "git.read" in tooling["enabled_capabilities"]  # nosec B101
        assert "git.read" in preset.profile.policy_document.capabilities  # nosec B101
        assert "git" not in tooling["progressive_disclosure"]["direct_categories"]  # nosec B101
        assert "git" in tooling["progressive_disclosure"]["deferred_categories"]  # nosec B101


def test_product_owner_and_documentation_writer_do_not_enable_git_by_default() -> None:
    git_read_tools = {
        "git.status",
        "git.diff",
        "git.log",
        "git.blame",
        "git.branches",
        "git.conflicts.list",
        "git.conflicts.read",
    }

    for preset_id in ("product-owner", "documentation-writer"):
        preset = presets.get_builtin_preset(preset_id)
        assert preset is not None  # nosec B101

        tooling = preset.profile.metadata["tooling"]
        assert git_read_tools.isdisjoint(tooling["enabled_tools"])  # nosec B101
        assert "git.read" not in tooling["enabled_capabilities"]  # nosec B101
        assert "git.read" not in preset.profile.policy_document.capabilities  # nosec B101
        assert "git" not in tooling["progressive_disclosure"]["direct_categories"]  # nosec B101


def test_recommendation_catalog_patch_does_not_grant_authority() -> None:
    from mcp_unified.profiles.tooling import merge_tooling_recommendations

    product_owner = presets.get_builtin_preset("product-owner")
    assert product_owner is not None

    patched_tooling = merge_tooling_recommendations(
        product_owner.profile.metadata["tooling"],
        {
            "recommended_tools": [
                {
                    "id": "shell.run",
                    "category": "shell",
                    "activation": "requires_operator_enablement",
                }
            ]
        },
    )

    assert any(item["id"] == "shell.run" for item in patched_tooling["recommended_tools"])
    assert "shell.run" not in product_owner.profile.policy_document.allowed_tools


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


def test_builtin_preset_template_timestamps_are_version_stable() -> None:
    """Keep built-in preset provenance timestamps tied to the release date."""

    bundled = presets.list_builtin_presets()

    assert presets.PRESET_RELEASE_DATE.strftime("%Y.%m.%d") == presets.PRESET_VERSION
    assert presets.PRESET_CREATED_AT.date() == presets.PRESET_RELEASE_DATE
    assert all(
        preset.profile.created_at == presets.PRESET_CREATED_AT
        for preset in bundled
    )
    assert all(
        preset.profile.updated_at == presets.PRESET_CREATED_AT
        for preset in bundled
    )


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
    assert profile.preset_version == "2026.06.04"
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


def test_safety_validation_accepts_reviewed_high_risk_classes_with_approval_and_provenance() -> None:
    profile = MCPProfile(
        id="safe-reviewed-risk",
        name="Safe Reviewed Risk",
        policy_document={
            "risk_classes": [
                "browser_mutation",
                "git_mutation",
                "deployment_mutation",
                "memory_mutation",
                "test_execution",
            ],
        },
        approval_policy={
            "required_for": [
                "browser_mutation",
                "git_mutation",
                "deployment_mutation",
                "memory_mutation",
                "test_execution",
            ],
        },
        provenance={
            "high_risk": {
                "browser_mutation": "reviewed",
                "git_mutation": "reviewed",
                "deployment_mutation": "reviewed",
                "memory_mutation": "reviewed",
                "test_execution": "reviewed",
            },
        },
    )
    preset = presets.ProfilePreset(
        id="safe-reviewed-risk",
        version="test",
        profile=profile,
    )

    assert presets.validate_preset_safety(preset) == []


def test_safety_validation_rejects_reviewed_high_risk_class_without_approval() -> None:
    profile = MCPProfile(
        id="unsafe-browser",
        name="Unsafe Browser",
        policy_document={"risk_classes": ["browser_mutation"]},
        provenance={"high_risk": {"browser_mutation": "reviewed"}},
    )
    preset = presets.ProfilePreset(
        id="unsafe-browser",
        version="test",
        profile=profile,
    )

    assert "browser_mutation_requires_approval" in presets.validate_preset_safety(preset)


def test_safety_validation_requires_explicit_reviewed_high_risk_approval() -> None:
    profile = MCPProfile(
        id="unsafe-generic-browser",
        name="Unsafe Generic Browser",
        policy_document={"risk_classes": ["browser_mutation"]},
        approval_policy={"required_for": ["mutating", "write"]},
        provenance={"high_risk": {"browser_mutation": "reviewed"}},
    )
    preset = presets.ProfilePreset(
        id="unsafe-generic-browser",
        version="test",
        profile=profile,
    )

    assert "browser_mutation_requires_approval" in presets.validate_preset_safety(preset)


def test_safety_validation_rejects_unknown_future_risk_classes() -> None:
    profile = MCPProfile(
        id="unsafe-future-risk",
        name="Unsafe Future Risk",
        policy_document={"risk_classes": ["future_mutation"]},
        approval_policy={"required_for": ["future_mutation"]},
        provenance={"high_risk": {"future_mutation": "reviewed"}},
    )
    preset = presets.ProfilePreset(
        id="unsafe-future-risk",
        version="test",
        profile=profile,
    )

    assert "unknown_high_risk_requires_review" in presets.validate_preset_safety(preset)


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
