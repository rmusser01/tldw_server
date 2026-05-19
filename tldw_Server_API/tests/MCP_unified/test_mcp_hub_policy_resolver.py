from __future__ import annotations

import pytest


class _FakeRepo:
    def __init__(self) -> None:
        self.profiles = {
            1: {
                "id": 1,
                "name": "Default Read",
                "owner_scope_type": "global",
                "owner_scope_id": None,
                "policy_document": {
                    "allowed_tools": ["notes.search"],
                    "capabilities": ["filesystem.read"],
                    "path_scope_mode": "workspace_root",
                    "path_scope_enforcement": "approval_required_when_unenforceable",
                    "path_allowlist_prefixes": ["src"],
                },
                "path_scope_object_id": None,
            },
            2: {
                "id": 2,
                "name": "Group External",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "policy_document": {
                    "allowed_tools": ["external.servers.list"],
                    "capabilities": ["network.external"],
                },
                "path_scope_object_id": None,
            },
        }
        self.approval_policies: dict[int, dict] = {}
        self.path_scope_objects = {
            100: {
                "id": 100,
                "name": "Profile Docs",
                "path_scope_document": {
                    "path_scope_mode": "workspace_root",
                    "path_scope_enforcement": "approval_required_when_unenforceable",
                    "path_allowlist_prefixes": ["docs"],
                },
                "is_active": True,
            },
            101: {
                "id": 101,
                "name": "Assignment Current Folder",
                "path_scope_document": {
                    "path_scope_mode": "cwd_descendants",
                    "path_scope_enforcement": "approval_required_when_unenforceable",
                },
                "is_active": True,
            },
        }
        self.assignments = [
            {
                "id": 10,
                "target_type": "default",
                "target_id": None,
                "owner_scope_type": "global",
                "owner_scope_id": None,
                "profile_id": 1,
                "path_scope_object_id": None,
                "inline_policy_document": {},
                "is_active": True,
            },
            {
                "id": 11,
                "target_type": "group",
                "target_id": "ops",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "profile_id": 2,
                "path_scope_object_id": None,
                "inline_policy_document": {"denied_tools": ["external.tools.refresh"]},
                "is_active": True,
            },
            {
                "id": 12,
                "target_type": "persona",
                "target_id": "researcher",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "profile_id": None,
                "path_scope_object_id": None,
                "inline_policy_document": {
                    "allowed_tools": ["Bash(git *)"],
                    "approval_mode": "ask_every_time",
                    "path_scope_mode": "cwd_descendants",
                },
                "is_active": True,
            },
        ]
        self.overrides: dict[int, dict] = {}
        self.assignment_workspaces: dict[int, list[str]] = {}
        self.capability_mappings: dict[tuple[str, int | None, str], dict] = {}
        self.governance_packs: dict[tuple[str, str, str, int | None], dict] = {}

    async def list_policy_assignments(
        self,
        *,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
        target_type: str | None = None,
        target_id: str | None = None,
    ) -> list[dict]:
        rows = list(self.assignments)
        if owner_scope_type is not None:
            rows = [row for row in rows if row["owner_scope_type"] == owner_scope_type]
        if owner_scope_id is not None:
            rows = [row for row in rows if row["owner_scope_id"] == owner_scope_id]
        if target_type is not None:
            rows = [row for row in rows if row["target_type"] == target_type]
        if target_id is not None:
            rows = [row for row in rows if row["target_id"] == target_id]
        return rows

    async def get_permission_profile(self, profile_id: int) -> dict | None:
        return self.profiles.get(profile_id)

    async def get_policy_override_by_assignment(self, assignment_id: int) -> dict | None:
        return self.overrides.get(assignment_id)

    async def get_path_scope_object(self, path_scope_object_id: int) -> dict | None:
        return self.path_scope_objects.get(path_scope_object_id)

    async def get_approval_policy(self, approval_policy_id: int) -> dict | None:
        return self.approval_policies.get(approval_policy_id)

    async def list_policy_assignment_workspaces(self, assignment_id: int) -> list[dict]:
        return [
            {"assignment_id": assignment_id, "workspace_id": workspace_id}
            for workspace_id in self.assignment_workspaces.get(assignment_id, [])
        ]

    async def find_active_capability_mapping(
        self,
        *,
        owner_scope_type: str,
        owner_scope_id: int | None,
        capability_name: str,
    ) -> dict | None:
        return self.capability_mappings.get((owner_scope_type, owner_scope_id, capability_name))

    async def get_governance_pack_by_identity(
        self,
        *,
        pack_id: str,
        pack_version: str,
        owner_scope_type: str,
        owner_scope_id: int | None,
    ) -> dict | None:
        return self.governance_packs.get(
            (pack_id, pack_version, owner_scope_type, owner_scope_id)
        )


@pytest.mark.asyncio
async def test_policy_resolver_merges_default_group_and_persona_targets() -> None:
    from tldw_Server_API.app.services.mcp_hub_policy_resolver import McpHubPolicyResolver

    resolver = McpHubPolicyResolver(repo=_FakeRepo())

    policy = await resolver.resolve_for_context(
        user_id=7,
        metadata={
            "mcp_policy_context_enabled": True,
            "group_id": "ops",
            "persona_id": "researcher",
        },
    )

    assert policy["enabled"] is True
    assert policy["allowed_tools"] == [
        "notes.search",
        "external.servers.list",
        "Bash(git *)",
    ]
    assert policy["denied_tools"] == ["external.tools.refresh"]
    assert policy["capabilities"] == ["filesystem.read", "network.external"]
    assert policy["approval_mode"] == "ask_every_time"
    # resolved_policy_document gains setdefault keys (tool_tier_overrides, conditions)
    for key in policy["authored_policy_document"]:
        assert policy["authored_policy_document"][key] == policy["policy_document"][key]
    assert policy["resolved_policy_document"] == policy["policy_document"]
    assert policy["policy_document"]["path_scope_mode"] == "cwd_descendants"
    assert policy["policy_document"]["path_scope_enforcement"] == "approval_required_when_unenforceable"
    assert [source["assignment_id"] for source in policy["sources"]] == [10, 11, 12]


@pytest.mark.asyncio
async def test_policy_resolver_returns_disabled_policy_when_no_assignments_apply() -> None:
    from tldw_Server_API.app.services.mcp_hub_policy_resolver import McpHubPolicyResolver

    repo = _FakeRepo()
    repo.assignments = []
    resolver = McpHubPolicyResolver(repo=repo)

    policy = await resolver.resolve_for_context(
        user_id=99,
        metadata={"mcp_policy_context_enabled": True, "persona_id": "unknown"},
    )

    assert policy["enabled"] is False
    assert policy["allowed_tools"] == []
    assert policy["denied_tools"] == []
    assert policy["sources"] == []


@pytest.mark.asyncio
async def test_policy_resolver_ignores_superseded_governance_pack_objects() -> None:
    from tldw_Server_API.app.services.mcp_hub_policy_resolver import McpHubPolicyResolver

    repo = _FakeRepo()
    repo.governance_packs = {
        ("legacy-pack", "0.9.0", "user", 7): {
            "id": 90,
            "is_active_install": False,
        },
    }
    repo.assignments.append(
        {
            "id": 13,
            "target_type": "group",
            "target_id": "ops",
            "owner_scope_type": "user",
            "owner_scope_id": 7,
            "profile_id": None,
            "path_scope_object_id": None,
            "inline_policy_document": {
                "allowed_tools": ["legacy.assignment"],
                "governance_pack": {
                    "pack_id": "legacy-pack",
                    "pack_version": "0.9.0",
                    "source_object_id": "legacy.assignment",
                },
            },
            "approval_policy_id": None,
            "is_active": True,
        }
    )
    repo.profiles[3] = {
        "id": 3,
        "name": "Legacy Profile",
        "owner_scope_type": "user",
        "owner_scope_id": 7,
        "policy_document": {
            "allowed_tools": ["legacy.profile"],
            "governance_pack": {
                "pack_id": "legacy-pack",
                "pack_version": "0.9.0",
                "source_object_id": "legacy.profile",
            },
        },
        "path_scope_object_id": None,
        "is_active": True,
    }
    repo.assignments.append(
        {
            "id": 14,
            "target_type": "persona",
            "target_id": "researcher",
            "owner_scope_type": "user",
            "owner_scope_id": 7,
            "profile_id": 3,
            "path_scope_object_id": None,
            "inline_policy_document": {},
            "approval_policy_id": 55,
            "is_active": True,
        }
    )
    repo.approval_policies[55] = {
        "id": 55,
        "name": "Legacy Approval",
        "owner_scope_type": "user",
        "owner_scope_id": 7,
        "mode": "ask_every_time",
        "rules": {
            "governance_pack": {
                "pack_id": "legacy-pack",
                "pack_version": "0.9.0",
                "source_object_id": "legacy.approval",
            }
        },
        "is_active": True,
    }
    resolver = McpHubPolicyResolver(repo=repo)

    policy = await resolver.resolve_for_context(
        user_id=7,
        metadata={
            "mcp_policy_context_enabled": True,
            "group_id": "ops",
            "persona_id": "researcher",
        },
    )

    assert "legacy.assignment" not in policy["allowed_tools"]
    assert "legacy.profile" not in policy["allowed_tools"]
    assert policy["approval_policy_id"] is None
    assert [source["assignment_id"] for source in policy["sources"]] == [10, 11, 12, 14]


@pytest.mark.asyncio
async def test_policy_resolver_allows_assignment_override_to_replace_path_scope_mode() -> None:
    from tldw_Server_API.app.services.mcp_hub_policy_resolver import McpHubPolicyResolver

    repo = _FakeRepo()
    repo.overrides[12] = {
        "id": 99,
        "assignment_id": 12,
        "override_policy_document": {"path_scope_mode": "workspace_root"},
        "is_active": True,
    }
    resolver = McpHubPolicyResolver(repo=repo)

    policy = await resolver.resolve_for_context(
        user_id=7,
        metadata={
            "mcp_policy_context_enabled": True,
            "group_id": "ops",
            "persona_id": "researcher",
        },
    )

    assert policy["policy_document"]["path_scope_mode"] == "workspace_root"
    assert any(
        entry["field"] == "path_scope_mode"
        and entry["source_kind"] == "assignment_override"
        and entry["effect"] == "replaced"
        for entry in policy["provenance"]
    )


@pytest.mark.asyncio
async def test_policy_resolver_allows_null_override_to_clear_inherited_field() -> None:
    from tldw_Server_API.app.services.mcp_hub_policy_resolver import McpHubPolicyResolver

    repo = _FakeRepo()
    repo.overrides[10] = {
        "id": 501,
        "assignment_id": 10,
        "override_policy_document": {"path_scope_mode": None},
        "is_active": True,
    }
    resolver = McpHubPolicyResolver(repo=repo)

    policy = await resolver.resolve_for_context(
        user_id=7,
        metadata={"mcp_policy_context_enabled": True},
    )

    assert "path_scope_mode" in policy["policy_document"]
    assert policy["policy_document"]["path_scope_mode"] is None


@pytest.mark.asyncio
async def test_policy_resolver_replaces_path_allowlist_prefixes_in_assignment_override() -> None:
    from tldw_Server_API.app.services.mcp_hub_policy_resolver import McpHubPolicyResolver

    repo = _FakeRepo()
    repo.overrides[12] = {
        "id": 99,
        "assignment_id": 12,
        "override_policy_document": {"path_allowlist_prefixes": ["docs/api"]},
        "is_active": True,
    }
    resolver = McpHubPolicyResolver(repo=repo)

    policy = await resolver.resolve_for_context(
        user_id=7,
        metadata={
            "mcp_policy_context_enabled": True,
            "group_id": "ops",
            "persona_id": "researcher",
        },
    )

    assert policy["policy_document"]["path_allowlist_prefixes"] == ["docs/api"]
    assert any(
        entry["field"] == "path_allowlist_prefixes"
        and entry["source_kind"] == "assignment_override"
        and entry["effect"] == "replaced"
        for entry in policy["provenance"]
    )


@pytest.mark.asyncio
async def test_policy_resolver_applies_path_scope_object_layers_before_inline_and_override() -> None:
    from tldw_Server_API.app.services.mcp_hub_policy_resolver import McpHubPolicyResolver

    repo = _FakeRepo()
    repo.profiles[1]["policy_document"] = {
        "allowed_tools": ["notes.search"],
        "capabilities": ["filesystem.read"],
    }
    repo.profiles[1]["path_scope_object_id"] = 100
    repo.assignments[2]["path_scope_object_id"] = 101
    repo.assignments[2]["inline_policy_document"] = {
        "allowed_tools": ["Bash(git *)"],
        "approval_mode": "ask_every_time",
        "path_allowlist_prefixes": ["persona"],
    }
    resolver = McpHubPolicyResolver(repo=repo)

    policy = await resolver.resolve_for_context(
        user_id=7,
        metadata={
            "mcp_policy_context_enabled": True,
            "group_id": "ops",
            "persona_id": "researcher",
        },
    )

    assert policy["policy_document"]["path_scope_mode"] == "cwd_descendants"
    assert policy["policy_document"]["path_allowlist_prefixes"] == ["persona"]
    assert any(
        entry["field"] == "path_scope_mode"
        and entry["source_kind"] == "assignment_path_scope_object"
        for entry in policy["provenance"]
    )


@pytest.mark.asyncio
async def test_policy_resolver_returns_authored_and_resolved_documents_when_mapping_applies() -> None:
    from tldw_Server_API.app.services.mcp_hub_policy_resolver import McpHubPolicyResolver

    repo = _FakeRepo()
    repo.profiles[1]["policy_document"] = {
        "allowed_tools": ["notes.search"],
        "capabilities": ["tool.invoke.research"],
        "path_scope_enforcement": "approval_required_when_unenforceable",
    }
    repo.capability_mappings[("global", None, "tool.invoke.research")] = {
        "mapping_id": "research.global",
        "owner_scope_type": "global",
        "owner_scope_id": None,
        "capability_name": "tool.invoke.research",
        "resolved_policy_document": {
            "allowed_tools": ["web.search"],
            "approval_mode": "allow_silently",
            "path_scope_mode": "workspace_root",
        },
        "supported_environment_requirements": ["workspace_bounded_read"],
        "is_active": True,
    }
    resolver = McpHubPolicyResolver(repo=repo)

    policy = await resolver.resolve_for_context(
        user_id=7,
        metadata={
            "mcp_policy_context_enabled": True,
            "group_id": "ops",
            "persona_id": "researcher",
        },
    )

    assert policy["authored_policy_document"]["allowed_tools"] == [
        "notes.search",
        "external.servers.list",
        "Bash(git *)",
    ]
    assert policy["authored_policy_document"]["capabilities"] == [
        "tool.invoke.research",
        "network.external",
    ]
    assert policy["resolved_policy_document"]["allowed_tools"] == [
        "notes.search",
        "external.servers.list",
        "Bash(git *)",
        "web.search",
    ]
    assert policy["resolved_policy_document"]["approval_mode"] == "ask_every_time"
    assert policy["resolved_policy_document"]["path_scope_mode"] == "cwd_descendants"
    assert policy["allowed_tools"] == [
        "notes.search",
        "external.servers.list",
        "Bash(git *)",
        "web.search",
    ]
    assert policy["resolved_capabilities"] == ["tool.invoke.research"]
    assert policy["unresolved_capabilities"] == ["network.external"]
    assert policy["capability_mapping_summary"] == [
        {
            "capability_name": "tool.invoke.research",
            "resolution_intent": "allow",
            "mapping_id": "research.global",
            "mapping_scope_type": "global",
            "mapping_scope_id": None,
            "resolved_effects": {
                "allowed_tools": ["web.search"],
                "approval_mode": "allow_silently",
                "path_scope_mode": "workspace_root",
            },
            "supported_environment_requirements": ["workspace_bounded_read"],
            "unsupported_environment_requirements": [],
        }
    ]
    assert any(
        entry["source_kind"] == "capability_mapping"
        and entry["capability_name"] == "tool.invoke.research"
        and entry["mapping_id"] == "research.global"
        and entry["effect"] == "merged"
        for entry in policy["provenance"]
    )


@pytest.mark.asyncio
async def test_policy_resolver_keeps_unresolved_capabilities_visible_without_grants() -> None:
    from tldw_Server_API.app.services.mcp_hub_policy_resolver import McpHubPolicyResolver

    repo = _FakeRepo()
    repo.assignments = [repo.assignments[0]]
    repo.profiles[1]["policy_document"] = {
        "capabilities": ["tool.invoke.unmapped"],
    }
    resolver = McpHubPolicyResolver(repo=repo)

    policy = await resolver.resolve_for_context(
        user_id=7,
        metadata={"mcp_policy_context_enabled": True},
    )

    assert policy["allowed_tools"] == []
    assert policy["authored_policy_document"] == {"capabilities": ["tool.invoke.unmapped"]}
    assert policy["resolved_policy_document"] == {
        "capabilities": ["tool.invoke.unmapped"],
        "tool_tier_overrides": {},
        "conditions": {},
    }
    assert policy["resolved_capabilities"] == []
    assert policy["unresolved_capabilities"] == ["tool.invoke.unmapped"]
    assert policy["capability_warnings"] == [
        "No active capability adapter mapping found for 'tool.invoke.unmapped'"
    ]
    assert any(
        entry["source_kind"] == "capability_mapping"
        and entry["capability_name"] == "tool.invoke.unmapped"
        and entry["effect"] == "blocked"
        for entry in policy["provenance"]
    )


@pytest.mark.asyncio
async def test_policy_resolver_turns_denied_capabilities_into_denied_tool_patterns() -> None:
    from tldw_Server_API.app.services.mcp_hub_policy_resolver import McpHubPolicyResolver

    repo = _FakeRepo()
    repo.assignments = [repo.assignments[0]]
    repo.profiles[1]["policy_document"] = {
        "capabilities": ["filesystem.read"],
        "denied_capabilities": ["tool.invoke.docs"],
    }
    repo.capability_mappings = {
        ("global", None, "filesystem.read"): {
            "mapping_id": "filesystem.read.global",
            "owner_scope_type": "global",
            "owner_scope_id": None,
            "capability_name": "filesystem.read",
            "resolved_policy_document": {"allowed_tools": ["files.read"]},
            "supported_environment_requirements": [],
            "is_active": True,
        },
        ("global", None, "tool.invoke.docs"): {
            "mapping_id": "docs.global",
            "owner_scope_type": "global",
            "owner_scope_id": None,
            "capability_name": "tool.invoke.docs",
            "resolved_policy_document": {"allowed_tools": ["docs.search"]},
            "supported_environment_requirements": [],
            "is_active": True,
        },
    }
    resolver = McpHubPolicyResolver(repo=repo)

    policy = await resolver.resolve_for_context(
        user_id=7,
        metadata={"mcp_policy_context_enabled": True},
    )

    assert policy["allowed_tools"] == ["files.read"]
    assert policy["denied_tools"] == ["docs.search"]
    assert sorted(summary["resolution_intent"] for summary in policy["capability_mapping_summary"]) == [
        "allow",
        "deny",
    ]
    assert any(
        entry["source_kind"] == "capability_mapping"
        and entry["capability_name"] == "tool.invoke.docs"
        and entry["effect"] == "narrowed"
        for entry in policy["provenance"]
    )
