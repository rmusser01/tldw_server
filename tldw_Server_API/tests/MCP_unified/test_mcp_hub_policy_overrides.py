from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ("tldw_Server_API.tests.AuthNZ.conftest",)


async def _make_repo(tmp_path, monkeypatch):
    from tldw_Server_API.app.core.AuthNZ.database import get_db_pool, reset_db_pool
    from tldw_Server_API.app.core.AuthNZ.migrations import ensure_authnz_tables
    from tldw_Server_API.app.core.AuthNZ.repos.mcp_hub_repo import McpHubRepo
    from tldw_Server_API.app.core.AuthNZ.settings import reset_settings

    db_path = tmp_path / "users.db"
    monkeypatch.setenv("AUTH_MODE", "multi_user")
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    reset_settings()
    await reset_db_pool()

    pool = await get_db_pool()
    ensure_authnz_tables(Path(str(db_path)))

    repo = McpHubRepo(pool)
    await repo.ensure_tables()
    return repo


@pytest.mark.asyncio
async def test_repo_policy_override_is_one_to_one_and_enriches_assignment_summary(
    tmp_path, monkeypatch
) -> None:
    repo = await _make_repo(tmp_path, monkeypatch)

    assignment = await repo.create_policy_assignment(
        target_type="persona",
        target_id="researcher",
        owner_scope_type="user",
        owner_scope_id=7,
        profile_id=None,
        inline_policy_document={"approval_mode": "ask_every_time"},
        approval_policy_id=None,
        actor_id=7,
        is_active=True,
    )

    created = await repo.upsert_policy_override(
        assignment_id=int(assignment["id"]),
        override_policy_document={
            "allowed_tools": ["remote.fetch"],
            "approval_mode": "ask_outside_profile",
        },
        broadens_access=True,
        grant_authority_snapshot={"permissions": ["grant.tool.invoke"]},
        actor_id=7,
        is_active=True,
    )

    updated = await repo.upsert_policy_override(
        assignment_id=int(assignment["id"]),
        override_policy_document={"denied_tools": ["sandbox.run"]},
        broadens_access=False,
        grant_authority_snapshot={"permissions": []},
        actor_id=8,
        is_active=False,
    )

    assert int(updated["id"]) == int(created["id"])
    assert updated["override_policy_document"] == {"denied_tools": ["sandbox.run"]}
    assert updated["is_active"] is False

    listed = await repo.list_policy_assignments(owner_scope_type="user", owner_scope_id=7)
    assert len(listed) == 1
    assert listed[0]["has_override"] is True
    assert int(listed[0]["override_id"]) == int(created["id"])
    assert listed[0]["override_active"] is False
    assert listed[0]["override_updated_at"] is not None


@pytest.mark.asyncio
async def test_repo_delete_policy_assignment_removes_override(tmp_path, monkeypatch) -> None:
    repo = await _make_repo(tmp_path, monkeypatch)

    assignment = await repo.create_policy_assignment(
        target_type="persona",
        target_id="researcher",
        owner_scope_type="user",
        owner_scope_id=7,
        profile_id=None,
        inline_policy_document={},
        approval_policy_id=None,
        actor_id=7,
        is_active=True,
    )
    await repo.upsert_policy_override(
        assignment_id=int(assignment["id"]),
        override_policy_document={"allowed_tools": ["remote.fetch"]},
        broadens_access=True,
        grant_authority_snapshot={"permissions": ["grant.tool.invoke"]},
        actor_id=7,
        is_active=True,
    )

    deleted = await repo.delete_policy_assignment(int(assignment["id"]))

    assert deleted is True
    assert await repo.get_policy_override_by_assignment(int(assignment["id"])) is None


class _ResolverRepo:
    def __init__(self, *, override_active: bool) -> None:
        self.profiles = {
            1: {
                "id": 1,
                "name": "Base Research",
                "is_active": True,
                "policy_document": {
                    "allowed_tools": ["notes.search"],
                    "capabilities": ["filesystem.read"],
                },
            }
        }
        self.assignments = [
            {
                "id": 12,
                "target_type": "persona",
                "target_id": "researcher",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "profile_id": 1,
                "inline_policy_document": {
                    "allowed_tools": ["Bash(git *)"],
                    "approval_mode": "ask_every_time",
                },
                "approval_policy_id": None,
                "is_active": True,
            }
        ]
        self.overrides = {
            12: {
                "id": 101,
                "assignment_id": 12,
                "override_policy_document": {
                    "allowed_tools": ["remote.fetch"],
                    "approval_mode": "ask_outside_profile",
                },
                "is_active": override_active,
            }
        }

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

    async def list_policy_assignment_workspaces(self, assignment_id: int) -> list[dict]:  # noqa: ARG002
        return []

    async def find_active_capability_mapping(
        self,
        *,
        owner_scope_type: str,  # noqa: ARG002
        owner_scope_id: int | None,  # noqa: ARG002
        capability_name: str,  # noqa: ARG002
    ) -> dict | None:
        return None


@pytest.mark.asyncio
async def test_policy_resolver_applies_assignment_override_and_emits_provenance() -> None:
    from tldw_Server_API.app.services.mcp_hub_policy_resolver import McpHubPolicyResolver

    resolver = McpHubPolicyResolver(repo=_ResolverRepo(override_active=True))

    policy = await resolver.resolve_for_context(
        user_id=7,
        metadata={"mcp_policy_context_enabled": True, "persona_id": "researcher"},
    )

    assert policy["enabled"] is True
    assert policy["allowed_tools"] == ["notes.search", "Bash(git *)", "remote.fetch"]
    assert policy["capabilities"] == ["filesystem.read"]
    assert policy["approval_mode"] == "ask_outside_profile"
    expected_authored_policy = {
        "allowed_tools": ["notes.search", "Bash(git *)", "remote.fetch"],
        "capabilities": ["filesystem.read"],
        "approval_mode": "ask_outside_profile",
    }
    assert policy["authored_policy_document"] == expected_authored_policy
    assert policy["resolved_policy_document"] == {
        **expected_authored_policy,
        "conditions": {},
        "tool_tier_overrides": {},
    }
    assert policy["sources"] == [
        {
            "assignment_id": 12,
            "target_type": "persona",
            "target_id": "researcher",
            "owner_scope_type": "user",
            "owner_scope_id": 7,
            "profile_id": 1,
            "path_scope_object_id": None,
        }
    ]
    assert policy["provenance"] == [
        {
            "field": "allowed_tools",
            "value": ["notes.search"],
            "source_kind": "profile",
            "assignment_id": 12,
            "profile_id": 1,
            "override_id": None,
            "effect": "merged",
        },
        {
            "field": "capabilities",
            "value": ["filesystem.read"],
            "source_kind": "profile",
            "assignment_id": 12,
            "profile_id": 1,
            "override_id": None,
            "effect": "merged",
        },
        {
            "field": "allowed_tools",
            "value": ["Bash(git *)"],
            "source_kind": "assignment_inline",
            "assignment_id": 12,
            "profile_id": 1,
            "override_id": None,
            "effect": "merged",
        },
        {
            "field": "approval_mode",
            "value": "ask_every_time",
            "source_kind": "assignment_inline",
            "assignment_id": 12,
            "profile_id": 1,
            "override_id": None,
            "effect": "replaced",
        },
        {
            "field": "allowed_tools",
            "value": ["remote.fetch"],
            "source_kind": "assignment_override",
            "assignment_id": 12,
            "profile_id": 1,
            "override_id": 101,
            "effect": "merged",
        },
        {
            "field": "approval_mode",
            "value": "ask_outside_profile",
            "source_kind": "assignment_override",
            "assignment_id": 12,
            "profile_id": 1,
            "override_id": 101,
            "effect": "replaced",
        },
        {
            "field": "capabilities",
            "value": "filesystem.read",
            "source_kind": "capability_mapping",
            "assignment_id": None,
            "profile_id": None,
            "override_id": None,
            "capability_name": "filesystem.read",
            "mapping_id": None,
            "mapping_scope_type": None,
            "mapping_scope_id": None,
            "resolved_effects": {},
            "resolution_intent": "allow",
            "effect": "blocked",
        },
    ]


@pytest.mark.asyncio
async def test_policy_resolver_ignores_inactive_assignment_override() -> None:
    from tldw_Server_API.app.services.mcp_hub_policy_resolver import McpHubPolicyResolver

    resolver = McpHubPolicyResolver(repo=_ResolverRepo(override_active=False))

    policy = await resolver.resolve_for_context(
        user_id=7,
        metadata={"mcp_policy_context_enabled": True, "persona_id": "researcher"},
    )

    assert policy["enabled"] is True
    assert policy["allowed_tools"] == ["notes.search", "Bash(git *)"]
    assert policy["approval_mode"] == "ask_every_time"
    assert policy["unresolved_capabilities"] == ["filesystem.read"]
    assert not any(
        entry["source_kind"] == "assignment_override" for entry in policy["provenance"]
    )
