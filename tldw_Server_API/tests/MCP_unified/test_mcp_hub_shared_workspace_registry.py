from __future__ import annotations

import pytest


class _FakeRepo:
    def __init__(self) -> None:
        self.assignments = [
            {
                "id": 18,
                "target_type": "group",
                "target_id": "team-red",
                "owner_scope_type": "team",
                "owner_scope_id": 21,
                "profile_id": None,
                "path_scope_object_id": None,
                "workspace_source_mode": "named",
                "workspace_set_object_id": 601,
                "inline_policy_document": {
                    "allowed_tools": ["files.read"],
                    "path_scope_mode": "workspace_root",
                    "path_scope_enforcement": "approval_required_when_unenforceable",
                },
                "is_active": True,
            }
        ]
        self.workspace_set_objects = {
            601: {
                "id": 601,
                "name": "Team Workspaces",
                "owner_scope_type": "team",
                "owner_scope_id": 21,
                "is_active": True,
            }
        }
        self.workspace_set_members = {
            601: ["shared-docs"],
        }
        self.shared_workspace_entries = [
            {
                "id": 71,
                "workspace_id": "shared-docs",
                "display_name": "Shared Docs",
                "absolute_root": "/srv/shared/docs",
                "owner_scope_type": "team",
                "owner_scope_id": 21,
                "is_active": True,
            }
        ]

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

    async def get_permission_profile(self, profile_id: int) -> dict | None:  # noqa: ARG002
        return None

    async def get_policy_override_by_assignment(self, assignment_id: int) -> dict | None:  # noqa: ARG002
        return None

    async def get_path_scope_object(self, path_scope_object_id: int) -> dict | None:  # noqa: ARG002
        return None

    async def list_policy_assignment_workspaces(self, assignment_id: int) -> list[dict]:
        return []

    async def get_workspace_set_object(self, workspace_set_object_id: int) -> dict | None:
        return self.workspace_set_objects.get(workspace_set_object_id)

    async def list_workspace_set_members(self, workspace_set_object_id: int) -> list[dict]:
        return [
            {
                "workspace_set_object_id": workspace_set_object_id,
                "workspace_id": workspace_id,
            }
            for workspace_id in self.workspace_set_members.get(workspace_set_object_id, [])
        ]

    async def list_shared_workspace_entries(
        self,
        *,
        workspace_id: str | None = None,
        owner_scope_type: str | None = None,
        owner_scope_id: int | None = None,
    ) -> list[dict]:
        rows = list(self.shared_workspace_entries)
        if workspace_id is not None:
            rows = [row for row in rows if row["workspace_id"] == workspace_id]
        if owner_scope_type is not None:
            rows = [row for row in rows if row["owner_scope_type"] == owner_scope_type]
        if owner_scope_id is not None:
            rows = [row for row in rows if row["owner_scope_id"] == owner_scope_id]
        return rows


@pytest.mark.asyncio
async def test_policy_resolver_marks_shared_workspace_set_as_shared_registry_source() -> None:
    from tldw_Server_API.app.services.mcp_hub_policy_resolver import McpHubPolicyResolver

    resolver = McpHubPolicyResolver(repo=_FakeRepo())

    policy = await resolver.resolve_for_context(
        user_id=7,
        metadata={
            "mcp_policy_context_enabled": True,
            "group_id": "team-red",
            "team_id": 21,
        },
    )

    assert policy["selected_workspace_source_mode"] == "named"
    assert policy["selected_workspace_set_object_id"] == 601
    assert policy["selected_workspace_set_object_name"] == "Team Workspaces"
    assert policy["selected_workspace_trust_source"] == "shared_registry"
    assert policy["selected_assignment_workspace_ids"] == ["shared-docs"]
    assert policy["authored_policy_document"] == {
        "allowed_tools": ["files.read"],
        "path_scope_mode": "workspace_root",
        "path_scope_enforcement": "approval_required_when_unenforceable",
    }
    assert policy["resolved_policy_document"] == {
        "allowed_tools": ["files.read"],
        "conditions": {},
        "path_scope_mode": "workspace_root",
        "path_scope_enforcement": "approval_required_when_unenforceable",
        "tool_tier_overrides": {},
    }
