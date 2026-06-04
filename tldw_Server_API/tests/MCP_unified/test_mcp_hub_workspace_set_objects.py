from __future__ import annotations

import pytest


class _FakeRepo:
    def __init__(self) -> None:
        self.assignments = [
            {
                "id": 12,
                "target_type": "persona",
                "target_id": "researcher",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "profile_id": None,
                "path_scope_object_id": None,
                "workspace_source_mode": "named",
                "workspace_set_object_id": 501,
                "inline_policy_document": {
                    "allowed_tools": ["files.read"],
                    "path_scope_mode": "workspace_root",
                    "path_scope_enforcement": "approval_required_when_unenforceable",
                },
                "is_active": True,
            }
        ]
        self.assignment_workspaces = {
            12: ["workspace-inline"],
        }
        self.workspace_set_objects = {
            501: {
                "id": 501,
                "name": "Research Workspaces",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "is_active": True,
            }
        }
        self.workspace_set_members = {
            501: ["workspace-alpha", "workspace-beta"],
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

    async def get_permission_profile(self, profile_id: int) -> dict | None:  # noqa: ARG002
        return None

    async def get_policy_override_by_assignment(self, assignment_id: int) -> dict | None:  # noqa: ARG002
        return None

    async def get_path_scope_object(self, path_scope_object_id: int) -> dict | None:  # noqa: ARG002
        return None

    async def list_policy_assignment_workspaces(self, assignment_id: int) -> list[dict]:
        return [
            {"assignment_id": assignment_id, "workspace_id": workspace_id}
            for workspace_id in self.assignment_workspaces.get(assignment_id, [])
        ]

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


@pytest.mark.asyncio
async def test_policy_resolver_uses_named_workspace_set_members_over_preserved_inline_rows() -> None:
    from tldw_Server_API.app.services.mcp_hub_policy_resolver import McpHubPolicyResolver

    resolver = McpHubPolicyResolver(repo=_FakeRepo())

    policy = await resolver.resolve_for_context(
        user_id=7,
        metadata={
            "mcp_policy_context_enabled": True,
            "persona_id": "researcher",
        },
    )

    assert policy["selected_workspace_source_mode"] == "named"
    assert policy["selected_workspace_set_object_id"] == 501
    assert policy["selected_workspace_set_object_name"] == "Research Workspaces"
    assert policy["selected_assignment_workspace_ids"] == ["workspace-alpha", "workspace-beta"]
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
