from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tldw_Server_API.app.core.exceptions import BadRequestError


def _service_normalized_root(value: str) -> str:
    return str(Path(value).expanduser().resolve(strict=False))


class _FakeRepo:
    def __init__(self) -> None:
        self.permission_profiles = {
            5: {
                "id": 5,
                "name": "Workspace Root Profile",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "path_scope_object_id": 41,
                "policy_document": {},
                "is_active": True,
            }
        }
        self.path_scope_objects = {
            41: {
                "id": 41,
                "name": "Workspace Root",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "path_scope_document": {
                    "path_scope_mode": "workspace_root",
                    "path_scope_enforcement": "approval_required_when_unenforceable",
                },
                "is_active": True,
            }
        }
        self.workspace_set_objects = {
            51: {
                "id": 51,
                "name": "Named Workspaces",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "is_active": True,
            },
            61: {
                "id": 61,
                "name": "Team Shared Workspaces",
                "owner_scope_type": "team",
                "owner_scope_id": 21,
                "is_active": True,
            },
        }
        self.workspace_set_members = {
            51: ["workspace-alpha", "workspace-beta"],
            61: ["shared-alpha", "shared-beta"],
        }
        self.assignments = {
            11: {
                "id": 11,
                "target_type": "persona",
                "target_id": "researcher",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "profile_id": None,
                "path_scope_object_id": None,
                "workspace_source_mode": "inline",
                "workspace_set_object_id": None,
                "inline_policy_document": {
                    "path_scope_mode": "workspace_root",
                },
                "is_active": True,
            },
            12: {
                "id": 12,
                "target_type": "persona",
                "target_id": "cwd-only",
                "owner_scope_type": "user",
                "owner_scope_id": 7,
                "profile_id": None,
                "path_scope_object_id": None,
                "workspace_source_mode": "inline",
                "workspace_set_object_id": None,
                "inline_policy_document": {
                    "path_scope_mode": "cwd_descendants",
                },
                "is_active": True,
            },
        }
        self.assignment_workspaces = {
            11: ["workspace-alpha", "workspace-missing"],
            12: ["workspace-alpha", "workspace-beta"],
        }

    async def get_permission_profile(self, profile_id: int) -> dict[str, Any] | None:
        return self.permission_profiles.get(profile_id)

    async def get_path_scope_object(self, path_scope_object_id: int) -> dict[str, Any] | None:
        return self.path_scope_objects.get(path_scope_object_id)

    async def get_workspace_set_object(self, workspace_set_object_id: int) -> dict[str, Any] | None:
        return self.workspace_set_objects.get(workspace_set_object_id)

    async def list_workspace_set_members(self, workspace_set_object_id: int) -> list[dict[str, Any]]:
        return [
            {
                "workspace_set_object_id": workspace_set_object_id,
                "workspace_id": workspace_id,
            }
            for workspace_id in self.workspace_set_members.get(workspace_set_object_id, [])
        ]

    async def get_policy_assignment(self, assignment_id: int) -> dict[str, Any] | None:
        return self.assignments.get(assignment_id)

    async def list_policy_assignment_workspaces(self, assignment_id: int) -> list[dict[str, Any]]:
        return [
            {
                "assignment_id": assignment_id,
                "workspace_id": workspace_id,
            }
            for workspace_id in self.assignment_workspaces.get(assignment_id, [])
        ]

    async def get_policy_override_by_assignment(self, assignment_id: int) -> dict[str, Any] | None:  # noqa: ARG002
        return None


class _FakeWorkspaceRootResolver:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.results = {
            ("user_local", "workspace-alpha"): {
                "workspace_root": "/repo",
                "workspace_id": "workspace-alpha",
                "source": "sandbox_workspace_lookup",
                "reason": None,
            },
            ("user_local", "workspace-beta"): {
                "workspace_root": "/repo/docs",
                "workspace_id": "workspace-beta",
                "source": "sandbox_workspace_lookup",
                "reason": None,
            },
            ("user_local", "workspace-missing"): {
                "workspace_root": None,
                "workspace_id": "workspace-missing",
                "source": "sandbox_workspace_lookup",
                "reason": "workspace_root_unavailable",
            },
            ("shared_registry", "shared-alpha"): {
                "workspace_root": "/srv/shared/a",
                "workspace_id": "shared-alpha",
                "source": "shared_registry",
                "reason": None,
            },
            ("shared_registry", "shared-beta"): {
                "workspace_root": "/srv/shared/b",
                "workspace_id": "shared-beta",
                "source": "shared_registry",
                "reason": None,
            },
        }

    async def resolve_for_context(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        trust_source = str(kwargs.get("workspace_trust_source") or "user_local")
        workspace_id = str(kwargs.get("workspace_id") or "")
        return dict(
            self.results.get(
                (trust_source, workspace_id),
                {
                    "workspace_root": None,
                    "workspace_id": workspace_id,
                    "source": trust_source,
                    "reason": "workspace_root_unavailable",
                },
            )
        )


@pytest.mark.asyncio
async def test_validate_multi_root_assignment_readiness_rejects_overlapping_named_roots_from_inherited_workspace_root() -> None:
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

    repo = _FakeRepo()
    resolver = _FakeWorkspaceRootResolver()
    svc = McpHubService(repo=repo, workspace_root_resolver=resolver)

    with pytest.raises(BadRequestError) as exc_info:
        await svc.validate_multi_root_assignment_readiness(
            actor_id=7,
            assignment_id=None,
            owner_scope_type="user",
            owner_scope_id=7,
            profile_id=5,
            path_scope_object_id=None,
            inline_policy_document={},
            workspace_source_mode="named",
            workspace_set_object_id=51,
            inline_workspace_ids=None,
        )

    assert getattr(exc_info.value, "detail", {}) == {
        "code": "assignment_multi_root_overlap",
        "message": "Named workspace source contains overlapping roots for multi-root execution.",
        "conflicting_workspace_ids": ["workspace-alpha", "workspace-beta"],
        "conflicting_workspace_roots": [
            _service_normalized_root("/repo"),
            _service_normalized_root("/repo/docs"),
        ],
        "workspace_source_mode": "named",
        "workspace_trust_source": "user_local",
    }


@pytest.mark.asyncio
async def test_validate_multi_root_assignment_readiness_rejects_unresolved_inline_workspaces() -> None:
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

    repo = _FakeRepo()
    resolver = _FakeWorkspaceRootResolver()
    svc = McpHubService(repo=repo, workspace_root_resolver=resolver)

    with pytest.raises(BadRequestError) as exc_info:
        await svc.validate_multi_root_assignment_readiness(
            actor_id=7,
            assignment_id=11,
            owner_scope_type="user",
            owner_scope_id=7,
            profile_id=None,
            path_scope_object_id=None,
            inline_policy_document={"path_scope_mode": "workspace_root"},
            workspace_source_mode="inline",
            workspace_set_object_id=None,
            inline_workspace_ids=None,
        )

    assert getattr(exc_info.value, "detail", {}) == {
        "code": "assignment_workspace_unresolvable",
        "message": "Workspace source cannot resolve every workspace for multi-root execution.",
        "unresolved_workspace_ids": ["workspace-missing"],
        "workspace_source_mode": "inline",
        "workspace_trust_source": "user_local",
    }


@pytest.mark.asyncio
async def test_validate_multi_root_assignment_readiness_skips_overlap_checks_for_cwd_descendants_mode() -> None:
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

    repo = _FakeRepo()
    resolver = _FakeWorkspaceRootResolver()
    svc = McpHubService(repo=repo, workspace_root_resolver=resolver)

    await svc.validate_multi_root_assignment_readiness(
        actor_id=7,
        assignment_id=12,
        owner_scope_type="user",
        owner_scope_id=7,
        profile_id=None,
        path_scope_object_id=None,
        inline_policy_document={"path_scope_mode": "cwd_descendants"},
        workspace_source_mode="inline",
        workspace_set_object_id=None,
        inline_workspace_ids=None,
    )

    assert resolver.calls == []


@pytest.mark.asyncio
async def test_validate_multi_root_assignment_readiness_uses_shared_registry_for_shared_named_sets() -> None:
    from tldw_Server_API.app.services.mcp_hub_service import McpHubService

    repo = _FakeRepo()
    resolver = _FakeWorkspaceRootResolver()
    svc = McpHubService(repo=repo, workspace_root_resolver=resolver)

    await svc.validate_multi_root_assignment_readiness(
        actor_id=7,
        assignment_id=None,
        owner_scope_type="team",
        owner_scope_id=21,
        profile_id=None,
        path_scope_object_id=41,
        inline_policy_document={},
        workspace_source_mode="named",
        workspace_set_object_id=61,
        inline_workspace_ids=None,
    )

    assert resolver.calls
    assert {call["workspace_trust_source"] for call in resolver.calls} == {"shared_registry"}
