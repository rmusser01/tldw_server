from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp_unified.interfaces.path_scope import PathScopeCandidate


class _FakePathScopeService:
    def __init__(self, result: dict) -> None:
        self.result = dict(result)

    async def resolve_for_context(self, *, effective_policy, context):  # noqa: ANN001
        return dict(self.result)


class _FakeMultiRootPathService:
    def __init__(self, workspace_root: str, workspace_id: str = "ws-1") -> None:
        self.workspace_root = workspace_root
        self.workspace_id = workspace_id
        self.calls: list[dict] = []

    async def resolve_path_bundle(self, **kwargs):  # noqa: ANN001
        self.calls.append(dict(kwargs))
        normalized_paths = [
            str((Path(self.workspace_root) / raw_path).resolve()) for raw_path in kwargs.get("raw_paths", [])
        ]
        return {
            "ok": True,
            "reason": None,
            "normalized_paths": normalized_paths,
            "workspace_bundle_ids": [self.workspace_id],
            "workspace_bundle_roots": [self.workspace_root],
            "path_workspace_map": dict.fromkeys(normalized_paths, self.workspace_id),
            "resolved_workspace_roots_by_id": {self.workspace_id: self.workspace_root},
        }


def _workspace_scope() -> dict:
    return {
        "enabled": True,
        "path_scope_mode": "workspace_root",
        "path_scope_enforcement": "approval_required_when_unenforceable",
        "workspace_root": "/tmp/mcp-hub-path-enforcer/project",
        "cwd": "/tmp/mcp-hub-path-enforcer/project",
        "reason": None,
    }


def _filesystem_tool_def(*, action: str = "read") -> dict:
    return {
        "name": f"fs.{action}",
        "metadata": {
            "uses_filesystem": True,
            "path_boundable": True,
            "path_argument_hints": ["path"],
            "path_scope_action": action,
        },
    }


@pytest.mark.asyncio
async def test_path_enforcement_allows_path_boundable_tool_within_scope() -> None:
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )

    svc = McpHubPathEnforcementService(
        path_scope_service=_FakePathScopeService(
            {
                "enabled": True,
                "path_scope_mode": "workspace_root",
                "path_scope_enforcement": "approval_required_when_unenforceable",
                "workspace_root": "/tmp/mcp-hub-path-enforcer/project",
                "cwd": "/tmp/mcp-hub-path-enforcer/project/src",
                "reason": None,
            }
        )
    )
    workspace_root = "/tmp/mcp-hub-path-enforcer/project"
    cwd_root = str((Path(workspace_root).resolve() / "src").resolve())
    expected_path = str((Path(cwd_root) / "docs/readme.md").resolve())

    result = await svc.evaluate_tool_call(
        effective_policy={
            "enabled": True,
            "policy_document": {"path_scope_mode": "workspace_root"},
        },
        context=SimpleNamespace(metadata={"cwd": "src"}),
        tool_name="files.read",
        tool_args={"path": "docs/readme.md"},
        tool_def={
            "name": "files.read",
            "metadata": {
                "uses_filesystem": True,
                "path_boundable": True,
                "path_argument_hints": ["path"],
            },
        },
    )

    assert result["enabled"] is True
    assert result["within_scope"] is True
    assert result["reason"] is None
    assert result["force_approval"] is False
    assert result["normalized_paths"] == [expected_path]
    assert result["scope_payload"] == {
        "path_scope_mode": "workspace_root",
        "workspace_root": workspace_root,
        "scope_root": str(Path(workspace_root).resolve()),
        "normalized_paths": [expected_path],
    }


@pytest.mark.asyncio
async def test_path_enforcement_requires_approval_for_path_outside_cwd_scope() -> None:
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )

    svc = McpHubPathEnforcementService(
        path_scope_service=_FakePathScopeService(
            {
                "enabled": True,
                "path_scope_mode": "cwd_descendants",
                "path_scope_enforcement": "approval_required_when_unenforceable",
                "workspace_root": "/tmp/mcp-hub-path-enforcer/project",
                "cwd": "/tmp/mcp-hub-path-enforcer/project/src",
                "reason": None,
            }
        )
    )
    workspace_root = "/tmp/mcp-hub-path-enforcer/project"
    cwd_root = str((Path(workspace_root).resolve() / "src").resolve())
    expected_path = str((Path(cwd_root) / "../README.md").resolve())

    result = await svc.evaluate_tool_call(
        effective_policy={
            "enabled": True,
            "policy_document": {"path_scope_mode": "cwd_descendants"},
        },
        context=SimpleNamespace(metadata={"cwd": "src"}),
        tool_name="files.read",
        tool_args={"path": "../README.md"},
        tool_def={
            "name": "files.read",
            "metadata": {
                "uses_filesystem": True,
                "path_boundable": True,
                "path_argument_hints": ["path"],
            },
        },
    )

    assert result["enabled"] is True
    assert result["within_scope"] is False
    assert result["reason"] == "path_outside_current_folder_scope"
    assert result["force_approval"] is True
    assert result["normalized_paths"] == [expected_path]
    assert result["scope_payload"] == {
        "path_scope_mode": "cwd_descendants",
        "workspace_root": workspace_root,
        "scope_root": cwd_root,
        "normalized_paths": [expected_path],
        "reason": "path_outside_current_folder_scope",
    }


@pytest.mark.asyncio
async def test_path_enforcement_requires_approval_for_non_path_boundable_filesystem_tool() -> None:
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )

    svc = McpHubPathEnforcementService(
        path_scope_service=_FakePathScopeService(
            {
                "enabled": True,
                "path_scope_mode": "workspace_root",
                "path_scope_enforcement": "approval_required_when_unenforceable",
                "workspace_root": "/tmp/mcp-hub-path-enforcer/project",
                "cwd": "/tmp/mcp-hub-path-enforcer/project",
                "reason": None,
            }
        )
    )
    workspace_root = "/tmp/mcp-hub-path-enforcer/project"

    result = await svc.evaluate_tool_call(
        effective_policy={
            "enabled": True,
            "policy_document": {"path_scope_mode": "workspace_root"},
        },
        context=SimpleNamespace(metadata={}),
        tool_name="sandbox.run",
        tool_args={"files": [{"path": "src/app.py", "content_b64": "QUJD"}]},
        tool_def={
            "name": "sandbox.run",
            "metadata": {
                "uses_filesystem": True,
                "path_boundable": False,
                "path_argument_hints": ["files[].path"],
            },
        },
    )

    assert result["enabled"] is True
    assert result["within_scope"] is False
    assert result["reason"] == "tool_not_path_boundable"
    assert result["force_approval"] is True
    assert result["normalized_paths"] == []
    assert result["scope_payload"] == {
        "path_scope_mode": "workspace_root",
        "workspace_root": workspace_root,
        "scope_root": str(Path(workspace_root).resolve()),
        "reason": "tool_not_path_boundable",
    }


@pytest.mark.asyncio
async def test_path_enforcement_requires_candidate_to_match_allowlist_root() -> None:
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )

    svc = McpHubPathEnforcementService(
        path_scope_service=_FakePathScopeService(
            {
                "enabled": True,
                "path_scope_mode": "workspace_root",
                "path_scope_enforcement": "approval_required_when_unenforceable",
                "workspace_root": "/tmp/mcp-hub-path-enforcer/project",
                "cwd": "/tmp/mcp-hub-path-enforcer/project",
                "reason": None,
            }
        )
    )
    workspace_root = "/tmp/mcp-hub-path-enforcer/project"
    expected_path = str((Path(workspace_root).resolve() / "src2/notes.md").resolve())

    result = await svc.evaluate_tool_call(
        effective_policy={
            "enabled": True,
            "policy_document": {
                "path_scope_mode": "workspace_root",
                "path_allowlist_prefixes": ["src"],
            },
        },
        context=SimpleNamespace(metadata={}),
        tool_name="files.read",
        tool_args={"path": "src2/notes.md"},
        tool_def={
            "name": "files.read",
            "metadata": {
                "uses_filesystem": True,
                "path_boundable": True,
                "path_argument_hints": ["path"],
            },
        },
    )

    assert result["enabled"] is True
    assert result["within_scope"] is False
    assert result["reason"] == "path_outside_allowlist_scope"
    assert result["force_approval"] is True
    assert result["normalized_paths"] == [expected_path]
    assert result["scope_payload"] == {
        "path_scope_mode": "workspace_root",
        "workspace_root": workspace_root,
        "scope_root": str(Path(workspace_root).resolve()),
        "normalized_paths": [expected_path],
        "path_allowlist_prefixes": ["src"],
        "reason": "path_outside_allowlist_scope",
    }


@pytest.mark.asyncio
async def test_path_enforcement_allows_candidate_within_scope_and_allowlist_root() -> None:
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )

    svc = McpHubPathEnforcementService(
        path_scope_service=_FakePathScopeService(
            {
                "enabled": True,
                "path_scope_mode": "workspace_root",
                "path_scope_enforcement": "approval_required_when_unenforceable",
                "workspace_root": "/tmp/mcp-hub-path-enforcer/project",
                "cwd": "/tmp/mcp-hub-path-enforcer/project",
                "reason": None,
            }
        )
    )
    workspace_root = "/tmp/mcp-hub-path-enforcer/project"
    expected_path = str((Path(workspace_root).resolve() / "src/docs/readme.md").resolve())

    result = await svc.evaluate_tool_call(
        effective_policy={
            "enabled": True,
            "policy_document": {
                "path_scope_mode": "workspace_root",
                "path_allowlist_prefixes": ["src"],
            },
        },
        context=SimpleNamespace(metadata={}),
        tool_name="files.read",
        tool_args={"path": "src/docs/readme.md"},
        tool_def={
            "name": "files.read",
            "metadata": {
                "uses_filesystem": True,
                "path_boundable": True,
                "path_argument_hints": ["path"],
            },
        },
    )

    assert result["enabled"] is True
    assert result["within_scope"] is True
    assert result["reason"] is None
    assert result["force_approval"] is False
    assert result["normalized_paths"] == [expected_path]
    assert result["scope_payload"] == {
        "path_scope_mode": "workspace_root",
        "workspace_root": workspace_root,
        "scope_root": str(Path(workspace_root).resolve()),
        "normalized_paths": [expected_path],
        "path_allowlist_prefixes": ["src"],
    }


@pytest.mark.asyncio
async def test_path_grants_allow_candidate_action_and_report_safe_decision() -> None:
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )

    svc = McpHubPathEnforcementService(path_scope_service=_FakePathScopeService(_workspace_scope()))

    result = await svc.evaluate_tool_call(
        effective_policy={
            "enabled": True,
            "policy_document": {
                "path_scope_mode": "workspace_root",
                "path_grants": [
                    {"prefix": "documents", "actions": ["read", "edit", "write"]},
                ],
            },
        },
        context=SimpleNamespace(metadata={}),
        tool_name="fs.write",
        tool_args={"path": "ignored-by-derived-candidates.md"},
        tool_def=_filesystem_tool_def(action="write"),
        path_scope_candidates=[
            PathScopeCandidate(path="documents/story.md", action="write", source="module"),
        ],
    )

    assert result["within_scope"] is True
    assert result["reason"] is None
    assert result["path_decisions"] == [
        {
            "requested_action": "write",
            "normalized_path": "documents/story.md",
            "grant_outcome": "allowed",
            "grant_source": "path_grants",
            "matched_grant_prefix": "documents",
            "matched_grant_effect": "allow",
            "reason_code": None,
            "redacted": True,
        }
    ]
    assert "/tmp/mcp-hub-path-enforcer" not in repr(result["path_decisions"])


@pytest.mark.asyncio
async def test_effective_permission_preview_returns_redacted_path_grant_decision() -> None:
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )

    svc = McpHubPathEnforcementService(path_scope_service=_FakePathScopeService(_workspace_scope()))

    result = await svc.preview_effective_path_permission(
        effective_policy={
            "enabled": True,
            "selected_assignment_id": 11,
            "policy_document": {
                "path_scope_mode": "workspace_root",
                "path_grants": [
                    {"prefix": "documents", "actions": ["read", "edit", "write"]},
                ],
            },
            "sources": [
                {
                    "assignment_id": 11,
                    "target_type": "persona",
                    "target_id": "researcher",
                    "owner_scope_type": "user",
                    "owner_scope_id": 7,
                    "profile_id": 5,
                }
            ],
        },
        context=SimpleNamespace(metadata={}),
        tool_name="fs.write",
        action="write",
        path="documents/story.md",
    )

    assert result["tool_name"] == "fs.write"
    assert result["requested_action"] == "write"
    assert result["normalized_path"] == "documents/story.md"
    assert result["outcome"] == "allow"
    assert result["within_scope"] is True
    assert result["reason_code"] is None
    assert result["selected_assignment_id"] == 11
    assert result["profile_id"] == 5
    assert result["grant_source"] == "path_grants"
    assert result["grant_outcome"] == "allowed"
    assert result["matched_grant_prefix"] == "documents"
    assert result["matched_grant_effect"] == "allow"
    assert result["redacted"] is True
    assert "/tmp/mcp-hub-path-enforcer" not in repr(result)


@pytest.mark.asyncio
async def test_effective_permission_preview_explains_reserved_file_policy_action() -> None:
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )

    svc = McpHubPathEnforcementService(path_scope_service=_FakePathScopeService(_workspace_scope()))

    result = await svc.preview_effective_path_permission(
        effective_policy={
            "enabled": True,
            "policy_document": {
                "path_scope_mode": "workspace_root",
                "path_grants": [
                    {"prefix": "documents", "actions": ["share"]},
                ],
            },
        },
        context=SimpleNamespace(metadata={}),
        tool_name="fs.share",
        action="share",
        path="documents/story.md",
    )

    assert result["requested_action"] == "share"
    assert result["outcome"] == "allow"
    assert result["grant_outcome"] == "allowed"
    assert result["matched_grant_prefix"] == "documents"
    assert result["redacted"] is True
    assert "/tmp/mcp-hub-path-enforcer" not in repr(result)


@pytest.mark.asyncio
async def test_effective_permission_preview_denies_ungranted_reserved_file_policy_action() -> None:
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )

    svc = McpHubPathEnforcementService(path_scope_service=_FakePathScopeService(_workspace_scope()))

    result = await svc.preview_effective_path_permission(
        effective_policy={
            "enabled": True,
            "policy_document": {
                "path_scope_mode": "workspace_root",
                "path_grants": [
                    {"prefix": "documents", "actions": ["read"]},
                ],
            },
        },
        context=SimpleNamespace(metadata={}),
        tool_name="fs.export",
        action="export",
        path="documents/story.md",
    )

    assert result["requested_action"] == "export"
    assert result["outcome"] == "deny"
    assert result["reason_code"] == "path_action_not_granted"
    assert result["grant_outcome"] == "not_granted"
    assert result["redacted"] is True


@pytest.mark.asyncio
async def test_effective_permission_preview_returns_ask_for_force_approval_scope_block() -> None:
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )

    scope = _workspace_scope()
    scope["workspace_id"] = "ws-1"
    svc = McpHubPathEnforcementService(path_scope_service=_FakePathScopeService(scope))

    result = await svc.preview_effective_path_permission(
        effective_policy={
            "enabled": True,
            "selected_assignment_workspace_ids": ["ws-2"],
            "policy_document": {"path_scope_mode": "workspace_root"},
        },
        context=SimpleNamespace(metadata={}),
        tool_name="fs.read",
        action="read",
        path="documents/story.md",
    )

    assert result["outcome"] == "ask"
    assert result["reason_code"] == "workspace_not_allowed_but_trusted"
    assert result.get("path_decisions", []) == []


@pytest.mark.asyncio
async def test_effective_permission_preview_does_not_guess_profile_for_ambiguous_sources() -> None:
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )

    svc = McpHubPathEnforcementService(path_scope_service=_FakePathScopeService(_workspace_scope()))

    result = await svc.preview_effective_path_permission(
        effective_policy={
            "enabled": True,
            "selected_assignment_id": 99,
            "policy_document": {
                "path_scope_mode": "workspace_root",
                "path_grants": [
                    {"prefix": "documents", "actions": ["read"]},
                ],
            },
            "sources": [
                {"assignment_id": 11, "profile_id": 5},
                {"assignment_id": 12, "profile_id": 6},
            ],
        },
        context=SimpleNamespace(metadata={}),
        tool_name="fs.read",
        action="read",
        path="documents/story.md",
    )

    assert result["profile_id"] is None


@pytest.mark.asyncio
async def test_effective_permission_preview_does_not_guess_profile_without_selected_assignment() -> None:
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )

    svc = McpHubPathEnforcementService(path_scope_service=_FakePathScopeService(_workspace_scope()))

    result = await svc.preview_effective_path_permission(
        effective_policy={
            "enabled": True,
            "policy_document": {
                "path_scope_mode": "workspace_root",
                "path_grants": [
                    {"prefix": "documents", "actions": ["read"]},
                ],
            },
            "sources": [
                {"assignment_id": 11, "profile_id": 5},
            ],
        },
        context=SimpleNamespace(metadata={}),
        tool_name="fs.read",
        action="read",
        path="documents/story.md",
    )

    assert result["profile_id"] is None


@pytest.mark.asyncio
async def test_path_grants_deny_overrides_broader_allow_grant() -> None:
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )

    svc = McpHubPathEnforcementService(path_scope_service=_FakePathScopeService(_workspace_scope()))

    result = await svc.evaluate_tool_call(
        effective_policy={
            "enabled": True,
            "policy_document": {
                "path_scope_mode": "workspace_root",
                "path_grants": [
                    {"prefix": "documents", "actions": ["read", "edit", "write"]},
                    {"prefix": "documents/private", "actions": ["edit", "write"], "effect": "deny"},
                ],
            },
        },
        context=SimpleNamespace(metadata={}),
        tool_name="fs.patch",
        tool_args={"diff": "not-inspected-here"},
        tool_def=_filesystem_tool_def(action="edit"),
        path_scope_candidates=[
            PathScopeCandidate(path="documents/private/secret.md", action="edit", source="module"),
        ],
    )

    assert result["within_scope"] is False
    assert result["reason"] == "path_action_denied"
    assert result["force_approval"] is False
    assert result["path_decisions"] == [
        {
            "requested_action": "edit",
            "normalized_path": "documents/private/secret.md",
            "grant_outcome": "denied",
            "grant_source": "path_grants",
            "matched_grant_prefix": "documents/private",
            "matched_grant_effect": "deny",
            "reason_code": "path_action_denied",
            "redacted": True,
        }
    ]
    assert "/tmp/mcp-hub-path-enforcer" not in repr(result)


@pytest.mark.asyncio
async def test_authored_path_grants_compile_for_runtime_enforcement() -> None:
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )

    svc = McpHubPathEnforcementService(path_scope_service=_FakePathScopeService(_workspace_scope()))
    effective_policy = {
        "enabled": True,
        "policy_document": {
            "path_scope_mode": "workspace_root",
            "path_grant_authoring": {
                "workspace": [
                    {"prefix": "documents", "actions": ["read", "edit", "write"]},
                ],
                "folders": [
                    {"prefix": "documents/private", "actions": ["edit", "write"], "effect": "deny"},
                ],
            },
        },
    }

    allowed = await svc.evaluate_tool_call(
        effective_policy=effective_policy,
        context=SimpleNamespace(metadata={}),
        tool_name="fs.patch",
        tool_args={"path": "documents/public/story.md"},
        tool_def=_filesystem_tool_def(action="edit"),
    )
    denied = await svc.evaluate_tool_call(
        effective_policy=effective_policy,
        context=SimpleNamespace(metadata={}),
        tool_name="fs.patch",
        tool_args={"path": "documents/private/secret.md"},
        tool_def=_filesystem_tool_def(action="edit"),
    )

    assert allowed["within_scope"] is True
    assert allowed["path_decisions"][0]["grant_source"] == "path_grants"
    assert denied["within_scope"] is False
    assert denied["reason"] == "path_action_denied"
    assert denied["path_decisions"][0]["matched_grant_prefix"] == "documents/private"

    root_allowed = await svc.evaluate_tool_call(
        effective_policy={
            "enabled": True,
            "policy_document": {
                "path_scope_mode": "workspace_root",
                "path_grant_authoring": {
                    "workspace": [
                        {"prefix": "./", "actions": ["read"]},
                    ],
                },
            },
        },
        context=SimpleNamespace(metadata={}),
        tool_name="fs.read",
        tool_args={"path": "README.md"},
        tool_def=_filesystem_tool_def(action="read"),
    )

    assert root_allowed["within_scope"] is True
    assert root_allowed["path_decisions"][0]["grant_source"] == "path_grants"
    assert root_allowed["path_decisions"][0]["matched_grant_prefix"] == "."


@pytest.mark.asyncio
async def test_invalid_authored_path_grants_do_not_fall_back_to_legacy_allowlist() -> None:
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )

    svc = McpHubPathEnforcementService(path_scope_service=_FakePathScopeService(_workspace_scope()))

    result = await svc.evaluate_tool_call(
        effective_policy={
            "enabled": True,
            "policy_document": {
                "path_scope_mode": "workspace_root",
                "path_allowlist_prefixes": ["documents"],
                "path_grant_authoring": {
                    "workspace": [
                        {"prefix": "/documents", "actions": ["read"]},
                    ],
                },
            },
        },
        context=SimpleNamespace(metadata={}),
        tool_name="fs.read",
        tool_args={"path": "documents/story.md"},
        tool_def=_filesystem_tool_def(action="read"),
    )

    assert result["within_scope"] is False
    assert result["reason"] == "path_action_not_granted"
    assert result["path_decisions"][0]["grant_outcome"] == "not_granted"
    assert result["scope_payload"]["path_grant_diagnostic_codes"] == ["invalid_prefix"]
    assert result["scope_payload"]["path_grant_diagnostics"] == [
        {"code": "invalid_prefix", "source": "workspace[0]", "severity": "error"}
    ]


@pytest.mark.asyncio
async def test_path_grants_are_authoritative_when_legacy_allowlist_also_present() -> None:
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )

    svc = McpHubPathEnforcementService(path_scope_service=_FakePathScopeService(_workspace_scope()))

    result = await svc.evaluate_tool_call(
        effective_policy={
            "enabled": True,
            "policy_document": {
                "path_scope_mode": "workspace_root",
                "path_allowlist_prefixes": ["documents"],
                "path_grants": [
                    {"prefix": "downloads", "actions": ["read"]},
                ],
            },
        },
        context=SimpleNamespace(metadata={}),
        tool_name="fs.write",
        tool_args={"path": "documents/story.md"},
        tool_def=_filesystem_tool_def(action="write"),
    )

    assert result["within_scope"] is False
    assert result["reason"] == "path_action_not_granted"
    assert result["path_decisions"][0]["normalized_path"] == "documents/story.md"
    assert result["path_decisions"][0]["grant_source"] == "path_grants"


@pytest.mark.asyncio
async def test_empty_path_grants_fail_closed_even_with_legacy_allowlist() -> None:
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )

    svc = McpHubPathEnforcementService(path_scope_service=_FakePathScopeService(_workspace_scope()))

    result = await svc.evaluate_tool_call(
        effective_policy={
            "enabled": True,
            "policy_document": {
                "path_scope_mode": "workspace_root",
                "path_allowlist_prefixes": ["documents"],
                "path_grants": [],
            },
        },
        context=SimpleNamespace(metadata={}),
        tool_name="fs.read",
        tool_args={"path": "documents/story.md"},
        tool_def=_filesystem_tool_def(action="read"),
    )

    assert result["within_scope"] is False
    assert result["reason"] == "path_action_not_granted"
    assert result["path_decisions"][0]["grant_outcome"] == "not_granted"


@pytest.mark.asyncio
async def test_multi_root_path_grants_keep_deduped_paths_and_actions_aligned() -> None:
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )

    workspace_root = "/tmp/mcp-hub-path-enforcer/project"
    scope = {
        **_workspace_scope(),
        "workspace_id": "ws-1",
        "selected_workspace_trust_source": "sandbox_workspace_lookup",
        "selected_workspace_scope_type": "user",
        "selected_workspace_scope_id": 7,
    }
    multi_root = _FakeMultiRootPathService(workspace_root=workspace_root)
    svc = McpHubPathEnforcementService(
        path_scope_service=_FakePathScopeService(scope),
        multi_root_path_service=multi_root,
    )

    result = await svc.evaluate_tool_call(
        effective_policy={
            "enabled": True,
            "selected_assignment_workspace_ids": ["ws-1", "ws-2"],
            "policy_document": {
                "path_scope_mode": "workspace_root",
                "path_allowlist_prefixes": ["legacy-only"],
                "path_grants": [
                    {"prefix": "documents", "actions": ["read"]},
                    {"prefix": "downloads", "actions": ["write"]},
                ],
            },
        },
        context=SimpleNamespace(user_id="1", metadata={}),
        tool_name="fs.patch",
        tool_args={"diff": "not-inspected-here"},
        tool_def=_filesystem_tool_def(action="edit"),
        path_scope_candidates=[
            PathScopeCandidate(path="documents/story.md", action="read", source="module"),
            PathScopeCandidate(path="documents/story.md", action="read", source="module"),
            PathScopeCandidate(path="downloads/export.md", action="write", source="module"),
        ],
    )

    assert multi_root.calls[0]["raw_paths"] == ["documents/story.md", "downloads/export.md"]
    assert result["within_scope"] is True
    assert result["reason"] is None
    assert [decision["requested_action"] for decision in result["path_decisions"]] == ["read", "write"]
    assert result["scope_payload"]["path_decisions"] == result["path_decisions"]
    assert "/tmp/mcp-hub-path-enforcer" not in repr(result["path_decisions"])


@pytest.mark.asyncio
async def test_multi_root_path_grants_deny_override_blocks_matching_bundle_path() -> None:
    from tldw_Server_API.app.services.mcp_hub_path_enforcement_service import (
        McpHubPathEnforcementService,
    )

    workspace_root = "/tmp/mcp-hub-path-enforcer/project"
    scope = {
        **_workspace_scope(),
        "workspace_id": "ws-1",
        "selected_workspace_trust_source": "sandbox_workspace_lookup",
    }
    multi_root = _FakeMultiRootPathService(workspace_root=workspace_root)
    svc = McpHubPathEnforcementService(
        path_scope_service=_FakePathScopeService(scope),
        multi_root_path_service=multi_root,
    )

    result = await svc.evaluate_tool_call(
        effective_policy={
            "enabled": True,
            "selected_assignment_workspace_ids": ["ws-1", "ws-2"],
            "policy_document": {
                "path_scope_mode": "workspace_root",
                "path_allowlist_prefixes": ["legacy-only"],
                "path_grants": [
                    {"prefix": "documents", "actions": ["read", "edit", "write"]},
                    {"prefix": "documents/private", "actions": ["edit", "write"], "effect": "deny"},
                ],
            },
        },
        context=SimpleNamespace(user_id="1", metadata={}),
        tool_name="fs.patch",
        tool_args={"diff": "not-inspected-here"},
        tool_def=_filesystem_tool_def(action="edit"),
        path_scope_candidates=[
            PathScopeCandidate(path="documents/private/secret.md", action="edit", source="module"),
        ],
    )

    assert result["within_scope"] is False
    assert result["reason"] == "path_action_denied"
    assert result["path_decisions"] == [
        {
            "requested_action": "edit",
            "normalized_path": "documents/private/secret.md",
            "grant_outcome": "denied",
            "grant_source": "path_grants",
            "matched_grant_prefix": "documents/private",
            "matched_grant_effect": "deny",
            "reason_code": "path_action_denied",
            "redacted": True,
        }
    ]
    assert result["scope_payload"]["path_decisions"] == result["path_decisions"]
    assert "/tmp/mcp-hub-path-enforcer" not in repr(result)
