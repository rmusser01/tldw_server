from __future__ import annotations

from tldw_Server_API.app.core.Workspaces.context import build_workspace_core_context


def test_research_workspace_context_has_fail_closed_project_capabilities() -> None:
    context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "research"},
        primary_root=None,
        source_summary={"total": 0, "queryable": 0},
        service_capabilities={},
        partial_errors=[],
    )

    assert context["workspace_profile"] == "research"
    assert context["workspace_kind"] == "research_workspace"
    assert context["project_root"]["state"] == "not_configured"
    assert context["resolution"]["status"] == "complete"
    assert context["allowed_actions"]["write_files"] == {
        "allowed": False,
        "reason_code": "project_root_not_configured",
    }


def test_project_workspace_context_represents_sandbox_root() -> None:
    context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "project"},
        primary_root={
            "root_id": "root-1",
            "backend": "sandbox_volume",
            "display_name": "Sandbox project",
            "root_state": "attached",
            "absolute_root": "/srv/sandbox/volume-1",
            "sandbox_volume_id": "volume-1",
            "git_state": "absent",
            "file_inventory_state": "not_started",
            "indexing_state": "disabled",
            "sandbox_mount_state": "mounted",
            "mcp_trust_state": "trusted",
        },
        source_summary={"total": 3, "queryable": 2},
        service_capabilities={
            "workspace_services": {
                "sandbox": {"state": "available", "reason_code": None},
                "mcp": {"state": "available", "reason_code": None},
                "acp": {"state": "available", "reason_code": None},
            },
            "allowed_actions": {
                "use_sandbox": {"allowed": True, "reason_code": None},
                "run_mcp_tools": {"allowed": True, "reason_code": None},
                "use_acp_agents": {"allowed": True, "reason_code": None},
            },
        },
        partial_errors=[],
    )

    assert context["workspace_profile"] == "project"
    assert context["workspace_kind"] == "project_workspace"
    assert context["project_root"] == {
        "state": "attached",
        "root_id": "root-1",
        "backend": "sandbox_volume",
        "display_name": "Sandbox project",
        "path_hint": "volume-1",
        "git_state": "absent",
        "file_inventory_state": "not_started",
        "indexing_state": "disabled",
        "sandbox_mount_state": "mounted",
        "mcp_trust_state": "trusted",
        "file_inventory": {
            "state": "not_started",
            "indexed_file_count": 0,
            "total_file_count": 0,
            "updated_at": None,
        },
    }
    assert context["allowed_actions"]["write_files"]["allowed"] is True
    assert context["allowed_actions"]["run_sandbox"]["allowed"] is True
    assert context["allowed_actions"]["run_mcp_tools"]["allowed"] is True
    assert context["allowed_actions"]["use_mcp_tools"]["allowed"] is True


def test_sandbox_volume_root_fails_closed_until_mount_ready() -> None:
    context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "project"},
        primary_root={
            "root_id": "primary",
            "backend": "sandbox_volume",
            "root_state": "attached",
            "sandbox_volume_id": "volume-1",
            "sandbox_mount_state": "not_configured",
        },
        source_summary={},
        service_capabilities={
            "workspace_services": {
                "sandbox": {"state": "available", "reason_code": None},
                "acp": {"state": "available", "reason_code": None},
            },
            "allowed_actions": {
                "use_sandbox": {"allowed": True, "reason_code": None},
                "use_acp_agents": {"allowed": True, "reason_code": None},
            },
        },
        partial_errors=[],
    )

    assert context["allowed_actions"]["write_files"] == {
        "allowed": False,
        "reason_code": "sandbox_mount_not_configured",
    }
    assert context["allowed_actions"]["run_sandbox"] == {
        "allowed": False,
        "reason_code": "sandbox_mount_not_configured",
    }
    assert context["allowed_actions"]["use_acp_agents"] == {
        "allowed": False,
        "reason_code": "sandbox_mount_not_configured",
    }


def test_context_resolution_becomes_partial_for_dependency_failures() -> None:
    context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "project"},
        primary_root={"root_id": "root-1", "backend": "host_local", "root_state": "attached"},
        source_summary={},
        service_capabilities={},
        partial_errors=[
            {"scope": "mcp", "code": "mcp_policy_resolution_failed", "message": "MCP unavailable"}
        ],
    )

    assert context["resolution"] == {
        "status": "partial",
        "partial_errors": [
            {
                "scope": "mcp",
                "code": "dependency_resolution_partial",
                "message": "Workspace dependency resolution failed.",
            }
        ],
    }
    assert context["allowed_actions"]["run_mcp_tools"] == {
        "allowed": False,
        "reason_code": "dependency_resolution_partial",
    }
    assert context["allowed_actions"]["use_mcp_tools"] == {
        "allowed": False,
        "reason_code": "dependency_resolution_partial",
    }
    assert context["allowed_actions"]["use_acp_agents"] == {
        "allowed": False,
        "reason_code": "dependency_resolution_partial",
    }


def test_context_partial_errors_do_not_echo_upstream_messages() -> None:
    context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "project"},
        primary_root={"root_id": "root-1", "backend": "host_local", "root_state": "attached"},
        source_summary={},
        service_capabilities={},
        partial_errors=[
            {
                "scope": "mcp",
                "code": "custom_backend_code",
                "message": "/Users/alice/private/project failed",
            },
            "raw /Users/alice/private/project failure",
        ],
    )

    assert context["resolution"]["partial_errors"] == [
        {
            "scope": "mcp",
            "code": "dependency_resolution_partial",
            "message": "Workspace dependency resolution failed.",
        },
        {
            "scope": "workspace",
            "code": "dependency_resolution_partial",
            "message": "Workspace dependency resolution failed.",
        },
    ]
    assert "/Users/alice" not in str(context["resolution"]["partial_errors"])


def test_project_workspace_preview_and_file_indexing_fail_closed_until_ready() -> None:
    context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "project"},
        primary_root={
            "root_id": "root-1",
            "backend": "host_local",
            "root_state": "attached",
            "indexing_state": "disabled",
        },
        source_summary={},
        service_capabilities={
            "workspace_services": {
                "sandbox": {"state": "available", "reason_code": None},
            },
            "allowed_actions": {
                "use_sandbox": {"allowed": True, "reason_code": None},
            },
        },
        partial_errors=[],
    )

    assert context["allowed_actions"]["create_preview"] == {
        "allowed": False,
        "reason_code": "preview_not_configured",
    }
    assert context["allowed_actions"]["index_file_content"] == {
        "allowed": False,
        "reason_code": "file_indexing_disabled",
    }


def test_project_workspace_context_includes_file_inventory_summary_and_actions() -> None:
    context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "project"},
        primary_root={
            "root_id": "root-1",
            "backend": "host_local",
            "root_state": "attached",
            "file_inventory_state": "current",
            "file_inventory": {
                "state": "current",
                "total_file_count": 12,
                "indexed_file_count": 0,
                "updated_at": "2026-06-03T12:00:00Z",
            },
            "indexing_state": "disabled",
        },
        source_summary={},
        service_capabilities={},
        partial_errors=[],
    )

    assert context["project_root"]["file_inventory"] == {
        "state": "current",
        "indexed_file_count": 0,
        "total_file_count": 12,
        "updated_at": "2026-06-03T12:00:00Z",
    }
    assert context["allowed_actions"]["scan_files"] == {
        "allowed": True,
        "reason_code": None,
    }
    assert context["allowed_actions"]["view_file_inventory"] == {
        "allowed": True,
        "reason_code": None,
    }
    assert context["allowed_actions"]["index_file_content"] == {
        "allowed": False,
        "reason_code": "file_indexing_disabled",
    }


def test_file_inventory_scan_fails_closed_when_inventory_disabled() -> None:
    context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "project"},
        primary_root={
            "root_id": "root-1",
            "backend": "host_local",
            "root_state": "attached",
            "file_inventory_state": "disabled",
        },
        source_summary={},
        service_capabilities={},
        partial_errors=[],
    )

    assert context["allowed_actions"]["scan_files"] == {
        "allowed": False,
        "reason_code": "file_inventory_disabled",
    }
    assert context["allowed_actions"]["view_file_inventory"] == {
        "allowed": True,
        "reason_code": None,
    }


def test_failed_file_inventory_scan_remains_viewable_and_rescannable() -> None:
    context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "project"},
        primary_root={
            "root_id": "root-1",
            "backend": "host_local",
            "root_state": "attached",
            "file_inventory_state": "failed",
            "file_inventory": {
                "state": "failed",
                "total_file_count": 0,
                "indexed_file_count": 0,
                "updated_at": "2026-06-03T12:00:00Z",
            },
        },
        source_summary={},
        service_capabilities={},
        partial_errors=[],
    )

    assert context["project_root"]["file_inventory"]["state"] == "failed"
    assert context["allowed_actions"]["view_file_inventory"] == {
        "allowed": True,
        "reason_code": None,
    }
    assert context["allowed_actions"]["scan_files"] == {
        "allowed": True,
        "reason_code": None,
    }


def test_acp_agents_require_project_root_and_available_acp_service() -> None:
    missing_root_context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "project"},
        primary_root=None,
        source_summary={},
        service_capabilities={
            "workspace_services": {
                "acp": {"state": "available", "reason_code": None},
            },
            "allowed_actions": {
                "use_acp_agents": {"allowed": True, "reason_code": None},
            },
        },
        partial_errors=[],
    )
    unknown_acp_context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "project"},
        primary_root={"root_id": "root-1", "backend": "host_local", "root_state": "attached"},
        source_summary={},
        service_capabilities={
            "workspace_services": {
                "acp": {"state": "unknown", "reason_code": "acp_status_resolution_failed"},
            },
            "allowed_actions": {
                "use_acp_agents": {"allowed": True, "reason_code": None},
            },
        },
        partial_errors=[],
    )

    assert missing_root_context["allowed_actions"]["use_acp_agents"] == {
        "allowed": False,
        "reason_code": "project_root_not_configured",
    }
    assert unknown_acp_context["allowed_actions"]["use_acp_agents"] == {
        "allowed": False,
        "reason_code": "acp_status_resolution_failed",
    }


def test_acp_agents_fail_closed_for_unscoped_partial_resolution() -> None:
    context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "project"},
        primary_root={"root_id": "root-1", "backend": "host_local", "root_state": "attached"},
        source_summary={},
        service_capabilities={
            "workspace_services": {
                "acp": {"state": "available", "reason_code": None},
            },
            "allowed_actions": {
                "use_acp_agents": {"allowed": True, "reason_code": None},
            },
        },
        partial_errors=[{"message": "Dependency status unavailable"}],
    )

    assert context["allowed_actions"]["use_acp_agents"] == {
        "allowed": False,
        "reason_code": "dependency_resolution_partial",
    }


def test_project_root_actions_fail_closed_for_unscoped_partial_resolution() -> None:
    context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "project"},
        primary_root={
            "root_id": "root-1",
            "backend": "host_local",
            "root_state": "attached",
            "indexing_state": "ready",
        },
        source_summary={},
        service_capabilities={
            "workspace_services": {
                "preview": {"state": "available", "reason_code": None},
            },
        },
        partial_errors=[{"message": "Dependency status unavailable"}],
    )

    assert context["allowed_actions"]["write_files"] == {
        "allowed": False,
        "reason_code": "dependency_resolution_partial",
    }
    assert context["allowed_actions"]["create_preview"] == {
        "allowed": False,
        "reason_code": "dependency_resolution_partial",
    }
    assert context["allowed_actions"]["index_file_content"] == {
        "allowed": False,
        "reason_code": "dependency_resolution_partial",
    }


def test_legacy_mcp_action_fails_closed_for_unscoped_partial_resolution() -> None:
    context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "project"},
        primary_root={
            "root_id": "root-1",
            "backend": "host_local",
            "root_state": "attached",
            "mcp_trust_state": "trusted",
        },
        source_summary={},
        service_capabilities={
            "workspace_services": {
                "mcp": {"state": "available", "reason_code": None},
            },
            "allowed_actions": {
                "run_mcp_tools": {"allowed": True, "reason_code": None},
            },
        },
        partial_errors=[{"message": "Dependency status unavailable"}],
    )

    assert context["allowed_actions"]["run_mcp_tools"] == {
        "allowed": False,
        "reason_code": "dependency_resolution_partial",
    }
    assert context["allowed_actions"]["use_mcp_tools"] == {
        "allowed": False,
        "reason_code": "dependency_resolution_partial",
    }


def test_sandbox_actions_require_project_root_even_when_service_allows_sandbox() -> None:
    context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "project"},
        primary_root=None,
        source_summary={},
        service_capabilities={
            "workspace_services": {
                "sandbox": {"state": "available", "reason_code": None},
            },
            "allowed_actions": {
                "use_sandbox": {"allowed": True, "reason_code": None},
            },
        },
        partial_errors=[],
    )

    assert context["allowed_actions"]["run_sandbox"] == {
        "allowed": False,
        "reason_code": "project_root_not_configured",
    }
    assert context["allowed_actions"]["use_sandbox"] == {
        "allowed": False,
        "reason_code": "project_root_not_configured",
    }


def test_acp_agents_fail_closed_for_malformed_partial_resolution() -> None:
    context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "project"},
        primary_root={"root_id": "root-1", "backend": "host_local", "root_state": "attached"},
        source_summary={},
        service_capabilities={
            "workspace_services": {
                "acp": {"state": "available", "reason_code": None},
            },
            "allowed_actions": {
                "use_acp_agents": {"allowed": True, "reason_code": None},
            },
        },
        partial_errors=["malformed"],
    )

    assert context["resolution"]["status"] == "partial"
    assert context["allowed_actions"]["use_acp_agents"] == {
        "allowed": False,
        "reason_code": "dependency_resolution_partial",
    }


def test_project_root_path_hint_redacts_windows_absolute_paths() -> None:
    context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "project"},
        primary_root={
            "root_id": "root-1",
            "backend": "host_local",
            "root_state": "attached",
            "absolute_root": r"C:\Users\researcher\secret-project",
        },
        source_summary={},
        service_capabilities={},
        partial_errors=[],
    )
    explicit_hint_context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "project"},
        primary_root={
            "root_id": "root-1",
            "backend": "host_local",
            "root_state": "attached",
            "path_hint": r"\\server\share\secret-project",
        },
        source_summary={},
        service_capabilities={},
        partial_errors=[],
    )

    assert context["project_root"]["path_hint"] == "secret-project"
    assert explicit_hint_context["project_root"]["path_hint"] == "secret-project"


def test_project_root_path_hint_redacts_path_like_display_names() -> None:
    posix_context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "project"},
        primary_root={
            "root_id": "root-1",
            "backend": "host_local",
            "root_state": "attached",
            "display_name": "/Users/researcher/secret-project",
        },
        source_summary={},
        service_capabilities={},
        partial_errors=[],
    )
    tilde_context = build_workspace_core_context(
        workspace={"id": "ws-1", "workspace_profile": "project"},
        primary_root={
            "root_id": "root-1",
            "backend": "host_local",
            "root_state": "attached",
            "display_name": "~/secret-project",
        },
        source_summary={},
        service_capabilities={},
        partial_errors=[],
    )

    assert posix_context["project_root"]["path_hint"] == "secret-project"
    assert tilde_context["project_root"]["path_hint"] == "secret-project"
