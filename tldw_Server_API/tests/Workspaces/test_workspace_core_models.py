from tldw_Server_API.app.core.Workspaces.models import (
    fail_closed_action,
    normalize_project_root_state,
    normalize_workspace_profile,
    project_sandbox_volume_projection,
    workspace_attention_state,
    workspace_file_inventory_available,
    workspace_kind_for_profile,
)


def test_workspace_profile_defaults_to_research() -> None:
    assert normalize_workspace_profile(None) == "research"
    assert normalize_workspace_profile("") == "research"
    assert normalize_workspace_profile("project") == "project"


def test_workspace_kind_is_compatibility_alias() -> None:
    assert workspace_kind_for_profile("research") == "research_workspace"
    assert workspace_kind_for_profile("project") == "project_workspace"


def test_project_root_state_fails_closed_for_unknown_values() -> None:
    assert normalize_project_root_state("attached") == "attached"
    assert normalize_project_root_state("provisioning") == "provisioning"
    assert normalize_project_root_state("unavailable") == "unavailable"
    assert normalize_project_root_state("cleanup_pending") == "cleanup_pending"
    assert normalize_project_root_state("unexpected") == "failed"


def test_attention_state_projects_project_without_root_to_setup_pending() -> None:
    assert workspace_attention_state(
        workspace_profile="project",
        project_root_state="not_configured",
        inventory_state="not_started",
        service_errors=[],
        archived=False,
    ) == "setup_pending"


def test_attention_state_keeps_research_workspace_ready_without_root() -> None:
    assert workspace_attention_state(
        workspace_profile="research",
        project_root_state="not_configured",
        inventory_state="not_started",
    ) == "ready"


def test_attention_state_projects_active_inventory_to_working() -> None:
    assert workspace_attention_state(
        workspace_profile="project",
        project_root_state="attached",
        inventory_state="queued",
    ) == "working"


def test_attention_state_archived_overrides_root_and_errors() -> None:
    assert workspace_attention_state(
        workspace_profile="project",
        project_root_state="failed",
        inventory_state="failed",
        service_errors=["sandbox_mount_unavailable"],
        archived=True,
    ) == "archived"


def test_attention_state_projects_provisioning_root_to_working() -> None:
    assert workspace_attention_state(
        workspace_profile="project",
        project_root_state="provisioning",
        inventory_state="not_started",
    ) == "working"


def test_attention_state_projects_unavailable_root_to_blocked() -> None:
    assert workspace_attention_state(
        workspace_profile="project",
        project_root_state="unavailable",
        inventory_state="not_started",
    ) == "blocked"


def test_attention_state_projects_cleanup_pending_root_to_needs_attention() -> None:
    assert workspace_attention_state(
        workspace_profile="project",
        project_root_state="cleanup_pending",
        inventory_state="not_started",
    ) == "needs_attention"


def test_attention_state_fails_closed_for_unknown_project_root_state() -> None:
    assert workspace_attention_state(
        workspace_profile="project",
        project_root_state="surprising",
        inventory_state="current",
    ) == "blocked"


def test_sandbox_projection_ready_without_mount_fails_closed() -> None:
    projection = project_sandbox_volume_projection("ready", usable_mount=False)

    assert projection["root_state"] == "attached"
    assert projection["mount_state"] == "not_ready"
    assert projection["file_inventory"]["available"] is False
    assert projection["attention_state"] == "needs_attention"


def test_sandbox_projection_table_matches_workspace_contract() -> None:
    cases = [
        (
            "provisioning",
            False,
            {
                "root_state": "provisioning",
                "mount_state": "not_ready",
                "available": False,
                "attention_state": "working",
            },
        ),
        (
            "ready",
            True,
            {
                "root_state": "attached",
                "mount_state": "ready",
                "available": True,
                "attention_state": "ready",
            },
        ),
        (
            "not_configured",
            False,
            {
                "root_state": "unavailable",
                "mount_state": "not_configured",
                "available": False,
                "attention_state": "blocked",
            },
        ),
        (
            "unavailable",
            False,
            {
                "root_state": "unavailable",
                "mount_state": "unavailable",
                "available": False,
                "attention_state": "blocked",
            },
        ),
        (
            "failed",
            False,
            {
                "root_state": "failed",
                "mount_state": "failed",
                "available": False,
                "attention_state": "blocked",
            },
        ),
        (
            "cleanup_pending",
            False,
            {
                "root_state": "cleanup_pending",
                "mount_state": "unavailable",
                "available": False,
                "attention_state": "needs_attention",
            },
        ),
    ]

    for sandbox_state, usable_mount, expected in cases:
        projection = project_sandbox_volume_projection(sandbox_state, usable_mount=usable_mount)

        assert projection["root_state"] == expected["root_state"]
        assert projection["mount_state"] == expected["mount_state"]
        assert projection["file_inventory"]["available"] is expected["available"]
        assert projection["attention_state"] == expected["attention_state"]


def test_sandbox_projection_unavailable_blocks() -> None:
    projection = project_sandbox_volume_projection("unavailable", usable_mount=False)

    assert projection["root_state"] == "unavailable"
    assert projection["mount_state"] == "unavailable"
    assert projection["file_inventory"]["available"] is False
    assert projection["attention_state"] == "blocked"


def test_file_inventory_available_uses_shared_root_readiness_rules() -> None:
    assert workspace_file_inventory_available(
        project_root_state="attached",
        root_id="primary",
        backend="host_local",
        sandbox_mount_state=None,
        inventory_state="not_started",
    ) is True
    assert workspace_file_inventory_available(
        project_root_state="attached",
        root_id="primary",
        backend="host_local",
        sandbox_mount_state=None,
        inventory_state="disabled",
    ) is False
    assert workspace_file_inventory_available(
        project_root_state="attached",
        root_id="primary",
        backend="sandbox_volume",
        sandbox_mount_state="not_configured",
        inventory_state="not_started",
    ) is False
    assert workspace_file_inventory_available(
        project_root_state="attached",
        root_id="primary",
        backend="sandbox_volume",
        sandbox_mount_state="ready",
        inventory_state="not_started",
    ) is True


def test_fail_closed_action_uses_reason_code() -> None:
    assert fail_closed_action("root_unresolved") == {
        "allowed": False,
        "reason_code": "root_unresolved",
    }
