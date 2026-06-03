from tldw_Server_API.app.core.Workspaces.models import (
    fail_closed_action,
    normalize_project_root_state,
    normalize_workspace_profile,
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
    assert normalize_project_root_state("unexpected") == "failed"


def test_fail_closed_action_uses_reason_code() -> None:
    assert fail_closed_action("root_unresolved") == {
        "allowed": False,
        "reason_code": "root_unresolved",
    }
